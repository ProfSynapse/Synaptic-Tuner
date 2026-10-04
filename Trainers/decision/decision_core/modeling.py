"""
DecisionModel: causal-LM torso + LoRA + a readout that scores options.

Location: Trainers/decision/decision_core/modeling.py
Used by:  trainer.py, evaluate.py, inference.py, train_decision.py.

The full causal LM is kept (not just the decoder) because the letter-logit
readout and the frozen-readout KL term both read the LM's output embedding.
LoRA is injected in place, so calling the inner decoder still runs the adapter,
and ``disable_adapter()`` gives the untouched torso for the KL reference.

Saved layout (``final_model/``):
    adapter_config.json, adapter_model.safetensors   PEFT LoRA adapter
    readout_head.safetensors                          pointer head (pointer only)
    decision_config.json                              DecisionModelConfig
    tokenizer files
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .model_config import CONFIG_FILE, DecisionModelConfig
from .prompting import option_markers
from .readouts import (
    PointerHead,
    gather_answer,
    gather_positions,
    marker_logits,
    mask_logits,
)

HEAD_FILE = "readout_head.safetensors"

def single_token_marker_ids(tokenizer: Any, marker_style: str, limit: int = 26) -> list[int]:
    """Token ids of option markers, in order, up to the first marker that is not one token.

    The readout reads the token that directly follows ``<answer>``, so each marker
    is encoded with no leading space. Numbers stop at 9 because "10" is two digit
    tokens in Qwen-style tokenisers.
    """
    ids: list[int] = []
    style_cap = 9 if marker_style == "numbers" else 26
    for marker in option_markers(marker_style, min(limit, style_cap)):
        enc = tokenizer.encode(marker, add_special_tokens=False)
        if len(enc) != 1:
            break
        ids.append(int(enc[0]))
    return ids


def _load_causal_lm(config: DecisionModelConfig) -> nn.Module:
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    # transformers 5 renamed `torch_dtype` to `dtype`; 4.x only knows the old name.
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kwargs: dict[str, Any] = {
        dtype_key: getattr(torch, config.torch_dtype),
        "revision": config.revision,
    }
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation
    if config.loader == "text_tower":
        # Multimodal wrapper checkpoints (e.g. Qwen3.5): load only the text tower
        # through the registry-named causal-LM class, which maps the checkpoint's
        # weight names.
        lm_cls = getattr(transformers, config.causal_lm_class, None)
        if lm_cls is None:
            raise ValueError(
                f"transformers {transformers.__version__} has no {config.causal_lm_class}; "
                "the registry entry needs a newer transformers"
            )
        base_cfg = AutoConfig.from_pretrained(config.hf_id, revision=config.revision)
        return lm_cls.from_pretrained(config.hf_id, config=base_cfg.get_text_config(), **kwargs)
    return AutoModelForCausalLM.from_pretrained(config.hf_id, **kwargs)


class DecisionModel(nn.Module):
    def __init__(self, config: DecisionModelConfig, lm: nn.Module, tokenizer: Any):
        super().__init__()
        self.decision_config = config
        self.lm = lm
        self.tokenizer = tokenizer
        hidden_size = int(self._inner().config.hidden_size)
        self.head: nn.Module | None = None
        if config.readout == "pointer":
            self.head = PointerHead(hidden_size, config.pointer_dim, config.head_dropout).to(torch.float32)
        marker_ids = single_token_marker_ids(tokenizer, config.marker_style)
        if not marker_ids and config.readout == "letter_logits":
            raise ValueError(f"no {config.marker_style} marker is a single token in this tokenizer")
        self.register_buffer("marker_ids", torch.tensor(marker_ids, dtype=torch.long), persistent=False)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_base(cls, config: DecisionModelConfig) -> "DecisionModel":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.hf_id, revision=config.revision)
        lm = _load_causal_lm(config)
        return cls.wrap(config, lm, tokenizer)

    @classmethod
    def wrap(cls, config: DecisionModelConfig, lm: nn.Module, tokenizer: Any) -> "DecisionModel":
        """Attach LoRA (if enabled) to an already-loaded causal LM."""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(lm.config, "use_cache", None):
            lm.config.use_cache = False
        if config.use_lora:
            from peft import LoraConfig, get_peft_model

            lm = get_peft_model(
                lm,
                LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=config.lora_targets,
                    bias="none",
                ),
            )
        else:
            for p in lm.parameters():
                p.requires_grad_(False)
        return cls(config, lm, tokenizer)

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device | None = None) -> "DecisionModel":
        from transformers import AutoTokenizer

        path = Path(path)
        config = DecisionModelConfig.from_dict(json.loads((path / CONFIG_FILE).read_text(encoding="utf-8")))
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        lm = _load_causal_lm(config)
        if config.use_lora:
            from peft import PeftModel

            lm = PeftModel.from_pretrained(lm, str(path))
        model = cls(config, lm, tokenizer)
        if model.head is not None:
            from safetensors.torch import load_file

            model.head.load_state_dict(load_file(str(path / HEAD_FILE)))
        if device is not None:
            model.to(device)
        model.eval()
        return model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.decision_config.use_lora:
            self.lm.save_pretrained(str(path))
        if self.head is not None:
            from safetensors.torch import save_file

            save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
                      str(path / HEAD_FILE))
        self.tokenizer.save_pretrained(str(path))
        (path / CONFIG_FILE).write_text(json.dumps(self.decision_config.to_dict(), indent=2), encoding="utf-8")

    # ---- internals --------------------------------------------------------

    def _inner(self) -> nn.Module:
        """The causal LM with LoRA layers injected (PEFT unwrapped)."""
        get_base = getattr(self.lm, "get_base_model", None)
        return get_base() if callable(get_base) else self.lm

    def _decoder(self) -> nn.Module:
        inner = self._inner()
        get_decoder = getattr(inner, "get_decoder", None)
        decoder = get_decoder() if callable(get_decoder) else None
        return decoder if decoder is not None else inner.model

    def _output_layer(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self._inner().get_output_embeddings()
        if out is None:
            raise RuntimeError("base model exposes no output embedding for the letter-logit readout")
        return out.weight, getattr(out, "bias", None)

    def gradient_checkpointing_enable(self, **_: Any) -> None:
        self._inner().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    @property
    def max_marker_options(self) -> int:
        return int(self.marker_ids.numel())

    # ---- forward ----------------------------------------------------------

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self._decoder()(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state

    def readout_logits(
        self,
        hidden: torch.Tensor,
        option_index: torch.Tensor,
        answer_index: torch.Tensor,
        n_options: torch.Tensor,
        readout: str | None = None,
    ) -> torch.Tensor:
        readout = readout or self.decision_config.readout
        answer = gather_answer(hidden, answer_index).to(torch.float32)
        if readout == "pointer":
            options = gather_positions(hidden, option_index).to(torch.float32)
            logits = self.head(answer, options)
        else:
            weight, bias = self._output_layer()
            logits = marker_logits(answer, weight, self.marker_ids, option_index.size(1), bias)
        return mask_logits(logits, n_options)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_index: torch.Tensor,
        answer_index: torch.Tensor,
        n_options: torch.Tensor,
    ) -> torch.Tensor:
        """Uncalibrated, masked option logits [B, K]."""
        hidden = self.hidden_states(input_ids, attention_mask)
        return self.readout_logits(hidden, option_index, answer_index, n_options)

    @torch.no_grad()
    def frozen_marker_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_index: torch.Tensor,
        answer_index: torch.Tensor,
        n_options: torch.Tensor,
    ) -> torch.Tensor:
        """The untouched torso's own option-marker readout (adapter disabled)."""
        ctx = self.lm.disable_adapter() if hasattr(self.lm, "disable_adapter") else contextlib.nullcontext()
        was_training = self.training
        self.eval()
        try:
            with ctx:
                hidden = self.hidden_states(input_ids, attention_mask)
                return self.readout_logits(hidden, option_index, answer_index, n_options,
                                           readout="letter_logits")
        finally:
            self.train(was_training)
