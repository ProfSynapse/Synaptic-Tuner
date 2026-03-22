"""
Transformers-based model loader for exact post-training loss computation.

This loader intentionally avoids Unsloth. It resolves a saved training artifact
as either:
  - a merged Hugging Face model directory, or
  - a LoRA adapter directory backed by a base model name in adapter_config.json

The returned model is moved to the requested device and placed in eval mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - handled in tests / non-PEFT environments
    PeftModel = None  # type: ignore[assignment]


def _default_loss_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _load_adapter_base_model_name(model_dir: Path) -> str:
    adapter_config_path = model_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {model_dir}")

    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model = str(payload.get("base_model_name_or_path", "")).strip()
    if not base_model:
        raise RuntimeError("adapter_config.json is missing base_model_name_or_path")
    return base_model


def _preferred_tokenizer_source(model_dir: Path, base_model_name: str) -> str:
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    ]
    if any((model_dir / filename).exists() for filename in tokenizer_files):
        return str(model_dir)
    return base_model_name


def load_transformers_loss_model(
    model_dir: Path | str,
    *,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """Load a saved training artifact for exact loss scoring.

    Args:
        model_dir: Path to the saved `final_model` or merged model directory.
        device: Target device. Defaults to CUDA if available, else CPU.
        torch_dtype: Optional dtype override. Defaults to bf16/fp16 on CUDA.
        trust_remote_code: Passed through to Transformers loaders.

    Returns:
        Tuple of `(model, tokenizer)`.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    load_dtype = torch_dtype or _default_loss_dtype()
    merged_model_path = model_path / "config.json"
    adapter_config_path = model_path / "adapter_config.json"

    if merged_model_path.exists() and not adapter_config_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=load_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    else:
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapter checkpoints for exact loss scoring.")
        base_model_name = _load_adapter_base_model_name(model_path)
        tokenizer_source = _preferred_tokenizer_source(model_path, base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=load_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(model_path), is_trainable=False)

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer
