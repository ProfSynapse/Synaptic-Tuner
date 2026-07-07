"""Transformers batched engine for both batch verbs.

Location: tuner/batch/engines/hf_batched.py
Purpose: The default engine. Length-sorted micro-batching turns the bs=1
    sequential decode (memory-bandwidth-bound, a few percent of GPU capacity)
    into full batches; a single batched forward with ``output_hidden_states``
    replaces the duplicate per-row capture pass.
Used by: tuner.batch.engines.get_generate_engine / get_capture_engine.

Padding discipline
------------------
* Generation uses LEFT padding: causal decoding must have the real tokens
  flush against the position where the first new token is produced, so every
  row in a batch shares the same "next token" column. The pad token falls back
  to EOS *without mutating the special-id set* (we set ``pad_token_id`` on the
  generate call, not on the tokenizer's special tokens).
* Capture uses RIGHT padding with an attention mask: absolute position indices
  supplied by the caller index into the real (unpadded-prefix) sequence, which
  right padding preserves; ``"last"`` resolves to the final real token via the
  attention mask.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from tuner.batch.engines.base import (
    CaptureEngine,
    CaptureItem,
    CaptureResult,
    GenerateEngine,
    GenerateItem,
    GenerateResult,
    OutOfMemoryError,
)


def _is_cuda_oom(exc: BaseException) -> bool:
    """Best-effort detection of a CUDA OOM from any torch version."""
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "OutOfMemory"}:
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _select_dtype(torch, device: str):
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


class _ModelBundle:
    """Lazily loads and holds a transformers model + tokenizer on a device."""

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = True,
        model: Any = None,
        tokenizer: Any = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision, trust_remote_code=trust_remote_code
            )
        self.tokenizer = tokenizer

        if model is None:
            torch_dtype = (
                getattr(torch, dtype) if dtype else _select_dtype(torch, device)
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
            )
            model.to(device)
        self.model = model
        self.model.eval()
        self._ensure_pad_token()

    def _ensure_pad_token(self) -> None:
        """Designate a pad token so batched tokenization can pad.

        If the tokenizer already has a pad token, nothing changes. Otherwise the
        EOS token is designated as the pad token. EOS is already in the special
        set, so this adds NO new special id — it only tells the tokenizer which
        existing token to use for padding (the transformers-recommended pattern
        for pad-less tokenizers like GPT-2).
        """
        tok = self.tokenizer
        if tok.pad_token_id is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:  # pragma: no cover - extremely rare (no EOS at all)
                tok.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(tok))

    def pad_token_id(self) -> int:
        """The pad id used for batching (set up by ``_ensure_pad_token``)."""
        tok = self.tokenizer
        if tok.pad_token_id is not None:
            return int(tok.pad_token_id)
        if tok.eos_token_id is not None:
            return int(tok.eos_token_id)
        return 0


def _run_with_oom_halving(
    fn: Callable[[List[Any], int], List[Any]],
    items: List[Any],
    batch_size: int,
    on_oom: Optional[Callable[[int, int], None]],
    torch,
) -> List[Any]:
    """Process ``items`` in micro-batches, halving batch size on CUDA OOM.

    ``fn(chunk, bs)`` processes one micro-batch. On OOM the current chunk is
    retried at a halved size, down to 1; below 1 the OOM is re-raised.
    """
    results: List[Any] = []
    idx = 0
    bs = max(1, int(batch_size))
    n = len(items)
    while idx < n:
        chunk = items[idx : idx + bs]
        try:
            results.extend(fn(chunk, bs))
            idx += len(chunk)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it's OOM
            if not _is_cuda_oom(exc):
                raise
            if bs <= 1:
                raise OutOfMemoryError(
                    "CUDA OOM even at batch size 1; sequence too large for this GPU."
                ) from exc
            new_bs = max(1, bs // 2)
            if on_oom is not None:
                on_oom(bs, new_bs)
            bs = new_bs
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


class HFBatchedGenerateEngine(GenerateEngine):
    """Batched, length-sorted, left-padded ``model.generate``."""

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        max_new_tokens: int = 48,
        min_new_tokens: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        extra_eos_tokens: Optional[List[str]] = None,
        stop: Optional[List[str]] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = True,
        model: Any = None,
        tokenizer: Any = None,
    ):
        self.bundle = _ModelBundle(
            model_name,
            device=device,
            dtype=dtype,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model=model,
            tokenizer=tokenizer,
        )
        self.max_new_tokens = int(max_new_tokens)
        self.min_new_tokens = int(min_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = seed
        self.extra_eos_tokens = list(extra_eos_tokens) if extra_eos_tokens else []
        self.stop = list(stop) if stop else []

    def generate(
        self,
        items: List[GenerateItem],
        *,
        batch_size: int,
        on_oom: Optional[Callable[[int, int], None]] = None,
    ) -> List[GenerateResult]:
        torch = self.bundle.torch
        # Length-sort so each micro-batch has similar prompt lengths (less pad,
        # less wasted compute). Keyed skip means order is irrelevant downstream.
        order = sorted(range(len(items)), key=lambda i: len(items[i].prompt))
        ordered = [items[i] for i in order]

        def _process(chunk: List[GenerateItem], _bs: int) -> List[GenerateResult]:
            return self._generate_chunk(chunk)

        results = _run_with_oom_halving(_process, ordered, batch_size, on_oom, torch)
        return results

    def _generate_chunk(self, chunk: List[GenerateItem]) -> List[GenerateResult]:
        torch = self.bundle.torch
        tok = self.bundle.tokenizer
        model = self.bundle.model
        pad_id = self.bundle.pad_token_id()

        if self.seed is not None:
            torch.manual_seed(self.seed)

        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(
                [it.prompt for it in chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
        finally:
            tok.padding_side = prev_side

        input_ids = enc["input_ids"].to(self.bundle.device)
        attention_mask = enc["attention_mask"].to(self.bundle.device)
        prompt_len = input_ids.shape[1]

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=pad_id,
            use_cache=True,
        )
        eos_ids = self._eos_token_ids()
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids[0] if len(eos_ids) == 1 else eos_ids
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # New tokens are everything past the (left-padded) prompt columns.
        new_tokens = out[:, prompt_len:]
        results: List[GenerateResult] = []
        eos_ids = set(self._eos_token_ids())
        for i, it in enumerate(chunk):
            row_ids = new_tokens[i].tolist()
            trimmed, finish_reason = self._trim(row_ids, pad_id, eos_ids)
            text = tok.decode(trimmed, skip_special_tokens=True)
            text, stop_hit = self._apply_stop(text)
            if stop_hit:
                finish_reason = "stop"
            # Real (non-pad) prompt length for this row.
            real_prompt_len = int(attention_mask[i].sum().item())
            results.append(
                GenerateResult(
                    id=it.id,
                    completion_text=text,
                    completion_token_ids=trimmed,
                    prompt_token_len=real_prompt_len,
                    finish_reason=finish_reason,
                    passthrough=it.passthrough,
                )
            )
        return results

    def _eos_token_ids(self) -> List[int]:
        tok = self.bundle.tokenizer
        ids = set()
        if tok.eos_token_id is not None:
            ids.add(int(tok.eos_token_id))
        for token in self.extra_eos_tokens:
            try:
                tok_id = tok.convert_tokens_to_ids(token)
            except Exception:
                continue
            unk_id = getattr(tok, "unk_token_id", None)
            if isinstance(tok_id, int) and tok_id >= 0 and tok_id != unk_id:
                ids.add(int(tok_id))
        return sorted(ids)

    def _trim(self, ids: List[int], pad_id: int, eos_ids: set[int]):
        """Trim trailing pad and cut at the first EOS; report a finish reason."""
        finish_reason = "length"
        out: List[int] = []
        for tid in ids:
            if tid in eos_ids:
                finish_reason = "eos"
                break
            out.append(tid)
        # Strip trailing pad ids that can appear when a shorter row in the batch
        # already stopped (pad fills the rest of its row).
        while out and out[-1] == pad_id and pad_id not in eos_ids:
            out.pop()
        return out, finish_reason

    def _apply_stop(self, text: str):
        """Cut ``text`` at the earliest stop string, if any."""
        best = None
        for s in self.stop:
            if not s:
                continue
            pos = text.find(s)
            if pos != -1 and (best is None or pos < best):
                best = pos
        if best is not None:
            return text[:best], True
        return text, False


class HFBatchedCaptureEngine(CaptureEngine):
    """Batched forward with ``output_hidden_states``, right-padded."""

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        layers: str = "all",
        revision: Optional[str] = None,
        trust_remote_code: bool = True,
        model: Any = None,
        tokenizer: Any = None,
    ):
        self.bundle = _ModelBundle(
            model_name,
            device=device,
            dtype=dtype,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model=model,
            tokenizer=tokenizer,
        )
        self.layers_spec = layers

    def _resolve_layers(self, n_hidden_states: int) -> List[int]:
        """Turn the layer spec into concrete hidden_states indices.

        ``output_hidden_states`` returns ``num_hidden_layers + 1`` tensors
        (index 0 = embeddings, 1..N = block outputs). ``all`` selects every one;
        a comma list selects those indices.
        """
        if self.layers_spec in (None, "", "all"):
            return list(range(n_hidden_states))
        out: List[int] = []
        for part in str(self.layers_spec).split(","):
            part = part.strip()
            if part == "":
                continue
            idx = int(part)
            if idx < 0 or idx >= n_hidden_states:
                raise ValueError(
                    f"layer index {idx} out of range for a model with "
                    f"{n_hidden_states} hidden-state layers (0..{n_hidden_states - 1})"
                )
            out.append(idx)
        if not out:
            raise ValueError(f"No valid layers parsed from {self.layers_spec!r}")
        return out

    def capture(
        self,
        items: List[CaptureItem],
        *,
        batch_size: int,
        on_oom: Optional[Callable[[int, int], None]] = None,
    ) -> List[CaptureResult]:
        torch = self.bundle.torch

        def _len(it: CaptureItem) -> int:
            if it.token_ids is not None:
                return len(it.token_ids)
            return len(it.text or "")

        order = sorted(range(len(items)), key=lambda i: _len(items[i]))
        ordered = [items[i] for i in order]

        def _process(chunk: List[CaptureItem], _bs: int) -> List[CaptureResult]:
            return self._capture_chunk(chunk)

        return _run_with_oom_halving(_process, ordered, batch_size, on_oom, torch)

    def _tokenize_chunk(self, chunk: List[CaptureItem]):
        """Right-pad a chunk to a common length; return ids, mask, real lengths."""
        torch = self.bundle.torch
        tok = self.bundle.tokenizer
        pad_id = self.bundle.pad_token_id()

        if all(it.token_ids is not None for it in chunk):
            seqs = [list(it.token_ids) for it in chunk]
            lengths = [len(s) for s in seqs]
            maxlen = max(lengths)
            input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                attention_mask[i, : len(s)] = 1
            return input_ids, attention_mask, lengths

        prev_side = tok.padding_side
        tok.padding_side = "right"
        try:
            enc = tok(
                [it.text or "" for it in chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
        finally:
            tok.padding_side = prev_side
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        lengths = attention_mask.sum(dim=1).tolist()
        return input_ids, attention_mask, [int(x) for x in lengths]

    def _capture_chunk(self, chunk: List[CaptureItem]) -> List[CaptureResult]:
        torch = self.bundle.torch
        model = self.bundle.model

        input_ids, attention_mask, lengths = self._tokenize_chunk(chunk)
        input_ids = input_ids.to(self.bundle.device)
        attention_mask = attention_mask.to(self.bundle.device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = out.hidden_states  # tuple: (n_layers+1) x [B, T, H]
        n_hidden = len(hidden_states)
        layer_idxs = self._resolve_layers(n_hidden)
        hidden_dim = int(hidden_states[0].shape[-1])

        results: List[CaptureResult] = []
        for i, it in enumerate(chunk):
            real_len = lengths[i]
            resolved = self._resolve_positions(it.positions, real_len)
            tensors: Dict[str, Any] = {}
            for pos_name, tok_idx in resolved.items():
                for layer in layer_idxs:
                    vec = hidden_states[layer][i, tok_idx].to(torch.float32).cpu()
                    tensors[f"{pos_name}__L{layer}"] = vec
            results.append(
                CaptureResult(
                    id=it.id,
                    tensors=tensors,
                    n_layers=len(layer_idxs),
                    hidden_dim=hidden_dim,
                    positions=resolved,
                    passthrough=it.passthrough,
                )
            )
        return results

    def _resolve_positions(self, positions: Dict[str, Any], real_len: int) -> Dict[str, int]:
        """Resolve named positions to absolute (right-padded) token indices.

        ``"last"`` -> the final real token (``real_len - 1``). Negative indices
        count from the end of the real sequence. Out-of-range indices raise.
        """
        resolved: Dict[str, int] = {}
        for name, spec in positions.items():
            if isinstance(spec, str) and spec == "last":
                idx = real_len - 1
            else:
                idx = int(spec)
                if idx < 0:
                    idx = real_len + idx
            if idx < 0 or idx >= real_len:
                raise ValueError(
                    f"position {name!r}={spec!r} resolves to index {idx}, out of "
                    f"range for a sequence of {real_len} real tokens"
                )
            resolved[name] = idx
        return resolved
