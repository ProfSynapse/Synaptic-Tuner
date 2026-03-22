from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


def cuda_headroom_bytes(device: str | torch.device | None = None) -> int | None:
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda")
    else:
        device = torch.device(device)
    if device.type != "cuda":
        return None
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    return int(free_bytes)


def recommend_eval_max_workers(
    *,
    backend: str,
    requested_max_workers: int | None,
    cpu_count: int | None = None,
) -> int:
    cores = max(1, int(cpu_count or os.cpu_count() or 1))
    requested = int(requested_max_workers or 0)

    if backend == "vllm":
        auto_cap = max(2, min(16, cores // 2))
    elif backend in {"openrouter", "ollama", "lmstudio", "mlc"}:
        auto_cap = max(2, min(8, cores // 2))
    else:
        auto_cap = 1

    if requested > 0:
        return max(1, min(requested, auto_cap if backend != "unsloth" else requested))
    return auto_cap


@dataclass
class AdaptiveTokenBudget:
    initial_tokens: int
    min_tokens: int = 2048
    max_tokens: int = 65536
    growth_factor: float = 1.2
    shrink_factor: float = 0.5
    target_batch_seconds: float = 2.0
    low_headroom_bytes: int = 512 * 1024 * 1024
    high_headroom_bytes: int = 2 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        self.current_tokens = max(self.min_tokens, min(int(self.initial_tokens), self.max_tokens))

    def observe_success(self, *, padded_batch_tokens: int, elapsed_seconds: float, headroom_bytes: int | None) -> int:
        baseline = max(self.current_tokens, int(padded_batch_tokens))
        if headroom_bytes is not None and headroom_bytes < self.low_headroom_bytes:
            self.current_tokens = max(self.min_tokens, int(max(self.min_tokens, baseline) * self.shrink_factor))
            return self.current_tokens
        if elapsed_seconds <= self.target_batch_seconds and (
            headroom_bytes is None or headroom_bytes >= self.high_headroom_bytes
        ):
            self.current_tokens = min(self.max_tokens, int(max(baseline, self.current_tokens) * self.growth_factor))
        return self.current_tokens

    def observe_oom(self) -> int:
        self.current_tokens = max(self.min_tokens, int(self.current_tokens * self.shrink_factor))
        return self.current_tokens
