"""PivotRL pivot profiler — data selection for GRPO training.

Extracts (state, action) candidates from SFT trajectories, generates rollouts
with the frozen base model, scores them, and filters for "pivots" — turns with
high reward variance.  Output is consumable by ``load_raw_dataset()``.
"""
from __future__ import annotations

import hashlib, json, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch

logger = logging.getLogger(__name__)

@dataclass
class PivotCandidate:
    """A single (state, action) pair extracted from an SFT trajectory."""
    source_file: str
    source_line: int
    turn_index: int
    state_messages: list
    reference_action: str
    ground_truth_tool: str | None
    ground_truth_args_json: str | None
    jsonl_hash: str

@dataclass
class PivotResult:
    """Profiling result for a single candidate."""
    candidate: PivotCandidate
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    num_rollouts: int
    is_pivot: bool

# Tool-call extraction (mirrors rewards.py patterns)
_QWEN_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE)
_MISTRAL_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[[\s\S]*?\])")

def _safe_args(args: Any) -> dict:
    if isinstance(args, str):
        try: args = json.loads(args)
        except (json.JSONDecodeError, TypeError): return {}
    return args if isinstance(args, dict) else {}

def _extract_tool_info(text: str) -> Tuple[str | None, str | None]:
    """Return (tool_name, args_json) from an assistant message."""
    m = _QWEN_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj.get("name"), json.dumps(_safe_args(obj.get("arguments", {})), sort_keys=True)
        except (json.JSONDecodeError, TypeError): pass
    if "[TOOL_CALLS]" in text and (m2 := _MISTRAL_RE.search(text)):
        try:
            calls = json.loads(m2.group(1))
            if isinstance(calls, list) and calls:
                tc = calls[0]
                return tc.get("name"), json.dumps(_safe_args(tc.get("arguments", {})), sort_keys=True)
        except (json.JSONDecodeError, TypeError): pass
    return None, None

def extract_candidates(sft_file: Path) -> list[PivotCandidate]:
    """Walk an SFT JSONL file and extract every assistant turn as a candidate."""
    sft_file, candidates = Path(sft_file), []
    with open(sft_file, "r", encoding="utf-8") as fh:
        for line_idx, raw_line in enumerate(fh):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try: row = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s:%d", sft_file, line_idx + 1)
                continue
            convos = row.get("conversations", [])
            if not isinstance(convos, list):
                continue
            h = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()[:8]
            for i, msg in enumerate(convos):
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                tool_name, args_json = _extract_tool_info(content)
                candidates.append(PivotCandidate(
                    source_file=str(sft_file), source_line=line_idx + 1, turn_index=i,
                    state_messages=[{"role": m.get("role", ""), "content": m.get("content", "")}
                                    for m in convos[:i] if isinstance(m, dict)],
                    reference_action=content, ground_truth_tool=tool_name,
                    ground_truth_args_json=args_json, jsonl_hash=h,
                ))
    logger.info("Extracted %d candidates from %s", len(candidates), sft_file.name)
    return candidates

def profile_candidates(
    candidates: list[PivotCandidate], model: Any, tokenizer: Any,
    reward_fn: Callable, *, n_rollouts: int = 8, temperature: float = 1.0,
    max_completion_length: int = 512, batch_size: int = 16,
) -> list[PivotResult]:
    """Generate rollouts per candidate and compute reward statistics."""
    results: list[PivotResult] = []
    for idx, cand in enumerate(candidates):
        prompt = tokenizer.apply_chat_template(
            cand.state_messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
        completions: list[str] = []
        rem = n_rollouts
        while rem > 0:
            n = min(rem, batch_size)
            with torch.no_grad():
                outs = model.generate(ids.expand(n, -1), do_sample=True,
                                      temperature=temperature, max_new_tokens=max_completion_length)
            completions.extend(tokenizer.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in outs)
            rem -= n
        kw: Dict[str, Any] = {}
        if cand.ground_truth_tool is not None:
            kw["ground_truth_tool"] = [cand.ground_truth_tool] * n_rollouts
        if cand.ground_truth_args_json is not None:
            kw["ground_truth_args_json"] = [cand.ground_truth_args_json] * n_rollouts
        r = torch.tensor(reward_fn(completions, prompts=[prompt] * n_rollouts, **kw), dtype=torch.float32)
        results.append(PivotResult(
            candidate=cand, reward_mean=r.mean().item(), reward_std=r.std().item(),
            reward_min=r.min().item(), reward_max=r.max().item(),
            num_rollouts=n_rollouts, is_pivot=False))
        if (idx + 1) % 10 == 0 or idx + 1 == len(candidates):
            logger.info("Profiled %d / %d candidates", idx + 1, len(candidates))
    return results

def filter_pivots(
    results: list[PivotResult], *, variance_threshold: float = 0.1,
    min_candidates: int = 50, max_candidates: int | None = None,
    mean_reward_range: tuple[float, float] | None = None,
) -> list[PivotResult]:
    """Filter profiling results down to high-variance pivot candidates."""
    filtered = [r for r in results if r.reward_std >= variance_threshold]
    if mean_reward_range is not None:
        lo, hi = mean_reward_range
        filtered = [r for r in filtered if lo <= r.reward_mean <= hi]
    filtered.sort(key=lambda r: r.reward_std, reverse=True)
    if max_candidates is not None:
        filtered = filtered[:max_candidates]
    for r in filtered:
        r.is_pivot = True
    if len(filtered) < min_candidates:
        logger.warning("Only %d pivots found (min=%d). Consider lowering variance_threshold.",
                        len(filtered), min_candidates)
    return filtered

def pivots_to_dataset(pivots: list[PivotResult]) -> list[dict]:
    """Convert filtered pivots to JSONL-compatible dicts for the GRPO data loader."""
    rows: list[dict] = []
    for pr in pivots:
        c = pr.candidate
        rows.append({
            "prompt": c.state_messages,
            "ground_truth_tool": c.ground_truth_tool,
            "ground_truth_args_json": c.ground_truth_args_json,
            "pivot_metadata": {
                "source_file": c.source_file, "source_line": c.source_line,
                "turn_index": c.turn_index, "jsonl_hash": c.jsonl_hash,
                "reward_mean": pr.reward_mean, "reward_std": pr.reward_std,
                "reward_min": pr.reward_min, "reward_max": pr.reward_max,
                "num_rollouts": pr.num_rollouts,
            },
        })
    return rows

# -- Cache -------------------------------------------------------------------
_DEFAULT_CACHE_DIR = Path("Datasets/grpo/.pivot_cache")

def _cache_key(sft_file: Path, model_name: str, cfg: dict) -> str:
    with open(sft_file, "rb") as fh:
        fh_hash = hashlib.sha256(fh.read()).hexdigest()[:12]
    return hashlib.sha1(f"{fh_hash}{model_name}{json.dumps(cfg, sort_keys=True)}".encode()).hexdigest()[:12]

def _cache_path(sft_file: Path, h: str, d: Path) -> Path:
    return d / f"{sft_file.stem}_{h}.jsonl"

# -- Public entry point ------------------------------------------------------
def profile_pivots(
    sft_file: Path | str, model: Any, tokenizer: Any,
    reward_fn: Callable, pivot_config: dict,
) -> list[dict]:
    """Orchestrate: extract -> profile -> filter -> convert.

    Checks cache first; on miss runs the full pipeline and persists results.
    ``pivot_config`` keys: n_rollouts, temperature, max_completion_length,
    batch_size, variance_threshold, min_candidates, max_candidates,
    mean_reward_range, cache_dir.
    """
    sft_file = Path(sft_file)
    cache_dir = Path(pivot_config.get("cache_dir", _DEFAULT_CACHE_DIR))
    mname = getattr(getattr(model, "config", None), "_name_or_path", "unknown")
    cached = _cache_path(sft_file, _cache_key(sft_file, mname, pivot_config), cache_dir)

    if cached.exists():
        rows = [json.loads(ln) for ln in cached.read_text("utf-8").splitlines() if ln.strip()]
        logger.info("Loaded %d pivot candidates from cache (%s)", len(rows), cached.name)
        return rows

    candidates = extract_candidates(sft_file)
    profiled = profile_candidates(
        candidates, model, tokenizer, reward_fn,
        n_rollouts=int(pivot_config.get("n_rollouts", 8)),
        temperature=float(pivot_config.get("temperature", 1.0)),
        max_completion_length=int(pivot_config.get("max_completion_length", 512)),
        batch_size=int(pivot_config.get("batch_size", 16)))
    mr = pivot_config.get("mean_reward_range")
    pivots = filter_pivots(
        profiled, variance_threshold=float(pivot_config.get("variance_threshold", 0.1)),
        min_candidates=int(pivot_config.get("min_candidates", 50)),
        max_candidates=pivot_config.get("max_candidates"),
        mean_reward_range=tuple(mr) if mr is not None else None)
    rows = pivots_to_dataset(pivots)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cached, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info("Cached %d pivot candidates to %s", len(rows), cached.name)
    return rows
