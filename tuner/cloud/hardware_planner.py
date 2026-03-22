"""Blind hardware planning for HF Jobs stages.

This module intentionally works without prior run telemetry. It uses:
  - live HF Jobs hardware/pricing data
  - experiment/stage config
  - simple model-size and memory heuristics

The goal is not perfect prediction. The goal is to rank feasible flavors well
enough to choose sensible defaults before launch.
"""

from __future__ import annotations

import json
import math
import os
import re
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from shared.experiment_tracking.experiment_spec import ExperimentSpec


DEFAULT_HF_HARDWARE_URL = "https://huggingface.co/api/jobs/hardware"

_CPU_RE = re.compile(r"(\d+(?:\.\d+)?)")
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([TGMK]?B)", re.IGNORECASE)
_PARAM_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?:\s*)([bm])(?![a-z])", re.IGNORECASE)


@dataclass(frozen=True)
class HardwareFlavor:
    flavor: str
    pretty_name: str
    gpu_model: str
    gpus: int
    vram_gb: float
    ram_gb: float
    cpu: float
    price_hr: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class StageEstimate:
    stage: str
    flavor: str
    pretty_name: str
    feasible: bool
    reason: str
    gpu_model: str
    vram_gb: float
    price_hr: float
    recommended_batch_size: Optional[int]
    recommended_gradient_accumulation: Optional[int]
    estimated_memory_gb: float
    estimated_headroom_gb: float
    throughput_score: float
    score_per_dollar: float
    estimated_hours: Optional[float]
    estimated_cost: Optional[float]


@dataclass(frozen=True)
class StagePlan:
    stage: str
    optimize_for: str
    model_name: str
    rows: list[StageEstimate]

    @property
    def recommendation(self) -> Optional[StageEstimate]:
        feasible = [row for row in self.rows if row.feasible]
        return feasible[0] if feasible else None


def _parse_numeric_string(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    match = _CPU_RE.search(value.replace(",", ""))
    return float(match.group(1)) if match else 0.0


def _parse_memory_gb(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    match = _MEM_RE.search(value.replace(",", ""))
    if not match:
        return 0.0
    amount = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "TB":
        return amount * 1024.0
    if unit == "MB":
        return amount / 1024.0
    if unit == "KB":
        return amount / (1024.0 * 1024.0)
    return amount


def _cost_per_hour(item: dict[str, Any]) -> float:
    if isinstance(item.get("price_hr"), (int, float)):
        return float(item["price_hr"])

    unit_cost = item.get("unitCostUSD")
    if not isinstance(unit_cost, (int, float)):
        unit_cost = item.get("hourly_price") or item.get("price") or item.get("priceUsd")
    if not isinstance(unit_cost, (int, float)):
        return 0.0

    unit_label = str(item.get("unitLabel") or "hour").strip().lower()
    if unit_label.startswith("min"):
        return float(unit_cost) * 60.0
    if unit_label.startswith("sec"):
        return float(unit_cost) * 3600.0
    return float(unit_cost)


def _default_ssl_context() -> Optional[ssl.SSLContext]:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def fetch_hardware_payload(url: str = DEFAULT_HF_HARDWARE_URL) -> Any:
    headers = {"Accept": "application/json"}
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    ssl_context = _default_ssl_context()
    with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_hardware_rows(payload: Any) -> list[HardwareFlavor]:
    candidates: Iterable[Any]
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        for key in ("hardware", "flavors", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break
        else:
            candidates = []
    else:
        candidates = []

    rows: list[HardwareFlavor] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        flavor = str(item.get("name") or item.get("flavor") or item.get("id") or "").strip()
        if not flavor:
            continue

        accelerator = item.get("accelerator") or {}
        gpu_model = str(
            item.get("gpu")
            or item.get("gpu_name")
            or accelerator.get("model")
            or accelerator.get("label")
            or ""
        ).strip()
        gpus = int(_parse_numeric_string(item.get("gpus") or accelerator.get("quantity") or 1) or 1)
        vram_gb = _parse_memory_gb(item.get("vram_gb") or accelerator.get("vram") or item.get("gpu_memory_gb"))
        ram_gb = _parse_memory_gb(item.get("ram") or item.get("ram_gb") or item.get("memory"))
        cpu = _parse_numeric_string(item.get("cpu") or item.get("cpus") or item.get("vcpus"))
        rows.append(
            HardwareFlavor(
                flavor=flavor,
                pretty_name=str(item.get("prettyName") or item.get("pretty_name") or flavor),
                gpu_model=gpu_model,
                gpus=gpus,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                cpu=cpu,
                price_hr=_cost_per_hour(item),
                raw=item,
            )
        )
    return rows


def load_live_hf_hardware_rows() -> list[HardwareFlavor]:
    rows = normalize_hardware_rows(fetch_hardware_payload())
    return sorted(rows, key=lambda row: (row.price_hr, row.flavor))


def parse_model_params_billions(model_name: str) -> Optional[float]:
    matches = list(_PARAM_RE.finditer(model_name))
    if not matches:
        return None
    value = float(matches[-1].group(1))
    unit = matches[-1].group(2).lower()
    return value if unit == "b" else value / 1000.0


def _default_effective_batch(method: str) -> int:
    return {"sft": 32, "kto": 16, "grpo": 8}.get(method, 16)


def _gpu_speed_factor(gpu_model: str, *, stage: str) -> float:
    key = gpu_model.strip().upper()
    training = {
        "T4": 0.7,
        "L4": 0.9,
        "A10G": 1.0,
        "L40S": 1.6,
        "A100": 2.3,
        "H200": 3.0,
    }
    eval_like = {
        "T4": 1.0,
        "L4": 1.15,
        "A10G": 1.3,
        "L40S": 1.9,
        "A100": 2.5,
        "H200": 3.2,
    }
    table = training if stage == "training" else eval_like
    return table.get(key, 1.0)


def _training_resident_gb(model_b: float, *, load_in_4bit: bool) -> float:
    return 5.0 + ((1.2 if load_in_4bit else 2.2) * model_b)


def estimate_training_memory_gb(
    *,
    model_b: float,
    seq_len: int,
    batch_size: int,
    method: str,
    load_in_4bit: bool,
) -> float:
    resident = _training_resident_gb(model_b, load_in_4bit=load_in_4bit)
    activation_coeff = {"sft": 1.0, "kto": 1.05, "grpo": 1.2}.get(method, 1.0)
    activation = activation_coeff * 1.1 * math.sqrt(max(model_b, 0.1)) * max(batch_size, 1) * max(seq_len, 1) / 2048.0
    return resident + activation


def estimate_eval_memory_gb(*, model_b: float, runtime: str) -> float:
    runtime_key = (runtime or "unsloth").strip().lower()
    factor = 1.25 if runtime_key == "vllm" else 1.0
    return 3.0 + (factor * model_b) + 2.0


def estimate_loss_memory_gb(*, model_b: float, seq_len: int) -> float:
    return 5.0 + (2.2 * model_b) + (0.12 * model_b * max(seq_len, 1) / 2048.0)


def _usable_vram_gb(vram_gb: float) -> float:
    return max((vram_gb * 0.9) - 1.0, 0.0)


def _recommend_training_microbatch(
    *,
    model_b: float,
    seq_len: int,
    method: str,
    load_in_4bit: bool,
    vram_gb: float,
    requested_batch_size: Optional[int],
) -> tuple[Optional[int], float]:
    usable = _usable_vram_gb(vram_gb)
    feasible_batches: list[tuple[int, float]] = []
    search_limit = max(requested_batch_size or 0, 32)
    for batch_size in range(1, search_limit + 1):
        estimate = estimate_training_memory_gb(
            model_b=model_b,
            seq_len=seq_len,
            batch_size=batch_size,
            method=method,
            load_in_4bit=load_in_4bit,
        )
        if estimate <= usable:
            feasible_batches.append((batch_size, estimate))
        elif requested_batch_size and batch_size >= requested_batch_size:
            break
    if requested_batch_size:
        for batch_size, estimate in feasible_batches:
            if batch_size == requested_batch_size:
                return batch_size, estimate
    if not feasible_batches:
        return None, estimate_training_memory_gb(
            model_b=model_b,
            seq_len=seq_len,
            batch_size=1,
            method=method,
            load_in_4bit=load_in_4bit,
        )
    return feasible_batches[-1]


def _stage_hours_from_steps(
    *,
    step_count: Optional[int],
    throughput_score: float,
    gradient_accumulation: int,
    stage: str,
) -> Optional[float]:
    if not step_count or throughput_score <= 0:
        return None
    base_step_seconds = 12.0 if stage == "training" else 8.0
    seconds = step_count * max(gradient_accumulation, 1) * base_step_seconds / throughput_score
    return seconds / 3600.0


def _rank_estimates(rows: list[StageEstimate], optimize_for: str) -> list[StageEstimate]:
    feasible = [row for row in rows if row.feasible]
    infeasible = [row for row in rows if not row.feasible]

    if optimize_for == "speed":
        feasible.sort(key=lambda row: (-row.throughput_score, row.price_hr, -row.estimated_headroom_gb))
    elif optimize_for == "cost":
        feasible.sort(
            key=lambda row: (
                row.estimated_cost
                if row.estimated_cost is not None
                else row.price_hr / max(row.throughput_score, 0.001),
                row.price_hr,
                -row.throughput_score,
            )
        )
    else:
        if feasible:
            cheapest = min(feasible, key=lambda row: row.price_hr)
            cheapest_throughput = max(cheapest.throughput_score, 0.001)

            def _balanced_key(row: StageEstimate) -> tuple[float, float, float]:
                speedup = row.throughput_score / cheapest_throughput
                cost_multiple = row.price_hr / max(cheapest.price_hr, 0.001)
                # Default to the cheapest feasible option unless a pricier tier
                # clearly outperforms its own price multiplier.
                insufficient_gain = speedup < (cost_multiple * 1.15)
                if insufficient_gain:
                    return (1.0, row.price_hr, -row.throughput_score)
                return (0.0, -(speedup / max(cost_multiple, 0.001)), row.price_hr)

            feasible.sort(key=_balanced_key)
    infeasible.sort(key=lambda row: (row.estimated_headroom_gb, row.price_hr), reverse=True)
    return feasible + infeasible


def _stage_supports_flavor(*, stage: str, row: HardwareFlavor, spec: ExperimentSpec) -> tuple[bool, str]:
    if row.gpus > 1:
        return False, "multi-GPU flavors are not wired for this stage yet"
    return True, ""


def plan_stage_hardware(
    *,
    spec: ExperimentSpec,
    rows: list[HardwareFlavor],
    stage: str,
    optimize_for: str = "balanced",
    max_hourly_price: Optional[float] = None,
) -> StagePlan:
    model_b = parse_model_params_billions(spec.training.model_name) or 4.0
    filtered_rows = [
        row
        for row in rows
        if row.gpu_model and row.vram_gb > 0 and (max_hourly_price is None or row.price_hr <= max_hourly_price)
    ]

    estimates: list[StageEstimate] = []
    if stage == "training":
        requested_batch = spec.training.batch_size
        target_effective = (spec.training.batch_size or 0) * (spec.training.gradient_accumulation or 0) or _default_effective_batch(spec.method)
        seq_len = spec.training.max_seq_length or 2048
        for row in filtered_rows:
            supported, unsupported_reason = _stage_supports_flavor(stage=stage, row=row, spec=spec)
            batch_size, memory_estimate = _recommend_training_microbatch(
                model_b=model_b,
                seq_len=seq_len,
                method=spec.method,
                load_in_4bit=bool(spec.training.load_in_4bit),
                vram_gb=row.vram_gb,
                requested_batch_size=requested_batch,
            )
            feasible = batch_size is not None
            grad_acc = math.ceil(target_effective / batch_size) if feasible and batch_size else None
            throughput = _gpu_speed_factor(row.gpu_model, stage=stage) * ((batch_size or 1) ** 0.70) / max(grad_acc or 1, 1)
            hours = _stage_hours_from_steps(
                step_count=spec.training.max_steps,
                throughput_score=throughput,
                gradient_accumulation=grad_acc or 1,
                stage=stage,
            )
            cost = hours * row.price_hr if hours is not None else None
            headroom = _usable_vram_gb(row.vram_gb) - memory_estimate
            if feasible and (headroom < 1.5 or (grad_acc or 1) > 16):
                feasible = False
            if feasible and not supported:
                feasible = False
            if feasible:
                reason = "fits estimated training footprint"
            elif unsupported_reason:
                reason = unsupported_reason
            elif headroom < 1.5:
                reason = "estimated VRAM headroom is too tight for stable training"
            elif (grad_acc or 1) > 16:
                reason = "would require excessive gradient accumulation to stay within memory"
            else:
                reason = "estimated to exceed safe VRAM headroom"
            estimates.append(
                StageEstimate(
                    stage=stage,
                    flavor=row.flavor,
                    pretty_name=row.pretty_name,
                    feasible=feasible,
                    reason=reason,
                    gpu_model=row.gpu_model,
                    vram_gb=row.vram_gb,
                    price_hr=row.price_hr,
                    recommended_batch_size=batch_size,
                    recommended_gradient_accumulation=grad_acc,
                    estimated_memory_gb=memory_estimate,
                    estimated_headroom_gb=headroom,
                    throughput_score=throughput,
                    score_per_dollar=throughput / row.price_hr if row.price_hr else 0.0,
                    estimated_hours=hours,
                    estimated_cost=cost,
                )
            )
    elif stage == "evaluation":
        runtime = spec.evaluation.runtime or "unsloth"
        requested_gpu = spec.evaluation.gpu
        memory_estimate = estimate_eval_memory_gb(model_b=model_b, runtime=runtime)
        for row in filtered_rows:
            supported, unsupported_reason = _stage_supports_flavor(stage=stage, row=row, spec=spec)
            feasible = memory_estimate <= _usable_vram_gb(row.vram_gb)
            throughput = _gpu_speed_factor(row.gpu_model, stage=stage)
            hours = None
            if spec.evaluation.preset == "quick":
                hours = 0.35 / max(throughput, 0.1)
            elif spec.evaluation.enabled:
                hours = 0.9 / max(throughput, 0.1)
            cost = hours * row.price_hr if hours is not None else None
            if requested_gpu and row.flavor != requested_gpu:
                feasible = False
            if feasible and not supported:
                feasible = False
            headroom = _usable_vram_gb(row.vram_gb) - memory_estimate
            estimates.append(
                StageEstimate(
                    stage=stage,
                    flavor=row.flavor,
                    pretty_name=row.pretty_name,
                    feasible=feasible,
                    reason=(
                        "fits eval runtime footprint"
                        if feasible
                        else unsupported_reason or "estimated to exceed eval VRAM headroom"
                    ),
                    gpu_model=row.gpu_model,
                    vram_gb=row.vram_gb,
                    price_hr=row.price_hr,
                    recommended_batch_size=None,
                    recommended_gradient_accumulation=None,
                    estimated_memory_gb=memory_estimate,
                    estimated_headroom_gb=headroom,
                    throughput_score=throughput,
                    score_per_dollar=throughput / row.price_hr if row.price_hr else 0.0,
                    estimated_hours=hours,
                    estimated_cost=cost,
                )
            )
    elif stage == "loss":
        seq_len = spec.loss.max_seq_length or spec.training.max_seq_length or 2048
        memory_estimate = estimate_loss_memory_gb(model_b=model_b, seq_len=seq_len)
        for row in filtered_rows:
            supported, unsupported_reason = _stage_supports_flavor(stage=stage, row=row, spec=spec)
            feasible = memory_estimate <= _usable_vram_gb(row.vram_gb)
            throughput = _gpu_speed_factor(row.gpu_model, stage=stage) * max(row.vram_gb / 24.0, 1.0) ** 0.35
            hours = 0.75 / max(throughput, 0.1) if spec.loss.enabled else None
            cost = hours * row.price_hr if hours is not None else None
            if feasible and not supported:
                feasible = False
            headroom = _usable_vram_gb(row.vram_gb) - memory_estimate
            estimates.append(
                StageEstimate(
                    stage=stage,
                    flavor=row.flavor,
                    pretty_name=row.pretty_name,
                    feasible=feasible,
                    reason=(
                        "fits exact loss scorer footprint"
                        if feasible
                        else unsupported_reason or "estimated to exceed loss-stage VRAM headroom"
                    ),
                    gpu_model=row.gpu_model,
                    vram_gb=row.vram_gb,
                    price_hr=row.price_hr,
                    recommended_batch_size=None,
                    recommended_gradient_accumulation=None,
                    estimated_memory_gb=memory_estimate,
                    estimated_headroom_gb=headroom,
                    throughput_score=throughput,
                    score_per_dollar=throughput / row.price_hr if row.price_hr else 0.0,
                    estimated_hours=hours,
                    estimated_cost=cost,
                )
            )
    else:
        raise ValueError(f"Unsupported stage for hardware planning: {stage}")

    return StagePlan(
        stage=stage,
        optimize_for=optimize_for,
        model_name=spec.training.model_name,
        rows=_rank_estimates(estimates, optimize_for),
    )


def plan_experiment_hardware(
    *,
    spec: ExperimentSpec,
    optimize_for: str = "balanced",
    max_hourly_price: Optional[float] = None,
    stages: Optional[Iterable[str]] = None,
    rows: Optional[list[HardwareFlavor]] = None,
) -> dict[str, StagePlan]:
    live_rows = rows or load_live_hf_hardware_rows()
    selected = list(stages or ["training", "evaluation", "loss"])
    plans: dict[str, StagePlan] = {}
    if "training" in selected:
        plans["training"] = plan_stage_hardware(
            spec=spec,
            rows=live_rows,
            stage="training",
            optimize_for=optimize_for,
            max_hourly_price=max_hourly_price,
        )
    if spec.evaluation.enabled and "evaluation" in selected:
        plans["evaluation"] = plan_stage_hardware(
            spec=spec,
            rows=live_rows,
            stage="evaluation",
            optimize_for=optimize_for,
            max_hourly_price=max_hourly_price,
        )
    if spec.loss.enabled and "loss" in selected:
        plans["loss"] = plan_stage_hardware(
            spec=spec,
            rows=live_rows,
            stage="loss",
            optimize_for=optimize_for,
            max_hourly_price=max_hourly_price,
        )
    return plans


def format_stage_plan_json(stage_plan: StagePlan) -> dict[str, Any]:
    return {
        "stage": stage_plan.stage,
        "optimize_for": stage_plan.optimize_for,
        "model_name": stage_plan.model_name,
        "recommendation": asdict(stage_plan.recommendation) if stage_plan.recommendation else None,
        "alternatives": [asdict(row) for row in stage_plan.rows[:5]],
    }
