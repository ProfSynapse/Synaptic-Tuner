"""
Runtime capacity profiling helpers for training runs.

These helpers collect lightweight hardware and memory metrics that are useful
for comparing training configurations across local and cloud runs.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _round_gb(value_bytes: Optional[float]) -> Optional[float]:
    if value_bytes is None:
        return None
    return round(float(value_bytes) / (1024 ** 3), 3)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_oom_risk(max_reserved_pct: Optional[float]) -> Optional[str]:
    if max_reserved_pct is None:
        return None
    if max_reserved_pct >= 97.0:
        return "critical"
    if max_reserved_pct >= 92.0:
        return "high"
    if max_reserved_pct >= 85.0:
        return "moderate"
    return "low"


def _coerce_bool_flag(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1
    if text in {"false", "0", "no", "n"}:
        return 0
    return None


def _get_nested(mapping: Dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _detect_total_system_memory_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size)
    except Exception:
        pass

    return None


def _detect_system_memory_usage_bytes() -> Dict[str, Optional[int]]:
    info = {
        "total": _detect_total_system_memory_bytes(),
        "used": None,
        "available": None,
    }

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["used"] = int(vm.used)
        info["available"] = int(vm.available)
        return info
    except Exception:
        return info


def _detect_process_rss_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss * 1024)
    except Exception:
        return None


def _query_nvidia_smi() -> Dict[str, Optional[float]]:
    """Query NVIDIA runtime utilization if `nvidia-smi` is available."""
    result = {
        "gpu_utilization_pct": None,
        "gpu_vram_used_gb": None,
        "gpu_vram_total_gb": None,
        "gpu_vram_utilization_pct": None,
    }

    query = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            query,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return result

    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return result

    gpu_util = _safe_float(parts[0])
    mem_used_mb = _safe_float(parts[1])
    mem_total_mb = _safe_float(parts[2])

    result["gpu_utilization_pct"] = gpu_util
    if mem_used_mb is not None:
        result["gpu_vram_used_gb"] = round(mem_used_mb / 1024, 3)
    if mem_total_mb is not None:
        result["gpu_vram_total_gb"] = round(mem_total_mb / 1024, 3)
    if mem_used_mb is not None and mem_total_mb:
        result["gpu_vram_utilization_pct"] = round((mem_used_mb / mem_total_mb) * 100, 2)

    return result


def reset_capacity_peaks(torch_module=None) -> None:
    """Reset peak CUDA memory stats at the start of a training run."""
    torch_module = torch_module or _import_torch()
    if torch_module is None or not torch_module.cuda.is_available():
        return

    try:
        torch_module.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def capture_runtime_capacity_snapshot(torch_module=None) -> Dict[str, Any]:
    """Capture current runtime capacity metrics for logging."""
    snapshot: Dict[str, Any] = {}

    memory_info = _detect_system_memory_usage_bytes()
    if memory_info["total"] is not None:
        snapshot["system_ram_total_gb"] = _round_gb(memory_info["total"])
    if memory_info["used"] is not None:
        snapshot["system_ram_used_gb"] = _round_gb(memory_info["used"])
    if memory_info["available"] is not None:
        snapshot["system_ram_available_gb"] = _round_gb(memory_info["available"])

    process_rss = _detect_process_rss_bytes()
    if process_rss is not None:
        snapshot["process_ram_gb"] = _round_gb(process_rss)

    torch_module = torch_module or _import_torch()
    if torch_module is not None and torch_module.cuda.is_available():
        try:
            props = torch_module.cuda.get_device_properties(0)
            total = float(props.total_memory)
            reserved = float(torch_module.cuda.memory_reserved(0))
            allocated = float(torch_module.cuda.memory_allocated(0))
            max_reserved = float(torch_module.cuda.max_memory_reserved(0))
            max_allocated = float(torch_module.cuda.max_memory_allocated(0))

            snapshot.update(
                {
                    "gpu_name": props.name,
                    "gpu_total_memory_gb": _round_gb(total),
                    # Keep gpu_memory_gb as the UI-compatible current reserved value.
                    "gpu_memory_gb": _round_gb(reserved),
                    "gpu_memory_reserved_gb": _round_gb(reserved),
                    "gpu_memory_allocated_gb": _round_gb(allocated),
                    "max_gpu_memory_reserved_gb": _round_gb(max_reserved),
                    "max_gpu_memory_allocated_gb": _round_gb(max_allocated),
                    "gpu_memory_reserved_pct": round((reserved / total) * 100, 2) if total else None,
                    "gpu_memory_allocated_pct": round((allocated / total) * 100, 2) if total else None,
                    "gpu_memory_reserved_headroom_gb": _round_gb(max(total - reserved, 0.0)),
                    "gpu_memory_allocated_headroom_gb": _round_gb(max(total - allocated, 0.0)),
                    "max_gpu_memory_reserved_pct": round((max_reserved / total) * 100, 2) if total else None,
                    "max_gpu_memory_allocated_pct": round((max_allocated / total) * 100, 2) if total else None,
                    "max_gpu_memory_reserved_headroom_gb": _round_gb(max(total - max_reserved, 0.0)),
                    "max_gpu_memory_allocated_headroom_gb": _round_gb(max(total - max_allocated, 0.0)),
                }
            )
            snapshot["oom_risk_level"] = _classify_oom_risk(snapshot.get("max_gpu_memory_reserved_pct"))
        except Exception:
            pass

    snapshot.update(
        {
            key: value
            for key, value in _query_nvidia_smi().items()
            if value is not None
        }
    )

    return snapshot


def capture_hardware_info(torch_module=None) -> Dict[str, Any]:
    """Capture static hardware and cloud metadata for lineage."""
    torch_module = torch_module or _import_torch()

    hardware_info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "cuda_available": bool(torch_module is not None and torch_module.cuda.is_available()),
    }

    total_ram = _detect_total_system_memory_bytes()
    if total_ram is not None:
        hardware_info["system_memory_gb"] = _round_gb(total_ram)

    cloud_provider = os.environ.get("CLOUD_PROVIDER", "").strip()
    cloud_gpu_type = os.environ.get("CLOUD_GPU_TYPE", "").strip()
    if cloud_provider:
        hardware_info["cloud_provider"] = cloud_provider
    if cloud_gpu_type:
        hardware_info["cloud_gpu_type"] = cloud_gpu_type

    if torch_module is not None:
        hardware_info["pytorch_version"] = getattr(torch_module, "__version__", "unknown")
        if torch_module.cuda.is_available():
            hardware_info.update(
                {
                    "cuda_version": getattr(torch_module.version, "cuda", None),
                    "gpu_name": torch_module.cuda.get_device_name(0),
                    "gpu_memory_gb": round(
                        torch_module.cuda.get_device_properties(0).total_memory / 1e9, 1
                    ),
                }
            )

    return hardware_info


def summarize_capacity_from_logs(logs_dir: Path) -> Dict[str, Any]:
    """Summarize persisted capacity metrics from a training log directory."""
    logs_dir = Path(logs_dir)
    if not logs_dir.exists():
        return {}

    candidates: Iterable[Path] = []
    latest_link = logs_dir / "training_latest.jsonl"
    if latest_link.exists():
        candidates = [latest_link]
    else:
        candidates = sorted(logs_dir.glob("training_*.jsonl"))

    entries = []
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                entries = [json.loads(line) for line in handle if line.strip()]
            if entries:
                break
        except Exception:
            continue

    if not entries:
        return {}

    summary: Dict[str, Any] = {
        "logged_steps": len(entries),
    }

    peak_fields = {
        "peak_gpu_memory_reserved_gb": "gpu_memory_reserved_gb",
        "peak_gpu_memory_allocated_gb": "gpu_memory_allocated_gb",
        "peak_gpu_memory_reserved_pct": "gpu_memory_reserved_pct",
        "peak_gpu_memory_allocated_pct": "gpu_memory_allocated_pct",
        "peak_max_gpu_memory_reserved_gb": "max_gpu_memory_reserved_gb",
        "peak_max_gpu_memory_allocated_gb": "max_gpu_memory_allocated_gb",
        "peak_max_gpu_memory_reserved_pct": "max_gpu_memory_reserved_pct",
        "peak_max_gpu_memory_allocated_pct": "max_gpu_memory_allocated_pct",
        "peak_gpu_utilization_pct": "gpu_utilization_pct",
        "peak_gpu_vram_utilization_pct": "gpu_vram_utilization_pct",
        "peak_process_ram_gb": "process_ram_gb",
        "peak_system_ram_used_gb": "system_ram_used_gb",
        "peak_samples_per_sec": "samples_per_sec",
        "peak_steps_per_second": "steps_per_second",
    }

    for output_key, input_key in peak_fields.items():
        values = [_safe_float(entry.get(input_key)) for entry in entries]
        values = [value for value in values if value is not None]
        if values:
            summary[output_key] = round(max(values), 3)

    min_fields = {
        "min_gpu_memory_reserved_headroom_gb": "gpu_memory_reserved_headroom_gb",
        "min_gpu_memory_allocated_headroom_gb": "gpu_memory_allocated_headroom_gb",
        "min_max_gpu_memory_reserved_headroom_gb": "max_gpu_memory_reserved_headroom_gb",
        "min_max_gpu_memory_allocated_headroom_gb": "max_gpu_memory_allocated_headroom_gb",
    }
    for output_key, input_key in min_fields.items():
        values = [_safe_float(entry.get(input_key)) for entry in entries]
        values = [value for value in values if value is not None]
        if values:
            summary[output_key] = round(min(values), 3)

    latest = entries[-1]
    latest_fields = {
        "latest_gpu_memory_reserved_gb": "gpu_memory_reserved_gb",
        "latest_gpu_memory_allocated_gb": "gpu_memory_allocated_gb",
        "latest_gpu_memory_reserved_headroom_gb": "gpu_memory_reserved_headroom_gb",
        "latest_gpu_memory_allocated_headroom_gb": "gpu_memory_allocated_headroom_gb",
        "latest_max_gpu_memory_reserved_gb": "max_gpu_memory_reserved_gb",
        "latest_max_gpu_memory_reserved_pct": "max_gpu_memory_reserved_pct",
        "latest_max_gpu_memory_reserved_headroom_gb": "max_gpu_memory_reserved_headroom_gb",
        "latest_gpu_utilization_pct": "gpu_utilization_pct",
        "latest_gpu_vram_utilization_pct": "gpu_vram_utilization_pct",
        "latest_samples_per_sec": "samples_per_sec",
        "latest_steps_per_second": "steps_per_second",
    }
    for output_key, input_key in latest_fields.items():
        value = _safe_float(latest.get(input_key))
        if value is not None:
            summary[output_key] = round(value, 3)

    risk_pct = summary.get("peak_max_gpu_memory_reserved_pct")
    if risk_pct is None:
        risk_pct = summary.get("peak_gpu_memory_reserved_pct")
    risk_level = _classify_oom_risk(_safe_float(risk_pct))
    if risk_level:
        summary["oom_risk_level"] = risk_level

    for passthrough_key in (
        "gpu_name",
        "gpu_total_memory_gb",
        "system_ram_total_gb",
        "cloud_provider",
        "cloud_gpu_type",
    ):
        value = latest.get(passthrough_key)
        if value in (None, "", []):
            for entry in reversed(entries[:-1]):
                candidate = entry.get(passthrough_key)
                if candidate not in (None, "", []):
                    value = candidate
                    break
        if value not in (None, "", []):
            summary[passthrough_key] = value

    return summary


def build_capacity_feature_row(lineage: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten lineage into a model-friendly per-run feature row."""
    if not isinstance(lineage, dict):
        return {}

    model = lineage.get("model") if isinstance(lineage.get("model"), dict) else {}
    lora = lineage.get("lora") if isinstance(lineage.get("lora"), dict) else {}
    training = lineage.get("training") if isinstance(lineage.get("training"), dict) else {}
    dataset = lineage.get("dataset") if isinstance(lineage.get("dataset"), dict) else {}
    hardware = lineage.get("hardware") if isinstance(lineage.get("hardware"), dict) else {}
    capacity = lineage.get("capacity_profile") if isinstance(lineage.get("capacity_profile"), dict) else {}
    results = lineage.get("results") if isinstance(lineage.get("results"), dict) else {}

    row: Dict[str, Any] = {
        "features_version": 1,
        "training_type": lineage.get("training_type"),
        "timestamp": lineage.get("timestamp"),
        "run_directory": lineage.get("run_directory"),
        "status_completed": 1,
        "oom_observed": 0,
        "oom_risk_level": capacity.get("oom_risk_level"),
        "model_base_model": model.get("base_model"),
        "model_max_seq_length": model.get("max_seq_length"),
        "model_load_in_4bit": _coerce_bool_flag(model.get("load_in_4bit")),
        "model_dtype": model.get("dtype"),
        "lora_rank": lora.get("rank"),
        "lora_alpha": lora.get("alpha"),
        "lora_dropout": lora.get("dropout"),
        "lora_target_module_count": len(lora.get("target_modules", [])) if isinstance(lora.get("target_modules"), list) else None,
        "training_batch_size": training.get("batch_size"),
        "training_gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "training_effective_batch_size": training.get("effective_batch_size"),
        "training_learning_rate": training.get("learning_rate"),
        "training_num_epochs": training.get("num_epochs"),
        "training_max_steps": training.get("max_steps"),
        "training_warmup_ratio": training.get("warmup_ratio"),
        "training_lr_scheduler": training.get("lr_scheduler"),
        "training_optimizer": training.get("optimizer"),
        "training_max_grad_norm": training.get("max_grad_norm"),
        "training_packing": _coerce_bool_flag(training.get("packing")),
        "training_completion_only_loss": _coerce_bool_flag(training.get("completion_only_loss")),
        "training_gradient_checkpointing": _coerce_bool_flag(training.get("gradient_checkpointing")),
        "training_fp16": _coerce_bool_flag(training.get("fp16")),
        "training_bf16": _coerce_bool_flag(training.get("bf16")),
        "training_seed": training.get("seed"),
        "dataset_train_examples": dataset.get("train_examples"),
        "dataset_eval_examples": dataset.get("eval_examples"),
        "dataset_source": dataset.get("source"),
        "hardware_cloud_provider": hardware.get("cloud_provider"),
        "hardware_cloud_gpu_type": hardware.get("cloud_gpu_type"),
        "hardware_gpu_name": hardware.get("gpu_name"),
        "hardware_gpu_memory_gb": hardware.get("gpu_memory_gb"),
        "hardware_system_memory_gb": hardware.get("system_memory_gb"),
        "capacity_logged_steps": capacity.get("logged_steps"),
        "capacity_peak_gpu_memory_reserved_gb": capacity.get("peak_gpu_memory_reserved_gb"),
        "capacity_peak_gpu_memory_reserved_pct": capacity.get("peak_gpu_memory_reserved_pct"),
        "capacity_peak_max_gpu_memory_reserved_gb": capacity.get("peak_max_gpu_memory_reserved_gb"),
        "capacity_peak_max_gpu_memory_reserved_pct": capacity.get("peak_max_gpu_memory_reserved_pct"),
        "capacity_min_gpu_memory_reserved_headroom_gb": capacity.get("min_gpu_memory_reserved_headroom_gb"),
        "capacity_min_max_gpu_memory_reserved_headroom_gb": capacity.get("min_max_gpu_memory_reserved_headroom_gb"),
        "capacity_peak_samples_per_sec": capacity.get("peak_samples_per_sec"),
        "capacity_peak_steps_per_second": capacity.get("peak_steps_per_second"),
        "capacity_latest_max_gpu_memory_reserved_pct": capacity.get("latest_max_gpu_memory_reserved_pct"),
        "capacity_latest_max_gpu_memory_reserved_headroom_gb": capacity.get("latest_max_gpu_memory_reserved_headroom_gb"),
        "result_final_step": results.get("final_step"),
        "result_total_epochs": results.get("total_epochs"),
        "result_final_loss": results.get("final_loss"),
        "result_training_time_seconds": results.get("training_time_seconds"),
    }

    # Method-specific knobs that are useful predictors.
    if "max_length" in training:
        row["training_max_length"] = training.get("max_length")
    if "max_prompt_length" in training:
        row["training_max_prompt_length"] = training.get("max_prompt_length")
    if "beta" in training:
        row["training_beta"] = training.get("beta")
    if "desirable_weight" in training:
        row["training_desirable_weight"] = training.get("desirable_weight")
    if "undesirable_weight" in training:
        row["training_undesirable_weight"] = training.get("undesirable_weight")
    if "use_kto_s" in training:
        row["training_use_kto_s"] = _coerce_bool_flag(training.get("use_kto_s"))

    row = {key: value for key, value in row.items() if value is not None}
    return row
