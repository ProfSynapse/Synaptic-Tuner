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
                }
            )
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

    latest = entries[-1]
    latest_fields = {
        "latest_gpu_memory_reserved_gb": "gpu_memory_reserved_gb",
        "latest_gpu_memory_allocated_gb": "gpu_memory_allocated_gb",
        "latest_gpu_utilization_pct": "gpu_utilization_pct",
        "latest_gpu_vram_utilization_pct": "gpu_vram_utilization_pct",
        "latest_samples_per_sec": "samples_per_sec",
        "latest_steps_per_second": "steps_per_second",
    }
    for output_key, input_key in latest_fields.items():
        value = _safe_float(latest.get(input_key))
        if value is not None:
            summary[output_key] = round(value, 3)

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
