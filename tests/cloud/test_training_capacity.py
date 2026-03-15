from pathlib import Path

from shared import training_capacity


class _FakeCudaProps:
    name = "Fake GPU"
    total_memory = 24 * 1024 ** 3


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_properties(index):
        return _FakeCudaProps()

    @staticmethod
    def memory_reserved(index):
        return 12 * 1024 ** 3

    @staticmethod
    def memory_allocated(index):
        return 10 * 1024 ** 3

    @staticmethod
    def max_memory_reserved(index):
        return 14 * 1024 ** 3

    @staticmethod
    def max_memory_allocated(index):
        return 11 * 1024 ** 3

    @staticmethod
    def reset_peak_memory_stats():
        return None


class _FakeTorch:
    __version__ = "2.9.0"
    cuda = _FakeCuda()

    class version:
        cuda = "12.8"


def test_capture_runtime_capacity_snapshot(monkeypatch):
    monkeypatch.setattr(
        training_capacity,
        "_detect_system_memory_usage_bytes",
        lambda: {"total": 64 * 1024 ** 3, "used": 20 * 1024 ** 3, "available": 44 * 1024 ** 3},
    )
    monkeypatch.setattr(training_capacity, "_detect_process_rss_bytes", lambda: 3 * 1024 ** 3)
    monkeypatch.setattr(
        training_capacity,
        "_query_nvidia_smi",
        lambda: {
            "gpu_utilization_pct": 87.0,
            "gpu_vram_used_gb": 12.0,
            "gpu_vram_total_gb": 24.0,
            "gpu_vram_utilization_pct": 50.0,
        },
    )

    snapshot = training_capacity.capture_runtime_capacity_snapshot(_FakeTorch())

    assert snapshot["gpu_name"] == "Fake GPU"
    assert snapshot["gpu_total_memory_gb"] == 24.0
    assert snapshot["gpu_memory_reserved_gb"] == 12.0
    assert snapshot["gpu_memory_allocated_gb"] == 10.0
    assert snapshot["max_gpu_memory_reserved_gb"] == 14.0
    assert snapshot["gpu_utilization_pct"] == 87.0
    assert snapshot["system_ram_total_gb"] == 64.0
    assert snapshot["process_ram_gb"] == 3.0


def test_summarize_capacity_from_logs(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    log_file = logs_dir / "training_latest.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"step": 5, "gpu_memory_reserved_gb": 10.0, "gpu_memory_allocated_gb": 8.0, "gpu_utilization_pct": 55.0, "samples_per_sec": 12.0, "steps_per_second": 1.5, "cloud_provider": "hf_jobs", "cloud_gpu_type": "a10g-small"}',
                '{"step": 10, "gpu_memory_reserved_gb": 14.0, "gpu_memory_allocated_gb": 11.0, "gpu_utilization_pct": 81.0, "samples_per_sec": 14.5, "steps_per_second": 1.8, "gpu_name": "NVIDIA A10G", "gpu_total_memory_gb": 24.0, "system_ram_total_gb": 44.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = training_capacity.summarize_capacity_from_logs(logs_dir)

    assert summary["logged_steps"] == 2
    assert summary["peak_gpu_memory_reserved_gb"] == 14.0
    assert summary["peak_gpu_memory_allocated_gb"] == 11.0
    assert summary["peak_gpu_utilization_pct"] == 81.0
    assert summary["peak_samples_per_sec"] == 14.5
    assert summary["latest_steps_per_second"] == 1.8
    assert summary["cloud_provider"] == "hf_jobs"
    assert summary["cloud_gpu_type"] == "a10g-small"
