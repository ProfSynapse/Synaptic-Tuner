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
    assert snapshot["gpu_memory_reserved_headroom_gb"] == 12.0
    assert snapshot["max_gpu_memory_reserved_headroom_gb"] == 10.0
    assert snapshot["max_gpu_memory_reserved_pct"] == 58.33
    assert snapshot["oom_risk_level"] == "low"
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
                '{"step": 5, "gpu_memory_reserved_gb": 10.0, "gpu_memory_allocated_gb": 8.0, "gpu_memory_reserved_headroom_gb": 14.0, "max_gpu_memory_reserved_gb": 10.0, "max_gpu_memory_reserved_pct": 41.67, "max_gpu_memory_reserved_headroom_gb": 14.0, "gpu_utilization_pct": 55.0, "samples_per_sec": 12.0, "steps_per_second": 1.5, "cloud_provider": "hf_jobs", "cloud_gpu_type": "a10g-small"}',
                '{"step": 10, "gpu_memory_reserved_gb": 14.0, "gpu_memory_allocated_gb": 11.0, "gpu_memory_reserved_headroom_gb": 10.0, "max_gpu_memory_reserved_gb": 18.0, "max_gpu_memory_reserved_pct": 75.0, "max_gpu_memory_reserved_headroom_gb": 6.0, "gpu_utilization_pct": 81.0, "samples_per_sec": 14.5, "steps_per_second": 1.8, "gpu_name": "NVIDIA A10G", "gpu_total_memory_gb": 24.0, "system_ram_total_gb": 44.0}',
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
    assert summary["peak_max_gpu_memory_reserved_gb"] == 18.0
    assert summary["peak_max_gpu_memory_reserved_pct"] == 75.0
    assert summary["min_gpu_memory_reserved_headroom_gb"] == 10.0
    assert summary["min_max_gpu_memory_reserved_headroom_gb"] == 6.0
    assert summary["latest_max_gpu_memory_reserved_headroom_gb"] == 6.0
    assert summary["oom_risk_level"] == "low"
    assert summary["latest_steps_per_second"] == 1.8
    assert summary["cloud_provider"] == "hf_jobs"
    assert summary["cloud_gpu_type"] == "a10g-small"


def test_build_capacity_feature_row():
    lineage = {
        "training_type": "SFT",
        "timestamp": "2026-03-15T00:00:00Z",
        "run_directory": "/tmp/run",
        "model": {
            "base_model": "unsloth/test-model",
            "max_seq_length": 2048,
            "load_in_4bit": False,
            "dtype": "bfloat16",
        },
        "lora": {
            "rank": 64,
            "alpha": 128,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
        },
        "training": {
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 32,
            "learning_rate": 2e-4,
            "num_epochs": 1,
            "max_steps": 294,
            "warmup_ratio": 0.02,
            "lr_scheduler": "linear",
            "optimizer": "adamw_8bit",
            "max_grad_norm": 1.0,
            "packing": False,
            "completion_only_loss": True,
            "gradient_checkpointing": True,
            "fp16": False,
            "bf16": True,
            "seed": 42,
        },
        "dataset": {
            "source": "dataset/path.jsonl",
            "train_examples": 1000,
            "eval_examples": 100,
        },
        "hardware": {
            "cloud_provider": "hf_jobs",
            "cloud_gpu_type": "a10g-large",
            "gpu_name": "NVIDIA A10G",
            "gpu_memory_gb": 23.9,
            "system_memory_gb": 186.7,
        },
        "capacity_profile": {
            "logged_steps": 58,
            "peak_max_gpu_memory_reserved_pct": 91.2,
            "min_max_gpu_memory_reserved_headroom_gb": 2.1,
            "peak_samples_per_sec": 14.5,
            "peak_steps_per_second": 1.8,
            "oom_risk_level": "moderate",
        },
        "results": {
            "final_step": 294,
            "total_epochs": 1.0,
            "final_loss": 0.6989,
            "training_time_seconds": 2241.2,
        },
    }

    row = training_capacity.build_capacity_feature_row(lineage)

    assert row["features_version"] == 1
    assert row["training_type"] == "SFT"
    assert row["model_load_in_4bit"] == 0
    assert row["training_gradient_checkpointing"] == 1
    assert row["lora_target_module_count"] == 3
    assert row["capacity_peak_max_gpu_memory_reserved_pct"] == 91.2
    assert row["capacity_min_max_gpu_memory_reserved_headroom_gb"] == 2.1
    assert row["oom_risk_level"] == "moderate"
    assert row["result_final_loss"] == 0.6989
