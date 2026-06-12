import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from Evaluator import cloud_hf_job_vllm
from Evaluator.cloud_hf_job_vllm import _compute_exact_loss_outputs, _load_base_model_name, _parse_args
from shared.cloud_stage_logging import StageLogger
from shared.experiment_tracking.schema import LossResult


def test_load_base_model_name_reads_adapter_config(tmp_path: Path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"Qwen/Qwen3-4B"}',
        encoding="utf-8",
    )

    assert _load_base_model_name(model_dir) == "Qwen/Qwen3-4B"


def test_parse_args_supports_tensor_parallel_and_loss_workers(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "cloud_hf_job_vllm.py",
            "--bucket-id",
            "bucket",
            "--run-prefix",
            "runs/hf_jobs/sft/demo",
            "--eval-prefix",
            "runs/hf_jobs/sft/demo/evaluations/vllm/ts",
            "--vllm-tensor-parallel-size",
            "4",
            "--loss-workers",
            "2",
        ],
    )

    args = _parse_args()

    assert args.vllm_tensor_parallel_size == 4
    assert args.loss_workers == 2


def test_compute_exact_loss_outputs_emits_progress_events(tmp_path: Path):
    args = Namespace(
        loss_dataset_path=str(tmp_path / "dataset.jsonl"),
        loss_dataset_name=None,
        loss_dataset_file=None,
        loss_max_seq_length=2048,
        loss_no_completion_only=False,
        loss_workers=0,
    )
    Path(args.loss_dataset_path).write_text('{"messages":[]}\n', encoding="utf-8")
    results_dir = tmp_path / "results"
    stage_logger = StageLogger(results_dir / "logs", stage="evaluation", provider="hf_jobs", run_prefix="runs/demo")

    class _Syncer:
        def __init__(self) -> None:
            self.calls = 0

        def sync_once(self) -> None:
            self.calls += 1

    syncer = _Syncer()

    def _fake_compute(**kwargs):
        summary_path = Path(kwargs["output_root"]) / "partial" / "loss_summary.partial.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"rows_written": 2, "batch_count": 1}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        kwargs["on_aggregate"](Path(kwargs["output_root"]))
        return [
            LossResult(index=0, loss=0.3, num_completion_tokens=1, num_total_tokens=2, jsonl_hash="aaaa1111"),
            LossResult(index=1, loss=0.2, num_completion_tokens=1, num_total_tokens=2, jsonl_hash="bbbb2222"),
        ]

    with patch("Evaluator.cloud_hf_job_vllm.compute_per_example_losses_parallel", side_effect=_fake_compute):
        _compute_exact_loss_outputs(
            args=args,
            model_dir=tmp_path / "model",
            results_dir=results_dir,
            hf_token="hf-token",
            progress_syncer=syncer,
            stage_logger=stage_logger,
        )

    summary = json.loads((results_dir / "logs" / "stage_summary.json").read_text(encoding="utf-8"))
    assert syncer.calls >= 2
    assert summary["event"] == "progress"
    assert summary["details"]["phase"] == "exact_loss"
    assert summary["details"]["examples_done"] == 2


def test_main_installs_termination_handler_before_initial_download(tmp_path: Path):
    args = Namespace(
        bucket_id="bucket-id",
        run_prefix="runs/demo",
        eval_prefix="runs/demo/evaluations/vllm/ts",
        config_dir="Evaluator/config",
        output_root=str(tmp_path / "eval_outputs"),
        preset=None,
        scenarios=None,
        tags=None,
        env_backend="none",
        env_template=None,
        env_tool_schema=None,
        env_exec_config=None,
        upload_to_hf=None,
        update_model_card=False,
        with_loss=False,
        loss_dataset_path=None,
        loss_dataset_name=None,
        loss_dataset_file=None,
        loss_max_seq_length=2048,
        loss_no_completion_only=False,
        vllm_host="127.0.0.1",
        vllm_port=8000,
        vllm_timeout=600,
        vllm_gpu_memory_utilization=0.85,
        vllm_tensor_parallel_size=0,
        loss_workers=0,
    )
    installed = []

    def _fake_install(**kwargs):
        installed.append(kwargs.get("progress_syncer"))

    def _fake_sync_from_bucket(*_args, **_kwargs):
        assert installed == [None]
        raise RuntimeError("stop after bootstrap ordering check")

    with patch.object(cloud_hf_job_vllm, "_parse_args", return_value=args), patch.object(
        cloud_hf_job_vllm, "get_hf_token", return_value="hf-token"
    ), patch.object(cloud_hf_job_vllm, "_log_runtime_versions"), patch.object(
        cloud_hf_job_vllm, "_install_termination_handler", side_effect=_fake_install
    ), patch.object(
        cloud_hf_job_vllm, "_sync_from_bucket", side_effect=_fake_sync_from_bucket
    ):
        with pytest.raises(RuntimeError, match="stop after bootstrap ordering check"):
            cloud_hf_job_vllm.main()
