import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from shared.experiment_tracking.schema import LossResult


def test_cloud_loss_job_uses_transformers_loader_for_exact_loss(tmp_path: Path):
    import shared.experiment_tracking.cloud_loss_job as cloud_loss_job

    args = Namespace(
        bucket_id="test-bucket",
        run_prefix="runs/hf_jobs/sft/20260321_221651-57b16b6c",
        dataset_path=str(tmp_path / "dataset.jsonl"),
        dataset_name=None,
        dataset_file=None,
        results_prefix=None,
        output_root=str(tmp_path / "loss_outputs"),
        max_seq_length=2048,
        batch_max_tokens=512,
        max_batch_size=2,
        sync_every_batches=1,
        num_workers=2,
        aggregate_interval_seconds=1.0,
        no_completion_only=False,
    )
    Path(args.dataset_path).write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n',
        encoding="utf-8",
    )

    losses = [LossResult(index=0, loss=0.3, num_completion_tokens=1, num_total_tokens=2, jsonl_hash="abcd1234")]

    with patch.object(cloud_loss_job, "_parse_args", return_value=args):
        with patch.object(cloud_loss_job, "get_hf_token", return_value="hf-token"):
            with patch.object(cloud_loss_job, "_sync_from_bucket") as mock_sync_from_bucket:
                with patch.object(cloud_loss_job, "_sync_bucket") as mock_sync_bucket:
                    def _fake_compute(**kwargs):
                        summary_path = Path(kwargs["output_root"]) / "partial" / "loss_summary.partial.json"
                        summary_path.parent.mkdir(parents=True, exist_ok=True)
                        summary_path.write_text(
                            json.dumps({"rows_written": 1, "batch_count": 1, "mean_loss": 0.3}, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                        kwargs["on_aggregate"](Path(kwargs["output_root"]))
                        return losses

                    with patch(
                        "shared.experiment_tracking.per_example_loss.compute_per_example_losses_parallel",
                        side_effect=_fake_compute,
                    ) as mock_compute:
                            exit_code = cloud_loss_job.main()

    assert exit_code == 0
    mock_compute.assert_called_once()
    assert mock_compute.call_args.kwargs["num_workers"] == 2
    assert mock_sync_from_bucket.call_count == 1
    assert mock_sync_bucket.call_count >= 1
    summary = json.loads((Path(args.output_root) / "results" / "logs" / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["details"]["rows_written"] == 1
    assert summary["details"]["batch_count"] == 1


def test_materialize_hf_dataset_downloads_plain_dataset_file(tmp_path: Path):
    import shared.experiment_tracking.cloud_loss_job as cloud_loss_job

    downloaded = tmp_path / "downloaded.jsonl"
    downloaded.write_text('{"messages":[]}\n', encoding="utf-8")

    with patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)) as mock_download:
        materialized = cloud_loss_job._materialize_hf_dataset(
            "professorsynapse/claudesidian-synthetic-dataset",
            "train.jsonl",
            tmp_path / "out",
            token="hf-token",
        )

    assert materialized.read_text(encoding="utf-8") == '{"messages":[]}\n'
    assert materialized.name == "train.jsonl"
    mock_download.assert_called_once_with(
        repo_id="professorsynapse/claudesidian-synthetic-dataset",
        filename="train.jsonl",
        repo_type="dataset",
        token="hf-token",
    )


def test_cloud_loss_job_tolerates_partial_sync_failure(tmp_path: Path):
    import shared.experiment_tracking.cloud_loss_job as cloud_loss_job

    args = Namespace(
        bucket_id="test-bucket",
        run_prefix="runs/hf_jobs/sft/test-run",
        dataset_path=str(tmp_path / "dataset.jsonl"),
        dataset_name=None,
        dataset_file=None,
        results_prefix=None,
        output_root=str(tmp_path / "loss_outputs"),
        max_seq_length=2048,
        batch_max_tokens=512,
        max_batch_size=2,
        sync_every_batches=1,
        num_workers=2,
        aggregate_interval_seconds=1.0,
        no_completion_only=False,
    )
    Path(args.dataset_path).write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n',
        encoding="utf-8",
    )

    losses = [LossResult(index=0, loss=0.3, num_completion_tokens=1, num_total_tokens=2, jsonl_hash="abcd1234")]

    with patch.object(cloud_loss_job, "_parse_args", return_value=args):
        with patch.object(cloud_loss_job, "get_hf_token", return_value="hf-token"):
            with patch.object(cloud_loss_job, "_sync_from_bucket"):
                sync_results = [False, True, True]
                with patch.object(cloud_loss_job, "_sync_bucket", side_effect=lambda *a, **k: sync_results.pop(0)) as mock_sync_bucket:
                    def _fake_compute(**kwargs):
                        summary_path = Path(kwargs["output_root"]) / "partial" / "loss_summary.partial.json"
                        summary_path.parent.mkdir(parents=True, exist_ok=True)
                        summary_path.write_text(
                            json.dumps({"rows_written": 1, "batch_count": 1, "mean_loss": 0.3}, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                        kwargs["on_aggregate"](Path(kwargs["output_root"]))
                        return losses

                    with patch(
                        "shared.experiment_tracking.per_example_loss.compute_per_example_losses_parallel",
                        side_effect=_fake_compute,
                    ):
                        exit_code = cloud_loss_job.main()

    assert exit_code == 0
    assert mock_sync_bucket.call_count == 3
    events = (Path(args.output_root) / "results" / "logs" / "stage_events.jsonl").read_text(encoding="utf-8")
    assert "sync_degraded" in events
    summary = json.loads((Path(args.output_root) / "results" / "logs" / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
