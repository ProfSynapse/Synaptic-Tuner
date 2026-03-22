from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

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
        no_completion_only=False,
    )
    Path(args.dataset_path).write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n',
        encoding="utf-8",
    )

    model = MagicMock()
    model.parameters.return_value = iter([torch.tensor(1.0)])
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "<eos>"
    losses = [LossResult(index=0, loss=0.3, num_completion_tokens=1, num_total_tokens=2, jsonl_hash="abcd1234")]

    with patch.object(cloud_loss_job, "_parse_args", return_value=args):
        with patch.object(cloud_loss_job, "get_hf_token", return_value="hf-token"):
            with patch.object(cloud_loss_job, "_sync_from_bucket") as mock_sync_from_bucket:
                with patch.object(cloud_loss_job, "_sync_bucket") as mock_sync_bucket:
                    with patch(
                        "shared.experiment_tracking.transformers_loss_loader.load_transformers_loss_model",
                        return_value=(model, tokenizer),
                    ) as mock_loader:
                        with patch(
                            "shared.experiment_tracking.per_example_loss.compute_per_example_losses",
                            return_value=losses,
                        ) as mock_compute:
                            exit_code = cloud_loss_job.main()

    assert exit_code == 0
    mock_loader.assert_called_once()
    mock_compute.assert_called_once()
    assert mock_sync_from_bucket.call_count == 1
    assert mock_sync_bucket.call_count >= 1
