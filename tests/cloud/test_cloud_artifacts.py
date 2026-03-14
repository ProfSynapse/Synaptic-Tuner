from pathlib import Path
from unittest.mock import patch

from shared.cloud_artifacts import (
    HFBucketSyncCallback,
    build_manifest,
    build_run_paths,
    write_manifest,
)


def test_build_run_paths_uses_canonical_layout(tmp_path):
    paths = build_run_paths(tmp_path / "outputs", "modal", "sft", "20260314_120000", "abcdef123456")

    assert paths.run_dir == tmp_path / "outputs" / "runs" / "modal" / "sft" / "20260314_120000-abcdef12"
    assert paths.checkpoints_dir == paths.run_dir / "checkpoints"
    assert paths.logs_dir == paths.run_dir / "logs"
    assert paths.final_model_dir == paths.run_dir / "final_model"
    assert paths.lineage_path == paths.run_dir / "training_lineage.json"
    assert paths.manifest_path == paths.run_dir / "manifest.json"


def test_build_manifest_records_provider_native_artifact_metadata(tmp_path):
    run_paths = build_run_paths(tmp_path, "runpod", "kto", "20260314_120000", "abcdef123456")

    manifest = build_manifest(
        provider="runpod",
        method="kto",
        artifact_backend="runpod_network_volume",
        artifact_identifier="runpod-vol-123",
        run_paths=run_paths,
        repo_branch="main",
        repo_commit="abcdef123456",
        publish_final_model=False,
        publish_target_repo=None,
        status="running",
    )

    assert manifest["provider"] == "runpod"
    assert manifest["artifact_backend"] == "runpod_network_volume"
    assert manifest["artifact_identifier"] == "runpod-vol-123"
    assert manifest["publish_final_model"] is False
    assert manifest["paths"]["run_dir"] == str(run_paths.run_dir)


def test_write_manifest_writes_stable_json(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    payload = {"status": "completed", "provider": "hf_jobs"}

    write_manifest(manifest_path, payload)

    contents = manifest_path.read_text()
    assert contents.endswith("\n")
    assert '"provider": "hf_jobs"' in contents
    assert '"status": "completed"' in contents


def test_hf_bucket_sync_callback_syncs_run_dir(tmp_path):
    callback = HFBucketSyncCallback(
        run_dir=tmp_path / "run",
        bucket_id="toolset-training-artifacts",
        prefix="runs/hf_jobs/sft/20260314_120000-abcdef12",
        token="hf_test_token",
    )

    with patch("shared.cloud_artifacts.sync_directory_to_hf_bucket") as mock_sync:
        callback.on_save(args=None, state=None, control=None)

    mock_sync.assert_called_once_with(
        Path(tmp_path / "run"),
        "toolset-training-artifacts",
        "runs/hf_jobs/sft/20260314_120000-abcdef12",
        token="hf_test_token",
    )
