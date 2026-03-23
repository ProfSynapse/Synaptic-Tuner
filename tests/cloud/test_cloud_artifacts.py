from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from shared.cloud_artifacts import (
    HFBucketSyncCallback,
    build_manifest,
    build_run_paths,
    ensure_hf_bucket,
    normalize_hf_bucket_id,
    sync_file_to_hf_bucket,
    sync_directory_to_hf_bucket,
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


def test_hf_bucket_sync_callback_syncs_checkpoint_dir(tmp_path):
    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_dir = checkpoints_dir / "checkpoint-100"
    checkpoint_dir.mkdir(parents=True)

    callback = HFBucketSyncCallback(
        run_dir=run_dir,
        bucket_id="toolset-training-artifacts",
        prefix="runs/hf_jobs/sft/20260314_120000-abcdef12",
        token="hf_test_token",
    )

    args = type("Args", (), {"output_dir": str(checkpoints_dir)})()
    state = type("State", (), {"global_step": 100})()

    with patch("shared.cloud_artifacts.sync_directory_to_hf_bucket") as mock_sync:
        callback.on_save(args=args, state=state, control=None)

    mock_sync.assert_called_once_with(
        checkpoint_dir,
        "toolset-training-artifacts",
        "runs/hf_jobs/sft/20260314_120000-abcdef12/checkpoints/checkpoint-100",
        token="hf_test_token",
    )


def test_hf_bucket_sync_callback_falls_back_to_run_dir(tmp_path):
    """When args/state are None, falls back to syncing entire run_dir."""
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


def test_hf_bucket_sync_callback_syncs_latest_log_on_log(tmp_path):
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "training_20260314_120000.jsonl").write_text("{\"step\": 5}\n", encoding="utf-8")

    callback = HFBucketSyncCallback(
        run_dir=run_dir,
        bucket_id="toolset-training-artifacts",
        prefix="runs/hf_jobs/sft/20260314_120000-abcdef12",
        token="hf_test_token",
        log_every_n_steps=5,
    )

    state = type("State", (), {"global_step": 5})()

    with patch("shared.cloud_artifacts.sync_directory_to_hf_bucket") as mock_sync:
        callback.on_log(args=None, state=state, control=None, logs={"loss": 1.23})

    mock_sync.assert_called_once_with(
        logs_dir,
        "toolset-training-artifacts",
        "runs/hf_jobs/sft/20260314_120000-abcdef12/logs",
        token="hf_test_token",
    )


def test_normalize_hf_bucket_id_strips_transport_prefixes():
    assert normalize_hf_bucket_id("hf://buckets/user/toolset") == "user/toolset"
    assert normalize_hf_bucket_id("buckets/user/toolset/") == "user/toolset"


def test_ensure_hf_bucket_returns_namespaced_bucket_id():
    mock_hub = ModuleType("huggingface_hub")

    def create_bucket(bucket_id, **kwargs):
        class BucketInfo:
            pass

        bucket = BucketInfo()
        bucket.bucket_id = f"test-user/{bucket_id.split('/')[-1]}"
        return bucket

    mock_hub.create_bucket = create_bucket

    with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
        assert ensure_hf_bucket("toolset-training-artifacts", token="hf_test_token") == "test-user/toolset-training-artifacts"


def test_sync_directory_to_hf_bucket_uses_resolved_bucket_id(tmp_path):
    local_dir = tmp_path / "run"
    local_dir.mkdir()

    mock_hub = ModuleType("huggingface_hub")
    calls = []

    def create_bucket(bucket_id, **kwargs):
        class BucketInfo:
            pass

        bucket = BucketInfo()
        bucket.bucket_id = f"test-user/{bucket_id.split('/')[-1]}"
        return bucket

    def sync_bucket(src, dst, token=None):
        calls.append((src, dst, token))

    mock_hub.sync_bucket = sync_bucket
    mock_hub.create_bucket = create_bucket

    with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
        sync_directory_to_hf_bucket(
            local_dir,
            "toolset-training-artifacts",
            "runs/hf_jobs/sft/20260314_120000-abcdef12",
            token="hf_test_token",
        )

    assert (
        str(local_dir),
        "hf://buckets/test-user/toolset-training-artifacts/runs/hf_jobs/sft/20260314_120000-abcdef12",
        "hf_test_token",
    ) in calls


def test_sync_file_to_hf_bucket_uses_resolved_bucket_id(tmp_path):
    local_file = tmp_path / "training.jsonl"
    local_file.write_text("{\"step\": 5}\n", encoding="utf-8")

    mock_hub = ModuleType("huggingface_hub")
    calls = []

    def create_bucket(bucket_id, **kwargs):
        class BucketInfo:
            pass

        bucket = BucketInfo()
        bucket.bucket_id = f"test-user/{bucket_id.split('/')[-1]}"
        return bucket

    def sync_bucket(src, dst, token=None):
        calls.append((src, dst, token))

    mock_hub.sync_bucket = sync_bucket
    mock_hub.create_bucket = create_bucket

    with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
        sync_file_to_hf_bucket(
            local_file,
            "toolset-training-artifacts",
            "runs/hf_jobs/sft/20260314_120000-abcdef12/logs/training.jsonl",
            token="hf_test_token",
        )

    assert (
        str(local_file.parent),
        "hf://buckets/test-user/toolset-training-artifacts/runs/hf_jobs/sft/20260314_120000-abcdef12/logs",
        "hf_test_token",
    ) in calls


def test_sync_directory_to_hf_bucket_raises_real_sync_error(tmp_path):
    local_dir = tmp_path / "run"
    local_dir.mkdir()

    mock_hub = ModuleType("huggingface_hub")

    def create_bucket(bucket_id, **kwargs):
        class BucketInfo:
            pass

        bucket = BucketInfo()
        bucket.bucket_id = f"test-user/{bucket_id.split('/')[-1]}"
        return bucket

    def sync_bucket(src, dst, token=None):
        raise RuntimeError("permission denied")

    mock_hub.sync_bucket = sync_bucket
    mock_hub.create_bucket = create_bucket

    with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
        with pytest.raises(RuntimeError, match="HF bucket sync failed"):
            sync_directory_to_hf_bucket(
                local_dir,
                "toolset-training-artifacts",
                "runs/hf_jobs/sft/20260314_120000-abcdef12",
                token="hf_test_token",
            )


def test_sync_directory_to_hf_bucket_helper_omits_blank_auth_env(tmp_path, monkeypatch):
    local_dir = tmp_path / "run"
    local_dir.mkdir()

    monkeypatch.setenv("HF_BUCKET_SYNC_PYTHON", "/usr/bin/python3")
    monkeypatch.setenv("HF_TOKEN", "   ")
    monkeypatch.setenv("HF_API_KEY", "")

    with patch("shared.cloud_artifacts.subprocess.run") as mock_run:
        sync_directory_to_hf_bucket(
            local_dir,
            "toolset-training-artifacts",
            "runs/hf_jobs/sft/20260314_120000-abcdef12",
        )

    helper_env = mock_run.call_args.kwargs["env"]
    assert "HF_TOKEN" not in helper_env
    assert "HF_API_KEY" not in helper_env
