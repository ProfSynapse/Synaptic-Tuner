"""Tests for the remote-container env defaults in Trainers/cloud/train_modal.py.

Focus: apply_hf_xet_mitigation, which sets the hf_xet-hang workaround defaults
on the remote training container while still honoring an explicit override
forwarded from the local launch env.
"""

import pytest

# train_modal defines a Modal @app.function, so importing it needs the modal
# package; skip cleanly where modal is unavailable rather than erroring.
modal = pytest.importorskip("modal")

from Trainers.cloud.train_modal import (  # noqa: E402
    HF_XET_MITIGATION,
    _sha256,
    _validate_staged_sft_config,
    _write_wrapper_state,
    apply_hf_xet_mitigation,
    build_training_command,
    commit_success_volumes,
)


def test_mitigation_applied_when_unset():
    env = {}
    apply_hf_xet_mitigation(env)
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_mitigation_applied_when_empty_string():
    # The @app.function secrets dict forwards these as "" when they are absent
    # from the local env; an empty value must still fall through to the default.
    env = {"HF_HUB_DISABLE_XET": "", "HF_HUB_ENABLE_HF_TRANSFER": ""}
    apply_hf_xet_mitigation(env)
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_explicit_local_value_overrides_default():
    env = {"HF_HUB_DISABLE_XET": "0"}
    apply_hf_xet_mitigation(env)
    assert env["HF_HUB_DISABLE_XET"] == "0"  # explicit local override wins
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"  # unset -> default


def test_mitigation_keys_are_the_two_expected():
    assert set(HF_XET_MITIGATION) == {"HF_HUB_DISABLE_XET", "HF_HUB_ENABLE_HF_TRANSFER"}


def test_staged_config_and_dataset_are_hash_verified_inside_input_mount(
    tmp_path, monkeypatch
):
    import yaml
    import Trainers.cloud.train_modal as wrapper

    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"conversations": []}\n', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"dataset": {"local_file": str(dataset)}}), encoding="utf-8"
    )
    monkeypatch.setattr(wrapper, "INPUT_VOLUME_NAME", "private-inputs")
    monkeypatch.setattr(wrapper, "INPUT_MOUNT_PATH", str(tmp_path))

    resolved = _validate_staged_sft_config(
        str(config), _sha256(config), _sha256(dataset)
    )
    assert resolved[0] == config.resolve()
    assert resolved[1] == dataset.resolve()


def test_staged_dataset_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    import yaml
    import Trainers.cloud.train_modal as wrapper

    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"dataset": {"local_file": str(dataset)}}), encoding="utf-8"
    )
    monkeypatch.setattr(wrapper, "INPUT_VOLUME_NAME", "private-inputs")
    monkeypatch.setattr(wrapper, "INPUT_MOUNT_PATH", str(tmp_path))
    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        _validate_staged_sft_config(str(config), _sha256(config), "0" * 64)


def test_config_driven_command_is_an_argv_not_a_shell_string(monkeypatch):
    import Trainers.cloud.train_modal as wrapper

    monkeypatch.setattr(wrapper, "OUTPUT_MOUNT_PATH", "/vol/artifacts")
    command = build_training_command(
        train_script="train_sft.py",
        run_timestamp="20260722_120000",
        config_path="/vol/inputs/smoke/config.yaml",
    )
    assert command == [
        "python", "train_sft.py",
        "--run-timestamp", "20260722_120000",
        "--output-root", "/vol/artifacts/outputs",
        "--cloud-provider", "modal",
        "--artifact-backend", "modal_volume",
        "--config", "/vol/inputs/smoke/config.yaml",
    ]


def test_success_commits_output_before_cache(monkeypatch):
    import Trainers.cloud.train_modal as wrapper
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(
        wrapper, "output_volume", SimpleNamespace(commit=lambda: calls.append("output"))
    )
    monkeypatch.setattr(
        wrapper, "model_cache", SimpleNamespace(commit=lambda: calls.append("cache"))
    )
    commit_success_volumes()
    assert calls == ["output", "cache"]


def test_wrapper_failure_updates_manifest_binds_provenance_and_redacts(tmp_path):
    provenance = {
        "source": {"branch": "feature/smoke", "commit": "a" * 40},
        "inputs": {"verified": True},
        "runtime": {"image": "example/train@sha256:" + "b" * 64},
    }
    (tmp_path / "manifest.json").write_text('{"status": "running"}\n')
    _write_wrapper_state(
        tmp_path,
        provenance,
        "failed",
        "token=supersecret https://user:pass@example.invalid/repo.git",
    )
    manifest_text = (tmp_path / "manifest.json").read_text()
    assert "supersecret" not in manifest_text
    assert "user:pass" not in manifest_text
    manifest = __import__("json").loads(manifest_text)
    assert manifest["status"].startswith("failed:")
    assert manifest["cloud_job_provenance"]["source"]["commit"] == "a" * 40
