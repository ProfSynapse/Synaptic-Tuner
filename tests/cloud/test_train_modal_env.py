"""Tests for the remote-container env defaults in Trainers/cloud/train_modal.py.

Focus: apply_hf_xet_mitigation, which sets the hf_xet-hang workaround defaults
on the remote training container while still honoring an explicit override
forwarded from the local launch env.
"""

import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

# train_modal defines a Modal @app.function, so importing it needs the modal
# package; skip cleanly where modal is unavailable rather than erroring.
modal = pytest.importorskip("modal")

from Trainers.cloud.train_modal import (  # noqa: E402
    HF_XET_MITIGATION,
    MODULE_IMPORT_MODAL_ENV_KEYS,
    _config_overrides_mapping,
    _checkout_workspace,
    _commit_failed_state,
    _commit_stable_completion,
    _function_secret_env,
    _is_completed_retry,
    _latest_valid_checkpoint,
    _modal_provenance,
    _promote_ready_completion,
    _provenance_identity,
    _reload_output_volume,
    _run_with_periodic_output_commits,
    _sha256,
    _validate_stable_namespace,
    _validate_staged_sft_config,
    _write_done_marker,
    _write_completion_ready_marker,
    _write_wrapper_state,
    apply_hf_xet_mitigation,
    build_training_command,
    commit_success_volumes,
    run_stable_training,
    run_training,
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


def test_modal_cli_can_build_help_for_remote_function_annotations():
    """Exercise Modal's real CLI annotation parser without creating an app run."""
    from modal.cli.run import (
        _add_click_options,
        _get_cli_runnable_signature,
        safe_get_type_hints,
    )

    raw_function = run_training.get_raw_f()
    signature = _get_cli_runnable_signature(
        inspect.signature(raw_function), safe_get_type_hints(raw_function)
    )

    def command(**_kwargs):
        pass

    command = click.command()(_add_click_options(command, signature.parameters))
    result = CliRunner().invoke(command, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--config-overrides TEXT" in result.output
    assert "--run-id TEXT" not in result.output

    stable_raw = run_stable_training.get_raw_f()
    stable_signature = _get_cli_runnable_signature(
        inspect.signature(stable_raw), safe_get_type_hints(stable_raw)
    )
    stable_command = click.command()(
        _add_click_options(command.callback, stable_signature.parameters)
    )
    stable_result = CliRunner().invoke(stable_command, ["--help"])
    assert stable_result.exit_code == 0, stable_result.output
    assert "--run-id TEXT" in stable_result.output


def _import_modal_wrapper(module_name):
    path = Path(__file__).resolve().parents[2] / "Trainers/cloud/train_modal.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _explicit_dependency_signature(module):
    from modal._utils.async_utils import synchronizer

    function = synchronizer._translate_in(module.run_training)
    dependencies = function.deps(only_explicit_mounts=True)
    secret_environment = dependencies[0]._load_env_dict()
    return {
        "types": [type(dependency).__name__ for dependency in dependencies],
        "secret": repr(dependencies[0]),
        "module_environment": {
            key: secret_environment.get(key, "") for key in MODULE_IMPORT_MODAL_ENV_KEYS
        },
        "volumes": [repr(dependency) for dependency in dependencies[2:]],
        "mounts": list(module.function_volumes),
    }


def _roundtrip_dependency_signatures(monkeypatch, launch_env):
    with monkeypatch.context() as local_environment:
        for key in MODULE_IMPORT_MODAL_ENV_KEYS:
            local_environment.delenv(key, raising=False)
        for key, value in launch_env.items():
            local_environment.setenv(key, value)
        local_module = _import_modal_wrapper("_test_train_modal_local_submit")
        submitted_signature = _explicit_dependency_signature(local_module)
        forwarded_environment = dict(local_module.FUNCTION_SECRET_ENV)

    with monkeypatch.context() as remote_environment:
        for key in MODULE_IMPORT_MODAL_ENV_KEYS:
            remote_environment.delenv(key, raising=False)
        assert not any(os.environ.get(key) for key in MODULE_IMPORT_MODAL_ENV_KEYS)
        # Modal injects the submitted Function Secret before importing user code.
        for key, value in forwarded_environment.items():
            remote_environment.setenv(key, value)
        remote_module = _import_modal_wrapper("_test_train_modal_remote_reimport")
        remote_signature = _explicit_dependency_signature(remote_module)

    return submitted_signature, remote_signature


def test_remote_reimport_reconstructs_exact_dependency_graph_from_function_secret(
    monkeypatch,
):
    """Reproduce Modal hydration with no inherited ambient launch config."""
    launch_env = {
        "MODAL_TRAINING_IMAGE": "registry.example/train@sha256:" + "b" * 64,
        "MODAL_TRAINING_PIP_PACKAGES_JSON": '["peft==0.18.1"]',
        "MODAL_GPU": "A10G",
        "MODAL_TIMEOUT_SECONDS": "7200",
        "MODAL_CACHE_VOLUME_NAME": "cache-selected-by-config",
        "MODAL_OUTPUT_VOLUME_NAME": "output-selected-by-config",
        "MODAL_OUTPUT_MOUNT_PATH": "/vol/custom-artifacts",
        "MODAL_INPUT_VOLUME_NAME": "input-selected-by-config",
        "MODAL_INPUT_MOUNT_PATH": "/vol/custom-inputs",
    }
    submitted_signature, remote_signature = _roundtrip_dependency_signatures(
        monkeypatch, launch_env
    )

    assert submitted_signature == remote_signature
    assert submitted_signature["module_environment"] == launch_env
    assert submitted_signature["types"] == [
        "_Secret",
        "_Image",
        "_Volume",
        "_Volume",
        "_Volume",
    ]
    assert submitted_signature["volumes"] == [
        "modal.Volume.from_name('cache-selected-by-config')",
        "modal.Volume.from_name('output-selected-by-config')",
        "modal.Volume.from_name('input-selected-by-config')",
    ]
    assert submitted_signature["mounts"] == [
        "/cache/huggingface",
        "/vol/custom-artifacts",
        "/vol/custom-inputs",
    ]


def test_remote_reimport_preserves_legacy_graph_without_private_input(monkeypatch):
    submitted_signature, remote_signature = _roundtrip_dependency_signatures(
        monkeypatch,
        {
            "MODAL_CACHE_VOLUME_NAME": "legacy-cache",
            "MODAL_OUTPUT_VOLUME_NAME": "legacy-output",
            "MODAL_OUTPUT_MOUNT_PATH": "/vol/legacy-output",
        },
    )
    assert submitted_signature == remote_signature
    assert submitted_signature["types"] == ["_Secret", "_Image", "_Volume", "_Volume"]
    assert submitted_signature["volumes"] == [
        "modal.Volume.from_name('legacy-cache')",
        "modal.Volume.from_name('legacy-output')",
    ]
    assert submitted_signature["mounts"] == [
        "/cache/huggingface",
        "/vol/legacy-output",
    ]


def test_existing_function_secret_carries_every_module_import_modal_key():
    payload = _function_secret_env(
        {
            key: f"value-{index}"
            for index, key in enumerate(MODULE_IMPORT_MODAL_ENV_KEYS)
        }
    )
    assert {key: payload[key] for key in MODULE_IMPORT_MODAL_ENV_KEYS} == {
        key: f"value-{index}" for index, key in enumerate(MODULE_IMPORT_MODAL_ENV_KEYS)
    }


def test_function_secret_omits_empty_import_values_so_remote_defaults_survive():
    payload = _function_secret_env(
        {
            "MODAL_INPUT_VOLUME_NAME": "",
            "MODAL_OUTPUT_VOLUME_NAME": "configured-output",
        }
    )
    assert "MODAL_INPUT_VOLUME_NAME" not in payload
    assert payload["MODAL_OUTPUT_VOLUME_NAME"] == "configured-output"


@pytest.mark.parametrize(
    "payload, expected",
    [
        (None, {}),
        ("", {}),
        ({"max_steps": 1}, {"max_steps": 1}),
        ('{"max_steps": 1}', {"max_steps": 1}),
    ],
)
def test_config_overrides_accepts_cli_json_and_legacy_mappings(payload, expected):
    assert _config_overrides_mapping(payload) == expected


@pytest.mark.parametrize("payload", ["[]", "not-json", False, 1])
def test_config_overrides_rejects_non_object_payloads(payload):
    with pytest.raises(ValueError, match="config_overrides"):
        _config_overrides_mapping(payload)


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


def test_resume_checkpoint_is_forwarded_as_shell_free_trainer_argv(monkeypatch):
    import Trainers.cloud.train_modal as wrapper

    monkeypatch.setattr(wrapper, "OUTPUT_MOUNT_PATH", "/vol/artifacts")
    command = build_training_command(
        train_script="train_sft.py",
        run_timestamp="stable-run",
        config_path="/vol/inputs/config.yaml",
        resume_from_checkpoint=(
            "/vol/artifacts/outputs/runs/modal/sft/stable-run-aaaaaaaa/"
            "checkpoints/checkpoint-20"
        ),
    )
    assert command[-2:] == [
        "--resume-from-checkpoint",
        "/vol/artifacts/outputs/runs/modal/sft/stable-run-aaaaaaaa/"
        "checkpoints/checkpoint-20",
    ]


def test_warm_workspace_fetches_and_checks_out_exact_commit(tmp_path, monkeypatch):
    import Trainers.cloud.train_modal as wrapper

    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    calls = []

    def fake_git(args, *, cwd=None):
        calls.append((args, cwd))
        if args[:3] == ["remote", "get-url", "origin"]:
            return SimpleNamespace(stdout="https://example.invalid/repo.git\n")
        if args[0] == "status":
            return SimpleNamespace(stdout="")
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(wrapper, "_run_git", fake_git)
    _checkout_workspace(
        repo_url="https://example.invalid/repo.git",
        repo_branch="main",
        repo_commit="a" * 40,
        workspace=workspace,
    )

    commands = [call[0] for call in calls]
    assert ["status", "--porcelain", "--untracked-files=no"] in commands
    assert ["fetch", "--depth", "1", "origin", "a" * 40] in commands
    assert ["checkout", "--detach", "a" * 40] in commands
    assert not any(command[0] == "clone" for command in commands)


def test_fresh_workspace_clones_before_exact_checkout(tmp_path, monkeypatch):
    import Trainers.cloud.train_modal as wrapper

    calls = []

    def fake_git(args, *, cwd=None):
        calls.append((args, cwd))
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(wrapper, "_run_git", fake_git)
    workspace = tmp_path / "workspace"
    _checkout_workspace(
        repo_url="https://example.invalid/repo.git",
        repo_branch="main",
        repo_commit="a" * 40,
        workspace=workspace,
    )

    commands = [call[0] for call in calls]
    assert commands[0] == [
        "clone", "--branch", "main", "--depth", "1", "--no-checkout",
        "https://example.invalid/repo.git", str(workspace),
    ]
    assert commands[1] == ["fetch", "--depth", "1", "origin", "a" * 40]
    assert commands[2] == ["checkout", "--detach", "a" * 40]


def _checkpoint(run_dir: Path, step: int, *, complete: bool = True) -> Path:
    import torch
    from safetensors.torch import save_file

    checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        __import__("json").dumps({"global_step": step}), encoding="utf-8"
    )
    if complete:
        torch.save({"state": {}, "param_groups": []}, checkpoint / "optimizer.pt")
        torch.save({"last_epoch": step}, checkpoint / "scheduler.pt")
        torch.save({"cpu": torch.get_rng_state()}, checkpoint / "rng_state.pth")
        save_file({"adapter.weight": torch.ones(1)}, checkpoint / "adapter_model.safetensors")
    return checkpoint


def test_latest_checkpoint_uses_highest_complete_checkpoint(tmp_path):
    run_dir = tmp_path / "stable-aaaaaaaa"
    expected = _checkpoint(run_dir, 10)
    _checkpoint(run_dir, 20, complete=False)

    assert _latest_valid_checkpoint(run_dir) == expected


def test_latest_checkpoint_falls_back_when_highest_binary_state_is_torn(tmp_path):
    run_dir = tmp_path / "stable-aaaaaaaa"
    expected = _checkpoint(run_dir, 10)
    corrupt = _checkpoint(run_dir, 20)
    (corrupt / "rng_state.pth").write_bytes(b"not a torch checkpoint")

    assert _latest_valid_checkpoint(run_dir) == expected


def test_stable_namespace_fails_closed_on_commit_drift(tmp_path):
    namespace = tmp_path / "sft"
    namespace.mkdir()
    (namespace / "stable-bbbbbbbb").mkdir()
    provenance = {"source": {"commit": "a" * 40}}

    with pytest.raises(RuntimeError, match="ambiguous or source-drifted"):
        _validate_stable_namespace(
            namespace=namespace,
            run_dir=namespace / "stable-aaaaaaaa",
            run_id="stable",
            expected_provenance=provenance,
        )


def test_periodic_commit_errors_are_contained_and_lifecycle_stops(monkeypatch):
    import Trainers.cloud.train_modal as wrapper

    commits = []

    def commit():
        commits.append(time.monotonic())
        if len(commits) == 1:
            raise RuntimeError("transient")

    class Process:
        def wait(self):
            time.sleep(0.035)
            return 0

    monkeypatch.setattr(wrapper, "output_volume", SimpleNamespace(commit=commit))
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *args, **kwargs: Process())
    assert _run_with_periodic_output_commits(
        ["python", "train_sft.py"], cwd="/tmp", env={}, interval_seconds=0.005
    ) == 0
    count_after_return = len(commits)
    time.sleep(0.015)
    assert count_after_return >= 2
    assert len(commits) == count_after_return


def test_wait_exception_terminates_then_kills_and_reaps_child(monkeypatch):
    import Trainers.cloud.train_modal as wrapper

    events = []

    class Process:
        def wait(self, timeout=None):
            if timeout is None:
                raise RuntimeError("wait channel failed")
            events.append(("wait", timeout))
            if not any(event == "kill" for event in events):
                raise wrapper.subprocess.TimeoutExpired("trainer", timeout)
            return -9

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(wrapper, "output_volume", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(RuntimeError, match="wait channel failed"):
        _run_with_periodic_output_commits(
            ["python", "train_sft.py"], cwd="/tmp", env={}, interval_seconds=0.01
        )

    assert events[0] == "terminate"
    assert "kill" in events
    assert events[-1] == ("wait", wrapper.PROCESS_STOP_TIMEOUT_SECONDS)


def test_commit_thread_join_is_bounded_and_fails_closed(monkeypatch):
    import Trainers.cloud.train_modal as wrapper

    joins = []

    class Process:
        def wait(self):
            return 0

    class StuckThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            joins.append(timeout)

        def is_alive(self):
            return True

    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(wrapper.threading, "Thread", StuckThread)
    with pytest.raises(RuntimeError, match="did not stop within its bound"):
        _run_with_periodic_output_commits(
            ["python", "train_sft.py"], cwd="/tmp", env={}, interval_seconds=1
        )
    assert joins == [wrapper.COMMIT_THREAD_JOIN_TIMEOUT_SECONDS]


def _write_completed_artifacts(
    run_dir: Path, provenance: dict, *, special_tokens: bool = True
) -> None:
    import torch
    import yaml
    from safetensors.torch import save_file

    config_record = (provenance.setdefault("inputs", {}).get("config") or {})
    if not config_record.get("mounted_path"):
        config_path = run_dir.parent / f"{run_dir.name}-config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        model = (
            {"tokenizer": {"additional_special_tokens": ["<GENERIC>"]}}
            if special_tokens
            else {}
        )
        config_path.write_text(
            yaml.safe_dump({"model": model, "dataset": {"local_file": "/inputs/data.jsonl"}}),
            encoding="utf-8",
        )
        provenance["inputs"]["config"] = {
            "mounted_path": str(config_path),
            "sha256": _sha256(config_path),
        }
    _write_wrapper_state(run_dir, provenance, "completed")
    (run_dir / "training_lineage.json").write_text(
        json.dumps(
            {
                "method": "sft",
                "training": {"completed": True},
                "cloud_job_provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    final_model = run_dir / "final_model"
    final_model.mkdir()
    (final_model / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}), encoding="utf-8"
    )
    save_file({"adapter.weight": torch.ones(1)}, final_model / "adapter_model.safetensors")
    (final_model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "GenericTokenizer"}), encoding="utf-8"
    )
    (final_model / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    if special_tokens:
        (final_model / "special_tokens_lineage.json").write_text(
            json.dumps(
                {
                    "resolved_config": {"tokens": ["<GENERIC>"]},
                    "config_sha256": "c" * 64,
                    "vocab_sha256_after": "d" * 64,
                }
            ),
            encoding="utf-8",
        )


def test_done_marker_identity_makes_completed_retry_a_no_op(tmp_path, monkeypatch):
    import yaml
    import Trainers.cloud.train_modal as wrapper

    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"conversations": []}\n', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"dataset": {"local_file": str(dataset)}}))
    commit = "a" * 40
    run_id = "stable-smoke"
    run_dir = tmp_path / "outputs" / "runs" / "modal" / "sft" / f"{run_id}-aaaaaaaa"
    monkeypatch.setattr(wrapper, "INPUT_VOLUME_NAME", "private-inputs")
    monkeypatch.setattr(wrapper, "INPUT_MOUNT_PATH", str(tmp_path))
    monkeypatch.setattr(wrapper, "OUTPUT_MOUNT_PATH", str(tmp_path))
    provenance = _modal_provenance(
        repo_branch="main",
        repo_commit=commit,
        config_path=str(config.resolve()),
        config_sha256=_sha256(config),
        dataset_path=str(dataset.resolve()),
        dataset_sha256=_sha256(dataset),
        inputs_verified=True,
        run_id=run_id,
    )
    _write_completed_artifacts(run_dir, provenance, special_tokens=False)
    _write_completion_ready_marker(run_dir, provenance)
    _write_done_marker(run_dir, provenance)
    assert _is_completed_retry(run_dir, provenance)

    reloads = []
    monkeypatch.setattr(
        wrapper,
        "output_volume",
        SimpleNamespace(reload=lambda: reloads.append("reload"), commit=lambda: None),
    )
    monkeypatch.setattr(
        wrapper,
        "_checkout_workspace",
        lambda **kwargs: pytest.fail("completed retry must not touch source checkout"),
    )
    result = run_stable_training.get_raw_f()(
        trainer_type="sft",
        repo_url="https://example.invalid/repo.git",
        repo_branch="main",
        repo_commit=commit,
        config_path=str(config),
        config_sha256=_sha256(config),
        dataset_sha256=_sha256(dataset),
        run_id=run_id,
    )
    assert result["status"] == "completed"
    assert result["no_op"] is True
    assert result["artifact_root"] == str(run_dir)
    assert reloads == ["reload"]


def test_done_without_complete_canonical_artifacts_is_rejected(tmp_path):
    provenance = {
        "run": {"id": "stable", "stable": True},
        "source": {"branch": "main", "commit": "a" * 40},
        "inputs": {},
        "runtime": {},
        "artifacts": {},
        "cache": {},
        "publish_final_model": False,
    }
    run_dir = tmp_path / "stable-aaaaaaaa"
    _write_wrapper_state(run_dir, provenance, "completed")
    _write_completion_ready_marker(run_dir, provenance)
    _write_done_marker(run_dir, provenance)

    with pytest.raises(RuntimeError, match="training lineage"):
        _is_completed_retry(run_dir, provenance)


def test_retry_reload_discards_uncommitted_done_before_it_can_short_circuit(
    tmp_path, monkeypatch
):
    import Trainers.cloud.train_modal as wrapper

    run_dir = tmp_path / "stable-aaaaaaaa"
    run_dir.mkdir()
    marker = run_dir / "DONE"
    marker.write_text("locally written but not committed")
    monkeypatch.setattr(
        wrapper,
        "output_volume",
        SimpleNamespace(reload=lambda: marker.unlink(), commit=lambda: None),
    )

    _reload_output_volume()
    assert not marker.exists()


def test_stable_completion_commits_artifacts_before_done_then_cache(tmp_path, monkeypatch):
    import Trainers.cloud.train_modal as wrapper

    provenance = {
        "run": {"id": "stable", "stable": True},
        "source": {"branch": "main", "commit": "a" * 40},
        "inputs": {},
        "runtime": {},
        "artifacts": {},
        "cache": {},
        "publish_final_model": False,
    }
    run_dir = tmp_path / "stable-aaaaaaaa"
    _write_completed_artifacts(run_dir, provenance)
    events = []

    def output_commit():
        events.append(("output", (run_dir / "DONE").exists()))

    monkeypatch.setattr(wrapper, "output_volume", SimpleNamespace(commit=output_commit))
    monkeypatch.setattr(
        wrapper, "model_cache", SimpleNamespace(commit=lambda: events.append(("cache", True)))
    )
    _commit_stable_completion(run_dir, provenance)

    assert events == [("output", False), ("output", True), ("cache", True)]


def test_ordinary_config_completion_does_not_require_special_token_lineage(
    tmp_path, monkeypatch
):
    import Trainers.cloud.train_modal as wrapper

    provenance = {
        "run": {"id": "ordinary", "stable": True},
        "source": {"branch": "main", "commit": "a" * 40},
        "inputs": {},
        "runtime": {},
        "artifacts": {},
        "cache": {},
        "publish_final_model": False,
    }
    run_dir = tmp_path / "ordinary-aaaaaaaa"
    _write_completed_artifacts(run_dir, provenance, special_tokens=False)
    monkeypatch.setattr(wrapper, "output_volume", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(wrapper, "model_cache", SimpleNamespace(commit=lambda: None))

    _commit_stable_completion(run_dir, provenance)
    assert not (run_dir / "final_model" / "special_tokens_lineage.json").exists()
    assert (run_dir / "DONE").is_file()


def test_special_token_config_completion_requires_special_token_lineage(tmp_path):
    provenance = {
        "run": {"id": "special", "stable": True},
        "source": {"branch": "main", "commit": "a" * 40},
        "inputs": {},
        "runtime": {},
        "artifacts": {},
        "cache": {},
        "publish_final_model": False,
    }
    run_dir = tmp_path / "special-aaaaaaaa"
    _write_completed_artifacts(run_dir, provenance, special_tokens=True)
    (run_dir / "final_model" / "special_tokens_lineage.json").unlink()

    with pytest.raises(RuntimeError, match="special-token lineage"):
        _commit_stable_completion(run_dir, provenance)


def test_training_lineage_must_bind_full_stable_provenance(tmp_path):
    provenance = {
        "run": {"id": "stable", "stable": True},
        "source": {"branch": "main", "commit": "a" * 40},
        "inputs": {},
        "runtime": {},
        "artifacts": {},
        "cache": {"volume_name": "cache-a", "mount_path": "/cache"},
        "publish_final_model": False,
    }
    run_dir = tmp_path / "stable-aaaaaaaa"
    _write_completed_artifacts(run_dir, provenance)
    lineage_path = run_dir / "training_lineage.json"
    lineage = json.loads(lineage_path.read_text())
    lineage["cloud_job_provenance"]["cache"]["volume_name"] = "cache-drift"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Training lineage provenance identity"):
        _commit_stable_completion(run_dir, provenance)


def test_failed_done_commit_removes_local_marker_for_same_container_retry(
    tmp_path, monkeypatch
):
    import Trainers.cloud.train_modal as wrapper

    provenance = {
        "run": {"id": "stable", "stable": True},
        "source": {"branch": "main", "commit": "a" * 40},
        "inputs": {},
        "runtime": {},
        "artifacts": {},
        "cache": {},
        "publish_final_model": False,
    }
    run_dir = tmp_path / "stable-aaaaaaaa"
    _write_completed_artifacts(run_dir, provenance)
    commits = []

    def commit():
        commits.append((run_dir / "DONE").exists())
        if len(commits) == 2:
            raise RuntimeError("commit failed")

    monkeypatch.setattr(wrapper, "output_volume", SimpleNamespace(commit=commit))
    monkeypatch.setattr(wrapper, "model_cache", SimpleNamespace(commit=lambda: None))
    with pytest.raises(RuntimeError, match="commit failed"):
        _commit_stable_completion(run_dir, provenance)
    assert commits == [False, True]
    assert not (run_dir / "DONE").exists()


def test_committed_ready_phase_promotes_to_done_without_training(tmp_path, monkeypatch):
    import Trainers.cloud.train_modal as wrapper

    provenance = {
        "run": {"id": "stable", "stable": True},
        "source": {"branch": "main", "commit": "a" * 40},
        "inputs": {},
        "runtime": {},
        "artifacts": {},
        "cache": {},
        "publish_final_model": False,
    }
    run_dir = tmp_path / "stable-aaaaaaaa"
    _write_completed_artifacts(run_dir, provenance)
    _write_completion_ready_marker(run_dir, provenance)
    events = []
    monkeypatch.setattr(
        wrapper, "output_volume", SimpleNamespace(commit=lambda: events.append("output"))
    )
    monkeypatch.setattr(
        wrapper, "model_cache", SimpleNamespace(commit=lambda: events.append("cache"))
    )

    assert _promote_ready_completion(run_dir, provenance) is True
    assert (run_dir / "DONE").is_file()
    assert events == ["output", "cache"]


def test_provenance_identity_binds_all_volume_names_mounts_and_input_paths():
    base = {
        "run": {"id": "stable", "stable": True},
        "source": {"branch": "main", "commit": "a" * 40},
        "inputs": {
            "config": {"mounted_path": "/inputs/config.yaml", "sha256": "b" * 64},
            "dataset": {"mounted_path": "/inputs/data.jsonl", "sha256": "c" * 64},
            "verified": True,
            "volume_name": "inputs-a",
            "mount_path": "/inputs",
        },
        "runtime": {"image": "image@sha256:" + "d" * 64},
        "artifacts": {"volume_name": "outputs-a", "mount_path": "/outputs"},
        "cache": {"volume_name": "cache-a", "mount_path": "/cache"},
        "publish_final_model": False,
    }
    mutations = [
        ("inputs", "volume_name", "inputs-b"),
        ("inputs", "mount_path", "/other-inputs"),
        ("artifacts", "volume_name", "outputs-b"),
        ("artifacts", "mount_path", "/other-outputs"),
        ("cache", "volume_name", "cache-b"),
        ("cache", "mount_path", "/other-cache"),
    ]
    for section, field, value in mutations:
        changed = json.loads(json.dumps(base))
        changed[section][field] = value
        assert _provenance_identity(changed) != _provenance_identity(base)

    for kind in ("config", "dataset"):
        changed = json.loads(json.dumps(base))
        changed["inputs"][kind]["mounted_path"] += ".other"
        assert _provenance_identity(changed) != _provenance_identity(base)


def test_only_stable_modal_function_declares_three_retries():
    source = (Path(__file__).resolve().parents[2] / "Trainers/cloud/train_modal.py").read_text()
    legacy_start = source.index("@app.function(**_modal_function_options())")
    legacy_end = source.index("def run_training(", legacy_start)
    stable_start = source.index("@app.function(", legacy_end)
    stable_end = source.index("def run_stable_training(", stable_start)
    assert "retries=" not in source[legacy_start:legacy_end]
    assert "modal.Retries(max_retries=3" in source[stable_start:stable_end]


def test_retry_enabled_entrypoint_rejects_missing_stable_contract():
    raw = run_stable_training.get_raw_f()
    with pytest.raises(ValueError, match="non-empty stable run_id"):
        raw(run_id="")


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


def test_failure_state_is_written_before_output_commit_and_never_commits_cache(
    tmp_path, monkeypatch
):
    import Trainers.cloud.train_modal as wrapper

    calls = []
    monkeypatch.setattr(
        wrapper,
        "_write_wrapper_state",
        lambda *args: calls.append(("state", args[2])),
    )
    monkeypatch.setattr(
        wrapper, "output_volume", SimpleNamespace(commit=lambda: calls.append(("output", None)))
    )
    monkeypatch.setattr(
        wrapper, "model_cache", SimpleNamespace(commit=lambda: calls.append(("cache", None)))
    )

    _commit_failed_state(tmp_path, {"source": {}}, "failed")
    assert calls == [("state", "failed"), ("output", None)]


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
