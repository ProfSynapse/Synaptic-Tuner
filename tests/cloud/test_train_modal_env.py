"""Tests for the remote-container env defaults in Trainers/cloud/train_modal.py.

Focus: apply_hf_xet_mitigation, which sets the hf_xet-hang workaround defaults
on the remote training container while still honoring an explicit override
forwarded from the local launch env.
"""

import importlib.util
import inspect
import os
import sys
from pathlib import Path

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
    _function_secret_env,
    _sha256,
    _validate_staged_sft_config,
    _write_wrapper_state,
    apply_hf_xet_mitigation,
    build_training_command,
    commit_success_volumes,
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
