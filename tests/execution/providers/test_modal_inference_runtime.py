"""Provider-free tests for the packaged Modal inference runtime verifier."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tuner.execution.providers.modal import inference_runtime as runtime
from tuner.execution.providers.modal.inference_preparation import (
    ModalInferencePreparationConfig,
)

_DEPENDENCY = "tuner/execution/providers/modal/inference-dependencies.lock"
_CLOSURE = "tuner/execution/providers/modal/inference-worker-closure.json"
_DISTRIBUTIONS = {
    "modal": "1.5.4",
    "safetensors": "0.6.2",
    "tokenizers": "0.22.0",
    "torch": "2.8.0",
    "transformers": "4.56.0",
    "vllm": "0.10.2",
}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _entry(root: Path, relative: str) -> dict[str, object]:
    payload = (root / relative).read_bytes()
    return {
        "path": relative,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _configuration(
    runtime_values: dict[str, object],
) -> ModalInferencePreparationConfig:
    document = {
        "schema_version": "synaptic-modal-inference-preparation-config/v1",
        "provider": {"provider_id": "modal", "profile_ref": "chat-a10"},
        "client": {
            "account_ref": "account-a",
            "workspace_ref": "workspace-a",
            "environment_ref": "environment-a",
            "client_ref": "client-a",
            "sdk_version": "1.5.4",
        },
        "application": {
            "app_name": "synaptic-chat-v1",
            "app_ref": "chat-app",
            "sandbox_entrypoint": "chat-entry",
            "worker_ref": "chat-worker",
        },
        "image": {
            "registry_reference": "registry.example/synaptic/chat@sha256:" + "1" * 64,
            "image_digest": "1" * 64,
        },
        "runtime": runtime_values,
        "volumes": {
            "source_artifact_volume_ref": "source-artifacts",
            "source_artifact_volume_id": "source-artifacts-id",
            "chat_control_volume_ref": "chat-control",
            "chat_control_volume_id": "chat-control-id",
            "model_cache_volume_ref": "model-cache",
            "model_cache_volume_id": "model-cache-id",
            "key_ref": "chat-evidence-key",
        },
        "resources": {
            "accelerator": "A10G",
            "accelerator_count": 1,
            "cpu_millicores": 4000,
            "memory_mb": 16384,
            "service_port": 8000,
            "provider_timeout_seconds": 900,
            "provider_idle_timeout_seconds": 300,
            "max_retries": 0,
        },
        "serving": {
            "served_model_name": "fixture-chat",
            "gpu_memory_utilization_milli": 730,
            "enforce_eager": False,
            "tokenizer_mode": "mistral",
            "max_lora_rank": 32,
            "readiness_request_timeout_milliseconds": 750,
            "max_tokens": 73,
            "temperature_milli": 250,
            "top_p_milli": 875,
        },
        "policy": {
            "startup_timeout_seconds": 600,
            "request_timeout_seconds": 60,
            "idle_timeout_seconds": 300,
            "absolute_lifetime_seconds": 900,
            "max_turns": 32,
            "max_history_bytes": 65536,
            "max_request_bytes": 8192,
            "max_response_bytes": 65536,
        },
        "secrets": [{"name": "model-download", "required_keys": ["HF_TOKEN"]}],
        "evidence": {
            "issuer_ref": "host-config",
            "evidence_ref": "config-1",
            "audience_ref": "chat-session",
            "challenge_nonce": "config-nonce",
            "key_ref": "config-key",
            "verified_at": "2026-09-09T12:01:00Z",
            "expires_at": "2026-09-09T12:05:00Z",
        },
    }
    return ModalInferencePreparationConfig.build(document)


class _Distribution:
    def __init__(self, name: str, version: str):
        self.metadata = {"Name": name}
        self.version = version


def _case(tmp_path: Path, monkeypatch):
    root = tmp_path / "package"
    sources = sorted(runtime._REQUIRED_SOURCES)
    for relative in [*sources, _DEPENDENCY]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode())

    source_members = [_entry(root, relative) for relative in sources]
    unsigned_closure = {
        "schema_version": "synaptic-modal-inference-worker-closure/v1",
        "entrypoint": "tuner/execution/providers/modal/inference_bootstrap.py",
        "member_count": len(source_members),
        "payload_bytes": sum(item["size_bytes"] for item in source_members),
        "members": source_members,
    }
    closure = dict(unsigned_closure)
    closure["closure_digest"] = hashlib.sha256(_canonical(unsigned_closure)).hexdigest()
    closure_path = root / _CLOSURE
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    closure_path.write_bytes(_canonical(closure))

    inventory = sorted(
        [_entry(root, relative) for relative in [*sources, _DEPENDENCY, _CLOSURE]],
        key=lambda item: item["path"],
    )
    executable = Path(sys.executable).resolve(strict=True)
    executable_digest = runtime._hash_executable(executable)
    manifest = {
        "schema_version": "synaptic-modal-inference-runtime-lock/v1",
        "registry_reference": "registry.example/synaptic/chat@sha256:" + "1" * 64,
        "sdk_version": "1.5.4",
        "python": {
            "implementation": "cpython",
            "version": runtime._actual_version(),
            "executable": str(executable),
            "executable_sha256": executable_digest,
        },
        "dependency_lock_path": _DEPENDENCY,
        "worker_closure_manifest_path": _CLOSURE,
        "distributions": _DISTRIBUTIONS,
        "source_inventory": inventory,
    }
    manifest_payload = _canonical(manifest)
    manifest_path = root / "tuner/execution/providers/modal/inference-runtime.lock.json"
    manifest_path.write_bytes(manifest_payload)
    by_path = {item["path"]: item for item in inventory}
    values = {
        "dependency_lock_digest": by_path[_DEPENDENCY]["sha256"],
        "runtime_lock_digest": hashlib.sha256(manifest_payload).hexdigest(),
        "source_lock_digest": hashlib.sha256(_canonical(inventory)).hexdigest(),
        "worker_closure_digest": closure["closure_digest"],
        "python_version": runtime._actual_version(),
        "python_executable": str(executable),
        "python_executable_digest": executable_digest,
    }
    monkeypatch.setattr(runtime, "_runtime_root", lambda: root)
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "distributions",
        lambda: [
            _Distribution(name, version) for name, version in _DISTRIBUTIONS.items()
        ],
    )
    return root, manifest, values


def test_packaged_runtime_verifies_complete_exact_environment(tmp_path, monkeypatch):
    _, _, values = _case(tmp_path, monkeypatch)
    runtime.verify_modal_inference_runtime(_configuration(values))


def test_missing_packaged_manifest_fails_closed(tmp_path, monkeypatch):
    root, _, values = _case(tmp_path, monkeypatch)
    (root / "tuner/execution/providers/modal/inference-runtime.lock.json").unlink()
    with pytest.raises(runtime.ModalInferenceRuntimeError) as raised:
        runtime.verify_modal_inference_runtime(_configuration(values))
    assert str(raised.value) == "modal_inference_runtime_invalid"


def test_non_config_is_rejected_before_packaged_reads(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_manifest",
        lambda: (_ for _ in ()).throw(AssertionError("manifest read")),
    )
    with pytest.raises(runtime.ModalInferenceRuntimeError):
        runtime.verify_modal_inference_runtime(object())


@pytest.mark.parametrize(
    "field",
    [
        "dependency_lock_digest",
        "runtime_lock_digest",
        "source_lock_digest",
        "worker_closure_digest",
        "python_executable_digest",
    ],
)
def test_configuration_commitment_substitution_fails_closed(
    tmp_path, monkeypatch, field
):
    _, _, values = _case(tmp_path, monkeypatch)
    values[field] = "f" * 64
    with pytest.raises(runtime.ModalInferenceRuntimeError):
        runtime.verify_modal_inference_runtime(_configuration(values))


@pytest.mark.parametrize(
    "relative", sorted(runtime._REQUIRED_SOURCES)[:2] + [_DEPENDENCY]
)
def test_packaged_member_change_or_link_fails_closed(tmp_path, monkeypatch, relative):
    root, _, values = _case(tmp_path, monkeypatch)
    path = root / relative
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(runtime.ModalInferenceRuntimeError):
        runtime.verify_modal_inference_runtime(_configuration(values))


def test_worker_closure_must_be_complete_and_distinct_from_dependency(
    tmp_path, monkeypatch
):
    root, _, values = _case(tmp_path, monkeypatch)
    path = root / _CLOSURE
    closure = json.loads(path.read_text())
    closure["members"].append(_entry(root, _DEPENDENCY))
    closure["member_count"] += 1
    closure["payload_bytes"] += closure["members"][-1]["size_bytes"]
    unsigned = dict(closure)
    unsigned.pop("closure_digest")
    closure["closure_digest"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    path.write_bytes(_canonical(closure))
    with pytest.raises(runtime.ModalInferenceRuntimeError):
        runtime.verify_modal_inference_runtime(_configuration(values))


@pytest.mark.parametrize("change", ["missing", "extra", "version", "duplicate"])
def test_installed_distribution_set_is_exact(tmp_path, monkeypatch, change):
    _, _, values = _case(tmp_path, monkeypatch)
    distributions = dict(_DISTRIBUTIONS)
    if change == "missing":
        distributions.pop("vllm")
    elif change == "extra":
        distributions["unlisted"] = "1"
    elif change == "version":
        distributions["torch"] = "0"
    items = [_Distribution(name, version) for name, version in distributions.items()]
    if change == "duplicate":
        items.append(_Distribution("SafeTensors", "0.6.2"))
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda: items)
    with pytest.raises(runtime.ModalInferenceRuntimeError):
        runtime.verify_modal_inference_runtime(_configuration(values))


@pytest.mark.parametrize("change", ("missing", "different"))
def test_manifest_requires_the_same_installed_modal_sdk(tmp_path, monkeypatch, change):
    root, manifest, values = _case(tmp_path, monkeypatch)
    manifest = deepcopy(manifest)
    if change == "missing":
        manifest["distributions"].pop("modal")
    else:
        manifest["distributions"]["modal"] = "1.5.3"
    payload = _canonical(manifest)
    (root / "tuner/execution/providers/modal/inference-runtime.lock.json").write_bytes(
        payload
    )
    values["runtime_lock_digest"] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "distributions",
        lambda: [
            _Distribution(name, version)
            for name, version in manifest["distributions"].items()
        ],
    )
    with pytest.raises(runtime.ModalInferenceRuntimeError):
        runtime.verify_modal_inference_runtime(_configuration(values))


def test_fixed_commitment_paths_reject_manifest_selected_files(tmp_path, monkeypatch):
    root, manifest, values = _case(tmp_path, monkeypatch)
    manifest = deepcopy(manifest)
    manifest["dependency_lock_path"] = "alternate.lock"
    (root / "alternate.lock").write_bytes(b"alternate\n")
    payload = _canonical(manifest)
    path = root / "tuner/execution/providers/modal/inference-runtime.lock.json"
    path.write_bytes(payload)
    values["runtime_lock_digest"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(runtime.ModalInferenceRuntimeError):
        runtime.verify_modal_inference_runtime(_configuration(values))


def test_manifest_link_and_noncanonical_bytes_fail_closed(tmp_path, monkeypatch):
    root, _, values = _case(tmp_path, monkeypatch)
    path = root / "tuner/execution/providers/modal/inference-runtime.lock.json"
    target = path.with_suffix(".target")
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(runtime.ModalInferenceRuntimeError):
        runtime.verify_modal_inference_runtime(_configuration(values))


def test_metadata_control_flow_is_preserved(tmp_path, monkeypatch):
    _, _, values = _case(tmp_path, monkeypatch)

    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime.importlib.metadata, "distributions", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runtime.verify_modal_inference_runtime(_configuration(values))


def test_interpreter_read_interrupt_is_not_masked_by_close_failure(monkeypatch):
    monkeypatch.setattr(runtime.os, "open", lambda *args: 71)
    monkeypatch.setattr(
        runtime.os,
        "fstat",
        lambda descriptor: SimpleNamespace(
            st_mode=0o100600,
            st_size=1,
            st_dev=1,
            st_ino=2,
            st_mtime_ns=3,
        ),
    )
    monkeypatch.setattr(
        runtime.os,
        "read",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        runtime.os,
        "close",
        lambda descriptor: (_ for _ in ()).throw(OSError("close detail")),
    )
    with pytest.raises(KeyboardInterrupt):
        runtime._hash_executable(Path("/python"))


def test_verification_does_not_import_ml_distributions(tmp_path, monkeypatch):
    _, _, values = _case(tmp_path, monkeypatch)
    before = {name for name in sys.modules if name.split(".", 1)[0] in _DISTRIBUTIONS}
    runtime.verify_modal_inference_runtime(_configuration(values))
    after = {name for name in sys.modules if name.split(".", 1)[0] in _DISTRIBUTIONS}
    assert after == before
