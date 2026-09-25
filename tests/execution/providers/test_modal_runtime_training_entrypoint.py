"""Provider-free wiring tests for the installed Modal training function."""

from __future__ import annotations

import base64
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.runtime import runtime_release_modal_training as entrypoint
from tuner.execution.providers.modal.runtime_release_deployment import ModalRuntimeReleaseDeployer
from tests.execution.providers.test_modal_runtime_release_deployment import _plan


def test_outer_entrypoint_failure_is_fixed_and_does_not_expose_exception(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace())
    def fail_setup(*args, **kwargs):
        raise RuntimeError("private token and path")
    monkeypatch.setattr(entrypoint, "_run_with_modal", fail_setup)
    assert entrypoint.run_modal_packaged_training(b"signed-dispatch") == {
        "schema_version": "synaptic-modal-packaged-worker-result/v2",
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
        "failure_stage": "ENTRYPOINT_SETUP",
    }


def test_training_callable_has_exact_installed_global_identity() -> None:
    spec = replace(
        _plan().functions[0],
        module="tuner.runtime.runtime_release_modal_training",
        qualname="run_modal_packaged_training",
    )
    assert ModalRuntimeReleaseDeployer._validate_entrypoint(
        spec, entrypoint.run_modal_packaged_training,
    ) is entrypoint.run_modal_packaged_training


def test_remote_entrypoint_binds_image_volumes_and_model_cache_before_worker(monkeypatch, tmp_path: Path) -> None:
    from tuner.execution.providers.modal import model_snapshot, packaged_dispatch, packaged_worker
    from tuner.execution.providers.modal.runtime_release_qualification import QUALIFICATION_HMAC_ENV_KEY

    control = tmp_path / "control"
    artifacts = tmp_path / "artifacts"
    model_cache = tmp_path / "model-cache"
    private = tmp_path / "private"
    for path in (control, artifacts, model_cache, private):
        path.mkdir()
    monkeypatch.setattr(entrypoint, "_CONTROL_ROOT", control)
    monkeypatch.setattr(entrypoint, "_ARTIFACT_ROOT", artifacts)
    monkeypatch.setattr(entrypoint, "_MODEL_CACHE_ROOT", model_cache)
    monkeypatch.setattr(entrypoint, "_PRIVATE_SCRATCH_ROOT", private)
    monkeypatch.setenv(QUALIFICATION_HMAC_ENV_KEY, base64.b64encode(b"k" * 32).decode())
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "production")
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-exact")
    monkeypatch.setenv("HF_TOKEN", "test-token-not-logged")
    facts = SimpleNamespace(environment_ref="production", image_id="im-exact",
                            control_volume_id="vo-control", artifact_volume_id="vo-artifacts",
                            model_cache_volume_id="vo-cache")
    monkeypatch.setattr(packaged_dispatch, "parse_modal_packaged_dispatch",
                        lambda payload, verifier: SimpleNamespace(provider_facts=facts))
    events: list[str] = []
    client = object()

    class Volume:
        is_hydrated = True

        def __init__(self, identity: str):
            self.object_id = identity

        @classmethod
        def from_id(cls, identity, *, client: object):
            assert client is client_value
            events.append("volume:" + identity)
            return cls(identity)

        def hydrate(self, selected):
            assert selected is client
            return self

        def commit(self):
            events.append("commit:" + self.object_id)

    client_value = client

    class SDK:
        __version__ = "1.5.4"

        class Client:
            @staticmethod
            def from_env():
                events.append("client")
                return client

        @staticmethod
        def current_function_call_id():
            return "fc-test"

    SDK.Volume = Volume

    def prepare_model_snapshot(**kwargs):
        events.append("model")
        assert kwargs["model_ref"] == "Qwen/Qwen3.5-4B"
        assert kwargs["revision"] == "1" * 40
        assert kwargs["token"] == "test-token-not-logged"
        assert kwargs["persistent_root"] == model_cache
        assert kwargs["scratch_root"].is_relative_to(private)
        return kwargs["destination_root"]

    monkeypatch.setattr(model_snapshot, "prepare_model_snapshot", prepare_model_snapshot)

    class Worker:
        def __init__(self, **kwargs):
            assert kwargs["expected_facts"] is facts
            assert kwargs["roots"].control == control
            assert kwargs["roots"].cache == model_cache
            self.executor = kwargs["trainer_executor"]

        def __call__(self, payload, call_id, *, commit_artifacts, commit_control):
            assert payload == b"signed-dispatch" and call_id == "fc-test"
            self.executor._model_preparer({"ref": "Qwen/Qwen3.5-4B", "revision": "1" * 40}, model_cache / "run")
            commit_artifacts()
            commit_control()
            return {"schema_version": "synaptic-modal-packaged-worker-result/v1",
                    "effect_id": "effect:test", "status_code": "completed", "completion_sha256": "a" * 64}

    monkeypatch.setattr(packaged_worker, "ModalPackagedWorker", Worker)
    result = entrypoint._run_with_modal(b"signed-dispatch", sdk=SDK)
    assert result["status_code"] == "completed"
    assert events == ["client", "volume:vo-control", "volume:vo-artifacts", "volume:vo-cache", "model",
                      "commit:vo-cache", "commit:vo-artifacts", "commit:vo-control"]


def test_image_substitution_fails_before_client_or_storage(monkeypatch) -> None:
    from tuner.execution.providers.modal import packaged_dispatch
    from tuner.execution.providers.modal.runtime_release_qualification import QUALIFICATION_HMAC_ENV_KEY

    monkeypatch.setenv(QUALIFICATION_HMAC_ENV_KEY, base64.b64encode(b"k" * 32).decode())
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "production")
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-other")
    facts = SimpleNamespace(environment_ref="production", image_id="im-exact")
    monkeypatch.setattr(packaged_dispatch, "parse_modal_packaged_dispatch",
                        lambda payload, verifier: SimpleNamespace(provider_facts=facts))

    class SDK:
        __version__ = "1.5.4"

        class Client:
            @staticmethod
            def from_env():
                raise AssertionError("client must not be created")

    with pytest.raises(ValueError, match="provider identity differs"):
        entrypoint._run_with_modal(b"signed-dispatch", sdk=SDK)
