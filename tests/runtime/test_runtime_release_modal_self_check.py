"""The installed Modal self-check parent has one closed live signature."""
from __future__ import annotations

import base64
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tuner.execution.providers.modal.runtime_release_qualification import (
    QUALIFICATION_HMAC_ENV_KEY,
    QUALIFICATION_HMAC_KEY_REF,
    ModalRuntimeQualificationHmacAuthenticator,
    ModalRuntimeReleaseFixtureReceiptV1,
    ModalRuntimeReleaseQualificationDispatchV1,
    ModalRuntimeReleaseQualificationPolicyV1,
    build_modal_runtime_release_qualification_dispatch,
)
from tuner.runtime.releases import ProviderRuntimeBindingV1
from tuner.runtime.runtime_release_modal_self_check import (
    run_runtime_release_self_check,
)

from tests.execution.providers.test_modal_packaged_binding import _release_and_execution
from tests.execution.providers.test_modal_runtime_release_qualification import _real_facts
from tuner.runtime.packaged_training_worker import LOCAL_CPU_DATA


class _Client:
    value = object()
    calls = 0

    @classmethod
    def from_env(cls):
        cls.calls += 1
        return cls.value


class _VolumeHandle:
    def __init__(self, object_id, events):
        self.object_id = object_id
        self.is_hydrated = False
        self.events = events

    def hydrate(self, client):
        assert client is _Client.value
        self.is_hydrated = True

    def commit(self):
        self.events.append(self.object_id)


class _Volume:
    registry = {}
    calls = []

    @classmethod
    def from_name(cls, name, **kwargs):
        cls.calls.append((name, kwargs))
        return cls.registry[name]


def _case(monkeypatch):
    release, packaged, _, _, _ = _release_and_execution(LOCAL_CPU_DATA)
    facts = _real_facts(release, packaged.client_binding)
    binding = ProviderRuntimeBindingV1.build(
        provider_ref="modal", runtime_release=release,
        provider_facts_schema=facts.schema_version,
        provider_facts_digest=facts.facts_digest,
    )
    dispatch = ModalRuntimeReleaseQualificationDispatchV1(
        "qualify-live", release, binding, facts,
        ModalRuntimeReleaseFixtureReceiptV1.create(
            effect_id="qualify-live", artifact_volume_id="vo-artifact",
        ),
        ModalRuntimeReleaseQualificationPolicyV1(), QUALIFICATION_HMAC_KEY_REF,
    )
    key = b"q" * 32
    raw = build_modal_runtime_release_qualification_dispatch(
        dispatch, ModalRuntimeQualificationHmacAuthenticator(key),
    )
    assert base64.b64encode(key) not in raw
    modal = ModuleType("modal")
    modal.__version__ = "1.5.4"
    modal.Client = _Client
    modal.Volume = _Volume
    modal.current_function_call_id = lambda: "fc-live"
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setenv(QUALIFICATION_HMAC_ENV_KEY, base64.b64encode(key).decode("ascii"))
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    monkeypatch.setenv("MODAL_ENVIRONMENT", facts.client_binding.environment_ref)
    monkeypatch.setenv("MODAL_IMAGE_ID", facts.image_id)
    return raw, facts


def test_live_parent_has_exact_one_bytes_argument_and_wires_verified_resources(
    monkeypatch,
) -> None:
    raw, facts = _case(monkeypatch)
    events = []
    _Client.calls = 0
    _Volume.calls = []
    _Volume.registry = {
        "control-name": _VolumeHandle("vo-control", events),
        "artifact-name": _VolumeHandle("vo-artifact", events),
    }
    captures = []

    class Worker:
        def __init__(self, **kwargs):
            captures.append(kwargs)

        def __call__(self, payload, *, commit_artifacts, commit_control):
            assert payload == raw
            commit_artifacts()
            commit_control()
            return {
                "schema_version": "synaptic-modal-runtime-release-qualification-result/v1",
                "status_code": "completed",
            }

    monkeypatch.setattr(
        "tuner.execution.providers.modal.runtime_release_qualification."
        "ModalRuntimeReleaseQualificationWorker",
        Worker,
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.runtime_release_qualification."
        "ModalRuntimeReleaseQualificationRoots",
        lambda control, artifacts: SimpleNamespace(
            control=control, artifacts=artifacts,
        ),
    )
    assert tuple(inspect.signature(run_runtime_release_self_check).parameters) == (
        "dispatch_bytes",
    )
    result = run_runtime_release_self_check(raw)
    assert result["status_code"] == "completed"
    assert _Client.calls == 1
    assert events == ["vo-artifact", "vo-control"]
    assert [item[0] for item in _Volume.calls] == ["control-name", "artifact-name"]
    assert all(item[1]["client"] is _Client.value for item in _Volume.calls)
    assert all(item[1]["create_if_missing"] is False for item in _Volume.calls)
    assert captures[0]["expected_facts"] == facts
    assert captures[0]["call_id_provider"]() == "fc-live"


@pytest.mark.parametrize("fault", ("missing_key", "wrong_image", "wrong_volume"))
def test_live_parent_fails_closed_before_worker_for_auth_or_resource_drift(
    monkeypatch, fault: str,
) -> None:
    raw, _ = _case(monkeypatch)
    _Volume.calls = []
    _Volume.registry = {
        "control-name": _VolumeHandle("vo-control", []),
        "artifact-name": _VolumeHandle(
            "vo-other" if fault == "wrong_volume" else "vo-artifact", [],
        ),
    }
    if fault == "missing_key":
        monkeypatch.delenv(QUALIFICATION_HMAC_ENV_KEY)
    elif fault == "wrong_image":
        monkeypatch.setenv("MODAL_IMAGE_ID", "im-other")
    result = run_runtime_release_self_check(raw)
    assert result == {
        "schema_version": "synaptic-modal-runtime-release-qualification-result/v1",
        "status_code": "failed",
    }


def test_generic_installed_child_remains_provider_neutral() -> None:
    import tuner.runtime.packaged_training_worker as child

    source = inspect.getsource(child)
    assert "import modal" not in source
    assert "from modal" not in source
