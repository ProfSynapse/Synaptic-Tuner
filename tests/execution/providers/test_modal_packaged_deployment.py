"""Provider-free qualification of read-only packaged deployment admission."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.packaged_deployment import (
    ModalPackagedDeploymentObserver,
    observe_modal_packaged_deployment,
)

from tests.execution.providers.test_modal_packaged_binding import (
    _release_and_execution,
)
from tests.execution.providers.test_modal_sdk154_adapter import SDK


class Reader:
    def __init__(self, value) -> None:
        self.value = value
        self.calls: list[dict[str, object]] = []

    def observe(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


def _observer(reader: Reader):
    _, facts, _, _, _ = _release_and_execution()
    client = object()
    observer = ModalPackagedDeploymentObserver(
        sdk=SDK, client=client, client_binding=facts.client_binding, reader=reader,
    )
    return observer, client, facts


def test_observes_exact_existing_deployment_once_with_explicit_client() -> None:
    release, facts, provider, _, _ = _release_and_execution()
    observer, client, _ = _observer(Reader(facts.canonical_bytes))
    assert observe_modal_packaged_deployment(
        observer, facts, runtime_release=release, provider_binding=provider,
    ) == facts
    assert observer._reader.calls == [{
        "client": client,
        "app_name": facts.app_name,
        "function_name": facts.function_name,
        "environment_name": facts.environment_ref,
    }]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("app_id", "ap-other"),
        ("deployment_generation", 8),
        ("function_id", "fu-other"),
        ("image_id", "im-other"),
        ("control_volume_id", "vo-other"),
        ("artifact_volume_id", "vo-other"),
    ),
)
def test_release_deployment_generation_or_layout_drift_is_rejected(
    field: str, value: object,
) -> None:
    release, facts, provider, _, _ = _release_and_execution()
    observed = replace(facts, **{field: value})
    observer, _, _ = _observer(Reader(observed))
    with pytest.raises(ValueError, match="deployment changed"):
        observer.observe(
            facts, runtime_release=release, provider_binding=provider,
        )
    assert len(observer._reader.calls) == 1


def test_provider_binding_substitution_is_rejected() -> None:
    release, facts, _, _, _ = _release_and_execution()
    changed = replace(facts, deployment_generation=8)
    observer, _, _ = _observer(Reader(facts))
    with pytest.raises(ValueError, match="deployment changed"):
        observer.observe(
            facts, runtime_release=release,
            provider_binding=changed.build_provider_binding(release),
        )


def test_explicit_client_scope_mismatch_stops_before_reader() -> None:
    release, facts, provider, _, _ = _release_and_execution()
    reader = Reader(facts)
    observer = ModalPackagedDeploymentObserver(
        sdk=SDK, client=object(),
        client_binding=ModalClientBinding(
            "other-account", facts.workspace_ref, facts.environment_ref,
            facts.client_ref, facts.sdk_version,
        ),
        reader=reader,
    )
    with pytest.raises(ValueError, match="scope differs"):
        observer.observe(
            facts, runtime_release=release, provider_binding=provider,
        )
    assert reader.calls == []


@pytest.mark.parametrize("version", (None, "1.5.3", "1.5.5"))
def test_exact_sdk_version_is_required_without_reader_access(version) -> None:
    _, facts, _, _, _ = _release_and_execution()
    reader = Reader(facts)
    sdk = type("SDKVersion", (), {"__version__": version})
    with pytest.raises(ValueError, match="SDK version"):
        ModalPackagedDeploymentObserver(
            sdk=sdk, client=object(), client_binding=facts.client_binding,
            reader=reader,
        )
    assert reader.calls == []


def test_provider_failure_is_closed_and_does_not_leak_detail() -> None:
    release, facts, provider, _, _ = _release_and_execution()
    observer, _, _ = _observer(Reader(RuntimeError("credential=private-value")))
    with pytest.raises(ValueError) as caught:
        observer.observe(
            facts, runtime_release=release, provider_binding=provider,
        )
    assert str(caught.value) == "Modal packaged deployment observation unavailable"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_current_state_admission_does_not_claim_version_pinned_invocation() -> None:
    """One exact observation admits now; the later admin race remains external."""
    release, facts, provider, _, _ = _release_and_execution()
    reader = Reader(facts)
    observer, _, _ = _observer(reader)
    admitted = observer.observe(
        facts, runtime_release=release, provider_binding=provider,
    )
    reader.value = replace(facts, deployment_generation=8)
    # The retained admission remains only the prior observation, not proof that
    # the current floating deployment cannot change after the final check.
    assert admitted.deployment_generation == 7
    with pytest.raises(ValueError, match="deployment changed"):
        observer.observe(
            facts, runtime_release=release, provider_binding=provider,
        )
