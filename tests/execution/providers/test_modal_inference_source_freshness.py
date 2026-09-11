"""Read-only Modal source freshness does not create verification transitions."""

from dataclasses import replace

import pytest

from tests.execution.providers.test_modal_inference_source_integration import _case
from tuner.execution.providers.modal.inference_binding import ModalInferenceBindingError


def test_repeated_bind_reverifies_and_changes_revision(monkeypatch):
    binder, runs, run, operations, store, *_ = _case(monkeypatch)
    first = binder.bind(runs, run)
    first_revision = store.value.revision
    second = binder.bind(runs, run)
    assert store.value.revision == first_revision + 1
    assert first.source_revision != second.source_revision
    assert first.read_request_digest != second.read_request_digest
    assert operations.calls == ["reverify", "outcome", "reverify", "outcome"]


def test_repeated_current_checks_are_read_only(monkeypatch):
    binder, runs, run, operations, store, *_ = _case(monkeypatch)
    source = binder.bind(runs, run)
    revision = store.value.revision
    record_digest = store.value.record_digest
    operations.calls.clear()
    binder.assert_current(source)
    binder.assert_current(source)
    assert store.value.revision == revision
    assert store.value.record_digest == record_digest
    assert operations.calls == ["outcome", "outcome"]


def test_workflow_drift_is_denied_without_another_reverify(monkeypatch):
    binder, runs, run, operations, store, *_ = _case(monkeypatch)
    source = binder.bind(runs, run)
    operations.calls.clear()
    object.__setattr__(store.value, "revision", store.value.revision + 1)
    with pytest.raises(ModalInferenceBindingError):
        binder.assert_current(source)
    assert operations.calls == ["outcome"]


def test_source_mutation_is_denied(monkeypatch):
    binder, runs, run, operations, _, *_ = _case(monkeypatch)
    source = binder.bind(runs, run)
    operations.calls.clear()
    object.__setattr__(source, "manifest_digest", "0" * 64)
    with pytest.raises(ModalInferenceBindingError):
        binder.assert_current(source)
    assert operations.calls == ["outcome"]


def test_run_mutation_is_denied_before_public_or_native_read(monkeypatch):
    binder, runs, run, operations, _, *_ = _case(monkeypatch)
    source = binder.bind(runs, run)
    native_calls = []
    original = binder._reader.native_artifacts

    def counted(request):
        native_calls.append(request)
        return original(request)

    monkeypatch.setattr(binder._reader, "native_artifacts", counted)
    operations.calls.clear()
    object.__setattr__(source.run, "run_id", "alternate-local-run")
    with pytest.raises(ModalInferenceBindingError):
        binder.assert_current(source)
    assert operations.calls == []
    assert native_calls == []


def test_native_inventory_drift_is_denied(monkeypatch):
    binder, runs, run, operations, _, *_ = _case(monkeypatch)
    source = binder.bind(runs, run)
    original = binder._reader.native_artifacts

    def changed(request):
        binding, reference, inventory, manifest = original(request)
        return (
            binding,
            reference,
            inventory,
            replace(manifest, manifest_digest="0" * 64),
        )

    monkeypatch.setattr(binder._reader, "native_artifacts", changed)
    operations.calls.clear()
    with pytest.raises(ModalInferenceBindingError):
        binder.assert_current(source)
    assert operations.calls == ["outcome"]
