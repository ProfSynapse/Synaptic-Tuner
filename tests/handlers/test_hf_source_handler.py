from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import tuner.handlers.hf_source_handler as module
from tuner.cloud.hf_provisioning_operator import (
    HFProvisioningFailure,
    HFProvisioningOutcome,
)
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.hf_source_handler import (
    HFSourceHandler,
    _persist_immutable,
    _require_explicit_env_file,
    _resolve_explicit_hf_token,
)
from tuner.project import ProjectContext


class FakeTracking:
    def __init__(self, *, project_context=None):
        self.base_dir = project_context.tracking_root
        self.project_context = project_context
        self.transitions: list[str] = []

    def resolve_uri(self, value):
        return self.base_dir / value.removeprefix("tracking://")

    def tracking_uri(self, path):
        return f"tracking://{path.relative_to(self.base_dir).as_posix()}"

    def record_provisioning_acknowledged(self, experiment, *, uri, sha256):
        self.transitions.append("ACKNOWLEDGED")
        experiment.provisioning_evidence_uri = uri
        experiment.provisioning_evidence_sha256 = sha256
        experiment.source_transport_state = "ACKNOWLEDGED"

    def mark_source_transport_consumable(self, experiment):
        self.transitions.append("CONSUMABLE")
        experiment.source_transport_state = "CONSUMABLE"


def _args(env_file: Path) -> Namespace:
    return Namespace(
        experiment_id="exp-test",
        actor="operator-1",
        authority="operator",
        env_file=env_file,
        json=True,
    )


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.standalone(engine_root=tmp_path)


def _experiment():
    return SimpleNamespace(
        experiment_id="exp-test",
        source_transport_state="PREPARED",
        source_transport_uri="tracking://experiments/exp-test/cloud/hf/source-transport/descriptor.json",
        source_transport_sha256="1" * 64,
        source_lock_uri="tracking://experiments/exp-test/source-lock.json",
        provisioning_evidence_uri=None,
        provisioning_evidence_sha256=None,
    )


def test_handler_requires_explicit_env_file(tmp_path) -> None:
    context = _context(tmp_path)
    handler = HFSourceHandler(_args(tmp_path / "missing"), context=context)
    with pytest.raises(CloudProviderError, match="unavailable"):
        handler.provision()
    handler = HFSourceHandler(
        Namespace(experiment_id="x", actor="y", env_file=None),
        context=context,
    )
    with pytest.raises(CloudProviderError, match="explicit"):
        handler.provision()


def test_handler_persists_then_acknowledges_consumes_and_marks_consumable(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env.jp"
    env_file.write_text("HF_TOKEN=not-recorded\n", encoding="utf-8")
    context = _context(tmp_path)
    experiment = _experiment()
    tracking = FakeTracking(project_context=context)
    evidence = {"safe": "bounded"}
    outcome = HFProvisioningOutcome(
        evidence=evidence,
        evidence_sha256="a" * 64,
        mutated=True,
    )
    order = []
    provider_call = {}

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setattr(module, "TrackingService", lambda project_context=None: tracking)
    monkeypatch.setattr(module, "load_experiment", lambda *args: experiment)
    monkeypatch.setattr(
        module,
        "provision_hf_source_transport",
        lambda *args, **kwargs: outcome,
    )
    monkeypatch.setattr(
        module,
        "consume_hf_source_transport",
        lambda *args, **kwargs: order.append("consume") or SimpleNamespace(evidence_sha256="a" * 64),
    )
    original_ack = tracking.record_provisioning_acknowledged
    original_mark = tracking.mark_source_transport_consumable
    tracking.record_provisioning_acknowledged = lambda *args, **kwargs: (
        order.append("ack"), original_ack(*args, **kwargs)
    )[-1]
    tracking.mark_source_transport_consumable = lambda *args, **kwargs: (
        order.append("mark"), original_mark(*args, **kwargs)
    )[-1]

    handler = HFSourceHandler(
        _args(env_file),
        context=context,
        provider_factory=lambda **kwargs: provider_call.update(kwargs) or SimpleNamespace(),
    )
    result = handler.provision()
    assert result is outcome
    assert order == ["ack", "consume", "mark"]
    evidence_path = context.tracking_root / "experiments/exp-test/cloud/hf/source-transport/provisioning-evidence.json"
    assert evidence_path.exists()
    assert b"not-recorded" not in evidence_path.read_bytes()
    assert provider_call == {"token": "not-recorded"}
    assert "HF_TOKEN" not in module.os.environ
    assert tracking.transitions == ["ACKNOWLEDGED", "CONSUMABLE"]


def test_ambiguous_mutation_never_persists_or_transitions(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env.jp"
    env_file.write_text("HF_TOKEN=hidden\n", encoding="utf-8")
    context = _context(tmp_path)
    experiment = _experiment()
    tracking = FakeTracking(project_context=context)
    failure = HFProvisioningOutcome(
        mutated=True,
        failure=HFProvisioningFailure("mutation_ambiguous", "bounded", False),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setattr(module, "TrackingService", lambda project_context=None: tracking)
    monkeypatch.setattr(module, "load_experiment", lambda *args: experiment)
    monkeypatch.setattr(module, "provision_hf_source_transport", lambda *args, **kwargs: failure)
    handler = HFSourceHandler(_args(env_file), context=context, provider_factory=lambda **kwargs: object())
    assert handler.provision() is failure
    assert tracking.transitions == []
    assert not list(context.tracking_root.rglob("provisioning-evidence.json"))


def test_immutable_evidence_persistence_is_idempotent_and_never_overwrites(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    _persist_immutable(path, b"same")
    _persist_immutable(path, b"same")
    with pytest.raises(CloudProviderError, match="different"):
        _persist_immutable(path, b"changed")
    assert path.read_bytes() == b"same"


@pytest.mark.parametrize(
    ("name", "ambient_value", "file_value"),
    [
        ("HF_TOKEN", "ambient-different", "file-authority"),
        ("HF_TOKEN", "same-value", "same-value"),
        ("HF_API_KEY", "ambient-alias", "file-authority"),
    ],
)
def test_ambient_hf_credentials_are_rejected_even_when_token_matches(
    tmp_path, monkeypatch, name, ambient_value, file_value
) -> None:
    path = tmp_path / ".env.jp"
    path.write_text(f"HF_TOKEN={file_value}\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setenv(name, ambient_value)
    with pytest.raises(CloudProviderError, match="ambient") as caught:
        _resolve_explicit_hf_token(path)
    assert ambient_value not in str(caught.value)
    assert file_value not in str(caught.value)


@pytest.mark.parametrize(
    "contents",
    [
        "HF_API_KEY=alias-only\n",
        "HF_TOKEN=file-authority\nHF_API_KEY=alias-too\n",
    ],
)
def test_file_hf_api_key_alias_is_always_rejected(tmp_path, monkeypatch, contents) -> None:
    path = tmp_path / ".env.jp"
    path.write_text(contents, encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    with pytest.raises(CloudProviderError, match="rejects HF_API_KEY"):
        _resolve_explicit_hf_token(path)


@pytest.mark.parametrize("contents", ["", "# no credential\n", "HF_TOKEN=\n", "HF_TOKEN='  '\n"])
def test_file_hf_token_must_be_present_and_nonblank(tmp_path, monkeypatch, contents) -> None:
    path = tmp_path / ".env.jp"
    path.write_text(contents, encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    with pytest.raises(CloudProviderError, match="non-empty HF_TOKEN"):
        _resolve_explicit_hf_token(path)


def test_dotenv_quotes_and_comments_are_parsed_without_mutating_environment(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env.jp"
    path.write_text('export HF_TOKEN="file-authority" # selected\nOTHER=value\n', encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    assert _resolve_explicit_hf_token(path) == "file-authority"
    assert "HF_TOKEN" not in module.os.environ


def test_env_file_must_be_contained_in_project_boundary(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.env"
    outside.write_text("HF_TOKEN=value\n", encoding="utf-8")
    context = _context(project)
    with pytest.raises(CloudProviderError, match="project/config boundary"):
        _require_explicit_env_file(outside, context=context)


def test_env_file_cannot_traverse_symlink(tmp_path) -> None:
    target = tmp_path / "target.env"
    target.write_text("HF_TOKEN=value\n", encoding="utf-8")
    link = tmp_path / "selected.env"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(CloudProviderError, match="cannot traverse links"):
        _require_explicit_env_file(link, context=_context(tmp_path))
