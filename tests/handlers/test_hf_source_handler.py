from __future__ import annotations

import json
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import tuner.handlers.hf_source_handler as module
from tuner.cloud.hf_provisioning_operator import HFProvisioningOutcome
from tuner.core.exceptions import CloudProviderError
from tuner.handlers._hf_secret_file import preflight_hf_secret_file, read_claimed_hf_token
from tuner.handlers.hf_source_handler import (
    HFSourceHandler,
    _load_committed_volume_settings,
    _persist_immutable,
    _require_external_base_dir,
    _require_recoverable_bootstrap_experiment,
)
from tuner.project import ProjectContext


def _context(root: Path) -> ProjectContext:
    engine = root / "engine"
    engine.mkdir(exist_ok=True)
    return ProjectContext.standalone(engine_root=engine, invocation_cwd=engine)


def _args(base: Path, **overrides) -> Namespace:
    values = dict(
        subcommand="provision", experiment_id="exp-test", actor="operator-1",
        authority="operator", env_file=None, source_config=None, source_mode=None,
        base_dir=str(base), json=True, events=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_handler_dispatches_only_explicit_prepare_or_provision(monkeypatch, capsys):
    handler = HFSourceHandler(_args(Path("C:/external"), subcommand=None))
    assert handler.handle() == 1
    assert "prepare or provision" in capsys.readouterr().out
    handler = HFSourceHandler(_args(Path("C:/external"), subcommand="prepare"))
    monkeypatch.setattr(handler, "prepare", lambda: {"status": "PREPARED"})
    assert handler.handle() == 0


def test_external_base_dir_must_be_absolute_and_outside_source(tmp_path):
    context = _context(tmp_path)
    with pytest.raises(CloudProviderError, match="absolute"):
        _require_external_base_dir("relative", context=context)
    with pytest.raises(CloudProviderError, match="outside source"):
        _require_external_base_dir(context.engine_root / "tracking", context=context)
    external = tmp_path / "external"
    assert _require_external_base_dir(external.resolve(), context=context) == external.resolve()


def _recovery_experiment(**overrides):
    values = dict(
        source_lock_uri=None, source_lock_sha256=None,
        source_transport_uri=None, source_transport_sha256=None,
        source_transport_state=None, provisioning_evidence_uri=None,
        provisioning_evidence_sha256=None, hf_run_approval_uri=None,
        hf_submission_state=None, provider="hf_jobs", method="bootstrap",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_prepare_recovery_accepts_source_lock_partial_and_exact_prepared_state():
    _require_recoverable_bootstrap_experiment(_recovery_experiment())
    _require_recoverable_bootstrap_experiment(_recovery_experiment(
        source_lock_uri="tracking://source-lock.json", source_lock_sha256="a" * 64,
    ))
    _require_recoverable_bootstrap_experiment(_recovery_experiment(
        source_lock_uri="tracking://source-lock.json", source_lock_sha256="a" * 64,
        source_transport_uri="tracking://descriptor.json", source_transport_sha256="d" * 64,
        source_transport_state="PREPARED",
    ))
    with pytest.raises(CloudProviderError, match="ACKNOWLEDGED|neutral or PREPARED"):
        _require_recoverable_bootstrap_experiment(_recovery_experiment(
            source_lock_uri="tracking://source-lock.json", source_lock_sha256="a" * 64,
            source_transport_uri="tracking://descriptor.json", source_transport_sha256="d" * 64,
            source_transport_state="ACKNOWLEDGED",
            provisioning_evidence_uri="tracking://evidence.json",
            provisioning_evidence_sha256="e" * 64,
        ))


def test_volume_policy_is_parsed_from_exact_committed_blob_not_worktree(tmp_path, monkeypatch):
    context = _context(tmp_path)
    config = context.engine_root / "source.yaml"
    config.write_text("cloud: {hf_jobs: {bootstrap_volume: {source: attacker/bucket, path_prefix: changed}}}\n", encoding="utf-8")
    committed = b"cloud: {hf_jobs: {bootstrap_volume: {source: owner/bucket, path_prefix: committed/source}}}\n"
    monkeypatch.setattr(
        module.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=committed),
    )
    source = SimpleNamespace(commit="1" * 40)
    lock = SimpleNamespace(engine_source=source, project_source=source)
    settings = _load_committed_volume_settings(config, context=context, source_lock=lock)
    assert settings == {"source": "owner/bucket", "path_prefix": "committed/source"}


@pytest.mark.parametrize(
    "contents",
    [
        "HF_API_KEY=alias\n", "OTHER=value\nHF_TOKEN=hf_value\n",
        "HF_TOKEN=one\nHF_TOKEN=two\n", "export HF_TOKEN=hf_value\n",
        "HF_TOKEN=${OTHER}\n", "HF_TOKEN=hf_value # inline\n", "\x00HF_TOKEN=hf_value\n",
    ],
)
def test_shared_secret_boundary_rejects_noncanonical_documents(tmp_path, monkeypatch, contents):
    context = _context(tmp_path)
    selected = context.engine_root / "selected.env"
    selected.write_text(contents, encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    claim = preflight_hf_secret_file(selected, context=context)
    with pytest.raises(CloudProviderError):
        read_claimed_hf_token(claim)


def test_shared_secret_boundary_rejects_ambient_before_read(tmp_path, monkeypatch):
    context = _context(tmp_path)
    selected = context.engine_root / "selected.env"
    selected.write_text("HF_TOKEN=hf_file\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_ambient")
    with pytest.raises(CloudProviderError, match="ambient"):
        preflight_hf_secret_file(selected, context=context)


def test_non_prepared_state_rejects_before_secret_selection_or_read(
    tmp_path, monkeypatch
):
    context = _context(tmp_path)
    base = (tmp_path / "external").resolve()
    base.mkdir()
    experiment = SimpleNamespace(
        experiment_id="exp-test",
        source_transport_state=None,
    )
    calls: list[str] = []

    class Tracking:
        base_dir = base

    monkeypatch.setattr(module, "TrackingService", lambda base_dir=None: Tracking())
    monkeypatch.setattr(module, "load_experiment", lambda *args: experiment)
    monkeypatch.setattr(
        module,
        "preflight_hf_secret_file",
        lambda *args, **kwargs: calls.append("metadata"),
    )
    monkeypatch.setattr(
        module,
        "read_claimed_hf_token",
        lambda claim: calls.append("read") or "hf_forbidden",
    )
    handler = HFSourceHandler(
        _args(base, env_file=str(context.engine_root / "missing.env")),
        context=context,
    )
    with pytest.raises(CloudProviderError, match="PREPARED"):
        handler.provision()
    assert calls == []


def test_provision_reauthenticates_before_credential_read_or_provider_import(tmp_path, monkeypatch):
    context = _context(tmp_path)
    base = (tmp_path / "external").resolve()
    base.mkdir()
    secret = context.engine_root / "selected.env"
    secret.write_text("HF_TOKEN=hf_selected\n", encoding="utf-8")
    descriptor = base / "experiments/exp-test/cloud/hf/source-transport/descriptor.json"
    descriptor.parent.mkdir(parents=True)
    experiment = SimpleNamespace(
        experiment_id="exp-test", source_transport_state="PREPARED",
        source_transport_uri="tracking://experiments/exp-test/cloud/hf/source-transport/descriptor.json",
        source_transport_sha256="d" * 64,
        source_lock_uri="tracking://experiments/exp-test/source-lock.json",
        source_lock_sha256="s" * 64,
        hf_provisioning_state=None, hf_provisioning_event_uri=None,
        provisioning_evidence_uri=None, provisioning_evidence_sha256=None,
    )
    order: list[str] = []

    class Tracking:
        base_dir = base
        def verify_experiment_provenance(self, value): order.append("durable")
        def resolve_uri(self, uri): return descriptor
        def tracking_uri(self, path): return "tracking://evidence.json"
        @contextmanager
        def hf_provisioning_execution_lock(self, experiment_id):
            order.append("lock")
            yield
        def claim_hf_provisioning(self, exp, claim):
            order.append("claim")
            return SimpleNamespace(
                document=claim, event_uri="tracking://claim.json",
                event_sha256="c" * 64, state="CLAIMED",
                provider_attempt_authorized=True,
            )
        def record_hf_provisioning_succeeded(self, *args, **kwargs):
            order.append("succeeded")
            experiment.source_transport_state = "ACKNOWLEDGED"
            experiment.hf_provisioning_state = "SUCCEEDED"
        def mark_source_transport_consumable(self, *args): order.append("consume-state")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setattr(module, "TrackingService", lambda base_dir=None: Tracking())
    monkeypatch.setattr(module, "load_experiment", lambda *args: experiment)
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.load_hf_source_transport",
        lambda *args, **kwargs: order.append("bundle") or SimpleNamespace(
            descriptor_sha256="d" * 64,
            descriptor={"source_lock": {"sha256": "s" * 64}},
        ),
    )
    outcome = HFProvisioningOutcome(evidence={"safe": True}, evidence_sha256="e" * 64, mutated=False)
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning_operator.provision_hf_source_transport",
        lambda *args, **kwargs: order.append("provider") or outcome,
    )
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.consume_hf_source_transport",
        lambda *args, **kwargs: SimpleNamespace(evidence_sha256="e" * 64),
    )
    monkeypatch.setattr(
        module, "preflight_hf_secret_file",
        lambda *args, **kwargs: order.append("secret-metadata") or object(),
    )
    monkeypatch.setattr(
        module, "read_claimed_hf_token",
        lambda claim: order.append("secret-read") or "hf_selected",
    )
    monkeypatch.setattr(
        module, "build_hf_provisioning_claim",
        lambda **kwargs: {"state": "CLAIMED"},
    )
    monkeypatch.setattr(
        module, "build_hf_provisioning_succeeded_event",
        lambda *args, **kwargs: {"state": "SUCCEEDED"},
    )
    handler = HFSourceHandler(
        _args(base, env_file=str(secret)), context=context,
        provider_factory=lambda **kwargs: order.append("provider-import") or object(),
    )
    assert handler.provision() is outcome
    assert order[:5] == ["durable", "bundle", "lock", "secret-metadata", "claim"]
    assert order[5:8] == ["secret-read", "provider-import", "provider"]
    assert order[-2:] == ["succeeded", "consume-state"]


def test_consumed_provisioning_claim_never_reads_secret_or_calls_provider(
    tmp_path, monkeypatch
):
    context = _context(tmp_path)
    base = (tmp_path / "external").resolve()
    base.mkdir()
    descriptor = base / "experiments/exp-test/cloud/hf/source-transport/descriptor.json"
    descriptor.parent.mkdir(parents=True)
    experiment = SimpleNamespace(
        experiment_id="exp-test", source_transport_state="PREPARED",
        source_transport_uri="tracking://experiments/exp-test/cloud/hf/source-transport/descriptor.json",
        source_transport_sha256="d" * 64,
        source_lock_uri="tracking://experiments/exp-test/source-lock.json",
        source_lock_sha256="s" * 64,
        hf_provisioning_state=None, hf_provisioning_event_uri=None,
        provisioning_evidence_uri=None, provisioning_evidence_sha256=None,
    )
    calls: list[str] = []

    class Tracking:
        base_dir = base
        def verify_experiment_provenance(self, value): pass
        def resolve_uri(self, uri): return descriptor
        @contextmanager
        def hf_provisioning_execution_lock(self, experiment_id): yield
        def claim_hf_provisioning(self, exp, claim):
            return SimpleNamespace(
                document=claim, event_uri="tracking://claim.json",
                event_sha256="c" * 64, state="CLAIMED",
                provider_attempt_authorized=False,
            )

    monkeypatch.setattr(module, "TrackingService", lambda base_dir=None: Tracking())
    monkeypatch.setattr(module, "load_experiment", lambda *args: experiment)
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.load_hf_source_transport",
        lambda *args, **kwargs: SimpleNamespace(
            descriptor_sha256="d" * 64,
            descriptor={"source_lock": {"sha256": "s" * 64}},
        ),
    )
    monkeypatch.setattr(
        module, "preflight_hf_secret_file",
        lambda *args, **kwargs: calls.append("metadata") or object(),
    )
    monkeypatch.setattr(
        module, "read_claimed_hf_token",
        lambda claim: calls.append("read") or "hf_forbidden",
    )
    monkeypatch.setattr(
        module, "build_hf_provisioning_claim",
        lambda **kwargs: {"state": "CLAIMED"},
    )
    handler = HFSourceHandler(
        _args(base, env_file=str(context.engine_root / "selected.env")),
        context=context,
        provider_factory=lambda **kwargs: calls.append("provider") or object(),
    )
    with pytest.raises(CloudProviderError, match="already consumed"):
        handler.provision()
    assert calls == ["metadata"]


def test_claimed_recovery_without_terminal_or_evidence_closes_interrupted_without_retry(
    tmp_path, monkeypatch
):
    context = _context(tmp_path)
    base = (tmp_path / "external").resolve()
    base.mkdir()
    descriptor = base / "experiments/exp-test/cloud/hf/source-transport/descriptor.json"
    descriptor.parent.mkdir(parents=True)
    experiment = SimpleNamespace(
        experiment_id="exp-test", source_transport_state="PREPARED",
        source_transport_uri="tracking://experiments/exp-test/cloud/hf/source-transport/descriptor.json",
        source_transport_sha256="d" * 64,
        source_lock_uri="tracking://experiments/exp-test/source-lock.json",
        source_lock_sha256="s" * 64,
        hf_provisioning_state="CLAIMED", hf_provisioning_event_uri="tracking://claim.json",
        provisioning_evidence_uri=None, provisioning_evidence_sha256=None,
    )
    reasons: list[str] = []
    claim = {"state": "CLAIMED", "actor": "operator-1", "authority": "operator"}

    class Tracking:
        base_dir = base
        def verify_experiment_provenance(self, value): pass
        def resolve_uri(self, uri): return descriptor
        @contextmanager
        def hf_provisioning_execution_lock(self, experiment_id): yield
        def claim_hf_provisioning(self, exp, value):
            return SimpleNamespace(
                document=claim, event_uri="tracking://claim.json",
                event_sha256="c" * 64, state="CLAIMED",
                provider_attempt_authorized=False,
            )
        def find_hf_provisioning_terminal(self, exp): return None
        def record_hf_provisioning_ambiguous(self, exp, event):
            reasons.append(event["reason_code"])
            exp.hf_provisioning_state = "AMBIGUOUS"

    monkeypatch.setattr(module, "TrackingService", lambda base_dir=None: Tracking())
    monkeypatch.setattr(module, "load_experiment", lambda *args: experiment)
    monkeypatch.setattr(module, "_recover_hf_provisioning_claim", lambda *args: claim)
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.load_hf_source_transport",
        lambda *args, **kwargs: SimpleNamespace(
            descriptor_sha256="d" * 64,
            descriptor={"source_lock": {"sha256": "s" * 64}},
        ),
    )
    monkeypatch.setattr(
        module, "preflight_hf_secret_file",
        lambda *args, **kwargs: pytest.fail("CLAIMED recovery selected a credential"),
    )
    monkeypatch.setattr(
        module, "read_claimed_hf_token",
        lambda claim: pytest.fail("CLAIMED recovery read credential content"),
    )
    monkeypatch.setattr(
        module, "build_hf_provisioning_ambiguous_event",
        lambda *args, reason_code, **kwargs: {"state": "AMBIGUOUS", "reason_code": reason_code},
    )
    handler = HFSourceHandler(
        _args(base, env_file=str(context.engine_root / "selected.env")),
        context=context,
        provider_factory=lambda **kwargs: pytest.fail("CLAIMED recovery called provider"),
    )
    with pytest.raises(CloudProviderError, match="interrupted"):
        handler.provision()
    assert reasons == ["INTERRUPTED_AFTER_CLAIM"]


def test_postclaim_credential_rejection_records_closed_reason_before_error(
    tmp_path, monkeypatch
):
    context = _context(tmp_path)
    base = (tmp_path / "external").resolve()
    base.mkdir()
    descriptor = base / "experiments/exp-test/cloud/hf/source-transport/descriptor.json"
    descriptor.parent.mkdir(parents=True)
    experiment = SimpleNamespace(
        experiment_id="exp-test", source_transport_state="PREPARED",
        source_transport_uri="tracking://experiments/exp-test/cloud/hf/source-transport/descriptor.json",
        source_transport_sha256="d" * 64,
        source_lock_uri="tracking://experiments/exp-test/source-lock.json",
        source_lock_sha256="s" * 64,
        hf_provisioning_state=None, hf_provisioning_event_uri=None,
        provisioning_evidence_uri=None, provisioning_evidence_sha256=None,
    )
    reasons: list[str] = []

    class Tracking:
        base_dir = base
        def verify_experiment_provenance(self, value): pass
        def resolve_uri(self, uri): return descriptor
        @contextmanager
        def hf_provisioning_execution_lock(self, experiment_id): yield
        def claim_hf_provisioning(self, exp, claim):
            exp.hf_provisioning_state = "CLAIMED"
            return SimpleNamespace(
                document=claim, event_uri="tracking://claim.json",
                event_sha256="c" * 64, state="CLAIMED", provider_attempt_authorized=True,
            )
        def record_hf_provisioning_ambiguous(self, exp, event):
            reasons.append(event["reason_code"])

    monkeypatch.setattr(module, "TrackingService", lambda base_dir=None: Tracking())
    monkeypatch.setattr(module, "load_experiment", lambda *args: experiment)
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.load_hf_source_transport",
        lambda *args, **kwargs: SimpleNamespace(
            descriptor_sha256="d" * 64,
            descriptor={"source_lock": {"sha256": "s" * 64}},
        ),
    )
    monkeypatch.setattr(module, "preflight_hf_secret_file", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "read_claimed_hf_token", lambda claim: (_ for _ in ()).throw(CloudProviderError("bad token")))
    monkeypatch.setattr(module, "build_hf_provisioning_claim", lambda **kwargs: {"state": "CLAIMED"})
    monkeypatch.setattr(
        module, "build_hf_provisioning_ambiguous_event",
        lambda *args, reason_code, **kwargs: {"state": "AMBIGUOUS", "reason_code": reason_code},
    )
    handler = HFSourceHandler(_args(base, env_file="selected.env"), context=context)
    with pytest.raises(CloudProviderError, match="credential was rejected"):
        handler.provision()
    assert reasons == ["CREDENTIAL_REJECTED"]


def test_immutable_evidence_persistence_is_idempotent_and_never_overwrites(tmp_path):
    path = tmp_path / "evidence.json"
    _persist_immutable(path, b"same")
    _persist_immutable(path, b"same")
    with pytest.raises(CloudProviderError, match="different"):
        _persist_immutable(path, b"changed")
    assert path.read_bytes() == b"same"


def test_prepare_output_is_portable_and_contains_no_local_paths(tmp_path, monkeypatch):
    context = _context(tmp_path)
    base = (tmp_path / "external").resolve()
    config = context.engine_root / "source.yaml"
    config.write_text(
        "cloud:\n  hf_jobs:\n    bootstrap_volume:\n      source: owner/bucket\n      path_prefix: prepared/source\n",
        encoding="utf-8",
    )
    experiment = SimpleNamespace(
        experiment_id="exp-1", source_lock_uri=None, source_lock_sha256=None,
        source_transport_uri=None, source_transport_sha256=None,
        source_transport_state=None, provisioning_evidence_uri=None,
        provisioning_evidence_sha256=None, hf_run_approval_uri=None,
        hf_submission_state=None, provider="hf_jobs", method="bootstrap",
    )

    class Tracking:
        base_dir = base
        def create_experiment(self, **kwargs): return experiment
        def tracking_uri(self, path): return f"tracking://{path.relative_to(base).as_posix()}"
        @contextmanager
        def hf_source_preparation_execution_lock(self, experiment_id): yield
        def persist_source_lock(self, exp, lock):
            exp.source_lock_uri = "tracking://experiments/exp-1/source-lock.json"
            exp.source_lock_sha256 = "a" * 64
            return lock
        def record_source_transport_prepared(self, exp, *, uri, sha256):
            exp.source_transport_uri, exp.source_transport_sha256 = uri, sha256

    transport = base / "experiments/exp-1/cloud/hf/source-transport"
    transport.mkdir(parents=True)
    (transport / "descriptor.json").write_text(json.dumps({
        "volume": {"type": "bucket", "source": "owner/bucket", "path": "prepared/source/exp-1/digest", "mount_path": "/workspace/input", "read_only": True}
    }), encoding="utf-8")
    prepared = SimpleNamespace(
        source_lock=object(), descriptor_uri="tracking://experiments/exp-1/cloud/hf/source-transport/descriptor.json",
        descriptor_sha256="d" * 64, staging_root=transport,
        descriptor={"volume": {"type": "bucket", "source": "owner/bucket", "path": "prepared/source/exp-1/digest", "mount_path": "/workspace/input", "read_only": True}},
    )
    monkeypatch.setattr(module, "TrackingService", lambda base_dir=None: Tracking())
    monkeypatch.setattr(module, "load_experiment", lambda *args, **kwargs: experiment)
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.load_canonical_json",
        lambda *args, **kwargs: json.loads((transport / "descriptor.json").read_text(encoding="utf-8")),
    )
    monkeypatch.setattr("tuner.handlers.stages._util.preflight_hf_source_lock", lambda *args, **kwargs: object())
    monkeypatch.setattr("tuner.handlers.stages._util.finalize_hf_source_lock", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "_load_committed_volume_settings", lambda *args, **kwargs: {"source": "owner/bucket", "path_prefix": "prepared/source"})
    monkeypatch.setattr("tuner.handlers.stages._util.prepare_hf_source", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(module, "_load_exact_prepared_transport", lambda **kwargs: prepared)
    result = HFSourceHandler(
        _args(
            base, subcommand="prepare", source_config=str(config), experiment_id=None,
            actor=None, authority=None,
        ),
        context=context,
    ).prepare()
    serialized = json.dumps(result)
    assert result["status"] == "PREPARED"
    assert str(base) not in serialized and str(context.engine_root) not in serialized
