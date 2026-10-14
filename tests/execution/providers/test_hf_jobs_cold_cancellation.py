"""Configured known-job cancellation on reopened SQLite; no live effects."""

import pytest
import json
import os
import selectors
import shutil
import signal
import subprocess

sdk = pytest.importorskip("huggingface_hub")

from tests.execution.providers.test_hf_jobs_public_family import material as material
from tests.execution.providers.test_hf_jobs_public_lifecycle import (
    host,
    start,
    cancellation,
)
from tuner.execution.providers.hf_jobs_provider_v1.preparation import HFJobsPreparation
from tuner.execution.providers.hf_jobs_provider_v1.retained_reader import (
    open_retained_hf_lifecycle,
    open_retained_hf_runs,
)
from tuner.execution.providers.hf_jobs_provider_v1.model import HFJobsError
from synaptic_tuner.api.v1.reference import compose_reference_authority
from synaptic_tuner.api.v1.reference.provider_family import ReferenceHostPortsV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.runs_facade import RunOperationError

pytestmark = pytest.mark.skipif(
    sdk.__version__ != "1.33.0", reason="exact isolated HF host SDK"
)


def configured(value, enabled=True):
    original = value.preparation
    value.preparation = HFJobsPreparation(
        selection=original.selection,
        training_plan=original.rich_plan,
        bundle=original.bundle,
        closure_members={
            p.removeprefix("engine/"): raw
            for p, raw in original.source_files.items()
            if p.startswith("engine/")
        },
        parent_members={
            p.removeprefix("parent/"): raw
            for p, raw in original.source_files.items()
            if p.startswith("parent/")
        },
        input_lease=original.input_lease,
        request_id=original.resolved.request_id,
        project_ref=original.resolved.project_ref,
        clock=original.clock,
        maximum_cost_minor_units=100,
        currency="USD",
        allow_cancel_reconcile=enabled,
    )
    return value


def cold(value, store, ports, run, scratch, *, lifecycle=True):
    authority = compose_reference_authority(
        clock=ports.clock,
        secrets=ports.secrets,
        authority_secret=SecretRef("env", "AUTHORITY"),
    )
    options = dict(
        run=run,
        authority=authority,
        sdk=sdk,
        api=value.provider,
        scratch_root=scratch,
        reader_authority_ref="fixture-reader",
        reader_key_ref="fixture-reader-key",
        reader_key=b"r" * 32,
    )
    if lifecycle:
        return open_retained_hf_lifecycle(
            **options,
            ports=ReferenceHostPortsV1(
                store, store, ports.clock, ports.grants, ports.secrets
            ),
            allow_current_cancel_authorization=True,
        )
    return open_retained_hf_runs(
        **options, records=store, streams=store, secrets=ports.secrets
    )


def test_genuine_configured_cold_cancel_ack_and_lookup(material, tmp_path):
    value = configured(material)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    database = root / "workflow.sqlite3"
    with SQLiteDurableStoreV1(database) as store:
        api, _, ports = host(value, records=store, streams=store, current=True)
        run, _, _ = start(api, value)
    cancellation(value)
    with SQLiteDurableStoreV1(database, create=False) as store:
        view = cold(value, store, ports, run, value.scratch)
        outcome = view.runs.cancel(run, "bounded fixture cancellation")
        assert outcome.state.value == "cancel_requested"
        assert len([c for c in value.provider.calls if c[0] == "cancel"]) == 1
    with SQLiteDurableStoreV1(database, create=False) as store:
        view = cold(value, store, ports, run, value.scratch)
        view.runs.reconcile(run)
        assert len([c for c in value.provider.calls if c[0] == "cancel"]) == 1
        assert len([c for c in value.provider.calls if c[0] == "submit"]) == 1


def test_historical_false_descriptor_cannot_be_promoted(material, tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    with SQLiteDurableStoreV1(root / "workflow.sqlite3") as store:
        api, _, ports = host(material, records=store, streams=store)
        run, _, _ = start(api, material)
        calls = list(material.provider.calls)
        with pytest.raises(HFJobsError):
            cold(material, store, ports, run, material.scratch)
        assert material.provider.calls == calls
        readonly = cold(material, store, ports, run, material.scratch, lifecycle=False)
        with pytest.raises(RunOperationError):
            readonly.runs.cancel(run, "not enabled")


def test_retained_mode_cannot_stage_submit_or_restore_source_authority(
    material, tmp_path
):
    from tuner.execution.foundation_v2.commands import parse_exact_command

    value = configured(material)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    with SQLiteDurableStoreV1(root / "workflow.sqlite3") as store:
        api, _, ports = host(value, records=store, streams=store, current=True)
        run, _, _ = start(api, value)
        view = cold(value, store, ports, run, value.scratch)
        operations = view.runs._operations._operations
        transport = operations._reader.transport
        workflow = view.runs._operations._workflows.get(run)
        assert transport.preparation is None
        calls = list(value.provider.calls)
        for operation, intent in (
            (transport.stage, workflow.stage),
            (transport.submit, workflow.submit),
            (transport.lookup, workflow.submit),
        ):
            with pytest.raises(HFJobsError):
                operation(parse_exact_command(intent.canonical_command_bytes))
        with pytest.raises(RunOperationError):
            view.runs.reconcile(run)
        readonly = cold(value, store, ports, run, value.scratch, lifecycle=False)
        with pytest.raises(RunOperationError):
            readonly.runs.cancel(run, "enabled descriptor is not a read grant")
        with pytest.raises(RunOperationError):
            readonly.runs.reconcile(run)
        assert value.provider.calls == calls


def test_changed_current_provider_binding_cannot_receive_cancel(material, tmp_path):
    from tests.execution.providers.test_hf_jobs_public_family import _job

    value = configured(material)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    database = root / "workflow.sqlite3"
    with SQLiteDurableStoreV1(database) as store:
        api, _, ports = host(value, records=store, streams=store, current=True)
        run, _, _ = start(api, value)
    arguments = next(c[1] for c in value.provider.calls if c[0] == "submit")
    value.provider.job = _job(
        {**arguments, "labels": {**arguments["labels"], "synaptic-material": "f" * 64}}
    )
    cancellation(value)
    with SQLiteDurableStoreV1(database, create=False) as store:
        view = cold(value, store, ports, run, value.scratch)
        assert (
            view.runs.cancel(run, "mismatched current job").state.value
            == "reconcile_required"
        )
        assert not [c for c in value.provider.calls if c[0] == "cancel"]
        view.runs.reconcile(run)
        assert not [c for c in value.provider.calls if c[0] == "cancel"]
        assert not view.runs.show(run).artifacts


@pytest.mark.parametrize("invalid", [0, 1, "true", None])
def test_initial_candidate_requires_exact_bool(material, invalid):
    with pytest.raises(HFJobsError):
        configured(material, invalid)


def test_absent_candidate_preserves_original_bytes(material):
    original = material.preparation
    value = configured(material, False).preparation
    assert value.descriptor == original.descriptor
    assert value.plan == original.plan
    assert value.context_value == original.context_value
    assert value.binding == original.binding
    assert value.material_bytes == original.material_bytes


@pytest.mark.parametrize("fault", ["receipt_insert", "after_receipt_commit"])
def test_cold_receipt_gap_never_repeats_cancel(material, tmp_path, monkeypatch, fault):
    from tuner.execution.providers.hf_jobs_provider_v1.transport import HFJobsTransport

    value = configured(material)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    database = root / "workflow.sqlite3"
    with SQLiteDurableStoreV1(database) as store:
        api, _, ports = host(value, records=store, streams=store, current=True)
        run, _, _ = start(api, value)
        cancellation(value)
        if fault == "receipt_insert":
            store._connection.execute(
                "CREATE TEMP TRIGGER reject_cancel_receipt BEFORE INSERT ON records "
                "WHEN NEW.partition='effects' AND instr(NEW.key,'/cancel/')>0 "
                "BEGIN SELECT RAISE(ABORT,'scripted retained-receipt fault'); END"
            )
        else:
            original = HFJobsTransport._retain

            def lose_result(self, kind, command, document):
                result = original(self, kind, command, document)
                if kind == "cancel":
                    raise HFJobsError("read_unavailable")
                return result

            monkeypatch.setattr(HFJobsTransport, "_retain", lose_result)
        assert (
            api.runs.cancel(run, "bounded original cancel").state.value
            == "reconcile_required"
        )
        retained = api.runs.show(run)
        assert retained.run == run
        assert len([c for c in value.provider.calls if c[0] == "cancel"]) == 1
    ports.clock.epoch += (
        1000  # New cancel/lookup policy never renews original submit assessment.
    )
    with SQLiteDurableStoreV1(database, create=False) as store:
        view = cold(value, store, ports, run, value.scratch)
        before = view.runs._operations._workflows.get(run)
        outcome = view.runs.reconcile(run)
        assert outcome.state.value == (
            "reconcile_required" if fault == "receipt_insert" else "cancel_requested"
        )
        after = view.runs._operations._workflows.get(run)
        assert after.submit.foundation_bindings == before.submit.foundation_bindings
        assert after.cancel.command_digest == before.cancel.command_digest
        assert (
            after.bound_cancellation is None
            if fault == "receipt_insert"
            else after.bound_cancellation is not None
        )
        view.runs.cancel(run, "bounded original cancel")
        assert len([c for c in value.provider.calls if c[0] == "cancel"]) == 1
        assert len([c for c in value.provider.calls if c[0] == "submit"]) == 1


@pytest.mark.parametrize("lost", [False, True])
def test_killed_configured_producer_fresh_cancel_and_reconcile(tmp_path, lost):
    from tests.execution.providers.test_hf_jobs_process_readback import (
        _command,
        _census,
    )

    root = tmp_path / "proof"
    root.mkdir(mode=0o700)
    scratch = root / "tmp"
    scratch.mkdir(mode=0o700)
    environment = {"PATH": os.defpath, "TMPDIR": str(scratch)}
    producer = subprocess.Popen(
        _command(root, "producer-cancel"),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(producer.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=60)
            raw = os.read(producer.stdout.fileno(), 4096)
            if not raw:
                producer.wait(timeout=5)
                raise AssertionError("configured producer did not publish readiness")
            ready = json.loads(raw)
        assert ready["pid"] == producer.pid
        producer.kill()
        assert producer.wait(timeout=5) == -signal.SIGKILL
    finally:
        if producer.poll() is None:
            producer.kill()
            producer.wait(timeout=5)
        producer.stdout.close()
        producer.stderr.close()
    shutil.rmtree(root / "original-inputs")
    shutil.rmtree(root / "provider-execution")

    def consumer(role):
        result = subprocess.run(
            _command(root, role),
            env=environment,
            capture_output=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, "fresh known-run consumer rejected"
        return json.loads(result.stdout)

    # No parent database open after SIGKILL: this fresh lifecycle reader is first.
    pending = consumer("cancel-pending")
    assert pending["census"] == ready["census"]
    assert pending["before"] == "queued"
    canceled = consumer("cancel-lost" if lost else "cancel-ack")
    assert canceled["cancel_disposition"] == ("indeterminate" if lost else "found")
    assert canceled["bound_cancellation"] is (not lost)
    reconciled = consumer("cancel-reconcile")
    assert reconciled["cancel_command_digest"] == canceled["cancel_command_digest"]
    assert (
        reconciled["submit_assessment_sha256"] == canceled["submit_assessment_sha256"]
    )
    if lost:
        repeated = consumer("cancel-repeat")
        assert "cancel" not in repeated["events"]
        assert repeated["cancel_disposition"] == "indeterminate"
    assert (root / "cancel-count").read_bytes() == b"cancel\n"
    state = json.loads((root / "provider.json").read_bytes())
    state["status"] = "CANCELED" if lost else "COMPLETED"
    (root / "provider.json").write_text(json.dumps(state))
    terminal = consumer("cancel-terminal")
    assert terminal["phase"] == ("cancelled" if lost else "verified")
    assert terminal["cancel_command_digest"] == canceled["cancel_command_digest"]
    assert terminal["cancel_disposition"] == canceled["cancel_disposition"]
    assert terminal["bound_cancellation"] is (not lost)
    assert terminal["submit_assessment_sha256"] == canceled["submit_assessment_sha256"]
    assert (root / "cancel-count").read_bytes() == b"cancel\n"
    assert (
        len(
            {
                ready["pid"],
                pending["pid"],
                canceled["pid"],
                reconciled["pid"],
                terminal["pid"],
            }
        )
        == 5
    )
    assert _census(root) == terminal["census"]
