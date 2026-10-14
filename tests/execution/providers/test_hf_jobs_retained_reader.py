"""Source-free original-command reads; provider transport is an explicit fixture."""

import pytest

sdk = pytest.importorskip("huggingface_hub")
from tests.execution.providers.test_hf_jobs_public_family import (
    material as material,
    _native_values,
    _complete,
)
from tests.execution.providers.test_hf_jobs_public_lifecycle import (
    host,
    start,
    enabled_material as enabled_material,
    cancellation,
)
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference import compose_reference_authority
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunOperationError
from tuner.execution.providers.hf_jobs_provider_v1.retained_reader import (
    open_retained_hf_runs,
)


def test_cold_known_job_reads_without_original_preparation(material, tmp_path):
    value = material
    private = tmp_path / "private-db"
    private.mkdir(mode=0o700)
    path = private / "workflow.sqlite3"
    with SQLiteDurableStoreV1(path) as store:
        api, composed, ports = host(value, records=store, streams=store)
        run, _, _ = start(api, value)
        expected = _native_values(value.preparation)
        _complete(value, expected)
    # The read path only gets the existing stores/keys and the external provider
    # boundary; no preparation object is passed to it.
    authority = compose_reference_authority(
        clock=ports.clock,
        secrets=ports.secrets,
        authority_secret=SecretRef("env", "AUTHORITY"),
    )
    scratch = tmp_path / "cold-scratch"
    scratch.mkdir(mode=0o700)
    with SQLiteDurableStoreV1(path, create=False) as store:
        view = open_retained_hf_runs(
            run=run,
            records=store,
            streams=store,
            authority=authority,
            secrets=ports.secrets,
            sdk=sdk,
            api=value.provider,
            scratch_root=scratch,
            reader_authority_ref="fixture-reader",
            reader_key_ref="fixture-reader-key",
            reader_key=b"r" * 32,
        )
        assert view.runs.outcome(run).state.value == "succeeded"
        assert view.runs.verify(run).verified
        for role, (_, raw) in expected.items():
            assert (
                b"".join(
                    view.runs.artifacts(
                        RunArtifactRequest(run, role, len(raw))
                    ).iter_bytes()
                )
                == raw
            )
        assert len([item for item in value.provider.calls if item[0] == "submit"]) == 1


@pytest.mark.parametrize(
    "capture_error",
    ["missing", "altered", "descriptor_changed", "wrong_key", "missing_submit_receipt"],
)
def test_invalid_retention_rejects_before_provider(material, tmp_path, capture_error):
    from tuner.execution.providers.hf_jobs_provider_v1.model import HFJobsError

    value = material
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "workflow.sqlite3"
    with SQLiteDurableStoreV1(path) as store:
        api, _, ports = host(value, records=store, streams=store)
        run, _, _ = start(api, value)
        key = (
            "hf-jobs/"
            + value.preparation.material_digest
            + "/read-material/"
            + value.preparation.material_digest
        )
        if capture_error == "missing":
            store._connection.execute(
                "DELETE FROM records WHERE partition='effects' AND key=?", (key,)
            )
            store._connection.commit()
        elif capture_error == "altered":
            store._connection.execute(
                "UPDATE records SET canonical=? WHERE partition='effects' AND key=?",
                (b"{}", key),
            )
            store._connection.commit()
        elif capture_error == "descriptor_changed":
            import json
            from tuner.execution.providers.hf_jobs_provider_v1.model import (
                document_bytes,
            )

            previous = store.read(partition="effects", key=key)
            document = json.loads(previous.canonical)
            document["original_descriptor"]["capabilities"]["cancel"] = True
            assert store.compare_and_swap(
                partition="effects",
                key=key,
                expected_revision=previous.revision,
                canonical=document_bytes(document),
            )
        elif capture_error == "missing_submit_receipt":
            store._connection.execute(
                "DELETE FROM records WHERE partition='effects' AND key LIKE '%/submit/%'"
            )
            store._connection.commit()
    clock = ports.clock
    secrets = ports.secrets
    if capture_error == "wrong_key":

        class WrongSecrets:
            def resolve(self, ref):
                return "wrong-authority-" + "z" * 40

        secrets = WrongSecrets()
    authority = compose_reference_authority(
        clock=clock, secrets=secrets, authority_secret=SecretRef("env", "AUTHORITY")
    )
    scratch = tmp_path / "reader-scratch"
    scratch.mkdir(mode=0o700)
    before = list(value.provider.calls)
    with SQLiteDurableStoreV1(path, create=False) as store:
        with pytest.raises(HFJobsError, match="read_unavailable"):
            open_retained_hf_runs(
                run=run,
                records=store,
                streams=store,
                authority=authority,
                secrets=secrets,
                sdk=sdk,
                api=value.provider,
                scratch_root=scratch,
                reader_authority_ref="fixture-reader",
                reader_key_ref="fixture-reader-key",
                reader_key=b"r" * 32,
            )
    assert value.provider.calls == before


def test_capture_is_exact_create_or_identical_warm_read(material):
    from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1
    from tuner.execution.providers.hf_jobs_provider_v1.retained_reader import _capture
    from tuner.execution.providers.hf_jobs_provider_v1.model import HFJobsError

    records = InMemoryDurableRecordStoreV1()
    first = _capture(material.preparation, records)
    second = _capture(material.preparation, records)
    assert first["workload"].canonical_bytes == second["workload"].canonical_bytes
    key = (
        "hf-jobs/"
        + first["material_digest"]
        + "/read-material/"
        + first["material_digest"]
    )
    old = records.read(partition="effects", key=key)
    assert records.compare_and_swap(
        partition="effects", key=key, expected_revision=old.revision, canonical=b"{}"
    )
    with pytest.raises(HFJobsError, match="binding_mismatch"):
        _capture(material.preparation, records)
    assert records.read(partition="effects", key=key).canonical == b"{}"


def test_original_enabled_descriptor_stays_read_only_on_cold_cancel_unknown(
    enabled_material, tmp_path
):

    value = enabled_material
    cancellation(value, lost=True)
    private = tmp_path / "original-enabled"
    private.mkdir(mode=0o700)
    path = private / "workflow.sqlite3"
    with SQLiteDurableStoreV1(path) as store:
        api, _, ports = host(value, records=store, streams=store, current=True)
        run, _, _ = start(api, value)
        assert api.runs.cancel(run, "stop").state.value == "reconcile_required"
    scratch = tmp_path / "readonly"
    scratch.mkdir(mode=0o700)
    authority = compose_reference_authority(
        clock=ports.clock,
        secrets=ports.secrets,
        authority_secret=SecretRef("env", "AUTHORITY"),
    )
    before = list(value.provider.calls)
    with SQLiteDurableStoreV1(path, create=False) as store:
        view = open_retained_hf_runs(
            run=run,
            records=store,
            streams=store,
            authority=authority,
            secrets=ports.secrets,
            sdk=sdk,
            api=value.provider,
            scratch_root=scratch,
            reader_authority_ref="fixture-reader",
            reader_key_ref="fixture-reader-key",
            reader_key=b"r" * 32,
        )
        assert view.runs.outcome(run).state.value == "reconcile_required"
        with pytest.raises(RunOperationError, match="capability_unavailable"):
            view.runs.cancel(run, "not authorized here")
        with pytest.raises(RunOperationError, match="capability_unavailable"):
            view.runs.reconcile(run)
        value.provider.job.status.stage = "CANCELED"
        assert view.runs.outcome(run).state.value == "cancelled"
    assert [
        row for row in value.provider.calls if row[0] in {"cancel", "submit", "upload"}
    ] == [row for row in before if row[0] in {"cancel", "submit", "upload"}]
    assert len([row for row in value.provider.calls if row[0] == "cancel"]) == 1
    assert len([row for row in value.provider.calls if row[0] == "submit"]) == 1


def test_actual_lost_submission_never_adopts_provider_job(material, tmp_path):
    from tuner.execution.providers.hf_jobs_provider_v1.model import HFJobsError

    original = material.provider.run_job

    def lost(**kwargs):
        original(**kwargs)
        raise TimeoutError("controlled lost submission reply")

    material.provider.run_job = lost
    private = tmp_path / "unknown"
    private.mkdir(mode=0o700)
    with SQLiteDurableStoreV1(private / "workflow.sqlite3") as store:
        api, _, ports = host(material, records=store, streams=store)
        run, _, _ = start(api, material)
        assert api.runs.show(run).state.value == "reconcile_required"
    authority = compose_reference_authority(
        clock=ports.clock,
        secrets=ports.secrets,
        authority_secret=SecretRef("env", "AUTHORITY"),
    )
    scratch = tmp_path / "cold-unknown"
    scratch.mkdir(mode=0o700)
    before = list(material.provider.calls)
    with SQLiteDurableStoreV1(private / "workflow.sqlite3", create=False) as store:
        with pytest.raises(HFJobsError, match="read_unavailable"):
            open_retained_hf_runs(
                run=run,
                records=store,
                streams=store,
                authority=authority,
                secrets=ports.secrets,
                sdk=sdk,
                api=material.provider,
                scratch_root=scratch,
                reader_authority_ref="fixture-reader",
                reader_key_ref="fixture-reader-key",
                reader_key=b"r" * 32,
            )
    assert material.provider.calls == before
    assert material.provider.job is not None


@pytest.mark.parametrize(
    "fault",
    [
        "missing_descriptor",
        "descriptor_extra",
        "bad_flavor",
        "bool_timeout",
        "metadata_bool",
        "duplicate_key",
    ],
)
def test_capture_parser_closed_metadata_controls(material, fault):
    import json
    import base64
    from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1
    from tuner.execution.providers.hf_jobs_provider_v1.retained_reader import (
        _capture,
        _parse_capture,
    )
    from tuner.execution.providers.hf_jobs_provider_v1.model import (
        HFJobsError,
        document_bytes,
    )

    records = InMemoryDurableRecordStoreV1()
    _capture(material.preparation, records)
    digest = material.preparation.material_digest
    raw = records.read(
        partition="effects", key="hf-jobs/" + digest + "/read-material/" + digest
    ).canonical
    envelope = json.loads(raw)
    if fault == "missing_descriptor":
        del envelope["original_descriptor"]
    elif fault == "descriptor_extra":
        envelope["original_descriptor"]["unexpected"] = "not admitted"
    elif fault == "duplicate_key":
        raw = raw.replace(
            b'"schema_version":"synaptic-hf-read-material/v1"',
            b'"schema_version":"synaptic-hf-read-material/v1","schema_version":"synaptic-hf-read-material/v1"',
        )
    else:
        value = json.loads(base64.b64decode(envelope["material_base64"]))
        if fault == "bad_flavor":
            value["selection"]["flavor"] = 42
        elif fault == "bool_timeout":
            value["selection"]["timeout_seconds"] = True
        else:
            value["members"][0]["size_bytes"] = False
        envelope["material_base64"] = base64.b64encode(document_bytes(value)).decode()
    if fault != "duplicate_key":
        raw = document_bytes(envelope)
    with pytest.raises(HFJobsError, match="binding_mismatch"):
        _parse_capture(raw)


def test_base64_bound_is_checked_before_decoding(monkeypatch):
    from types import SimpleNamespace
    from tuner.execution.providers.hf_jobs_provider_v1 import retained_reader
    from tuner.execution.providers.hf_jobs_provider_v1.model import HFJobsError

    def forbidden(*args, **kwargs):
        pytest.fail("oversized base64 must reject before decoding")

    monkeypatch.setattr(retained_reader, "base64", SimpleNamespace(b64decode=forbidden))
    with pytest.raises(HFJobsError, match="binding_mismatch"):
        retained_reader._decode("A" * 16, 8)
