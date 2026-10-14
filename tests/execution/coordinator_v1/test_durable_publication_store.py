"""Real local ownership/store processes; destination is a signed test fixture."""

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_publication as fixture
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference.publication_store import DurablePublicationStoreV1
from tuner.execution.coordinator_v1.publication import (
    AuthenticatedPublicationReceiptV1,
    PublicationRecordV1,
    PublicationPhaseV1,
    PublicationErrorV1,
    TransferDispositionV1,
)
from synaptic_tuner.api.v1.artifacts_facade import PublicationRequest
import synaptic_tuner.api.v1.reference.publication_store as production

STAMP = "2026-08-27T12:00:00Z"
_SOURCE = Path(production.__file__).resolve().parents[4]
_TEST_DIRECTORY = Path(__file__).resolve().parent


def command():
    service, _, registry, sources, _, _ = fixture._stack()
    descriptor, _ = registry.resolve("opaque/local-like")
    return service._command(fixture.RUN, descriptor, sources.describe(fixture.RUN))


@pytest.fixture
def roots(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    locks = private / "locks"
    locks.mkdir(mode=0o700)
    return private, locks


def test_real_sqlite_publication_verified_reopen(roots):
    private, locks = roots
    with SQLiteDurableStoreV1(private / "db") as records:
        with DurablePublicationStoreV1(records, locks) as store:
            service, _, _, _, _, adapter = fixture._stack(store=store)
            result = service.publish(
                PublicationRequest(fixture.RUN, "opaque/local-like")
            )
            assert result.state.value == "verified" and adapter.publish_calls == 1
    with SQLiteDurableStoreV1(private / "db", create=False) as records:
        with DurablePublicationStoreV1(records, locks, create=False) as store:
            service, _, _, sources, _, adapter = fixture._stack(store=store)
            assert service.publications("opaque/local-like").publications[0] == result
            assert service.verify(result.publication).verified
            assert adapter.publish_calls == adapter.lookup_calls == sources.opens == 0


def test_claim_cas_codec_and_wrong_lock_root(roots):
    private, locks = roots
    initial = PublicationRecordV1.claim(command(), STAMP)
    with SQLiteDurableStoreV1(private / "db") as records:
        with DurablePublicationStoreV1(records, locks) as store:
            assert store.claim(initial) == (initial, True)
            assert store.claim(initial) == (initial, False)
            failed = initial.transition(PublicationPhaseV1.FAILED_BEFORE_EFFECT, STAMP)
            assert not store.compare_and_swap("f" * 64, failed)
            assert store.get(initial.command.publication_id) == initial
            assert store.compare_and_swap(initial.record_digest, failed)
            assert store.get(initial.command.publication_id) == failed
            assert (
                PublicationRecordV1.from_canonical_bytes(failed.canonical_bytes)
                == failed
            )
        alternate = private / "other"
        alternate.mkdir(mode=0o700)
        with pytest.raises(PublicationErrorV1):
            DurablePublicationStoreV1(records, alternate)
        before = records.read(partition="effects", key="publication-lock-root/v1")
        assert before is not None


def test_existing_only_missing_binding_has_no_setup_mutation(roots):
    private, locks = roots
    with SQLiteDurableStoreV1(private / "db") as records:
        with pytest.raises(PublicationErrorV1):
            DurablePublicationStoreV1(records, locks, create=False)
        assert not records.list_page(
            partition="effects", prefix="publication", after_key=None, limit=100
        ).records
        assert not list(locks.iterdir())


def test_lock_replacement_refuses_without_fencing_live_owner(roots):
    private, locks = roots
    initial = PublicationRecordV1.claim(command(), STAMP)
    with SQLiteDurableStoreV1(private / "db") as records:
        store = DurablePublicationStoreV1(records, locks)
        store.claim(initial)
        acquired = store.begin_transfer(
            initial.command.publication_id, initial.record_digest, STAMP
        )
        path = locks / initial.command.publication_id
        path.rename(locks / "original")
        path.touch(mode=0o600)
        with pytest.raises(PublicationErrorV1):
            store.recover_transfer(
                initial.command.publication_id, initial.command.command_digest, STAMP
            )
        with pytest.raises(PublicationErrorV1):
            store.close()
        assert store._closed and not store._owned
        assert acquired.disposition is TransferDispositionV1.ACQUIRED


def test_thread_affinity_leaves_owner_and_records_intact(roots):
    private, locks = roots
    with SQLiteDurableStoreV1(private / "db") as records:
        with DurablePublicationStoreV1(records, locks) as store:
            initial = PublicationRecordV1.claim(command(), STAMP)
            store.claim(initial)
            errors = []

            def foreign():
                try:
                    store.get(initial.command.publication_id)
                except Exception as error:
                    errors.append(type(error))

            thread = threading.Thread(target=foreign)
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive() and errors == [PublicationErrorV1]
            assert store.get(initial.command.publication_id) == initial


def test_record_codec_cap_is_enforced_before_generic_store_write(roots, monkeypatch):
    import tuner.execution.coordinator_v1.publication as module

    private, locks = roots
    with SQLiteDurableStoreV1(private / "db") as records:
        with DurablePublicationStoreV1(records, locks) as store:
            initial = PublicationRecordV1.claim(command(), STAMP)
            monkeypatch.setattr(module, "_MAX_PUBLICATION_RECORD_BYTES", 8)
            with pytest.raises(ValueError):
                store.claim(initial)
            assert (
                records.read(
                    partition="effects",
                    key="publication/v1/" + initial.command.publication_id,
                )
                is None
            )


def _child(role, private):
    locks = private / "locks"
    with SQLiteDurableStoreV1(private / "db", create=role == "producer") as records:
        with DurablePublicationStoreV1(
            records, locks, create=role == "producer"
        ) as store:
            service, _, _, sources, _, adapter = fixture._stack(store=store)
            if role == "producer":
                original = adapter.publish_once

                def publish(*args):
                    receipt = original(*args)
                    with (private / "receipt.json").open("x") as output:
                        json.dump(receipt.to_dict(), output)
                        output.flush()
                        os.fsync(output.fileno())
                    print(json.dumps({"ready": True, "pid": os.getpid()}), flush=True)
                    signal.alarm(30)
                    signal.pause()
                    raise AssertionError("must be killed")

                adapter.publish_once = publish
                service.publish(PublicationRequest(fixture.RUN, "opaque/local-like"))
            elif role == "active":
                result = service.publish(
                    PublicationRequest(fixture.RUN, "opaque/local-like")
                )
                assert result.state.value == "transferring"
                assert (
                    adapter.publish_calls == adapter.lookup_calls == sources.opens == 0
                )
                print(json.dumps({"active": True, "pid": os.getpid()}), flush=True)
            else:
                adapter.receipt = AuthenticatedPublicationReceiptV1.from_dict(
                    json.loads((private / "receipt.json").read_bytes())
                )
                adapter.published = {"adapter": b"adapter"}
                result = service.publish(
                    PublicationRequest(fixture.RUN, "opaque/local-like")
                )
                assert result.state.value == "verified"
                assert (
                    adapter.publish_calls == sources.opens == 0
                    and adapter.lookup_calls == 1
                )
                record = store.get(result.publication.publication_id)
                assert (
                    len(record.ownership_history) == len(record.recovery_permits) == 1
                )
                print(json.dumps({"verified": True, "pid": os.getpid()}), flush=True)


def child_command(role, private):
    return [
        sys.executable,
        "-I",
        "-B",
        "-c",
        "import sys; from pathlib import Path; sys.path[:0]=sys.argv[1:3]; from test_durable_publication_store import _child; _child(sys.argv[3],Path(sys.argv[4]))",
        str(_TEST_DIRECTORY),
        str(_SOURCE),
        role,
        str(private),
    ]


def test_real_active_owner_then_sigkill_first_fresh_reader_lookup_only(roots):
    private, _ = roots
    process = subprocess.Popen(
        child_command("producer", private),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=20)
            ready = json.loads(os.read(process.stdout.fileno(), 4096))
        assert ready["pid"] == process.pid
        active = subprocess.run(
            child_command("active", private), capture_output=True, timeout=15
        )
        assert active.returncode == 0, active.stderr.decode()
        process.kill()
        assert process.wait(timeout=5) == -signal.SIGKILL
        # Parent does not open SQLite after kill: first recovery opener is fresh child.
        recovered = subprocess.run(
            child_command("recovered", private), capture_output=True, timeout=15
        )
        assert recovered.returncode == 0, recovered.stderr.decode()
        assert (
            len(
                {
                    ready["pid"],
                    json.loads(active.stdout)["pid"],
                    json.loads(recovered.stdout)["pid"],
                }
            )
            == 3
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def test_second_opener_refuses_replaced_lock_inode(roots):
    private, locks = roots
    initial = PublicationRecordV1.claim(command(), STAMP)
    with SQLiteDurableStoreV1(private / "db") as records:
        owner = DurablePublicationStoreV1(records, locks)
        owner.claim(initial)
        active = owner.begin_transfer(
            initial.command.publication_id, initial.record_digest, STAMP
        )
        path = locks / initial.command.publication_id
        path.rename(locks / "retained-original")
        path.touch(mode=0o600)
        with DurablePublicationStoreV1(records, locks, create=False) as other:
            with pytest.raises(PublicationErrorV1):
                other.recover_transfer(
                    initial.command.publication_id,
                    initial.command.command_digest,
                    STAMP,
                )
        assert (
            records.read(
                partition="effects",
                key="publication/v1/" + initial.command.publication_id,
            ).canonical
            == active.record.canonical_bytes
        )
        with pytest.raises(PublicationErrorV1):
            owner.close()


def test_existing_publication_missing_lock_binding_refuses_even_create_mode(roots):
    private, locks = roots
    initial = PublicationRecordV1.claim(command(), STAMP)
    with SQLiteDurableStoreV1(private / "db") as records:
        with DurablePublicationStoreV1(records, locks) as store:
            store.claim(initial)
        records._connection.execute(
            "DELETE FROM records WHERE key=?",
            ("publication-lock/v1/" + initial.command.publication_id,),
        )
        with DurablePublicationStoreV1(records, locks) as other:
            with pytest.raises(PublicationErrorV1):
                other.begin_transfer(
                    initial.command.publication_id, initial.record_digest, STAMP
                )
            assert other.get(initial.command.publication_id) == initial
        assert (
            records.read(
                partition="effects",
                key="publication-lock/v1/" + initial.command.publication_id,
            )
            is None
        )


def test_cold_orphan_becomes_lookup_only_without_new_transfer(roots):
    private, locks = roots
    initial = PublicationRecordV1.claim(command(), STAMP)
    with SQLiteDurableStoreV1(private / "db") as records:
        with DurablePublicationStoreV1(records, locks) as owner:
            owner.claim(initial)
            admission = owner.begin_transfer(
                initial.command.publication_id, initial.record_digest, STAMP
            )
        with DurablePublicationStoreV1(records, locks, create=False) as cold:
            decision = cold.recover_transfer(
                initial.command.publication_id, initial.command.command_digest, STAMP
            )
            assert (
                decision.record.phase is PublicationPhaseV1.AMBIGUOUS
                and decision.permit is not None
            )
            second = cold.begin_transfer(
                initial.command.publication_id, decision.record.record_digest, STAMP
            )
            assert second.disposition is TransferDispositionV1.TERMINAL
            assert cold.get(initial.command.publication_id).ownership_history == (
                admission.ownership,
            )


@pytest.mark.parametrize(
    "mutation", ["root_mode", "leaf_mode", "leaf_symlink", "ancestor"]
)
def test_private_custody_mutations_refuse(roots, mutation):
    private, locks = roots
    initial = PublicationRecordV1.claim(command(), STAMP)
    with SQLiteDurableStoreV1(private / "db") as records:
        store = DurablePublicationStoreV1(records, locks)
        store.claim(initial)
        store.begin_transfer(
            initial.command.publication_id, initial.record_digest, STAMP
        )
        leaf = locks / initial.command.publication_id
        if mutation == "root_mode":
            locks.chmod(0o755)
        elif mutation == "leaf_mode":
            leaf.chmod(0o644)
        elif mutation == "leaf_symlink":
            leaf.rename(locks / "original")
            leaf.symlink_to("original")
        else:
            locks.rename(private / "original-root")
            locks.mkdir(mode=0o700)
        with pytest.raises(PublicationErrorV1):
            store.get(initial.command.publication_id)
        with pytest.raises(PublicationErrorV1):
            store.close()
        assert store._closed and not store._anchors and not store._owned


def test_wrong_thread_close_keeps_real_competing_opener_active(roots):
    private, locks = roots
    initial = PublicationRecordV1.claim(command(), STAMP)
    with SQLiteDurableStoreV1(private / "db") as records:
        with DurablePublicationStoreV1(records, locks) as owner:
            owner.claim(initial)
            admission = owner.begin_transfer(
                initial.command.publication_id, initial.record_digest, STAMP
            )
            failures = []

            def foreign_close():
                try:
                    owner.close()
                except Exception as error:
                    failures.append(type(error))

            thread = threading.Thread(target=foreign_close)
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive() and failures == [PublicationErrorV1]
            assert not owner._closed and owner._owned
            active = subprocess.run(
                child_command("active", private), capture_output=True, timeout=15
            )
            assert active.returncode == 0, active.stderr.decode()
            retained = owner.get(initial.command.publication_id)
            assert retained == admission.record and not retained.recovery_permits
            assert len(retained.ownership_history) == 1
