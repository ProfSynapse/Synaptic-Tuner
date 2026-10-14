"""Actual public native verification/local publication with scripted Docker/ML.

Cold objects reopen real SQLite; this is not live Docker or model qualification.
"""

import time

import pytest

from tests.training.test_docker_host_composition import host_fixture as host_fixture
from tests.training.test_docker_host_composition import (
    start,
    native_values,
)  # noqa: F401
from tests.execution.providers.docker_provider_v1.conftest import profile  # noqa: F401
from tests.execution.providers.docker_provider_v1.test_local_source import material  # noqa: F401
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference.publication_store import DurablePublicationStoreV1
from synaptic_tuner.api.v1.reference.local_publication import (
    LocalArtifactSpoolV1,
    LocalPublicationDestinationV1,
    LocalDestinationRegistryV1,
)
from synaptic_tuner.api.v1.reference.publication_sources import (
    RunsArtifactSourceV1,
    compose_publication_artifacts,
)
from synaptic_tuner.api.v1.artifacts_facade import PublicationRequest
from synaptic_tuner.api.v1.runs_facade import RunsAPI, RunArtifactRequest
from tuner.execution.coordinator_v1.publication import (
    PublicationErrorV1,
    PublicationCodeV1,
)

REF = "local/native"


def local_ports(root, records, host, *, create=True, spool=True):
    if create:
        for name in ("locks", "spool", "destination"):
            (root / name).mkdir(mode=0o700)
    authority = host.composition.authority.publication_authority
    deadline = time.monotonic() + 60
    store = DurablePublicationStoreV1(records, root / "locks", create=create)
    staging = (
        LocalArtifactSpoolV1(
            root / "spool",
            maximum_artifact_bytes=4 << 20,
            maximum_total_bytes=8 << 20,
            deadline=deadline,
        )
        if spool
        else None
    )
    destination = LocalPublicationDestinationV1(
        root / "destination",
        destination_ref=REF,
        display_name="Original native files",
        role_paths={
            role: "native/" + role + ".bin"
            for role in (
                "final_model",
                "tokenizer",
                "training_lineage",
                "training_metrics",
                "workload_record",
            )
        },
        maximum_artifact_bytes=4 << 20,
        maximum_total_bytes=8 << 20,
        authority_ref="host-publication",
        authority=authority,
        clock=host.composition.authority.clock.now,
        deadline=deadline,
        spool=staging,
    )
    api = compose_publication_artifacts(
        runs=host.api.runs,
        store=store,
        spool=staging,
        destinations=LocalDestinationRegistryV1((destination,)),
        authority_ref="host-publication",
        authority=authority,
        clock=host.composition.authority.clock.now,
    )
    return api, store, staging, destination


def close(ports):
    _, store, spool, destination = ports
    destination.close()
    if spool is not None:
        spool.close()
    store.close()


def complete(host, kwargs, state, roles, monkeypatch):
    _, _, _, run = start(host, kwargs)
    state["status"] = "exited"
    (kwargs["artifact_root"] / "native").mkdir(mode=0o700)
    values = native_values(kwargs, monkeypatch)
    for role, relative in roles.items():
        (kwargs["artifact_root"] / relative).write_bytes(values[role])
    assert host.api.runs.outcome(run).state.value == "succeeded"
    assert host.api.runs.verify(run).verified
    assert len(host.api.runs.show(run).artifacts) == 5
    return run, values


def test_native_five_publication_and_existing_only_cold_lookup(
    host_fixture, monkeypatch
):
    compose, kwargs, commands, state, roles, grants, root = host_fixture
    database = root / "workflow.sqlite3"
    with SQLiteDurableStoreV1(database) as records, compose(records) as host:
        run, values = complete(host, kwargs, state, roles, monkeypatch)
        ports = local_ports(root, records, host)
        try:
            api, store, spool, destination = ports
            original_source = RunsArtifactSourceV1(
                host.api.runs,
                authority_ref="host-publication",
                authority=host.composition.authority.publication_authority,
            ).describe(run)
            result = api.publish(PublicationRequest(run, REF))
            assert result.state.value == "verified" and len(result.artifacts) == 5
            record = store.get(result.publication.publication_id)
            for artifact in record.receipt.inventory.inventory.artifacts:
                assert (
                    b"".join(destination.iter_bytes(record.command, artifact, 4 << 20))
                    == values[artifact.role]
                )
            assert api.verify(result.publication).verified
            assert not spool.root.closed and not destination.root.closed
            assert (
                records.read(partition="effects", key="publication-lock-root/v1")
                is not None
            )
            before = len(commands)
        finally:
            close(ports)

    def no_effect(*args, **kwargs):
        pytest.fail(
            "cold publication reopened provider/native source or bound authority"
        )

    kwargs["cli_configuration"]["runner"] = no_effect
    monkeypatch.setattr(grants, "bind", no_effect)
    with (
        SQLiteDurableStoreV1(database, create=False) as records,
        compose(records, cold=True) as cold,
    ):
        ports = local_ports(root, records, cold, create=False, spool=False)
        try:
            source = RunsArtifactSourceV1(
                cold.api.runs,
                authority_ref="host-publication",
                authority=cold.composition.authority.publication_authority,
            )
            descriptor = source.describe(run)
            assert descriptor == original_source
            assert descriptor.artifacts == result.artifacts
            monkeypatch.setattr(RunsAPI, "artifacts", no_effect)
            assert ports[0].publish(PublicationRequest(run, REF)) == result
            assert ports[0].verify(result.publication).verified
            assert ports[0].publications(REF).publications == (result,)
            assert len(commands) == before
        finally:
            close(ports)


def test_unverified_public_run_refuses_before_spool_or_destination(host_fixture):
    compose, kwargs, _, _, _, _, root = host_fixture
    with SQLiteDurableStoreV1(root / "db") as records, compose(records) as host:
        _, _, _, run = start(host, kwargs)
        ports = local_ports(root, records, host)
        try:
            with pytest.raises(PublicationErrorV1) as error:
                ports[0].publish(PublicationRequest(run, REF))
            assert error.value.code is PublicationCodeV1.SOURCE_INVALID
            assert not list((root / "spool").iterdir())
            assert not list((root / "destination").iterdir())
        finally:
            close(ports)


def test_open_rejects_current_manifest_mismatch(host_fixture, monkeypatch):
    from dataclasses import replace

    compose, kwargs, _, state, roles, _, root = host_fixture
    with SQLiteDurableStoreV1(root / "db") as records, compose(records) as host:
        run, _ = complete(host, kwargs, state, roles, monkeypatch)
        source = RunsArtifactSourceV1(
            host.api.runs,
            authority_ref="host-publication",
            authority=host.composition.authority.publication_authority,
        )
        original = RunsAPI.artifacts

        def wrong(self, request):
            stream = original(self, request)

            class Changed:
                run = stream.run
                maximum_bytes = stream.maximum_bytes
                artifact = replace(stream.artifact, sha256="0" * 64)

                def iter_bytes(self):
                    pytest.fail("mismatched stream consumed")

            return Changed()

        monkeypatch.setattr(RunsAPI, "artifacts", wrong)
        with pytest.raises(ValueError):
            source.open(RunArtifactRequest(run, "final_model", 4 << 20))


def test_changed_native_file_refuses_publication_without_destination_write(
    host_fixture, monkeypatch
):
    compose, kwargs, _, state, roles, _, root = host_fixture
    with SQLiteDurableStoreV1(root / "db") as records, compose(records) as host:
        run, _ = complete(host, kwargs, state, roles, monkeypatch)
        (kwargs["artifact_root"] / roles["final_model"]).write_bytes(
            b"changed after native verification"
        )
        ports = local_ports(root, records, host)
        try:
            with pytest.raises(PublicationErrorV1) as error:
                ports[0].publish(PublicationRequest(run, REF))
            assert error.value.code is PublicationCodeV1.SOURCE_CONTENT_INVALID
            assert not list((root / "destination").iterdir())
        finally:
            close(ports)


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_native_join_sigkill_first_new_process_lookup_only(
    host_fixture, monkeypatch, boundary
):
    """Forked processes retain scripted provider inputs, never a warm SQLite FD."""
    import multiprocessing
    import os
    import signal

    compose, kwargs, _, state, roles, grants, root = host_fixture
    database = root / "workflow.sqlite3"
    with SQLiteDurableStoreV1(database) as records, compose(records) as host:
        run, _ = complete(host, kwargs, state, roles, monkeypatch)
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def produce():
        receive.close()
        with (
            SQLiteDurableStoreV1(database, create=False) as records,
            compose(records, cold=True) as host,
        ):
            ports = local_ports(root, records, host)
            try:
                destination = ports[3]

                def pause():
                    send.send("marker-boundary")
                    send.close()
                    signal.alarm(30)
                    signal.pause()
                    raise AssertionError("producer must be killed")

                if boundary == "before":
                    original = destination._receipt

                    def receipt(*args):
                        pause()
                        return original(*args)

                    destination._receipt = receipt
                else:
                    original = destination.publish_once

                    def publish(*args):
                        value = original(*args)
                        pause()
                        return value

                    destination.publish_once = publish
                ports[0].publish(PublicationRequest(run, REF))
            finally:
                close(ports)

    producer = context.Process(target=produce)
    producer.start()
    send.close()
    try:
        assert receive.poll(15), "producer did not reach original marker boundary"
        assert receive.recv() == "marker-boundary"
        os.kill(producer.pid, signal.SIGKILL)
        producer.join(timeout=5)
        assert producer.exitcode == -signal.SIGKILL
        before = {
            str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in (root / "destination").rglob("*")
        }
        replies, report = context.Pipe(duplex=False)

        def recover():
            replies.close()

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "cold recovery reopened native source/provider/effect"
                )

            kwargs["cli_configuration"]["runner"] = forbidden
            grants.bind = forbidden
            RunsAPI.artifacts = forbidden
            # FIRST SQLite opener after kill, in a different process.
            with (
                SQLiteDurableStoreV1(database, create=False) as records,
                compose(records, cold=True) as host,
            ):
                ports = local_ports(root, records, host, create=False, spool=False)
                try:
                    value = ports[0].publish(PublicationRequest(run, REF))
                    record = ports[1].get(value.publication.publication_id)
                    assert (
                        len(record.ownership_history)
                        == len(record.recovery_permits)
                        == 1
                    )
                    if boundary == "after":
                        assert ports[0].verify(value.publication).verified
                        assert len(value.artifacts) == 5
                    assert {
                        str(p.relative_to(root)): (
                            p.stat().st_size,
                            p.stat().st_mtime_ns,
                        )
                        for p in (root / "destination").rglob("*")
                    } == before
                    report.send(value.state.value)
                finally:
                    close(ports)
            report.close()

        consumer = context.Process(target=recover)
        consumer.start()
        report.close()
        try:
            assert replies.poll(15), "new owner did not finish lookup"
            assert replies.recv() == (
                "verified" if boundary == "after" else "ambiguous"
            )
            consumer.join(timeout=5)
            assert consumer.exitcode == 0
        finally:
            if consumer.is_alive():
                consumer.kill()
            consumer.join(timeout=5)
            replies.close()
    finally:
        if producer.is_alive():
            producer.kill()
        producer.join(timeout=5)
        receive.close()
