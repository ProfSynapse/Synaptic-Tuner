"""Real local publication/SQLite; signed source is a test-owned byte fixture."""

from dataclasses import replace
import os
from pathlib import Path
import time
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_publication as fixture
from synaptic_tuner.api.v1.reference.authority import (
    ReferencePublicationEvidenceAuthorityV1,
)
from synaptic_tuner.api.v1.reference.local_publication import (
    LocalArtifactSpoolV1,
    LocalPublicationDestinationV1,
    LocalDestinationRegistryV1,
)
from synaptic_tuner.api.v1.reference.publication_store import DurablePublicationStoreV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.artifacts_facade import PublicationRequest
from tuner.execution.coordinator_v1.publication import (
    PublicationOperationsV1,
    PublicationErrorV1,
    PublicationCodeV1,
)

AUTH = ReferencePublicationEvidenceAuthorityV1("local-key", b"x" * 32)
STAMP = "2026-10-14T00:00:00Z"
REF = "local/target"


def stack(root, *, data=b"adapter", create=True, spool=True):
    deadline = time.monotonic() + 60
    if create:
        for name in ("locks", "spool", "destination"):
            (root / name).mkdir(mode=0o700)
    records = SQLiteDurableStoreV1(root / "db", create=create)
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
        display_name="Local target",
        role_paths={"adapter": "native/adapter.bin"},
        maximum_artifact_bytes=4 << 20,
        maximum_total_bytes=8 << 20,
        authority_ref="host-authority",
        authority=AUTH,
        clock=lambda: STAMP,
        deadline=deadline,
        spool=staging,
    )
    artifacts = [fixture._artifact(data=data)]
    source = fixture._source(artifacts)
    source = replace(source, key_ref=AUTH.key_ref, tag="0" * 64)
    source = replace(
        source, tag=AUTH.sign("publication-verified-source/v1", source.payload)
    )
    sources = fixture.Sources(
        artifacts,
        descriptor=source,
        stream_factory=lambda run, artifact, raw: fixture.Stream(
            run,
            artifact,
            raw,
            chunks=tuple(raw[i : i + (1 << 20)] for i in range(0, len(raw), 1 << 20)),
        ),
    )
    service = PublicationOperationsV1(
        store=store,
        destinations=LocalDestinationRegistryV1((destination,)),
        sources=sources,
        spool=staging,
        authority=AUTH,
        clock=lambda: STAMP,
    )
    return service, store, records, staging, destination, sources


def close(values):
    _, store, records, spool, destination, _ = values
    destination.close()
    if spool:
        spool.close()
    store.close()
    records.close()


@pytest.mark.parametrize(
    "data",
    [b"", b"adapter", b"x" * ((2 << 20) + 7)],
    ids=["empty", "small", "multi-chunk"],
)
def test_actual_publication_hash_chunks_and_cold_read(tmp_path, data):
    values = stack(tmp_path, data=data)
    try:
        service, store, _, _, destination, sources = values
        result = service.publish(PublicationRequest(fixture.RUN, REF))
        assert result.state.value == "verified" and sources.opens == 1
        record = store.get(result.publication.publication_id)
        artifact = record.receipt.inventory.inventory.artifacts[0]
        chunks = list(destination.iter_bytes(record.command, artifact, 4 << 20))
        assert b"".join(chunks) == data
        assert all(0 < len(chunk) <= 1 << 20 for chunk in chunks)
        before = {
            p: p.stat().st_mtime_ns for p in (tmp_path / "destination").rglob("*")
        }
    finally:
        close(values)
    cold = stack(tmp_path, data=data, create=False, spool=False)
    try:
        assert cold[0].verify(result.publication).verified
        assert cold[5].opens == 0
        assert {
            p: p.stat().st_mtime_ns for p in (tmp_path / "destination").rglob("*")
        } == before
    finally:
        close(cold)


def test_spool_total_bound_partial_preserved(tmp_path):
    spool = LocalArtifactSpoolV1(
        tmp_path,
        maximum_artifact_bytes=4,
        maximum_total_bytes=5,
        deadline=time.monotonic() + 10,
    )
    try:
        first = spool.open("a" * 64, "one", 4)
        first.write(b"1234")
        first.finish()
        second = spool.open("a" * 64, "two", 4)
        with pytest.raises(ValueError):
            second.write(b"12")
        second.abort()
        assert len(list((tmp_path / ("a" * 64)).iterdir())) == 2
    finally:
        spool.close()


@pytest.mark.parametrize(
    "mutation", ["chmod", "replace", "symlink", "hardlink", "extra"]
)
def test_original_files_tamper_refused(tmp_path, mutation):
    values = stack(tmp_path)
    try:
        service, store, _, _, destination, _ = values
        result = service.publish(PublicationRequest(fixture.RUN, REF))
        record = store.get(result.publication.publication_id)
        artifact = record.receipt.inventory.inventory.artifacts[0]
        base = tmp_path / "destination" / record.command.publication_id
        leaf = base / artifact.path
        if mutation == "chmod":
            leaf.chmod(0o644)
        elif mutation == "replace":
            leaf.unlink()
            leaf.write_bytes(b"adapter")
            leaf.chmod(0o600)
        elif mutation == "symlink":
            leaf.unlink()
            leaf.symlink_to(tmp_path / "db")
        elif mutation == "hardlink":
            os.link(leaf, base / "alias")
        else:
            (base / "extra").write_bytes(b"extra")
        # An independently rewritten same-byte file is still a valid cold file:
        # no descriptor identity commitment survives closing the publication.
        if mutation == "replace":
            assert (
                b"".join(destination.iter_bytes(record.command, artifact, 4 << 20))
                == b"adapter"
            )
        else:
            with pytest.raises((ValueError, OSError)):
                list(destination.iter_bytes(record.command, artifact, 4 << 20))
    finally:
        close(values)


def test_mutation_during_stream_refuses_eof(tmp_path):
    values = stack(tmp_path, data=b"x" * ((2 << 20) + 1))
    try:
        service, store, _, _, destination, _ = values
        result = service.publish(PublicationRequest(fixture.RUN, REF))
        record = store.get(result.publication.publication_id)
        artifact = record.receipt.inventory.inventory.artifacts[0]
        stream = destination.iter_bytes(record.command, artifact, 4 << 20)
        assert len(next(stream)) == 1 << 20
        path = tmp_path / "destination" / record.command.publication_id / artifact.path
        path.write_bytes(b"changed")
        with pytest.raises(ValueError):
            list(stream)
    finally:
        close(values)


def _process(role, root):
    import signal

    values = stack(
        root, create=role.startswith("producer"), spool=role.startswith("producer")
    )
    try:
        service, store, _, _, destination, sources = values
        if role.startswith("producer"):

            def paused():
                print("ready", flush=True)
                signal.alarm(30)
                signal.pause()
                raise AssertionError("producer must be killed")

            if role == "producer-before":
                original = destination._receipt

                def receipt(*args):
                    paused()
                    return original(*args)

                destination._receipt = receipt
            else:
                original = destination.publish_once

                def publish(*args):
                    result = original(*args)
                    paused()
                    return result

                destination.publish_once = publish
            service.publish(PublicationRequest(fixture.RUN, REF))
        else:
            before = {
                str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in (root / "destination").rglob("*")
            }
            result = service.publish(PublicationRequest(fixture.RUN, REF))
            record = store.get(result.publication.publication_id)
            assert sources.opens == 0
            assert len(record.ownership_history) == len(record.recovery_permits) == 1
            assert {
                str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in (root / "destination").rglob("*")
            } == before
            assert result.state.value == (
                "verified" if role == "recover-after" else "ambiguous"
            )
            print(result.state.value, flush=True)
    finally:
        close(values)


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_sigkill_fresh_first_opener_marker_lookup_only(tmp_path, boundary):
    import selectors
    import signal
    import subprocess
    import synaptic_tuner.api.v1.reference.local_publication as production

    source = Path(production.__file__).resolve().parents[4]
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "sys.path.insert(0, sys.argv[2]); import test_local_publication as t; "
        "from pathlib import Path; t._process(sys.argv[3], Path(sys.argv[4]))"
    )
    args = [sys.executable, "-I", "-c", script, str(source), str(Path(__file__).parent)]
    child = subprocess.Popen(
        args + ["producer-" + boundary, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selected:
            selected.register(child.stdout, selectors.EVENT_READ)
            assert selected.select(15), "producer did not reach retained boundary"
            assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=5)
        assert child.returncode == -signal.SIGKILL
        # First store opener after the kill is this new interpreter.
        result = subprocess.run(
            args + ["recover-" + boundary, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == (
            "verified" if boundary == "after" else "ambiguous"
        )
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdout.close()
        child.stderr.close()


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "native/../escape",
        "publication-receipt.json",
        "nested\\windows",
    ],
)
def test_role_mapping_rejects_before_files(tmp_path, path):
    with pytest.raises(ValueError):
        LocalPublicationDestinationV1(
            tmp_path,
            destination_ref=REF,
            display_name="Target",
            role_paths={"adapter": path},
            maximum_artifact_bytes=10,
            maximum_total_bytes=10,
            authority_ref="host-authority",
            authority=AUTH,
            clock=lambda: STAMP,
            deadline=time.monotonic() + 10,
        )
    assert not list(tmp_path.iterdir())


def test_deadline_and_constructor_failure_close_descriptors(tmp_path):
    before = len(list(Path("/proc/self/fd").iterdir()))
    for deadline in (False, float("inf"), 10**1000, time.monotonic() - 1):
        with pytest.raises(ValueError):
            LocalArtifactSpoolV1(
                tmp_path,
                maximum_artifact_bytes=10,
                maximum_total_bytes=10,
                deadline=deadline,
            )
    with pytest.raises(ValueError):
        LocalPublicationDestinationV1(
            tmp_path,
            destination_ref="invalid ref",
            display_name="Target",
            role_paths={"adapter": "adapter"},
            maximum_artifact_bytes=10,
            maximum_total_bytes=10,
            authority_ref="host-authority",
            authority=AUTH,
            clock=lambda: STAMP,
            deadline=time.monotonic() + 10,
        )
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_wrong_key_or_role_mapping_cannot_read_original_marker(tmp_path):
    values = stack(tmp_path)
    try:
        service, store, _, _, _, _ = values
        result = service.publish(PublicationRequest(fixture.RUN, REF))
        record = store.get(result.publication.publication_id)
        artifact = record.receipt.inventory.inventory.artifacts[0]
        for key, path in ((b"z" * 32, "native/adapter.bin"), (b"x" * 32, "other.bin")):
            other = LocalPublicationDestinationV1(
                tmp_path / "destination",
                destination_ref=REF,
                display_name="Local target",
                role_paths={"adapter": path},
                maximum_artifact_bytes=4 << 20,
                maximum_total_bytes=8 << 20,
                authority_ref="host-authority",
                authority=ReferencePublicationEvidenceAuthorityV1("local-key", key),
                clock=lambda: STAMP,
                deadline=time.monotonic() + 10,
            )
            try:
                with pytest.raises(ValueError):
                    list(other.iter_bytes(record.command, artifact, 4 << 20))
            finally:
                other.close()
    finally:
        close(values)


def test_root_replacement_and_expired_deadline_refuse(tmp_path):
    values = stack(tmp_path)
    try:
        _, _, _, _, destination, _ = values
        original = tmp_path / "destination"
        original.rename(tmp_path / "retained")
        original.mkdir(mode=0o700)
        with pytest.raises(ValueError):
            destination.root.check()
    finally:
        close(values)
    spool = LocalArtifactSpoolV1(
        tmp_path / "spool",
        maximum_artifact_bytes=1,
        maximum_total_bytes=1,
        deadline=time.monotonic() + 10,
    )
    try:
        spool.root.deadline = time.monotonic() - 1
        with pytest.raises(ValueError):
            spool.open("a" * 64, "adapter", 1)
    finally:
        spool.close()


def test_clock_mutation_cannot_publish_authenticated_marker(tmp_path):
    values = stack(tmp_path)
    try:
        service, _, _, _, destination, _ = values
        count = 0

        def clock():
            nonlocal count
            count += 1
            if count == 2:
                destination.root.path.chmod(0o777)
            return STAMP

        destination.clock = clock
        result = service.publish(PublicationRequest(fixture.RUN, REF))
        assert result.state.value != "verified"
        assert not list((tmp_path / "destination").rglob("publication-receipt.json"))
    finally:
        close(values)


def test_spool_bound_counts_other_publications_and_partial_bytes(tmp_path):
    spool = LocalArtifactSpoolV1(
        tmp_path,
        maximum_artifact_bytes=4,
        maximum_total_bytes=5,
        deadline=time.monotonic() + 10,
    )
    try:
        sink = spool.open("a" * 64, "adapter", 4)
        sink.write(b"1234")
        sink.abort()
        other = spool.open("b" * 64, "adapter", 4)
        with pytest.raises(ValueError):
            other.write(b"12")
    finally:
        spool.close()


@pytest.mark.parametrize("target", ["spool", "destination"])
@pytest.mark.parametrize("mutation", ["chmod", "hardlink"])
def test_post_write_identity_mutation_never_promoted(
    tmp_path, monkeypatch, target, mutation
):
    import synaptic_tuner.api.v1.reference.local_publication as production

    values = stack(tmp_path)
    original = production._write
    mutated = []

    def write(fd, chunk):
        original(fd, chunk)
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if target in path.parts and not mutated:
            mutated.append(path)
            if mutation == "chmod":
                os.fchmod(fd, 0o644)
            else:
                os.link(path, path.parent / "unexpected-link")

    monkeypatch.setattr(production, "_write", write)
    try:
        if target == "spool":
            with pytest.raises(PublicationErrorV1) as error:
                values[0].publish(PublicationRequest(fixture.RUN, REF))
            assert error.value.code is PublicationCodeV1.SOURCE_CONTENT_INVALID
        else:
            result = values[0].publish(PublicationRequest(fixture.RUN, REF))
            assert result.state.value != "verified"
        assert mutated
        assert not list((tmp_path / "destination").rglob("publication-receipt.json"))
    finally:
        close(values)


def test_lookup_callback_mutation_does_not_sign_found(tmp_path):
    values = stack(tmp_path)
    try:
        service, store, _, _, destination, _ = values
        original = destination.publish_once

        def lost(*args):
            original(*args)
            raise TimeoutError("fixture lost receipt")

        destination.publish_once = lost
        first = service.publish(PublicationRequest(fixture.RUN, REF))
        assert first.state.value == "ambiguous"
        record = store.get(first.publication.publication_id)
        path = (
            tmp_path
            / "destination"
            / record.command.publication_id
            / "native/adapter.bin"
        )
        count = 0

        def clock():
            nonlocal count
            count += 1
            if count == 2:
                path.chmod(0o644)
            return STAMP

        destination.clock = clock
        recovered = service.publish(PublicationRequest(fixture.RUN, REF))
        assert recovered.state.value == "ambiguous"
        assert store.get(first.publication.publication_id).receipt is None
    finally:
        close(values)


def test_fifo_original_leaf_rejected_without_blocking(tmp_path):
    values = stack(tmp_path)
    try:
        service, store, _, _, destination, _ = values
        result = service.publish(PublicationRequest(fixture.RUN, REF))
        record = store.get(result.publication.publication_id)
        artifact = record.receipt.inventory.inventory.artifacts[0]
        path = tmp_path / "destination" / record.command.publication_id / artifact.path
        path.unlink()
        os.mkfifo(path, 0o600)
        started = time.monotonic()
        with pytest.raises(ValueError):
            list(destination.iter_bytes(record.command, artifact, 4 << 20))
        assert time.monotonic() - started < 1
    finally:
        close(values)


def test_partial_write_reservation_sticks_and_sink_cannot_finish(tmp_path, monkeypatch):
    import synaptic_tuner.api.v1.reference.local_publication as production

    spool = LocalArtifactSpoolV1(
        tmp_path,
        maximum_artifact_bytes=4,
        maximum_total_bytes=5,
        deadline=time.monotonic() + 10,
    )
    try:
        first = spool.open("a" * 64, "adapter", 4)

        def partial(fd, chunk):
            assert os.write(fd, chunk[:2]) == 2
            raise OSError("test interrupted partial write")

        monkeypatch.setattr(production, "_write", partial)
        with pytest.raises(OSError):
            first.write(b"1234")
        assert os.fstat(first.leaf.fd).st_size == 2
        assert spool._total_bytes == 4
        with pytest.raises(ValueError):
            first.finish()
        with pytest.raises(ValueError):
            first.write(b"x")
        second = spool.open("b" * 64, "other", 4)
        with pytest.raises(ValueError):
            second.write(b"12")
        assert os.fstat(second.leaf.fd).st_size == 0
        assert spool._total_bytes == 4
    finally:
        spool.close()
