"""Public native SFT -> real SQLite/spool -> scripted HF commit/readback.

No provider requests or model loading. Exact pinned SDK operation values and
actual HTTPX streaming run under a controlled transport.
"""

from contextlib import contextmanager
import json
import time

import pytest

httpx = pytest.importorskip("httpx")
sdk = pytest.importorskip("huggingface_hub")
pytestmark = pytest.mark.skipif(
    sdk.__version__ != "1.33.0", reason="pinned HF 1.33 publication transport proof"
)

from tests.training.test_docker_host_composition import host_fixture as host_fixture
from tests.execution.providers.docker_provider_v1.conftest import profile  # noqa: F401
from tests.execution.providers.docker_provider_v1.test_local_source import material  # noqa: F401
from tests.training.test_local_run_artifact_publication import complete
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference.publication_store import DurablePublicationStoreV1
from synaptic_tuner.api.v1.reference.local_publication import LocalArtifactSpoolV1
from synaptic_tuner.api.v1.reference.hf_publication import (
    HFModelPublicationDestinationV1,
)
from synaptic_tuner.api.v1.reference.publication_sources import (
    compose_publication_artifacts,
)
from synaptic_tuner.api.v1.artifacts_facade import PublicationRequest
from synaptic_tuner.api.v1.secrets import SecretRef

PARENT = "a" * 40
OID = "b" * 40
REF = "hf/model"


class Hub:
    def __init__(self):
        self.head = PARENT
        self.files = {}
        self.calls = []
        self.reads = []
        self.lose_reply = False
        self.secrets = 0

    def resolve(self, reference):
        assert reference == SecretRef("env", "TEST_HF_PUBLICATION_TOKEN")
        self.secrets += 1
        return "synthetic-test-token"

    def api(self, **kwargs):
        assert kwargs == {
            "endpoint": "https://huggingface.co",
            "token": "synthetic-test-token",
        }
        return self

    def preupload_lfs_files(self, repo_id, additions, **kwargs):
        assert repo_id == "fixture/model"
        assert kwargs == {
            "token": "synthetic-test-token",
            "repo_type": "model",
            "revision": "main",
            "create_pr": False,
            "num_threads": 1,
            "free_memory": False,
        }
        for operation in additions:
            operation._upload_mode = "regular"
            operation._should_ignore = False

    def http(self, **kwargs):
        assert kwargs["trust_env"] is False
        if kwargs["follow_redirects"]:
            assert kwargs["max_redirects"] == 3
        return httpx.Client(transport=httpx.MockTransport(self.request), **kwargs)

    def request(self, request):
        assert request.headers["authorization"] == "Bearer synthetic-test-token"
        assert request.headers["accept-encoding"] == "identity"
        path = request.url.path
        self.reads.append(path)
        if path == "/api/models/fixture/model/revision/main":
            data = json.dumps(
                {"id": "fixture/model", "private": True, "sha": self.head}
            ).encode()
        elif path == "/api/models/fixture/model/paths-info/" + PARENT:
            assert request.method == "POST"
            assert request.headers["content-type"].startswith(
                "application/x-www-form-urlencoded"
            )
            data = b"[]"
        elif path == "/api/models/fixture/model/commit/main":
            import base64

            documents = [json.loads(x) for x in request.content.splitlines()]
            assert (
                documents[0]["key"] == "header"
                and documents[0]["value"]["parentCommit"] == PARENT
            )
            assert self.head == PARENT
            files = {}
            for item in documents[1:]:
                assert item["key"] == "file" and item["value"]["encoding"] == "base64"
                files[item["value"]["path"]] = base64.b64decode(
                    item["value"]["content"], validate=True
                )
            self.calls.append((documents, files))
            self.files[OID] = files
            self.head = OID
            if self.lose_reply:
                raise httpx.ReadError("synthetic lost commit reply")
            data = json.dumps(
                {
                    "commitOid": OID,
                    "commitUrl": "https://huggingface.co/fixture/model/commit/" + OID,
                }
            ).encode()
        elif "/tree/" in path:
            prefix = path.split("/tree/", 1)[1].split("/", 1)[1]
            data = json.dumps(
                [
                    {"type": "file", "path": name, "size": len(raw)}
                    for name, raw in self.files.get(OID, {}).items()
                    if name.startswith(prefix + "/")
                ]
            ).encode()
        elif "/resolve/" in path:
            tail = path.split("/resolve/", 1)[1]
            oid, file = tail.split("/", 1)
            data = self.files.get(oid, {}).get(file)
            if data is None:
                return httpx.Response(404, stream=httpx.ByteStream(b""))
        else:
            raise AssertionError(path)
        return httpx.Response(200, stream=httpx.ByteStream(data))


class Registry:
    def __init__(self, destination):
        self.destination = destination

    def resolve(self, reference):
        assert reference == REF
        return self.destination.descriptor, self.destination

    def list(self, limit):
        return (self.destination.descriptor,), True


@contextmanager
def ports(root, records, host, hub, *, create=True):
    if create:
        for name in ("publication-locks", "spool", "render"):
            (root / name).mkdir(mode=0o700)
    deadline = time.monotonic() + 60
    store = DurablePublicationStoreV1(
        records, root / "publication-locks", create=create
    )
    spool = (
        LocalArtifactSpoolV1(
            root / "spool",
            maximum_artifact_bytes=4 << 20,
            maximum_total_bytes=8 << 20,
            deadline=deadline,
        )
        if create
        else None
    )
    authority = host.composition.authority.publication_authority
    destination = HFModelPublicationDestinationV1(
        destination_ref=REF,
        display_name="Scripted HF native model",
        repo_id="fixture/model",
        branch="main",
        parent_commit=PARENT,
        secret_ref=SecretRef("env", "TEST_HF_PUBLICATION_TOKEN"),
        secrets=hub,
        authority_ref="host-publication",
        authority=authority,
        clock=host.composition.authority.clock.now,
        deadline=deadline,
        maximum_artifact_bytes=4 << 20,
        maximum_total_bytes=8 << 20,
        maximum_upload_bytes=16 << 20,
        render_root=root / "render" if create else None,
        spool=spool,
        api_factory=hub.api,
        http_factory=hub.http,
    )
    api = compose_publication_artifacts(
        runs=host.api.runs,
        store=store,
        spool=spool,
        destinations=Registry(destination),
        authority_ref="host-publication",
        authority=authority,
        clock=host.composition.authority.clock.now,
    )
    try:
        yield api, store, spool, destination
    finally:
        destination.close()
        if spool:
            spool.close()
        store.close()


def test_public_native_publish_usable_projection_and_cold_immutable_readback(
    host_fixture, monkeypatch
):
    compose, kwargs, commands, state, roles, grants, root = host_fixture
    hub = Hub()
    database = root / "workflow.sqlite3"
    with SQLiteDurableStoreV1(database) as records, compose(records) as host:
        run, values = complete(host, kwargs, state, roles, monkeypatch)
        with ports(root, records, host, hub) as (api, store, spool, destination):
            value = api.publish(PublicationRequest(run, REF))
            assert value.state.value == "verified" and len(value.artifacts) == 5
            assert len(hub.calls) == 1
            record = store.get(value.publication.publication_id)
            prefix = "releases/" + record.command.publication_id
            marker = json.loads(hub.files[OID][prefix + "/publication.json"])
            assert marker["model"]["kind"] == "lora"
            assert any(
                x["path"].endswith("/adapter_config.json") for x in marker["rendered"]
            )
            assert any(
                x["path"].endswith("/tokenizer.json") for x in marker["rendered"]
            )
            for item in marker["native"]:
                assert hub.files[OID][item["path"]] == values[item["role"]]
            for artifact in record.receipt.inventory.inventory.artifacts:
                assert artifact.path.startswith(OID + "/")
                assert (
                    b"".join(destination.iter_bytes(record.command, artifact, 4 << 20))
                    == values[artifact.role]
                )
            original = value

    def forbidden(*args, **kwargs):
        pytest.fail("cold publication reopened source or provider effect")

    kwargs["cli_configuration"]["runner"] = forbidden
    monkeypatch.setattr(grants, "bind", forbidden)
    from synaptic_tuner.api.v1.runs_facade import RunsAPI

    monkeypatch.setattr(RunsAPI, "artifacts", forbidden)
    with (
        SQLiteDurableStoreV1(database, create=False) as records,
        compose(records, cold=True) as host,
    ):
        with ports(root, records, host, hub, create=False) as (
            api,
            store,
            _,
            destination,
        ):
            assert api.publish(PublicationRequest(run, REF)) == original
            assert api.verify(original.publication).verified
            assert len(hub.calls) == 1


def test_lost_reply_reopens_readonly_owned_snapshot_without_second_commit(
    host_fixture, monkeypatch
):
    compose, kwargs, _, state, roles, grants, root = host_fixture
    hub = Hub()
    hub.lose_reply = True
    database = root / "workflow.sqlite3"
    with SQLiteDurableStoreV1(database) as records, compose(records) as host:
        run, _ = complete(host, kwargs, state, roles, monkeypatch)
        with ports(root, records, host, hub) as (api, store, _, destination):
            value = api.publish(PublicationRequest(run, REF))
            assert value.state.value == "ambiguous"
            assert len(hub.calls) == 1
    with (
        SQLiteDurableStoreV1(database, create=False) as records,
        compose(records, cold=True) as host,
    ):
        with ports(root, records, host, hub, create=False) as (
            api,
            store,
            _,
            destination,
        ):
            recovered = api.publish(PublicationRequest(run, REF))
            assert recovered.state.value == "verified" and len(hub.calls) == 1
            record = store.get(recovered.publication.publication_id)
            assert len(record.ownership_history) == len(record.recovery_permits) == 1
            assert all(
                x.path.startswith(OID + "/")
                for x in record.receipt.inventory.inventory.artifacts
            )


@pytest.mark.parametrize("status", [307, 308])
def test_final_commit_redirect_never_replays_post(host_fixture, monkeypatch, status):  # noqa: F811
    compose, kwargs, _, state, roles, _, root = host_fixture
    hub = Hub()
    original = hub.request
    targets = []

    def redirect(request):
        targets.append(request.url.path)
        if request.url.path == "/api/models/fixture/model/commit/main":
            hub.calls.append(("redirect", request.content))
            return httpx.Response(
                status,
                headers={"Location": "https://huggingface.co/replay"},
                stream=httpx.ByteStream(b"bounded failure"),
            )
        return original(request)

    hub.request = redirect
    with SQLiteDurableStoreV1(root / "db") as records, compose(records) as host:
        run, _ = complete(host, kwargs, state, roles, monkeypatch)
        with ports(root, records, host, hub) as (api, store, _, destination):
            result = api.publish(PublicationRequest(run, REF))
            assert result.state.value == "ambiguous"
            assert len(hub.calls) == 1 and "/replay" not in targets and not hub.files


@pytest.mark.parametrize("mutation", ["extra", "tensor", "marker", "missing"])
def test_cold_unknown_commit_requires_exact_owned_native_and_rendered_snapshot(
    host_fixture, monkeypatch, mutation
):  # noqa: F811
    compose, kwargs, _, state, roles, _, root = host_fixture
    hub = Hub()
    hub.lose_reply = True
    database = root / "db"
    with SQLiteDurableStoreV1(database) as records, compose(records) as host:
        run, _ = complete(host, kwargs, state, roles, monkeypatch)
        with ports(root, records, host, hub) as (api, store, _, destination):
            result = api.publish(PublicationRequest(run, REF))
            assert result.state.value == "ambiguous" and len(hub.calls) == 1
            command = store.get(result.publication.publication_id).command
            prefix = "releases/" + command.publication_id
    files = hub.files[OID]
    if mutation == "extra":
        files[prefix + "/model/unreviewed.py"] = b"not part of the owned model"
    elif mutation == "tensor":
        path = next(x for x in files if "/model/" in x and x.endswith(".safetensors"))
        files[path] = b"changed model tensor"
    elif mutation == "marker":
        value = json.loads(files[prefix + "/publication.json"])
        value["ownership"]["ownership_id"] = "0" * 64
        from tuner.execution.foundation_v2.canonical import canonical_bytes

        files[prefix + "/publication.json"] = canonical_bytes(value)
    else:
        del files[prefix + "/publication.json"]
    with (
        SQLiteDurableStoreV1(database, create=False) as records,
        compose(records, cold=True) as host,
    ):
        with ports(root, records, host, hub, create=False) as (
            api,
            store,
            _,
            destination,
        ):
            recovered = api.publish(PublicationRequest(run, REF))
            assert recovered.state.value == "ambiguous" and not recovered.artifacts
            assert len(hub.calls) == 1


def bare_destination(hub, **overrides):
    from synaptic_tuner.api.v1.reference.authority import (
        ReferencePublicationEvidenceAuthorityV1,
    )

    values = dict(
        destination_ref=REF,
        display_name="Bounded HTTP unit control",
        repo_id="fixture/model",
        branch="main",
        parent_commit=PARENT,
        secret_ref=SecretRef("env", "TEST_HF_PUBLICATION_TOKEN"),
        secrets=hub,
        authority_ref="host-publication",
        authority=ReferencePublicationEvidenceAuthorityV1("fixture/key", b"x" * 32),
        clock=lambda: "2026-10-14T00:00:00Z",
        deadline=time.monotonic() + 10,
        maximum_artifact_bytes=4 << 20,
        maximum_total_bytes=8 << 20,
        maximum_upload_bytes=16 << 20,
        api_factory=hub.api,
        http_factory=hub.http,
    )
    values.update(overrides)
    return HFModelPublicationDestinationV1(**values)


@pytest.mark.parametrize("size,accepted", [(2 << 20, True), ((2 << 20) + 1, False)])
def test_actual_raw_stream_enforces_multimegabyte_boundary_and_closes(size, accepted):
    hub = Hub()
    closed = []

    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * size

        def close(self):
            closed.append(True)

    hub.request = lambda request: httpx.Response(200, stream=Body())
    destination = bare_destination(hub)
    try:
        if accepted:
            chunks = list(
                destination._stream_http("/bounded", "synthetic-test-token", 2 << 20)
            )
            assert sum(map(len, chunks)) == size and all(
                len(x) <= 1 << 20 for x in chunks
            )
        else:
            with pytest.raises(ValueError):
                list(
                    destination._stream_http(
                        "/bounded", "synthetic-test-token", 2 << 20
                    )
                )
        assert closed
    finally:
        destination.close()


def test_compressed_response_rejected_without_decoding_or_reading_body():
    hub = Hub()
    reads = []
    closed = []

    class Body(httpx.SyncByteStream):
        def __iter__(self):
            reads.append(True)
            yield b"hostile compressed bytes"

        def close(self):
            closed.append(True)

    hub.request = lambda request: httpx.Response(
        200, headers={"Content-Encoding": "gzip"}, stream=Body()
    )
    destination = bare_destination(hub)
    try:
        with pytest.raises(ValueError):
            list(destination._stream_http("/compressed", "synthetic-test-token", 100))
        assert closed and not reads
    finally:
        destination.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_artifact_bytes", True),
        ("maximum_upload_bytes", 0),
        ("maximum_commit_bytes", True),
        ("private", 1),
        ("parent_commit", "main"),
        ("deadline", float("inf")),
    ],
)
def test_invalid_finite_scope_rejects_before_named_secret_or_http(field, value):
    hub = Hub()
    with pytest.raises(ValueError):
        bare_destination(hub, **{field: value})
    assert hub.secrets == 0 and not hub.calls and not hub.reads


def test_hash_consistent_malformed_native_archive_refuses_before_credentials(
    host_fixture, monkeypatch
):  # noqa: F811
    from dataclasses import replace
    import hashlib
    from types import SimpleNamespace
    from synaptic_tuner.api.v1.reference.publication_sources import RunsArtifactSourceV1
    from synaptic_tuner.api.v1.artifacts_facade import ArtifactsAPI
    from tuner.execution.coordinator_v1.publication import PublicationOperationsV1

    compose, kwargs, _, state, roles, _, root = host_fixture
    hub = Hub()
    bad = b"hash-consistent malformed model archive"
    with SQLiteDurableStoreV1(root / "db") as records, compose(records) as host:
        run, _ = complete(host, kwargs, state, roles, monkeypatch)
        authority = host.composition.authority.publication_authority
        actual = RunsArtifactSourceV1(
            host.api.runs, authority_ref="host-publication", authority=authority
        )
        original = actual.describe(run)
        model = replace(
            original.artifacts[0],
            sha256=hashlib.sha256(bad).hexdigest(),
            size_bytes=len(bad),
        )

        # Deliberate trusted-source port fault. Original run really verified; the
        # independent renderer must reject even an authenticated wrong-byte declaration.
        class Fault:
            def describe(self, selected):
                assert selected == run
                value = replace(
                    original, artifacts=(model, *original.artifacts[1:]), tag="0" * 64
                )
                return replace(
                    value,
                    tag=authority.sign("publication-verified-source/v1", value.payload),
                )

            def open(self, request):
                if request.role == "final_model":
                    return SimpleNamespace(
                        run=run,
                        artifact=model,
                        maximum_bytes=request.maximum_bytes,
                        iter_bytes=lambda: iter((bad,)),
                    )
                return actual.open(request)

        with ports(root, records, host, hub) as (_, store, spool, destination):
            api = ArtifactsAPI(
                PublicationOperationsV1(
                    store=store,
                    spool=spool,
                    destinations=Registry(destination),
                    sources=Fault(),
                    authority=authority,
                    clock=host.composition.authority.clock.now,
                )
            )
            result = api.publish(PublicationRequest(run, REF))
            assert result.state.value == "ambiguous"
            assert hub.secrets == 0 and not hub.calls and not hub.reads


def test_late_sdk_owns_duplicate_handles_after_parent_closes_custody(
    host_fixture, monkeypatch
):  # noqa: F811
    from threading import Event
    from tuner.execution.providers.hf_jobs_provider_v1 import transport

    compose, kwargs, _, state, roles, _, root = host_fixture
    hub = Hub()
    entered, release, posted = Event(), Event(), Event()
    handles = []
    original_preupload = hub.preupload_lfs_files
    original_http = hub.request

    def held(repo_id, additions, **controls):
        handles.extend(
            x.path_or_fileobj for x in additions if hasattr(x.path_or_fileobj, "closed")
        )
        entered.set()
        assert release.wait(5)
        assert handles and all(not x.closed for x in handles)
        for handle in handles:
            handle.seek(0)
            assert handle.read()
            handle.seek(0)
        return original_preupload(repo_id, additions, **controls)

    def observe(request):
        value = original_http(request)
        if "/commit/" in request.url.path:
            posted.set()
        return value

    hub.preupload_lfs_files, hub.request = held, observe
    actual_bounded = transport._bounded
    # Finite stage-specific waiting clamp; no production clock/bound changes.
    monkeypatch.setattr(
        transport,
        "_bounded",
        lambda operation, deadline: actual_bounded(
            operation, min(deadline, time.monotonic() + 0.2)
        ),
    )
    try:
        with SQLiteDurableStoreV1(root / "db") as records, compose(records) as host:
            run, _ = complete(host, kwargs, state, roles, monkeypatch)
            with ports(root, records, host, hub) as (api, store, spool, destination):
                result = api.publish(PublicationRequest(run, REF))
                assert result.state.value == "ambiguous" and entered.is_set()
                destination.close()
                spool.close()
                assert all(not x.closed for x in handles)
                release.set()
                assert posted.wait(5)
                deadline = time.monotonic() + 3
                while not all(x.closed for x in handles):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                assert len(hub.calls) == 1
                # Late worker never signs/updates owner journal or claims success.
                assert (
                    store.get(result.publication.publication_id).phase.value
                    == "ambiguous"
                )
    finally:
        release.set()


def test_sigkill_after_provider_commit_fresh_first_sqlite_opener_lookup_only(
    host_fixture, monkeypatch
):  # noqa: F811
    import multiprocessing
    import os
    import signal
    from synaptic_tuner.api.v1.runs_facade import RunsAPI

    compose, kwargs, _, state, roles, grants, root = host_fixture
    database = root / "db"
    remote = root / "scripted-hf-service"
    remote.mkdir(mode=0o700)
    with SQLiteDurableStoreV1(database) as records, compose(records) as host:
        run, _ = complete(host, kwargs, state, roles, monkeypatch)
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    class DurableHub(Hub):
        def __init__(self):
            super().__init__()
            if (remote / "head.json").exists():
                self.head = json.loads((remote / "head.json").read_text())["oid"]
                self.files[self.head] = {
                    str(p.relative_to(remote / "files")): p.read_bytes()
                    for p in (remote / "files").rglob("*")
                    if p.is_file()
                }

        def request(self, request):
            value = super().request(request)
            if "/commit/" in request.url.path:
                # Scripted service owns an actual durable ledger independently
                # of producer memory. No engine/private-store restoration shortcut.
                files_root = remote / "files"
                files_root.mkdir(mode=0o700)
                for name, raw in self.files[OID].items():
                    target = files_root / name
                    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                    with target.open("xb") as file:
                        file.write(raw)
                        file.flush()
                        os.fsync(file.fileno())
                with (remote / "head.json").open("x") as file:
                    json.dump({"oid": OID, "commits": 1}, file)
                    file.flush()
                    os.fsync(file.fileno())
                for directory in sorted(
                    (p for p in remote.rglob("*") if p.is_dir()),
                    key=lambda p: len(p.parts),
                    reverse=True,
                ):
                    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                fd = os.open(remote, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                send.send("remote-commit-before-receipt")
                send.close()
                signal.pause()
                raise AssertionError("producer must be killed")
            return value

    def produce():
        receive.close()
        with (
            SQLiteDurableStoreV1(database, create=False) as records,
            compose(records, cold=True) as host,
        ):
            with ports(root, records, host, DurableHub()) as (
                api,
                store,
                _,
                destination,
            ):
                api.publish(PublicationRequest(run, REF))

    producer = context.Process(target=produce)
    producer.start()
    send.close()
    try:
        assert receive.poll(15), "producer did not commit scripted provider bytes"
        assert receive.recv() == "remote-commit-before-receipt"
        os.kill(producer.pid, signal.SIGKILL)
        producer.join(timeout=5)
        assert producer.exitcode == -signal.SIGKILL
        before = {
            str(p.relative_to(remote)): p.read_bytes()
            for p in remote.rglob("*")
            if p.is_file()
        }
        replies, report = context.Pipe(duplex=False)

        def recover():
            replies.close()

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "cold recovery reopened source or submitted provider effect"
                )

            kwargs["cli_configuration"]["runner"] = forbidden
            grants.bind = forbidden
            RunsAPI.artifacts = forbidden
            hub = DurableHub()
            hub.preupload_lfs_files = forbidden
            # FIRST database open since SIGKILL, new process and fresh objects.
            with (
                SQLiteDurableStoreV1(database, create=False) as records,
                compose(records, cold=True) as host,
            ):
                with ports(root, records, host, hub, create=False) as (
                    api,
                    store,
                    _,
                    destination,
                ):
                    result = api.publish(PublicationRequest(run, REF))
                    assert (
                        result.state.value == "verified" and len(result.artifacts) == 5
                    )
                    assert api.verify(result.publication).verified
                    record = store.get(result.publication.publication_id)
                    assert (
                        len(record.ownership_history)
                        == len(record.recovery_permits)
                        == 1
                    )
                    assert not hub.calls and all(
                        x.path.startswith(OID + "/")
                        for x in record.receipt.inventory.inventory.artifacts
                    )
                    assert before == {
                        str(p.relative_to(remote)): p.read_bytes()
                        for p in remote.rglob("*")
                        if p.is_file()
                    }
                    report.send("verified-readonly")
            report.close()

        consumer = context.Process(target=recover)
        consumer.start()
        report.close()
        try:
            assert replies.poll(15), "fresh consumer failed owned snapshot recovery"
            assert replies.recv() == "verified-readonly"
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


@pytest.mark.parametrize("mode", ["valid", "cycle", "crossrepo", "over_limit"])
def test_exact_inventory_finite_pagination_rejects_aliases_and_repeated_cursor(mode):
    from types import SimpleNamespace

    hub = Hub()
    destination = bare_destination(hub)
    publication = "0" * 64
    prefix = "releases/" + publication
    path = "/api/models/fixture/model/tree/" + OID + "/releases%2F" + publication
    calls = []
    body = {
        "native": [{"path": prefix + "/native/a", "size_bytes": 2}],
        "rendered": [{"path": prefix + "/model/b", "size_bytes": 3}],
    }

    def response(request):
        calls.append(str(request.url))
        assert b"releases%2F" in request.url.raw_path
        headers = {}
        if mode == "valid":
            if len(calls) == 1:
                entries = [
                    {"type": "directory", "path": prefix + "/native"},
                    {"type": "file", "path": prefix + "/native/a", "size": 2},
                ]
                headers["Link"] = (
                    "<https://huggingface.co" + path + '?cursor=next>; rel="next"'
                )
            else:
                entries = [
                    {"type": "directory", "path": prefix + "/model"},
                    {"type": "file", "path": prefix + "/model/b", "size": 3},
                    {"type": "file", "path": prefix + "/publication.json", "size": 4},
                ]
        else:
            entries = []
            target = (
                path + "?cursor=" + ("repeat" if mode == "cycle" else str(len(calls)))
            )
            if mode == "crossrepo":
                target = target.replace("fixture/model", "foreign/model")
            headers["Link"] = "<https://huggingface.co" + target + '>; rel="next"'
        return httpx.Response(
            200, headers=headers, stream=httpx.ByteStream(json.dumps(entries).encode())
        )

    hub.request = response
    try:
        if mode == "valid":
            destination._inventory(
                SimpleNamespace(publication_id=publication),
                OID,
                "synthetic-test-token",
                body,
                4,
            )
            assert len(calls) == 2
        else:
            with pytest.raises(ValueError):
                destination._inventory(
                    SimpleNamespace(publication_id=publication),
                    OID,
                    "synthetic-test-token",
                    body,
                    4,
                )
            assert len(calls) == {"cycle": 2, "crossrepo": 1, "over_limit": 32}[mode]
    finally:
        destination.close()
