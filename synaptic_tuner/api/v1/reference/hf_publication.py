"""Optional create-only HF model-repository publication destination.

Native artifacts stay byte-identical; a bounded admitted model/tokenizer projection
is committed beside them. Caller deadlines bound waiting, not SDK termination.
Late transfers may continue; an unknown commit is never retried by this adapter.
No repository provisioning, model loading, merging or implicit credentials.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from queue import Empty, Full, Queue
import re
from threading import Event, Thread, get_ident, Lock
import time
from urllib.parse import quote, urlsplit

from synaptic_tuner.api.v1.secrets import SecretRef
from .authority import ReferencePublicationEvidenceAuthorityV1
from .local_publication import _Root, _Tree, _Leaf
from tuner.execution.coordinator_v1.publication import (
    AuthenticatedDestinationV1,
    AuthenticatedDestinationInventoryV1,
    AuthenticatedPublicationReceiptV1,
    AuthenticatedLookupV1,
    LookupOutcomeV1,
    DestinationArtifactV1,
    DestinationInventoryV1,
    MaterializedSourceV1,
    PublicationCommandV1,
    PublicationOperationsV1,
    PublicationRecordV1,
    TransferOwnershipV1,
    LookupRecoveryPermitV1,
    MAX_ARTIFACT_BYTES,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes

_ROLES = (
    "final_model",
    "tokenizer",
    "training_lineage",
    "training_metrics",
    "workload_record",
)
_MARKER_MAX = 1 << 20
_OID = re.compile(r"[0-9a-f]{40}")
_PURPOSE = "publication-hf-model-marker/v1"


def _invalid():
    return ValueError("hf_publication_unavailable")


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise _invalid()
            result[key] = value
        return result

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise _invalid()
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_float=number,
        parse_constant=lambda _: (_ for _ in ()).throw(_invalid()),
    )


class HFModelPublicationDestinationV1:
    """Existing destination port; caller owns stores, keys, roots and repo scope.

    ``api_factory`` and ``http_factory`` are explicit host transport seams, not
    authority. Defaults lazily use pinned Hub1.33 and HTTPX. One commit attempt
    can involve multiple content-addressed preupload calls. Historical lookup
    verifies an immutable owned snapshot, not the original RPC's causal history.
    Repo namespace is target scope, not proof of the token's authenticated owner.
    Host validation/serialization and OS I/O are synchronous; no total hard
    completion deadline is promised.
    """

    def __init__(
        self,
        *,
        destination_ref,
        display_name,
        repo_id,
        branch,
        parent_commit,
        secret_ref,
        secrets,
        authority_ref,
        authority,
        clock,
        deadline,
        maximum_artifact_bytes,
        maximum_total_bytes,
        maximum_upload_bytes,
        render_root=None,
        spool=None,
        endpoint="https://huggingface.co",
        private=True,
        api_factory=None,
        http_factory=None,
        maximum_commit_bytes=4 << 20,
    ):
        if (
            type(authority) is not ReferencePublicationEvidenceAuthorityV1
            or type(secret_ref) is not SecretRef
            or not callable(getattr(secrets, "resolve", None))
            or not callable(clock)
            or type(private) is not bool
            or type(deadline) not in (int, float)
            or not math.isfinite(deadline)
            or deadline <= time.monotonic()
            or type(endpoint) is not str
            or not re.fullmatch(r"https://[A-Za-z0-9.-]+(?::[0-9]+)?", endpoint)
            or type(repo_id) is not str
            or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo_id)
            or ".." in repo_id
            or type(branch) is not str
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", branch)
            or type(parent_commit) is not str
            or not _OID.fullmatch(parent_commit)
            or any(
                type(n) is not int or not 1 <= n <= MAX_ARTIFACT_BYTES
                for n in (
                    maximum_artifact_bytes,
                    maximum_total_bytes,
                    maximum_upload_bytes,
                )
            )
            or type(maximum_commit_bytes) is not int
            or not 1 <= maximum_commit_bytes <= 64 << 20
            or maximum_artifact_bytes > maximum_total_bytes
            or maximum_total_bytes > maximum_upload_bytes
            or any(
                x is not None and not callable(x) for x in (api_factory, http_factory)
            )
        ):
            raise _invalid()
        self.endpoint, self.repo_id, self.branch, self.parent_commit = (
            endpoint,
            repo_id,
            branch,
            parent_commit,
        )
        self.secret_ref, self.secrets = secret_ref, secrets
        self.authority, self.clock, self.deadline = authority, clock, float(deadline)
        self.maximum_upload_bytes, self.private = maximum_upload_bytes, private
        self.api_factory, self.http_factory = api_factory, http_factory
        self.maximum_commit_bytes = maximum_commit_bytes
        self.thread, self.pid = get_ident(), os.getpid()
        self.closed = False
        self._verified_snapshots = set()
        self.root = _Root(render_root, deadline) if render_root is not None else None
        self.spool = spool
        config = canonical_bytes(
            {
                **self._destination(),
                "private": private,
                "maximum_upload_bytes": maximum_upload_bytes,
                "maximum_commit_bytes": maximum_commit_bytes,
            }
        )
        self._configuration = config
        self._original_deadline = self.deadline
        value = AuthenticatedDestinationV1(
            "synaptic-publication-destination/v1",
            destination_ref,
            display_name,
            hashlib.sha256(config).hexdigest(),
            hashlib.sha256(b"owned-hf-native-plus-model/v1").hexdigest(),
            maximum_artifact_bytes,
            maximum_total_bytes,
            authority_ref,
            authority.key_ref,
            "0" * 64,
        )
        self.descriptor = self._signed(value, "publication-destination/v1")

    def _destination(self):
        return {
            "endpoint": self.endpoint,
            "repo_id": self.repo_id,
            "repo_type": "model",
            "branch": self.branch,
            "parent_commit": self.parent_commit,
            "layout": "native-plus-model/v1",
        }

    def _guard(self):
        if (
            self.closed
            or get_ident() != self.thread
            or os.getpid() != self.pid
            or self.deadline != self._original_deadline
            or canonical_bytes(
                {
                    **self._destination(),
                    "private": self.private,
                    "maximum_upload_bytes": self.maximum_upload_bytes,
                    "maximum_commit_bytes": self.maximum_commit_bytes,
                }
            )
            != self._configuration
            or time.monotonic() >= self.deadline
        ):
            raise _invalid()
        if self.root is not None:
            self.root.check()

    def _signed(self, value, purpose):
        self._guard()
        result = replace(value, tag=self.authority.sign(purpose, value.payload))
        self._guard()
        return result

    def _command(self, command):
        self._guard()
        if (
            type(command) is not PublicationCommandV1
            or not PublicationOperationsV1._descriptor_matches_command(
                self.descriptor, command
            )
            or tuple(x.role for x in command.source_inventory) != _ROLES
        ):
            raise _invalid()

    def _token(self):
        self._guard()
        token = self.secrets.resolve(self.secret_ref)
        if type(token) is not str or not token.strip():
            raise _invalid()
        self._guard()
        return token

    def _stream_http(
        self, path, token, maximum, *, method="GET", body=None, headers_out=None
    ):
        """Raw identity bytes, one bounded queued chunk; late blocked I/O may remain."""
        self._guard()
        import httpx

        messages = Queue(maxsize=1)
        stopped = Event()
        endpoint, deadline, factory = self.endpoint, self.deadline, self.http_factory

        def offer(value):
            while not stopped.is_set():
                try:
                    messages.put(value, timeout=0.05)
                    return
                except Full:
                    continue

        def execute():
            try:
                if time.monotonic() >= deadline:
                    raise _invalid()
                kwargs = {
                    "timeout": min(30, max(0.001, deadline - time.monotonic())),
                    "follow_redirects": True,
                    "max_redirects": 3,
                    "trust_env": False,
                }
                client = (
                    factory(**kwargs)
                    if factory
                    else httpx.Client(
                        transport=httpx.HTTPTransport(retries=0, trust_env=False),
                        **kwargs,
                    )
                )
                with client:
                    with client.stream(
                        method,
                        endpoint + path,
                        headers={
                            "Authorization": "Bearer " + token,
                            "Accept-Encoding": "identity",
                        },
                        data=body,
                    ) as response:
                        if response.status_code == 404:
                            offer(("missing", None))
                            return
                        if (
                            response.status_code != 200
                            or response.headers.get(
                                "Content-Encoding", "identity"
                            ).lower()
                            != "identity"
                        ):
                            raise _invalid()
                        link = response.headers.get("Link", "")
                        if len(link.encode("utf-8")) > 8192:
                            raise _invalid()
                        if headers_out is not None:
                            headers_out["next"] = response.links.get("next", {}).get(
                                "url"
                            )
                        total = 0
                        for chunk in response.iter_raw(chunk_size=1 << 20):
                            if stopped.is_set() or time.monotonic() >= deadline:
                                raise _invalid()
                            if not chunk:
                                continue
                            total += len(chunk)
                            if total > maximum:
                                raise _invalid()
                            offer(("chunk", chunk))
                        offer(("done", None))
            except BaseException:
                offer(("error", None))

        Thread(target=execute, daemon=True, name="hf-publication-read").start()
        try:
            while True:
                self._guard()
                try:
                    kind, value = messages.get(
                        timeout=max(0.001, self.deadline - time.monotonic())
                    )
                except Empty:
                    raise _invalid() from None
                self._guard()
                if kind == "done":
                    return
                if kind == "missing":
                    raise FileNotFoundError("hf_publication_unavailable")
                if kind != "chunk":
                    raise _invalid()
                yield value
        finally:
            stopped.set()

    def _metadata(self, path, token, *, method="GET", body=None, include_next=False):
        headers = {}
        raw = b"".join(
            self._stream_http(
                path, token, _MARKER_MAX, method=method, body=body, headers_out=headers
            )
        )
        value = _json(raw)
        self._guard()
        return (value, headers.get("next")) if include_next else value

    def _head(self, token):
        value = self._metadata(
            "/api/models/" + self.repo_id + "/revision/" + self.branch, token
        )
        if (
            type(value) is not dict
            or value.get("id") != self.repo_id
            or value.get("private") is not self.private
            or type(value.get("sha")) is not str
            or not _OID.fullmatch(value["sha"])
        ):
            raise _invalid()
        return value["sha"]

    def _prefix(self, command):
        return "releases/" + command.publication_id

    def _file(self, oid, path, token, maximum):
        if not _OID.fullmatch(oid):
            raise _invalid()
        return self._stream_http(
            "/" + self.repo_id + "/resolve/" + oid + "/" + quote(path, safe="/"),
            token,
            maximum,
        )

    def _receipt(self, command, body, oid):
        ownership = TransferOwnershipV1(**body["ownership"])
        inventory = DestinationInventoryV1(
            tuple(
                DestinationArtifactV1(
                    x["role"], oid + "/" + x["path"], x["sha256"], x["size_bytes"]
                )
                for x in body["native"]
            )
        )
        stamp = body["recorded_at"]
        bound = AuthenticatedDestinationInventoryV1(
            inventory,
            command.publication_id,
            command.command_digest,
            command.mutation_id,
            ownership.ownership_id,
            stamp,
            self.descriptor.authority_ref,
            self.authority.key_ref,
            "0" * 64,
        )
        bound = self._signed(bound, "publication-destination-inventory/v1")
        value = AuthenticatedPublicationReceiptV1(
            "synaptic-publication-receipt/v1",
            command.publication_id,
            command.command_digest,
            command.run,
            command.source_identity_digest,
            command.destination_ref,
            command.destination_identity_digest,
            command.mutation_id,
            ownership.claim_digest,
            ownership.ownership_id,
            bound,
            stamp,
            self.descriptor.authority_ref,
            self.authority.key_ref,
            "0" * 64,
        )
        return self._signed(value, "publication-receipt/v1")

    def _inventory(self, command, oid, token, body, marker_size):
        path = (
            "/api/models/"
            + self.repo_id
            + "/tree/"
            + oid
            + "/"
            + quote(self._prefix(command), safe="")
        )
        next_path = path + "?recursive=true&expand=false&limit=100"
        expected = {
            x["path"]: x["size_bytes"] for x in body["native"] + body["rendered"]
        }
        expected[self._prefix(command) + "/publication.json"] = marker_size
        directories = {
            self._prefix(command) + "/native",
            self._prefix(command) + "/model",
        }
        observed, seen_pages = {}, set()
        for _ in range(32):
            if next_path in seen_pages:
                raise _invalid()
            seen_pages.add(next_path)
            entries, link = self._metadata(next_path, token, include_next=True)
            if type(entries) is not list or len(entries) > 1000:
                raise _invalid()
            for entry in entries:
                if type(entry) is not dict or type(entry.get("path")) is not str:
                    raise _invalid()
                name = entry["path"]
                if name in observed or len(observed) >= 2056:
                    raise _invalid()
                if entry.get("type") == "directory":
                    if name not in directories:
                        raise _invalid()
                    observed[name] = None
                elif entry.get("type") == "file":
                    if (
                        name not in expected
                        or type(entry.get("size")) is not int
                        or entry["size"] != expected[name]
                    ):
                        raise _invalid()
                    observed[name] = entry["size"]
                else:
                    raise _invalid()
            if link is None:
                if {x: y for x, y in observed.items() if y is not None} != expected:
                    raise _invalid()
                return
            if type(link) is not str or len(link) > 8192:
                raise _invalid()
            target = urlsplit(link)
            if (
                target.scheme + "://" + target.netloc != self.endpoint
                or target.path != path
                or target.fragment
            ):
                raise _invalid()
            next_path = target.path + ("?" + target.query if target.query else "")
        raise _invalid()

    def _verified_marker(
        self, command, oid, token, *, ownership_id=None, verify_content=True
    ):
        raw = b"".join(
            self._file(
                oid, self._prefix(command) + "/publication.json", token, _MARKER_MAX
            )
        )
        value = _json(raw)
        expected = {
            "schema_version",
            "command",
            "ownership",
            "destination",
            "native",
            "rendered",
            "model",
            "recorded_at",
            "authority_ref",
            "key_ref",
            "tag",
        }
        if (
            type(value) is not dict
            or set(value) != expected
            or canonical_bytes(value) != raw
        ):
            raise _invalid()
        body = {key: val for key, val in value.items() if key != "tag"}
        if (
            body["schema_version"] != "synaptic-hf-model-publication-marker/v1"
            or body["command"] != _json(canonical_bytes(asdict(command)))
            or body["destination"] != self._destination()
            or body["authority_ref"] != self.descriptor.authority_ref
            or body["key_ref"] != self.authority.key_ref
            or not self.authority.verify(
                _PURPOSE, canonical_bytes(body), value["tag"], body["key_ref"]
            )
        ):
            raise _invalid()
        ownership = TransferOwnershipV1(**body["ownership"])
        if (
            ownership.publication_id,
            ownership.command_digest,
            ownership.mutation_id,
            ownership.claim_digest,
        ) != (
            command.publication_id,
            command.command_digest,
            command.mutation_id,
            PublicationRecordV1.claim(command, self.clock()).claim_digest,
        ):
            raise _invalid()
        if ownership_id is not None and ownership.ownership_id != ownership_id:
            raise _invalid()
        native = [
            {
                "role": x.role,
                "path": self._prefix(command) + "/native/" + x.role + ".bin",
                "sha256": x.sha256,
                "size_bytes": x.size_bytes,
            }
            for x in command.source_inventory
        ]
        if (
            body["native"] != native
            or type(body["rendered"]) is not list
            or not 1 <= len(body["rendered"]) <= 2048
        ):
            raise _invalid()
        model = body["model"]
        if (
            type(model) is not dict
            or set(model)
            != {"kind", "model_ref", "model_revision", "tokenizer_revision"}
            or model["kind"] not in ("lora", "full")
            or any(
                type(model[field]) is not str
                or not 1 <= len(model[field].encode("utf-8")) <= 4096
                for field in ("model_ref", "model_revision", "tokenizer_revision")
            )
        ):
            raise _invalid()
        total = len(raw)
        paths = []
        for item in body["native"] + body["rendered"]:
            if (
                type(item) is not dict
                or set(item)
                != (
                    {"role", "path", "sha256", "size_bytes"}
                    if "role" in item
                    else {"kind", "path", "sha256", "size_bytes"}
                )
                or type(item["size_bytes"]) is not int
                or not 1 <= item["size_bytes"] <= self.descriptor.maximum_artifact_bytes
                or type(item["sha256"]) is not str
                or not re.fullmatch("[0-9a-f]{64}", item["sha256"])
            ):
                raise _invalid()
            if "kind" in item:
                name = item["path"].removeprefix(self._prefix(command) + "/model/")
                if (
                    item["kind"] not in ("model", "tokenizer")
                    or not name
                    or "/" in name
                    or "\\" in name
                    or name in (".", "..")
                    or item["path"] != self._prefix(command) + "/model/" + name
                ):
                    raise _invalid()
            paths.append(item["path"])
            total += item["size_bytes"]
            if total > self.maximum_upload_bytes:
                raise _invalid()
            if verify_content:
                digest, size = hashlib.sha256(), 0
                for chunk in self._file(oid, item["path"], token, item["size_bytes"]):
                    size += len(chunk)
                    digest.update(chunk)
                if (size, digest.hexdigest()) != (item["size_bytes"], item["sha256"]):
                    raise _invalid()
        if len(paths) != len(set(paths)) or body["rendered"] != sorted(
            body["rendered"], key=lambda x: x["path"]
        ):
            raise _invalid()
        if verify_content:
            self._inventory(command, oid, token, body, len(raw))
            self._verified_snapshots.add((command.command_digest, oid))
        return body

    def publish_once(self, command, source, ownership):
        self._command(command)
        if (
            type(source) is not MaterializedSourceV1
            or source.inventory != command.source_inventory
            or source.source_identity_digest != command.source_identity_digest
            or type(ownership) is not TransferOwnershipV1
            or self.spool is None
            or self.root is None
            or (
                ownership.publication_id,
                ownership.command_digest,
                ownership.mutation_id,
                ownership.claim_digest,
            )
            != (
                command.publication_id,
                command.command_digest,
                command.mutation_id,
                PublicationRecordV1.claim(command, self.clock()).claim_digest,
            )
        ):
            raise _invalid()
        from importlib.metadata import version

        if version("huggingface_hub") != "1.33.0":
            raise _invalid()
        from tuner.inference.retrieved_model import _materialize_admitted_sft_model
        from tuner.execution.providers.hf_jobs_provider_v1.transport import _bounded
        from huggingface_hub import HfApi, CommitOperationAdd

        artifacts = {x.artifact.role: x for x in source.artifacts}

        def read_artifact(run, artifact):
            leaf = self.spool.retained(command.publication_id, artifacts[artifact.role])

            class Stream:
                def iter_bytes(inner):
                    return leaf.chunks(artifact.size_bytes)

            result = Stream()
            result.run, result.artifact = run, artifact
            return result

        retrieved = _materialize_admitted_sft_model(
            run=command.run,
            artifacts=command.source_inventory,
            root=self.root.path,
            root_fd=self.root.fd,
            read_artifact=read_artifact,
            maximum_artifact_bytes=command.maximum_artifact_bytes,
            maximum_total_bytes=command.maximum_total_bytes,
        )
        retrieved.validate()
        tree = _Tree(self.root, retrieved.attempt)
        files = []
        try:
            native, rendered = [], []
            for artifact in command.source_inventory:
                leaf = _Leaf(tree, artifact.role + ".tar", artifact.size_bytes)
                files.append(leaf)
                leaf.verify(artifact)
                native.append(
                    {
                        "role": artifact.role,
                        "path": self._prefix(command)
                        + "/native/"
                        + artifact.role
                        + ".bin",
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                    }
                )
            for kind, members in (
                ("model", retrieved.model_files),
                ("tokenizer", retrieved.tokenizer_files),
            ):
                for item in members:
                    leaf = _Leaf(tree, kind + "/" + item.name, item.size_bytes)
                    files.append(leaf)
                    leaf.verify(item)
                    rendered.append(
                        {
                            "kind": kind,
                            "path": self._prefix(command) + "/model/" + item.name,
                            "sha256": item.sha256,
                            "size_bytes": item.size_bytes,
                        }
                    )
            paired = list(zip(native + rendered, files))
            rendered.sort(key=lambda x: x["path"])
            body = {
                "schema_version": "synaptic-hf-model-publication-marker/v1",
                "command": asdict(command),
                "ownership": asdict(ownership),
                "destination": self._destination(),
                "native": native,
                "rendered": rendered,
                "model": {
                    "kind": retrieved.model_kind,
                    "model_ref": retrieved.model_ref,
                    "model_revision": retrieved.model_revision,
                    "tokenizer_revision": retrieved.tokenizer_revision,
                },
                "recorded_at": self.clock(),
                "authority_ref": self.descriptor.authority_ref,
                "key_ref": self.authority.key_ref,
            }
            raw = canonical_bytes(
                {**body, "tag": self.authority.sign(_PURPOSE, canonical_bytes(body))}
            )
            if (
                len(raw) > _MARKER_MAX
                or len(raw) + sum(x["size_bytes"] for x in native + rendered)
                > self.maximum_upload_bytes
            ):
                raise _invalid()
            token = self._token()
            if self._head(token) != self.parent_commit:
                raise _invalid()
            existing = self._metadata(
                "/api/models/" + self.repo_id + "/paths-info/" + self.parent_commit,
                token,
                method="POST",
                body={"paths": [self._prefix(command)], "expand": False},
            )
            if existing != []:
                raise _invalid()
            self._guard()
            # Duplicates belong to the SDK operation and remain open on caller timeout.
            handles = []
            try:
                for item, leaf in paired:
                    leaf.check()
                    handle = os.fdopen(os.dup(leaf.fd), "rb")
                    handle.seek(0)
                    handles.append((item["path"], handle))
                self._guard()
                scope = self._destination().copy()
                prefix = self._prefix(command)
                provider_deadline = self.deadline
                commit_bound = self.maximum_commit_bytes
                api_factory, http_factory = self.api_factory, self.http_factory
                handoff = Lock()
                custody = {"entered": False, "abandoned": False}

                def commit():
                    with handoff:
                        if custody["abandoned"]:
                            raise _invalid()
                        custody["entered"] = True
                    try:
                        api = (
                            api_factory(endpoint=scope["endpoint"], token=token)
                            if api_factory
                            else HfApi(endpoint=scope["endpoint"], token=token)
                        )
                        operations = [
                            CommitOperationAdd(
                                path_in_repo=path, path_or_fileobj=handle
                            )
                            for path, handle in handles
                        ]
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=prefix + "/publication.json",
                                path_or_fileobj=raw,
                            )
                        )
                        from importlib.metadata import version
                        from huggingface_hub._commit_api import _prepare_commit_payload
                        import httpx

                        if version("huggingface_hub") != "1.33.0":
                            raise _invalid()
                        api.preupload_lfs_files(
                            scope["repo_id"],
                            operations,
                            token=token,
                            repo_type="model",
                            revision=scope["branch"],
                            create_pr=False,
                            num_threads=1,
                            free_memory=False,
                        )
                        estimate = 0
                        for operation in operations:
                            if (
                                operation._should_ignore
                                or operation._upload_mode not in ("regular", "lfs")
                            ):
                                raise _invalid()
                            estimate += 512 + len(operation.path_in_repo.encode())
                            if operation._upload_mode == "regular":
                                estimate += 4 * ((operation.upload_info.size + 2) // 3)
                        if estimate > commit_bound:
                            raise _invalid()
                        chunks = []
                        total = 0
                        for item in _prepare_commit_payload(
                            operations,
                            {},
                            "Publication " + command.mutation_id,
                            parent_commit=scope["parent_commit"],
                        ):
                            encoded = json.dumps(item, allow_nan=False).encode() + b"\n"
                            total += len(encoded)
                            if total > commit_bound:
                                raise _invalid()
                            chunks.append(encoded)
                        content = b"".join(chunks)
                        if time.monotonic() >= provider_deadline:
                            raise _invalid()
                        kwargs = {
                            "timeout": min(
                                30, max(0.001, provider_deadline - time.monotonic())
                            ),
                            "follow_redirects": False,
                            "trust_env": False,
                        }
                        client = (
                            http_factory(**kwargs)
                            if http_factory
                            else httpx.Client(
                                transport=httpx.HTTPTransport(
                                    retries=0, trust_env=False
                                ),
                                **kwargs,
                            )
                        )
                        with client:
                            with client.stream(
                                "POST",
                                scope["endpoint"]
                                + "/api/models/"
                                + scope["repo_id"]
                                + "/commit/"
                                + scope["branch"],
                                content=content,
                                headers={
                                    "Authorization": "Bearer " + token,
                                    "Accept-Encoding": "identity",
                                    "Content-Type": "application/x-ndjson",
                                },
                            ) as response:
                                if (
                                    response.status_code not in (200, 201)
                                    or response.headers.get(
                                        "Content-Encoding", "identity"
                                    ).lower()
                                    != "identity"
                                ):
                                    raise _invalid()
                                result = bytearray()
                                for chunk in response.iter_raw(chunk_size=65536):
                                    if (
                                        time.monotonic() >= provider_deadline
                                        or len(result) + len(chunk) > _MARKER_MAX
                                    ):
                                        raise _invalid()
                                    result.extend(chunk)
                                document = _json(result)
                        if (
                            type(document) is not dict
                            or type(document.get("commitOid")) is not str
                        ):
                            raise _invalid()
                        return document["commitOid"]
                    finally:
                        for _, handle in handles:
                            handle.close()

                dispatched = True
                response = _bounded(commit, self.deadline)
            except BaseException:
                if not locals().get("dispatched", False):
                    for _, handle in handles:
                        handle.close()
                else:
                    with handoff:
                        if not custody["entered"]:
                            custody["abandoned"] = True
                            for _, handle in handles:
                                handle.close()
                raise
            oid = response
            if type(oid) is not str or not _OID.fullmatch(oid):
                raise _invalid()
            for leaf in files:
                leaf.check()
            retrieved.validate()
            verified = self._verified_marker(
                command, oid, token, ownership_id=ownership.ownership_id
            )
            return self._receipt(command, verified, oid)
        finally:
            for leaf in files:
                leaf.close()
            tree.close()

    def lookup(self, command, permit):
        self._command(command)
        if type(permit) is not LookupRecoveryPermitV1 or (
            permit.publication_id,
            permit.command_digest,
            permit.mutation_id,
            permit.claim_digest,
        ) != (
            command.publication_id,
            command.command_digest,
            command.mutation_id,
            PublicationRecordV1.claim(command, self.clock()).claim_digest,
        ):
            raise _invalid()
        receipt = None
        try:
            token = self._token()
            oid = self._head(token)
            body = self._verified_marker(
                command, oid, token, ownership_id=permit.fenced_ownership_id
            )
            receipt = self._receipt(command, body, oid)
        except Exception:
            pass  # Missing/changed content never authorizes another commit.
        value = AuthenticatedLookupV1(
            "synaptic-publication-lookup/v1",
            LookupOutcomeV1.FOUND if receipt else LookupOutcomeV1.INDETERMINATE,
            command.publication_id,
            command.command_digest,
            command.destination_identity_digest,
            command.mutation_id,
            permit.fenced_ownership_id,
            permit.permit_id,
            hashlib.sha256(
                canonical_bytes(
                    {
                        "command": command.command_digest,
                        "receipt": None
                        if receipt is None
                        else hashlib.sha256(receipt.canonical_bytes).hexdigest(),
                    }
                )
            ).hexdigest(),
            self.clock(),
            None,
            receipt,
            self.descriptor.authority_ref,
            self.authority.key_ref,
            "0" * 64,
        )
        return self._signed(value, "publication-lookup/v1")

    def iter_bytes(self, command, artifact, maximum_bytes):
        self._command(command)
        if (
            type(artifact) is not DestinationArtifactV1
            or type(maximum_bytes) is not int
            or not 1 <= maximum_bytes <= command.maximum_artifact_bytes
        ):
            raise _invalid()
        oid, path = artifact.path.split("/", 1)
        token = self._token()
        body = self._verified_marker(
            command,
            oid,
            token,
            verify_content=(command.command_digest, oid)
            not in self._verified_snapshots,
        )
        if {
            "role": artifact.role,
            "path": path,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        } not in body["native"] or artifact.size_bytes > maximum_bytes:
            raise _invalid()
        yield from self._file(oid, path, token, artifact.size_bytes)

    def close(self):
        if get_ident() != self.thread or os.getpid() != self.pid:
            raise _invalid()
        if self.root is not None:
            self.root.close()
        self.closed = True
