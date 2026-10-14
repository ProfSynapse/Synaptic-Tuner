"""Private local disk spool and create-only publication destination.

Caller owns local POSIX roots, finite limits, role mappings and evidence keys.
Filesystem deadlines are cooperative, not syscall termination. Partial files are
retained; no adapter method overwrites, adopts, retries or deletes a publication.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import stat
import threading
import time

from synaptic_tuner.api.v1.reference.authority import (
    ReferencePublicationEvidenceAuthorityV1,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    safe_ref,
)
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
    MAX_CHUNK_BYTES,
)

_MARKER = "publication-receipt.json"
_MAX_MARKER = 1 << 20


def _invalid():
    return ValueError("local_publication_unavailable")


def _id(info):
    return info.st_dev, info.st_ino


def _file_id(info):
    return (
        *_id(info),
        info.st_mode,
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _parts(path):
    if type(path) is not str or not 1 <= len(path.encode()) <= 256:
        raise _invalid()
    p = PurePosixPath(path)
    if (
        not p.parts
        or p.is_absolute()
        or p.as_posix() != path
        or "\\" in path
        or len(p.parts) > 32
    ):
        raise _invalid()
    if any(
        x in {".", ".."} or not x or any(ord(c) < 32 or ord(c) == 127 for c in x)
        for x in p.parts
    ):
        raise _invalid()
    return p.parts


class _Root:
    def __init__(self, path, deadline):
        if os.name != "posix" or os.open not in os.supports_dir_fd:
            raise _invalid()
        if (
            type(deadline) not in (float, int)
            or abs(deadline) > 1e100
            or not math.isfinite(deadline)
            or deadline <= time.monotonic()
        ):
            raise _invalid()
        path = Path(path)
        if (
            not path.is_absolute()
            or path != Path(os.path.normpath(path))
            or len(path.parts) > 64
        ):
            raise _invalid()
        self.path, self.deadline = path, float(deadline)
        self.thread, self.pid = threading.get_ident(), os.getpid()
        self.anchors, self.closed = [], False
        try:
            fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self.anchors.append((fd, None, None))
            for part in path.parts[1:]:
                parent = fd
                fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                )
                self.anchors.append((fd, parent, part))
            self.fd = fd
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        if (
            self.closed
            or self.thread != threading.get_ident()
            or self.pid != os.getpid()
        ):
            raise _invalid()
        if time.monotonic() >= self.deadline:
            raise _invalid()
        for fd, parent, name in self.anchors:
            info = os.fstat(fd)
            if (
                not stat.S_ISDIR(info.st_mode)
                or parent is not None
                and _id(os.stat(name, dir_fd=parent, follow_symlinks=False))
                != _id(info)
            ):
                raise _invalid()
        self.private_dir(self.fd)

    @staticmethod
    def private_dir(fd):
        info = os.fstat(fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise _invalid()

    def close(self):
        if self.thread != threading.get_ident() or self.pid != os.getpid():
            raise _invalid()
        for fd, _, _ in reversed(self.anchors):
            os.close(fd)
        self.anchors.clear()
        self.closed = True


class _Tree:
    """Retain every selected directory through reads and publication signing."""

    def __init__(self, root, name, *, create=False):
        root.check()
        self.root, self.fds, self.names = root, [], {}
        self.base = self._directory(root.fd, name, create)
        self.names[()] = self.base

    def _directory(self, parent, name, create):
        if create:
            os.mkdir(name, 0o700, dir_fd=parent)
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            _Root.private_dir(fd)
            if _id(before) != _id(os.fstat(fd)):
                raise _invalid()
        except BaseException:
            os.close(fd)
            raise
        self.fds.append((fd, parent, name))
        return fd

    def parent(self, parts, *, create=False):
        prefix = ()
        fd = self.base
        for part in parts[:-1]:
            prefix += (part,)
            if prefix not in self.names:
                self.names[prefix] = self._directory(fd, part, create)
            fd = self.names[prefix]
        return fd

    def check(self):
        self.root.check()
        for fd, parent, name in self.fds:
            _Root.private_dir(fd)
            if _id(os.stat(name, dir_fd=parent, follow_symlinks=False)) != _id(
                os.fstat(fd)
            ):
                raise _invalid()

    def close(self):
        for fd, _, _ in reversed(self.fds):
            os.close(fd)
        self.fds.clear()

    def inventory(self, files):
        expected = {(): set()}
        for path in files:
            parts = _parts(path)
            prefix = ()
            for part in parts:
                expected.setdefault(prefix, set()).add(part)
                prefix += (part,)
            for n in range(1, len(parts)):
                expected.setdefault(parts[:n], set())
            self.parent(parts)
        for path, names in expected.items():
            fd = self.names[path]
            cursor = os.open(
                ".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            try:
                found = set()
                with os.scandir(cursor) as entries:
                    for entry in entries:
                        if len(found) >= len(names) or entry.name not in names:
                            raise _invalid()
                        found.add(entry.name)
                if found != names:
                    raise _invalid()
            finally:
                os.close(cursor)
        self.check()


class _Leaf:
    def __init__(self, tree, path, maximum, *, create=False):
        parts = _parts(path)
        parent = tree.parent(parts, create=create)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL if create else os.O_RDONLY
        fd = os.open(
            parts[-1], flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent
        )
        self.tree, self.parent, self.name, self.fd = tree, parent, parts[-1], fd
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size > maximum
            ):
                raise _invalid()
            self.identity = _file_id(info)
            self.check()
        except BaseException:
            os.close(fd)
            raise

    def check(self):
        self.tree.check()
        if (
            _file_id(os.fstat(self.fd)) != self.identity
            or _file_id(os.stat(self.name, dir_fd=self.parent, follow_symlinks=False))
            != self.identity
        ):
            raise _invalid()

    def refresh_written(self):
        current = _file_id(os.fstat(self.fd))
        if current[:5] != self.identity[:5]:
            raise _invalid()
        self.identity = current
        self.check()

    def chunks(self, maximum):
        self.check()
        os.lseek(self.fd, 0, os.SEEK_SET)
        total = 0
        while True:
            self.check()
            chunk = os.read(self.fd, min(MAX_CHUNK_BYTES, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise _invalid()
            yield chunk
        self.check()

    def verify(self, artifact):
        digest, total = hashlib.sha256(), 0
        for chunk in self.chunks(artifact.size_bytes):
            digest.update(chunk)
            total += len(chunk)
        if total != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
            raise _invalid()

    def close(self):
        os.close(self.fd)


def _write(fd, chunk):
    view = memoryview(chunk)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise _invalid()
        view = view[count:]


class _Sink:
    def __init__(self, spool, publication_id, role, maximum):
        self.spool, self.publication_id, self.role, self.maximum = (
            spool,
            publication_id,
            role,
            maximum,
        )
        self.size, self.finished, self.aborted = 0, False, False
        self.name = hashlib.sha256(role.encode()).hexdigest()
        self.tree = spool._trees[publication_id]
        self.leaf = _Leaf(self.tree, self.name, maximum, create=True)

    def write(self, chunk):
        if (
            self.finished
            or self.aborted
            or type(chunk) is not bytes
            or not 0 < len(chunk) <= MAX_CHUNK_BYTES
        ):
            raise _invalid()
        self.leaf.check()
        total = self.spool._total_bytes
        if (
            self.size + len(chunk) > self.maximum
            or total + len(chunk) > self.spool.maximum_total_bytes
        ):
            raise _invalid()
        _write(self.leaf.fd, chunk)
        self.size += len(chunk)
        self.spool._total_bytes += len(chunk)
        self.leaf.refresh_written()

    def finish(self):
        if self.finished or self.aborted:
            raise _invalid()
        self.leaf.check()
        os.fsync(self.leaf.fd)
        os.fsync(self.tree.base)
        self.leaf.check()
        self.finished = True
        return "spool/" + self.publication_id + "/" + self.name

    def abort(self):
        # Partial bytes remain private and consumed, never silently reused.
        self.aborted = True


class LocalArtifactSpoolV1:
    """Private spool with total bytes/100 files bounded across its entire lifetime.

    Per-file bytes are independently bounded; partial/aborted bytes still count.
    Deadline is an absolute monotonic instant and is never reset.
    """

    def __init__(self, root, *, maximum_artifact_bytes, maximum_total_bytes, deadline):
        if (
            any(
                type(x) is not int or not 1 <= x <= 2**63 - 1
                for x in (maximum_artifact_bytes, maximum_total_bytes)
            )
            or maximum_artifact_bytes > maximum_total_bytes
        ):
            raise _invalid()
        self.root = _Root(root, deadline)
        self.maximum_artifact_bytes, self.maximum_total_bytes = (
            maximum_artifact_bytes,
            maximum_total_bytes,
        )
        self._trees, self._sinks, self._total_bytes = {}, {}, 0

    def open(self, publication_id, role, maximum_bytes):
        self.root.check()
        digest_text(publication_id, "publication_id")
        safe_ref(role, "role")
        if (
            type(maximum_bytes) is not int
            or not 0 <= maximum_bytes <= self.maximum_artifact_bytes
            or (publication_id, role) in self._sinks
        ):
            raise _invalid()
        if publication_id not in self._trees:
            self._trees[publication_id] = _Tree(self.root, publication_id, create=True)
        if len(self._sinks) >= 100:
            raise _invalid()
        sink = _Sink(self, publication_id, role, maximum_bytes)
        self._sinks[publication_id, role] = sink
        return sink

    def retained(self, publication_id, spooled):
        sink = self._sinks.get((publication_id, spooled.artifact.role))
        if (
            sink is None
            or not sink.finished
            or sink.aborted
            or spooled.spool_ref != "spool/" + publication_id + "/" + sink.name
        ):
            raise _invalid()
        sink.leaf.verify(spooled.artifact)
        return sink.leaf

    def close(self):
        if self.root.thread != threading.get_ident() or self.root.pid != os.getpid():
            raise _invalid()
        for sink in self._sinks.values():
            sink.leaf.close()
        self._sinks.clear()
        for tree in self._trees.values():
            tree.close()
        self._trees.clear()
        self.root.close()


class LocalPublicationDestinationV1:
    """Exclusive local publication plus read-only original-marker reconciliation."""

    def __init__(
        self,
        root,
        *,
        destination_ref,
        display_name,
        role_paths,
        maximum_artifact_bytes,
        maximum_total_bytes,
        authority_ref,
        authority,
        clock,
        deadline,
        spool=None,
    ):
        if (
            type(authority) is not ReferencePublicationEvidenceAuthorityV1
            or not callable(clock)
            or type(role_paths) is not dict
            or not 1 <= len(role_paths) <= 100
            or any(
                type(x) is not int or not 1 <= x <= 2**63 - 1
                for x in (maximum_artifact_bytes, maximum_total_bytes)
            )
            or maximum_artifact_bytes > maximum_total_bytes
        ):
            raise _invalid()
        paths = {}
        for role, path in role_paths.items():
            safe_ref(role, "role")
            parts = _parts(path)
            if parts[0] == _MARKER:
                raise _invalid()
            paths[role] = path
        values = list(paths.values())
        if len(set(values)) != len(values) or any(
            a.startswith(b + "/") for a in values for b in values if a != b
        ):
            raise _invalid()
        self.root, self.spool = _Root(root, deadline), spool
        if spool is not None and (
            self.root.path == spool.root.path
            or self.root.path in spool.root.path.parents
            or spool.root.path in self.root.path.parents
        ):
            self.root.close()
            raise _invalid()
        try:
            self.authority, self.clock, self.role_paths = authority, clock, paths
            info = os.fstat(self.root.fd)
            config = canonical_bytes(
                {
                    "root": str(self.root.path),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "roles": paths,
                }
            )
            descriptor = AuthenticatedDestinationV1(
                "synaptic-publication-destination/v1",
                destination_ref,
                display_name,
                hashlib.sha256(config).hexdigest(),
                hashlib.sha256(b"create-only-local-publication/v1").hexdigest(),
                maximum_artifact_bytes,
                maximum_total_bytes,
                authority_ref,
                authority.key_ref,
                "0" * 64,
            )
            self.descriptor = self._signed(descriptor, "publication-destination/v1")
        except BaseException:
            self.root.close()
            raise

    def _signed(self, value, purpose):
        self.root.check()
        tag = self.authority.sign(purpose, value.payload)
        self.root.check()
        return replace(value, tag=tag)

    def _command(self, command):
        self.root.check()
        if (
            type(command) is not PublicationCommandV1
            or not PublicationOperationsV1._descriptor_matches_command(
                self.descriptor, command
            )
        ):
            raise _invalid()
        if any(item.role not in self.role_paths for item in command.source_inventory):
            raise _invalid()

    def _receipt(self, command, ownership, inventory):
        stamp = self.clock()
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
        receipt = AuthenticatedPublicationReceiptV1(
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
        return self._signed(receipt, "publication-receipt/v1")

    def publish_once(self, command, source, ownership):
        self._command(command)
        if (
            type(source) is not MaterializedSourceV1
            or source.inventory != command.source_inventory
            or source.source_identity_digest != command.source_identity_digest
            or type(ownership) is not TransferOwnershipV1
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
            or self.spool is None
        ):
            raise _invalid()
        tree = _Tree(self.root, command.publication_id, create=True)
        leaves = []
        try:
            entries = []
            for spooled in source.artifacts:
                original = self.spool.retained(command.publication_id, spooled)
                target = _Leaf(
                    tree,
                    self.role_paths[spooled.artifact.role],
                    spooled.artifact.size_bytes,
                    create=True,
                )
                leaves.append(target)
                for chunk in original.chunks(spooled.artifact.size_bytes):
                    self.root.check()
                    _write(target.fd, chunk)
                os.fsync(target.fd)
                target.refresh_written()
                target.verify(spooled.artifact)
                original.verify(spooled.artifact)
                entries.append(
                    DestinationArtifactV1(
                        spooled.artifact.role,
                        self.role_paths[spooled.artifact.role],
                        spooled.artifact.sha256,
                        spooled.artifact.size_bytes,
                    )
                )
            tree.inventory([item.path for item in entries])
            receipt = self._receipt(
                command, ownership, DestinationInventoryV1(tuple(entries))
            )
            raw = receipt.canonical_bytes
            if len(raw) > _MAX_MARKER:
                raise _invalid()
            marker = _Leaf(tree, _MARKER, _MAX_MARKER, create=True)
            leaves.append(marker)
            _write(marker.fd, raw)
            os.fsync(marker.fd)
            marker.refresh_written()
            for fd, _, _ in reversed(tree.fds):
                os.fsync(fd)
            os.fsync(self.root.fd)
            tree.inventory([item.path for item in entries] + [_MARKER])
            for leaf, artifact in zip(leaves, command.source_inventory):
                leaf.verify(artifact)
            marker.check()
            self.root.check()
            return receipt
        finally:
            for leaf in leaves:
                leaf.close()
            tree.close()

    @contextmanager
    def _original(self, command):
        self._command(command)
        tree = _Tree(self.root, command.publication_id)
        marker = None
        try:
            marker = _Leaf(tree, _MARKER, _MAX_MARKER)
            raw = b"".join(marker.chunks(_MAX_MARKER))
            receipt = AuthenticatedPublicationReceiptV1.from_canonical_bytes(raw)
            if (
                receipt.command_digest != command.command_digest
                or receipt.run != command.run
                or receipt.source_identity_digest != command.source_identity_digest
                or receipt.destination_ref != command.destination_ref
                or receipt.mutation_id != command.mutation_id
                or receipt.claim_digest
                != PublicationRecordV1.claim(command, receipt.recorded_at).claim_digest
                or receipt.publication_id != command.publication_id
                or receipt.destination_identity_digest
                != self.descriptor.identity_digest
                or receipt.authority_ref != self.descriptor.authority_ref
                or receipt.key_ref != self.authority.key_ref
                or not self.authority.verify(
                    "publication-receipt/v1",
                    receipt.payload,
                    receipt.tag,
                    receipt.key_ref,
                )
                or not self.authority.verify(
                    "publication-destination-inventory/v1",
                    receipt.inventory.payload,
                    receipt.inventory.tag,
                    receipt.inventory.key_ref,
                )
            ):
                raise _invalid()
            entries = receipt.inventory.inventory.artifacts
            if tuple(
                (e.role, e.sha256, e.size_bytes, e.path) for e in entries
            ) != tuple(
                (a.role, a.sha256, a.size_bytes, self.role_paths[a.role])
                for a in command.source_inventory
            ):
                raise _invalid()
            tree.inventory([entry.path for entry in entries] + [_MARKER])
            yield tree, marker, receipt
            marker.check()
            tree.check()
        finally:
            if marker is not None:
                marker.close()
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

        def result(receipt):
            outcome = AuthenticatedLookupV1(
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
                            "marker": None
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
            return self._signed(outcome, "publication-lookup/v1")

        try:
            with self._original(command) as (tree, marker, candidate):
                if candidate.ownership_id != permit.fenced_ownership_id:
                    raise _invalid()
                leaves = []
                try:
                    for expected in command.source_inventory:
                        leaf = _Leaf(
                            tree, self.role_paths[expected.role], expected.size_bytes
                        )
                        leaves.append(leaf)
                        leaf.verify(expected)
                    signed = result(candidate)
                    for leaf in leaves:
                        leaf.check()
                    marker.check()
                    return signed
                finally:
                    for leaf in leaves:
                        leaf.close()
        except (FileNotFoundError, ValueError, OSError):
            return result(None)

    def iter_bytes(self, command, artifact, maximum_bytes):
        self._command(command)
        if (
            type(artifact) is not DestinationArtifactV1
            or type(maximum_bytes) is not int
            or not 1 <= maximum_bytes <= self.descriptor.maximum_artifact_bytes
        ):
            raise _invalid()
        with self._original(command) as (tree, marker, receipt):
            if (
                artifact not in receipt.inventory.inventory.artifacts
                or artifact.size_bytes > maximum_bytes
            ):
                raise _invalid()
            leaf = _Leaf(tree, artifact.path, maximum_bytes)
            try:
                digest, total = hashlib.sha256(), 0
                for chunk in leaf.chunks(artifact.size_bytes):
                    digest.update(chunk)
                    total += len(chunk)
                    yield chunk
                if (
                    total != artifact.size_bytes
                    or digest.hexdigest() != artifact.sha256
                ):
                    raise _invalid()
                marker.check()
            finally:
                leaf.close()

    def close(self):
        self.root.close()


class LocalDestinationRegistryV1:
    def __init__(self, destinations):
        if (
            type(destinations) is not tuple
            or len(destinations) > 100
            or any(type(d) is not LocalPublicationDestinationV1 for d in destinations)
        ):
            raise _invalid()
        self.destinations = {d.descriptor.destination_ref: d for d in destinations}
        if len(self.destinations) != len(destinations):
            raise _invalid()

    def resolve(self, destination_ref):
        target = self.destinations[destination_ref]
        target.root.check()
        return target.descriptor, target

    def list(self, limit):
        if type(limit) is not int or limit != 101:
            raise _invalid()
        values = tuple(d.descriptor for _, d in sorted(self.destinations.items()))
        return values, True
