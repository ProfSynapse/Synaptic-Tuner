"""Durable publication records and local POSIX transfer ownership.

Caller owns the generic record store and private lock directory. Connections and
this adapter are thread/process affine. Locks prove live local ownership, not
remote termination or a multi-host lease. Closing releases ownership; recovery
then permits lookup only through the unchanged publication kernel.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import threading

from synaptic_tuner.api.v1.ports import StoragePartition
from tuner.execution.foundation_v2.canonical import digest_text, safe_ref
from tuner.execution.coordinator_v1.publication import (
    PublicationCodeV1,
    PublicationErrorV1,
    PublicationPhaseV1,
    PublicationRecordV1,
    PublicationTransitionKernelV1 as Kernel,
    TransferAdmissionV1,
    TransferDispositionV1,
    TransferOwnershipV1,
    LookupRecoveryPermitV1,
    MAX_PAGE,
)

_PARTITION = StoragePartition.EFFECTS.value
_PREFIX = "publication/v1/"
_ROOT_KEY = "publication-lock-root/v1"


def _conflict():
    return PublicationErrorV1(PublicationCodeV1.STATE_CONFLICT)


def _identity(info):
    return info.st_dev, info.st_ino


class DurablePublicationStoreV1:
    """Existing publication-store port over generic CAS and retained OS locks.

    All instances sharing records must use the same persisted lock-root identity.
    No lock file is unlinked: replacing it could split an active flock. The host
    must use a local filesystem with working POSIX flock/openat, not network FS.
    """

    def __init__(self, records, lock_root: Path, *, create: bool = True):
        if type(create) is not bool:
            raise TypeError("create must be exact bool")
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise _conflict()
        import fcntl

        for method in (
            "read",
            "create",
            "compare_and_swap",
            "put_if_absent",
            "list_page",
        ):
            if not callable(getattr(records, method, None)):
                raise TypeError("generic record store required")
        path = Path(lock_root)
        if (
            not path.is_absolute()
            or path != Path(os.path.normpath(path))
            or len(path.parts) > 64
        ):
            raise _conflict()
        self._records, self._fcntl, self._create = records, fcntl, create
        self._thread, self._pid = threading.get_ident(), os.getpid()
        self._anchors, self._owned, self._closed = [], {}, False
        try:
            fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._anchors.append((fd, None, None))
            for part in path.parts[1:]:
                parent = fd
                fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                )
                self._anchors.append((fd, parent, part))
            self._root = fd
            self._check()
            info = os.fstat(fd)
            raw = json.dumps(
                {
                    "schema_version": "synaptic-publication-lock-root/v1",
                    "path": str(path),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            if not create:
                existing = records.read(partition=_PARTITION, key=_ROOT_KEY)
                if existing is None or existing.canonical != raw:
                    raise _conflict()
                return
            created = records.put_if_absent(
                partition=_PARTITION, key=_ROOT_KEY, canonical=raw
            )
            if type(created) is not bool:
                raise _conflict()
            if not created:
                existing = records.read(partition=_PARTITION, key=_ROOT_KEY)
                if existing is None or existing.canonical != raw:
                    raise _conflict()
        except BaseException:
            self._close_fds()
            self._closed = True
            raise

    def _check(self):
        if (
            self._closed
            or self._pid != os.getpid()
            or self._thread != threading.get_ident()
        ):
            raise _conflict()
        for fd, parent, name in self._anchors:
            info = os.fstat(fd)
            if not stat.S_ISDIR(info.st_mode):
                raise _conflict()
            if parent is not None and _identity(
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            ) != _identity(info):
                raise _conflict()
        root = os.fstat(self._root)
        if root.st_uid != os.getuid() or stat.S_IMODE(root.st_mode) != 0o700:
            raise _conflict()
        for publication_id, (fd, _) in self._owned.items():
            self._check_leaf(publication_id, fd)

    def _leaf_binding(self, publication_id, fd):
        info = os.fstat(fd)
        return json.dumps(
            {
                "schema_version": "synaptic-publication-lock/v1",
                "publication_id": publication_id,
                "device": info.st_dev,
                "inode": info.st_ino,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def _check_leaf(self, publication_id, fd, *, bound=True):
        info = os.fstat(fd)
        named = os.stat(publication_id, dir_fd=self._root, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != 0
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or _identity(named) != _identity(info)
        ):
            raise _conflict()
        if bound:
            retained = self._records.read(
                partition=_PARTITION, key="publication-lock/v1/" + publication_id
            )
            if retained is None or retained.canonical != self._leaf_binding(
                publication_id, fd
            ):
                raise _conflict()

    def _acquire(self, publication_id):
        self._check()
        digest_text(publication_id, "publication_id")
        if publication_id in self._owned:
            return self._owned[publication_id][0], False
        fd = os.open(
            publication_id,
            os.O_RDWR
            | (os.O_CREAT if self._create else 0)
            | os.O_NOFOLLOW
            | os.O_NONBLOCK,
            0o600,
            dir_fd=self._root,
        )
        try:
            self._check_leaf(publication_id, fd, bound=False)
            key = "publication-lock/v1/" + publication_id
            retained = self._records.read(partition=_PARTITION, key=key)
            expected = self._leaf_binding(publication_id, fd)
            if retained is None:
                if (
                    not self._create
                    or self._records.read(
                        partition=_PARTITION, key=_PREFIX + publication_id
                    )
                    is not None
                ):
                    raise _conflict()
                created = self._records.put_if_absent(
                    partition=_PARTITION, key=key, canonical=expected
                )
                if type(created) is not bool or not created:
                    raise _conflict()
                os.fsync(self._root)
            elif retained.canonical != expected:
                raise _conflict()
            self._fcntl.flock(fd, self._fcntl.LOCK_EX | self._fcntl.LOCK_NB)
            self._check_leaf(publication_id, fd)
            return fd, True
        except BlockingIOError:
            os.close(fd)
            return None, False
        except BaseException:
            os.close(fd)
            raise

    def _release(self, publication_id):
        value = self._owned.pop(publication_id, None)
        if value is not None:
            os.close(value[0])

    @contextmanager
    def _temporary_lock(self, publication_id):
        fd, temporary = self._acquire(publication_id)
        try:
            yield fd
        finally:
            if temporary:
                os.close(fd)

    def _load(self, publication_id):
        self._check()
        digest_text(publication_id, "publication_id")
        stored = self._records.read(partition=_PARTITION, key=_PREFIX + publication_id)
        if stored is None:
            return None, None
        record = PublicationRecordV1.from_canonical_bytes(stored.canonical)
        if record.command.publication_id != publication_id:
            raise _conflict()
        return stored, record

    def _persist(self, stored, current, descendant):
        if descendant is current:
            return current
        raw = descendant.canonical_bytes
        updated = self._records.compare_and_swap(
            partition=_PARTITION,
            key=_PREFIX + descendant.command.publication_id,
            expected_revision=stored.revision,
            canonical=raw,
        )
        if type(updated) is not bool or not updated:
            raise _conflict()
        return descendant

    def claim(self, record):
        if (
            type(record) is not PublicationRecordV1
            or record.phase is not PublicationPhaseV1.CLAIMED
        ):
            raise TypeError("exact claimed record required")
        self._check()
        raw = record.canonical_bytes
        with self._temporary_lock(record.command.publication_id) as fd:
            if fd is None:
                return Kernel.claim(self.get(record.command.publication_id), record)
            created = self._records.create(
                partition=_PARTITION,
                key=_PREFIX + record.command.publication_id,
                canonical=raw,
            )
            if type(created) is not bool:
                raise _conflict()
            if created:
                return record, True
            return Kernel.claim(self.get(record.command.publication_id), record)

    def get(self, publication_id):
        return self._load(publication_id)[1]

    def compare_and_swap(self, expected_record_digest, descendant):
        if type(descendant) is not PublicationRecordV1:
            raise TypeError("exact publication record required")
        stored, current = self._load(descendant.command.publication_id)
        target, changed = Kernel.compare_and_swap(
            current, expected_record_digest, descendant
        )
        if not changed:
            return False
        updated = self._records.compare_and_swap(
            partition=_PARTITION,
            key=_PREFIX + descendant.command.publication_id,
            expected_revision=stored.revision,
            canonical=target.canonical_bytes,
        )
        if type(updated) is not bool:
            raise _conflict()
        return updated

    def begin_transfer(self, publication_id, expected_record_digest, timestamp):
        fd, fresh = self._acquire(publication_id)
        try:
            stored, current = self._load(publication_id)
            if fd is None:
                if current is None:
                    raise _conflict()
                return TransferAdmissionV1(TransferDispositionV1.ACTIVE, current, None)
            admission = Kernel.begin_transfer(
                current,
                publication_id,
                expected_record_digest,
                timestamp,
                1 if current is None else current.revision + 1,
            )
            self._persist(stored, current, admission.record)
            if admission.disposition is TransferDispositionV1.ACQUIRED:
                self._owned[publication_id] = (fd, admission.ownership)
                fresh = False
            return admission
        finally:
            if fresh:
                os.close(fd)

    def complete_transfer(self, ownership, receipt, verified, timestamp):
        if type(ownership) is not TransferOwnershipV1:
            raise TypeError("exact ownership required")
        self._check()
        stored, current = self._load(ownership.publication_id)
        active = self._owned.get(ownership.publication_id)
        descendant = Kernel.complete_transfer(
            current,
            ownership,
            receipt,
            verified,
            timestamp,
            ownership_active=active is not None and active[1] == ownership,
        )
        result = self._persist(stored, current, descendant)
        if verified:
            self._release(ownership.publication_id)
        return result

    def relinquish_uncertain(self, ownership, timestamp):
        if type(ownership) is not TransferOwnershipV1:
            raise TypeError("exact ownership required")
        self._check()
        stored, current = self._load(ownership.publication_id)
        active = self._owned.get(ownership.publication_id)
        decision = Kernel.relinquish_uncertain(
            current,
            ownership,
            timestamp,
            ownership_active=active is not None and active[1] == ownership,
        )
        self._persist(stored, current, decision.record)
        self._release(ownership.publication_id)
        return decision

    def recover_transfer(self, publication_id, command_digest, timestamp):
        with self._temporary_lock(publication_id) as fd:
            stored, current = self._load(publication_id)
            decision = Kernel.recover_transfer(
                current,
                publication_id,
                command_digest,
                timestamp,
                ownership_active=fd is None or publication_id in self._owned,
            )
            self._persist(stored, current, decision.record)
            return decision

    def finalize_recovery(
        self, permit, phase, timestamp, outcome, receipt=None, tombstone=None
    ):
        if type(permit) is not LookupRecoveryPermitV1:
            raise TypeError("exact recovery permit required")
        with self._temporary_lock(permit.publication_id) as fd:
            if fd is None or permit.publication_id in self._owned:
                raise _conflict()
            stored, current = self._load(permit.publication_id)
            descendant = Kernel.finalize_recovery(
                current,
                permit,
                phase,
                timestamp,
                outcome,
                receipt=receipt,
                tombstone=tombstone,
            )
            return self._persist(stored, current, descendant)

    def list(self, destination_ref, limit):
        self._check()
        safe_ref(destination_ref, "destination_ref")
        if type(limit) is not int or limit != MAX_PAGE + 1:
            raise ValueError("publication list probe limit must be 101")
        values, after = [], None
        # Scan finite pages; reject rather than silently omit a large catalog.
        for _ in range(100):
            page = self._records.list_page(
                partition=_PARTITION, prefix=_PREFIX, after_key=after, limit=100
            )
            for stored in page.records:
                record = PublicationRecordV1.from_canonical_bytes(stored.canonical)
                if stored.key != _PREFIX + record.command.publication_id:
                    raise _conflict()
                if record.command.destination_ref == destination_ref:
                    values.append(record)
                    if len(values) >= limit:
                        return tuple(values), False
            if not page.truncated:
                return tuple(values), len(values) <= MAX_PAGE
            after = page.next_cursor
        raise _conflict()

    def _close_fds(self):
        for fd, _ in self._owned.values():
            os.close(fd)
        self._owned.clear()
        for fd, _, _ in reversed(self._anchors):
            os.close(fd)
        self._anchors.clear()

    def close(self):
        if self._closed:
            return
        try:
            self._check()
        finally:
            self._close_fds()
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()
