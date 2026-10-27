"""Host-owned lifecycle journal custody and retention through existing ports.

No database, background store thread, signal authority, or execution admission.
The caller supplies independently bound launch material and its host authority.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat

from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_lifecycle_protocol import MAXIMUM_RECORD_BYTES, NativeLifecyclePolicyV1, verify_lifecycle_journal
from tuner.runtime.native_mlx_material import canonical, strict_json


@dataclass(frozen=True)
class NativeLifecycleLaunchV1:
    policy: NativeLifecyclePolicyV1
    protocol_sha256: str

    def __post_init__(self):
        if (type(self.policy) is not NativeLifecyclePolicyV1 or type(self.protocol_sha256) is not str
                or len(self.protocol_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.protocol_sha256)):
            raise ValueError("native lifecycle launch selection is invalid")

    def to_dict(self):
        return {"schema": "synaptic-native-lifecycle-launch/v1", "policy": self.policy.to_dict(),
                "protocol_sha256": self.protocol_sha256}

    def validate_source(self, invocation, request):
        document = strict_json(request)
        if (type(document) is not dict or canonical(document.get("lifecycle")) != canonical(self.to_dict())
                or self.policy.maximum_journal_bytes > invocation.limits.maximum_file_bytes):
            raise ValueError("native lifecycle request selection differs")
        path = Path(invocation.guardian).with_name("native_lifecycle_protocol.py")
        with NativeTree(path.parent) as tree:
            raw = tree.read(path.name, maximum=65536)
        if hashlib.sha256(raw).hexdigest() != self.protocol_sha256:
            raise ValueError("native lifecycle protocol source differs")


class NativeLifecycleJournal:
    def __init__(self, records, streams, authority, launch):
        if type(launch) is not NativeLifecycleLaunchV1:
            raise TypeError("exact native lifecycle launch selection required")
        for port, methods in ((streams, ("append", "read_page")), (authority, ("lifecycle_key",))):
            if any(not callable(getattr(port, name, None)) for name in methods):
                raise TypeError("native lifecycle host port is absent")
        self.records, self.streams, self.authority, self.launch = records, streams, authority, launch
        self._prefixes = {}

    def key(self, invocation):
        key = self.authority.lifecycle_key(invocation.digest, self.launch.policy.digest)
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("native lifecycle derived capability is invalid")
        return key

    def _member(self, invocation):
        return "lifecycle-" + invocation.digest + ".jsonl"

    def create(self, invocation, root):
        root.check()
        fd = os.open(self._member(invocation), os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=root.fd)
        try:
            info = os.fstat(fd)
            if stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError("native lifecycle journal private mode differs")
            self.records._retain(invocation, "lifecycle", {
                "launch": self.launch.to_dict(), "member": self._member(invocation),
                "device": info.st_dev, "inode": info.st_ino,
            })
            os.fsync(root.fd)
            root.check()
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _metadata(self, invocation):
        stored = self.records.records.read(partition="effects", key=self.records._key(invocation.attempt, "lifecycle"))
        if stored is None:
            return None
        document = strict_json(stored.canonical)
        value = document.get("value")
        if (document != {"schema": "synaptic-native-process-record/v1", "stage": "lifecycle",
                         "invocation_digest": invocation.digest, "binding_digest": invocation.binding_digest,
                         "value": value}
                or type(value) is not dict or set(value) != {"launch", "member", "device", "inode"}
                or canonical(value["launch"]) != canonical(self.launch.to_dict()) or value["member"] != self._member(invocation)
                or any(type(value[name]) is not int or value[name] < 0 for name in ("device", "inode"))):
            raise ValueError("native lifecycle journal binding differs")
        return value

    def _verify(self, invocation, raw, *, previous_raw=None):
        return verify_lifecycle_journal(raw, invocation_digest=invocation.digest,
                                       policy=self.launch.policy, key=self.key(invocation), previous_raw=previous_raw)

    def _retained(self, invocation):
        cached = self._prefixes.get(invocation.digest)
        chunks = [] if cached is None else cached.splitlines(keepends=True)
        after = len(chunks) - 1 if chunks else None
        maximum = self.launch.policy.maximum_journal_bytes
        size = 0 if cached is None else len(cached)
        for _ in range((self.launch.policy.maximum_progress_records + 1) // 1000 + 2):
            page = self.streams.read_page(partition="effects", stream_key="native-lifecycle/" + invocation.digest,
                                          after_sequence=after, limit=1000)
            for entry in page.entries:
                if (entry.sequence != len(chunks) or len(chunks) > self.launch.policy.maximum_progress_records
                        or not 0 < len(entry.canonical) < MAXIMUM_RECORD_BYTES
                        or size + len(entry.canonical) + 1 > maximum):
                    raise ValueError("native lifecycle retained sequence differs")
                raw = entry.canonical + b"\n"
                size += len(raw)
                chunks.append(raw)
            if not page.truncated:
                raw = b"".join(chunks)
                self._verify(invocation, raw)
                self._prefixes = {invocation.digest: raw}
                return raw
            if not page.entries or page.next_cursor != page.entries[-1].sequence:
                raise ValueError("native lifecycle retained cursor differs")
            after = page.next_cursor
        raise ValueError("native lifecycle retained page limit")

    def _capture(self, invocation, metadata, root, fd):
        root.check()
        before = os.fstat(fd)
        def check(info):
            if (not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (metadata["device"], metadata["inode"])
                    or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                    or not 0 <= info.st_size <= self.launch.policy.maximum_journal_bytes):
                raise ValueError("native lifecycle journal custody differs")
        check(before)
        chunks, offset = [], 0
        while offset < before.st_size:
            chunk = os.pread(fd, min(1 << 20, before.st_size - offset), offset)
            if not chunk:
                raise ValueError("native lifecycle journal shortened")
            chunks.append(chunk)
            offset += len(chunk)
        after = os.fstat(fd)
        current = os.stat(metadata["member"], dir_fd=root.fd, follow_symlinks=False)
        check(after)
        check(current)
        if after.st_size < before.st_size or current.st_size < before.st_size:
            raise ValueError("native lifecycle journal shortened")
        root.check()
        return b"".join(chunks)

    def snapshot(self, invocation, *, root=None, fd=None):
        metadata = self._metadata(invocation)
        if metadata is None:
            # A claim may survive interruption before journal creation. It
            # grants neither terminal/absence proof nor another start.
            return self._verify(invocation, b"")
        previous = self._retained(invocation)
        retained = self._verify(invocation, previous)
        # An already retained authenticated terminal needs no mutable file/source
        # to establish the same outcome. Artifact verification remains separate.
        if retained.terminal_bytes is not None:
            return retained
        if root is None:
            with NativeTree(Path(invocation.private_root), private=True) as opened:
                try:
                    descriptor = os.open(metadata["member"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                         dir_fd=opened.fd)
                except FileNotFoundError:
                    return retained  # Missing publication is uncertainty, never absence proof.
                try:
                    raw = self._capture(invocation, metadata, opened, descriptor)
                finally:
                    os.close(descriptor)
        else:
            raw = self._capture(invocation, metadata, root, fd)
        view = self._verify(invocation, raw, previous_raw=previous)
        frames = raw[:view.committed_bytes].splitlines(keepends=True)
        first = previous.count(b"\n")
        stream_key = "native-lifecycle/" + invocation.digest
        for sequence in range(first, len(frames)):
            frame = frames[sequence][:-1]
            # A false or lost acknowledgement is not a process replay. Check only
            # the exact append slot; identical retained bytes may be read back.
            self.streams.append(partition="effects", stream_key=stream_key, sequence=sequence, canonical=frame)
            page = self.streams.read_page(partition="effects", stream_key=stream_key,
                                          after_sequence=None if sequence == 0 else sequence - 1, limit=1)
            if len(page.entries) != 1 or page.entries[0].sequence != sequence or page.entries[0].canonical != frame:
                raise ValueError("native lifecycle retained frame readback differs")
        self._prefixes = {invocation.digest: raw[:view.committed_bytes]}
        return view
