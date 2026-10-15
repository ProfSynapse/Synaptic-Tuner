"""Fixed stdlib first-stage loader. Its exact bytes are in the submitted command.

Only after authenticating/copying the complete parent inventory does it use
normal Python package imports. This is separate from the isolated engine tree.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


# Ordinary Jobs only. Protected smoke codes 120--125 are a separate contract.
HF_JOB_PHASE_EXITS = {
    "BOOTSTRAP_ADMISSION": 80,
    "PARENT_MATERIALIZATION": 81,
    "WORKER_INPUT": 82,
    "SOURCE_MATERIALIZATION": 83,
    "RUNTIME_INVENTORY": 84,
    "PREPARED_INPUT": 85,
    "MODEL_PREPARATION": 86,
    "DISPATCH": 87,
    "ARTIFACT_PUBLICATION": 88,
    "UNCLASSIFIED": 89,
}
HF_NATIVE_STAGE_EXITS = frozenset({20, 21, 22, 23, 24, 25,
                                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39})


class HFJobFailure(RuntimeError):
    """Finite local diagnostic; never carries exception or provider text."""

    def __init__(self, phase, native_returncode=None):
        if type(phase) is not str or phase not in HF_JOB_PHASE_EXITS:
            raise ValueError("invalid HF diagnostic phase")
        if native_returncode is not None and (
            phase != "DISPATCH" or type(native_returncode) is not int
            or native_returncode not in HF_NATIVE_STAGE_EXITS
        ):
            raise ValueError("invalid HF native diagnostic")
        self.phase = phase
        self.exit_code = (native_returncode if native_returncode is not None
                          else HF_JOB_PHASE_EXITS[phase])
        super().__init__(phase)



def failure_projection(error):
    """Revalidate a local error at each boundary; mutable fields are not proof."""
    if type(error) is not HFJobFailure:
        return "UNCLASSIFIED", 89
    phase = getattr(error, "phase", None)
    code = getattr(error, "exit_code", None)
    if type(phase) is not str or phase not in HF_JOB_PHASE_EXITS or type(code) is not int:
        return "UNCLASSIFIED", 89
    if code != HF_JOB_PHASE_EXITS[phase] and not (
        phase == "DISPATCH" and code in HF_NATIVE_STAGE_EXITS
    ):
        return "UNCLASSIFIED", 89
    return phase, code


def _closed_failure(error):
    phase, code = failure_projection(error)
    return HFJobFailure(phase, code if code in HF_NATIVE_STAGE_EXITS else None)


def _relative(value):
    p = PurePosixPath(value)
    if (type(value) is not str or p.is_absolute() or p.as_posix() != value
            or not p.parts or any(part in {".", ".."} for part in p.parts)
            or "\\" in value or "\0" in value):
        raise ValueError("HF_BOOTSTRAP_INPUT")
    return p.parts


def _read(root, relative, size):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    held = []
    try:
        parent = os.open(root, flags)
        held.append(parent)
        chain = []
        parts = _relative(relative)
        for part in parts[:-1]:
            child = os.open(part, flags, dir_fd=parent)
            held.append(child)
            chain.append((parent, part, child))
            parent = child
        before = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ValueError("HF_BOOTSTRAP_MEMBER")
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                       dir_fd=parent)
        held.append(leaf)
        opened = os.fstat(leaf)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(opened) != identity(before):
            raise ValueError("HF_BOOTSTRAP_MEMBER")
        raw = bytearray()
        while True:
            chunk = os.read(leaf, min(1024 * 1024, size + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > size:
                raise ValueError("HF_BOOTSTRAP_MEMBER")
        if len(raw) != size or identity(os.fstat(leaf)) != identity(before):
            raise ValueError("HF_BOOTSTRAP_MEMBER")
        if identity(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)) != identity(before):
            raise ValueError("HF_BOOTSTRAP_MEMBER")
        for owner, name, child in chain:
            actual, expected = os.stat(name, dir_fd=owner, follow_symlinks=False), os.fstat(child)
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise ValueError("HF_BOOTSTRAP_MEMBER")
        return bytes(raw)
    finally:
        for fd in reversed(held):
            os.close(fd)


def _write(root, relative, raw, *, private=True):
    parts = _relative(relative)
    held = []
    try:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        held.append(parent)
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o700, dir_fd=parent)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=parent)
            held.append(child)
            info = os.fstat(child)
            if private and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
                raise ValueError("HF_BOOTSTRAP_PRIVATE")
            parent = child
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=parent)
        held.append(fd)
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise ValueError("HF_BOOTSTRAP_WRITE")
            view = view[count:]
        if private:
            os.fsync(fd)
            os.fsync(parent)
        # Provider mount writes are verified by readback and later host SDK
        # reads; local fsync is not an HF remote-persistence guarantee.
    finally:
        for fd in reversed(held):
            os.close(fd)


def run(payload):
    """Load only the command-bound parent members from the read-only mount."""
    phase = "BOOTSTRAP_ADMISSION"
    try:
        if len(payload) > 1024 * 1024:
            raise ValueError("HF_BOOTSTRAP_INPUT")
        document = json.loads(payload)
        if set(document) != {"schema_version", "parent_members", "worker"} or document["schema_version"] != "synaptic-hf-source-bootstrap/v1":
            raise ValueError("HF_BOOTSTRAP_INPUT")
        members = document["parent_members"]
        if type(members) is not list or not 1 <= len(members) <= 128:
            raise ValueError("HF_BOOTSTRAP_INPUT")
        phase = "PARENT_MATERIALIZATION"
        parent_root = Path("/workspace/hf-parent")
        parent_root.mkdir(mode=0o700)
        total, seen = 0, set()
        for row in members:
            if type(row) is not dict or set(row) != {"path", "size_bytes", "sha256"}:
                raise ValueError("HF_BOOTSTRAP_INPUT")
            name, size = row["path"], row["size_bytes"]
            _relative(name)
            if name in seen or type(size) is not int or not 0 < size <= 4 * 1024 * 1024:
                raise ValueError("HF_BOOTSTRAP_INPUT")
            seen.add(name)
            total += size
            if total > 16 * 1024 * 1024:
                raise ValueError("HF_BOOTSTRAP_INPUT")
            raw = _read("/source", "parent/" + name, size)
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("HF_BOOTSTRAP_MEMBER")
            _write(parent_root, name, raw)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(parent_root))
        from tuner.execution.providers.hf_jobs_provider_v1.source_worker import execute
        # This file is also embedded as __main__; imported bootstrap has a distinct
        # class identity. Translate only the exact authenticated worker error type.
        from tuner.execution.providers.hf_jobs_provider_v1.bootstrap import (
            HFJobFailure as WorkerFailure, failure_projection as worker_failure_projection,
        )
        try:
            return execute(document["worker"])
        except WorkerFailure as error:
            worker_phase, code = worker_failure_projection(error)
            native = code if code in HF_NATIVE_STAGE_EXITS else None
            raise HFJobFailure(worker_phase, native) from None
    except HFJobFailure as error:
        raise _closed_failure(error) from None
    except BaseException:
        raise HFJobFailure(phase) from None
