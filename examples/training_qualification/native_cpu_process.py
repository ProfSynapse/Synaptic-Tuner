"""Private, one-shot CPU observation of the native owned-process boundary.

This does not issue a training admission or load ML packages. A clean source
selection and exact local bytes are checked before an unused state root is
created. A consumed or uncertain state is never adopted or replayed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import re
import stat
import sys
from urllib.parse import urlsplit

ENGINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bounded_command import BoundedCommandError, run_bounded_command
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.runtime.native_process import NativeInvocation, NativeOwnedProcess, NativeProcessLimits


_SCHEMA = "synaptic-native-cpu-process-config/v1"
_FIELDS = {
    "schema", "engine_root", "engine_commit", "engine_ref", "engine_remote",
    "launcher", "executable", "executable_sha256", "python_version",
    "platform", "machine", "guardian_sha256", "worker_sha256",
    "request_sha256", "state_root", "attempt", "limits",
}
_LIMIT_FIELDS = {
    "timeout_seconds", "termination_grace_seconds", "maximum_output_bytes", "maximum_file_bytes",
}
_REQUEST = "examples/training_qualification/native_cpu_request.json"
_WORKER = "examples/training_qualification/native_cpu_worker.py"
_GUARDIAN = "tuner/runtime/native_process_guardian.py"
_RESULT = b'{"schema":"synaptic-native-cpu-worker-result/v1","status":"ok"}\n'


def _path(value):
    if (type(value) is not str or not value.startswith("/") or len(value) > 4096
            or str(Path(value)) != os.path.normpath(value)
            or ".." in Path(value).parts or any(ord(char) < 32 for char in value)):
        raise ValueError("native CPU path is invalid")
    return Path(value)


def _file(path, digest, maximum):
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("native CPU file digest is invalid")
    with NativeTree(path.parent) as parent:
        raw = parent.read(path.name, maximum=maximum)
        parent.check()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("native CPU file bytes differ")
    return raw


def _command(*args, cwd):
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LC_ALL": "C", "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
    }
    try:
        status, raw = run_bounded_command(
            ["git", "--no-optional-locks", "-C", str(cwd), *args],
            environment=environment, maximum=4096, timeout=10,
        )
        if status != 0:
            raise ValueError("native CPU source command refused")
        return raw.decode("utf-8").strip()
    except (BoundedCommandError, UnicodeError):
        raise ValueError("native CPU source command refused") from None


def _source(config, root):
    if root != ENGINE_ROOT or root.resolve(strict=True) != root:
        raise ValueError("native CPU source checkout differs")
    ref = config["engine_ref"]
    if (type(ref) is not str or re.fullmatch(r"refs/heads/[A-Za-z0-9._/-]+", ref) is None
            or ".." in ref or "//" in ref or ref.endswith("/")):
        raise ValueError("native CPU source ref is invalid")
    remote = config["engine_remote"]
    if type(remote) is not str:
        raise ValueError("native CPU source remote is invalid")
    parsed = urlsplit(remote)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.port is not None
            or parsed.query or parsed.fragment or not parsed.path.endswith(".git")):
        raise ValueError("native CPU source remote is invalid")
    commit = config["engine_commit"]
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("native CPU source commit is invalid")
    if (_command("status", "--porcelain=v1", "--untracked-files=all", cwd=root)
            or _command("rev-parse", "HEAD", cwd=root) != commit
            or _command("rev-parse", "--verify", ref, cwd=root) != commit
            or _command("remote", "get-url", "origin", cwd=root) != remote):
        raise ValueError("native CPU source selection differs")


def _launcher(config, executable):
    launcher = _path(config["launcher"])
    if str(launcher) != sys.executable or platform.python_implementation() != "CPython":
        raise ValueError("native CPU launcher differs")
    if platform.python_version() != config["python_version"]:
        raise ValueError("native CPU Python version differs")
    with NativeTree(launcher.parent) as parent:
        info = os.stat(launcher.name, dir_fd=parent.fd, follow_symlinks=False)
        if info.st_uid != os.getuid():
            raise ValueError("native CPU launcher owner differs")
        if stat.S_ISLNK(info.st_mode):
            if os.readlink(launcher.name, dir_fd=parent.fd) != str(executable):
                raise ValueError("native CPU launcher target differs")
        elif not stat.S_ISREG(info.st_mode) or launcher != executable:
            raise ValueError("native CPU launcher type differs")
        parent.check()


def _preflight(config_path):
    config_path = _path(config_path)
    with NativeTree(config_path.parent, private=True) as parent:
        info = os.stat(config_path.name, dir_fd=parent.fd, follow_symlinks=False)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("native CPU config custody differs")
        raw = parent.read(config_path.name, maximum=8192)
        parent.check()
    config = strict_json(raw, maximum=8192)
    if (type(config) is not dict or set(config) != _FIELDS or config["schema"] != _SCHEMA
            or canonical(config) != raw):
        raise ValueError("native CPU config fields differ")
    root = _path(config["engine_root"])
    state = _path(config["state_root"])
    if state == root or state.is_relative_to(root):
        raise ValueError("native CPU state overlaps source")
    if (platform.system() != config["platform"] or platform.machine() != config["machine"]
            or os.name != "posix"):
        raise ValueError("native CPU platform differs")
    limits_value = config["limits"]
    if type(limits_value) is not dict or set(limits_value) != _LIMIT_FIELDS:
        raise ValueError("native CPU limits differ")
    limits = NativeProcessLimits(**limits_value)
    if (limits.timeout_seconds > 30 or limits.termination_grace_seconds > 5
            or limits.maximum_output_bytes > 4096 or limits.maximum_file_bytes > 1024 * 1024):
        raise ValueError("native CPU bounds exceed qualification")
    _source(config, root)
    executable = _path(config["executable"])
    _launcher(config, executable)
    _file(executable, config["executable_sha256"], 64 * 1024 * 1024)
    _file(root / _GUARDIAN, config["guardian_sha256"], 1024 * 1024)
    _file(root / _WORKER, config["worker_sha256"], 1024 * 1024)
    request = _file(root / _REQUEST, config["request_sha256"], 1024)
    with NativeTree(state.parent, private=True) as parent:
        try:
            os.stat(state.name, dir_fd=parent.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("native CPU state is already used")
        parent.check()
    binding = hashlib.sha256(raw).hexdigest()
    invocation = NativeInvocation(
        config["attempt"], binding, str(executable), config["executable_sha256"],
        str(root / _GUARDIAN), config["guardian_sha256"],
        str(root / _WORKER), config["worker_sha256"],
        str(state / "request.json"), config["request_sha256"], str(state), limits,
    )
    return config, state, request, invocation, binding


def inspect(config_path):
    config, _, _, invocation, binding = _preflight(config_path)
    return {
        "schema": "synaptic-native-cpu-process-inspection/v1", "status": "REVIEW_REQUIRED",
        "source_commit": config["engine_commit"], "selection_digest": binding,
        "invocation_digest": invocation.digest, "executable_sha256": config["executable_sha256"],
        "guardian_sha256": config["guardian_sha256"],
        "worker_sha256": config["worker_sha256"], "request_sha256": config["request_sha256"],
        "limits": config["limits"],
    }


def run(config_path):
    _, state, request, invocation, binding = _preflight(config_path)
    with NativeTree(state.parent, private=True) as parent:
        os.mkdir(state.name, mode=0o700, dir_fd=parent.fd)
        parent.check()
    with NativeTree(state, private=True) as root:
        root.write_exclusive("request.json", request)
        if root.members() != ("request.json",):
            raise ValueError("native CPU fresh state differs")
    with SQLiteDurableStoreV1(state / "workflow.sqlite3", create=True) as store:
        process = NativeOwnedProcess(store)
        outcome = process.run_once(invocation)
        if process.observe(invocation) != {
            "state": "terminated", "outcome": outcome, "automatic_replay": False,
        }:
            raise ValueError("native CPU outcome readback differs")
    expected = {
        "schema": "synaptic-native-process-outcome/v1", "reason": "worker_exited",
        "exit_code": 0, "termination_confirmed": True,
        "stdout_bytes": 0, "stderr_bytes": 0,
    }
    if outcome != expected:
        raise ValueError("native CPU worker did not exit normally")
    receipt = {
        "schema": "synaptic-native-cpu-process-result/v1", "status": "NORMAL_EXIT_OBSERVED",
        "selection_digest": binding, "invocation_digest": invocation.digest,
        "outcome": outcome, "worker_result_sha256": hashlib.sha256(_RESULT).hexdigest(),
    }
    with NativeTree(state, private=True) as root:
        if (root.members() != ("request.json", "worker-result.json", "workflow.sqlite3")
                or root.read("worker-result.json", maximum=1024) != _RESULT):
            raise ValueError("native CPU worker publication differs")
        root.write_exclusive("qualification-result.json", canonical(receipt))
        if root.members() != (
            "qualification-result.json", "request.json", "worker-result.json", "workflow.sqlite3",
        ):
            raise ValueError("native CPU result inventory differs")
        root.check()
    return receipt


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) != 2 or argv[0] not in ("inspect", "run"):
            raise ValueError("native CPU action differs")
        result = inspect(argv[1]) if argv[0] == "inspect" else run(argv[1])
    except BaseException:
        # No exception text, paths, argv or environment data enter CLI output.
        sys.stdout.buffer.write(canonical({
            "schema": "synaptic-native-cpu-process-cli/v1", "status": "REJECTED_OR_UNCERTAIN",
        }) + b"\n")
        return 1
    sys.stdout.buffer.write(canonical(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
