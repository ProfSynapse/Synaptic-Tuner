"""One-shot native process transport using the existing host record port.

This internal transport does not replace coordinator/Foundation admission.
Claim-before-spawn prevents automatic replay after every uncertain boundary.
Only a live owner may cancel; restart reads retain uncertainty, never kill by PID.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
import threading

from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_material import canonical, strict_json


@dataclass(frozen=True)
class NativeProcessLimits:
    timeout_seconds: int
    termination_grace_seconds: int
    maximum_output_bytes: int
    maximum_file_bytes: int

    def __post_init__(self):
        for name, minimum, maximum in (
            ("timeout_seconds", 1, 86400), ("termination_grace_seconds", 1, 30),
            ("maximum_output_bytes", 1, 16 * 1024 * 1024),
            ("maximum_file_bytes", 1, 512 * 1024 * 1024),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("native process bound is invalid")


@dataclass(frozen=True)
class NativeInvocation:
    attempt: str
    binding_digest: str
    executable: str
    executable_digest: str
    guardian: str
    guardian_digest: str
    worker: str
    worker_digest: str
    request: str
    request_digest: str
    private_root: str
    limits: NativeProcessLimits

    def __post_init__(self):
        if type(self.attempt) is not str or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", self.attempt):
            raise ValueError("native attempt reference is invalid")
        if type(self.limits) is not NativeProcessLimits:
            raise TypeError("native process limits must be exact")
        for name in ("binding_digest", "executable_digest", "guardian_digest", "worker_digest", "request_digest"):
            value = getattr(self, name)
            if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("native process digest is invalid")
        for name in ("executable", "guardian", "worker", "request", "private_root"):
            value = getattr(self, name)
            if (type(value) is not str or not value.startswith("/") or str(Path(value)) != value
                    or ".." in Path(value).parts or any(ord(c) < 32 for c in value)):
                raise ValueError("native process path is invalid")
        if Path(self.request).parent != Path(self.private_root):
            raise ValueError("native request must be directly inside owned root")

    @property
    def digest(self):
        return hashlib.sha256(canonical(asdict(self))).hexdigest()


class NativeOwnedProcess:
    """Retained single-use transport. No durable state is opened by the engine."""

    def __init__(self, records):
        self.records = records
        self._live = {}
        self._lock = threading.Lock()

    def _key(self, attempt, stage):
        return f"native-mlx/{attempt}/{stage}"

    def _retain(self, invocation, stage, value):
        raw = canonical({
            "schema": "synaptic-native-process-record/v1",
            "invocation_digest": invocation.digest, "binding_digest": invocation.binding_digest,
            "stage": stage, "value": value,
        })
        if self.records.create(partition="effects", key=self._key(invocation.attempt, stage), canonical=raw) is not True:
            raise ValueError("native process record already exists")
        observed = self.records.read(partition="effects", key=self._key(invocation.attempt, stage))
        if observed is None or observed.canonical != raw:
            raise ValueError("native process record readback differs")

    def observe(self, invocation):
        """Known outcome only; missing acknowledgement never permits re-spawn."""
        claim = self.records.read(partition="effects", key=self._key(invocation.attempt, "claim"))
        if claim is None:
            return {"state": "unclaimed"}
        if strict_json(claim.canonical)["invocation_digest"] != invocation.digest:
            raise ValueError("native retained invocation differs")
        outcome = self.records.read(partition="effects", key=self._key(invocation.attempt, "outcome"))
        if outcome is None:
            return {"state": "uncertain", "automatic_replay": False}
        value = strict_json(outcome.canonical)
        if value["invocation_digest"] != invocation.digest:
            raise ValueError("native retained outcome differs")
        return {"state": "terminated", "outcome": value["value"], "automatic_replay": False}

    def cancel(self, invocation):
        """Signal only the current owner's pipe, never an observed numeric PID."""
        with self._lock:
            item = self._live.get(invocation.attempt)
            if item is None or item[0] != invocation.digest:
                return False
            try:
                os.write(item[1], b"stop")
                return True
            except (BrokenPipeError, OSError):
                return False

    def run_once(self, invocation: NativeInvocation):
        if type(invocation) is not NativeInvocation or os.name != "posix":
            raise TypeError("native invocation requires supported POSIX transport")
        # Revalidation and all request/transport reads precede the durable claim.
        with NativeTree(Path(invocation.private_root), private=True) as root:
            for name in ("executable", "guardian", "worker", "request"):
                path = Path(getattr(invocation, name))
                with NativeTree(path.parent) as source:
                    raw = source.read(path.name, maximum=64 * 1024 * 1024)
                if hashlib.sha256(raw).hexdigest() != getattr(invocation, name + "_digest"):
                    raise ValueError("native process input bytes differ")
            root.check()
            self._retain(invocation, "claim", {"automatic_replay": False})
            read_fd, write_fd = os.pipe()
            guardian = None
            try:
                # No inherited credentials, PYTHONPATH, DYLD_*, loader settings or
                # ambient cache/home; all worker writes stay in owned private roots.
                environment = {
                    "PATH": str(Path(invocation.executable).parent) + ":/usr/bin:/bin",
                    "HOME": invocation.private_root, "TMPDIR": invocation.private_root,
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
                spec = canonical({
                    "argv": [invocation.executable, "-I", "-S", "-B", invocation.worker,
                             "--request", invocation.request],
                    "cwd": invocation.private_root, "environment": environment,
                    **asdict(invocation.limits),
                }) + b"\n"
                if len(spec) > 65536:
                    raise ValueError("native process request exceeds guardian bound")
                guardian = subprocess.Popen(
                    [invocation.executable, "-I", "-S", "-B", invocation.guardian, str(read_fd)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    env=environment, cwd=invocation.private_root, start_new_session=True,
                    close_fds=True, pass_fds=(read_fd,),
                )
                os.close(read_fd)
                read_fd = None
                with self._lock:
                    self._live[invocation.attempt] = (invocation.digest, write_fd)
                self._retain(invocation, "spawn", {"guardian_pid": guardian.pid,
                                                 "restart_signal_authority": False})
                # communicate starts the guardian only after spawn is retained.
                timeout = invocation.limits.timeout_seconds + 2 * invocation.limits.termination_grace_seconds + 10
                stdout, _ = guardian.communicate(input=spec, timeout=timeout)
                if guardian.returncode != 0 or len(stdout) > 4096:
                    raise ValueError("native guardian outcome unavailable")
                outcome = strict_json(stdout, maximum=4096)
                if (type(outcome) is not dict or outcome.get("schema") != "synaptic-native-process-outcome/v1"
                        or outcome.get("termination_confirmed") is not True):
                    raise ValueError("native worker termination is uncertain")
                self._retain(invocation, "outcome", outcome)
                return outcome
            finally:
                with self._lock:
                    self._live.pop(invocation.attempt, None)
                # Closing our sole liveness writer makes a running guardian stop
                # its worker even when an intervening store operation failed.
                os.close(write_fd)
                if read_fd is not None:
                    os.close(read_fd)
                if guardian is not None:
                    if guardian.stdin and not guardian.stdin.closed:
                        guardian.stdin.close()
                    try:
                        guardian.wait(timeout=2 * invocation.limits.termination_grace_seconds + 2)
                    except subprocess.TimeoutExpired:
                        # Do not kill the guardian responsible for owned cleanup.
                        # Leave the claim uncertain and never auto-replay.
                        pass
                    if guardian.stdout:
                        guardian.stdout.close()
