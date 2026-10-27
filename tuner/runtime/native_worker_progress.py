"""Trusted child-local numeric callbacks; no keys, files or public authority."""

import fcntl
import os
import selectors
import stat
import threading
import time

from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
from tuner.runtime.native_lifecycle_protocol import MAXIMUM_RECORD_BYTES, encode_worker_progress


def validate_progress_budget(projection, launch):
    """Pure host/child check of the same compiled-control progress upper bound."""
    controls, method = projection.controls, projection.workload.method
    n = controls.iterations
    if method == "sft":
        # Pinned MLX-LM reports on interval/final and validates at first,
        # interval/final before updating. Upper bounds include duplicates.
        records = n // controls.steps_per_report + n // controls.steps_per_eval + 6
    elif method == "grpo":
        records = n // controls.logging_steps + 4
    else:
        raise ValueError("native progress method is unsupported")
    policy = launch.policy
    if (records > policy.maximum_progress_records
            or (records + 1) * MAXIMUM_RECORD_BYTES > policy.maximum_journal_bytes):
        raise ValueError("native progress budget cannot cover selected controls")


class NativeWorkerProgress:
    def __init__(self, fd, launch):
        if (type(fd) is not int or fd < 3 or type(launch) is not NativeLifecycleLaunchV1
                or not stat.S_ISFIFO(os.fstat(fd).st_mode)
                or fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_WRONLY):
            raise ValueError("native worker progress requires a write-only pipe")
        self._fd, self._launch = fd, launch
        self._owner = os.getpid(), threading.get_ident()
        self._iterations = None
        self._count, self._step = 0, 0
        self._failed = False
        os.set_blocking(fd, False)
        os.set_inheritable(fd, False)

    def bind(self, projection):
        if self._iterations is not None:
            raise ValueError("native progress is already bound")
        validate_progress_budget(projection, self._launch)
        self._iterations = projection.controls.iterations

    def emit(self, phase, step, metrics):
        if (self._failed or self._fd is None or self._iterations is None
                or self._owner != (os.getpid(), threading.get_ident())):
            raise ValueError("native progress owner/binding is unavailable")
        event = {"kind": "progress", "phase": phase, "step": step, "metrics": metrics}
        raw = encode_worker_progress(event)
        if (not self._step <= step <= self._iterations
                or self._count >= self._launch.policy.maximum_progress_records):
            raise ValueError("native progress update or record bound differs")
        deadline, offset = time.monotonic() + 5, 0
        try:
            with selectors.DefaultSelector() as ready:
                ready.register(self._fd, selectors.EVENT_WRITE)
                while offset < len(raw):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not ready.select(remaining):
                        raise ValueError("native progress delivery deadline")
                    try:
                        sent = os.write(self._fd, raw[offset:])
                    except BlockingIOError:
                        continue
                    if sent <= 0:
                        raise ValueError("native progress delivery failed")
                    offset += sent
        except BaseException:
            self._failed = True
            raise
        self._count += 1
        self._step = step

    def close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


class NativeSFTProgressCallback:
    def __init__(self, emit):
        self._emit = emit

    def on_train_loss_report(self, info):
        self._emit("training", info["iteration"], {"loss": info["train_loss"],
                   "learning_rate": info["learning_rate"], "tokens": info["trained_tokens"]})

    def on_val_loss_report(self, info):
        # This is upstream in-loop validation, never final saved-adapter loss.
        self._emit("validation", info["iteration"], {"loss": info["val_loss"]})
