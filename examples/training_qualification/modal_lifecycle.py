"""Caller-owned Linux supervisor; never submits a cancellation or opens an SDK.

Private decoded progress is an interruption hint, not independent authentication.
After interruption use the existing cold public reader before any cancellation.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
import re
from pathlib import Path
import select
import signal
import stat
import subprocess
import sys
import time

from synaptic_tuner.api.v1._contract import canonical_bytes
from synaptic_tuner.api.v1.observations import ObservationKind, ObservationRecordV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.training.modal_run_retention import load_retained_modal_run


class Refused(RuntimeError):
    def __init__(self, phase):
        self.phase = phase
        super().__init__(phase)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refused("INPUT")
        result[key] = value
    return result


def _json(raw):
    return json.loads(
        raw,
        object_pairs_hook=_object,
        parse_constant=lambda _: (_ for _ in ()).throw(Refused("INPUT")),
    )


def _private(path, directory=False):
    info = path.lstat()
    if (
        info.st_uid != os.geteuid()
        or info.st_mode & 0o077
        or (
            not stat.S_ISDIR(info.st_mode)
            if directory
            else not stat.S_ISREG(info.st_mode)
        )
    ):
        raise Refused("CUSTODY")
    return info


def progress_hint(root, run, *, maximum_step, maximum_records):
    """Read only the same private leaf as RunProgressConsumer; no store/SDK calls.

    A trailing partial record waits for completion. Malformed complete records
    refuse the entire snapshot, including a preceding valid metric.
    """
    original = _private(root, True)
    name = (
        "progress-"
        + hashlib.sha256(canonical_bytes(run.to_dict())).hexdigest()
        + ".jsonl"
    )
    with ExitStack() as stack:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, parent)
        held = os.fstat(parent)
        if (held.st_dev, held.st_ino) != (original.st_dev, original.st_ino):
            raise Refused("CUSTODY")
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent
        )
        stack.callback(os.close, descriptor)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o077
            or before.st_nlink != 1
            or before.st_size > maximum_records * 4097
        ):
            raise Refused("PROGRESS")
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 4096))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        extra = os.read(descriptor, 1)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        leaf = os.stat(name, dir_fd=parent, follow_symlinks=False)
        named_parent = _private(root, True)
        identity = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if (
            extra
            or identity(before) != identity(after)
            or identity(before) != identity(leaf)
            or (named_parent.st_dev, named_parent.st_ino) != (held.st_dev, held.st_ino)
        ):
            return None
        if not raw or not raw.endswith(b"\n"):
            return None
        lines = raw.splitlines()
        if len(lines) > maximum_records:
            raise Refused("PROGRESS")
        selected, last_sequence, last_step, exhausted = None, -1, 0, False
        for line in lines:
            if not line or len(line) > 4096:
                raise Refused("PROGRESS")
            record = ObservationRecordV1.from_dict(_json(line))
            if canonical_bytes(record.to_dict()) != line:
                raise Refused("PROGRESS")
            if (
                record.stream.entity_id != run.run_id
                or record.stream.project_ref != run.project_ref
                or record.stream.family.value != "training"
                or record.sequence <= last_sequence
            ):
                raise Refused("PROGRESS")
            last_sequence = record.sequence
            if record.kind is not ObservationKind.TRAINING_METRIC_OBSERVED:
                if (
                    record.kind is not ObservationKind.TRAINING_PHASE_OBSERVED
                    or record.payload.to_dict() != {"phase": "metric_capture_exhausted"}
                ):
                    raise Refused("PROGRESS")
                exhausted, selected = True, None
                continue
            if exhausted or record.payload.step <= last_step:
                raise Refused("PROGRESS")
            last_step = record.payload.step
            selected = last_step if 1 <= last_step <= maximum_step else None
        return selected


def _birth(pid):
    start = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    executable = os.stat(f"/proc/{pid}/exe")
    return start, (executable.st_dev, executable.st_ino)


def _exclusive(path, binary=False):
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        fd = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        os.fsync(parent)
        stream = os.fdopen(fd, "wb" if binary else "w")
        fd = None
        return stream
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


class OwnedProducer:
    """Retain the real process handle; errors/close never implicitly kill it."""

    def __init__(self, argv, *, cwd, environment, output_root):
        _private(output_root, True)
        self.output_root = output_root
        expected = os.stat(argv[0])
        self.executable = (expected.st_dev, expected.st_ino)
        # Reserve all leaves before spawn; collisions cannot occur only after an effect.
        with _exclusive(output_root / "producer-intent.json") as stream:
            json.dump(
                {
                    "new_session": True,
                    "status": "consumed_before_spawn",
                    "argv_sha256": hashlib.sha256(
                        canonical_bytes({"argv": argv})
                    ).hexdigest(),
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        self.stdout = self.stderr = None
        self.pidfd = None
        try:
            self.stdout = _exclusive(output_root / "producer.stdout", True)
            self.stderr = _exclusive(output_root / "producer.stderr", True)
            birth = _exclusive(output_root / "producer-birth.jsonl")
            try:
                self.process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=self.stdout,
                    stderr=self.stderr,
                    start_new_session=True,
                )
                # Record the actual PID even if subsequent local custody setup fails.
                json.dump(
                    {
                        "pid": self.process.pid,
                        "expected_executable_device": self.executable[0],
                        "expected_executable_inode": self.executable[1],
                    },
                    birth,
                )
                birth.write("\n")
                birth.flush()
                os.fsync(birth.fileno())
                self.pidfd = os.pidfd_open(self.process.pid)
                self.start, observed = _birth(self.process.pid)
                json.dump(
                    {
                        "pid": self.process.pid,
                        "start_time": self.start,
                        "observed_executable_device": observed[0],
                        "observed_executable_inode": observed[1],
                    },
                    birth,
                )
                birth.write("\n")
                birth.flush()
                os.fsync(birth.fileno())
            finally:
                birth.close()
            self.armed = False
            self.interrupted = False
        except BaseException:
            self.close()  # Never signal a possibly running child or make the consumed attempt retryable.
            raise

    def interrupt(self, *, validate):
        # All input/origin checks finish before any signal, even if validation raises.
        if self.interrupted:
            raise Refused("INTERRUPTION_CONSUMED")
        result = validate()
        if result is None:
            raise Refused("TRIGGER_NOT_READY")
        if self.process.poll() is not None:
            raise Refused("PRODUCER_TERMINAL")
        start, executable = _birth(self.process.pid)
        if start != self.start or executable != self.executable:
            raise Refused("PRODUCER_IDENTITY")
        with _exclusive(self.output_root / "interruption-intent.json") as stream:
            json.dump(
                {
                    "pid": self.process.pid,
                    "start_time": self.start,
                    "metric_step": result,
                    "status": "consumed_before_signal",
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        self.armed = True
        self.interrupted = (
            True  # An uncertain signal/wait must never permit another interrupt.
        )
        signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)
        return self.process.wait(timeout=5)

    def close(self):
        with ExitStack() as stack:
            if self.pidfd is not None:
                descriptor, self.pidfd = self.pidfd, None
                stack.callback(os.close, descriptor)
            for name in ("stdout", "stderr"):
                stream = getattr(self, name, None)
                if stream is not None:
                    setattr(self, name, None)
                    stack.callback(stream.close)


def _bounded_file(path, maximum):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
            raise Refused("INPUT")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(4096, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        named = path.lstat()
        identity = lambda x: (
            x.st_dev,
            x.st_ino,
            x.st_size,
            x.st_mtime_ns,
            x.st_ctime_ns,
        )
        if (
            len(raw) != info.st_size
            or identity(info) != identity(after)
            or identity(info) != identity(named)
        ):
            raise Refused("INPUT_CHANGED")
        return bytes(raw)
    finally:
        os.close(descriptor)


def load_config(path):
    raw = _bounded_file(path, 16384)
    value = _json(raw)
    fields = {
        "python",
        "project_root",
        "engine_root",
        "recipe",
        "state_root",
        "tmp_root",
        "output_root",
        "env_file",
        "profile",
        "environment",
        "maximum_step",
        "maximum_records",
        "wait_seconds",
    }
    if type(value) is not dict or set(value) != fields:
        raise Refused("INPUT")
    for key in fields - {"maximum_step", "maximum_records", "wait_seconds"}:
        if type(value[key]) is not str or not value[key]:
            raise Refused("INPUT")
    for key, upper in (
        ("maximum_step", 10000),
        ("maximum_records", 10001),
        ("wait_seconds", 7200),
    ):
        if type(value[key]) is not int or not 1 <= value[key] <= upper:
            raise Refused("INPUT")
    for key in (
        "python",
        "project_root",
        "engine_root",
        "recipe",
        "state_root",
        "tmp_root",
        "output_root",
        "env_file",
    ):
        if not Path(value[key]).is_absolute():
            raise Refused("INPUT")
    for key in ("state_root", "tmp_root", "output_root"):
        _private(Path(value[key]), True)
    return value


def local_run_hint(value, run, *, state_identity, recipe_sha256):
    """Local binding/custody only. Full authority validation remains cold/public."""
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run.run_id) is None
        or run.project_ref != "standalone-training"
    ):
        raise Refused("RUN_BINDING")
    root = Path(value["state_root"]) / "synaptic-training"
    current = _private(Path(value["state_root"]), True)
    if (current.st_dev, current.st_ino) != state_identity:
        raise Refused("CUSTODY")
    retained = load_retained_modal_run(root / run.run_id, expected_run=run)
    inputs = retained.inputs
    if (
        inputs["engine_root"] != value["engine_root"]
        or inputs["modal_profile"] != value["profile"]
        or inputs["modal_environment"] != value["environment"]
    ):
        raise Refused("RUN_BINDING")
    from tuner.training.modal_recipe import load_modal_sft_recipe

    path = Path(value["recipe"])
    if hashlib.sha256(_bounded_file(path, 65536)).hexdigest() != recipe_sha256:
        raise Refused("RECIPE_CHANGED")
    recipe = load_modal_sft_recipe(
        path, profiles_root=Path(value["engine_root"]) / "Trainers/runtime_profiles"
    )
    if hashlib.sha256(_bounded_file(path, 65536)).hexdigest() != recipe_sha256:
        raise Refused("RECIPE_CHANGED")
    prepared = inputs["components"]["execution_source"]["prepared_input"]
    if (
        recipe.training_input(prepared["ref"]).canonical_json()
        != inputs["prepared_request"]["canonical_json"]
        or recipe.dataset_digest != prepared["revision"]
        or inputs["components"]["resources"]
        != {
            "accelerator": recipe.accelerator,
            "accelerator_count": recipe.accelerator_count,
            "timeout_seconds": recipe.timeout_seconds,
        }
    ):
        raise Refused("RUN_BINDING")
    return progress_hint(
        root,
        run,
        maximum_step=value["maximum_step"],
        maximum_records=value["maximum_records"],
    )


def launch(value):
    # Do not inherit credentials/SDK overrides; the existing CLI resolves only its explicit env file/profile.
    environment = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "LANG", "LC_ALL")
        if key in os.environ
    }
    environment.update(
        SYNAPTIC_ENGINE_ROOT=value["engine_root"],
        XDG_STATE_HOME=value["state_root"],
        TMPDIR=value["tmp_root"],
        MODAL_IMAGE_BUILDER_VERSION="2025.06",
    )
    argv = [
        value["python"],
        "-I",
        "-B",
        "-m",
        "tuner",
        "train",
        "--job-config",
        value["recipe"],
        "--project-root",
        value["project_root"],
        "--fresh-attempt",
        "--modal-profile",
        value["profile"],
        "--modal-environment",
        value["environment"],
        "--env-file",
        value["env_file"],
        "--json",
    ]
    output_root = Path(value["output_root"])
    # Claim before Popen: any subsequent failure remains consumed, never an automatic retry.
    with _exclusive(output_root / "supervisor-intent.json") as stream:
        json.dump(
            {
                "config": value,
                "argv_sha256": hashlib.sha256(
                    canonical_bytes({"argv": argv})
                ).hexdigest(),
                "status": "consumed_before_spawn",
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    for key in ("state_root", "tmp_root"):
        if any(Path(value[key]).iterdir()):
            raise Refused("ATTEMPT_NOT_FRESH")
    return OwnedProducer(
        argv,
        cwd=value["project_root"],
        environment=environment,
        output_root=output_root,
    )


def supervise(value, *, controls=sys.stdin):
    state_info = _private(Path(value["state_root"]), True)
    recipe_path = Path(value["recipe"])
    recipe_sha256 = hashlib.sha256(_bounded_file(recipe_path, 65536)).hexdigest()
    control_fd = controls.fileno()
    was_blocking = os.get_blocking(control_fd)
    producer = launch(value)
    deadline = time.monotonic() + value["wait_seconds"]
    buffer = b""
    try:
        os.set_blocking(control_fd, False)
        print(
            json.dumps({"status": "started", "pid": producer.process.pid}), flush=True
        )
        while producer.process.poll() is None and time.monotonic() < deadline:
            if b"\n" not in buffer:
                ready, _, _ = select.select(
                    [controls], [], [], min(1, max(0, deadline - time.monotonic()))
                )
                if not ready:
                    continue
                try:
                    chunk = os.read(control_fd, 4096)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise Refused("CONTROL_CLOSED")
                buffer += chunk
                if len(buffer) > 4096:
                    raise Refused("CONTROL_BOUND")
            # Drain bounded coalesced lines before waiting for more input.
            for _ in range(16):
                if b"\n" not in buffer or time.monotonic() >= deadline:
                    break
                raw, buffer = buffer.split(b"\n", 1)
                try:
                    command = _json(raw)
                    if command == {"operation": "status"}:
                        print(json.dumps({"status": "producer_alive"}), flush=True)
                        continue
                    if (
                        type(command) is not dict
                        or set(command) != {"operation", "run"}
                        or command["operation"] != "interrupt"
                    ):
                        raise Refused("CONTROL")
                    run = TrainingRunRef.from_dict(command["run"])

                    def validate():
                        return local_run_hint(
                            value,
                            run,
                            state_identity=(state_info.st_dev, state_info.st_ino),
                            recipe_sha256=recipe_sha256,
                        )

                    producer.interrupt(validate=validate)
                    print(
                        json.dumps(
                            {
                                "status": "interrupted",
                                "cold_public_pending_required": True,
                            }
                        ),
                        flush=True,
                    )
                except Exception:
                    print(
                        json.dumps({"status": "refused", "phase": "INTERRUPTION"}),
                        flush=True,
                    )
        if producer.process.poll() is None:
            raise Refused("WAIT_EXPIRED")
        return producer.process.returncode
    finally:
        try:
            os.set_blocking(control_fd, was_blocking)
        finally:
            producer.close()  # Deliberately no terminate/kill on error or supervisor shutdown.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        return 0 if supervise(load_config(args.config)) in (0, -signal.SIGKILL) else 2
    except Exception:
        print(json.dumps({"status": "refused", "phase": "SUPERVISOR"}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
