"""Actual local process custody and typed decoded-progress interruption hints."""

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from synaptic_tuner.api.v1._contract import canonical_bytes
from synaptic_tuner.api.v1.observations import (
    ObservationKind,
    TrainingPhaseObservedPayloadV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from tests.execution.test_observation_logs import metric_record

FILE = Path("examples/training_qualification/modal_lifecycle.py").resolve()
spec = importlib.util.spec_from_file_location("modal_lifecycle", FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
RUN = TrainingRunRef("supervised-run", "standalone-training")


def progress(root, records, suffix=b""):
    root.chmod(0o700)
    p = root / (
        "progress-"
        + hashlib.sha256(canonical_bytes(RUN.to_dict())).hexdigest()
        + ".jsonl"
    )
    p.write_bytes(
        b"".join(canonical_bytes(x.to_dict()) + b"\n" for x in records) + suffix
    )
    p.chmod(0o600)
    return p


def hint(root):
    return m.progress_hint(root, RUN, maximum_step=31, maximum_records=33)


def metric(step=1, sequence=0):
    x = metric_record(sequence, run=RUN)
    return replace(x, payload=replace(x.payload, step=step, total_steps=32))


def test_metric_and_nonmetric_record_do_not_index_missing_step(tmp_path):
    x = metric()
    phase = replace(
        x,
        sequence=1,
        kind=ObservationKind.TRAINING_PHASE_OBSERVED,
        payload=TrainingPhaseObservedPayloadV1("metric_capture_exhausted"),
    )
    progress(tmp_path, [x, phase])
    assert hint(tmp_path) is None


def test_completed_last_step_does_not_reuse_old_eligible_step(tmp_path):
    progress(tmp_path, [metric(), metric(32, 1)])
    assert hint(tmp_path) is None


def test_partial_trailing_record_waits_without_using_prefix(tmp_path):
    progress(tmp_path, [metric()], b'{"kind":')
    assert hint(tmp_path) is None


@pytest.mark.parametrize(
    "failure", ["foreign", "duplicate", "nonfinite", "unknown_kind"]
)
def test_whole_snapshot_refuses_invalid_record_after_valid_metric(tmp_path, failure):
    p = progress(tmp_path, [metric()])
    x = metric(sequence=1).to_dict()
    if failure == "foreign":
        x["stream"]["entity_id"] = "different-run"
    elif failure == "duplicate":
        x["sequence"] = 0
    elif failure == "nonfinite":
        x["payload"]["loss"] = float("nan")
    else:
        x["kind"] = "secret_phase"
    with p.open("ab") as stream:
        stream.write(
            json.dumps(x, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        )
    with pytest.raises(Exception):
        hint(tmp_path)


def producer(tmp_path, argv=None):
    tmp_path.chmod(0o700)
    return m.OwnedProducer(
        argv or [sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        environment={"PATH": os.environ["PATH"]},
        output_root=tmp_path,
    )


def cleanup(p):
    if p.process.poll() is None:
        signal.pidfd_send_signal(p.pidfd, signal.SIGKILL)
        p.process.wait(timeout=5)
    p.close()


def test_actual_direct_python_pidfd_stop_only_held_child(tmp_path):
    p = producer(tmp_path)
    try:
        assert os.getsid(p.process.pid) == p.process.pid
        assert p.interrupt(validate=lambda: 1) == -signal.SIGKILL
        assert p.interrupted
    finally:
        cleanup(p)


@pytest.mark.parametrize(
    "mode", ["raise", "not_ready", "wrong_executable", "wrong_birth"]
)
def test_invalid_trigger_or_identity_never_signals(tmp_path, monkeypatch, mode):
    p = producer(tmp_path)
    calls = []
    real_signal = signal.pidfd_send_signal
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda *args: calls.append(args))
    try:
        if mode == "wrong_executable":
            p.executable = (0, 0)
        if mode == "wrong_birth":
            p.start = "invalid"

        def validate():
            if mode == "raise":
                raise KeyError("private missing step")
            return None if mode == "not_ready" else 1

        with pytest.raises(Exception):
            p.interrupt(validate=validate)
        assert calls == [] and p.process.poll() is None
    finally:
        monkeypatch.setattr(signal, "pidfd_send_signal", real_signal)
        cleanup(p)


def test_actual_env_exec_transition_is_not_mistaken_for_final_python(tmp_path):
    p = producer(
        tmp_path,
        [
            "/usr/bin/env",
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import time; time.sleep(60)",
        ],
    )
    try:
        deadline = time.monotonic() + 3
        expected = os.stat(sys.executable)
        while time.monotonic() < deadline:
            if m._birth(p.process.pid)[1] == (expected.st_dev, expected.st_ino):
                break
            time.sleep(0.01)
        with pytest.raises(m.Refused, match="PRODUCER_IDENTITY"):
            p.interrupt(validate=lambda: 1)
        assert p.process.poll() is None
    finally:
        cleanup(p)


def test_supervisor_close_does_not_kill_detached_producer(tmp_path):
    p = producer(tmp_path)
    pid = p.process.pid
    p.close()
    try:
        assert p.process.poll() is None and os.getsid(pid) == pid
    finally:
        p.process.kill()
        p.process.wait(timeout=5)


def test_fixed_launch_uses_direct_python_and_explicit_environment(
    tmp_path, monkeypatch
):
    captured = {}
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "private-test-value")
    monkeypatch.setenv("SYNAPTIC_TRAIN_AUTHORITY_KEY", "private-test-value")
    monkeypatch.setattr(
        m, "OwnedProducer", lambda argv, **kw: captured.update(argv=argv, **kw)
    )
    value = {
        k: str(tmp_path / k)
        for k in [
            "python",
            "project_root",
            "engine_root",
            "recipe",
            "state_root",
            "tmp_root",
            "output_root",
            "env_file",
        ]
    }
    value.update(profile="synaptic-labs", environment="test")
    for key in ("state_root", "tmp_root", "output_root"):
        Path(value[key]).mkdir(mode=0o700)
    m.launch(value)
    assert captured["argv"][0] == value["python"] and "env" not in captured["argv"]
    assert "MODAL_TOKEN_SECRET" not in captured["environment"]
    assert "SYNAPTIC_TRAIN_AUTHORITY_KEY" not in captured["environment"]
    assert "--fresh-attempt" in captured["argv"]


def _tty_supervisor(tmp_path, *, retention=True, wait_seconds=15):
    import pty
    import fcntl
    import termios

    root = tmp_path / "state" / "synaptic-training"
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    root.parent.chmod(0o700)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("test fixture only")
    code = f"""
import importlib.util,sys
from pathlib import Path
s=importlib.util.spec_from_file_location("controller", {str(FILE)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.launch=lambda value:m.OwnedProducer([sys.executable,"-I","-S","-c","import time; time.sleep(60)"],cwd={str(tmp_path)!r},environment={{}},output_root=Path({str(output)!r}))
"""
    if retention:
        code += 'm.local_run_hint=lambda value,run,**kw:m.progress_hint(Path(value["state_root"])/"synaptic-training",run,maximum_step=31,maximum_records=33)\n'
    code += f"""
try:
 m.supervise({{"state_root":{str(tmp_path / "state")!r},"maximum_step":31,"maximum_records":33,"wait_seconds":{wait_seconds},"recipe":{str(recipe)!r}}})
except Exception:
 print('{{"status":"supervisor_closed"}}',flush=True)
"""
    master, slave = pty.openpty()

    def controlling_tty():
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

    parent = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        preexec_fn=controlling_tty,
    )
    os.close(slave)
    import select

    pending = b""
    deadline = time.monotonic() + 5
    while b'"started"' not in pending and time.monotonic() < deadline:
        if select.select([master], [], [], 0.1)[0]:
            pending += os.read(master, 4096)
    started = next(
        json.loads(line) for line in pending.splitlines() if b'"started"' in line
    )
    child_fd = os.pidfd_open(started["pid"])
    return parent, master, started["pid"], child_fd, root


def _tty_until(master, token):
    import select

    result = b""
    deadline = time.monotonic() + 5
    while token not in result and time.monotonic() < deadline:
        if select.select([master], [], [], 0.1)[0]:
            try:
                result += os.read(master, 8192)
            except OSError:
                break
    assert token in result, result
    return result


def test_actual_tty_roundtrip_valid_metric_interrupts_only_detached_child(tmp_path):
    parent, master, pid, fd, root = _tty_supervisor(tmp_path)
    try:
        progress(root, [metric()])
        os.write(
            master,
            (
                json.dumps({"operation": "interrupt", "run": RUN.to_dict()}) + "\n"
            ).encode(),
        )
        _tty_until(master, b'"interrupted"')
        assert parent.wait(timeout=5) == 0
        assert not Path(f"/proc/{pid}").exists()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        os.close(master)
        os.close(fd)


def test_actual_tty_invalid_nonmetric_never_signals_then_supervisor_loss_preserves_child(
    tmp_path,
):
    parent, master, pid, fd, root = _tty_supervisor(tmp_path)
    try:
        progress(root, [metric()], b'{"bad":"complete"}\n')
        os.write(
            master,
            (
                json.dumps({"operation": "interrupt", "run": RUN.to_dict()}) + "\n"
            ).encode(),
        )
        _tty_until(master, b'"refused"')
        assert Path(f"/proc/{pid}").exists()
        # Closing the real controlling terminal may SIGHUP its supervisor; detached producer must survive.
        os.close(master)
        master = None
        parent.wait(timeout=5)
        assert Path(f"/proc/{pid}").exists()
        assert os.getsid(pid) == pid
    finally:
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        os.close(fd)
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if master is not None:
            os.close(master)


def test_actual_tty_missing_original_retention_never_signals(tmp_path):
    parent, master, pid, fd, root = _tty_supervisor(tmp_path, retention=False)
    try:
        progress(root, [metric()])
        os.write(
            master,
            (
                json.dumps({"operation": "interrupt", "run": RUN.to_dict()}) + "\n"
            ).encode(),
        )
        _tty_until(master, b'"refused"')
        assert Path(f"/proc/{pid}").exists()
    finally:
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        os.close(fd)
        parent.wait(timeout=5)
        os.close(master)


def test_fifo_progress_is_refused_without_blocking(tmp_path):
    tmp_path.chmod(0o700)
    name = (
        "progress-"
        + hashlib.sha256(canonical_bytes(RUN.to_dict())).hexdigest()
        + ".jsonl"
    )
    os.mkfifo(tmp_path / name, 0o600)
    before = time.monotonic()
    with pytest.raises(m.Refused):
        hint(tmp_path)
    assert time.monotonic() - before < 1


def test_short_reads_are_completed_to_exact_eof(tmp_path, monkeypatch):
    progress(tmp_path, [metric()])
    real_read = os.read
    monkeypatch.setattr(os, "read", lambda fd, count: real_read(fd, min(count, 7)))
    assert hint(tmp_path) == 1


def test_parent_substitution_cannot_supply_interruption_hint(tmp_path, monkeypatch):
    progress(tmp_path, [metric()])
    real_open = os.open
    moved = tmp_path.with_name(tmp_path.name + "-original")
    changed = False

    def opening(path, *args, **kwargs):
        nonlocal changed
        if str(path).startswith("progress-") and not changed:
            changed = True
            tmp_path.rename(moved)
            tmp_path.mkdir(mode=0o700)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", opening)
    assert hint(tmp_path) is None


@pytest.mark.parametrize("kind", ["sequence", "step"])
def test_nonmonotonic_progress_is_not_a_trigger(tmp_path, kind):
    first, second = metric(2, 1), metric(1, 2)
    if kind == "sequence":
        first, second = metric(1, 2), metric(2, 1)
    progress(tmp_path, [first, second])
    with pytest.raises(m.Refused):
        hint(tmp_path)


def test_intent_collision_precedes_any_process_spawn(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    (tmp_path / "producer-intent.json").write_text("previous consumed attempt")
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: pytest.fail("spawn after collision")
    )
    with pytest.raises(FileExistsError):
        producer(tmp_path)


def test_birth_setup_failure_closes_handles_but_does_not_signal(tmp_path, monkeypatch):
    created = []
    real_popen = subprocess.Popen

    def spawn(*a, **kw):
        p = real_popen(*a, **kw)
        created.append(p)
        return p

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(
        os,
        "pidfd_open",
        lambda *_: (_ for _ in ()).throw(OSError("private setup error")),
    )
    with pytest.raises(OSError):
        producer(tmp_path)
    p = created[0]
    try:
        assert p.poll() is None
        assert (
            json.loads((tmp_path / "producer-birth.jsonl").read_text().splitlines()[0])[
                "pid"
            ]
            == p.pid
        )
        assert (tmp_path / "producer-intent.json").is_file()
        # All producer stdout/stderr handles closed in the failed constructor.
        targets = [
            os.readlink(x) for x in Path("/proc/self/fd").iterdir() if x.exists()
        ]
        assert str(tmp_path / "producer.stdout") not in targets
        assert str(tmp_path / "producer.stderr") not in targets
    finally:
        p.kill()
        p.wait(timeout=5)


def test_partial_tty_control_cannot_extend_deadline_or_signal(tmp_path):
    parent, master, pid, fd, _ = _tty_supervisor(tmp_path, wait_seconds=1)
    try:
        os.write(master, b'{"operation":')
        parent.wait(timeout=4)
        assert Path(f"/proc/{pid}").exists()
    finally:
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        os.close(fd)
        os.close(master)


def test_partial_pipe_control_cannot_block_deadline(tmp_path):
    state, output = tmp_path / "state", tmp_path / "output"
    state.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    recipe = tmp_path / "recipe"
    recipe.write_text("fixture")
    code = f"""
import importlib.util,sys,json
from pathlib import Path
s=importlib.util.spec_from_file_location("controller",{str(FILE)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.launch=lambda value:m.OwnedProducer([sys.executable,"-I","-S","-c","import time; time.sleep(60)"],cwd={str(tmp_path)!r},environment={{}},output_root=Path({str(output)!r}))
try:
 m.supervise({{"state_root":{str(state)!r},"recipe":{str(recipe)!r},"maximum_step":31,"maximum_records":33,"wait_seconds":1}})
except Exception:
 pass
"""
    parent = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    import select

    assert select.select([parent.stdout], [], [], 5)[0]
    pid = json.loads(parent.stdout.readline())["pid"]
    fd = os.pidfd_open(pid)
    try:
        parent.stdin.write(b'{"operation":')
        parent.stdin.flush()
        assert parent.wait(timeout=4) == 0
        assert Path(f"/proc/{pid}").exists()
    finally:
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        os.close(fd)
        parent.stdin.close()
        parent.stdout.close()
        parent.stderr.close()


def test_real_public_retention_binds_local_hint_then_cold_authority_remains_separate(
    tmp_path, monkeypatch
):
    import yaml
    from tests.training.test_modal_cold_cancellation import _start, _sdk_call, _open
    from tuner.handlers.runs_handler import _CurrentCancellationGrant
    from tuner.training.modal_run_retention import load_retained_modal_run

    _, run, root, events, client = _start(tmp_path, monkeypatch)
    retained = load_retained_modal_run(root, expected_run=run)
    inputs = retained.inputs
    engine = Path(inputs["engine_root"])
    # Real supported recipe parser, selected fixture publication identity; no fake lease/source construction.
    document = yaml.safe_load(
        Path("Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml").read_text()
    )
    document["dataset"]["expected_digest"] = inputs["components"]["execution_source"][
        "prepared_input"
    ]["revision"]
    path = tmp_path / "supervisor-recipe.yaml"
    path.write_text(yaml.safe_dump(document))
    value = {
        "state_root": str(root.parent.parent),
        "engine_root": str(engine),
        "profile": "fixture",
        "environment": "main",
        "recipe": str(path),
        "maximum_step": 31,
        "maximum_records": 33,
    }
    record = metric_record(0, run=run)
    name = (
        "progress-"
        + hashlib.sha256(canonical_bytes(run.to_dict())).hexdigest()
        + ".jsonl"
    )
    leaf = root.parent / name
    leaf.write_bytes(canonical_bytes(record.to_dict()) + b"\n")
    leaf.chmod(0o600)
    info = Path(value["state_root"]).stat()
    identity = (info.st_dev, info.st_ino)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        m.local_run_hint(value, run, state_identity=identity, recipe_sha256=digest)
        == record.payload.step
    )
    with pytest.raises(m.Refused, match="RUN_BINDING"):
        m.local_run_hint(
            {**value, "environment": "wrong-scope"},
            run,
            state_identity=identity,
            recipe_sha256=digest,
        )
    # Only the existing public lifecycle opener authenticates original FOUND/current permission.
    # This does not claim a real pending provider poll or live interruption.
    calls = _sdk_call(monkeypatch, client)
    with _open(run, root, _CurrentCancellationGrant(run)) as view:
        assert (
            view.runs.cancel(run, "explicit fixture operator request").state.value
            == "cancel_requested"
        )
    assert calls == ["fc-owned"]
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_coalesced_status_and_interrupt_drain_without_more_input(tmp_path):
    parent, master, pid, fd, root = _tty_supervisor(tmp_path)
    try:
        progress(root, [metric()])
        packet = (
            json.dumps({"operation": "status"})
            + "\n"
            + json.dumps({"operation": "interrupt", "run": RUN.to_dict()})
            + "\n"
        )
        os.write(master, packet.encode())
        _tty_until(master, b'"interrupted"')
        assert parent.wait(timeout=5) == 0
        assert not Path(f"/proc/{pid}").exists()
    finally:
        if parent.poll() is None:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            parent.wait(timeout=5)
        os.close(fd)
        os.close(master)


def test_coalesced_invalid_and_status_do_not_signal_or_stall(tmp_path):
    parent, master, pid, fd, _ = _tty_supervisor(tmp_path)
    try:
        os.write(master, b'{"operation":"invalid"}\n{"operation":"status"}\n')
        result = _tty_until(master, b'"producer_alive"')
        assert b'"refused"' in result
        assert Path(f"/proc/{pid}").exists()
    finally:
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        os.close(fd)
        parent.wait(timeout=5)
        os.close(master)


def _config(tmp_path):
    value = {
        k: str(tmp_path / k)
        for k in [
            "python",
            "project_root",
            "engine_root",
            "recipe",
            "state_root",
            "tmp_root",
            "output_root",
            "env_file",
        ]
    }
    value.update(
        profile="fixture",
        environment="main",
        maximum_step=31,
        maximum_records=33,
        wait_seconds=10,
    )
    for key in ("state_root", "tmp_root", "output_root"):
        Path(value[key]).mkdir(mode=0o700)
    return value


def test_launch_claim_and_directory_fsync_precede_spawn(tmp_path, monkeypatch):
    value = _config(tmp_path)
    events = []
    real = os.fsync

    def fsync(fd):
        import stat

        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)

    def spawn(argv, **kwargs):
        claim = Path(value["output_root"]) / "supervisor-intent.json"
        assert claim.stat().st_mode & 0o777 == 0o600
        assert json.loads(claim.read_text())["status"] == "consumed_before_spawn"
        assert "directory" in events and "file" in events
        return "owned fixture"

    monkeypatch.setattr(m, "OwnedProducer", spawn)
    assert m.launch(value) == "owned fixture"
    with pytest.raises(FileExistsError):
        m.launch(value)


def test_nonfresh_state_refuses_before_spawn_and_preserves_consumed_intent(
    tmp_path, monkeypatch
):
    value = _config(tmp_path)
    (Path(value["state_root"]) / "old").write_text("existing")
    monkeypatch.setattr(
        m, "OwnedProducer", lambda *a, **kw: pytest.fail("spawn with old state")
    )
    with pytest.raises(m.Refused, match="ATTEMPT_NOT_FRESH"):
        m.launch(value)
    assert (Path(value["output_root"]) / "supervisor-intent.json").is_file()


@pytest.mark.parametrize("bad", ["bool", "extra", "relative", "oversize"])
def test_closed_bounded_config_rejects_invalid_inputs(tmp_path, bad):
    value = _config(tmp_path)
    if bad == "bool":
        value["maximum_step"] = True
    elif bad == "extra":
        value["secret_value"] = "private-test-value"
    elif bad == "relative":
        value["recipe"] = "relative.yaml"
    path = tmp_path / "config.json"
    path.write_text(" " * 16385 if bad == "oversize" else json.dumps(value))
    with pytest.raises(m.Refused):
        m.load_config(path)


def test_config_fifo_refuses_without_waiting(tmp_path):
    path = tmp_path / "config"
    os.mkfifo(path, 0o600)
    before = time.monotonic()
    with pytest.raises(m.Refused):
        m.load_config(path)
    assert time.monotonic() - before < 1


def test_signal_failure_keeps_consumed_intent_and_refuses_second_interrupt(
    tmp_path, monkeypatch
):
    p = producer(tmp_path)
    real_signal = signal.pidfd_send_signal
    calls = []

    def uncertain(*a):
        calls.append(a)
        raise OSError("private signal boundary")

    monkeypatch.setattr(signal, "pidfd_send_signal", uncertain)
    try:
        with pytest.raises(OSError):
            p.interrupt(validate=lambda: 1)
        with pytest.raises(m.Refused, match="INTERRUPTION_CONSUMED"):
            p.interrupt(validate=lambda: 1)
        assert len(calls) == 1 and p.process.poll() is None
        assert not p.interruption_completed
        assert (
            json.loads((tmp_path / "interruption-intent.json").read_text())["status"]
            == "consumed_before_signal"
        )
    finally:
        monkeypatch.setattr(signal, "pidfd_send_signal", real_signal)
        cleanup(p)


def test_real_external_sigkill_is_not_completed_owned_interruption(tmp_path):
    p = producer(tmp_path)
    try:
        p.process.kill()
        assert p.process.wait(timeout=5) == -signal.SIGKILL
        assert not p.interrupted and not p.interruption_completed
        assert not (tmp_path / "interruption-receipt.json").exists()
    finally:
        cleanup(p)


def test_owned_interrupt_has_completed_durable_receipt(tmp_path):
    p = producer(tmp_path)
    try:
        assert p.interrupt(validate=lambda: 1) == -signal.SIGKILL
        assert p.interruption_completed
        receipt = json.loads((tmp_path / "interruption-receipt.json").read_text())
        assert receipt["status"] == "owned_signal_returned_and_reaped"
        assert receipt["start_time"] == p.start
    finally:
        cleanup(p)


@pytest.mark.parametrize(
    "completed,returncode,expected", [(False, -9, 2), (True, -9, 0), (False, 0, 0)]
)
def test_cli_never_normalizes_external_or_uncertain_sigkill(
    tmp_path, monkeypatch, completed, returncode, expected
):
    monkeypatch.setattr(
        sys, "argv", ["supervisor", "--config", str(tmp_path / "config")]
    )
    monkeypatch.setattr(m, "load_config", lambda _: {})
    monkeypatch.setattr(
        m,
        "supervise",
        lambda _: {
            "producer_returncode": returncode,
            "interruption_completed": completed,
        },
    )
    assert m.main() == expected


def test_actual_full_status_pipe_cannot_block_supervisor_deadline(tmp_path):
    state, output = tmp_path / "state", tmp_path / "output"
    state.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    recipe = tmp_path / "recipe"
    recipe.write_text("fixture")
    code = f"""
import importlib.util,sys,os,fcntl
from pathlib import Path
s=importlib.util.spec_from_file_location("controller",{str(FILE)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.launch=lambda value:m.OwnedProducer([sys.executable,"-I","-S","-c","import time; time.sleep(60)"],cwd={str(tmp_path)!r},environment={{}},output_root=Path({str(output)!r}))
os.set_blocking(sys.stdout.fileno(),False)
# Actual pipe is deliberately full while its reading end remains open.
try:
 while True:os.write(sys.stdout.fileno(),b'x'*4096)
except BlockingIOError:
 pass
try:
 m.supervise({{"state_root":{str(state)!r},"recipe":{str(recipe)!r},"maximum_step":31,"maximum_records":33,"wait_seconds":1}})
except Exception:
 pass
"""
    parent = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert parent.wait(timeout=5) == 0
        born = json.loads((output / "producer-birth.jsonl").read_text().splitlines()[0])
        pid = born["pid"]
        fd = os.pidfd_open(pid)
        try:
            assert Path(f"/proc/{pid}").exists()
            signal.pidfd_send_signal(fd, signal.SIGKILL)
        finally:
            os.close(fd)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        parent.stdin.close()
        parent.stdout.close()
        parent.stderr.close()
