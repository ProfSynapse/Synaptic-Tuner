"""Real guardian and fixed public fixture keys; no ML or execution admission."""

from dataclasses import asdict, replace
import hashlib
import hmac
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1
from tests.runtime import test_native_process as fixtures
from tuner.runtime import native_lifecycle_protocol as protocol
from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
from tuner.runtime.native_process import NativeOwnedProcess
from tuner.runtime.native_process_session import NativeProcessSession
from tuner.training.modal_host_authority import HMACAuthenticator
from tuner.training.native_mlx_authority import NativeEvidenceAuthority

invocation = fixtures.invocation


def authority():
    return NativeEvidenceAuthority("fixture-reader", "fixture-key", HMACAuthenticator(
        {"fixture-key": b"k" * 32}, allowed_purposes=frozenset({"native-lifecycle-capability/v1"})))


@pytest.fixture
def launch(invocation):
    raw = Path(protocol.__file__).read_bytes()
    Path(invocation.guardian).with_name("native_lifecycle_protocol.py").write_bytes(raw)
    return NativeLifecycleLaunchV1(protocol.NativeLifecyclePolicyV1(32, 32768), hashlib.sha256(raw).hexdigest())


def prepared(invocation, launch, script):
    invocation = fixtures.worker(invocation, script)
    raw = json.dumps({"lifecycle": launch.to_dict()}, sort_keys=True, separators=(",", ":")).encode()
    Path(invocation.request).write_bytes(raw)
    return replace(invocation, request_digest=hashlib.sha256(raw).hexdigest())


def session(records, launch, streams=None):
    return NativeProcessSession(records, lifecycle=launch, lifecycle_authority=authority(),
                                streams=records if streams is None else streams)


def until(action, condition):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        value = action()
        if condition(value):
            return value
        time.sleep(.01)
    pytest.fail("bounded lifecycle fixture did not reach expected state")


EMIT = """import json, os, sys, time
from pathlib import Path
fd = int(sys.argv[sys.argv.index('--progress-fd') + 1])
def emit(step):
    event = {'kind':'progress','phase':'training','step':step,'metrics':{'loss':2.0 / step}}
    os.write(fd, json.dumps(event,sort_keys=True,separators=(',',':')).encode() + b'\\n')
"""


def test_live_progress_is_retained_before_completion_with_sqlite(invocation, launch):
    invocation = prepared(invocation, launch, EMIT + "emit(1)\nwhile not Path('release').exists(): time.sleep(.01)\n")
    with SQLiteDurableStoreV1(Path(invocation.private_root) / "session.sqlite3", create=True) as records:
        with session(records, launch) as owner:
            assert owner.start_once(invocation)["state"] == "running"
            view = until(lambda: owner.progress(invocation), lambda value: len(value.event_bytes) == 1)
            assert view.terminal_bytes is None
            assert json.loads(view.event_bytes[0])["metrics"] == {"loss": 2.0}
            assert owner.observe(invocation)["state"] == "running"
            Path(invocation.private_root, "release").write_text("release")
            done = until(lambda: owner.observe(invocation), lambda value: value["state"] == "terminated")
            assert done["outcome"]["exit_code"] == 0 and done["outcome"]["termination_confirmed"]
            final = owner.progress(invocation)
            assert final.terminal_bytes is not None and len(final.event_bytes) == 2
            page = records.read_page(partition="effects", stream_key="native-lifecycle/" + invocation.digest,
                                     after_sequence=None, limit=10)
            assert [item.sequence for item in page.entries] == [0, 1]
            assert b"k" * 32 not in b"".join(item.canonical for item in page.entries)
        # Independent host object derives the same scoped key; no new dispatch.
        with session(records, launch) as cold:
            assert cold.observe(invocation) == done
            assert cold.cancel(invocation) is False
            with pytest.raises(ValueError, match="already exists"):
                cold.start_once(invocation)


def test_worker_inherits_only_numeric_progress_pipe(invocation, launch):
    script = EMIT + """import fcntl
opened = []
for candidate in range(3, 4096):
    try: fcntl.fcntl(candidate, fcntl.F_GETFD)
    except OSError: continue
    opened.append(candidate)
Path('fds.json').write_text(json.dumps({'opened':opened,'progress':fd,'environment':dict(os.environ)}))
emit(1)
"""
    invocation = prepared(invocation, launch, script)
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        done = until(lambda: owner.observe(invocation), lambda value: value["state"] == "terminated")
        assert done["outcome"]["exit_code"] == 0
    report = json.loads(Path(invocation.private_root, "fds.json").read_bytes())
    assert report["opened"] == [report["progress"]]
    assert set(report["environment"]) <= {"PATH", "HOME", "TMPDIR", "PWD", "HF_HUB_OFFLINE",
                                         "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "LC_CTYPE"}
    # The fixture shell shim supplies PWD; production uses a captured Mach-O.
    assert report["environment"].get("PWD", invocation.private_root) == invocation.private_root


@pytest.mark.parametrize("stop", ["cancel", "close"])
def test_cold_authenticated_terminal_after_live_cancel_or_owner_close(invocation, launch, stop):
    invocation = prepared(invocation, launch, EMIT + "emit(1)\nwhile True: time.sleep(.01)\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    owner = session(records, launch, streams)
    try:
        owner.start_once(invocation)
        until(lambda: owner.progress(invocation), lambda value: bool(value.event_bytes))
        if stop == "cancel":
            assert owner.cancel(invocation)
        else:
            owner.close()
        with session(records, launch, streams) as cold:
            done = until(lambda: cold.observe(invocation), lambda value: value["state"] == "terminated")
            assert done["outcome"]["reason"] == ("cancel_requested" if stop == "cancel" else "owner_lost")
            assert done["outcome"]["termination_confirmed"] is True
            assert cold.cancel(invocation) is False
    finally:
        owner.close()


def test_actual_owner_process_death_leaves_authenticated_cold_terminal(invocation, launch):
    invocation = prepared(invocation, launch, EMIT + "emit(1)\nPath('worker-ready').write_text('ready')\nwhile True: time.sleep(.01)\n")
    root = Path(invocation.private_root)
    script = """import json, os, sys, time
from pathlib import Path
from tuner.runtime.native_process import NativeInvocation, NativeProcessLimits
from tuner.runtime.native_lifecycle_protocol import NativeLifecyclePolicyV1
from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
from tuner.runtime.native_process_session import NativeProcessSession
from tuner.training.native_mlx_authority import NativeEvidenceAuthority
from tuner.training.modal_host_authority import HMACAuthenticator
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
value=json.loads(sys.argv[1]); value['limits']=NativeProcessLimits(**value['limits'])
invocation=NativeInvocation(**value)
selected=json.loads(sys.argv[2]); selected['policy'].pop('schema')
launch=NativeLifecycleLaunchV1(NativeLifecyclePolicyV1(**selected['policy']),selected['protocol_sha256'])
authority=NativeEvidenceAuthority('fixture-reader','fixture-key',HMACAuthenticator(
    {'fixture-key':b'k'*32},allowed_purposes=frozenset({'native-lifecycle-capability/v1'})))
with SQLiteDurableStoreV1(Path(invocation.private_root)/'session.sqlite3',create=True) as store:
    owner=NativeProcessSession(store,lifecycle=launch,lifecycle_authority=authority,streams=store)
    owner.start_once(invocation)
    deadline=time.monotonic()+3
    while not Path(invocation.private_root,'worker-ready').exists():
        if time.monotonic()>deadline: raise SystemExit(99)
        time.sleep(.01)
    os._exit(0)
"""
    child = subprocess.run([sys.executable, "-B", "-c", script, json.dumps(asdict(invocation)),
                            json.dumps(launch.to_dict())], cwd=Path(__file__).resolve().parents[2],
                           env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
                           capture_output=True, timeout=8)
    assert child.returncode == 0, (child.stdout, child.stderr)
    with SQLiteDurableStoreV1(root / "session.sqlite3") as records:
        with session(records, launch) as cold:
            done = until(lambda: cold.observe(invocation), lambda value: value["state"] == "terminated")
            assert done["outcome"]["reason"] == "owner_lost"
            assert done["outcome"]["termination_confirmed"] is True
            with pytest.raises(ValueError, match="already exists"):
                cold.start_once(invocation)


@pytest.mark.parametrize("invalid", [
    b'{"kind":"terminal","outcome":{}}\n',
    b'{"kind":"progress","phase":"private text","step":1,"metrics":{}}\n',
    b"x" * 4096,
])
def test_worker_cannot_mint_terminal_or_free_text_evidence(invocation, launch, invalid):
    invocation = prepared(invocation, launch, EMIT + "os.write(fd," + repr(invalid) + ")\nwhile True: time.sleep(.01)\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        with pytest.raises(ValueError):
            until(lambda: owner.observe(invocation), lambda value: value["state"] == "terminated")
    with session(records, launch, streams) as cold:
        assert cold.observe(invocation) == {"state": "uncertain", "automatic_replay": False}
        assert cold.progress(invocation).terminal_bytes is None
        with pytest.raises(ValueError, match="already exists"):
            cold.start_once(invocation)


def test_unbound_launch_or_changed_protocol_rejects_before_claim(invocation, launch):
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        with pytest.raises(ValueError, match="selection differs"):
            owner.start_once(invocation)
        assert NativeOwnedProcess(records).observe(invocation) == {"state": "unclaimed"}
        invocation = prepared(invocation, launch, "pass\n")
        Path(invocation.guardian).with_name("native_lifecycle_protocol.py").write_text("changed")
        with pytest.raises(ValueError, match="source differs"):
            owner.start_once(invocation)
        assert NativeOwnedProcess(records).observe(invocation) == {"state": "unclaimed"}


def test_another_reader_can_confirm_terminal_before_live_owner_polls(invocation, launch):
    invocation = prepared(invocation, launch, EMIT + "emit(1)\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner, session(records, launch, streams) as cold:
        owner.start_once(invocation)
        done = until(lambda: cold.observe(invocation), lambda value: value["state"] == "terminated")
        assert owner.observe(invocation) == done


def test_outcome_acknowledgement_loss_reads_exact_existing_record(invocation, launch):
    class LostAck(InMemoryDurableRecordStoreV1):
        def create(self, *, partition, key, canonical):
            result = super().create(partition=partition, key=key, canonical=canonical)
            if key.endswith("/outcome") and result:
                raise RuntimeError("fixture lost terminal acknowledgement")
            return result
    invocation = prepared(invocation, launch, EMIT + "emit(1)\n")
    records, streams = LostAck(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        done = until(lambda: owner.observe(invocation), lambda value: value["state"] == "terminated")
        assert done["outcome"]["exit_code"] == 0
        assert NativeOwnedProcess(records).observe(invocation) == done


def test_stream_acknowledgement_loss_preserves_cold_recovery(invocation, launch):
    class LostAck(InMemoryDurableStreamStoreV1):
        lost = False
        def append(self, **kwargs):
            result = super().append(**kwargs)
            if result and not self.lost:
                self.lost = True
                raise RuntimeError("fixture lost progress acknowledgement")
            return result
    invocation = prepared(invocation, launch, EMIT + "emit(1)\n")
    records, streams = InMemoryDurableRecordStoreV1(), LostAck()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        with pytest.raises(RuntimeError, match="lost progress"):
            until(lambda: owner.observe(invocation), lambda value: value["state"] == "terminated")
    with session(records, launch, streams) as cold:
        done = until(lambda: cold.observe(invocation), lambda value: value["state"] == "terminated")
        assert done["outcome"]["termination_confirmed"] is True
        with pytest.raises(ValueError, match="already exists"):
            cold.start_once(invocation)


def test_retained_terminal_survives_mutable_journal_disappearance(invocation, launch):
    invocation = prepared(invocation, launch, EMIT + "emit(1)\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        done = until(lambda: owner.observe(invocation), lambda value: value["state"] == "terminated")
    Path(invocation.private_root, "lifecycle-" + invocation.digest + ".jsonl").unlink()
    # Valid retained authority does not depend on mutable source revalidation.
    Path(invocation.request).unlink()
    with session(records, launch, streams) as cold:
        assert cold.observe(invocation) == done


def test_latest_other_reader_prefix_prevents_cached_rollback(invocation, launch):
    invocation = prepared(invocation, launch, EMIT + """emit(1)
while not Path('second').exists(): time.sleep(.01)
emit(2)
while True: time.sleep(.01)
""")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner, session(records, launch, streams) as cold:
        owner.start_once(invocation)
        until(lambda: owner.progress(invocation), lambda view: len(view.event_bytes) == 1)
        journal = Path(invocation.private_root, "lifecycle-" + invocation.digest + ".jsonl")
        first = journal.read_bytes()
        Path(invocation.private_root, "second").write_text("second")
        until(lambda: cold.progress(invocation), lambda view: len(view.event_bytes) == 2)
        journal.write_bytes(first)
        with pytest.raises(ValueError, match="prefix changed"):
            owner.progress(invocation)


@pytest.mark.parametrize("change", ["replace", "symlink", "permissions"])
def test_journal_custody_failure_cannot_publish_fresh_progress(invocation, launch, change):
    invocation = prepared(invocation, launch, EMIT + "emit(1)\nwhile True: time.sleep(.01)\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        until(lambda: owner.progress(invocation), lambda view: len(view.event_bytes) == 1)
        journal = Path(invocation.private_root, "lifecycle-" + invocation.digest + ".jsonl")
        if change == "permissions":
            journal.chmod(0o644)
        else:
            saved = journal.with_suffix(".saved")
            journal.rename(saved)
            if change == "symlink": journal.symlink_to(saved)
            else:
                journal.write_bytes(saved.read_bytes())
                journal.chmod(0o600)
        with pytest.raises((ValueError, OSError)):
            owner.progress(invocation)


def test_exact_invocation_output_limit_applies_even_to_authentic_terminal(invocation, launch):
    invocation = fixtures.guardian_script(invocation, "import os,sys\nsys.stdin.buffer.readline(65537)\nos.read(int(sys.argv[1]),4)\n")
    invocation = prepared(invocation, launch, "pass\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        journal_fd = owner._live[invocation.attempt].journal_fd
        writer = protocol.NativeLifecycleWriter(journal_fd, invocation_digest=invocation.digest,
            policy=launch.policy, key=authority().lifecycle_key(invocation.digest, launch.policy.digest))
        writer.append({"kind": "terminal", "outcome": {
            "schema": "synaptic-native-process-outcome/v1", "reason": "worker_exited", "exit_code": 0,
            "termination_confirmed": True, "stdout_bytes": invocation.limits.maximum_output_bytes + 65537,
            "stderr_bytes": 0}})
        with pytest.raises(ValueError, match="output bound"):
            owner.observe(invocation)
    with session(records, launch, streams) as cold:
        with pytest.raises(ValueError, match="output bound"):
            cold.observe(invocation)
        with pytest.raises(ValueError, match="output bound"):
            cold.progress(invocation)
    assert NativeOwnedProcess(records).observe(invocation)["state"] == "uncertain"


def test_boolean_coercion_and_inconsistent_file_budget_reject_before_claim(invocation, launch):
    invocation = prepared(invocation, launch, "pass\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    request = json.loads(Path(invocation.request).read_bytes())
    request["lifecycle"]["policy"]["maximum_progress_records"] = 32.0
    raw = json.dumps(request).encode()
    Path(invocation.request).write_bytes(raw)
    changed = replace(invocation, request_digest=hashlib.sha256(raw).hexdigest())
    with session(records, launch, streams) as owner:
        with pytest.raises(ValueError, match="selection differs"):
            owner.start_once(changed)
        assert owner.observe(changed) == {"state": "unclaimed"}
    changed = prepared(invocation, launch, "pass\n")
    changed = replace(changed, limits=replace(changed.limits, maximum_file_bytes=8192))
    with session(records, launch, streams) as owner:
        with pytest.raises(ValueError, match="selection differs"):
            owner.start_once(changed)
        assert owner.observe(changed) == {"state": "unclaimed"}


def test_control_pipe_descriptors_do_not_collide_with_closed_stdio(tmp_path):
    report = tmp_path / "pipe-result"
    script = """import os,sys
from tuner.runtime.native_process_session import _pipe
path=sys.argv[1]
for fd in range(3):
    try: os.close(fd)
    except OSError: pass
r,w=_pipe()
assert min(r,w)>=3
os.write(w,b'bounded')
assert os.read(r,7)==b'bounded'
os.close(r);os.close(w)
with open(path,'w') as result: result.write('passed')
"""
    child = subprocess.run([sys.executable, "-B", "-c", script, str(report)],
        cwd=Path(__file__).resolve().parents[2], env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, timeout=5)
    assert child.returncode == 0 and report.read_text() == "passed"


def test_interruption_before_journal_creation_stays_uncertain(invocation, launch):
    invocation = prepared(invocation, launch, "pass\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    NativeOwnedProcess(records)._retain(invocation, "claim", {"automatic_replay": False})
    with session(records, launch, streams) as cold:
        assert cold.observe(invocation) == {"state": "uncertain", "automatic_replay": False}
        assert cold.progress(invocation).terminal_bytes is None
        assert cold.cancel(invocation) is False
        with pytest.raises(ValueError, match="already exists"):
            cold.start_once(invocation)


def test_guardian_revalidates_codec_before_any_worker_dispatch(invocation, launch, monkeypatch):
    invocation = prepared(invocation, launch, "from pathlib import Path\nPath('worker-started').write_text('unsafe')\n")
    from tuner.runtime import native_process_session as transport
    original = transport.subprocess.Popen
    def changed_before_spawn(*args, **kwargs):
        Path(invocation.guardian).with_name("native_lifecycle_protocol.py").write_text("changed")
        return original(*args, **kwargs)
    monkeypatch.setattr(transport.subprocess, "Popen", changed_before_spawn)
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        with pytest.raises(ValueError):
            until(lambda: owner.observe(invocation), lambda value: value["state"] == "terminated")
    assert not Path(invocation.private_root, "worker-started").exists()
    with session(records, launch, streams) as cold:
        assert cold.observe(invocation) == {"state": "uncertain", "automatic_replay": False}


def test_authenticated_terminal_survives_lost_stdout_acknowledgement(invocation, launch):
    invocation = prepared(invocation, launch, EMIT + "emit(1)\nwhile True: time.sleep(.01)\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    owner = session(records, launch, streams)
    try:
        owner.start_once(invocation)
        until(lambda: owner.progress(invocation), lambda view: len(view.event_bytes) == 1)
        guardian = owner._live[invocation.attempt].guardian
        guardian.stdout.close()
        assert owner.cancel(invocation)
        owner.close()
        assert guardian.returncode == 120  # Closed protocol acknowledgement failed.
        with session(records, launch, streams) as cold:
            done = cold.observe(invocation)
            assert done["state"] == "terminated"
            assert done["outcome"]["reason"] == "cancel_requested"
            assert done["outcome"]["termination_confirmed"] is True
    finally:
        owner.close()


@pytest.mark.parametrize("case", ["invalid_tag", "wrong_digest", "wrong_argv", "noncanonical", "duplicate_field"])
def test_launch_envelope_and_invocation_reject_tampering_before_worker(invocation, launch, monkeypatch, case):
    invocation = prepared(invocation, launch, "from pathlib import Path\nPath('worker-started').write_text('unsafe')\n")
    from tuner.runtime import native_process_session as transport
    original_spawn, original_write = transport.subprocess.Popen, transport.os.write
    target, changed = [], []
    def spawned(*args, **kwargs):
        child = original_spawn(*args, **kwargs)
        target.append(child.stdin.fileno())
        return child
    def altered(fd, raw):
        if target and fd == target[0] and not changed:
            changed.append(True)
            assert raw.endswith(b"\n")  # Fixture launch fits one startup chunk.
            document = json.loads(raw)
            if case == "invalid_tag": document["lifecycle_tag"] = "0" * 64
            if case == "wrong_digest": document["invocation_digest"] = "b" * 64
            if case == "wrong_argv": document["argv"][2] = "-B"
            if case in {"wrong_digest", "wrong_argv"}:
                document.pop("lifecycle_tag")
                # A valid test MAC cannot bypass the independent invocation join.
                key = authority().lifecycle_key(invocation.digest, launch.policy.digest)
                encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
                document["lifecycle_tag"] = hmac.new(
                    key, b"synaptic-native-lifecycle-launch/v1\0" + encoded, hashlib.sha256).hexdigest()
            raw = json.dumps(document, sort_keys=True,
                             separators=(",", ":") if case != "noncanonical" else (", ", ": ")).encode() + b"\n"
            if case == "duplicate_field":
                raw = raw.replace(b'{"argv":', b'{"argv":[],"argv":', 1)
        return original_write(fd, raw)
    monkeypatch.setattr(transport.subprocess, "Popen", spawned)
    monkeypatch.setattr(transport.os, "write", altered)
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        with pytest.raises(ValueError):
            until(lambda: owner.observe(invocation), lambda value: value["state"] == "terminated")
    assert changed and not Path(invocation.private_root, "worker-started").exists()
    with session(records, launch, streams) as cold:
        assert cold.observe(invocation) == {"state": "uncertain", "automatic_replay": False}


@pytest.mark.parametrize("sequence,raw", [(1, b"{}"), (0, b"x" * 4096)])
def test_untrusted_retained_stream_entries_obey_frame_and_sequence_bounds(invocation, launch, sequence, raw):
    invocation = prepared(invocation, launch, EMIT + "emit(1)\nwhile True: time.sleep(.01)\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        assert streams.append(partition="effects", stream_key="native-lifecycle/" + invocation.digest,
                              sequence=sequence, canonical=raw)
        with pytest.raises(ValueError, match="retained"):
            owner.progress(invocation)


def test_unsigned_outcome_boolean_cannot_replace_authenticated_integer(invocation, launch):
    invocation = prepared(invocation, launch, EMIT + "emit(1)\n")
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with session(records, launch, streams) as owner:
        owner.start_once(invocation)
        done = until(lambda: owner.observe(invocation), lambda value: value["state"] == "terminated")
        assert done["outcome"]["exit_code"] == 0
    key = "native-mlx/" + invocation.attempt + "/outcome"
    entry = records.read(partition="effects", key=key)
    document = json.loads(entry.canonical)
    document["value"]["exit_code"] = False
    assert records.compare_and_swap(partition="effects", key=key, expected_revision=entry.revision,
        canonical=json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    with session(records, launch, streams) as cold:
        with pytest.raises(ValueError, match="terminal record differs"):
            cold.observe(invocation)
