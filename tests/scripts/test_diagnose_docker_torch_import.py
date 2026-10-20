from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from scripts import diagnose_docker_torch_import as diagnostic


DAEMON = "24805003-b125-469b-8080-9da4daacd364"
IMAGE = "sha256:" + "a" * 64
CID = "b" * 64
ROOT = Path(__file__).parents[2]


def _release(root: Path) -> tuple[Path, str]:
    path = root / "release.json"
    path.write_text(json.dumps({"image": {"digest": "a" * 64,
        "reference": "example/image@" + IMAGE}, "python": {"executable": "/usr/bin/python3"},
        "manifest_digest": "c" * 64}, sort_keys=True, separators=(",", ":")))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_docker(root: Path) -> tuple[Path, Path, Path]:
    executable = root / "docker"
    state = root / "state.json"
    state.write_text(json.dumps({"mode": "normal", "status": "absent", "stops": 0}))
    config = root / "config"
    config.mkdir()
    executable.write_text(f"#!{sys.executable}\n" + r'''import json,sys,time
from pathlib import Path
state_path=Path(__file__).with_name('state.json')
s=json.loads(state_path.read_text())
a=sys.argv[sys.argv.index('--host')+2:]
def save(): state_path.write_text(json.dumps(s))
if a[0]=='info': print('24805003-b125-469b-8080-9da4daacd364')
elif a[:2]==['image','inspect']: print('sha256:'+'a'*64)
elif a[:2]==['container','create']:
 s['status']='created'; s['name']=a[a.index('--name')+1]
 s['guarded']='--mount' in a
 s['label']=a[a.index('--label')+1].split('=',1)[1]; save()
 if s['mode']=='create_hang': time.sleep(10)
 if s['mode']=='bad_id': print('invalid'); sys.exit(0)
 print('b'*64)
elif a[:2]==['container','start']:
 s['status']='running'; save()
 if s['mode']=='start_hang': time.sleep(10)
 if s['mode']=='start_error': sys.exit(7)
 if s['mode']=='start_stdout_flood': sys.stdout.write('x'*20000); sys.exit(0)
 if s['mode']=='start_stderr_flood': sys.stderr.write('x'*20000); sys.exit(0)
 print('b'*64)
elif a[:2]==['container','inspect']:
 name='/' + s.get('name','missing')
 if s['mode']=='wrong_identity' or (s['mode']=='wrong_running_identity' and s['status']=='running'):
  name='/another-container'
 for value in ('b'*64,name,'sha256:'+'a'*64,
               {'ai.synapticlabs.torch-import.claim':s.get('label','')},
               s['status'],0): print(json.dumps(value))
elif a[:2]==['container','stop']:
 s['stops']+=1
 if s['mode']!='stop_unconfirmed': s['status']='exited'
 save(); print('b'*64)
elif a[:2]==['container','logs']:
 if s['mode']=='stdout_flood': sys.stdout.write('x'*20000)
 elif s['mode']=='stderr_flood': sys.stderr.write('x'*20000)
 elif s['mode']=='both_flood':
  sys.stdout.write('x'*5000); sys.stderr.write('x'*5000)
 elif s.get('guarded'): print('{"category":"PASS","schema_version":"synaptic-docker-torch-import-child/v1","stage":"TORCH_IMPORT"}')
 else: print('{"category":"PASS","schema_version":"synaptic-docker-torch-import-child/v1"}')
else: sys.exit(8)
''')
    executable.chmod(0o700)
    return executable, config, state


def _call(root: Path, executable: Path, config: Path, *, output: str = "out", **extra):
    release, sha = _release(root)
    return diagnostic.diagnose(docker=executable, docker_config=config,
        release_path=release, daemon_id=DAEMON, output_dir=root / output,
        expected_release_sha256=sha, **extra)


def _state(path: Path, **changes):
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value))
    return value


def _stage_source(root: Path):
    source = root / "source"
    engine = source / "engine"
    document = json.loads((ROOT / "tuner/runtime/manifests/offline-sft-worker-v1.json").read_text())
    for member in document["members"]:
        target = engine / member["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / member["path"], target)
    (source / "control").mkdir()
    manifest = source / "control/offline-sft-worker-v1.json"
    shutil.copy2(ROOT / "tuner/runtime/manifests/offline-sft-worker-v1.json", manifest)
    lock = root / "source-lock.json"
    lock.write_text(json.dumps({"schema_version": "synaptic-source-lock/v1", "sources": {
        "engine": {"commit": "0d515ae1ae235d0858802228ce4af8dba8fccd03"}}}))
    flags = {"source_root": engine, "source_lock": lock,
        "expected_source_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "expected_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "expected_closure_digest": document["closure_digest"]}
    return flags, document, manifest


def test_real_cli_output_budget_and_timeout(tmp_path: Path) -> None:
    docker, config, state = _fake_docker(tmp_path)
    authority = diagnostic._authority(docker, config)
    _state(state, mode="create_hang")
    with pytest.raises(ValueError, match="DOCKER_COMMAND_REJECTED"):
        diagnostic._run(authority, ("container", "create", "--name", "test-name",
                                    "--label", "ai.synapticlabs.torch-import.claim=hash"), timeout=0.2)
    # Killing the client did not stop the fake daemon's created container.
    assert json.loads(state.read_text())["status"] == "created"
    for mode in ("stdout_flood", "stderr_flood", "both_flood"):
        _state(state, mode=mode)
        with pytest.raises(ValueError, match="DOCKER_COMMAND_REJECTED"):
            diagnostic._run(authority, ("container", "logs", CID), timeout=2)


def test_owned_lifecycle_and_closed_result(tmp_path: Path, monkeypatch) -> None:
    docker, config, state = _fake_docker(tmp_path)
    real = diagnostic._run
    monkeypatch.setattr(diagnostic, "_run", lambda authority, args, *, timeout=15:
                        real(authority, args, timeout=min(timeout, 0.4)))
    monkeypatch.setattr(diagnostic, "_RUN_SECONDS", 2)
    # The fake daemon advances to terminal independently after start.
    original_inspect = diagnostic._inspect_owned
    def inspect(*args, **kwargs):
        status, code = original_inspect(*args, **kwargs)
        if status == "running":
            _state(state, status="exited")
        return status, code
    monkeypatch.setattr(diagnostic, "_inspect_owned", inspect)
    report = _call(tmp_path, docker, config)
    assert report["category"] == "PASS" and report["terminal_observed"] is True
    assert report["container_id"] == CID and report["source_identity"] == {}
    assert json.loads((tmp_path / "out/result.json").read_text()) == report
    assert json.loads(state.read_text())["stops"] == 0


def test_guarded_happy_path_records_independent_source_identity(tmp_path: Path, monkeypatch) -> None:
    flags, document, _manifest = _stage_source(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "_RUN_SECONDS", 2)
    original_inspect = diagnostic._inspect_owned
    def inspect(*args, **kwargs):
        status, code = original_inspect(*args, **kwargs)
        if status == "running":
            _state(state, status="exited")
        return status, code
    monkeypatch.setattr(diagnostic, "_inspect_owned", inspect)
    report = _call(tmp_path, docker, config, **flags)
    assert report["scope"] == "guarded" and report["stage"] == "TORCH_IMPORT"
    assert report["source_identity"]["closure_digest"] == document["closure_digest"]
    assert report["source_identity"]["manifest_sha256"] == flags["expected_manifest_sha256"]


@pytest.mark.parametrize("mode,reason,expected_stops", [
    ("start_hang", "START_UNCERTAIN", 1),
    ("start_error", "START_UNCERTAIN", 1),
    ("start_stdout_flood", "START_UNCERTAIN", 1),
    ("start_stderr_flood", "START_UNCERTAIN", 1),
    ("normal", "DEADLINE_STOPPED", 1),
    ("stop_unconfirmed", "DEADLINE_UNCERTAIN", 1),
    ("wrong_identity", "CONTAINER_STATE_INVALID", 0),
    ("wrong_running_identity", "DEADLINE_UNCERTAIN", 0),
])
def test_uncertain_start_and_exact_owned_cleanup(tmp_path: Path, monkeypatch,
                                                   mode, reason, expected_stops) -> None:
    docker, config, state = _fake_docker(tmp_path)
    _state(state, mode=mode)
    real = diagnostic._run
    monkeypatch.setattr(diagnostic, "_run", lambda authority, args, *, timeout=15:
                        real(authority, args, timeout=min(timeout, 0.3)))
    monkeypatch.setattr(diagnostic, "_RUN_SECONDS",
                        1 if mode in {"normal", "stop_unconfirmed"} else 0.1)
    with pytest.raises(ValueError, match=reason):
        _call(tmp_path, docker, config)
    actual = json.loads(state.read_text())
    assert actual["stops"] == expected_stops
    if expected_stops and mode != "stop_unconfirmed":
        assert actual["status"] == "exited"
    assert (tmp_path / "out/claim.json").exists()
    assert not (tmp_path / "out/result.json").exists()


def test_uncertain_create_does_not_replay_or_start(tmp_path: Path, monkeypatch) -> None:
    docker, config, state = _fake_docker(tmp_path)
    _state(state, mode="create_hang")
    real = diagnostic._run
    monkeypatch.setattr(diagnostic, "_run", lambda authority, args, *, timeout=15:
                        real(authority, args, timeout=min(timeout, 0.2)))
    with pytest.raises(ValueError, match="CREATE_UNCERTAIN"):
        _call(tmp_path, docker, config)
    assert json.loads(state.read_text())["status"] == "created"
    assert json.loads(state.read_text())["stops"] == 0
    with pytest.raises(ValueError, match="OUTPUT_INVALID"):
        _call(tmp_path, docker, config)


def test_invalid_created_id_is_consumed_without_start(tmp_path: Path) -> None:
    docker, config, state = _fake_docker(tmp_path)
    _state(state, mode="bad_id")
    with pytest.raises(ValueError, match="CREATE_UNCERTAIN"):
        _call(tmp_path, docker, config)
    assert json.loads(state.read_text())["status"] == "created"
    assert (tmp_path / "out/claim.json").exists()


def test_interrupt_after_start_stops_only_owned_container_once(tmp_path: Path, monkeypatch) -> None:
    docker, config, state = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "time", type("Clock", (), {
        "monotonic": staticmethod(__import__("time").monotonic),
        "sleep": staticmethod(lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    })())
    with pytest.raises(KeyboardInterrupt):
        _call(tmp_path, docker, config)
    actual = json.loads(state.read_text())
    assert actual["stops"] == 1 and actual["status"] == "exited"


def test_storage_failure_after_start_still_stops_owned_container(tmp_path: Path, monkeypatch) -> None:
    docker, config, state = _fake_docker(tmp_path)
    original_append = diagnostic._append
    storage_failed = False

    def append(output_dir, event):
        nonlocal storage_failed
        if event["phase"] == "STARTED":
            storage_failed = True
        if storage_failed:
            raise OSError("simulated full evidence filesystem")
        original_append(output_dir, event)

    monkeypatch.setattr(diagnostic, "_append", append)
    with pytest.raises(ValueError, match="DEADLINE_UNCERTAIN"):
        _call(tmp_path, docker, config)
    actual = json.loads(state.read_text())
    assert actual["stops"] == 1 and actual["status"] == "exited"
    assert (tmp_path / "out/claim.json").exists()
    assert not (tmp_path / "out/result.json").exists()


def test_uncertain_stop_acknowledgement_is_independently_inspected(tmp_path: Path, monkeypatch) -> None:
    docker, config, state = _fake_docker(tmp_path)
    original_run = diagnostic._run
    inspected_after_stop = []
    stop_sent = False

    def run(authority, args, *, timeout=15):
        nonlocal stop_sent
        result = original_run(authority, args, timeout=timeout)
        if args[:2] == ("container", "stop"):
            stop_sent = True
            raise ValueError("DOCKER_COMMAND_REJECTED")
        if stop_sent and args[:2] == ("container", "inspect"):
            inspected_after_stop.append(args[-1])
        return result

    monkeypatch.setattr(diagnostic, "_run", run)
    monkeypatch.setattr(diagnostic, "_RUN_SECONDS", 0.1)
    with pytest.raises(ValueError, match="DEADLINE_STOPPED"):
        _call(tmp_path, docker, config)
    actual = json.loads(state.read_text())
    assert actual["stops"] == 1 and actual["status"] == "exited"
    assert inspected_after_stop == [CID]
    events = [json.loads(line) for line in (tmp_path / "out/events.jsonl").read_text().splitlines()]
    assert events[-1]["phase"] == "TERMINAL_OBSERVED"


def test_wrong_self_consistent_source_rejected_before_docker(tmp_path: Path, monkeypatch) -> None:
    from tuner.runtime.offline_sft_worker import closure_digest
    flags, document, manifest = _stage_source(tmp_path)
    engine = flags["source_root"]
    assert diagnostic._source_identity(engine, flags["source_lock"],
        flags["expected_source_lock_sha256"], flags["expected_manifest_sha256"],
        flags["expected_closure_digest"])["closure_digest"] == flags["expected_closure_digest"]
    # Alter a staged member and rewrite all its internal commitments coherently.
    target = engine / "shared/env_bootstrap.py"
    target.write_bytes(target.read_bytes() + b"\n# alternate closure\n")
    member = next(item for item in document["members"] if item["path"] == "shared/env_bootstrap.py")
    member["size_bytes"] = target.stat().st_size
    member["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    document["payload_bytes"] += len(b"\n# alternate closure\n")
    document["closure_digest"] = closure_digest(document)
    manifest.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    from tuner.runtime.offline_sft_worker import load_offline_sft_worker_closure
    assert load_offline_sft_worker_closure(manifest,
        expected_digest=document["closure_digest"], engine_root=engine).closure_digest == document["closure_digest"]
    docker, config, _state_path = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "_run", lambda *args, **kwargs: pytest.fail("Docker touched"))
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert not (tmp_path / "out").exists()


def test_cli_rejection_never_prints_exception_text(monkeypatch, capsys) -> None:
    def fail(**_kwargs):
        raise RuntimeError("private exception path /secret and credential")
    monkeypatch.setattr(diagnostic, "diagnose", fail)
    code = diagnostic.main(["--docker", "/usr/bin/docker", "--docker-config", "/private/config",
        "--release", "/private/release.json", "--output-dir", "/private/output",
        "--daemon-id", DAEMON, "--expected-release-sha256", "a" * 64])
    assert code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"schema_version": diagnostic.SCHEMA,
        "status": "REJECTED", "reason_code": "OTHER"}
