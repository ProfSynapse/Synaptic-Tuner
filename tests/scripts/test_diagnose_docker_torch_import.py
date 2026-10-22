from __future__ import annotations

import hashlib
import json
import base64
import re
import shutil
import subprocess
import sys
import ast
from pathlib import Path

import pytest

from scripts import diagnose_docker_torch_import as diagnostic


DAEMON = "24805003-b125-469b-8080-9da4daacd364"
IMAGE = "sha256:" + "a" * 64
CID = "b" * 64
ROOT = Path(__file__).parents[2]
DIRECT_TRAINER = "a7a8585e2a4607c7c0b49614260fda2814f29887a3b61e4b8b5df6f71f52639a"
DIRECT_MANIFEST = "c6c90c366f3b4043dc0cd651ae9ba130c05d76c370ac1570fab918dea25c31d1"
DIRECT_CLOSURE = "eb1b679c294462603f792cf2c66d8c0080443309b25e5114df654fb88b71ccce"
R4_ENGINE_COMMIT = "4a8bb138672d60318479286797ffd75572407054"
R6_ENGINE_COMMIT = "ba228d6c0e8859288b246c4b140774fa29c567a3"
R6_PROJECT_COMMIT = "9f7799659565481f05d3bc1ecfa8174c87cf1508"


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
 s['gpu_requested']='--gpus' in a and a[a.index('--gpus')+1]=='device=0' if '--gpus' in a else False
 s['gpu_probe_unhidden']=s['gpu_requested'] and "'NVIDIA_VISIBLE_DEVICES':'void'" not in a[-1] and "'CUDA_VISIBLE_DEVICES':''" not in a[-1]
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
 if 'DeviceRequests' in a[a.index('--format')+1]:
  if s['mode']=='wrong_gpu_request': print('null')
  else: print(json.dumps([{'Capabilities':[['gpu']],'Count':0,'DeviceIDs':['0'],'Driver':'','Options':{}}] if s.get('gpu_requested') else None))
  sys.exit(0)
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


def _stage_worker_context(root: Path):
    flags, _document, _manifest = _stage_source(root)
    engine = flags["source_root"]
    variables = {name: "1" for name in diagnostic._R2_ENV_KEYS}
    variables.update({diagnostic._ROOT_ENV[name]: value
                      for name, value in diagnostic._R2_ROOTS.items()})
    variables.update({"PYTHONPATH": "/source/engine",
        "PATH": "/opt/unsloth-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HF_HOME": "/artifacts/run/cache/huggingface",
        "TRANSFORMERS_CACHE": "/artifacts/run/cache/transformers",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    workload = {"execution_source": {"runtime": {
        "roots": diagnostic._R2_ROOTS,
        "environment": {"clear_inherited": True, "variables": variables},
        "interpreter": {"executable": "/opt/unsloth-venv/bin/python3"}}},
        "configuration": {"document": {"model": {
            "ref": "Example/Model", "revision": "a" * 40}}}}
    path = engine.parent / "control/workload.json"
    path.write_text(json.dumps(workload, sort_keys=True, separators=(",", ":")))
    flags.update({"worker_context": True,
        "expected_workload_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_runtime_sha256": hashlib.sha256(
            (engine / "Trainers/sft/runtime_v1.py").read_bytes()).hexdigest(),
        "expected_trainer_sha256": hashlib.sha256(
            (engine / "Trainers/sft/train_sft.py").read_bytes()).hexdigest()})
    return flags, path


def _stage_direct_inputs(root: Path):
    flags, workload_path = _stage_worker_context(root)
    lock = flags["source_lock"]
    source_lock = json.loads(lock.read_bytes())
    source_lock["sources"]["engine"]["commit"] = R4_ENGINE_COMMIT
    lock.write_text(json.dumps(source_lock))
    flags["expected_source_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    flags["expected_engine_commit"] = R4_ENGINE_COMMIT
    workload = json.loads(workload_path.read_bytes())
    content = b'{"text":"stub"}\n'
    digest = hashlib.sha256(content).hexdigest()
    dataset = {"ref": "prepared://sha256/" + "d" * 64, "revision": "d" * 64,
               "content_digest": digest, "size_bytes": len(content),
               "format": "synaptic-conversation-sft-row/v1"}
    workload["configuration"]["document"]["dataset"] = dataset
    workload_path.write_text(json.dumps(workload, sort_keys=True, separators=(",", ":")))
    flags["expected_workload_sha256"] = hashlib.sha256(workload_path.read_bytes()).hexdigest()
    artifact = root / "old-artifacts"
    snapshot = artifact / "run/cache/model/models--Example--Model/snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "tiny.bin").write_bytes(b"model-byte")
    prepared = artifact / "run/state/prepared-inputs" / digest / "private-dataset.jsonl"
    prepared.parent.mkdir(parents=True)
    prepared.write_bytes(content)
    rows = diagnostic._readonly_snapshot_inventory(snapshot)
    receipt = root / "old-model-receipt.json"
    receipt.write_text(json.dumps({"schema_version": "synaptic-docker-model-preparation/v1",
        "artifact_host_root": str(artifact), "snapshot": str(snapshot),
        "model": workload["configuration"]["document"]["model"], "files": rows}))
    flags.update({"worker_direct": True, "gpu_device": "0",
        "expected_direct_trainer_sha256": DIRECT_TRAINER,
        "expected_direct_manifest_sha256": DIRECT_MANIFEST,
        "expected_direct_closure_digest": DIRECT_CLOSURE,
        "artifact_root": artifact, "model_receipt": receipt,
        "expected_model_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()})
    return flags, workload_path, snapshot, prepared


def _stage_public_parent_inputs(root: Path):
    flags, workload_path, snapshot, prepared = _stage_direct_inputs(root)
    project = flags["source_root"].parent / "project"
    project.mkdir()
    lock = flags["source_lock"]
    lock_document = json.loads(lock.read_bytes())
    lock_document["sources"]["engine"]["commit"] = R6_ENGINE_COMMIT
    lock_document["sources"]["project"] = {"commit": R6_PROJECT_COMMIT}
    lock.write_text(json.dumps(lock_document, sort_keys=True, separators=(",", ":")))
    flags["expected_source_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    flags["expected_engine_commit"] = R6_ENGINE_COMMIT
    workload = json.loads(workload_path.read_bytes())
    workload["entrypoint"] = "Trainers/sft/runtime_v1.py"
    workload["execution_source"]["sources"] = {
        "engine": {"commit": R6_ENGINE_COMMIT}, "project": {"commit": R6_PROJECT_COMMIT}}
    workload_raw = json.dumps(workload, sort_keys=True, separators=(",", ":")).encode()
    workload_path.write_bytes(workload_raw)
    workload_sha = hashlib.sha256(workload_raw).hexdigest()
    fingerprint = hashlib.sha256(b"synaptic-training-workload/v1\0" + workload_raw).hexdigest()
    environment = diagnostic._context_environment(workload_raw, DIRECT_CLOSURE)
    command = ("/source/engine/Trainers/sft/runtime_v1.py",
               "--canonical-workload-file", "/source/control/workload.json",
               "--canonical-workload-control-root", "/source/control",
               "--canonical-workload-byte-count", str(len(workload_raw)),
               "--canonical-workload-sha256", workload_sha,
               "--canonical-workload-fingerprint", fingerprint)
    flags.update({"expected_workload_sha256": workload_sha, "public_parent": True,
                  "expected_project_commit": R6_PROJECT_COMMIT,
                  "expected_parent_environment_sha256": hashlib.sha256(json.dumps(
                      environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                  "expected_parent_command_sha256": hashlib.sha256(json.dumps(
                      command, separators=(",", ":")).encode()).hexdigest()})
    return flags, workload_path, snapshot, prepared, command


def test_public_parent_command_and_closed_terminal_use_real_entry(tmp_path: Path,
                                                                  monkeypatch) -> None:
    flags, _workload, snapshot, prepared, expected_command = _stage_public_parent_inputs(tmp_path)
    docker, config, _state_path = _fake_docker(tmp_path)
    original_run = diagnostic._run
    creates = []

    def run(authority, args, **kwargs):
        if args[:2] == ("image", "inspect") and "Config.Env" in args[3]:
            return subprocess.CompletedProcess(args, 0, b"[]\n", b"")
        if args[:2] == ("container", "create"):
            creates.append(args)
        if args[:2] == ("container", "inspect") and "Config.Entrypoint" in args[3]:
            created = creates[0]
            mounts = []
            for index, token in enumerate(created[:-1]):
                if token == "--mount":
                    parts = dict(part.split("=", 1) if "=" in part else (part, "")
                                 for part in created[index + 1].split(","))
                    mounts.append({"Type": "bind", "Source": parts["src"],
                                   "Destination": parts["dst"], "RW": "readonly" not in parts})
            env = [created[index + 1] for index, token in enumerate(created[:-1])
                   if token == "--env"]
            values = ([created[created.index("--entrypoint") + 1]], list(expected_command),
                      "root", "/workspace", env, "none", 16 * 1024**3, 2 * 10**9,
                      False, False, None, None, None, None, [], "runc",
                      {"Type": "json-file", "Config": {"max-size": "1m", "max-file": "1"}},
                      None, 64 * 1024**2, [], [{"Driver": "", "Count": 0,
                          "DeviceIDs": ["0"], "Capabilities": [["gpu"]], "Options": {}}],
                      "private", "",
                      mounts)
            return subprocess.CompletedProcess(args, 0,
                ("\n".join(json.dumps(value) for value in values) + "\n").encode(), b"")
        if args[:2] == ("container", "start"):
            tracking = tmp_path / "out/artifacts/run/tracking/trainer.stderr.log"
            tracking.write_bytes(
                b"OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=SYSTEM_EXIT "
                b"milestone=UNSLOTH_IMPORT\n")
        if args[:2] == ("container", "logs"):
            return subprocess.CompletedProcess(args, 0, b"", b"SFT_RUNTIME_V1_REJECTED\n")
        return original_run(authority, args, **kwargs)

    def inspect(_authority, _identifier, **_kwargs):
        status = json.loads((tmp_path / "state.json").read_bytes())["status"]
        return ("created", 0) if status == "created" else ("exited", 23)

    monkeypatch.setattr(diagnostic, "_run", run)
    monkeypatch.setattr(diagnostic, "_inspect_owned", inspect)
    report = _call(tmp_path, docker, config, **flags)
    assert report["scope"] == "public_parent"
    assert (report["category"], report["stage"]) == ("PASS", "UNSLOTH_IMPORT")
    assert report["training_executed"] is False
    assert len(creates) == 1
    command = creates[0]
    assert command[command.index(IMAGE) + 1:] == expected_command
    assert command[command.index("--entrypoint") + 1] == "/usr/bin/python3"
    assert "--workdir=/workspace" in command and "-c" not in command
    assert "--network=none" in command and "--memory=16g" in command
    mounts = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--mount"]
    assert any(f"src={snapshot}," in value and value.endswith(",readonly") for value in mounts)
    assert any(f"src={prepared}," in value and value.endswith(",readonly") for value in mounts)
    assert report["parent_identity"]["project_commit"] == R6_PROJECT_COMMIT


@pytest.mark.parametrize("drift", ["project", "workload", "environment", "command"])
def test_public_parent_drift_rejects_before_docker_or_claim(tmp_path: Path,
                                                            monkeypatch, drift: str) -> None:
    flags, workload, _snapshot, _prepared, _command = _stage_public_parent_inputs(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    if drift == "project":
        flags["expected_project_commit"] = "0" * 40
    elif drift == "workload":
        workload.write_bytes(workload.read_bytes() + b" ")
    elif drift == "environment":
        flags["expected_parent_environment_sha256"] = "0" * 64
    else:
        flags["expected_parent_command_sha256"] = "0" * 64
    monkeypatch.setattr(diagnostic, "_run", lambda *_args, **_kwargs:
                        pytest.fail("Docker touched"))
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert json.loads(state.read_text())["status"] == "absent"
    assert not (tmp_path / "out").exists()


def test_public_parent_project_drift_after_create_never_starts(tmp_path: Path,
                                                               monkeypatch) -> None:
    flags, _workload, _snapshot, _prepared, _command = _stage_public_parent_inputs(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    original_run = diagnostic._run
    calls = []

    def run(authority, args, **kwargs):
        calls.append(args[:2])
        if args[:2] == ("image", "inspect") and "Config.Env" in args[3]:
            return subprocess.CompletedProcess(args, 0, b"[]\n", b"")
        result = original_run(authority, args, **kwargs)
        if args[:2] == ("container", "create"):
            (flags["source_root"].parent / "project" / "unexpected").write_bytes(b"changed")
        return result

    monkeypatch.setattr(diagnostic, "_run", run)
    monkeypatch.setattr(diagnostic, "_require_public_parent_scope",
                        lambda *_args, **_kwargs: "a" * 64)
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert json.loads(state.read_text())["status"] == "created"
    assert ("container", "create") in calls
    assert ("container", "start") not in calls
    assert (tmp_path / "out/claim.json").exists()


def test_public_parent_label_read_rejects_parent_redirect(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    tracking = root / "run/tracking"
    tracking.mkdir(parents=True)
    label = b"OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=IMPORT milestone=TORCH_IMPORT\n"
    (tracking / "trainer.stderr.log").write_bytes(label)
    assert diagnostic._read_closed_worker_label(root) == label
    tracking.rename(root / "run/retained-tracking")
    tracking.symlink_to(root / "run/retained-tracking", target_is_directory=True)
    with pytest.raises(ValueError, match="CHILD_OUTPUT_INVALID"):
        diagnostic._read_closed_worker_label(root)


def test_public_parent_uncertain_create_consumes_claim(tmp_path: Path, monkeypatch) -> None:
    flags, _workload, _snapshot, _prepared, _command = _stage_public_parent_inputs(tmp_path)
    docker, config, _state_path = _fake_docker(tmp_path)
    original_run = diagnostic._run

    def run(authority, args, **kwargs):
        if args[:2] == ("image", "inspect") and "Config.Env" in args[3]:
            return subprocess.CompletedProcess(args, 0, b"[]\n", b"")
        if args[:2] == ("container", "create"):
            raise TimeoutError()
        return original_run(authority, args, **kwargs)

    monkeypatch.setattr(diagnostic, "_run", run)
    with pytest.raises(ValueError, match="CREATE_UNCERTAIN"):
        _call(tmp_path, docker, config, **flags)
    assert (tmp_path / "out/claim.json").exists()
    with pytest.raises(ValueError, match="OUTPUT_INVALID"):
        _call(tmp_path, docker, config, **flags)


def test_direct_derivative_keeps_exact_prefix_and_verifies_closure(tmp_path: Path) -> None:
    flags, _workload_path, _snapshot, _prepared = _stage_direct_inputs(tmp_path)
    identity = diagnostic._stage_direct_source(flags["source_root"], tmp_path / "derived",
        expected_trainer=DIRECT_TRAINER, expected_manifest=DIRECT_MANIFEST,
        expected_closure=DIRECT_CLOSURE)
    assert identity["closure_digest"] == DIRECT_CLOSURE
    original = (flags["source_root"] / "Trainers/sft/train_sft.py").read_bytes()
    actual = (tmp_path / "derived/engine/Trainers/sft/train_sft.py").read_bytes()
    boundary = diagnostic._trainer_prefix_boundary(original)
    assert actual == b"".join(original.splitlines(keepends=True)[:boundary + 1]) + b"raise SystemExit(42)\n"
    assert b"from unsloth import" not in actual
    assert b"TRAIN_CALL" not in actual
    assert b"model_loader" not in actual
    manifest = tmp_path / "derived/control/offline-sft-worker-v1.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == DIRECT_MANIFEST
    context = diagnostic._worker_context_identity(tmp_path / "derived/engine",
        flags["expected_workload_sha256"], flags["expected_runtime_sha256"],
        DIRECT_TRAINER, DIRECT_CLOSURE, direct_stop=True)
    probe = diagnostic._worker_probe(context, DIRECT_CLOSURE, direct_mode=True,
        prepared_sha256=hashlib.sha256(b'{"text":"stub"}\n').hexdigest(),
        prepared_size=len(b'{"text":"stub"}\n'))
    ast.parse(probe)
    assert "EXPECTED_" not in probe and "DIRECT_MODE" not in probe
    assert "[sys.executable,'-I',str(worker_path),*arguments]" in probe
    assert "stdout=subprocess.DEVNULL" in probe and "stderr=subprocess.PIPE" in probe
    assert "process.stderr.read(257)" in probe
    actual += b" "
    (tmp_path / "derived/engine/Trainers/sft/train_sft.py").write_bytes(actual)
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        diagnostic._direct_source_identity(tmp_path / "derived/engine",
            expected_trainer=DIRECT_TRAINER, expected_manifest=DIRECT_MANIFEST,
            expected_closure=DIRECT_CLOSURE)


@pytest.mark.parametrize("import_error", [False, True])
def test_real_direct_isolated_child_stops_at_fixed_marker(tmp_path: Path,
                                                           import_error: bool) -> None:
    engine = tmp_path / "engine"
    worker = engine / "tuner/runtime/offline_sft_worker.py"
    worker.parent.mkdir(parents=True)
    trainer = engine / "Trainers/sft/train_sft.py"
    trainer.parent.mkdir(parents=True)
    worker.write_text("""import runpy,sys
from pathlib import Path
phase=None
def mark(value):
 global phase
 phase=value
def main():
 if sys.flags.isolated!=1 or sys.argv[1:]!=['--','--max-steps','1']:
  return 3
 root=Path(__file__).parents[2]
 trainer=root/'Trainers/sft/train_sft.py'
 sys.path[:0]=[str(trainer.parent),str(root)]
 sys.argv=[str(trainer),'--max-steps','1']
 try: runpy.run_path(str(trainer),run_name='__main__',init_globals={'__packaged_runtime_phase__':mark})
 except SystemExit:
  print('OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=SYSTEM_EXIT milestone='+str(phase),file=sys.stderr)
  return 2
 except ImportError:
  print('OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=IMPORT milestone='+str(phase),file=sys.stderr)
  return 2
 return 0
if __name__=='__main__': raise SystemExit(main())
""")
    original = b"""import sys
def _mark_packaged_runtime_phase(value):
 marker=globals().get('__packaged_runtime_phase__')
 if marker is not None: marker(value)
assert __name__=='__main__'
assert sys.modules['__main__'].__dict__ is globals()
_mark_packaged_runtime_phase("BOOTSTRAP_ENV")
_mark_packaged_runtime_phase("TORCH_IMPORT")
import torch  # noqa: E402
_mark_packaged_runtime_phase("UNSLOTH_IMPORT")
from unsloth import is_bfloat16_supported  # noqa: E402
raise RuntimeError('TRAINER_FELL_THROUGH')
"""
    boundary = diagnostic._trainer_prefix_boundary(original)
    trainer.write_bytes(b"".join(original.splitlines(keepends=True)[:boundary + 1])
                        + diagnostic._DIRECT_STOP)
    (trainer.parent / "torch.py").write_text(
        "from pathlib import Path\nPath(__file__).with_name('TORCH_IMPORTED').write_text('yes')\n"
        + ("raise ImportError('private detail')\n" if import_error else ""))
    (trainer.parent / "unsloth.py").write_text(
        "from pathlib import Path\nPath(__file__).with_name('UNSLOTH_IMPORTED').write_text('bad')\n")
    completed = subprocess.run([sys.executable, "-I", str(worker), "--", "--max-steps", "1"],
        cwd=tmp_path, env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
                           "PYTHONSAFEPATH": "1"}, capture_output=True, timeout=20)
    assert completed.returncode == 2 and completed.stdout == b""
    expected = (b"OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=IMPORT milestone=TORCH_IMPORT\n"
                if import_error else
                b"OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=SYSTEM_EXIT milestone=UNSLOTH_IMPORT\n")
    assert completed.stderr == expected
    assert (trainer.parent / "TORCH_IMPORTED").exists()
    assert not (trainer.parent / "UNSLOTH_IMPORTED").exists()


def test_direct_drift_rejects_before_create_and_claim_is_not_replayed(
        tmp_path: Path, monkeypatch) -> None:
    flags, _workload, snapshot, _prepared = _stage_direct_inputs(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "_run", lambda *args, **kwargs:
                        pytest.fail("Docker touched"))
    (snapshot / "tiny.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert json.loads(state.read_text())["status"] == "absent"
    assert not (tmp_path / "out").exists()

    flags, _workload, _snapshot, _prepared = _stage_direct_inputs(tmp_path / "fresh")
    (tmp_path / "fresh").chmod(0o700)
    docker, config, state = _fake_docker(tmp_path / "fresh")
    def uncertain_create(_authority, args, **_kwargs):
        if args[:2] == ("container", "create"):
            raise TimeoutError()
        if args[0] == "info":
            return subprocess.CompletedProcess(args, 0, DAEMON.encode() + b"\n", b"")
        if args[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(args, 0, IMAGE.encode() + b"\n", b"")
        pytest.fail("unexpected Docker command")
    monkeypatch.setattr(diagnostic, "_run", uncertain_create)
    with pytest.raises(ValueError, match="CREATE_UNCERTAIN"):
        _call(tmp_path / "fresh", docker, config, **flags)
    assert (tmp_path / "fresh/out/claim.json").exists()
    with pytest.raises(ValueError, match="OUTPUT_INVALID"):
        _call(tmp_path / "fresh", docker, config, **flags)


def test_direct_derivative_drift_after_create_never_starts(tmp_path: Path,
                                                           monkeypatch) -> None:
    flags, _workload, _snapshot, _prepared = _stage_direct_inputs(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    original_run = diagnostic._run
    calls = []

    def mutating_run(authority, args, **kwargs):
        calls.append(args[:2])
        result = original_run(authority, args, **kwargs)
        if args[:2] == ("container", "create"):
            trainer = tmp_path / "out/derived-source/engine/Trainers/sft/train_sft.py"
            trainer.write_bytes(trainer.read_bytes() + b" ")
        return result

    monkeypatch.setattr(diagnostic, "_run", mutating_run)
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert json.loads(state.read_text())["status"] == "created"
    assert ("container", "create") in calls
    assert ("container", "start") not in calls
    assert (tmp_path / "out/claim.json").exists()


def test_direct_fake_docker_scope_and_readonly_nested_mounts(tmp_path: Path,
                                                             monkeypatch) -> None:
    flags, _workload, snapshot, prepared = _stage_direct_inputs(tmp_path)
    identity = diagnostic._source_identity(flags["source_root"], flags["source_lock"],
        flags["expected_source_lock_sha256"], flags["expected_manifest_sha256"],
        flags["expected_closure_digest"], R4_ENGINE_COMMIT)
    assert identity["engine_commit"] == R4_ENGINE_COMMIT
    assert diagnostic._worker_context_identity(flags["source_root"],
        flags["expected_workload_sha256"], flags["expected_runtime_sha256"],
        flags["expected_trainer_sha256"], flags["expected_closure_digest"])[
            "workload_sha256"] == flags["expected_workload_sha256"]
    docker, config, state = _fake_docker(tmp_path)
    original_run = diagnostic._run
    creates = []

    def run(authority, args, **kwargs):
        if args[:2] == ("container", "create"):
            creates.append(args)
        if args[:2] == ("container", "logs"):
            payload = {"schema_version": "synaptic-docker-torch-import-child/v1",
                "category": "PASS", "stage": "UNSLOTH_IMPORT", "context": {},
                "boundary": "BEFORE_UNSLOTH"}
            return subprocess.CompletedProcess(args, 0,
                (json.dumps(payload) + "\n").encode(), b"")
        return original_run(authority, args, **kwargs)

    def inspect(_authority, _identifier, **_kwargs):
        return ("created", 0) if json.loads(state.read_text())["status"] == "created" else ("exited", 0)

    monkeypatch.setattr(diagnostic, "_run", run)
    monkeypatch.setattr(diagnostic, "_inspect_owned", inspect)
    report = _call(tmp_path, docker, config, **flags)
    assert report["scope"] == "worker_direct" and report["stage"] == "UNSLOTH_IMPORT"
    assert report["source_identity"]["engine_commit"] == R4_ENGINE_COMMIT
    assert report["direct_identity"]["manifest_sha256"] == DIRECT_MANIFEST
    assert len(creates) == 1
    command = creates[0]
    assert "--network=none" in command and "--memory=16g" in command
    assert "--cpus=2" in command and command[command.index("--gpus") + 1] == "device=0"
    assert "--read-only" not in command and "--cap-drop=ALL" not in command
    mounts = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--mount"]
    assert any(f"src={snapshot}," in value and value.endswith(",readonly") for value in mounts)
    assert any(f"src={prepared}," in value and value.endswith(",readonly") for value in mounts)
    assert any("derived-source,dst=/source,readonly" in value for value in mounts)
    assert (tmp_path / "out/claim.json").exists()


@pytest.mark.parametrize("supplied", [None, "0d515ae1ae235d0858802228ce4af8dba8fccd03"])
def test_direct_r4_source_commit_requires_correct_explicit_pin_before_docker(
        tmp_path: Path, monkeypatch, supplied: str | None) -> None:
    flags, _workload, _snapshot, _prepared = _stage_direct_inputs(tmp_path)
    if supplied is None:
        del flags["expected_engine_commit"]
    else:
        flags["expected_engine_commit"] = supplied
    docker, config, state = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "_run",
                        lambda *_args, **_kwargs: pytest.fail("Docker touched"))
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert json.loads(state.read_text())["status"] == "absent"
    assert not (tmp_path / "out").exists()


def test_legacy_guarded_mode_keeps_r2_commit_default(tmp_path: Path, monkeypatch) -> None:
    flags, _document, _manifest = _stage_source(tmp_path)
    lock = flags["source_lock"]
    source_lock = json.loads(lock.read_bytes())
    source_lock["sources"]["engine"]["commit"] = R4_ENGINE_COMMIT
    lock.write_text(json.dumps(source_lock))
    flags["expected_source_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    docker, config, state = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "_run",
                        lambda *_args, **_kwargs: pytest.fail("Docker touched"))
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert json.loads(state.read_text())["status"] == "absent"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("expire_during_enumeration", [False, True])
def test_direct_inventory_enumeration_rejects_before_stat_hash_or_create(
        tmp_path: Path, monkeypatch, expire_during_enumeration: bool) -> None:
    flags, _workload, snapshot, _prepared = _stage_direct_inputs(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    snapshot_identity = (snapshot.stat().st_dev, snapshot.stat().st_ino)
    original_scandir = diagnostic.os.scandir
    original_stat = diagnostic.os.stat
    original_open = diagnostic.os.open
    yielded = 0
    clock = [0.0]

    def is_snapshot_fd(fd):
        info = diagnostic.os.fstat(fd)
        return (info.st_dev, info.st_ino) == snapshot_identity

    class FakeScandir:
        def __enter__(self):
            def entries():
                nonlocal yielded
                for index in range(4096):
                    yielded += 1
                    if expire_during_enumeration and yielded == 1:
                        clock[0] = 301.0
                    yield type("Entry", (), {"name": f"unexpected-{index:04d}"})()
            return entries()

        def __exit__(self, *_args):
            return False

    def scandir(path):
        return FakeScandir() if isinstance(path, int) and is_snapshot_fd(path) else original_scandir(path)

    def stat(path, *args, **kwargs):
        fd = kwargs.get("dir_fd")
        if fd is not None and is_snapshot_fd(fd):
            pytest.fail("model leaf was statted after enumeration should reject")
        return original_stat(path, *args, **kwargs)

    def open_file(path, flags, *args, **kwargs):
        fd = kwargs.get("dir_fd")
        if fd is not None and is_snapshot_fd(fd):
            pytest.fail("model leaf was opened for hashing")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(diagnostic.os, "scandir", scandir)
    monkeypatch.setattr(diagnostic.os, "stat", stat)
    monkeypatch.setattr(diagnostic.os, "open", open_file)
    if expire_during_enumeration:
        monkeypatch.setattr(diagnostic.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(diagnostic, "_run", lambda *args, **kwargs:
                        pytest.fail("Docker touched after inventory rejection"))
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert yielded == (1 if expire_during_enumeration else 321)
    assert json.loads(state.read_text())["status"] == "absent"
    assert not (tmp_path / "out").exists()


def test_worker_prefix_excludes_unsloth_and_all_trainer_calls() -> None:
    raw = (ROOT / "Trainers/sft/train_sft.py").read_bytes()
    boundary = diagnostic._trainer_prefix_boundary(raw)
    prefix = "".join(raw.decode().splitlines(keepends=True)[:boundary])
    tree = ast.parse(prefix)
    assert isinstance(tree.body[-1], ast.Import)
    assert [alias.name for alias in tree.body[-1].names] == ["torch"]
    assert "UNSLOTH_IMPORT" not in prefix
    assert "from unsloth import" not in prefix
    assert "TRAIN_CALL" not in prefix
    assert "MODEL_SNAPSHOT" not in prefix
    assert "if __name__" not in prefix
    for old, new in (('"TORCH_IMPORT"', '"UNSLOTH_IMPORT"'),
                     ('import torch  # noqa: E402', 'from unsloth import x')):
        with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
            diagnostic._trainer_prefix_boundary(raw.replace(old.encode(), new.encode(), 1))


def test_worker_context_source_and_workload_drift_reject_before_effect(tmp_path: Path,
                                                                      monkeypatch) -> None:
    flags, workload = _stage_worker_context(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "_run", lambda *args, **kwargs: pytest.fail("Docker touched"))
    workload.write_bytes(workload.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert json.loads(state.read_text())["status"] == "absent"
    assert not (tmp_path / "out").exists()


def test_worker_context_program_is_isolated_and_excludes_trainer_main(tmp_path: Path) -> None:
    flags, _workload = _stage_worker_context(tmp_path)
    identity = diagnostic._worker_context_identity(
        flags["source_root"], flags["expected_workload_sha256"],
        flags["expected_runtime_sha256"], flags["expected_trainer_sha256"],
        flags["expected_closure_digest"])
    probe = diagnostic._worker_probe(identity, flags["expected_closure_digest"])
    ast.parse(probe)
    ast.parse(diagnostic._WORKER_PREFIX_CHILD.replace(
        "EXPECTED_TRAINER_SHA256", flags["expected_trainer_sha256"]).replace(
        "EXPECTED_SUPPLIED_ENV_SHA256", identity["child_env_sha256"]))
    assert "EXPECTED_" not in probe
    assert "CHILD_SOURCE_B64" not in probe
    assert "process=subprocess.Popen([sys.executable,'-I','-c'" in probe
    assert "runpy.run_path(str(root/'tuner/runtime/offline_sft_worker.py')" in diagnostic._WORKER_PREFIX_CHILD
    assert "runpy.run_path(str(trainer)" not in diagnostic._WORKER_PREFIX_CHILD
    assert "exec(compile(tree,str(trainer),'exec'),namespace)" in diagnostic._WORKER_PREFIX_CHILD
    assert "lines[:positions[2]]" in diagnostic._WORKER_PREFIX_CHILD
    assert identity["child_env_sha256"] != hashlib.sha256(b"").hexdigest()
    assert identity["child_env_lc_ctype_sha256"] != identity["child_env_sha256"]


def _entry_fixture(tmp_path: Path, *, disable_trace: bool = False,
                   import_error: bool = False):
    """Use real isolated runpy and a non-ML Torch stub at the stop boundary."""
    engine = tmp_path / "engine"
    trainer = engine / "Trainers/sft/train_sft.py"
    trainer.parent.mkdir(parents=True)
    worker = engine / "tuner/runtime/offline_sft_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("""import runpy,sys
from pathlib import Path
def mark(value):
 pass
def main():
 if sys.argv[1:]!=['--','--max-steps','1','--no-load-in-4bit']:
  print('OFFLINE_SFT_WORKER_REJECTED phase=ARGUMENTS category=REJECTION',file=sys.stderr)
  return 2
 root=Path(__file__).parents[2]
 trainer=root/'Trainers/sft/train_sft.py'
 sys.path[:0]=[str(trainer.parent),str(root)]
 sys.argv=[str(trainer),*sys.argv[2:]]
 try:
  runpy.run_path(str(trainer),run_name='__main__',init_globals={'__packaged_runtime_phase__':mark})
 except ImportError:
  print('OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=IMPORT milestone=TORCH_IMPORT',file=sys.stderr)
  return 2
 except BaseException:
  print('OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=OTHER milestone=TORCH_IMPORT',file=sys.stderr)
  return 2
 return 0
if __name__=='__main__':
 raise SystemExit(main())
""")
    trainer.write_text("""import os,sys
from pathlib import Path
def _mark_packaged_runtime_phase(value):
 marker=globals().get('__packaged_runtime_phase__')
 if marker is not None: marker(value)
assert __name__=='__main__'
assert sys.modules['__main__'].__dict__ is globals()
_mark_packaged_runtime_phase("BOOTSTRAP_ENV")
os.environ['TORCH_COMPILE_DISABLE']='1'
_mark_packaged_runtime_phase("TORCH_IMPORT")
import torch  # noqa: E402
_mark_packaged_runtime_phase("UNSLOTH_IMPORT")
from unsloth import is_bfloat16_supported  # noqa: E402
Path(__file__).with_name('TRAINER_FELL_THROUGH').write_text('bad')
""")
    (trainer.parent / "torch.py").write_text(
        "from pathlib import Path\nimport sys\n"
        + ("import types\nsys.settrace(None)\n"
           "sys.modules['unsloth']=types.ModuleType('unsloth')\n"
           "sys.modules['unsloth'].is_bfloat16_supported=True\n"
           if disable_trace else "")
        + "Path(__file__).with_name('TORCH_IMPORTED').write_text('yes')\n"
        + ("raise ImportError('private detail')\n" if import_error else ""))
    (trainer.parent / "unsloth.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('UNSLOTH_IMPORTED').write_text('bad')\n"
        "is_bfloat16_supported=True\n")
    environment = {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
                   "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    observed_locale = {**environment, "LC_CTYPE": "C.UTF-8"}
    arguments = ["--", "--max-steps", "1", "--no-load-in-4bit"]
    digest = lambda obj: hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = diagnostic._WORKER_ENTRY_CHILD
    for key, value in {
        "DIAGNOSTIC_ENGINE_ROOT": str(engine),
        "DIAGNOSTIC_STDERR_PATH": str(tmp_path / "worker-entry.stderr"),
        "EXPECTED_TRAINER_SHA256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
        "EXPECTED_WORKER_SHA256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "EXPECTED_SUPPLIED_ENV_SHA256": digest(environment),
        "EXPECTED_LC_ENV_SHA256": digest(observed_locale),
        "EXPECTED_TRAINER_ARGV_SHA256": digest([str(trainer), *arguments[1:]]),
    }.items():
        source = source.replace(key, value)
    return trainer, source, environment, arguments


@pytest.mark.parametrize("disable_trace", [False, True])
def test_real_isolated_worker_entry_stops_before_unsloth(tmp_path: Path,
                                                         disable_trace: bool) -> None:
    trainer, source, environment, arguments = _entry_fixture(
        tmp_path, disable_trace=disable_trace)
    with (tmp_path / "worker-entry.stderr").open("xb") as stderr:
        completed = subprocess.run([sys.executable, "-I", "-c", source, *arguments],
            cwd=tmp_path, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=stderr, timeout=20)
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["boundary"] == "BEFORE_UNSLOTH"
    if disable_trace:
        assert result["category"] == "OTHER"
    else:
        assert result["category"] == "PASS"
        assert result["stage"] == "TORCH_IMPORT"
        assert result["context"]["runpy_main_bound"] is True
        assert (result["context"]["environment_before_torch_sha256"] ==
                result["context"]["environment_after_torch_sha256"])
        assert (result["context"]["environment_before_torch_sha256"] !=
                result["context"]["environment_observed_sha256"])
    assert (trainer.parent / "TORCH_IMPORTED").exists()
    assert not (trainer.parent / "UNSLOTH_IMPORTED").exists()
    assert not (trainer.parent / "TRAINER_FELL_THROUGH").exists()


def test_real_isolated_worker_entry_reports_closed_torch_import_failure(
        tmp_path: Path) -> None:
    trainer, source, environment, arguments = _entry_fixture(
        tmp_path, import_error=True)
    with (tmp_path / "worker-entry.stderr").open("xb") as stderr:
        completed = subprocess.run([sys.executable, "-I", "-c", source, *arguments],
            cwd=tmp_path, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=stderr, timeout=20)
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result == {"schema_version": "synaptic-docker-torch-import-child/v1",
        "category": "IMPORT", "stage": "TORCH_IMPORT", "context": {},
        "boundary": "BEFORE_UNSLOTH"}
    assert (trainer.parent / "TORCH_IMPORTED").exists()
    assert not (trainer.parent / "UNSLOTH_IMPORTED").exists()


def test_isolated_worker_entry_rejects_trainer_drift_before_worker(
        tmp_path: Path) -> None:
    trainer, source, environment, arguments = _entry_fixture(tmp_path)
    trainer.write_bytes(trainer.read_bytes() + b"\n")
    with (tmp_path / "worker-entry.stderr").open("xb") as stderr:
        completed = subprocess.run([sys.executable, "-I", "-c", source, *arguments],
            cwd=tmp_path, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=stderr, timeout=20)
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["category"] == "OTHER" and result["stage"] == "ENTRY_ADMISSION"
    assert result["context"] == {}
    assert not (trainer.parent / "TORCH_IMPORTED").exists()


def test_worker_entry_source_and_mode_bind_before_docker(tmp_path: Path,
                                                         monkeypatch) -> None:
    flags, workload_path = _stage_worker_context(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "_run", lambda *args, **kwargs:
                        pytest.fail("Docker touched"))
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, worker_entry=True, **flags)
    assert json.loads(state.read_text())["status"] == "absent"
    workload = json.loads(workload_path.read_text())
    workload["configuration"]["document"]["dataset"] = {
        "ref": "prepared://sha256/" + "a" * 64}
    workload_path.write_text(json.dumps(workload, sort_keys=True, separators=(",", ":")))
    flags["expected_workload_sha256"] = hashlib.sha256(workload_path.read_bytes()).hexdigest()
    identity = diagnostic._worker_context_identity(
        flags["source_root"], flags["expected_workload_sha256"],
        flags["expected_runtime_sha256"], flags["expected_trainer_sha256"],
        flags["expected_closure_digest"])
    probe = diagnostic._worker_probe(identity, flags["expected_closure_digest"],
                                     entry_mode=True)
    ast.parse(probe)
    assert "EXPECTED_" not in probe and "ENTRY_MODE" not in probe
    match = re.search(r"base64\.b64decode\('([A-Za-z0-9+/=]+)'\)", probe)
    assert match is not None
    child = base64.b64decode(match.group(1)).decode()
    ast.parse(child)
    assert "DIAGNOSTIC_ENGINE_ROOT" not in child
    assert "DIAGNOSTIC_STDERR_PATH" not in child
    assert "runpy.run_path(str(worker),run_name='__main__')" in child
    assert "sys.settrace(trace)" in child and "_UnslothBarrier" in child
    trainer = flags["source_root"] / "Trainers/sft/train_sft.py"
    trainer.write_bytes(trainer.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, worker_entry=True, **flags)
    assert json.loads(state.read_text())["status"] == "absent"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation,expected", [
    ("none", "LC_CTYPE_C_UTF8"),
    ("no_coercion", "NONE"),
    ("extra", "REJECTED"),
    ("changed", "REJECTED"),
    ("locale_value", "REJECTED"),
])
def test_real_isolated_python_startup_environment_evidence(mutation: str,
                                                           expected: str) -> None:
    """Run the exact child classifier after a real -I interpreter startup."""
    source = diagnostic._WORKER_PREFIX_CHILD
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and
                    node.name == "_environment_evidence")
    lines = source.splitlines(keepends=True)
    classifier = "".join(lines[function.lineno - 1:function.end_lineno])
    program = ("import hashlib,json,os,sys\n" + classifier +
        "\nsupplied=json.loads(sys.stdin.buffer.read(8193))\n" +
        ("os.environ['UNEXPECTED_DIAGNOSTIC_KEY']='x'\n" if mutation == "extra" else "") +
        ("os.environ['PATH']='/tampered'\n" if mutation == "changed" else "") +
        ("os.environ['LC_CTYPE']='C.utf8'\n" if mutation == "locale_value" else "") +
        "try:\n evidence=_environment_evidence(supplied,dict(os.environ))\n"
        "except BaseException:\n print(json.dumps({'status':'REJECTED'}))\n"
        "else:\n print(json.dumps({'status':'PASS','delta':evidence['startup_env_delta'],"
        "'supplied_sha256':evidence['environment_supplied_sha256'],"
        "'observed_sha256':evidence['environment_observed_sha256']}))\n")
    supplied = {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1"}
    if mutation == "no_coercion":
        supplied["LC_ALL"] = "C"
    result = subprocess.run([sys.executable, "-I", "-c", program],
        input=json.dumps(supplied, sort_keys=True, separators=(",", ":")),
        text=True, capture_output=True, env=supplied, timeout=5, check=True)
    assert result.stderr == ""
    report = json.loads(result.stdout)
    if expected == "REJECTED":
        assert report == {"status": "REJECTED"}
    else:
        assert report["status"] == "PASS"
        assert report["delta"] == expected
        expected_sha = hashlib.sha256(json.dumps(
            supplied, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert report["supplied_sha256"] == expected_sha
        assert (report["observed_sha256"] == expected_sha) == (expected == "NONE")


def test_worker_context_requires_independent_r2_hashes_before_docker(tmp_path: Path,
                                                                      monkeypatch) -> None:
    flags, _workload = _stage_worker_context(tmp_path)
    flags.pop("expected_workload_sha256")
    docker, config, state = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "_run", lambda *args, **kwargs: pytest.fail("Docker touched"))
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert json.loads(state.read_text())["status"] == "absent"
    assert not (tmp_path / "out").exists()


def test_worker_context_rechecks_drift_after_create_without_start(tmp_path: Path,
                                                                   monkeypatch) -> None:
    flags, workload = _stage_worker_context(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    original = diagnostic._inspect_owned
    changed = False

    def inspect(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed and result[0] == "created":
            workload.write_bytes(workload.read_bytes() + b" ")
            changed = True
        return result

    monkeypatch.setattr(diagnostic, "_inspect_owned", inspect)
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_INVALID"):
        _call(tmp_path, docker, config, **flags)
    assert json.loads(state.read_text())["status"] == "created"
    assert (tmp_path / "out/claim.json").exists()
    assert not (tmp_path / "out/result.json").exists()
    with pytest.raises(ValueError, match="OUTPUT_INVALID"):
        _call(tmp_path, docker, config, **flags)


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


def test_gpu_visibility_is_fixed_guarded_import_only(tmp_path: Path, monkeypatch) -> None:
    flags, _, _ = _stage_source(tmp_path)
    docker, config, state = _fake_docker(tmp_path)
    monkeypatch.setattr(diagnostic, "_RUN_SECONDS", 2)
    original_inspect = diagnostic._inspect_owned
    def inspect(*args, **kwargs):
        status, code = original_inspect(*args, **kwargs)
        if status == "running":
            _state(state, status="exited")
        return status, code
    monkeypatch.setattr(diagnostic, "_inspect_owned", inspect)
    report = _call(tmp_path, docker, config, gpu_device="0", **flags)
    claim = json.loads((tmp_path / "out/claim.json").read_text())
    actual = json.loads(state.read_text())
    assert claim["gpu_device"] == "0"
    assert actual["gpu_requested"] is True and actual["gpu_probe_unhidden"] is True
    assert actual["stops"] == 0
    assert report["category"] == "PASS" and report["stage"] == "TORCH_IMPORT"
    assert report["gpu_device"] == "0" and report["gpu_used"] is None
    assert report["training_executed"] is False


def test_gpu_scope_rejects_before_effect_or_start(tmp_path: Path) -> None:
    docker, config, state = _fake_docker(tmp_path)
    with pytest.raises(ValueError, match="GPU_SCOPE_INVALID"):
        _call(tmp_path, docker, config, gpu_device="0")
    assert json.loads(state.read_text())["status"] == "absent"
    assert not (tmp_path / "out").exists()
    flags, _, _ = _stage_source(tmp_path)
    with pytest.raises(ValueError, match="GPU_SCOPE_INVALID"):
        _call(tmp_path, docker, config, gpu_device="1", **flags)
    assert json.loads(state.read_text())["status"] == "absent"
    _state(state, mode="wrong_gpu_request")
    with pytest.raises(ValueError, match="GPU_SCOPE_INVALID"):
        _call(tmp_path, docker, config, gpu_device="0", **flags)
    assert json.loads(state.read_text())["status"] == "created"
    assert json.loads(state.read_text())["stops"] == 0
    assert (tmp_path / "out/claim.json").exists()
    assert not (tmp_path / "out/result.json").exists()


def test_invalid_gpu_cli_value_is_closed(tmp_path: Path, capsys) -> None:
    exit_code = diagnostic.main([
        "--docker", str(tmp_path / "docker"), "--docker-config", str(tmp_path / "config"),
        "--release", str(tmp_path / "release"), "--output-dir", str(tmp_path / "out"),
        "--daemon-id", DAEMON, "--expected-release-sha256", "a" * 64,
        "--gpu-device", "private-value",
    ])
    captured = capsys.readouterr()
    assert exit_code == 2 and captured.out == ""
    assert json.loads(captured.err) == {"schema_version": diagnostic.SCHEMA,
        "status": "REJECTED", "reason_code": "GPU_SCOPE_INVALID"}


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
    monkeypatch.setattr(diagnostic, "_RUN_SECONDS", 1)
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
        flags["expected_closure_digest"], diagnostic._LEGACY_ENGINE_COMMIT)[
            "closure_digest"] == flags["expected_closure_digest"]
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
