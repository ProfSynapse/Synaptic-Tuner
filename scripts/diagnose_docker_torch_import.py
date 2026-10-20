"""Bounded, credential-free Torch import diagnosis for an exact packaged image.

This is a diagnostic, never training or qualification authority.  It retains
the new container and writes only closed results to a fresh private directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path


SCHEMA = "synaptic-docker-torch-import-diagnostic/v1"
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CID = re.compile(r"[0-9a-f]{64}\Z")
_CATEGORIES = frozenset({"PASS", "MODULE_NOT_FOUND", "IMPORT", "OS", "RUNTIME", "OTHER"})
_REASONS = frozenset({"DAEMON_ID_INVALID", "INPUT_INVALID", "DOCKER_INVALID",
    "DOCKER_CHANGED", "DOCKER_COMMAND_REJECTED", "DIAGNOSTIC_OUTPUT_INVALID",
    "RELEASE_INVALID", "RELEASE_HASH_INVALID", "OUTPUT_INVALID", "SOURCE_INVALID", "DAEMON_INVALID", "IMAGE_INVALID",
    "CHILD_OUTPUT_INVALID", "CONTAINER_ID_INVALID", "CONTAINER_IMAGE_INVALID",
    "IMAGE_CHANGED", "SOURCE_IDENTITY_INVALID", "GPU_SCOPE_INVALID", "CREATE_UNCERTAIN", "START_UNCERTAIN",
    "DEADLINE_UNCERTAIN", "DEADLINE_STOPPED", "STOP_UNCONFIRMED", "CONTAINER_STATE_INVALID"})
_CLI_BUDGET = 8192
_RUN_SECONDS = 120
_HIDDEN_GPU_ENV = ",'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'"
_INSPECT = "{{json .Id}}\n{{json .Name}}\n{{json .Image}}\n{{json .Config.Labels}}\n{{json .State.Status}}\n{{json .State.ExitCode}}"

# This fixed program never formats an exception, inspects its attributes, or
# imports an engine/trainer module.  The exact packaged interpreter imports
# Torch under the isolated, offline child environment.
_PROBE = """import json,os,sys
report_fd=os.dup(1)
null_fd=os.open('/dev/null',os.O_WRONLY)
os.dup2(null_fd,1);os.dup2(null_fd,2);os.close(null_fd)
os.environ.clear()
os.environ.update({'PATH':os.path.dirname(sys.executable)+':/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','TMPDIR':'/tmp','LC_ALL':'C.UTF-8','PYTHONNOUSERSITE':'1','PYTHONSAFEPATH':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_DISABLED':'true','CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'})
try:
 import torch
 category='PASS'
except ModuleNotFoundError:
 category='MODULE_NOT_FOUND'
except ImportError:
 category='IMPORT'
except OSError:
 category='OS'
except RuntimeError:
 category='RUNTIME'
except BaseException:
 category='OTHER'
os.write(report_fd,(json.dumps({'schema_version':'synaptic-docker-torch-import-child/v1','category':category},sort_keys=True,separators=(',',':'))+'\\n').encode('ascii'))
os.close(report_fd)
"""

_GUARDED_PROBE = """import json,os,sys
report_fd=os.dup(1)
null_fd=os.open('/dev/null',os.O_WRONLY)
os.dup2(null_fd,1);os.dup2(null_fd,2);os.close(null_fd)
from pathlib import Path
os.environ.clear()
os.environ.update({'PATH':os.path.dirname(sys.executable)+':/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','TMPDIR':'/tmp','LC_ALL':'C.UTF-8','PYTHONNOUSERSITE':'1','PYTHONSAFEPATH':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_DISABLED':'true','CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void'})
root=Path('/source/engine')
sys.path.insert(0,str(root))
stage='IMPORT_WORKER'
try:
 from tuner.runtime.offline_sft_worker import load_offline_sft_worker_closure,install_owned_module_guard,verify_loaded_owned_module_origins
 stage='READ_MANIFEST'
 manifest=Path('/source/control/offline-sft-worker-v1.json')
 declared='EXPECTED_CLOSURE_DIGEST'
 stage='LOAD_CLOSURE'
 closure=load_offline_sft_worker_closure(manifest,expected_digest=declared,engine_root=root)
 stage='VERIFY_ORIGINS'
 verify_loaded_owned_module_origins(closure,engine_root=root)
 stage='INSTALL_GUARD'
 install_owned_module_guard(closure,engine_root=root)
 sys.path[:0]=[str(root),str(root/'Trainers/sft/src'),str(root/'Trainers/sft'),str(root)]
 stage='BOOTSTRAP_ENV'
 from shared.env_bootstrap import init_trainer_env
 init_trainer_env()
 stage='TORCH_IMPORT'
 import torch
 category='PASS'
except ModuleNotFoundError:
 category='MODULE_NOT_FOUND'
except ImportError:
 category='IMPORT'
except OSError:
 category='OS'
except RuntimeError:
 category='RUNTIME'
except BaseException:
 category='OTHER'
os.write(report_fd,(json.dumps({'schema_version':'synaptic-docker-torch-import-child/v1','category':category,'stage':stage},sort_keys=True,separators=(',',':'))+'\\n').encode('ascii'))
os.close(report_fd)
"""


def _read(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("INPUT_INVALID")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not raw or len(raw) > maximum:
        raise ValueError("INPUT_INVALID")
    return raw


def _identity(path: Path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


def _authority(docker: Path, docker_config: Path):
    if (not docker.is_absolute() or docker.is_symlink() or not docker.is_file()
            or not docker_config.is_absolute() or docker_config.is_symlink()
            or not docker_config.is_dir() or any(docker_config.iterdir())):
        raise ValueError("DOCKER_INVALID")
    return (docker, docker_config, _identity(docker), _identity(docker_config))


def _run(authority, args: tuple[str, ...], *, timeout: float = 15):
    """Bound both CLI lifetime and combined bytes before they enter memory."""
    docker, config, exe_stat, config_stat = authority
    if (_identity(docker) != exe_stat or _identity(config) != config_stat):
        raise ValueError("DOCKER_CHANGED")
    process = subprocess.Popen((str(docker), "--config", str(config),
        "--host", "unix:///var/run/docker.sock", *args),
        env={"PATH": os.defpath, "DOCKER_CONTENT_TRUST": "1"},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)
    output = {process.stdout: bytearray(), process.stderr: bytearray()}
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for stream in output:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("DOCKER_COMMAND_REJECTED")
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fileobj.fileno(), _CLI_BUDGET + 1)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif sum(map(len, output.values())) + len(chunk) > _CLI_BUDGET:
                        raise ValueError("DOCKER_COMMAND_REJECTED")
                    else:
                        output[key.fileobj].extend(chunk)
        if process.wait(timeout=max(0.01, deadline - time.monotonic())) != 0:
            raise ValueError("DOCKER_COMMAND_REJECTED")
        if (_identity(docker) != exe_stat or _identity(config) != config_stat):
            raise ValueError("DOCKER_CHANGED")
        return subprocess.CompletedProcess(args, 0, bytes(output[process.stdout]),
                                           bytes(output[process.stderr]))
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        for stream in output:
            stream.close()


def _one_line(raw: bytes) -> str:
    value = raw.decode("ascii")
    if value != value.strip() and value != value.strip() + "\n":
        raise ValueError("DIAGNOSTIC_OUTPUT_INVALID")
    if not value.strip() or "\n" in value.strip():
        raise ValueError("DIAGNOSTIC_OUTPUT_INVALID")
    return value.strip()


def _append(output_dir: Path, event: dict[str, object]) -> None:
    with (output_dir / "events.jsonl").open("ab") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _source_identity(source_root: Path, source_lock: Path, expected_lock: str,
                     expected_manifest: str, expected_closure: str) -> dict[str, str]:
    if (not source_root.is_absolute() or source_root.name != "engine"
            or source_root.is_symlink() or not source_root.is_dir()
            or not source_lock.is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", expected_lock)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_closure)):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    lock_raw = _read(source_lock, 128 * 1024)
    manifest = source_root.parent / "control/offline-sft-worker-v1.json"
    manifest_raw = _read(manifest, 128 * 1024)
    if (hashlib.sha256(lock_raw).hexdigest() != expected_lock
            or hashlib.sha256(manifest_raw).hexdigest() != expected_manifest):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    lock = json.loads(lock_raw)
    if (lock.get("schema_version") != "synaptic-source-lock/v1"
            or lock.get("sources", {}).get("engine", {}).get("commit") !=
            "0d515ae1ae235d0858802228ce4af8dba8fccd03"):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    # This trusted, checked-in verifier hashes every mounted worker member.
    # It is imported from this checkout, never from the staged source.
    checkout = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(checkout))
    from tuner.runtime import offline_sft_worker
    if Path(offline_sft_worker.__file__).resolve() != (
            checkout / "tuner/runtime/offline_sft_worker.py").resolve():
        raise ValueError("SOURCE_IDENTITY_INVALID")
    try:
        offline_sft_worker.load_offline_sft_worker_closure(
            manifest, expected_digest=expected_closure, engine_root=source_root)
    except Exception:
        raise ValueError("SOURCE_IDENTITY_INVALID") from None
    return {"source_lock_sha256": expected_lock, "manifest_sha256": expected_manifest,
            "closure_digest": expected_closure,
            "engine_commit": "0d515ae1ae235d0858802228ce4af8dba8fccd03"}


def _path_custody(path: Path) -> tuple[tuple[int, int, int], ...]:
    if not path.is_absolute():
        raise ValueError("SOURCE_IDENTITY_INVALID")
    observed = []
    for member in (path, *path.parents):
        info = member.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("SOURCE_IDENTITY_INVALID")
        observed.append((info.st_dev, info.st_ino, info.st_mode))
    return tuple(observed)


def _require_daemon(authority, expected: str, *, timeout: float = 5) -> None:
    result = _run(authority, ("info", "--format", "{{.ID}}"), timeout=timeout)
    if result.stderr or _one_line(result.stdout) != expected:
        raise ValueError("DAEMON_INVALID")


def _inspect_owned(authority, identifier: str, *, name: str, label: str, image: str,
                   daemon_id: str | None = None, timeout: float = 15):
    if not _CID.fullmatch(identifier):
        raise ValueError("CONTAINER_ID_INVALID")
    deadline = time.monotonic() + timeout
    if daemon_id is not None:
        _require_daemon(authority, daemon_id, timeout=max(0.05, min(5, timeout / 2)))
    result = _run(authority, ("container", "inspect", "--format", _INSPECT, identifier),
                  timeout=max(0.05, deadline - time.monotonic()))
    if result.stderr:
        raise ValueError("CONTAINER_STATE_INVALID")
    rows = result.stdout.decode("ascii").splitlines()
    if len(rows) != 6:
        raise ValueError("CONTAINER_STATE_INVALID")
    cid, actual_name, actual_image, labels, status, code = map(json.loads, rows)
    if (cid != identifier or actual_name != "/" + name or actual_image != image
            or type(labels) is not dict or labels.get("ai.synapticlabs.torch-import.claim") != label
            or status not in {"created", "running", "exited", "dead"} or type(code) is not int):
        raise ValueError("CONTAINER_STATE_INVALID")
    return status, code


def _require_gpu_zero(authority, identifier: str) -> None:
    """Confirm the fixed device request before any GPU-exposed code starts."""
    result = _run(authority, ("container", "inspect", "--format",
                              "{{json .HostConfig.DeviceRequests}}", identifier))
    if result.stderr:
        raise ValueError("GPU_SCOPE_INVALID")
    try:
        requests = json.loads(_one_line(result.stdout))
    except Exception:
        raise ValueError("GPU_SCOPE_INVALID") from None
    if (type(requests) is not list or len(requests) != 1
            or type(requests[0]) is not dict
            or requests[0].get("DeviceIDs") != ["0"]
            or requests[0].get("Capabilities") != [["gpu"]]
            or requests[0].get("Count") != 0
            or requests[0].get("Driver") != ""
            or requests[0].get("Options") != {}):
        raise ValueError("GPU_SCOPE_INVALID")


def _stop_once(authority, identifier: str, *, name: str, label: str,
               image: str, daemon_id: str, output_dir: Path) -> bool:
    """One stop request; only a separate exact inspection can prove terminal state."""
    evidence_retained = True

    def record(phase: str) -> None:
        nonlocal evidence_retained
        try:
            _append(output_dir, {"phase": phase, "container_id": identifier})
        except Exception:
            # Storage failure cannot suppress cleanup of the already claimed
            # container. It does prevent claiming durable terminal evidence.
            evidence_retained = False

    try:
        status, _ = _inspect_owned(authority, identifier, name=name, label=label,
                                   image=image, daemon_id=daemon_id)
        if status in {"exited", "dead"}:
            record("TERMINAL_OBSERVED")
            return evidence_retained
        if status != "running":
            return False
        record("STOP_CLAIMED")
        try:
            _run(authority, ("container", "stop", "--timeout", "2", identifier), timeout=10)
        except Exception:
            # A lost CLI acknowledgement does not establish daemon state.
            # Inspect independently without issuing a second stop.
            pass
        status, _ = _inspect_owned(authority, identifier, name=name, label=label,
                                   image=image, daemon_id=daemon_id)
        if status in {"exited", "dead"}:
            record("TERMINAL_OBSERVED")
            return evidence_retained
    except Exception:
        pass
    record("STOP_UNCONFIRMED")
    return False


def diagnose(*, docker: Path, docker_config: Path, release_path: Path,
             daemon_id: str, output_dir: Path, expected_release_sha256: str,
             source_root: Path | None = None, source_lock: Path | None = None,
             expected_source_lock_sha256: str | None = None,
             expected_manifest_sha256: str | None = None,
             expected_closure_digest: str | None = None,
             gpu_device: str | None = None) -> dict[str, object]:
    if gpu_device not in (None, "0") or (gpu_device is not None and source_root is None):
        raise ValueError("GPU_SCOPE_INVALID")
    if not re.fullmatch(r"[0-9a-f-]{36}", daemon_id):
        raise ValueError("DAEMON_ID_INVALID")
    release_raw = _read(release_path, 128 * 1024)
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_release_sha256)
            or hashlib.sha256(release_raw).hexdigest() != expected_release_sha256):
        raise ValueError("RELEASE_HASH_INVALID")
    release = json.loads(release_raw)
    if (type(release) is not dict or
            json.dumps(release, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True).encode("ascii") != release_raw):
        raise ValueError("RELEASE_INVALID")
    image = "sha256:" + release["image"]["digest"]
    python = release["python"]["executable"]
    digest = release["manifest_digest"]
    if (not _SHA.fullmatch(image) or type(python) is not str or not python.startswith("/")
            or type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or release["image"]["reference"].split("@")[-1] != image):
        raise ValueError("RELEASE_INVALID")
    parent_info = output_dir.parent.stat() if output_dir.parent.is_dir() else None
    if (output_dir.exists() or output_dir.is_symlink() or parent_info is None
            or output_dir.parent.is_symlink() or parent_info.st_uid != os.getuid()
            or stat.S_IMODE(parent_info.st_mode) != 0o700):
        raise ValueError("OUTPUT_INVALID")
    try:
        _path_custody(output_dir.parent)
    except Exception:
        raise ValueError("OUTPUT_INVALID") from None
    source_values = (source_lock, expected_source_lock_sha256,
                     expected_manifest_sha256, expected_closure_digest)
    if (source_root is None and any(value is not None for value in source_values)) or (
            source_root is not None and any(value is None for value in source_values)):
        raise ValueError("SOURCE_INVALID")
    source_identity = ({} if source_root is None else _source_identity(
        source_root, source_lock, expected_source_lock_sha256,
        expected_manifest_sha256, expected_closure_digest))
    custody = (() if source_root is None else tuple(_path_custody(path) for path in
        (source_root, source_root.parent / "control/offline-sft-worker-v1.json", source_lock)))
    authority = _authority(docker, docker_config)
    info = _run(authority, ("info", "--format", "{{.ID}}"))
    if info.stderr or _one_line(info.stdout) != daemon_id:
        raise ValueError("DAEMON_INVALID")
    before = _run(authority, ("image", "inspect", "--format", "{{.Id}}", image))
    if before.stderr or _one_line(before.stdout) != image:
        raise ValueError("IMAGE_INVALID")
    output_dir.mkdir(mode=0o700)
    nonce = secrets.token_hex(16)
    name = "synaptic-torch-import-" + nonce
    probe = _PROBE if source_root is None else _GUARDED_PROBE.replace(
        "EXPECTED_CLOSURE_DIGEST", expected_closure_digest)
    if gpu_device is not None:
        if probe.count(_HIDDEN_GPU_ENV) != 1:
            raise ValueError("GPU_SCOPE_INVALID")
        # The r2 isolated child did not inherit either visibility variable.
        # The host device request is separately inspected before start.
        probe = probe.replace(_HIDDEN_GPU_ENV, "")
    claim = {"schema_version": SCHEMA, "daemon_id": daemon_id, "image_id": image,
             "release_sha256": expected_release_sha256, "source_identity": source_identity,
             "name": name, "scope": "bare" if source_root is None else "guarded",
             "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "probe_sha256": hashlib.sha256(probe.encode("utf-8")).hexdigest(),
             "docker_file_identity": list(authority[2]),
             "docker_config_identity": list(authority[3])}
    if gpu_device is not None:
        claim["gpu_device"] = gpu_device
    claim_raw = json.dumps(claim, sort_keys=True, separators=(",", ":")).encode("ascii")
    label = hashlib.sha256(claim_raw).hexdigest()
    with (output_dir / "claim.json").open("xb") as stream:
        stream.write(claim_raw + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _append(output_dir, {"phase": "CREATE_CLAIMED", "claim_sha256": label})
    mount = (() if source_root is None else
             ("--mount", f"type=bind,src={source_root.parent},dst=/source,readonly"))
    user = (() if source_root is None else
            ("--user", f"{source_root.stat().st_uid}:{source_root.stat().st_gid}"))
    visibility = (("--env=NVIDIA_VISIBLE_DEVICES=void", "--env=CUDA_VISIBLE_DEVICES=")
                  if gpu_device is None else ("--gpus", "device=0"))
    command = (
        "container", "create", "--name", name, "--label", "ai.synapticlabs.torch-import.claim=" + label,
        "--pull=never", "--network=none", "--log-driver", "json-file",
        "--log-opt", "max-size=1m", "--log-opt", "max-file=1",
        "--read-only", "--cap-drop=ALL", "--runtime=runc",
        "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=4g",
        "--cpus=2", "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
        *mount, *user,
        *visibility,
        "--workdir=/tmp", "--entrypoint", python,
        image, "-I", "-c", probe,
    )
    if source_root is not None and tuple(_path_custody(path) for path in
            (source_root, source_root.parent / "control/offline-sft-worker-v1.json",
             source_lock)) != custody:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    _require_daemon(authority, daemon_id)
    try:
        created = _run(authority, command)
    except Exception:
        _append(output_dir, {"phase": "CREATE_UNCERTAIN"})
        raise ValueError("CREATE_UNCERTAIN") from None
    if created.stderr:
        _append(output_dir, {"phase": "CREATE_UNCERTAIN"})
        raise ValueError("CREATE_UNCERTAIN")
    cid = _one_line(created.stdout)
    if not _CID.fullmatch(cid):
        _append(output_dir, {"phase": "CREATE_UNCERTAIN"})
        raise ValueError("CREATE_UNCERTAIN")
    _append(output_dir, {"phase": "CREATED", "container_id": cid})
    if _inspect_owned(authority, cid, name=name, label=label, image=image,
                      daemon_id=daemon_id)[0] != "created":
        raise ValueError("CONTAINER_STATE_INVALID")
    if gpu_device is not None:
        _require_gpu_zero(authority, cid)
    # Recheck the exact source bytes immediately before starting mounted code.
    if source_root is not None and _source_identity(source_root, source_lock,
            expected_source_lock_sha256, expected_manifest_sha256,
            expected_closure_digest) != source_identity:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    if source_root is not None and tuple(_path_custody(path) for path in
            (source_root, source_root.parent / "control/offline-sft-worker-v1.json",
             source_lock)) != custody:
        raise ValueError("SOURCE_IDENTITY_INVALID")
    deadline = time.monotonic() + _RUN_SECONDS
    _append(output_dir, {"phase": "START_CLAIMED", "container_id": cid})
    cleanup_attempted = False
    start_acknowledged = False
    try:
        started = _run(authority, ("container", "start", cid), timeout=15)
        if started.stderr or _one_line(started.stdout) != cid:
            raise ValueError("START_UNCERTAIN")
        start_acknowledged = True
        _append(output_dir, {"phase": "STARTED", "container_id": cid})
        while True:
            if time.monotonic() >= deadline:
                cleanup_attempted = True
                confirmed = _stop_once(authority, cid, name=name, label=label,
                    image=image, daemon_id=daemon_id, output_dir=output_dir)
                raise ValueError("DEADLINE_STOPPED" if confirmed else "DEADLINE_UNCERTAIN")
            status, exit_code = _inspect_owned(authority, cid, name=name, label=label,
                image=image, daemon_id=daemon_id,
                timeout=max(0.1, min(5, deadline - time.monotonic())))
            if status in {"exited", "dead"}:
                _append(output_dir, {"phase": "TERMINAL_OBSERVED", "container_id": cid,
                                     "exit_code": exit_code})
                break
            if time.monotonic() >= deadline:
                cleanup_attempted = True
                confirmed = _stop_once(authority, cid, name=name, label=label,
                    image=image, daemon_id=daemon_id, output_dir=output_dir)
                raise ValueError("DEADLINE_STOPPED" if confirmed else "DEADLINE_UNCERTAIN")
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    except BaseException as error:
        if not cleanup_attempted:
            cleanup_attempted = True
            confirmed = _stop_once(authority, cid, name=name, label=label,
                image=image, daemon_id=daemon_id, output_dir=output_dir)
            if not confirmed:
                raise ValueError("DEADLINE_UNCERTAIN") from None
        if isinstance(error, KeyboardInterrupt):
            raise
        if isinstance(error, ValueError) and error.args and error.args[0] in {"DEADLINE_UNCERTAIN", "DEADLINE_STOPPED"}:
            raise
        raise ValueError("DEADLINE_UNCERTAIN" if start_acknowledged else "START_UNCERTAIN") from None
    if exit_code != 0:
        raise ValueError("CHILD_OUTPUT_INVALID")
    result = _run(authority, ("container", "logs", cid))
    if result.stderr:
        raise ValueError("CHILD_OUTPUT_INVALID")
    child = json.loads(_one_line(result.stdout))
    fields = {"schema_version", "category"} if source_root is None else {"schema_version", "category", "stage"}
    if (type(child) is not dict or set(child) != fields
            or child["schema_version"] != "synaptic-docker-torch-import-child/v1"
            or child["category"] not in _CATEGORIES
            or (source_root is not None and child["stage"] not in {"IMPORT_WORKER", "READ_MANIFEST",
                "LOAD_CLOSURE", "VERIFY_ORIGINS", "INSTALL_GUARD", "BOOTSTRAP_ENV", "TORCH_IMPORT"})):
        raise ValueError("CHILD_OUTPUT_INVALID")
    if _inspect_owned(authority, cid, name=name, label=label, image=image,
                      daemon_id=daemon_id)[0] not in {"exited", "dead"}:
        raise ValueError("CONTAINER_STATE_INVALID")
    after = _run(authority, ("image", "inspect", "--format", "{{.Id}}", image))
    if after.stderr or _one_line(after.stdout) != image:
        raise ValueError("IMAGE_CHANGED")
    report = {"schema_version": SCHEMA, "category": child["category"],
              "container_id": cid, "image_id": image, "daemon_id": daemon_id,
              "runtime_release_digest": digest,
              "training_executed": False, "gpu_used": None if gpu_device is not None else False,
              "scope": "bare" if source_root is None else "guarded",
              "claim_sha256": label, "source_identity": source_identity,
              "terminal_observed": True}
    if source_root is not None:
        report["stage"] = child["stage"]
    if gpu_device is not None:
        report["gpu_device"] = gpu_device
    with (output_dir / "result.json").open("x", encoding="ascii") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("docker", "docker-config", "release", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--daemon-id", required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--expected-source-lock-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-closure-digest")
    parser.add_argument("--gpu-device")
    args = parser.parse_args(argv)
    try:
        report = diagnose(docker=args.docker, docker_config=args.docker_config,
                          release_path=args.release, daemon_id=args.daemon_id,
                          output_dir=args.output_dir,
                          expected_release_sha256=args.expected_release_sha256,
                          source_root=args.source_root, source_lock=args.source_lock,
                          expected_source_lock_sha256=args.expected_source_lock_sha256,
                          expected_manifest_sha256=args.expected_manifest_sha256,
                          expected_closure_digest=args.expected_closure_digest,
                          gpu_device=args.gpu_device)
    except BaseException as error:
        reason = error.args[0] if type(error) is ValueError and len(error.args) == 1 else None
        if type(reason) is not str or reason not in _REASONS:
            reason = "OTHER"
        print(json.dumps({"schema_version": SCHEMA, "status": "REJECTED",
                          "reason_code": reason}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
