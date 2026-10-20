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
import subprocess
import sys
from pathlib import Path


SCHEMA = "synaptic-docker-torch-import-diagnostic/v1"
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CID = re.compile(r"[0-9a-f]{64}\Z")
_CATEGORIES = frozenset({"PASS", "MODULE_NOT_FOUND", "IMPORT", "OS", "RUNTIME", "OTHER"})
_REASONS = frozenset({"DAEMON_ID_INVALID", "INPUT_INVALID", "DOCKER_INVALID",
    "DOCKER_CHANGED", "DOCKER_COMMAND_REJECTED", "DIAGNOSTIC_OUTPUT_INVALID",
    "RELEASE_INVALID", "RELEASE_HASH_INVALID", "OUTPUT_INVALID", "SOURCE_INVALID", "DAEMON_INVALID", "IMAGE_INVALID",
    "CHILD_OUTPUT_INVALID", "CONTAINER_ID_INVALID", "CONTAINER_IMAGE_INVALID",
    "IMAGE_CHANGED"})

# This fixed program never formats an exception, inspects its attributes, or
# imports an engine/trainer module.  The exact packaged interpreter imports
# Torch under the isolated, offline child environment.
_PROBE = """import json,os,sys
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
print(json.dumps({'schema_version':'synaptic-docker-torch-import-child/v1','category':category},sort_keys=True,separators=(',',':')))
"""

_GUARDED_PROBE = """import json,os,sys
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
 declared=json.loads(manifest.read_bytes())['closure_digest']
 stage='LOAD_CLOSURE'
 closure=load_offline_sft_worker_closure(manifest,expected_digest=declared,engine_root=root)
 stage='VERIFY_ORIGINS'
 verify_loaded_owned_module_origins(closure,engine_root=root)
 stage='INSTALL_GUARD'
 install_owned_module_guard(closure,engine_root=root)
 sys.path[:0]=[str(root/'Trainers/sft'),str(root)]
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
print(json.dumps({'schema_version':'synaptic-docker-torch-import-child/v1','category':category,'stage':stage},sort_keys=True,separators=(',',':')))
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


def _run(authority, args: tuple[str, ...], *, timeout: int = 120):
    docker, config, exe_stat, config_stat = authority
    if (_identity(docker) != exe_stat or _identity(config) != config_stat):
        raise ValueError("DOCKER_CHANGED")
    result = subprocess.run((str(docker), "--config", str(config),
        "--host", "unix:///var/run/docker.sock", *args),
        env={"PATH": os.defpath, "DOCKER_CONTENT_TRUST": "1"},
        stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, check=False)
    if (_identity(docker) != exe_stat or _identity(config) != config_stat):
        raise ValueError("DOCKER_CHANGED")
    if result.returncode != 0 or len(result.stdout) > 8192 or len(result.stderr) > 8192:
        raise ValueError("DOCKER_COMMAND_REJECTED")
    return result


def _one_line(raw: bytes) -> str:
    value = raw.decode("ascii")
    if value != value.strip() and value != value.strip() + "\n":
        raise ValueError("DIAGNOSTIC_OUTPUT_INVALID")
    if not value.strip() or "\n" in value.strip():
        raise ValueError("DIAGNOSTIC_OUTPUT_INVALID")
    return value.strip()


def diagnose(*, docker: Path, docker_config: Path, release_path: Path,
             daemon_id: str, output_dir: Path, expected_release_sha256: str,
             source_root: Path | None = None) -> dict[str, object]:
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
    if output_dir.exists() or output_dir.is_symlink() or not output_dir.parent.is_dir():
        raise ValueError("OUTPUT_INVALID")
    if source_root is not None and (not source_root.is_absolute() or source_root.is_symlink()
            or not source_root.is_dir()
            or source_root.name != "engine"
            or not (source_root.parent / "control/offline-sft-worker-v1.json").is_file()):
        raise ValueError("SOURCE_INVALID")
    authority = _authority(docker, docker_config)
    info = _run(authority, ("info", "--format", "{{.ID}}"))
    if info.stderr or _one_line(info.stdout) != daemon_id:
        raise ValueError("DAEMON_INVALID")
    before = _run(authority, ("image", "inspect", "--format", "{{.Id}}", image))
    if before.stderr or _one_line(before.stdout) != image:
        raise ValueError("IMAGE_INVALID")
    output_dir.mkdir(mode=0o700)
    cidfile = output_dir / "container.id"
    mount = (() if source_root is None else
             ("--mount", f"type=bind,src={source_root.parent},dst=/source,readonly"))
    user = (() if source_root is None else
            ("--user", f"{source_root.stat().st_uid}:{source_root.stat().st_gid}"))
    command = (
        "run", "--cidfile", str(cidfile), "--pull=never", "--network=none",
        "--read-only", "--cap-drop=ALL", "--runtime=runc",
        "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=4g",
        "--cpus=2", "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
        *mount, *user,
        "--env=NVIDIA_VISIBLE_DEVICES=void", "--env=CUDA_VISIBLE_DEVICES=",
        "--workdir=/tmp", "--entrypoint", python,
        image, "-I", "-c", _PROBE if source_root is None else _GUARDED_PROBE,
    )
    result = _run(authority, command)
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
    cid = _one_line(_read(cidfile, 80))
    if not _CID.fullmatch(cid):
        raise ValueError("CONTAINER_ID_INVALID")
    inspected = _run(authority, ("container", "inspect", "--format", "{{.Image}}", cid))
    if inspected.stderr or _one_line(inspected.stdout) != image:
        raise ValueError("CONTAINER_IMAGE_INVALID")
    after = _run(authority, ("image", "inspect", "--format", "{{.Id}}", image))
    if after.stderr or _one_line(after.stdout) != image:
        raise ValueError("IMAGE_CHANGED")
    report = {"schema_version": SCHEMA, "category": child["category"],
              "container_id": cid, "image_id": image, "daemon_id": daemon_id,
              "runtime_release_digest": digest,
              "training_executed": False, "gpu_used": False,
              "scope": "bare" if source_root is None else "guarded"}
    if source_root is not None:
        report["stage"] = child["stage"]
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
    args = parser.parse_args(argv)
    try:
        report = diagnose(docker=args.docker, docker_config=args.docker_config,
                          release_path=args.release, daemon_id=args.daemon_id,
                          output_dir=args.output_dir,
                          expected_release_sha256=args.expected_release_sha256,
                          source_root=args.source_root)
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
