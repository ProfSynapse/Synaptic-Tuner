"""Derived-image GRPO constructor diagnostic; never grants training authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
from pathlib import Path
from tempfile import mkdtemp

from tuner.cloud.derived_training_image import (
    DerivedTrainingImageError, _canonical_bytes, _docker_argv, _docker_env,
    _inspect_command, _load_image_evidence, _load_runtime_release,
    _bound_capture, _release_from_input, _release_facts, _read_object, load_profile,
    _parse_inspect, _run_authorized, _validate_docker_inputs,
)
from tuner.cloud.hf_training_image_lock import CommandSpec, TrainingImageLockError, subprocess_runner
from tuner.runtime.packaged_worker_closure import stable_read


ADMISSION_SCHEMA = "synaptic-grpo-constructor-diagnostic-admission/v1"
RESULT_SCHEMA = "synaptic-grpo-constructor-diagnostic/v1"
CHILD_SCHEMA = "synaptic-grpo-constructor-probe-result/v1"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ID = re.compile(r"^[0-9a-f]{64}$")
_CHILD_SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "grpo_constructor_probe_child.py"
_CLI_SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "qualify_derived_training_image.py"


def _read_bounded(path: Path, bound: int) -> bytes:
    try:
        raw = stable_read(path, bound)
    except (OSError, ValueError) as exc:
        raise DerivedTrainingImageError("PROBE_INPUT_INVALID") from exc
    if not 0 < len(raw) <= bound:
        raise DerivedTrainingImageError("PROBE_INPUT_INVALID")
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path, bound: int) -> tuple[int, str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > bound:
            raise ValueError
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > bound:
                raise ValueError
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or size != before.st_size:
            raise ValueError
        return size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _copy_model_snapshot(source_root: Path, destination_root: Path, inventory: list[dict]) -> None:
    """Revalidate and copy admitted bytes into a new private read-only tree."""
    destination_root.mkdir(mode=0o700)
    for item in inventory:
        parts = item["path"].split("/")
        parent = destination_root
        for part in parts[:-1]:
            parent = parent / part
            if not parent.exists():
                parent.mkdir(mode=0o700)
        source = source_root.joinpath(*parts)
        target = parent / parts[-1]
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            before = os.fstat(source_fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or before.st_size != item["size_bytes"]):
                raise DerivedTrainingImageError("PROBE_MODEL_CHANGED")
            target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            digest = hashlib.sha256()
            size = 0
            try:
                while chunk := os.read(source_fd, 1024 * 1024):
                    size += len(chunk)
                    if size > item["size_bytes"]:
                        raise DerivedTrainingImageError("PROBE_MODEL_CHANGED")
                    digest.update(chunk)
                    pending = memoryview(chunk)
                    while pending:
                        written = os.write(target_fd, pending)
                        if written <= 0:
                            raise DerivedTrainingImageError("PROBE_MODEL_CHANGED")
                        pending = pending[written:]
                os.fsync(target_fd)
            finally:
                os.close(target_fd)
            after = os.fstat(source_fd)
            if (size != item["size_bytes"] or digest.hexdigest() != item["sha256"]
                    or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                    or _file_digest(target, item["size_bytes"])[1] != item["sha256"]):
                raise DerivedTrainingImageError("PROBE_MODEL_CHANGED")
            os.chmod(target, 0o400)
        finally:
            os.close(source_fd)


def validate_probe_admission(admission: object, *, source_sha256: str, input_sha256: str,
        recipe_sha256: str, dataset_sha256: str, plan_ready_sha256: str,
        profile_sha256: str, runtime_release_digest: str, image_config_digest: str,
        host_helper_sha256: str, cli_sha256: str, model_ref: str, model_revision: str) -> None:
    expected_keys = {"schema_version", "source_commit", "probe_source_sha256", "input_sha256",
        "host_helper_sha256", "cli_sha256",
        "recipe_sha256", "dataset_sha256", "plan_ready_sha256", "profile_sha256",
        "runtime_release_digest", "image_config_digest", "model_inventory", "gpu_device"}
    expected_keys |= {"model_ref", "model_revision"}
    digests = {"probe_source_sha256": source_sha256, "input_sha256": input_sha256,
        "host_helper_sha256": host_helper_sha256, "cli_sha256": cli_sha256,
        "recipe_sha256": recipe_sha256, "dataset_sha256": dataset_sha256,
        "plan_ready_sha256": plan_ready_sha256, "profile_sha256": profile_sha256,
        "runtime_release_digest": runtime_release_digest,
        "image_config_digest": image_config_digest}
    if (type(admission) is not dict or set(admission) != expected_keys
            or admission["schema_version"] != ADMISSION_SCHEMA
            or type(admission["source_commit"]) is not str
            or _COMMIT.fullmatch(admission["source_commit"]) is None
            or any(type(admission[key]) is not str or not _HEX.fullmatch(admission[key])
                   for key in digests if key != "image_config_digest")
            or admission["image_config_digest"] != image_config_digest
            or admission["model_ref"] != model_ref
            or admission["model_revision"] != model_revision
            or any(admission[key] != value for key, value in digests.items())
            or type(admission["model_inventory"]) is not list
            or not 1 <= len(admission["model_inventory"]) <= 64
            or type(admission["gpu_device"]) is not int
            or not 0 <= admission["gpu_device"] <= 15):
        raise DerivedTrainingImageError("PROBE_ADMISSION_INVALID")


def derive_probe_input(*, recipe_path: Path, plan_ready_path: Path, dataset_path: Path) -> bytes:
    """Project the normal GRPO compiler's exact four-row input without effects."""
    try:
        plan = json.loads(_read_bounded(plan_ready_path, 65536))["data"]
        recipe, _ = _read_object(recipe_path)
        dataset_raw = _read_bounded(dataset_path, 65536)
        from synaptic_tuner.api.v1.training_input import TrainingInputV1
        from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
        from tuner.training.grpo_compilation import compile_grpo_input, admit_grpo_snapshot
        if recipe["training_input"] != plan["training_input"]:
            raise ValueError
        training = TrainingInputV1.from_dict(plan["training_input"])
        prepared = PreparedTrainingInputIdentity.from_dict(plan["prepared_dataset"])
        compiled = compile_grpo_input(training, prepared)
        config, rows, documents, schema = admit_grpo_snapshot(training, prepared, dataset_raw)
        if compiled.fingerprint != plan["core_workload_digest"] or len(rows) != 4:
            raise ValueError
        return _canonical_bytes({"schema_version": "synaptic-grpo-constructor-probe-input/v1",
            "config": config, "rows": list(rows), "reward_documents": documents, "reward_schema": schema,
            "qualification_policy": recipe["limits"]["qualification_evidence"]})
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DerivedTrainingImageError("PROBE_INPUT_INVALID") from exc


def _admit(*, admission_path: Path, plan_ready_path: Path, snapshot_root: Path,
           recipe_path: Path, dataset_path: Path,
           release_path: Path, capture_path: Path,
           profile_path: Path, build_receipt_path: Path, candidate_path: Path,
           image_verification_path: Path):
    profile = load_profile(profile_path)
    release, _ = _load_runtime_release(release_path)
    image, image_raw = _load_image_evidence(profile_path=profile_path,
        build_receipt_path=build_receipt_path, candidate_path=candidate_path,
        image_verification_path=image_verification_path)
    capture, _ = _bound_capture(capture_path, image=image, image_raw=image_raw)
    if release.to_dict() != _release_from_input(_release_facts(
            capture, release.to_dict()["compatibility"], release.release_ref)).to_dict():
        raise DerivedTrainingImageError("RELEASE_CAPTURE_MISMATCH")
    try:
        raw = _read_bounded(admission_path, 65536)
        admission = json.loads(raw)
        source = _read_bounded(_CHILD_SOURCE, 65536)
        helper_source = _read_bounded(Path(__file__), 65536)
        cli_source = _read_bounded(_CLI_SOURCE, 65536)
        plan_raw = _read_bounded(plan_ready_path, 65536)
        dataset_raw = _read_bounded(dataset_path, 65536)
        input_raw = derive_probe_input(recipe_path=recipe_path, plan_ready_path=plan_ready_path,
                                       dataset_path=dataset_path)
        plan_model = json.loads(plan_raw)["data"]["training_input"]["model"]
        if (type(plan_model) is not dict or type(plan_model.get("ref")) is not str
                or type(plan_model.get("revision")) is not str
                or (plan_model["ref"], plan_model["revision"]) not in release.compatible_models):
            raise ValueError
        validate_probe_admission(admission, source_sha256=_sha(source), input_sha256=_sha(input_raw),
            host_helper_sha256=_sha(helper_source), cli_sha256=_sha(cli_source),
            recipe_sha256=_file_digest(recipe_path, 65536)[1], dataset_sha256=_sha(dataset_raw),
            plan_ready_sha256=_sha(plan_raw),
            profile_sha256=profile.canonical_sha256.removeprefix("sha256:"),
            runtime_release_digest=release.manifest_digest,
            image_config_digest=image["image_config_digest"],
            model_ref=plan_model["ref"], model_revision=plan_model["revision"])
        repo = _CHILD_SOURCE.parents[1]
        head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=repo, capture_output=True,
                              timeout=5, check=True).stdout.decode("ascii").strip()
        committed = [subprocess.run(("git", "show", "HEAD:" + member), cwd=repo,
                                   capture_output=True, timeout=5, check=True).stdout
            for member in ("scripts/grpo_constructor_probe_child.py",
                           "tuner/cloud/grpo_constructor_diagnostic.py",
                           "scripts/qualify_derived_training_image.py")]
        if head != admission["source_commit"] or committed != [source, helper_source, cli_source]:
            raise ValueError
        root = snapshot_root.resolve(strict=True)
        if root != snapshot_root.absolute() or not root.is_dir() or root.stat().st_mode & 0o077:
            raise ValueError
        observed = {}
        total_size = 0
        directory_count = 0
        for directory, folders, files in os.walk(root, followlinks=False):
            directory_count += 1
            if directory_count > 128:
                raise ValueError
            for name in folders + files:
                path = Path(directory) / name
                if path.is_symlink():
                    raise ValueError
            for name in files:
                path = Path(directory) / name
                if not path.is_file() or path.stat().st_nlink != 1:
                    raise ValueError
                relative = path.relative_to(root).as_posix()
                if len(observed) >= 64 or path.stat().st_size > 1024 * 1024 * 1024:
                    raise ValueError
                size, digest = _file_digest(path, 1024 * 1024 * 1024)
                total_size += size
                if total_size > 2 * 1024 * 1024 * 1024:
                    raise ValueError
                observed[relative] = {"path": relative, "size_bytes": size, "sha256": digest}
        expected = admission["model_inventory"]
        if (any(type(item) is not dict or set(item) != {"path", "size_bytes", "sha256"}
                for item in expected) or [observed[key] for key in sorted(observed)] != expected):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        raise DerivedTrainingImageError("PROBE_ADMISSION_INVALID") from exc
    return profile, release, image, admission, source, input_raw


def _command(docker: Path, docker_config: Path, args: tuple[str, ...], *, timeout=15, maximum=8192):
    return CommandSpec(_docker_argv(docker, docker_config) + args, _docker_env(docker), timeout, maximum)


def diagnose_derived_grpo_constructor(*, admission_path: Path, plan_ready_path: Path,
        snapshot_root: Path, recipe_path: Path, dataset_path: Path,
        release_path: Path,
        capture_path: Path, profile_path: Path, build_receipt_path: Path,
        candidate_path: Path, image_verification_path: Path, docker: Path,
        docker_config: Path, output: Path, execute: bool = False,
        runner=subprocess_runner) -> dict:
    """One fresh owned container. An uncertain command is consumed, never replayed."""
    profile, release, image, admission, source, input_raw = _admit(**{
        "admission_path": admission_path, "plan_ready_path": plan_ready_path, "snapshot_root": snapshot_root,
        "recipe_path": recipe_path, "dataset_path": dataset_path,
        "release_path": release_path,
        "capture_path": capture_path, "profile_path": profile_path,
        "build_receipt_path": build_receipt_path, "candidate_path": candidate_path,
        "image_verification_path": image_verification_path})
    if output.exists() or output.is_symlink() or output.suffix != ".json":
        raise DerivedTrainingImageError("OUTPUT_INVALID")
    if os.name != "posix":
        raise DerivedTrainingImageError("PROBE_HOST_UNSUPPORTED")
    parent = output.parent.resolve(strict=True)
    if (parent != output.parent.absolute() or parent.stat().st_uid != os.getuid()
            or parent.stat().st_mode & 0o077):
        raise DerivedTrainingImageError("OUTPUT_INVALID")
    if not execute:
        return {"schema_version": RESULT_SCHEMA, "status": "PLAN_ONLY",
                "source_commit": admission["source_commit"],
                "probe_source_sha256": admission["probe_source_sha256"],
                "image_config_digest": image["image_config_digest"],
                "runtime_release_digest": release.manifest_digest,
                "model_inventory_sha256": _sha(_canonical_bytes(admission["model_inventory"])),
                "training_executed": False, "generation_executed": False}
    authority = _validate_docker_inputs(docker, docker_config)
    before = _parse_inspect(_run_authorized(authority, runner, _inspect_command(
        docker=authority.executable, docker_config=authority.config_directory,
        image=image["image_config_digest"])).stdout)
    if before[0] != image["image_config_digest"] or before[2].get("ai.synapticlabs.derived-training-profile") != profile.canonical_sha256:
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID")
    nonce = secrets.token_hex(16)
    name = "synaptic-grpo-constructor-" + nonce
    claim = output.with_suffix(".claim.json")
    if claim.exists() or claim.is_symlink():
        raise DerivedTrainingImageError("PROBE_CLAIM_EXISTS")
    stage = Path(mkdtemp(prefix="grpo-probe-", dir=output.parent))
    os.chmod(stage, 0o700)
    input_path = stage / "input.json"
    descriptor = os.open(input_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(input_raw); stream.flush(); os.fsync(stream.fileno())
    claim_raw = _canonical_bytes({"schema_version": "synaptic-grpo-constructor-claim/v1",
        "name": name, "nonce": nonce, "admission_sha256": _sha(_read_bounded(admission_path, 65536)),
        "stage": str(stage), "status": "CONSUMED"})
    descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(claim_raw); stream.flush(); os.fsync(stream.fileno())
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    model_stage = stage / "model"
    _copy_model_snapshot(snapshot_root, model_stage, admission["model_inventory"])
    container_id = None
    cleanup_ok = False
    report = None
    try:
        bootstrap = ("import os,sys;d=os.open('/dev/null',os.O_WRONLY);os.dup2(d,2);os.close(d);"
            "sys.path.insert(0," + repr(profile.packaged_runtime["python"]["purelib"]) + ");"
            "exec(compile(bytes.fromhex(" + repr(source.hex()) + "),'<bound-probe>','exec'))")
        command = _command(docker, docker_config, (
            "create", "--name", name, "--label", "ai.synapticlabs.probe-nonce=" + nonce,
            "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--runtime=runc", "--pids-limit=64",
            "--memory=8g", "--cpus=2", "--gpus", "device=" + str(admission["gpu_device"]),
            "--user", str(os.getuid()) + ":" + str(os.getgid()),
            "--log-driver=local", "--log-opt=max-size=8k", "--log-opt=max-file=1",
            "--tmpfs", "/probe:rw,nosuid,nodev,size=512m,mode=0700,uid=" + str(os.getuid()) + ",gid=" + str(os.getgid()),
            "--mount", "type=bind,src=" + str(input_path) + ",dst=/probe-input/input.json,readonly",
            "--mount", "type=bind,src=" + str(model_stage) + ",dst=/probe-model,readonly",
            "--env=HF_HUB_OFFLINE=1", "--env=TRANSFORMERS_OFFLINE=1",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--env=WANDB_DISABLED=true",
            "--env=HOME=/probe", "--env=HF_HOME=/probe/hf",
            "--env=XDG_CACHE_HOME=/probe/cache", "--env=TRITON_CACHE_DIR=/probe/triton",
            "--workdir=/probe",
            "--entrypoint", release.python_executable, image["image_config_digest"],
            "-I", "-S", "-c", bootstrap), timeout=20)
        created = _run_authorized(authority, runner, command).stdout.decode("ascii").strip()
        if _ID.fullmatch(created) is None:
            raise DerivedTrainingImageError("PROBE_CREATE_UNCERTAIN")
        container_id = created
        _run_authorized(authority, runner, _command(docker, docker_config, ("start", container_id), timeout=15))
        waited = _run_authorized(authority, runner, _command(docker, docker_config, ("wait", container_id), timeout=120)).stdout.strip()
        if waited != b"0":
            raise DerivedTrainingImageError("PROBE_CHILD_UNAVAILABLE")
        terminal = _run_authorized(authority, runner, _command(docker, docker_config,
            ("container", "inspect", "--format", '{{json .Id}}\n{{json .State.Running}}\n{{json .State.ExitCode}}\n{{json .Config.Labels}}', container_id), timeout=10)).stdout
        try:
            terminal_lines = terminal.decode("utf-8").splitlines()
            terminal_values = [json.loads(line) for line in terminal_lines]
        except (UnicodeDecodeError, ValueError):
            raise DerivedTrainingImageError("PROBE_TERMINAL_INVALID") from None
        if (len(terminal_values) != 4 or terminal_values[0] != container_id
                or terminal_values[1] is not False or terminal_values[2] != 0
                or type(terminal_values[3]) is not dict
                or terminal_values[3].get("ai.synapticlabs.probe-nonce") != nonce):
            raise DerivedTrainingImageError("PROBE_TERMINAL_INVALID")
        child_raw = _run_authorized(authority, runner, _command(docker, docker_config,
            ("logs", container_id), timeout=15)).stdout
        if not 0 < len(child_raw) <= 8192:
            raise DerivedTrainingImageError("PROBE_CHILD_INVALID")
        child = json.loads(child_raw)
        if (type(child) is not dict or set(child) != {"schema_version", "status", "stage", "category",
            "class_digest", "signature_digest", "training_executed", "generation_executed"}
            or child["schema_version"] != CHILD_SCHEMA
            or child["status"] not in {"CONSTRUCTED", "REJECTED"}
            or child["stage"] not in {"INPUT", "DEPENDENCIES", "CONFIG", "MODEL_LOAD", "LORA", "DATA", "REWARD", "CALLBACKS", "CONSTRUCTOR"}
            or child["category"] not in {"NONE", "TYPE", "VALUE", "ATTRIBUTE", "KEY", "IMPORT", "MEMORY", "OS", "RUNTIME", "OTHER"}
            or child["training_executed"] is not False or child["generation_executed"] is not False
            or any(value is not None and (type(value) is not str or not _HEX.fullmatch(value))
                   for value in (child["class_digest"], child["signature_digest"]))):
            raise DerivedTrainingImageError("PROBE_CHILD_INVALID")
        if child_raw != _canonical_bytes(child):
            raise DerivedTrainingImageError("PROBE_CHILD_INVALID")
        after = _parse_inspect(_run_authorized(authority, runner, _inspect_command(
            docker=authority.executable, docker_config=authority.config_directory,
            image=image["image_config_digest"])).stdout)
        if before != after:
            raise DerivedTrainingImageError("DOCKER_EVIDENCE_CHANGED")
        report = {"schema_version": RESULT_SCHEMA, "status": "DIAGNOSTIC_ONLY", "child": child,
            "source_commit": admission["source_commit"], "probe_source_sha256": _sha(source),
            "input_sha256": _sha(input_raw), "runtime_release_digest": release.manifest_digest,
            "image_config_digest": image["image_config_digest"], "container_id": container_id,
            "training_executed": False, "generation_executed": False,
            "gpu_qualified": False, "retry_authorized": False}
    except TrainingImageLockError as exc:
        raise DerivedTrainingImageError("PROBE_COMMAND_" + ("TIMEOUT" if exc.reason_code == "OPERATION_TIMEOUT" else "UNAVAILABLE")) from None
    finally:
        # A timed-out Docker CLI can leave a live container. Resolve by the
        # unique name and remove only when both nonce and immutable ID match.
        try:
            observed = _run_authorized(authority, runner, _command(docker, docker_config,
                ("container", "inspect", "--format", '{{json .Id}}\n{{json .Config.Labels}}', name), timeout=10)).stdout
            lines = observed.decode("utf-8").splitlines()
            actual_id, labels = json.loads(lines[0]), json.loads(lines[1])
            if (len(lines) == 2 and _ID.fullmatch(actual_id) and type(labels) is dict
                    and labels.get("ai.synapticlabs.probe-nonce") == nonce
                    and (container_id is None or container_id == actual_id)):
                removed = _run_authorized(authority, runner, _command(docker, docker_config,
                    ("rm", "--force", actual_id), timeout=15)).stdout.strip()
                cleanup_ok = removed in {actual_id.encode("ascii"), name.encode("ascii")}
        except BaseException:
            pass
    if not cleanup_ok or report is None:
        raise DerivedTrainingImageError("PROBE_CLEANUP_UNCERTAIN")
    shutil.rmtree(stage)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_bytes(report)); stream.flush(); os.fsync(stream.fileno())
    return report
