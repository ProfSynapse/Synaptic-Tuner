"""
Location: Trainers/cloud/train_modal.py

Purpose:
    Standalone Modal wrapper script for running SFT or KTO training in the cloud.
    Invoked via `modal run --detach Trainers/cloud/train_modal.py::run_training` or
    programmatically from the ModalBackend in tuner/backends/training/cloud/.

    The script defines a Modal App with GPU configuration, persistent volumes for
    caching model weights, and a container image with all training dependencies.
    Training runs execute the existing train_sft.py or train_kto.py scripts inside
    the Modal container and retain canonical artifacts on the output Volume.
    Hub publication remains explicit opt-in behavior for the legacy CLI path.

Usage:
    # Run SFT training on an L40S GPU (default)
    modal run --detach Trainers/cloud/train_modal.py::run_training --trainer-type sft

    # Run KTO training on an A100 GPU
    MODAL_GPU=A100 modal run --detach Trainers/cloud/train_modal.py::run_training --trainer-type kto

    # Specify a custom model and dataset
    modal run --detach Trainers/cloud/train_modal.py::run_training \\
        --trainer-type sft \\
        --model-name "unsloth/mistral-7b-v0.3-bnb-4bit" \\
        --dataset-path "Datasets/my_dataset.jsonl" \\
        --publish-target-repo "myuser/my-model"

Dependencies:
    - modal (pip install modal)
    - Modal account with token configured (modal setup)
    - HF_TOKEN environment variable (for model downloads and Hub uploads)
"""

import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

# shared/ ships with the repo, not the container image: inside the Modal
# container this module is imported BEFORE the repo is cloned, so the import
# must not be fatal at module scope. run_training() re-imports it from the
# cloned workspace after checkout.
try:
    from shared.utilities.paths import (
        get_canonical_output_dir_name,
        get_canonical_trainer_dir_name,
    )
except ImportError:
    get_canonical_output_dir_name = None
    get_canonical_trainer_dir_name = None

try:
    import modal
except ImportError:
    print(
        "Error: modal package is not installed.\n"
        "Install it with: pip install modal\n"
        "Then authenticate with: modal setup"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Modal App Configuration
# ---------------------------------------------------------------------------

HOURS = 60 * 60  # seconds
VALID_GPU_TYPES = ["T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100"]
DEFAULT_GPU = "L40S"
_OCI_DIGEST = re.compile(r"^.+@sha256:[0-9a-fA-F]{64}$")
_EXACT_PIP = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^*<>=!~,;\s]+$")
_HASHED_URL_PIP = re.compile(
    r"^(?:[A-Za-z0-9_.-]+\s*@\s*)?https://\S+#sha256=[0-9a-fA-F]{64}$"
)
_PINNED_GIT_PIP = re.compile(r"^(?:[A-Za-z0-9_.-]+\s*@\s*)?git\+https://\S+@[0-9a-fA-F]{40}(?:#\S+)?$")
_STABLE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CHECKPOINT_NAME = re.compile(r"^checkpoint-([1-9][0-9]*)$")
DONE_MARKER_NAME = "DONE"
COMPLETION_READY_MARKER_NAME = "COMPLETION_READY.json"
OUTPUT_COMMIT_INTERVAL_SECONDS = 60
PROCESS_STOP_TIMEOUT_SECONDS = 10
COMMIT_THREAD_JOIN_TIMEOUT_SECONDS = 10


def _redact_text(value: str) -> str:
    text = str(value)
    text = re.sub(r"(https?://)[^/@\s]+@", r"\1[REDACTED]@", text)
    text = re.sub(
        r"(?i)(token|secret|password|api[_-]?key)(\s*[=:]\s*)[^\s,;]+",
        r"\1\2[REDACTED]",
        text,
    )
    return text


def _credential_free_repo_url(value: str) -> str:
    url = value.strip()
    parsed = urlsplit(url)
    if parsed.scheme in {"http", "https"} and (parsed.username or parsed.password):
        raise ValueError(
            "Credential-bearing repository URLs are forbidden; use a credential-free remote URL."
        )
    if not url or any(char in url for char in ("\n", "\r", "\x00")):
        raise ValueError("Repository URL is empty or contains control characters.")
    return url


def _exact_pip_requirement(value: str) -> str:
    requirement = value.strip()
    if re.search(r"https?://[^/@\s]+@", requirement):
        raise RuntimeError("Credential-bearing pip requirement URLs are forbidden.")
    if not (
        _EXACT_PIP.fullmatch(requirement)
        or _HASHED_URL_PIP.fullmatch(requirement)
        or _PINNED_GIT_PIP.fullmatch(requirement)
    ):
        raise RuntimeError(
            "Modal pip overlays must be immutable: name==version, a URL with a full "
            "#sha256 digest, or git+https pinned to a full commit."
        )
    return requirement


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")
    return value


def _config_overrides_mapping(value) -> dict:
    """Normalize programmatic mappings and Modal CLI JSON strings.

    Modal's generated CLI only supports scalar parameter annotations.  Keep the
    remote function callable with the historical mapping payload while exposing
    a string annotation that Modal 1.5 can turn into ``--config-overrides``.
    """
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("config_overrides must be a JSON object string.") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("config_overrides must be a mapping or a JSON object string.")


def _stable_run_id(value: str) -> str:
    """Validate a path-safe identifier supplied by a generic job planner."""
    run_id = (value or "").strip()
    if run_id and not _STABLE_RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-128 characters using only letters, digits, '.', '_', or '-', "
            "and must start with a letter or digit."
        )
    return run_id


# Persistent volume for caching HuggingFace model weights between runs.
# This avoids re-downloading multi-GB models on every training run, saving
# both time and bandwidth costs.
model_cache = modal.Volume.from_name(
    os.environ.get("MODAL_CACHE_VOLUME_NAME", "toolset-model-cache"), create_if_missing=True
)

# Persistent volume for storing provider-native training artifacts.
output_volume = modal.Volume.from_name(
    os.environ.get("MODAL_OUTPUT_VOLUME_NAME", "toolset-training-artifacts"), create_if_missing=True
)
OUTPUT_MOUNT_PATH = os.environ.get("MODAL_OUTPUT_MOUNT_PATH", "/vol/artifacts")

# A private input Volume is opt-in so historical repo-local / Hub-dataset jobs
# do not acquire a new prerequisite.  Config-driven private-input jobs set the
# name before `modal run`; unlike output/cache storage it must already exist.
INPUT_MOUNT_PATH = os.environ.get("MODAL_INPUT_MOUNT_PATH", "/vol/inputs")
INPUT_VOLUME_NAME = os.environ.get("MODAL_INPUT_VOLUME_NAME", "").strip()
input_volume = (
    modal.Volume.from_name(INPUT_VOLUME_NAME, create_if_missing=False)
    if INPUT_VOLUME_NAME
    else None
)

# Container image with all training dependencies pre-installed.
# Using debian_slim as the base keeps the image small while providing
# a stable foundation. Dependencies are installed via pip_install since
# unsloth has complex CUDA dependency resolution that works better with pip.
#
# Version pins follow Modal's official unsloth example pattern. The CUDA 12.8 /
# PyTorch 2.7.0 extras align with the unsloth Docker image used by HF Jobs and
# RunPod (see cloud_config.yaml). Update these together when upgrading unsloth.
def _runtime_pip_packages() -> list[str]:
    """Parse optional exact runtime requirements without invoking a shell."""
    raw = os.environ.get("MODAL_TRAINING_PIP_PACKAGES_JSON", "").strip()
    if not raw:
        return []
    try:
        packages = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MODAL_TRAINING_PIP_PACKAGES_JSON must be valid JSON.") from exc
    if not isinstance(packages, list) or any(
        not isinstance(item, str) or not item.strip() for item in packages
    ):
        raise RuntimeError("MODAL_TRAINING_PIP_PACKAGES_JSON must be a list of non-empty strings.")
    return [_exact_pip_requirement(item) for item in packages]


runtime_image_ref = os.environ.get("MODAL_TRAINING_IMAGE", "").strip()
if runtime_image_ref:
    if not _OCI_DIGEST.fullmatch(runtime_image_ref):
        raise RuntimeError("MODAL_TRAINING_IMAGE must include a full immutable @sha256 digest.")
    # Registry images such as Unsloth run a supervisor entrypoint by default;
    # clear it so Modal can invoke the function.  The inspect planner requires
    # an immutable @sha256 digest for this path.
    training_image = modal.Image.from_registry(runtime_image_ref).entrypoint([])
else:
    # Preserve the historical default image byte-for-byte when no explicit
    # runtime is configured.
    training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
        # Core ML stack — pinned to exact versions for reproducibility
        "torch==2.7.0",
        "unsloth[cu128-torch270]==2025.7.8",
        "trl==0.19.1",
        "transformers==4.54.0",
        "datasets==3.6.0",
        "peft==0.15.2",
        "accelerate==1.6.0",
        "bitsandbytes==0.45.5",
        # transformers 4.54.0 requires huggingface_hub>=0.34.0,<1.0; the old
        # ==0.30.2 pin made the image unresolvable (ResolutionImpossible at build).
        "huggingface_hub>=0.34.0,<1.0",
        # Project utilities — lighter deps, less sensitive to version drift
        "pyyaml",
        "wandb",
        "hf_transfer",
        "python-dotenv",
        "rich",
    ).apt_install("git")

runtime_pips = _runtime_pip_packages()
if runtime_pips:
    training_image = training_image.pip_install(*runtime_pips)

FUNCTION_GPU = os.environ.get("MODAL_GPU", DEFAULT_GPU).strip() or DEFAULT_GPU
if FUNCTION_GPU not in VALID_GPU_TYPES:
    raise RuntimeError(
        f"MODAL_GPU must be one of {', '.join(VALID_GPU_TYPES)}; got {FUNCTION_GPU!r}."
    )
FUNCTION_TIMEOUT_SECONDS = _positive_int_env("MODAL_TIMEOUT_SECONDS", 6 * HOURS)

app = modal.App("toolset-training", image=training_image)


# ---------------------------------------------------------------------------
# GPU Pricing Reference (display-only cost estimation for local entrypoint)
# ---------------------------------------------------------------------------
# Approximate Modal GPU prices per hour as of early 2026.
# Canonical pricing data is in tuner/backends/training/cloud/base_cloud.py
# GPU_PRICING["modal"]. This standalone script cannot import from tuner.*
# (runs via `modal run`), so a display-only copy is kept here.
# Check https://modal.com/pricing for current rates.
GPU_PRICING = {
    "T4": 0.59,
    "L4": 0.73,
    "A10G": 1.10,
    "L40S": 1.40,
    "A100": 2.78,       # A100-40GB
    "A100-80GB": 3.72,
    "H100": 4.89,
}

# The hf_xet CAS backend hangs without timeout on multi-GB model pulls (py-spy:
# workers frozen in xet_get at file_download.py:626), which froze two Modal A0
# attempts on 2026-07-05. It supersedes hf_transfer, so DISABLE_XET is the
# load-bearing one; hf_transfer is turned off too so the fallback is the plain
# resolve-endpoint HTTP path.
HF_XET_MITIGATION = {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}

# Modal re-imports this module inside the remote container before hydrating the
# submitted Function dependencies.  Every environment key that changes a
# module-scope Modal object or its mount/config must therefore be present before
# that import.  Keep these values on the existing Function Secret dependency so
# local submission and remote re-import construct the same ordered graph:
# secret, image, cache Volume, output Volume, optional input Volume.
MODULE_IMPORT_MODAL_ENV_KEYS = (
    "MODAL_TRAINING_IMAGE",
    "MODAL_TRAINING_PIP_PACKAGES_JSON",
    "MODAL_GPU",
    "MODAL_TIMEOUT_SECONDS",
    "MODAL_CACHE_VOLUME_NAME",
    "MODAL_OUTPUT_VOLUME_NAME",
    "MODAL_OUTPUT_MOUNT_PATH",
    "MODAL_INPUT_VOLUME_NAME",
    "MODAL_INPUT_MOUNT_PATH",
)


def _function_secret_env(env=os.environ) -> dict[str, str]:
    """Build the exact pre-import environment for the remote Function."""
    payload = {
        "HF_TOKEN": env.get("HF_TOKEN", ""),
        "WANDB_API_KEY": env.get("WANDB_API_KEY", ""),
        "UNSLOTH_COMPILE_DISABLE": env.get("UNSLOTH_COMPILE_DISABLE", ""),
        "HF_HUB_DISABLE_XET": env.get("HF_HUB_DISABLE_XET", ""),
        "HF_HUB_ENABLE_HF_TRANSFER": env.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
    }
    # Omit unset/empty import keys instead of injecting empty strings. Several
    # module-scope declarations use ``os.environ.get(key, default)``; an empty
    # injected value would suppress that default and change the remote graph.
    payload.update(
        {
            key: value
            for key in MODULE_IMPORT_MODAL_ENV_KEYS
            if (value := env.get(key, ""))
        }
    )
    return payload


FUNCTION_SECRET_ENV = _function_secret_env()


def apply_hf_xet_mitigation(env) -> None:
    """Set the hf_xet-hang mitigation defaults on a mutable env mapping.

    Treats empty as unset: the @app.function secrets dict forwards these as ""
    when they are absent from the local launch env, and an empty value would
    neither disable xet nor trip a plain setdefault. So an explicit local value
    (e.g. "0") is preserved, while an unset-or-empty var falls through to the
    mitigation default -- the fix applies even on a bare `modal run`.
    """
    for key, default in HF_XET_MITIGATION.items():
        if not env.get(key):
            env[key] = default


# ---------------------------------------------------------------------------
# Training Function (runs remotely on Modal)
# ---------------------------------------------------------------------------

function_volumes = {
    "/cache/huggingface": model_cache,
    OUTPUT_MOUNT_PATH: output_volume,
}
if input_volume is not None:
    function_volumes[INPUT_MOUNT_PATH] = input_volume


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mounted_input_file(path_value: str, label: str) -> Path:
    """Resolve a staged file and prove it stays inside the input mount."""
    if not INPUT_VOLUME_NAME:
        raise ValueError(f"{label} requires MODAL_INPUT_VOLUME_NAME to be set.")
    if not path_value:
        raise ValueError(f"{label} is required for a config-driven Modal run.")
    mount = Path(INPUT_MOUNT_PATH).resolve(strict=True)
    path = Path(path_value).resolve(strict=True)
    try:
        path.relative_to(mount)
    except ValueError as exc:
        raise ValueError(f"{label} must resolve inside {INPUT_MOUNT_PATH}: {path}") from exc
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path


def _validate_staged_sft_config(
    config_path: str,
    expected_config_sha256: str,
    expected_dataset_sha256: str,
) -> tuple[Path, Path, str, str]:
    """Hash-bind a staged direct SFT YAML and its configured private dataset."""
    import yaml

    config = _mounted_input_file(config_path, "config_path")
    if config.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Config-driven Modal SFT requires a .yaml or .yml config file.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_config_sha256 or ""):
        raise ValueError("config_sha256 must be a full 64-character SHA-256 digest.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_dataset_sha256 or ""):
        raise ValueError("dataset_sha256 must be a full 64-character SHA-256 digest.")

    config_sha = _sha256(config)
    if not hmac.compare_digest(config_sha.lower(), expected_config_sha256.lower()):
        raise ValueError(
            f"Staged config SHA-256 mismatch: expected {expected_config_sha256}, got {config_sha}."
        )
    try:
        document = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot parse staged SFT YAML config {config}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("dataset"), dict):
        raise ValueError("Staged SFT YAML must contain a top-level dataset mapping.")
    dataset_value = document["dataset"].get("local_file")
    if not isinstance(dataset_value, str) or not dataset_value:
        raise ValueError("Staged SFT YAML dataset.local_file must be a non-empty string.")
    dataset = _mounted_input_file(dataset_value, "dataset.local_file")
    dataset_sha = _sha256(dataset)
    if not hmac.compare_digest(dataset_sha.lower(), expected_dataset_sha256.lower()):
        raise ValueError(
            f"Staged dataset SHA-256 mismatch: expected {expected_dataset_sha256}, got {dataset_sha}."
        )
    return config, dataset, config_sha, dataset_sha


def build_training_command(
    *,
    train_script: str,
    run_timestamp: str,
    config_path: str = "",
    resume_from_checkpoint: str = "",
) -> list[str]:
    """Construct the shell-free trainer argv shared by remote execution/tests."""
    command = [
        "python",
        train_script,
        "--run-timestamp",
        run_timestamp,
        "--output-root",
        f"{OUTPUT_MOUNT_PATH}/outputs",
        "--cloud-provider",
        "modal",
        "--artifact-backend",
        "modal_volume",
    ]
    if config_path:
        command.extend(["--config", config_path])
    if resume_from_checkpoint:
        command.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    return command


def commit_success_volumes() -> None:
    """Persist canonical outputs before the replaceable model cache."""
    output_volume.commit()
    model_cache.commit()


def _run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True
    )
    if result.returncode != 0:
        error = _redact_text(result.stderr or result.stdout or "git command failed")
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {error}")
    return result


def _checkout_workspace(
    *, repo_url: str, repo_branch: str, repo_commit: str, workspace: Path
) -> None:
    """Idempotently materialize an exact source commit in a warm Modal container."""
    if workspace.exists():
        if not workspace.is_dir() or not (workspace / ".git").is_dir():
            raise RuntimeError(f"Existing workspace is not a git checkout: {workspace}")
        existing_url = _credential_free_repo_url(
            _run_git(["remote", "get-url", "origin"], cwd=workspace).stdout.strip()
        )
        if existing_url != repo_url:
            raise RuntimeError(
                "Existing workspace origin does not match the requested credential-free repo URL."
            )
        # Trainers and optional trackers may leave untracked runtime byproducts
        # (for example wandb/). They cannot alter the pinned tracked source and
        # must not make a warm retry unrecoverable. Any tracked modification is
        # source drift and remains a hard failure.
        dirty = _run_git(["status", "--porcelain", "--untracked-files=no"], cwd=workspace)
        if dirty.stdout.strip():
            raise RuntimeError(
                "Existing workspace is dirty; refusing to overwrite warm-container state."
            )
        _run_git(["fetch", "--depth", "1", "origin", repo_commit], cwd=workspace)
    else:
        workspace.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            [
                "clone",
                "--branch",
                repo_branch,
                "--depth",
                "1",
                "--no-checkout",
                repo_url,
                str(workspace),
            ]
        )
        # A shallow branch clone can race with a branch-tip advance between
        # planning and execution. Fetch the immutable requested object before
        # checkout even on a cold container.
        _run_git(["fetch", "--depth", "1", "origin", repo_commit], cwd=workspace)

    _run_git(["checkout", "--detach", repo_commit], cwd=workspace)
    resolved_head = _run_git(["rev-parse", "HEAD"], cwd=workspace).stdout.strip().lower()
    if resolved_head != repo_commit.lower():
        raise RuntimeError(
            f"Exact source verification failed: requested {repo_commit}, "
            f"got {resolved_head or 'unknown'}."
        )


def _provenance_identity(provenance: dict) -> dict:
    """Select immutable fields that must agree across retries."""
    inputs = provenance.get("inputs") or {}
    return {
        "run": provenance.get("run"),
        "source": provenance.get("source"),
        "inputs": inputs,
        "runtime": provenance.get("runtime"),
        "artifacts": provenance.get("artifacts"),
        "cache": provenance.get("cache"),
        "publish_final_model": provenance.get("publish_final_model"),
    }


def _read_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read valid {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} at {path} must contain a JSON object.")
    return payload


def _validate_stable_namespace(
    *, namespace: Path, run_dir: Path, run_id: str, expected_provenance: dict
) -> None:
    """Reject source drift, collisions, or unowned state in a stable namespace."""
    _validate_stable_namespace_location(
        namespace=namespace, run_dir=run_dir, run_id=run_id
    )
    if not run_dir.exists():
        return
    provenance_path = run_dir / "modal_job_provenance.json"
    if not provenance_path.is_file():
        if any(run_dir.iterdir()):
            raise RuntimeError(
                f"Stable run directory contains state without Modal provenance: {run_dir}"
            )
        return
    existing = _read_json_object(provenance_path, "Modal job provenance")
    if _provenance_identity(existing) != _provenance_identity(expected_provenance):
        raise RuntimeError(
            f"Stable run namespace {run_id!r} does not match requested source, inputs, or runtime."
        )


def _validate_stable_namespace_location(
    *, namespace: Path, run_dir: Path, run_id: str
) -> None:
    """Validate only the path ownership needed before staged input verification."""
    if namespace.exists():
        pattern = re.compile(rf"^{re.escape(run_id)}-[0-9a-fA-F]{{8}}$")
        collisions = [path for path in namespace.iterdir() if pattern.fullmatch(path.name)]
        unexpected = [path for path in collisions if path != run_dir]
        if unexpected:
            names = ", ".join(sorted(path.name for path in unexpected))
            raise RuntimeError(
                f"Stable run namespace {run_id!r} is ambiguous or source-drifted: {names}."
            )
    if not run_dir.exists():
        return
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError(f"Stable run path is not a directory: {run_dir}")


def _latest_valid_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest complete Hugging Face checkpoint, failing on ambiguity."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    if checkpoints_dir.is_symlink():
        raise RuntimeError(f"Checkpoint root must not be a symlink: {checkpoints_dir}")
    candidates: dict[int, Path] = {}
    for path in checkpoints_dir.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if not match or not path.is_dir():
            continue
        if path.is_symlink():
            raise RuntimeError(f"Checkpoint directory must not be a symlink: {path}")
        step = int(match.group(1))
        if step in candidates:
            raise RuntimeError(f"Ambiguous checkpoints for step {step} in {checkpoints_dir}.")
        state_path = path / "trainer_state.json"
        if not state_path.is_file():
            continue
        try:
            state = _read_json_object(state_path, "trainer checkpoint state")
        except RuntimeError:
            continue
        if state.get("global_step") != step:
            continue
        optimizer = path / "optimizer.pt"
        scheduler = path / "scheduler.pt"
        rng_files = sorted(path.glob("rng_state*.pth"))
        model_files = [
            path / name
            for name in ("adapter_model.safetensors", "model.safetensors", "pytorch_model.bin")
            if (path / name).is_file()
        ]
        if not optimizer.is_file() or not scheduler.is_file() or not rng_files or not model_files:
            continue
        try:
            _validate_torch_state(optimizer)
            _validate_torch_state(scheduler)
            for rng_file in rng_files:
                _validate_torch_state(rng_file)
            for model_file in model_files:
                _validate_model_state(model_file)
        except Exception:
            # Public restricted readers use several library-specific exception
            # types for truncation/corruption; all mean this candidate is not a
            # valid resume point, so continue to the next-lower complete step.
            continue
        candidates[step] = path
    return candidates[max(candidates)] if candidates else None


def _validate_torch_state(path: Path) -> None:
    """CRC-check a modern torch.save archive without unsafe pickle execution."""
    import zipfile

    if not zipfile.is_zipfile(path):
        raise ValueError(f"Expected a torch zip archive: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not any(name.endswith("/data.pkl") for name in names):
            raise ValueError(f"Torch archive has no data.pkl payload: {path}")
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Torch archive failed CRC validation at {corrupt_member}: {path}")


def _validate_model_state(path: Path) -> None:
    """Fully read a model/adapter artifact so truncated tensor data is rejected."""
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys:
                raise ValueError(f"Safetensors artifact contains no tensors: {path}")
            for key in keys:
                handle.get_tensor(key)
        return
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected non-empty tensor mapping in model state: {path}")
    if not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError(f"Model state contains non-tensor values: {path}")


def _config_requires_special_token_lineage(provenance: dict) -> bool:
    """Read the hash-bound staged YAML to determine the required artifact set."""
    import yaml

    config_record = ((provenance.get("inputs") or {}).get("config") or {})
    config_path = Path(str(config_record.get("mounted_path") or ""))
    expected_sha = str(config_record.get("sha256") or "")
    if not config_path.is_file() or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
        raise RuntimeError("Stable provenance has no valid authoritative staged SFT config.")
    if not hmac.compare_digest(_sha256(config_path), expected_sha.lower()):
        raise RuntimeError("Authoritative staged SFT config no longer matches its bound hash.")
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot parse authoritative staged SFT config: {exc}") from exc
    if not isinstance(document, dict):
        raise RuntimeError("Authoritative staged SFT config must contain a mapping.")
    tokenizer = ((document.get("model") or {}).get("tokenizer") or {})
    if not isinstance(tokenizer, dict):
        raise RuntimeError("model.tokenizer must be a mapping when present.")
    tokens = tokenizer.get("additional_special_tokens")
    if tokens is None:
        return False
    if not isinstance(tokens, list):
        raise RuntimeError("model.tokenizer.additional_special_tokens must be a list.")
    return bool(tokens)


def _validate_completed_artifacts(run_dir: Path, provenance: dict) -> None:
    """Require the complete canonical SFT result before accepting completion."""
    manifest = _read_json_object(run_dir / "manifest.json", "completed run manifest")
    if manifest.get("status") != "completed":
        raise RuntimeError(f"Canonical manifest is not completed: {run_dir / 'manifest.json'}")
    if manifest.get("repo_commit") != provenance["source"]["commit"]:
        raise RuntimeError("Completed manifest repo commit does not match Modal provenance.")
    manifest_provenance = manifest.get("cloud_job_provenance") or {}
    if _provenance_identity(manifest_provenance) != _provenance_identity(provenance):
        raise RuntimeError("Completed manifest provenance identity does not match the stable run.")

    lineage = _read_json_object(run_dir / "training_lineage.json", "training lineage")
    if not lineage:
        raise RuntimeError("training_lineage.json must contain a non-empty object.")
    lineage_provenance = lineage.get("cloud_job_provenance")
    if not isinstance(lineage_provenance, dict):
        raise RuntimeError("training_lineage.json is missing cloud_job_provenance.")
    if _provenance_identity(lineage_provenance) != _provenance_identity(provenance):
        raise RuntimeError("Training lineage provenance identity does not match the stable run.")

    final_model = run_dir / "final_model"
    if final_model.is_symlink() or not final_model.is_dir():
        raise RuntimeError(f"Completed run is missing a real final_model directory: {final_model}")
    adapter_config = _read_json_object(final_model / "adapter_config.json", "adapter config")
    if not adapter_config:
        raise RuntimeError("adapter_config.json must contain a non-empty object.")
    adapter_weights = [
        path
        for path in (
            final_model / "adapter_model.safetensors",
            final_model / "adapter_model.bin",
        )
        if path.is_file() and not path.is_symlink()
    ]
    if len(adapter_weights) != 1:
        raise RuntimeError("final_model must contain exactly one supported adapter weight artifact.")
    _validate_model_state(adapter_weights[0])

    tokenizer_config = _read_json_object(
        final_model / "tokenizer_config.json", "tokenizer config"
    )
    if not tokenizer_config:
        raise RuntimeError("tokenizer_config.json must contain a non-empty object.")
    tokenizer_payloads = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
    )
    if not any(
        (final_model / name).is_file() and not (final_model / name).is_symlink()
        for name in tokenizer_payloads
    ):
        raise RuntimeError("final_model is missing a serialized tokenizer vocabulary artifact.")
    for name in tokenizer_payloads:
        payload_path = final_model / name
        if not payload_path.is_file():
            continue
        if payload_path.stat().st_size <= 0:
            raise RuntimeError(f"Tokenizer payload is empty: {payload_path}")
        if payload_path.suffix == ".json":
            _read_json_object(payload_path, "tokenizer payload")

    special_lineage_path = final_model / "special_tokens_lineage.json"
    if _config_requires_special_token_lineage(provenance):
        special_lineage = _read_json_object(special_lineage_path, "special-token lineage")
        if not isinstance(special_lineage.get("resolved_config"), dict):
            raise RuntimeError("special-token lineage is missing resolved_config.")
        for field in ("config_sha256", "vocab_sha256_after"):
            if not re.fullmatch(r"[0-9a-fA-F]{64}", str(special_lineage.get(field, ""))):
                raise RuntimeError(f"special-token lineage has no valid {field}.")


def _write_done_marker(run_dir: Path, provenance: dict) -> None:
    payload = {
        "schema_version": 1,
        "status": "completed",
        "identity": _provenance_identity(provenance),
    }
    (run_dir / DONE_MARKER_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_completion_ready_marker(run_dir: Path, provenance: dict) -> None:
    (run_dir / COMPLETION_READY_MARKER_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "identity": _provenance_identity(provenance),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_completion_ready(run_dir: Path, provenance: dict) -> bool:
    marker = run_dir / COMPLETION_READY_MARKER_NAME
    if not marker.exists():
        return False
    payload = _read_json_object(marker, "completion-ready marker")
    if payload.get("status") != "ready" or payload.get("identity") != _provenance_identity(
        provenance
    ):
        raise RuntimeError(f"Completion-ready marker identity does not match: {marker}")
    _validate_completed_artifacts(run_dir, provenance)
    return True


def _is_completed_retry(run_dir: Path, provenance: dict) -> bool:
    marker = run_dir / DONE_MARKER_NAME
    if not marker.exists():
        return False
    payload = _read_json_object(marker, "DONE marker")
    if payload.get("status") != "completed" or payload.get("identity") != _provenance_identity(
        provenance
    ):
        raise RuntimeError(f"DONE marker identity does not match stable run: {marker}")
    if not _validate_completion_ready(run_dir, provenance):
        raise RuntimeError(f"DONE marker has no committed completion-ready phase: {marker}")
    return True


def _reload_output_volume() -> None:
    """Discard local uncommitted mount state before trusting retry markers."""
    output_volume.reload()


def _commit_stable_completion(run_dir: Path, provenance: dict) -> None:
    """Two-phase commit canonical outputs, then a commit-confirmed DONE marker."""
    _write_wrapper_state(run_dir, provenance, "completed")
    _validate_completed_artifacts(run_dir, provenance)
    _write_completion_ready_marker(run_dir, provenance)
    output_volume.commit()
    _promote_ready_completion(run_dir, provenance)


def _promote_ready_completion(run_dir: Path, provenance: dict) -> bool:
    """Finish a committed artifact phase without rerunning expensive training."""
    if not _validate_completion_ready(run_dir, provenance):
        return False
    _write_done_marker(run_dir, provenance)
    try:
        output_volume.commit()
    except Exception:
        # A same-container retry reloads the committed Volume before reading
        # DONE. Removing the local copy also prevents accidental in-process use.
        (run_dir / DONE_MARKER_NAME).unlink(missing_ok=True)
        raise
    model_cache.commit()
    return True


def _run_with_periodic_output_commits(
    cmd: list[str], *, cwd: str, env: dict, interval_seconds: float = OUTPUT_COMMIT_INTERVAL_SECONDS
) -> int:
    """Run a trainer while best-effort commits bound crash exposure on the output Volume."""
    if interval_seconds <= 0 or interval_seconds > 120:
        raise ValueError("Periodic output commit interval must be >0 and <=120 seconds.")
    process = subprocess.Popen(cmd, cwd=cwd, env=env)
    stopped = threading.Event()

    def commit_loop() -> None:
        while not stopped.wait(interval_seconds):
            try:
                output_volume.commit()
            except Exception as exc:  # A transient commit error must not kill or mask training.
                print(
                    "[Modal] Periodic output Volume commit failed (will retry): "
                    f"{_redact_text(exc)}"
                )

    thread = threading.Thread(target=commit_loop, name="modal-output-commit", daemon=True)
    thread.start()
    wait_error = None
    try:
        return process.wait()
    except BaseException as exc:
        wait_error = exc
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            finally:
                process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        except Exception:
            try:
                process.kill()
            finally:
                process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        raise
    finally:
        stopped.set()
        thread.join(timeout=COMMIT_THREAD_JOIN_TIMEOUT_SECONDS)
        if thread.is_alive():
            message = "Periodic output Volume commit thread did not stop within its bound."
            if wait_error is not None:
                raise RuntimeError(message) from wait_error
            raise RuntimeError(message)


def _modal_provenance(
    *,
    repo_branch: str,
    repo_commit: str,
    config_path: str,
    config_sha256: str,
    dataset_path: str,
    dataset_sha256: str,
    inputs_verified: bool,
    run_id: str = "",
) -> dict:
    return {
        "schema_version": 1,
        "provider": "modal",
        "run": {"id": run_id or None, "stable": bool(run_id)},
        "source": {"branch": repo_branch, "commit": repo_commit.lower()},
        "inputs": {
            "config": {"mounted_path": config_path or None, "sha256": config_sha256 or None},
            "dataset": {"mounted_path": dataset_path or None, "sha256": dataset_sha256 or None},
            "verified": inputs_verified,
            "volume_name": INPUT_VOLUME_NAME or None,
            "mount_path": INPUT_MOUNT_PATH if INPUT_VOLUME_NAME else None,
        },
        "runtime": {
            "image": runtime_image_ref or "legacy_builtin_modal_image",
            "pip_packages": list(runtime_pips),
            "gpu": FUNCTION_GPU,
            "timeout_seconds": FUNCTION_TIMEOUT_SECONDS,
        },
        "artifacts": {
            "volume_name": os.environ.get(
                "MODAL_OUTPUT_VOLUME_NAME", "toolset-training-artifacts"
            ),
            "mount_path": OUTPUT_MOUNT_PATH,
        },
        "cache": {
            "volume_name": os.environ.get("MODAL_CACHE_VOLUME_NAME", "toolset-model-cache"),
            "mount_path": "/cache/huggingface",
        },
        "publish_final_model": False,
    }


def _write_wrapper_state(run_dir: Path, provenance: dict, status: str, error: str = "") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(provenance)
    payload["status"] = status
    if error:
        payload["error"] = _redact_text(error)[:2000]
    (run_dir / "modal_job_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}
    manifest.update(
        {
            "provider": "modal",
            "method": "sft",
            "status": status if not error else f"failed: {_redact_text(error)[:500]}",
            "repo_branch": provenance["source"]["branch"],
            "repo_commit": provenance["source"]["commit"],
            "cloud_job_provenance": payload,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _commit_failed_state(run_dir: Path, provenance: dict | None, error: str) -> None:
    """Persist failure evidence to the output Volume; never commit replaceable cache state."""
    if provenance:
        _write_wrapper_state(run_dir, provenance, "failed", _redact_text(error))
    output_volume.commit()


def _commit_preflight_failure(run_dir: Path, provenance: dict, error: str) -> None:
    """Record failure before full input identity without claiming the run namespace."""
    failure_dir = run_dir.parent / "_preflight_failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(provenance)
    payload.update({"status": "failed", "error": _redact_text(error)[:2000]})
    (failure_dir / f"{run_dir.name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_volume.commit()


def _run_training_impl(
    trainer_type: str = "sft",
    repo_url: str = "",
    repo_branch: str = "main",
    repo_commit: str = "",
    model_name: str = "",
    dataset_path: str = "",
    dataset_name: str = "",
    dataset_file: str = "",
    publish_final_model: bool = False,
    publish_target_repo: str = "",
    config_overrides: str = "",
    config_path: str = "",
    config_sha256: str = "",
    dataset_sha256: str = "",
    run_id: str = "",
):
    """Run SFT or KTO training inside the Modal container.

    This function executes remotely on Modal's GPU infrastructure. It:
    1. Clones the Toolset-Training repo into the container
    2. Sets up the HuggingFace cache to use the persistent volume
    3. Runs the appropriate training script (train_sft.py or train_kto.py)
        4. Publishes final_model to HuggingFace Hub only when explicitly requested

    Args:
        trainer_type: Training method - "sft" or "kto"
        repo_url: Git URL to clone the training repo from
        repo_branch: Git branch to checkout
        model_name: Override model name (uses config.yaml default if empty)
        dataset_path: Override dataset path relative to repo root
        publish_final_model: Whether to publish final_model to HuggingFace Hub
        publish_target_repo: HuggingFace Hub repo ID to push trained model to
        config_overrides: JSON object of CLI argument overrides. Programmatic
            callers may continue to pass a mapping for backward compatibility.
        run_id: Optional stable, path-safe run identifier. Config-driven SFT
            retries with the same identifier resume the same canonical run.
    """
    config_overrides = _config_overrides_mapping(config_overrides)

    # Validate trainer type and exact source metadata.
    if trainer_type not in ("sft", "kto"):
        raise ValueError(f"Invalid trainer_type: {trainer_type}. Must be 'sft' or 'kto'.")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", repo_commit or ""):
        raise ValueError("repo_commit must be a full 40-character git commit SHA.")
    if config_path and trainer_type != "sft":
        raise ValueError("External YAML config_path is currently supported only for SFT.")
    if config_path and any((dataset_path, dataset_name, dataset_file, model_name, config_overrides)):
        raise ValueError(
            "config_path is authoritative and cannot be combined with model, dataset, or training overrides."
        )
    if config_path and (publish_final_model or publish_target_repo):
        raise ValueError(
            "config-driven Modal SFT jobs retain private Volume artifacts and cannot publish to the Hub."
        )
    run_id = _stable_run_id(run_id)
    if run_id and not config_path:
        raise ValueError("run_id is currently supported only for config-driven Modal SFT jobs.")
    repo_url = _credential_free_repo_url(repo_url)
    run_timestamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_namespace = Path(OUTPUT_MOUNT_PATH) / "outputs" / "runs" / "modal" / trainer_type
    modal_run_dir = (
        run_namespace / f"{run_timestamp}-{repo_commit[:8].lower()}"
    )
    if run_id:
        _reload_output_volume()
        _validate_stable_namespace_location(
            namespace=run_namespace, run_dir=modal_run_dir, run_id=run_id
        )
    job_provenance = None
    staged_inputs = None
    if config_path:
        job_provenance = _modal_provenance(
            repo_branch=repo_branch,
            repo_commit=repo_commit,
            config_path=config_path,
            config_sha256=config_sha256,
            dataset_path="",
            dataset_sha256=dataset_sha256,
            inputs_verified=False,
            run_id=run_id,
        )
        try:
            config_file, dataset_file_path, config_sha, dataset_sha = _validate_staged_sft_config(
                config_path, config_sha256, dataset_sha256
            )
        except Exception as exc:
            error = _redact_text(str(exc))
            # Never claim or overwrite a stable canonical namespace before its
            # complete mounted-input identity has been verified.
            if run_id:
                _commit_preflight_failure(modal_run_dir, job_provenance, error)
            else:
                _commit_failed_state(modal_run_dir, job_provenance, error)
            raise RuntimeError(error) from None
        staged_inputs = {
            "config_path": str(config_file),
            "config_sha256": config_sha,
            "dataset_path": str(dataset_file_path),
            "dataset_sha256": dataset_sha,
        }
        job_provenance = _modal_provenance(
            repo_branch=repo_branch,
            repo_commit=repo_commit,
            config_path=str(config_file),
            config_sha256=config_sha,
            dataset_path=str(dataset_file_path),
            dataset_sha256=dataset_sha,
            inputs_verified=True,
            run_id=run_id,
        )
        if run_id:
            _validate_stable_namespace(
                namespace=run_namespace,
                run_dir=modal_run_dir,
                run_id=run_id,
                expected_provenance=job_provenance,
            )
            completed = _is_completed_retry(modal_run_dir, job_provenance)
            if not completed:
                completed = _promote_ready_completion(modal_run_dir, job_provenance)
            if completed:
                print(
                    f"[Modal] Stable run {run_id!r} has committed complete artifacts; "
                    "retry is a no-op."
                )
                return {
                    "status": "completed",
                    "no_op": True,
                    "trainer_type": trainer_type,
                    "repo_commit": repo_commit.lower(),
                    "staged_inputs": staged_inputs,
                    "artifact_root": str(modal_run_dir),
                }
        _write_wrapper_state(modal_run_dir, job_provenance, "preparing_source")
        output_volume.commit()

    # Point HuggingFace cache at the persistent volume to avoid re-downloading models
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True

    # Dodge the hf_xet download hang (see apply_hf_xet_mitigation).
    apply_hf_xet_mitigation(os.environ)

    # Bridge the repo provenance into the env contract the training scripts read
    # (train_sft/train_kto/train_dpo consume CLOUD_REPO_BRANCH / CLOUD_REPO_COMMIT
    # from os.environ to stamp manifest.json and name the run dir). The
    # ModalBackend sets these in the LOCAL `modal run` process env, but Modal
    # only forwards explicitly-declared secrets into the remote container, so
    # the local env never reaches this function -- without this bridge the
    # manifest records repo_commit=null and the run dir is stamped "-local".
    # These args are the exact commit this container checked out above, so they
    # are the authoritative provenance regardless of provider.
    if repo_branch:
        os.environ["CLOUD_REPO_BRANCH"] = repo_branch
    if repo_commit:
        os.environ["CLOUD_REPO_COMMIT"] = repo_commit

    workspace = Path("/workspace/toolset-training")
    try:
        print(f"[Modal] Materializing exact source (branch: {repo_branch})")
        _checkout_workspace(
            repo_url=repo_url,
            repo_branch=repo_branch,
            repo_commit=repo_commit,
            workspace=workspace,
        )
    except Exception as exc:
        error = _redact_text(str(exc))
        _commit_failed_state(modal_run_dir, job_provenance, error)
        raise RuntimeError(error) from None

    # The cloned repo provides shared/; resolve the path helpers here because
    # the container image does not carry the repo source at module-import time.
    workspace_str = str(workspace)
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
    from shared.utilities.paths import (  # noqa: F811 (module-scope import is best-effort)
        get_canonical_output_dir_name,
        get_canonical_trainer_dir_name,
    )

    # Determine trainer directory and script
    trainer_dir = os.path.join(
        workspace_str, "Trainers", get_canonical_trainer_dir_name(trainer_type)
    )
    train_script = f"train_{trainer_type}.py"

    if not os.path.isfile(os.path.join(trainer_dir, train_script)):
        error = f"Training script not found: {os.path.join(trainer_dir, train_script)}"
        _commit_failed_state(modal_run_dir, job_provenance, error)
        raise FileNotFoundError(
            error
        )

    try:
        resume_checkpoint = _latest_valid_checkpoint(modal_run_dir) if run_id else None
    except Exception as exc:
        error = _redact_text(str(exc))
        _commit_failed_state(modal_run_dir, job_provenance, error)
        raise RuntimeError(error) from None
    cmd = build_training_command(
        train_script=train_script,
        run_timestamp=run_timestamp,
        config_path=str(config_file) if config_path else "",
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else "",
    )

    if job_provenance:
        os.environ["SYNAPTIC_CLOUD_JOB_PROVENANCE_JSON"] = json.dumps(
            job_provenance, sort_keys=True, separators=(",", ":")
        )
        _write_wrapper_state(modal_run_dir, job_provenance, "running")
        output_volume.commit()
    if resume_checkpoint:
        print(f"[Modal] Resuming stable run from {resume_checkpoint}")

    if model_name:
        # Model name override requires modifying config before training.
        # For now, this is noted as a future enhancement -- the config.yaml
        # in the repo should be pre-configured with the desired model.
        print(f"[Modal] Note: model_name override '{model_name}' requires config.yaml update")

    if dataset_path:
        # Convert relative dataset path to absolute within the workspace
        abs_dataset = os.path.join(workspace_str, dataset_path)
        cmd.extend(["--local-file", abs_dataset])
    if dataset_name:
        # Pull the dataset from the Hugging Face Hub instead of the config
        # default. dataset_file selects a specific file inside that repo.
        cmd.extend(["--dataset-name", dataset_name])
    if dataset_file:
        cmd.extend(["--dataset-file", dataset_file])
    if publish_final_model:
        cmd.append("--publish-final-model")
    if publish_target_repo:
        cmd.extend(["--publish-target-repo", publish_target_repo])

    # Apply config overrides as CLI arguments
    if config_overrides.get("learning_rate"):
        cmd.extend(["--learning-rate", str(config_overrides["learning_rate"])])
    if config_overrides.get("num_epochs"):
        cmd.extend(["--num-epochs", str(config_overrides["num_epochs"])])
    if config_overrides.get("batch_size"):
        cmd.extend(["--batch-size", str(config_overrides["batch_size"])])
    if config_overrides.get("max_seq_length"):
        cmd.extend(["--max-seq-length", str(config_overrides["max_seq_length"])])
    if config_overrides.get("max_steps"):
        cmd.extend(["--max-steps", str(config_overrides["max_steps"])])

    print(f"[Modal] Running: {' '.join(cmd)}")
    print(f"[Modal] Working directory: {trainer_dir}")

    try:
        returncode = _run_with_periodic_output_commits(
            cmd, cwd=trainer_dir, env={**os.environ}
        )
        if returncode != 0:
            raise RuntimeError(f"Training script exited with code {returncode}")
    except Exception as exc:
        error = _redact_text(str(exc))
        print(f"[Modal] Training failed: {error}")
        _commit_failed_state(modal_run_dir, job_provenance, error)
        raise RuntimeError(error) from None

    print("[Modal] Training completed successfully")

    # Stable jobs require complete canonical artifacts and a commit-confirmed
    # DONE marker. Legacy jobs retain their historical single output/cache
    # commit behavior and never acquire retry markers.
    if run_id:
        _commit_stable_completion(modal_run_dir, job_provenance)
    else:
        if job_provenance:
            _write_wrapper_state(modal_run_dir, job_provenance, "completed")
        commit_success_volumes()

    return {
        "status": "completed",
        "trainer_type": trainer_type,
        "repo_commit": repo_commit.lower(),
        "staged_inputs": staged_inputs,
        "artifact_root": (
            str(modal_run_dir)
            if run_id
            else f"{OUTPUT_MOUNT_PATH}/outputs/runs/modal/{trainer_type}"
        ),
    }


def _modal_function_options() -> dict:
    """Shared dependency graph for both public direct-remote entrypoints."""
    return {
        "gpu": FUNCTION_GPU,
        "timeout": FUNCTION_TIMEOUT_SECONDS,
        "volumes": function_volumes,
        # Scope credentials and the module-import Modal contract explicitly.
        "secrets": [modal.Secret.from_dict(FUNCTION_SECRET_ENV)],
    }


@app.function(**_modal_function_options())
def run_training(
    trainer_type: str = "sft",
    repo_url: str = "",
    repo_branch: str = "main",
    repo_commit: str = "",
    model_name: str = "",
    dataset_path: str = "",
    dataset_name: str = "",
    dataset_file: str = "",
    publish_final_model: bool = False,
    publish_target_repo: str = "",
    config_overrides: str = "",
    config_path: str = "",
    config_sha256: str = "",
    dataset_sha256: str = "",
):
    """Legacy direct Modal entrypoint; retains its historical no-retry semantics."""
    return _run_training_impl(
        trainer_type=trainer_type,
        repo_url=repo_url,
        repo_branch=repo_branch,
        repo_commit=repo_commit,
        model_name=model_name,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        publish_final_model=publish_final_model,
        publish_target_repo=publish_target_repo,
        config_overrides=config_overrides,
        config_path=config_path,
        config_sha256=config_sha256,
        dataset_sha256=dataset_sha256,
        run_id="",
    )


@app.function(
    **_modal_function_options(),
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=1.0),
)
def run_stable_training(
    trainer_type: str = "sft",
    repo_url: str = "",
    repo_branch: str = "main",
    repo_commit: str = "",
    config_path: str = "",
    config_sha256: str = "",
    dataset_sha256: str = "",
    run_id: str = "",
):
    """Retry-enabled direct entrypoint for stable config-driven SFT jobs only."""
    if not _stable_run_id(run_id):
        raise ValueError("run_stable_training requires a non-empty stable run_id.")
    if not config_path:
        raise ValueError("run_stable_training requires a config_path.")
    return _run_training_impl(
        trainer_type=trainer_type,
        repo_url=repo_url,
        repo_branch=repo_branch,
        repo_commit=repo_commit,
        config_path=config_path,
        config_sha256=config_sha256,
        dataset_sha256=dataset_sha256,
        run_id=run_id,
    )


def _upload_to_hub(trainer_type: str, trainer_dir: str, hub_repo: str):
    """Upload trained model to HuggingFace Hub.

    Looks for the most recent training run output directory and uploads
    the final_model subdirectory to the specified Hub repository.

    Args:
        trainer_type: "sft" or "kto" (determines output directory name)
        trainer_dir: Path to the trainer directory
        hub_repo: HuggingFace Hub repo ID (e.g., "username/model-name")
    """
    from pathlib import Path
    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[Modal] Warning: HF_TOKEN not set, skipping Hub upload")
        return

    output_dir_name = get_canonical_output_dir_name(trainer_type)
    output_base = Path(trainer_dir) / output_dir_name

    if not output_base.exists():
        print(f"[Modal] Warning: Output directory not found: {output_base}")
        return

    # Find the most recent run directory (sorted by timestamp name)
    run_dirs = sorted(output_base.iterdir(), reverse=True)
    if not run_dirs:
        print(f"[Modal] Warning: No training runs found in {output_base}")
        return

    latest_run = run_dirs[0]
    final_model = latest_run / "final_model"

    if not final_model.exists():
        print(f"[Modal] Warning: No final_model directory in {latest_run}")
        return

    print(f"[Modal] Uploading {final_model} to {hub_repo}")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hub_repo, exist_ok=True, private=True)
    api.upload_folder(
        folder_path=str(final_model),
        repo_id=hub_repo,
        commit_message=f"Modal cloud training ({trainer_type})",
    )
    print(f"[Modal] Upload complete: https://huggingface.co/{hub_repo}")


# ---------------------------------------------------------------------------
# Local Entrypoint (runs locally, dispatches to Modal)
# ---------------------------------------------------------------------------

def main(
    trainer_type: str = "sft",
    gpu: str = DEFAULT_GPU,
    repo_url: str = "",
    repo_branch: str = "main",
    repo_commit: str = "",
    model_name: str = "",
    dataset_path: str = "",
    dataset_name: str = "",
    dataset_file: str = "",
    publish_final_model: bool = False,
    publish_target_repo: str = "",
    learning_rate: float = 0.0,
    num_epochs: int = 0,
    batch_size: int = 0,
    max_steps: int = 0,
    timeout_hours: int = 6,
    config_path: str = "",
    config_sha256: str = "",
    dataset_sha256: str = "",
):
    """Local entrypoint for Modal cloud training.

    This function runs on your local machine and dispatches the training
    job to Modal's cloud infrastructure. Use `modal run` to invoke it.

    Args:
        trainer_type: Training method - "sft" or "kto"
        gpu: GPU type (T4, L4, A10G, L40S, A100, A100-80GB, H100)
        repo_url: Git URL to clone (auto-detected from local git remote if empty)
        repo_branch: Git branch to checkout (default: main)
        model_name: Override model name in config
        dataset_path: Override dataset path (relative to repo root)
        publish_final_model: Publish final_model to HuggingFace Hub
        publish_target_repo: HuggingFace Hub repo ID for publishing final_model
        learning_rate: Override learning rate (0 = use config default)
        num_epochs: Override number of epochs (0 = use config default)
        batch_size: Override batch size (0 = use config default)
        max_steps: Override max training steps (0 = use config default)
        timeout_hours: Maximum job duration in hours (default: 6)
    """
    # Validate GPU type
    if gpu not in VALID_GPU_TYPES:
        print(f"Error: Invalid GPU type '{gpu}'")
        print(f"Valid options: {', '.join(VALID_GPU_TYPES)}")
        sys.exit(1)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", repo_commit or ""):
        print("Error: --repo-commit must be a full 40-character git commit SHA.")
        sys.exit(1)

    # Auto-detect repo URL from local git remote if not provided.
    # Mirrors tuner/backends/training/cloud/base_cloud.resolve_repo_url()
    # but kept inline because this standalone script cannot import from tuner.*.
    if not repo_url:
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                repo_url = result.stdout.strip()
                print(f"[Modal] Auto-detected repo URL: {repo_url}")
            else:
                print("Error: No --repo-url provided and could not auto-detect from git remote.")
                print("Set CLOUD_REPO_URL env var or pass --repo-url explicitly.")
                sys.exit(1)
        except FileNotFoundError:
            print("Error: git not found. Please provide --repo-url explicitly.")
            sys.exit(1)

    # Also check CLOUD_REPO_URL env var as fallback
    if not repo_url:
        repo_url = os.environ.get("CLOUD_REPO_URL", "")
        if not repo_url:
            print("Error: Could not determine repo URL.")
            sys.exit(1)

    # Build config overrides dict (only include non-default values)
    config_overrides = {}
    if learning_rate > 0:
        config_overrides["learning_rate"] = learning_rate
    if num_epochs > 0:
        config_overrides["num_epochs"] = num_epochs
    if batch_size > 0:
        config_overrides["batch_size"] = batch_size
    if max_steps > 0:
        config_overrides["max_steps"] = max_steps

    # Display job configuration
    estimated_cost = GPU_PRICING.get(gpu, 0) * timeout_hours
    print("\n" + "=" * 60)
    print("MODAL CLOUD TRAINING")
    print("=" * 60)
    print(f"  Trainer:     {trainer_type.upper()}")
    print(f"  GPU:         {gpu}")
    print(f"  Timeout:     {timeout_hours}h")
    print(f"  Est. Cost:   ~${estimated_cost:.2f} (max)")
    print(f"  Repo:        {repo_url}")
    print(f"  Branch:      {repo_branch}")
    if model_name:
        print(f"  Model:       {model_name}")
    if dataset_path:
        print(f"  Dataset:     {dataset_path}")
    if publish_target_repo:
        print(f"  Publish Repo:{publish_target_repo}")
    if config_overrides:
        print(f"  Overrides:   {config_overrides}")
    print("=" * 60 + "\n")

    # Override the GPU and timeout for this specific invocation.
    # Modal's `with_options` allows runtime overrides of function config.
    training_fn = run_training.with_options(
        gpu=gpu,
        timeout=timeout_hours * HOURS,
    )

    raise RuntimeError(
        "The local Modal entrypoint is intentionally disabled because nested "
        "remote/spawn submission can create an app with zero running tasks. "
        "Launch the remote function directly with: "
        "modal run --detach Trainers/cloud/train_modal.py::run_training ..."
    )
