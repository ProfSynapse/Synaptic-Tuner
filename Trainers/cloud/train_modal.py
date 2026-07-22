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
    *, train_script: str, run_timestamp: str, config_path: str = ""
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
    return command


def commit_success_volumes() -> None:
    """Persist canonical outputs before the replaceable model cache."""
    output_volume.commit()
    model_cache.commit()


def _modal_provenance(
    *,
    repo_branch: str,
    repo_commit: str,
    config_path: str,
    config_sha256: str,
    dataset_path: str,
    dataset_sha256: str,
    inputs_verified: bool,
) -> dict:
    return {
        "schema_version": 1,
        "provider": "modal",
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


@app.function(
    gpu=FUNCTION_GPU,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    volumes=function_volumes,
    # Scope secrets to only the env vars needed for training, rather than
    # exposing the entire .env file via Secret.from_dotenv().
    #
    # UNSLOTH_COMPILE_DISABLE lets older GPU archs (e.g. T4 / sm_75) fall back
    # off the fused cut_cross_entropy Triton kernel, which fails to compile
    # there ("PassManager::run failed"). Forwarded only when set locally; empty
    # by default so it is inert on modern cards.
    #
    # HF_HUB_DISABLE_XET / HF_HUB_ENABLE_HF_TRANSFER: forwarded so a local
    # override reaches the container; run_training also sets them via setdefault
    # so the xet-hang mitigation applies even on a bare `modal run` (see there).
    # An empty string here does NOT override the setdefault (only a real value
    # forwarded from the local env does).
    secrets=[modal.Secret.from_dict({
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "UNSLOTH_COMPILE_DISABLE": os.environ.get("UNSLOTH_COMPILE_DISABLE", ""),
        "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", ""),
        "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
    })],
)
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
    repo_url = _credential_free_repo_url(repo_url)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    modal_run_dir = (
        Path(OUTPUT_MOUNT_PATH)
        / "outputs"
        / "runs"
        / "modal"
        / trainer_type
        / f"{run_timestamp}-{repo_commit[:8].lower()}"
    )
    job_provenance = None
    if config_path:
        job_provenance = _modal_provenance(
            repo_branch=repo_branch,
            repo_commit=repo_commit,
            config_path=config_path,
            config_sha256=config_sha256,
            dataset_path="",
            dataset_sha256=dataset_sha256,
            inputs_verified=False,
        )
        _write_wrapper_state(modal_run_dir, job_provenance, "validating_inputs")
        output_volume.commit()

    # Point HuggingFace cache at the persistent volume to avoid re-downloading models
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

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

    # Clone the repo
    workspace = "/workspace/toolset-training"
    if repo_url:
        print(f"[Modal] Cloning credential-free repo (branch: {repo_branch})")
        clone_result = subprocess.run(
            ["git", "clone", "--branch", repo_branch, "--depth", "1", repo_url, workspace],
            capture_output=True,
            text=True,
        )
        if clone_result.returncode != 0:
            # Scrub any credentials from stderr before logging/raising
            safe_stderr = re.sub(r'https?://[^@\s]+@', 'https://[REDACTED]@', clone_result.stderr)
            error = _redact_text(safe_stderr)
            if job_provenance:
                _write_wrapper_state(modal_run_dir, job_provenance, "failed", error)
                output_volume.commit()
            print(f"[Modal] Git clone failed: {error}")
            raise RuntimeError(f"Failed to clone repo: {error}")
        if repo_commit:
            checkout_result = subprocess.run(
                ["git", "checkout", repo_commit],
                cwd=workspace,
                capture_output=True,
                text=True,
            )
            if checkout_result.returncode != 0:
                error = _redact_text(checkout_result.stderr)
                if job_provenance:
                    _write_wrapper_state(modal_run_dir, job_provenance, "failed", error)
                    output_volume.commit()
                raise RuntimeError(f"Failed to checkout commit {repo_commit}: {error}")
            head_result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=workspace, capture_output=True, text=True
            )
            resolved_head = head_result.stdout.strip().lower()
            if head_result.returncode != 0 or resolved_head != repo_commit.lower():
                error = (
                    f"Exact source verification failed: requested {repo_commit}, "
                    f"got {resolved_head or 'unknown'}."
                )
                if job_provenance:
                    _write_wrapper_state(modal_run_dir, job_provenance, "failed", error)
                    output_volume.commit()
                raise RuntimeError(error)
    else:
        print("[Modal] No repo_url provided, skipping clone.")
        print("[Modal] Ensure training code is available in the container.")
        raise ValueError(
            "repo_url is required. Provide the git URL of your Toolset-Training repo."
        )

    # The cloned repo provides shared/; resolve the path helpers here because
    # the container image does not carry the repo source at module-import time.
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from shared.utilities.paths import (  # noqa: F811 (module-scope import is best-effort)
        get_canonical_output_dir_name,
        get_canonical_trainer_dir_name,
    )

    # Determine trainer directory and script
    trainer_dir = os.path.join(workspace, "Trainers", get_canonical_trainer_dir_name(trainer_type))
    train_script = f"train_{trainer_type}.py"

    if not os.path.isfile(os.path.join(trainer_dir, train_script)):
        raise FileNotFoundError(
            f"Training script not found: {os.path.join(trainer_dir, train_script)}"
        )

    # Build training command with CLI overrides
    cmd = build_training_command(train_script=train_script, run_timestamp=run_timestamp)

    staged_inputs = None
    if config_path:
        try:
            config_file, dataset_file_path, config_sha, dataset_sha = _validate_staged_sft_config(
                config_path, config_sha256, dataset_sha256
            )
        except Exception as exc:
            error = _redact_text(str(exc))
            _write_wrapper_state(modal_run_dir, job_provenance, "failed", error)
            output_volume.commit()
            raise RuntimeError(error) from None
        cmd.extend(["--config", str(config_file)])
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
        )
        os.environ["SYNAPTIC_CLOUD_JOB_PROVENANCE_JSON"] = json.dumps(
            job_provenance, sort_keys=True, separators=(",", ":")
        )
        _write_wrapper_state(modal_run_dir, job_provenance, "running")
        output_volume.commit()

    if model_name:
        # Model name override requires modifying config before training.
        # For now, this is noted as a future enhancement -- the config.yaml
        # in the repo should be pre-configured with the desired model.
        print(f"[Modal] Note: model_name override '{model_name}' requires config.yaml update")

    if dataset_path:
        # Convert relative dataset path to absolute within the workspace
        abs_dataset = os.path.join(workspace, dataset_path)
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

    # Run training
    process = subprocess.run(
        cmd,
        cwd=trainer_dir,
        env={**os.environ},
    )

    if process.returncode != 0:
        print(f"[Modal] Training failed with exit code {process.returncode}")
        if job_provenance:
            _write_wrapper_state(
                modal_run_dir,
                job_provenance,
                "failed",
                f"Training script exited with code {process.returncode}",
            )
        output_volume.commit()
        raise RuntimeError(f"Training script exited with code {process.returncode}")

    print("[Modal] Training completed successfully")

    # Persist result/provenance before opportunistic cache state.
    if job_provenance:
        _write_wrapper_state(modal_run_dir, job_provenance, "completed")
    commit_success_volumes()

    return {
        "status": "completed",
        "trainer_type": trainer_type,
        "repo_commit": repo_commit.lower(),
        "staged_inputs": staged_inputs,
        "artifact_root": f"{OUTPUT_MOUNT_PATH}/outputs/runs/modal/{trainer_type}",
    }


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
