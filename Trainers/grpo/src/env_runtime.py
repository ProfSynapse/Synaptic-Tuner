"""Cloud runtime helpers for environment-backed GRPO.

The goal is to keep using the Unsloth Docker image as the CUDA/PyTorch base,
while isolating newer TRL/OpenEnv dependencies in a dedicated virtualenv.
"""

from __future__ import annotations

import json
import inspect
import subprocess
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List


def build_cloud_bootstrap_commands(
    config: Dict[str, Any],
    *,
    repo_root: str,
    config_path: str,
) -> List[str]:
    runtime_cfg = ((config.get("env_training") or {}).get("runtime") or {})
    cloud_repo_root = str(runtime_cfg.get("repo_root_in_container") or repo_root)
    venv_dir = str(runtime_cfg.get("isolated_venv_dir") or "/workspace/.venvs/grpo-openenv")
    python_packages = list(runtime_cfg.get("python_packages") or [])
    project_pip_deps = list(runtime_cfg.get("project_pip_deps") or [])

    pip_packages = ["pip", "setuptools", "wheel", *project_pip_deps, *python_packages]
    install_args = " ".join(_shell_quote(part) for part in pip_packages)

    return [
        "mkdir -p /tmp/grpo-openenv-bootstrap",
        "python -m pip install --upgrade --target /tmp/grpo-openenv-bootstrap virtualenv",
        "export PYTHONPATH=/tmp/grpo-openenv-bootstrap:$PYTHONPATH",
        f"python -m virtualenv --no-download {_shell_quote(venv_dir)}",
        f". {_shell_quote(venv_dir)}/bin/activate",
        f"python -m pip install --upgrade {install_args}",
        f"cd {_shell_quote(str(Path(cloud_repo_root) / 'Trainers' / 'grpo'))}",
        f"python train_env_grpo.py --config {_shell_quote(config_path)}",
    ]


def detect_openenv_runtime_support() -> Dict[str, Any]:
    """Inspect the active Python env for the APIs required by env-GRPO."""
    support: Dict[str, Any] = {
        "trl_version": _safe_version("trl"),
        "transformers_version": _safe_version("transformers"),
        "openenv_version": _safe_version("openenv"),
        "has_rollout_func": False,
        "has_environment_factory": False,
        "has_generate_rollout_completions": False,
        "errors": [],
    }

    try:
        trl_mod = import_module("trl")
        trainer_cls = getattr(trl_mod, "GRPOTrainer", None)
        if trainer_cls is not None:
            signature = inspect.signature(trainer_cls.__init__)
            support["has_rollout_func"] = "rollout_func" in signature.parameters
            support["has_environment_factory"] = "environment_factory" in signature.parameters
    except Exception as exc:
        support["errors"].append(f"trl import failed: {exc}")
        return support

    for module_path in (
        "trl.experimental.openenv",
        "trl.experimental.open_env",
        "trl.extras.openenv",
    ):
        try:
            module = import_module(module_path)
        except Exception:
            continue
        if hasattr(module, "generate_rollout_completions"):
            support["has_generate_rollout_completions"] = True
            break

    return support


def ensure_local_openenv_runtime(
    config: Dict[str, Any],
    *,
    repo_root: str,
    bootstrap_python: str,
) -> str:
    """Ensure the local NVIDIA env-GRPO runtime exists and is usable.

    Creates an isolated virtualenv if needed, installs the configured runtime
    packages when support is missing, then returns the Python executable that
    should be used to launch `train_env_grpo.py`.
    """
    runtime_cfg = ((config.get("env_training") or {}).get("runtime") or {})
    venv_dir = _resolve_local_venv_dir(runtime_cfg, repo_root=repo_root)
    venv_python = _venv_python(venv_dir)

    if not venv_python.exists():
        subprocess.run([bootstrap_python, "-m", "venv", str(venv_dir)], check=True)

    support = _probe_runtime_support(str(venv_python))
    if not _runtime_support_sufficient(support):
        pip_packages = [
            "pip",
            "setuptools",
            "wheel",
            *list(runtime_cfg.get("project_pip_deps") or []),
            *list(runtime_cfg.get("python_packages") or []),
        ]
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", *pip_packages],
            check=True,
        )
        support = _probe_runtime_support(str(venv_python))

    if not _runtime_support_sufficient(support):
        raise RuntimeError(
            "Local env-GRPO runtime is missing required support: "
            f"{json.dumps(support, sort_keys=True)}"
        )

    return str(venv_python)


def _safe_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _shell_quote(value: str) -> str:
    escaped = value.replace("'", "'\"'\"'")
    return f"'{escaped}'"


def _resolve_local_venv_dir(runtime_cfg: Dict[str, Any], *, repo_root: str) -> Path:
    configured = str(runtime_cfg.get("local_venv_dir") or ".venvs/grpo-openenv")
    path = Path(configured)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _probe_runtime_support(python_executable: str) -> Dict[str, Any]:
    probe_script = """
import json
import inspect
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

def safe_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None

support = {
    "trl_version": safe_version("trl"),
    "transformers_version": safe_version("transformers"),
    "openenv_version": safe_version("openenv"),
    "has_rollout_func": False,
    "has_environment_factory": False,
    "has_generate_rollout_completions": False,
    "errors": [],
}

try:
    trl_mod = import_module("trl")
    trainer_cls = getattr(trl_mod, "GRPOTrainer", None)
    if trainer_cls is not None:
        signature = inspect.signature(trainer_cls.__init__)
        support["has_rollout_func"] = "rollout_func" in signature.parameters
        support["has_environment_factory"] = "environment_factory" in signature.parameters
except Exception as exc:
    support["errors"].append(f"trl import failed: {exc}")

for module_path in (
    "trl.experimental.openenv",
    "trl.experimental.open_env",
    "trl.extras.openenv",
):
    try:
        module = import_module(module_path)
    except Exception:
        continue
    if hasattr(module, "generate_rollout_completions"):
        support["has_generate_rollout_completions"] = True
        break

print(json.dumps(support))
""".strip()
    result = subprocess.run(
        [python_executable, "-c", probe_script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "trl_version": None,
            "transformers_version": None,
            "openenv_version": None,
            "has_rollout_func": False,
            "has_environment_factory": False,
            "has_generate_rollout_completions": False,
            "errors": [result.stderr.strip() or result.stdout.strip() or "probe failed"],
        }
    return json.loads(result.stdout)


def _runtime_support_sufficient(support: Dict[str, Any]) -> bool:
    return bool(support.get("has_rollout_func"))
