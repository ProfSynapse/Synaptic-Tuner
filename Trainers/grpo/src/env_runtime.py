"""Cloud runtime helpers for environment-backed GRPO.

The goal is to keep using the Unsloth Docker image as the CUDA/PyTorch base,
while isolating newer TRL/OpenEnv dependencies in a dedicated virtualenv.
"""

from __future__ import annotations

import inspect
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


def _safe_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _shell_quote(value: str) -> str:
    escaped = value.replace("'", "'\"'\"'")
    return f"'{escaped}'"
