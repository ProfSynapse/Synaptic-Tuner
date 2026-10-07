"""The editable ``synaptic`` console must work outside the source checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
import venv
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _declared_console_target() -> str:
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("Node B dependency: pyproject.toml is not present yet")
    document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    target = document.get("project", {}).get("scripts", {}).get("synaptic")
    if not isinstance(target, str) or not target.strip():
        pytest.skip("Node B dependency: the synaptic console script is not declared yet")
    return target


def _stage_editable_source(destination: Path) -> None:
    shutil.copy2(REPO_ROOT / "pyproject.toml", destination / "pyproject.toml")
    for filename in ("README.md", "LICENSE", "tuner.py"):
        source = REPO_ROOT / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)
    for package_name in ("tuner", "synaptic_tuner", "shared"):
        source = REPO_ROOT / package_name
        if source.is_dir():
            shutil.copytree(
                source,
                destination / package_name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )


def _venv_python(environment_root: Path) -> Path:
    if os.name == "nt":
        return environment_root / "Scripts" / "python.exe"
    return environment_root / "bin" / "python"


def _venv_console(environment_root: Path) -> Path:
    if os.name == "nt":
        return environment_root / "Scripts" / "synaptic.exe"
    return environment_root / "bin" / "synaptic"


def test_editable_console_runs_from_unrelated_working_directory(
    tmp_path: Path,
) -> None:
    target = _declared_console_target()
    staged_source = tmp_path / "editable source with spaces"
    staged_source.mkdir()
    _stage_editable_source(staged_source)
    environment_root = tmp_path / "editable environment"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment_root)

    install = subprocess.run(
        [
            str(_venv_python(environment_root)),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-build-isolation",
            "--editable",
            str(staged_source),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    unrelated_cwd = tmp_path / "unrelated working directory"
    unrelated_cwd.mkdir()
    command = _venv_console(environment_root)
    assert command.is_file(), f"{target!r} did not produce {command}"
    result = subprocess.run(
        [str(command), "--help"],
        cwd=unrelated_cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={key: value for key, value in os.environ.items() if key != "SYNAPTIC_PROJECT_ROOT"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
    assert str(REPO_ROOT) not in result.stderr
