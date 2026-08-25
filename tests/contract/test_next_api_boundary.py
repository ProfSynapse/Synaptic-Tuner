"""Boundary gates for the private next-API staging package."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

import synaptic_tuner
import synaptic_tuner.api.v1 as public_api_v1


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_PACKAGE = "synaptic_tuner._next_api_v1"
STAGING_CONTRACT_TESTS = {
    "tests/contract/test_execution_api_v1.py",
    "tests/contract/test_next_api_boundary.py",
    "tests/contract/test_training_api_v1.py",
    "tests/contract/test_training_schemas_v1.py",
}
JOBSTORE_PRIVATE_TESTS = {
    "tests/jobstore/_fake_provider.py",
    "tests/jobstore/test_jobstore_v1.py",
    "tests/jobstore/test_jobstore_v1_hostile.py",
    "tests/jobstore/test_jobstore_v1_recovery.py",
}
EXPLICIT_IMPLEMENTATION_ALLOWLIST: set[str] = set()
FINAL_CUTOVER_REQUIRES_STAGING_PACKAGE_DELETION = True


def _repository_python_files() -> tuple[Path, ...]:
    commands = (
        ("git", "ls-files", "--", "*.py"),
        ("git", "ls-files", "--others", "--exclude-standard", "--", "*.py"),
    )
    paths: set[Path] = set()
    for command in commands:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        )
        paths.update(
            REPO_ROOT / line
            for line in completed.stdout.splitlines()
            if line.strip()
        )
    return tuple(sorted(paths))


def _is_private_target(target: str) -> bool:
    return target == PRIVATE_PACKAGE or target.startswith(PRIVATE_PACKAGE + ".")


def _source_imports_private_package(source: str, *, filename: str = "<source>") -> bool:
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_private_target(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _is_private_target(node.module):
                return True
            if node.level > 0:
                if node.module and (
                    node.module == "_next_api_v1"
                    or node.module.startswith("_next_api_v1.")
                ):
                    return True
                if node.module is None and any(
                    alias.name == "_next_api_v1"
                    or alias.name.startswith("_next_api_v1.")
                    for alias in node.names
                ):
                    return True
        elif isinstance(node, ast.Call) and node.args:
            target = node.args[0]
            if not isinstance(target, ast.Constant) or not isinstance(target.value, str):
                continue
            is_importlib_call = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            )
            is_builtin_call = isinstance(node.func, ast.Name) and node.func.id == "__import__"
            if (is_importlib_call or is_builtin_call) and _is_private_target(target.value):
                return True
    return False


def _imports_private_package(path: Path) -> bool:
    return _source_imports_private_package(
        path.read_text(encoding="utf-8"), filename=str(path),
    )


@pytest.mark.parametrize(
    "source",
    [
        "import synaptic_tuner._next_api_v1",
        "from synaptic_tuner._next_api_v1 import TrainingRequest",
        "from . import _next_api_v1",
        "from ._next_api_v1 import TrainingRequest",
        "importlib.import_module('synaptic_tuner._next_api_v1')",
        "__import__('synaptic_tuner._next_api_v1.training')",
    ],
)
def test_private_import_scanner_detects_static_relative_and_dynamic_bypasses(
    source: str,
) -> None:
    assert _source_imports_private_package(source)


@pytest.mark.parametrize(
    "source",
    [
        "from . import helpers",
        "from ..support import helper",
        "importlib.import_module('synaptic_tuner.api.v1')",
        "__import__('json')",
    ],
)
def test_private_import_scanner_ignores_benign_relative_and_dynamic_imports(
    source: str,
) -> None:
    assert not _source_imports_private_package(source)


def test_next_api_staging_package_is_not_publicly_exported() -> None:
    assert "_next_api_v1" not in synaptic_tuner.__all__
    assert "_next_api_v1" not in public_api_v1.__all__


def test_private_staging_imports_are_confined_to_explicit_boundaries() -> None:
    private_root = REPO_ROOT / "synaptic_tuner" / "_next_api_v1"
    observed = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in _repository_python_files()
        if not path.is_relative_to(private_root) and _imports_private_package(path)
    }
    allowed = (
        STAGING_CONTRACT_TESTS
        | JOBSTORE_PRIVATE_TESTS
        | EXPLICIT_IMPLEMENTATION_ALLOWLIST
    )
    assert observed <= allowed


def test_staging_boundary_declares_atomic_cutover_deletion() -> None:
    """Staging exists now, but the final cutover must delete it."""
    assert FINAL_CUTOVER_REQUIRES_STAGING_PACKAGE_DELETION is True
    assert (REPO_ROOT / "synaptic_tuner" / "_next_api_v1").is_dir()
