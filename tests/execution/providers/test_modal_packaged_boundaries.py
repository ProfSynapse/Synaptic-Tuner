"""Static provider-free boundaries for the packaged Modal adapter.

These tests intentionally inspect source rather than import Modal's optional SDK.
They keep the shared packaged-runtime contracts provider-neutral and prevent the
Modal adapter from growing job-time source/bootstrap behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]
SHARED_PACKAGED_CORE = (
    "tuner/runtime/releases.py",
    "tuner/runtime/packaged_training_worker.py",
    "tuner/runtime/packaged_worker_closure.py",
    "tuner/training/input_preparation.py",
    "tuner/training/packaged_boundary.py",
    "tuner/training/packaged_compilation.py",
    "tuner/training/coordinator_material.py",
)
MODAL_PACKAGED_MODULES = (
    "packaged_binding.py",
    "packaged_deployment.py",
    "packaged_staging.py",
    "packaged_dispatch.py",
    "packaged_effects.py",
    "packaged_transport.py",
    "packaged_worker.py",
    "packaged_reader.py",
    "packaged_composition.py",
)


def _imports(relative: str) -> set[str]:
    tree = ast.parse(ROOT.joinpath(relative).read_text(encoding="utf-8"), relative)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _calls(relative: str) -> set[str]:
    tree = ast.parse(ROOT.joinpath(relative).read_text(encoding="utf-8"), relative)
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            calls.add(target.id)
        elif isinstance(target, ast.Attribute):
            parts = [target.attr]
            value = target.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(value.id)
            calls.add(".".join(reversed(parts)))
    return calls


@pytest.mark.parametrize("relative", SHARED_PACKAGED_CORE)
def test_shared_packaged_core_has_no_provider_imports(relative: str) -> None:
    forbidden = (
        "tuner.execution.providers.modal",
        "modal",
        "huggingface_hub",
        "runpod",
        "tuner.execution.providers.docker_provider_v1",
    )
    assert not {
        name for name in _imports(relative)
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    }


def test_modal_packaged_module_inventory_is_exact() -> None:
    package = ROOT / "tuner" / "execution" / "providers" / "modal"
    present = tuple(name for name in MODAL_PACKAGED_MODULES if (package / name).is_file())
    assert present == MODAL_PACKAGED_MODULES


@pytest.mark.parametrize("name", MODAL_PACKAGED_MODULES)
def test_modal_packaged_adapter_has_no_job_time_source_or_install_surface(name: str) -> None:
    relative = f"tuner/execution/providers/modal/{name}"
    imported = _imports(relative)
    assert not imported.intersection({"git", "subprocess", "pip"})
    calls = _calls(relative)
    forbidden_calls = {
        "add_local_python_source",
        "Image.add_local_python_source",
        "subprocess.Popen",
        "subprocess.run",
        "os.system",
        "App.deploy",
    }
    assert not calls.intersection(forbidden_calls)

    source = ROOT.joinpath(relative).read_text(encoding="utf-8")
    tree = ast.parse(source, relative)
    assert not any(
        isinstance(node, ast.keyword)
        and node.arg == "serialized"
        and isinstance(node.value, ast.Constant)
        and node.value is True
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"remote", "deploy"}
        for node in ast.walk(tree)
    )


def test_shared_core_does_not_name_modal_native_objects() -> None:
    forbidden = {
        "Modal", "Volume", "Secret", "Function", "FunctionCall",
        "app_id", "function_id", "call_id", "deployment_generation",
    }
    for relative in SHARED_PACKAGED_CORE:
        source = ROOT.joinpath(relative).read_text(encoding="utf-8")
        tree = ast.parse(source, relative)
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        strings = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert not names.intersection(forbidden), relative
        assert not strings.intersection(forbidden), relative
