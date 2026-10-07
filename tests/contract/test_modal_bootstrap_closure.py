"""Audit the checked-in lock against the candidate Modal bootstrap closure."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.execution.providers.modal.config import ModalRuntimeLockV1


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = "tuner.execution.providers.modal.coordinator_deployment"
MANIFEST = "tuner/runtime/manifests/offline-sft-worker-v1.json"
OWNED_PREFIXES = ("tuner", "synaptic_tuner")
REVIEWED_LAZY_TARGETS = (
    "tuner/training/resolution.py",
    "tuner/training/service.py",
)
DECLARED_ADDITIONS = (*REVIEWED_LAZY_TARGETS, MANIFEST)


def _module_path(module: str) -> Path | None:
    base = ROOT.joinpath(*module.split("."))
    package = base / "__init__.py"
    if package.is_file():
        return package
    source = base.with_suffix(".py")
    return source if source.is_file() else None


def _imports(node: ast.AST, package: str) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()
    if node.level:
        parts = package.split(".")
        base = ".".join(parts[: len(parts) - node.level + 1])
    else:
        base = ""
    module = ".".join(part for part in (base, node.module or "") if part)
    result = [module] if module else []
    # `from . import worker_source` and `from package import submodule` both
    # execute the named submodule when it exists. Attribute imports do not.
    for alias in node.names:
        candidate = ".".join(part for part in (module, alias.name) if part)
        if _module_path(candidate) is not None:
            result.append(candidate)
    return tuple(result)


class _RuntimeImports(ast.NodeVisitor):
    def __init__(self, package: str) -> None:
        self.package = package
        self.modules: list[str] = []

    def visit_If(self, node: ast.If) -> None:
        # With postponed annotations these branches are not runtime imports.
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.modules.extend(_imports(node, self.package))

    visit_ImportFrom = visit_Import


def _bootstrap_paths() -> tuple[str, ...]:
    pending = [ENTRYPOINT]
    seen: set[str] = set()
    paths: set[str] = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        source = _module_path(module)
        if source is None:
            if module == ENTRYPOINT:
                raise AssertionError("candidate Modal coordinator deployment is absent")
            continue
        paths.add(source.relative_to(ROOT).as_posix())

        # Importing a submodule executes every existing package initializer.
        for parent in source.parents:
            if parent == ROOT:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                paths.add(initializer.relative_to(ROOT).as_posix())
                pending.append(".".join(parent.relative_to(ROOT).parts))

        package = module if source.name == "__init__.py" else module.rsplit(".", 1)[0]
        visitor = _RuntimeImports(package)
        visitor.visit(ast.parse(source.read_bytes(), filename=str(source)))
        pending.extend(
            imported for imported in visitor.modules
            if imported.startswith(OWNED_PREFIXES)
        )
    # These inputs are deliberately reviewed declarations, not generic dynamic
    # discovery. TrainingService is resolved through tuner.training.__getattr__;
    # its service module imports resolution. The manifest is package data.
    paths.update(DECLARED_ADDITIONS)
    return tuple(sorted(paths))


def _assert_reviewed_additions(paths: tuple[str, ...]) -> None:
    missing = tuple(path for path in (*REVIEWED_LAZY_TARGETS, MANIFEST) if path not in paths)
    assert not missing, "Modal bootstrap audit omits reviewed additions: " + ", ".join(missing)


def _locked_paths() -> dict[str, str]:
    document = json.loads(
        (ROOT / "tuner/execution/providers/modal/modal-runtime-v1.lock.json")
        .read_text(encoding="utf-8")
    )
    return {name: value["path"] for name, value in document["locked_files"].items()}


def test_modal_runtime_lock_covers_candidate_bootstrap_closure():
    reachable = _bootstrap_paths()
    _assert_reviewed_additions(reachable)
    locked = set(_locked_paths().values())
    missing = tuple(path for path in reachable if path not in locked)
    assert not missing, "Modal runtime lock omits bootstrap paths:\n" + "\n".join(missing)


def test_modal_bootstrap_lock_paths_are_unique_and_drop_legacy_worker_entries():
    locked = _locked_paths()
    assert len(set(locked.values())) == len(locked)
    assert "tuner/execution/providers/modal/remote.py" not in locked.values()
    assert "tuner/execution/providers/modal/producer.py" not in locked.values()


def test_deployment_wrapper_is_candidate_and_selection_is_digest_sensitive():
    lock = ModalRuntimeLockV1.packaged()
    assert lock.document["locked_files"]["deployment_wrapper"]["path"] == (
        "tuner/execution/providers/modal/coordinator_deployment.py"
    )
    selection = SimpleNamespace(
        sdk_version=lock.sdk_version,
        image_digest=lock.image_digest,
        dependency_lock_digest=lock.locked_digest("dependency_lock"),
        wrapper_digest=lock.locked_digest("deployment_wrapper"),
        runtime_digest=lock.locked_digest("sft_runtime"),
        python_version=lock.python_version,
        python_executable=lock.python_executable,
        python_executable_digest=lock.python_executable_digest,
    )
    assert selection.wrapper_digest == lock.locked_digest("deployment_wrapper")
    with pytest.raises(ValueError, match="packaged runtime lock"):
        lock.validate_selection(SimpleNamespace(**{
            **vars(selection), "wrapper_digest": "0" * 64,
        }))


def test_bootstrap_audit_includes_dynamic_worker_inputs_and_resource():
    paths = set(_bootstrap_paths())
    assert {
        "tuner/execution/providers/modal/model_snapshot.py",
        "tuner/execution/providers/modal/worker_source.py",
        MANIFEST,
        *REVIEWED_LAZY_TARGETS,
    } <= paths


def test_dropping_a_reviewed_lazy_target_is_detected(monkeypatch):
    monkeypatch.setattr(
        __import__(__name__, fromlist=["DECLARED_ADDITIONS"]),
        "DECLARED_ADDITIONS", (MANIFEST,),
    )
    with pytest.raises(AssertionError, match="reviewed additions"):
        _assert_reviewed_additions(_bootstrap_paths())
