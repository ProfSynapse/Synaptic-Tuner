"""tests/inference/test_inference_evaluator_imports.py

AST gate for the slice 7 relocation (docs/plans/api-facade-slices.md, "Slice 7"):
``chat_session``, ``vllm_runtime`` and ``owned_process`` live under
``tuner/inference/`` and no module in that package may reach them through their
former ``Evaluator`` paths. The residual coupling of ``tuner/inference/`` to
``Evaluator/`` is pinned exactly so it can only shrink: ``chat_session.py`` keeps
``Evaluator.protocols`` because ``ChatSession`` checks ``BackendResponse`` by
class identity against what the Evaluator backend clients return, and
``model_chat.py`` keeps the local vLLM client arm that slice 8 revisits. Any new
``Evaluator`` import under ``tuner/inference/`` fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tuner" / "inference"
MOVED_MODULES = ("chat_session", "vllm_runtime", "owned_process")
FORBIDDEN = frozenset(f"Evaluator.{name}" for name in MOVED_MODULES)
ALLOWED_EVALUATOR_IMPORTS: dict[str, frozenset[str]] = {
    "chat_session.py": frozenset({"Evaluator.protocols"}),
    "model_chat.py": frozenset({"Evaluator.config", "Evaluator.vllm_client"}),
}


def _evaluator_imports(source: str, filename: str) -> set[str]:
    """Return every ``Evaluator`` module the source imports, in dotted form.

    ``import Evaluator.x``, ``from Evaluator.x import y`` and ``from Evaluator
    import x`` all record ``Evaluator.x``; a bare ``import Evaluator`` records
    ``Evaluator``. Relative imports cannot name ``Evaluator`` from this package.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "Evaluator" or alias.name.startswith("Evaluator."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if node.module == "Evaluator":
                found.update(f"Evaluator.{alias.name}" for alias in node.names)
            elif node.module.startswith("Evaluator."):
                found.add(node.module)
    return found


def _package_sources() -> dict[str, set[str]]:
    sources = sorted(PACKAGE.rglob("*.py"))
    assert sources, "tuner/inference/ has no modules"
    return {
        path.relative_to(PACKAGE).as_posix(): _evaluator_imports(
            path.read_text(encoding="utf-8"), str(path)
        )
        for path in sources
    }


def test_moved_modules_live_under_tuner_inference_only() -> None:
    for name in MOVED_MODULES:
        assert (PACKAGE / f"{name}.py").is_file(), name
        assert not (ROOT / "Evaluator" / f"{name}.py").exists(), name


def test_no_inference_module_imports_moved_modules_via_evaluator() -> None:
    offenders = {
        name: sorted(imports & FORBIDDEN)
        for name, imports in _package_sources().items()
        if imports & FORBIDDEN
    }
    assert offenders == {}


def test_residual_evaluator_coupling_is_pinned_exactly() -> None:
    actual = {
        name: frozenset(imports)
        for name, imports in _package_sources().items()
        if imports
    }
    assert actual == ALLOWED_EVALUATOR_IMPORTS


def test_detector_records_every_import_form() -> None:
    # The probe is assembled from MOVED_MODULES rather than written out, so the
    # tree-wide grep for the former Evaluator paths stays empty.
    chat, runtime, owned = MOVED_MODULES
    source = (
        f"import Evaluator.{owned} as owned\n"
        f"from Evaluator.{runtime} import VLLMRuntimeLease\n"
        f"from Evaluator import {chat} as chat_session_module\n"
        "from Evaluator import base_client, vllm_setup\n"
        "import Evaluator\n"
        "from .serving_target import ServingTarget\n"
        f"from tuner.inference.{chat} import ChatSession\n"
        "def later():\n"
        "    from Evaluator.protocols import BackendClient\n"
    )
    assert _evaluator_imports(source, "<probe>") == FORBIDDEN | {
        "Evaluator.base_client",
        "Evaluator.vllm_setup",
        "Evaluator",
        "Evaluator.protocols",
    }
