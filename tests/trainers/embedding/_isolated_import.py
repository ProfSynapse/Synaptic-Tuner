"""Isolated importer for ``Trainers/embedding/src`` modules.

Why this exists (the shadow hazard): ``tests/trainers/sft/test_data_loader.py``
does ``sys.path.insert(0, ROOT/Trainers/sft/src)`` then a BARE ``import
data_loader``. ``Trainers/embedding/src/data_loader.py`` and
``Trainers/embedding/src/registry.py`` would collide on those bare top-level
names under combined pytest collection — whichever ``sys.path`` entry lands
first wins ``sys.modules["data_loader"]`` / ``sys.modules["registry"]``, and a
later test in the other trainer silently imports the wrong file.

``Trainers/embedding/src/model_loader.py`` itself does ``from registry import
EmbeddingModelSpec`` (a bare import resolved via the trainer's own
``sys.path.insert``). To load it cleanly for testing WITHOUT polluting the
global ``registry`` name, we load the embedding ``registry`` module under a
NAMESPACED key first, also alias it under the bare ``"registry"`` key *only for
the duration of the model_loader load*, then restore whatever was there before.

The result: each embedding ``src`` module is importable by an explicit file
path, under a unique ``embedding_src.<name>`` module key, with no leakage into
the bare top-level namespace that the sft/kto/grpo trainer tests rely on.
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Trainers/embedding/src — resolved from this file's location (no CWD assumption).
_SRC_DIR = (
    Path(__file__).resolve().parents[3] / "Trainers" / "embedding" / "src"
)


def load_embedding_src(module_basename: str) -> ModuleType:
    """Load ``Trainers/embedding/src/<module_basename>.py`` in isolation.

    The module is registered under ``embedding_src.<module_basename>`` so it
    never shadows (or is shadowed by) a bare top-level name used by another
    trainer's tests.

    ``model_loader`` is special-cased: it executes ``from registry import
    EmbeddingModelSpec`` at import time, so the embedding ``registry`` module is
    temporarily aliased under the bare ``"registry"`` key for the duration of
    the ``model_loader`` exec, then the prior ``sys.modules["registry"]`` (if
    any) is restored.
    """
    file_path = _SRC_DIR / f"{module_basename}.py"
    qualified = f"embedding_src.{module_basename}"

    if qualified in sys.modules:
        return sys.modules[qualified]

    spec = importlib.util.spec_from_file_location(qualified, file_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load embedding src module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module

    with _registry_bare_alias_if_needed(module_basename):
        spec.loader.exec_module(module)

    return module


@contextlib.contextmanager
def _registry_bare_alias_if_needed(module_basename: str):
    """Temporarily expose the embedding ``registry`` under the bare key.

    Only needed when loading ``model_loader`` (which does ``from registry
    import ...``). The previous ``sys.modules["registry"]`` is saved and
    restored so we don't strand a sibling trainer's ``registry`` shadow.
    """
    if module_basename != "model_loader":
        yield
        return

    registry_module = load_embedding_src("registry")
    saved = sys.modules.get("registry")
    sys.modules["registry"] = registry_module
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["registry"] = saved
        else:
            sys.modules.pop("registry", None)
