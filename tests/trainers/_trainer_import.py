"""Load ``Trainers/<method>/src/<name>.py`` under a unique module name.

Each trainer's ``src`` directory is a flat set of modules that import their
siblings by bare name (``from data_loader import ...``), because the trainer
entrypoints put ``src`` on ``sys.path``. Several trainers share module names
(``data_loader``, ``model_loader``, ``training_callbacks``, ...), so a test that
copies that ``sys.path.insert`` + bare ``import data_loader`` gets whichever
trainer's module the pytest process imported first.

``load_trainer_module("sft", "data_loader")`` instead executes the file as
``trainer_src_sft_data_loader``. While it runs, the bare sibling names resolve
only to the same trainer's ``src`` files (each also loaded under its unique
name), and afterwards the bare entries of ``sys.modules`` are restored exactly
as they were. ``sys.path`` is never modified, so trainer runtime imports are
unchanged and nothing leaks between trainers.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable
from unittest.mock import MagicMock

TRAINERS_ROOT = Path(__file__).resolve().parents[2] / "Trainers"

_PREFIX = "trainer_src"


def trainer_module_name(method: str, name: str) -> str:
    """The unique ``sys.modules`` key of ``Trainers/<method>/src/<name>.py``."""
    return f"{_PREFIX}_{method}_{name}"


def load_trainer_module(
    method: str, name: str, *, stub_if_missing: Iterable[str] = ()
) -> ModuleType:
    """Import ``Trainers/<method>/src/<name>.py`` under its unique module name.

    ``stub_if_missing`` names third-party modules (e.g. ``torch``) that the file
    imports at module level but that are not installed: a ``MagicMock`` stands in
    for each one only while the file executes, and is removed from
    ``sys.modules`` again afterwards. Installed modules are imported for real.
    """
    qualified = trainer_module_name(method, name)
    if qualified in sys.modules:
        return sys.modules[qualified]

    src = TRAINERS_ROOT / method / "src"
    siblings = {path.stem for path in src.glob("*.py")}
    if name not in siblings:
        raise ImportError(f"No trainer module {src / (name + '.py')}")
    stubs = {
        module: MagicMock(name=module)
        for module in stub_if_missing
        if module not in sys.modules and importlib.util.find_spec(module) is None
    }

    saved = {bare: sys.modules.pop(bare) for bare in siblings if bare in sys.modules}
    sys.modules.update(stubs)
    finder = _SiblingFinder(method, src, siblings)
    sys.meta_path.insert(0, finder)
    try:
        return finder.load(name)
    finally:
        sys.meta_path.remove(finder)
        for bare in siblings | set(stubs):
            sys.modules.pop(bare, None)
        sys.modules.update(saved)


class _SiblingFinder(importlib.abc.MetaPathFinder):
    """Resolve bare sibling imports to the same trainer's ``src`` files."""

    def __init__(self, method: str, src: Path, siblings: set[str]):
        self.method = method
        self.src = src
        self.siblings = siblings

    def load(self, name: str) -> ModuleType:
        qualified = trainer_module_name(self.method, name)
        if qualified in sys.modules:
            return sys.modules[qualified]
        spec = importlib.util.spec_from_file_location(qualified, self.src / f"{name}.py")
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"Cannot load trainer module {self.src / name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(qualified, None)
            raise
        return module

    def find_spec(self, fullname, path=None, target=None):
        if path is not None or fullname not in self.siblings:
            return None
        return importlib.util.spec_from_loader(fullname, _LoadedModule(self.load(fullname)))


class _LoadedModule(importlib.abc.Loader):
    """Hand an already-executed sibling module to the import system."""

    def __init__(self, module: ModuleType):
        self.module = module

    def create_module(self, spec):
        return self.module

    def exec_module(self, module):
        return None
