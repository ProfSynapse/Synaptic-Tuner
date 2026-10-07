"""Import-light facade for HF Jobs experiment stage runners."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "HFEvalStageRunner": ".hf_eval_stage",
    "HFLossStageRunner": ".hf_loss_stage",
    "HFTrainingStageRunner": ".hf_training_stage",
}

__all__ = ["HFEvalStageRunner", "HFLossStageRunner", "HFTrainingStageRunner"]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))
