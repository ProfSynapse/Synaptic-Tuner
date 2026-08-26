"""Provider-neutral training planning and workload compilation."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CompiledWorkload": ".recipes",
    "RecipeAlreadyRegistered": ".recipes",
    "RecipeNotRegistered": ".recipes",
    "RecipeRegistry": ".recipes",
    "TrainingRecipe": ".recipes",
    "ResolvedTrainingComponents": ".resolution",
    "TrainingRequestResolver": ".resolution",
    "TrainingResolutionError": ".resolution",
    "validate_source_topology": ".resolution",
    "TrainingService": ".service",
    "SFTRecipe": ".methods",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def default_recipe_registry() -> RecipeRegistry:
    from .methods import SFTRecipe
    from .recipes import RecipeRegistry

    registry = RecipeRegistry()
    registry.register(SFTRecipe())
    return registry


__all__ = [
    "CompiledWorkload",
    "RecipeAlreadyRegistered",
    "RecipeNotRegistered",
    "RecipeRegistry",
    "ResolvedTrainingComponents",
    "SFTRecipe",
    "TrainingRecipe",
    "TrainingRequestResolver",
    "TrainingResolutionError",
    "TrainingService",
    "default_recipe_registry",
    "validate_source_topology",
]
