"""Provider-neutral training planning and workload compilation."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CompiledWorkload": ".recipes",
    "RecipeAlreadyRegistered": ".recipes",
    "RecipeNotRegistered": ".recipes",
    "RecipeRegistry": ".recipes",
    "TrainingRecipe": ".recipes",
    "ResolvedTrainingComponents": ".contracts",
    "PreparedTrainingInputIdentity": ".contracts",
    "RetainedTrainingInputStreamLease": ".contracts",
    "VerifiedTrainingInputSource": ".contracts",
    "TrainingRequestResolver": ".contracts",
    "TrainingResolutionError": ".contracts",
    "validate_source_topology": ".resolution",
    "TrainingService": ".service",
    "SFTRecipe": ".methods",
    "DATASET_PREP_NORMALIZER_V1": ".input_preparation",
    "DatasetPrepNormalizerConfigV1": ".input_preparation",
    "DatasetPrepTrainingInputNormalizerV1": ".input_preparation",
    "DatasetPreparedInputFormatVerifierV1": ".input_preparation",
    "NormalizedTrainingInputV1": ".input_preparation",
    "PosixPrivatePreparedRootAuthorityV1": ".input_preparation",
    "PreparedInputFormatVerifierV1": ".input_preparation",
    "PreparedInputVerificationV1": ".input_preparation",
    "PrivatePreparedRootAttestationV1": ".input_preparation",
    "PrivatePreparedRootAuthorityV1": ".input_preparation",
    "TrainingInputNormalizerV1": ".input_preparation",
    "TrainingInputPreparationServiceV1": ".input_preparation",
    "default_dataset_format_verifiers_v1": ".input_preparation",
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
    "DATASET_PREP_NORMALIZER_V1",
    "DatasetPrepNormalizerConfigV1",
    "DatasetPrepTrainingInputNormalizerV1",
    "DatasetPreparedInputFormatVerifierV1",
    "RecipeAlreadyRegistered",
    "RecipeNotRegistered",
    "RecipeRegistry",
    "ResolvedTrainingComponents",
    "NormalizedTrainingInputV1",
    "PosixPrivatePreparedRootAuthorityV1",
    "PreparedInputFormatVerifierV1",
    "PreparedInputVerificationV1",
    "PreparedTrainingInputIdentity",
    "PrivatePreparedRootAttestationV1",
    "PrivatePreparedRootAuthorityV1",
    "RetainedTrainingInputStreamLease",
    "SFTRecipe",
    "TrainingRecipe",
    "TrainingInputNormalizerV1",
    "TrainingInputPreparationServiceV1",
    "TrainingRequestResolver",
    "TrainingResolutionError",
    "TrainingService",
    "VerifiedTrainingInputSource",
    "default_recipe_registry",
    "validate_source_topology",
    "default_dataset_format_verifiers_v1",
]
