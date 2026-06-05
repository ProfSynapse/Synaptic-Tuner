"""Structure verifier: wrap the shared FitnessEvaluator.

Registered under the ``structure`` type. This builtin adapts
:class:`shared.validation.fitness.FitnessEvaluator` to the verifier contract:
the FitnessEvaluator is constructed once at build time and reused per call.
``FitnessResult.score`` maps to ``VerifierOutput.score``,
``FitnessResult.is_valid`` to ``passed``, and the validation errors are placed
in ``detail``.

The evaluator can be configured from a ``config_path`` YAML or from inline
``validations`` + ``scoring_method`` (+ ``max_errors``), mirroring
:func:`shared.validation.fitness.create_fitness_evaluator`.
"""

from __future__ import annotations

from typing import Mapping

from shared.validation.fitness import FitnessEvaluator, create_fitness_evaluator

from ..contract import VerifierInput, VerifierOutput
from ..registry import register


class StructureVerifier:
    """Verifier backed by a :class:`FitnessEvaluator`.

    Args:
        name: Verifier name.
        evaluator: A constructed :class:`FitnessEvaluator` to reuse per call.
    """

    def __init__(self, name: str, evaluator: FitnessEvaluator):
        self.name = name
        self._evaluator = evaluator

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        result = self._evaluator.evaluate(sample.completion_text)
        return VerifierOutput(
            score=result.score,
            passed=result.is_valid,
            detail={
                "errors": list(result.errors),
                "scoring_method": result.scoring_method,
            },
        )


@register("structure")
def _build_structure(spec: Mapping) -> StructureVerifier:
    params = spec.get("params", spec)
    config_path = params.get("config_path")
    validations = params.get("validations")
    scoring_method = params.get("scoring_method", "error_count")
    max_errors = params.get("max_errors", 5)

    evaluator = create_fitness_evaluator(
        config_path=config_path,
        validations=validations,
        scoring_method=scoring_method,
        max_errors=max_errors,
    )
    return StructureVerifier(name=spec.get("name", "structure"), evaluator=evaluator)
