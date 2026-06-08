"""CompositeVerifier: combine child verifiers into a single score.

A :class:`CompositeVerifier` runs a list of child verifiers and combines their
:class:`VerifierOutput` scores into one according to a ``combine`` mode. The
five supported modes reproduce — exactly — the three score-combination
semantics that already exist in the codebase:

- ``weighted_sum`` matches GRPO ``combined_reward`` (``rewards.py:706-715``):
  ``Σ weightᵢ · childᵢ.score`` with no normalization by total weight.
- ``max_tier`` matches the Evaluator "highest score among matched paths"
  behavior (``runner.py:714``): the score of the highest-scoring *passing*
  child, or ``0.0`` when none passed.
- ``mean`` / ``min`` / ``all_pass`` match the judge aggregation strategies
  ``mean_score`` / ``min_score`` / ``all_pass`` (see
  ``shared.verifiers.builtins.llm_judge.aggregate``).

Children are supplied as ``(verifier, weight)`` tuples; the weight is only
consulted by ``weighted_sum``.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

from .contract import Verifier, VerifierInput, VerifierOutput

_COMBINE_MODES = ("weighted_sum", "max_tier", "mean", "min", "all_pass")


class CompositeVerifier:
    """Combine several child verifiers into one composite score.

    Args:
        name: Human-readable name for this composite.
        children: List of ``(verifier, weight)`` tuples. The weight is used
            only by the ``weighted_sum`` combine mode.
        combine: Combination mode; one of ``weighted_sum``, ``max_tier``,
            ``mean``, ``min``, ``all_pass``.
        pass_threshold: Threshold used to decide ``passed`` for the
            threshold-based modes (``weighted_sum``, ``mean``, ``min``).
    """

    def __init__(
        self,
        name: str,
        children: Sequence[Tuple[Verifier, float]],
        combine: str,
        pass_threshold: float = 0.5,
    ):
        if combine not in _COMBINE_MODES:
            raise ValueError(
                f"unknown combine mode: {combine!r}; "
                f"must be one of {', '.join(_COMBINE_MODES)}"
            )
        self.name = name
        self.children: List[Tuple[Verifier, float]] = list(children)
        self.combine = combine
        self.pass_threshold = pass_threshold

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        """Run every child and combine their outputs by ``self.combine``."""
        outputs: List[Tuple[Verifier, float, VerifierOutput]] = [
            (verifier, weight, verifier.verify(sample))
            for verifier, weight in self.children
        ]

        child_details = [
            {
                "name": verifier.name,
                "score": out.score,
                "passed": out.passed,
                "weight": weight,
            }
            for verifier, weight, out in outputs
        ]

        score, passed = self._combine(outputs)

        return VerifierOutput(
            score=score,
            passed=passed,
            detail={"children": child_details, "combine": self.combine},
        )

    def _combine(
        self,
        outputs: List[Tuple[Verifier, float, VerifierOutput]],
    ) -> Tuple[float, bool]:
        mode = self.combine

        if mode == "weighted_sum":
            # GRPO combined_reward: Σ weightᵢ · scoreᵢ, no normalization.
            score = sum(weight * out.score for _, weight, out in outputs)
            return score, score >= self.pass_threshold

        if mode == "max_tier":
            # Evaluator: highest score among children whose .passed is True.
            best: float | None = None
            for _, _, out in outputs:
                if out.passed and (best is None or out.score >= best):
                    best = out.score
            if best is None:
                return 0.0, False
            return best, True

        scores = [out.score for _, _, out in outputs]

        if mode == "mean":
            score = sum(scores) / len(scores) if scores else 0.0
            return score, score >= self.pass_threshold

        if mode == "min":
            score = min(scores) if scores else 0.0
            return score, score >= self.pass_threshold

        if mode == "all_pass":
            score = 1.0 if outputs and all(out.passed for _, _, out in outputs) else 0.0
            return score, score == 1.0

        # Unreachable: __init__ validates the mode.
        raise ValueError(f"unknown combine mode: {mode!r}")


def build_composite(
    specs: List[Mapping],
    combine: str,
    pass_threshold: float = 0.5,
    name: str = "composite",
) -> CompositeVerifier:
    """Build a :class:`CompositeVerifier` from child specs.

    Each child verifier is constructed via :func:`build_verifier`; its weight
    is read from ``spec.get("weight", 1.0)``.

    Args:
        specs: List of child verifier spec mappings (each carrying a ``type``).
        combine: Combine mode for the composite.
        pass_threshold: Pass threshold forwarded to the composite.
        name: Name for the composite verifier.

    Returns:
        A constructed :class:`CompositeVerifier`.
    """
    # Imported here to avoid a circular import at module load (registry imports
    # nothing from composite, builtins import registry).
    from .registry import build_verifier

    children: List[Tuple[Verifier, float]] = []
    for spec in specs:
        child = build_verifier(spec)
        weight = float(spec.get("weight", 1.0))
        children.append((child, weight))

    return CompositeVerifier(
        name=name,
        children=children,
        combine=combine,
        pass_threshold=pass_threshold,
    )
