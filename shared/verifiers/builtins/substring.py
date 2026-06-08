"""Substring verifier: check that a needle appears in an extracted haystack.

Registered under the ``substring`` type. The haystack is pulled from the model
completion via :func:`shared.verifiers.extraction.extract` (text modes use
``ExtractedAnswer.answer_text``, falling back to the raw completion when the
extractor finds nothing). The needle comes from a literal ``target`` or from a
field of the sample's ``ground_truth`` named by ``target_field``.

Scoring is binary: ``1.0`` when the needle is contained in the haystack, else
``0.0``. An empty needle scores ``1.0`` (vacuously contained).
"""

from __future__ import annotations

from typing import Mapping

from ..contract import VerifierInput, VerifierOutput
from ..extraction import extract
from ..registry import register


class SubstringVerifier:
    """Verifier that scores by substring containment.

    Args:
        name: Verifier name.
        target: Literal needle string (takes precedence over ``target_field``).
        target_field: ``ground_truth`` key to read the needle from when
            ``target`` is not supplied.
        mode: Extraction mode for resolving the haystack (see
            :func:`shared.verifiers.extraction.extract`).
        output_regex: Optional regex override for haystack extraction.
        case_sensitive: When ``False`` (default), comparison is lowercased.
    """

    def __init__(
        self,
        name: str = "substring",
        target: str | None = None,
        target_field: str | None = None,
        mode: str = "verbatim",
        output_regex: str | None = None,
        case_sensitive: bool = False,
    ):
        self.name = name
        self.target = target
        self.target_field = target_field
        self.mode = mode
        self.output_regex = output_regex
        self.case_sensitive = case_sensitive

    def _resolve_haystack(self, sample: VerifierInput) -> str:
        extracted = extract(
            sample.completion_text,
            mode=self.mode,
            output_regex=self.output_regex,
        )
        if extracted.found and extracted.answer_text:
            return extracted.answer_text
        # Fall back to the raw completion when no text answer was extracted.
        return sample.completion_text

    def _resolve_needle(self, sample: VerifierInput) -> str:
        if self.target is not None:
            return self.target
        if self.target_field is not None:
            value = sample.ground_truth.get(self.target_field)
            return "" if value is None else str(value)
        return ""

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        haystack = self._resolve_haystack(sample)
        needle = self._resolve_needle(sample)

        if needle == "":
            # Vacuously contained.
            return VerifierOutput(
                score=1.0,
                passed=True,
                detail={"needle": needle, "haystack": haystack, "vacuous": True},
            )

        if self.case_sensitive:
            contained = needle in haystack
        else:
            contained = needle.lower() in haystack.lower()

        score = 1.0 if contained else 0.0
        return VerifierOutput(
            score=score,
            passed=score == 1.0,
            detail={"needle": needle, "haystack": haystack, "contained": contained},
        )


@register("substring")
def _build_substring(spec: Mapping) -> SubstringVerifier:
    params = spec.get("params", spec)
    return SubstringVerifier(
        name=spec.get("name", "substring"),
        target=params.get("target"),
        target_field=params.get("target_field"),
        mode=params.get("mode", "verbatim"),
        output_regex=params.get("output_regex"),
        case_sensitive=params.get("case_sensitive", False),
    )
