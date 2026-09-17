"""Completion return types and usage accounting for ``shared.llm`` clients.

Location: ``shared/llm/usage.py``.

``BaseLLMClient.chat`` returns ``LLMCompletionV1`` and ``structured_output``
returns ``LLMStructuredV1``. Both carry the provider's answer and, when the
provider reported prompt and completion token counts, a ``measured``
``UsageRecordV1``; when the provider reported nothing, ``usage`` is ``None``
(the ``unavailable`` case has exactly one representation, ``None``). No
adapter estimates tokens and none invents a cost: ``cost_minor_units`` is
filled only when a provider returns an exact integer minor-unit amount, which
none of the five bundled adapters does today.

``UsageAccumulator`` folds the usage of every call in one run into a single
record: the sum of the measured counters, or an ``indeterminate`` verdict when
any call returned no usage. The reference Evaluation and Data implementations
use it to publish one ``usage`` per run and to close one ``spend`` effect.

``shared/llm/metering.py`` wraps a ``BaseLLMClient`` so every call feeds an
accumulator; the reference Data implementation uses it.

This module is implementation, not part of the public ``synaptic_tuner.api.v1``
closure, so importing the usage contract from there is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from synaptic_tuner.api.v1.usage import USAGE_SCHEMA_VERSION, SpendRef, UsageAvailability, UsageRecordV1



def _exact_usage(value: object) -> UsageRecordV1 | None:
    if value is None:
        return None
    if type(value) is not UsageRecordV1:
        raise TypeError("usage must be exact UsageRecordV1 or None")
    if value.availability is not UsageAvailability.MEASURED:
        raise ValueError("unavailable usage is represented by None, not a record")
    return value


@dataclass(frozen=True, slots=True)
class LLMCompletionV1:
    """One chat completion: the text and the provider-reported usage, if any."""

    text: str
    usage: UsageRecordV1 | None = None

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("text must be an exact string")
        object.__setattr__(self, "usage", _exact_usage(self.usage))


@dataclass(frozen=True, slots=True)
class LLMStructuredV1:
    """One structured completion: the parsed JSON object and the usage, if any."""

    value: Dict[str, Any]
    usage: UsageRecordV1 | None = None

    def __post_init__(self) -> None:
        if type(self.value) is not dict:
            raise TypeError("value must be an exact dict")
        object.__setattr__(self, "usage", _exact_usage(self.usage))


def _count(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def measured_usage(
    input_tokens: int,
    output_tokens: int,
    *,
    cost_minor_units: int | None = None,
    currency: str | None = None,
    spend: SpendRef | None = None,
) -> UsageRecordV1:
    """A ``measured`` record from exact token counts."""
    return UsageRecordV1(
        USAGE_SCHEMA_VERSION, UsageAvailability.MEASURED, input_tokens, output_tokens,
        cost_minor_units, currency, spend,
    )


def usage_from_counts(input_tokens: object, output_tokens: object) -> UsageRecordV1 | None:
    """``measured`` usage when both counts are non-negative integers, else ``None``.

    Providers report the counts under different names (``prompt_tokens`` /
    ``completion_tokens``, ``input_tokens`` / ``output_tokens``,
    ``prompt_eval_count`` / ``eval_count``); adapters pick the pair and pass the
    raw values here. Floats, booleans, negative and missing values all mean
    the provider did not measure, so no record is fabricated.
    """
    prompt = _count(input_tokens)
    completion = _count(output_tokens)
    if prompt is None or completion is None:
        return None
    return measured_usage(prompt, completion)


def usage_from_openai_block(usage: object) -> UsageRecordV1 | None:
    """Usage from an OpenAI-style ``usage`` object (chat completions or Responses)."""
    if not isinstance(usage, dict):
        return None
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return usage_from_counts(usage.get("prompt_tokens"), usage.get("completion_tokens"))
    return usage_from_counts(usage.get("input_tokens"), usage.get("output_tokens"))


class UsageAccumulator:
    """Folds per-call usage into one per-run record.

    ``record(spend)`` returns the aggregated ``measured`` record once at least
    one call was measured and no call went unmeasured; ``None`` when no call
    was made. A run with any unmeasured call is ``indeterminate``: its total is
    unknown and ``record`` refuses to publish a partial sum. Cost is summed
    only when every measured call priced itself in one currency; otherwise the
    total carries tokens alone.
    """

    __slots__ = ("calls", "unmeasured", "input_tokens", "output_tokens", "_cost", "_currency", "_cost_complete")

    def __init__(self) -> None:
        self.calls = 0
        self.unmeasured = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._cost = 0
        self._currency: str | None = None
        self._cost_complete = True

    def add(self, usage: object) -> None:
        self.calls += 1
        if usage is None:
            self.unmeasured += 1
            return
        if type(usage) is not UsageRecordV1 or usage.availability is not UsageAvailability.MEASURED:
            self.unmeasured += 1
            return
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        if usage.cost_minor_units is None or usage.currency is None:
            self._cost_complete = False
        elif self._currency is None:
            self._currency = usage.currency
            self._cost += usage.cost_minor_units
        elif usage.currency != self._currency:
            self._cost_complete = False
        else:
            self._cost += usage.cost_minor_units

    @property
    def indeterminate(self) -> bool:
        return self.unmeasured > 0

    @property
    def measured(self) -> bool:
        return self.calls > 0 and self.unmeasured == 0

    def record(self, spend: SpendRef | None = None) -> UsageRecordV1 | None:
        if self.indeterminate:
            raise ValueError("usage is indeterminate: a call reported no usage")
        if self.calls == 0:
            return None
        priced = self._cost_complete and self._currency is not None
        return measured_usage(
            self.input_tokens, self.output_tokens,
            cost_minor_units=self._cost if priced else None,
            currency=self._currency if priced else None,
            spend=spend,
        )


__all__ = [
    "LLMCompletionV1",
    "LLMStructuredV1",
    "UsageAccumulator",
    "measured_usage",
    "usage_from_counts",
    "usage_from_openai_block",
]
