"""Generic, declarative, config-driven rollout filtering engine.

Location: ``shared/validation/rollout_filters.py``

Summary:
    A standalone, dependency-light engine for filtering arbitrary record dicts
    on ANY field via a declarative list of filter specs. It knows nothing about
    "turns", "tools", or any specific rollout shape — every field is addressed
    by a dot-path and every comparison goes through a generic operator table.

How it is used:
    The SynthChat rollout -> training-set projector
    (``SynthChat/scripts/project_rollout_datasets.py``) builds a
    :class:`RolloutFilterSet` from a ``projection.filters`` config list and
    applies it per projection target (``sft``, ``grpo``, ``kto_positive``,
    ``kto_negative``). The engine is intentionally generic so the flywheel
    stager (``shared/flywheel/...``) and any future projection path can import
    and reuse it without modification.

Config (YAML) example::

    projection:
      filters:
        # Drop short rollouts from the positive/GRPO targets (the default set).
        - field: environment.episode_trace.total_turns
          op: gte
          value: 5
          on_missing: keep          # records lacking the field pass through

        # Only keep a specific scenario family on the SFT target.
        - field: metadata.scenario_family
          op: in
          value: ["tool_calling", "agentic_search"]
          applies_to: ["sft"]

        # Explicitly filter hard negatives too (rare; opt-in).
        - field: environment.episode_trace.stop_reason
          op: ne
          value: "max_tool_steps_exceeded"
          applies_to: ["kto_negative"]
          on_missing: drop

Semantics:
    * A record PASSES a ``target`` iff ALL filters that apply to that target
      match (logical AND). A filter "applies" to ``target`` when its
      ``applies_to`` list includes ``target``, or — if ``applies_to`` is
      omitted — when ``target`` is in the caller-supplied default target set.
    * An empty filter set always passes (no-op).
    * ``on_missing`` (``keep``|``drop``, default ``keep``) decides the verdict
      when the addressed field resolves to :data:`MISSING`.

Fail-closed posture:
    Spec validation is eager. Unknown ``op``, bad ``on_missing``, or missing
    required keys raise :class:`ValueError` at construction time rather than
    being silently ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


class _Missing:
    """Sentinel type for an absent dot-path value.

    A dedicated singleton (rather than ``None``) so that a field whose real
    stored value is ``None`` is distinguishable from a field that is absent.
    """

    _instance: Optional["_Missing"] = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "MISSING"

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return False


MISSING = _Missing()
"""Module-level sentinel returned by :func:`get_path` when a field is absent."""


def get_path(record: Dict[str, Any], path: str) -> Any:
    """Resolve a dot-path like ``"a.b.c"`` into nested dicts.

    Mirrors the nested-get style already used in the projector
    (e.g. ``(environment.get("episode_trace") or {}).get("total_tool_calls")``)
    but generalized to an arbitrary depth.

    Args:
        record: The record dict to traverse.
        path: Dot-separated key path. Each segment indexes one dict level.

    Returns:
        The resolved value, or :data:`MISSING` if any segment is absent or a
        non-dict is encountered before the path is exhausted.
    """
    if not isinstance(path, str) or not path:
        return MISSING
    current: Any = record
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return MISSING
        current = current[segment]
    return current


# ---------------------------------------------------------------------------
# Operator table
# ---------------------------------------------------------------------------

def _safe_compare(op: Callable[[Any, Any], bool], left: Any, right: Any) -> bool:
    """Run a comparison operator, treating incomparable operands as non-match.

    Numeric/ordered comparisons (gt/gte/lt/lte) raise ``TypeError`` when the
    operands are not mutually comparable (e.g. ``"a" > 5``). Per spec these
    fail safe to ``False`` (non-match) rather than raising.
    """
    try:
        return bool(op(left, right))
    except TypeError:
        return False


def _op_in(value: Any, expected: Any) -> bool:
    if not isinstance(expected, (list, tuple, set)):
        return False
    return value in expected


def _op_not_in(value: Any, expected: Any) -> bool:
    if not isinstance(expected, (list, tuple, set)):
        return False
    return value not in expected


# Each entry maps an op name to a callable (field_value, spec_value) -> bool.
# ``field_value`` is the resolved record value (never MISSING here — MISSING is
# handled upstream by on_missing). ``exists``/``missing`` are handled specially
# in RolloutFilter.matches because they reason about presence, not value.
_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: _safe_compare(lambda x, y: x > y, a, b),
    "gte": lambda a, b: _safe_compare(lambda x, y: x >= y, a, b),
    "lt": lambda a, b: _safe_compare(lambda x, y: x < y, a, b),
    "lte": lambda a, b: _safe_compare(lambda x, y: x <= y, a, b),
    "in": _op_in,
    "not_in": _op_not_in,
}

# Operators that ignore ``value`` and reason purely about field presence.
_PRESENCE_OPERATORS = frozenset({"exists", "missing"})

# Operators that require ``value`` to be a list.
_LIST_VALUE_OPERATORS = frozenset({"in", "not_in"})

VALID_OPERATORS = frozenset(_OPERATORS) | _PRESENCE_OPERATORS
"""All operator names accepted in a filter spec."""

VALID_ON_MISSING = frozenset({"keep", "drop"})
"""Accepted values for the ``on_missing`` key."""

_DEFAULT_ON_MISSING = "keep"


# ---------------------------------------------------------------------------
# Filter spec + evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterDecision:
    """Outcome of evaluating a :class:`RolloutFilterSet` against one record.

    Attributes:
        passed: True if the record passes ALL applicable filters for the target.
        dropped_by: The filter that caused the drop (None when ``passed``).
        dropped_on_missing: True when the drop was caused by ``on_missing: drop``
            (the field resolved to MISSING) rather than by a predicate mismatch.
    """

    passed: bool
    dropped_by: Optional["RolloutFilter"] = None
    dropped_on_missing: bool = False


class RolloutFilter:
    """One parsed, validated filter spec.

    Construct via :meth:`from_spec` (which validates eagerly) rather than the
    constructor directly when consuming raw config.
    """

    __slots__ = ("field", "op", "value", "applies_to", "on_missing", "_index")

    def __init__(
        self,
        field: str,
        op: str,
        value: Any = None,
        applies_to: Optional[Sequence[str]] = None,
        on_missing: str = _DEFAULT_ON_MISSING,
        index: int = 0,
    ) -> None:
        self.field = field
        self.op = op
        self.value = value
        self.applies_to: Optional[List[str]] = (
            list(applies_to) if applies_to is not None else None
        )
        self.on_missing = on_missing
        self._index = index

    @classmethod
    def from_spec(cls, spec: Dict[str, Any], index: int = 0) -> "RolloutFilter":
        """Validate a raw spec dict and build a :class:`RolloutFilter`.

        Raises:
            ValueError: on any structural problem (unknown op, bad on_missing,
                missing required key, wrong value type for list operators).
        """
        prefix = f"filter[{index}]"
        if not isinstance(spec, dict):
            raise ValueError(f"{prefix}: filter spec must be a mapping, got {type(spec).__name__}")

        field_name = spec.get("field")
        if not isinstance(field_name, str) or not field_name:
            raise ValueError(f"{prefix}: 'field' is required and must be a non-empty string")

        op = spec.get("op")
        if not isinstance(op, str) or op not in VALID_OPERATORS:
            raise ValueError(
                f"{prefix}: 'op' must be one of {sorted(VALID_OPERATORS)}, got {op!r}"
            )

        is_presence = op in _PRESENCE_OPERATORS
        has_value = "value" in spec
        if not is_presence and not has_value:
            raise ValueError(f"{prefix}: 'value' is required for op {op!r}")

        value = spec.get("value")
        if op in _LIST_VALUE_OPERATORS and not isinstance(value, (list, tuple)):
            raise ValueError(f"{prefix}: op {op!r} requires 'value' to be a list")

        applies_to = spec.get("applies_to")
        if applies_to is not None:
            if not isinstance(applies_to, (list, tuple)) or not all(
                isinstance(item, str) for item in applies_to
            ):
                raise ValueError(f"{prefix}: 'applies_to' must be a list of strings")

        on_missing = spec.get("on_missing", _DEFAULT_ON_MISSING)
        if on_missing not in VALID_ON_MISSING:
            raise ValueError(
                f"{prefix}: 'on_missing' must be one of {sorted(VALID_ON_MISSING)}, "
                f"got {on_missing!r}"
            )

        # Reject unknown keys to fail closed on typos (e.g. 'applies' / 'op_').
        known_keys = {"field", "op", "value", "applies_to", "on_missing"}
        unknown = set(spec) - known_keys
        if unknown:
            raise ValueError(f"{prefix}: unknown key(s) {sorted(unknown)}")

        return cls(
            field=field_name,
            op=op,
            value=value,
            applies_to=applies_to,
            on_missing=on_missing,
            index=index,
        )

    def applies(self, target: str, default_targets: Sequence[str]) -> bool:
        """Whether this filter participates in evaluating ``target``."""
        if self.applies_to is not None:
            return target in self.applies_to
        return target in default_targets

    def matches(self, record: Dict[str, Any]) -> bool:
        """Whether ``record`` matches this filter, honoring ``on_missing``.

        Returns True for a MATCH (record is kept by this filter). On a MISSING
        field, the verdict follows ``on_missing``: ``keep`` -> True (match),
        ``drop`` -> False (non-match). Presence operators (``exists``/
        ``missing``) reason about presence directly and ignore ``on_missing``.
        """
        resolved = get_path(record, self.field)

        if self.op == "exists":
            return resolved is not MISSING
        if self.op == "missing":
            return resolved is MISSING

        if resolved is MISSING:
            return self.on_missing == "keep"

        operator = _OPERATORS[self.op]
        return operator(resolved, self.value)

    def describe(self) -> Dict[str, Any]:
        """Compact, serializable description for auditing/logging."""
        desc: Dict[str, Any] = {"field": self.field, "op": self.op}
        if self.op not in _PRESENCE_OPERATORS:
            desc["value"] = self.value
        if self.applies_to is not None:
            desc["applies_to"] = list(self.applies_to)
        desc["on_missing"] = self.on_missing
        return desc

    def key(self) -> str:
        """Stable per-filter key for stats tallies (field+op+value+index)."""
        return f"{self._index}:{self.field}:{self.op}:{self.value!r}"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"RolloutFilter({self.describe()!r})"


class RolloutFilterSet:
    """An ordered set of filters with per-target AND evaluation.

    Args:
        filters: Raw list of spec dicts (validated eagerly).
        default_targets: Targets a filter applies to when it omits
            ``applies_to``. The caller decides this set; the projector passes
            the positive/GRPO targets only (excluding hard negatives).
    """

    def __init__(
        self,
        filters: Optional[Sequence[Dict[str, Any]]] = None,
        default_targets: Optional[Sequence[str]] = None,
    ) -> None:
        raw = filters or []
        if not isinstance(raw, (list, tuple)):
            raise ValueError("filters config must be a list of filter specs")
        self.filters: List[RolloutFilter] = [
            RolloutFilter.from_spec(spec, index=i) for i, spec in enumerate(raw)
        ]
        self.default_targets: List[str] = list(default_targets or [])

    @property
    def is_empty(self) -> bool:
        """True when there are no filters (apply is a no-op)."""
        return not self.filters

    def filters_for(self, target: str) -> List[RolloutFilter]:
        """Filters that participate in evaluating ``target``."""
        return [f for f in self.filters if f.applies(target, self.default_targets)]

    def apply(self, record: Dict[str, Any], target: str) -> FilterDecision:
        """Evaluate ``record`` for ``target``.

        A record passes iff every applicable filter matches (AND). The first
        non-matching filter is reported as the drop cause.
        """
        for filt in self.filters:
            if not filt.applies(target, self.default_targets):
                continue
            resolved_missing = get_path(record, filt.field) is MISSING
            if not filt.matches(record):
                return FilterDecision(
                    passed=False,
                    dropped_by=filt,
                    dropped_on_missing=resolved_missing and filt.op not in _PRESENCE_OPERATORS,
                )
        return FilterDecision(passed=True)


# ---------------------------------------------------------------------------
# Stats accumulator
# ---------------------------------------------------------------------------

@dataclass
class _FilterTargetTally:
    passed: int = 0
    dropped_by_predicate: int = 0
    dropped_on_missing: int = 0


class FilterStats:
    """Tally pass/drop counts per (target, filter) for breakdown logging.

    Usage::

        stats = FilterStats()
        decision = filter_set.apply(record, "grpo")
        stats.record("grpo", filter_set, decision)
        ...
        breakdown = stats.as_dict()  # serializable summary
    """

    def __init__(self) -> None:
        # (target -> filter_key -> tally) plus a per-target total.
        self._per_filter: Dict[str, Dict[str, _FilterTargetTally]] = {}
        self._target_totals: Dict[str, _FilterTargetTally] = {}
        # Stable mapping of filter_key -> human description, for output.
        self._filter_desc: Dict[str, Dict[str, Any]] = {}

    def _tally(self, target: str, filter_key: str) -> _FilterTargetTally:
        per_target = self._per_filter.setdefault(target, {})
        return per_target.setdefault(filter_key, _FilterTargetTally())

    def _target_total(self, target: str) -> _FilterTargetTally:
        return self._target_totals.setdefault(target, _FilterTargetTally())

    def record(
        self,
        target: str,
        filter_set: RolloutFilterSet,
        decision: FilterDecision,
    ) -> None:
        """Record one record's decision for ``target``.

        Per-target totals count whether the record ultimately passed or which
        kind of drop occurred. Per-filter counts attribute the drop to the
        specific filter that caused it; passing records credit a ``passed`` to
        every filter that applied to the target (so each filter's pass count
        reflects records it saw and let through).
        """
        total = self._target_total(target)
        applicable = filter_set.filters_for(target)
        for filt in applicable:
            self._filter_desc.setdefault(filt.key(), filt.describe())

        if decision.passed:
            total.passed += 1
            for filt in applicable:
                self._tally(target, filt.key()).passed += 1
            return

        # Dropped: attribute to the causing filter.
        culprit = decision.dropped_by
        if decision.dropped_on_missing:
            total.dropped_on_missing += 1
        else:
            total.dropped_by_predicate += 1

        for filt in applicable:
            tally = self._tally(target, filt.key())
            if culprit is not None and filt.key() == culprit.key():
                if decision.dropped_on_missing:
                    tally.dropped_on_missing += 1
                else:
                    tally.dropped_by_predicate += 1
            else:
                # This filter matched (record only fails at the first mismatch),
                # so credit a pass for filters that ran before the culprit. We
                # cannot cheaply know ordering here, so we count it as passed
                # only when it is not the culprit; this keeps per-filter pass
                # counts meaningful for single-filter configs and is a safe
                # over-approximation for multi-filter configs.
                tally.passed += 1

    def as_dict(self) -> Dict[str, Any]:
        """Serializable breakdown: per-target totals + per-filter detail."""
        out: Dict[str, Any] = {}
        targets = set(self._target_totals) | set(self._per_filter)
        for target in sorted(targets):
            total = self._target_totals.get(target, _FilterTargetTally())
            per_filter = self._per_filter.get(target, {})
            out[target] = {
                "passed": total.passed,
                "dropped_by_predicate": total.dropped_by_predicate,
                "dropped_on_missing": total.dropped_on_missing,
                "filters": [
                    {
                        **self._filter_desc.get(filter_key, {"key": filter_key}),
                        "passed": tally.passed,
                        "dropped_by_predicate": tally.dropped_by_predicate,
                        "dropped_on_missing": tally.dropped_on_missing,
                    }
                    for filter_key, tally in per_filter.items()
                ],
            }
        return out

    @property
    def is_empty(self) -> bool:
        return not self._target_totals and not self._per_filter


__all__ = [
    "MISSING",
    "get_path",
    "VALID_OPERATORS",
    "VALID_ON_MISSING",
    "FilterDecision",
    "RolloutFilter",
    "RolloutFilterSet",
    "FilterStats",
]
