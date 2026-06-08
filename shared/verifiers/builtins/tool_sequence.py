"""Tool-sequence verifier: pure tool-name predicates over a call sequence.

Registered under the ``tool_sequence`` type. This module is the canonical home
for the pure tool-name predicates extracted (verbatim) from the Evaluator's
``_matches_scoring_path``: ``all_tools``, ``any_tools``, ``ordered_tools``,
``first_tool``, ``first_tool_any_of``, ``max_tool_calls`` and
``min_tool_calls`` — plus the ``_contains_subsequence`` and ``_string_list``
helpers they depend on.

These predicates are pure: they reference only the observed tool-name list and
the path config. The ``require_*_pass`` gates (which reference Evaluator result
objects) deliberately stay in ``Evaluator/runner.py``; only the tool portion
lives here so it can be reused without importing Evaluator.

``shared.verifiers`` MUST NOT import ``Evaluator/`` or ``Trainers/``.
"""
from __future__ import annotations

from typing import Any, List, Mapping

from ..contract import VerifierInput, VerifierOutput
from ..registry import register


def evaluate_tool_sequence(
    tool_names: List[str],
    path_cfg: Mapping[str, Any],
) -> tuple[bool, List[str]]:
    """Evaluate the pure tool-name predicates of a scoring path.

    Reproduces, exactly, the tool portion of the Evaluator's
    ``_matches_scoring_path``: the order of checks and the exact reason strings
    are preserved so that callers accumulating reasons see identical output.

    Args:
        tool_names: Observed tool-call names, in call order.
        path_cfg: Path config carrying any of ``all_tools``, ``any_tools``,
            ``ordered_tools``, ``first_tool``, ``first_tool_any_of``,
            ``max_tool_calls``, ``min_tool_calls``.

    Returns:
        ``(matched, reasons)`` where ``matched`` is ``True`` iff no predicate
        produced a reason, and ``reasons`` is the accumulated failure reasons.
    """
    reasons: List[str] = []

    all_tools = _string_list(path_cfg.get("all_tools"))
    if all_tools:
        missing = [tool for tool in all_tools if tool not in tool_names]
        if missing:
            reasons.append(f"missing tools: {', '.join(missing)}")

    any_tools = _string_list(path_cfg.get("any_tools"))
    if any_tools and not any(tool in tool_names for tool in any_tools):
        reasons.append(f"needs any of: {', '.join(any_tools)}")

    ordered_tools = _string_list(path_cfg.get("ordered_tools"))
    if ordered_tools and not _contains_subsequence(tool_names, ordered_tools):
        reasons.append(f"ordered tools not matched: {', '.join(ordered_tools)}")

    first_tool = str(path_cfg.get("first_tool", "")).strip()
    if first_tool and (not tool_names or tool_names[0] != first_tool):
        reasons.append(f"first tool should be {first_tool}")

    first_tool_any_of = _string_list(path_cfg.get("first_tool_any_of"))
    if first_tool_any_of and (not tool_names or tool_names[0] not in first_tool_any_of):
        reasons.append(f"first tool should be one of: {', '.join(first_tool_any_of)}")

    max_tool_calls = path_cfg.get("max_tool_calls")
    if max_tool_calls is not None and len(tool_names) > int(max_tool_calls):
        reasons.append(f"too many tool calls: {len(tool_names)} > {int(max_tool_calls)}")

    min_tool_calls = path_cfg.get("min_tool_calls")
    if min_tool_calls is not None and len(tool_names) < int(min_tool_calls):
        reasons.append(f"too few tool calls: {len(tool_names)} < {int(min_tool_calls)}")

    return len(reasons) == 0, reasons


def _contains_subsequence(items: List[str], subsequence: List[str]) -> bool:
    if not subsequence:
        return True
    pos = 0
    for item in items:
        if item == subsequence[pos]:
            pos += 1
            if pos == len(subsequence):
                return True
    return False


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        clean = value.strip()
        return [clean] if clean else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


# ---------------------------------------------------------------------------
# Verifier wiring
# ---------------------------------------------------------------------------

def _derive_tool_names(sample: VerifierInput) -> List[str]:
    """Derive tool-call names from ``sample.parsed`` when no signal is given.

    Used only when no ``tool_names`` signal is supplied. Keeps the shared
    package free of any Evaluator import while reading the common
    ``tool_calls[*].name`` shape that parsed responses expose.
    """
    parsed = sample.parsed
    tool_calls = getattr(parsed, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return []
    names: List[str] = []
    for call in tool_calls:
        name = getattr(call, "name", None)
        if name is None and isinstance(call, Mapping):
            name = call.get("name")
        if name is not None:
            names.append(str(name))
    return names


class ToolSequenceVerifier:
    """Verifier that scores by the pure tool-name predicates of a scoring path.

    The path config is read from the spec ``params`` (the spec itself, minus the
    ``type``/``name`` keys, also serves as the config). The observed tool names
    are read from ``sample.signals['tool_names']`` when present, falling back to
    names derived from ``sample.parsed``.

    Scoring is binary: ``1.0`` when all configured predicates match, else
    ``0.0``.
    """

    def __init__(self, name: str = "tool_sequence", path_cfg: Mapping[str, Any] | None = None):
        self.name = name
        self.path_cfg = dict(path_cfg) if path_cfg else {}

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        signal_names = sample.signals.get("tool_names")
        if isinstance(signal_names, list):
            tool_names = [str(n) for n in signal_names]
        else:
            tool_names = _derive_tool_names(sample)

        matched, reasons = evaluate_tool_sequence(tool_names, self.path_cfg)
        return VerifierOutput(
            score=1.0 if matched else 0.0,
            passed=matched,
            detail={"reasons": reasons, "tool_names": tool_names},
        )


@register("tool_sequence")
def _build_tool_sequence(spec: Mapping) -> ToolSequenceVerifier:
    params = spec.get("params", spec)
    return ToolSequenceVerifier(
        name=spec.get("name", "tool_sequence"),
        path_cfg=params,
    )
