"""
Render a (state, question) pair into the prompt both readouts read.

Location: Trainers/decision/decision_core/prompting.py
Used by:  collate.py (training), inference.py (serving), evaluate.py.

The layout follows the strands-decider prompt so a strands corpus and its
held-out evaluations transfer unchanged:

    <state>
    {state}
    </state>
    <question type="{kind}">
    {header}
    {instructions}
    <options>
    1. {name} — {description}
    2. ...
    </options>
    </question>
    <answer>

Each option is exactly one line. The pointer readout scores option k from the
hidden state of the last token of its line; the letter-logit readout scores it
from the LM's logit for option k's marker token ("1".."9" or "A".."Z") at the
``<answer>`` position. The option order passed in is the order rendered, and
``slot_names`` records the binding so callers never track the permutation.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from typing import Any, Sequence

PROMPT_FORMAT = "decision-prompt/v1"

# Default per-kind instruction lines (the strands-decider wording). Runs set
# their own under the config's `prompt.headers`.
HEADERS = {
    "noul": "Decide whether the statement is true of the state.",
    "choice": "Select exactly one option.",
    "score": "Rate the state against the ordered levels below (lowest first).",
}

MARKER_STYLES = ("numbers", "letters")


def option_markers(style: str, n: int) -> list[str]:
    """The visible marker of each option line, in rendered order."""
    if style == "numbers":
        return [str(i + 1) for i in range(n)]
    if style == "letters":
        if n > len(string.ascii_uppercase):
            raise ValueError(f"letter markers support at most 26 options, got {n}")
        return list(string.ascii_uppercase[:n])
    raise ValueError(f"unknown option marker style {style!r}; expected one of {MARKER_STYLES}")


def render_content(content: Any) -> str:
    """Flatten a state or instruction into text deterministically."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    kind: str
    # slot_names[k] is the canonical option name rendered at position k.
    slot_names: tuple[str, ...]
    # (start, end) character span of each option line within `text`.
    option_spans: tuple[tuple[int, int], ...]
    markers: tuple[str, ...]

    @property
    def n_options(self) -> int:
        return len(self.slot_names)


def render_prompt(
    state: Any,
    kind: str,
    instructions: str,
    options: Sequence[Sequence[str]],
    *,
    order: Sequence[int] | None = None,
    marker_style: str = "numbers",
    headers: dict[str, str] | None = None,
) -> RenderedPrompt:
    """Render one question over a state. ``order[k]`` is the canonical option shown at slot k.

    ``headers`` (one line per kind) comes from the run config's ``prompt.headers``
    and is saved with the checkpoint; ``HEADERS`` is the default.
    """
    headers = HEADERS if headers is None else headers
    if kind not in headers:
        raise ValueError(f"unknown question kind {kind!r}")
    if order is None:
        order = list(range(len(options)))
    elif sorted(order) != list(range(len(options))):
        raise ValueError("order must be a permutation of the option indices")

    shown = [options[i] for i in order]
    markers = option_markers(marker_style, len(shown))

    head = (
        f"<state>\n{render_content(state)}\n</state>\n"
        f'<question type="{kind}">\n'
        f"{headers[kind]}\n"
        f"{render_content(instructions)}\n"
        f"<options>\n"
    )
    lines: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = len(head)
    for marker, (name, desc) in zip(markers, shown):
        desc = " ".join((desc or "").split())
        line = f"{marker}. {name}" + (f" — {desc}" if desc else "")
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    text = head + "\n".join(lines) + "\n</options>\n</question>\n<answer>"
    return RenderedPrompt(
        text=text,
        kind=kind,
        slot_names=tuple(name for name, _ in shown),
        option_spans=tuple(spans),
        markers=tuple(markers),
    )


def last_token_in_span(offsets: Sequence[Sequence[int]], span: Sequence[int]) -> int:
    """Index of the last token lying wholly inside a character span.

    Raises when truncation removed the option: scoring it from a neighbour's
    hidden state would be silently wrong.
    """
    a, b = span
    last = -1
    for j, (lo, hi) in enumerate(offsets):
        if hi <= lo:
            continue
        if lo >= a and hi <= b:
            last = j
    if last < 0:
        raise ValueError(f"option span ({a}, {b}) has no tokens; the prompt was truncated through it")
    return last
