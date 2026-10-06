"""
Decision training rows: one (state, typed question, gold answer) triple per row.

Location: Trainers/decision/decision_core/examples.py
Used by:  collate.py, evaluate.py, train_decision.py, build_corpus.py.

Two on-disk shapes load into the same DecisionExample:

1. ``decision-row/v1`` -- field-compatible with the strands-decider ``Example``
   JSONL (``kind, state, instructions, options, label, task, weight,
   instruction_variants``), so a corpus built by ``strands-decider data build``
   trains here unchanged.

2. Jev request rows -- ``{"state": ..., "questions": {name: {type, instructions,
   criteria}}, "answers": {name: gold}}``, the shape the Jev / Strands serving
   APIs accept, plus gold answers. Each question becomes one DecisionExample.

``label`` always indexes the *canonical* option order. Option shuffling happens
at collate time, so one stored row yields a different slot binding every epoch.
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Iterable, Iterator, cast

KINDS = ("noul", "choice", "score")

# Noul questions always render as the two-option list (false, true).
NOUL_OPTION_NAMES = ("false", "true")
NOUL_DEFAULT_CRITERIA = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}


@dataclass
class DecisionExample:
    kind: str
    state: Any
    instructions: str
    # Canonical option order as [name, description] pairs. For score questions the
    # name is the level index as a string and the description is the rubric text.
    options: list[list[str]]
    label: int
    task: str = "unknown"
    weight: float = 1.0
    instruction_variants: list[str] = field(default_factory=list)
    # Free-form provenance carried through IO untouched (e.g. a prior-knowledge
    # label or a source-popularity score). Never rendered.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown question kind {self.kind!r}; expected one of {KINDS}")
        if len(self.options) < 2:
            raise ValueError(f"a {self.kind} question needs at least 2 options (task {self.task!r})")
        if not 0 <= self.label < len(self.options):
            raise ValueError(
                f"label {self.label} out of range for {len(self.options)} options (task {self.task!r})"
            )
        if self.kind == "noul" and [o[0] for o in self.options] != list(NOUL_OPTION_NAMES):
            raise ValueError("noul options must be [['false', ...], ['true', ...]] in that order")

    @property
    def n_options(self) -> int:
        return len(self.options)

    def all_instructions(self) -> list[str]:
        return [self.instructions, *self.instruction_variants]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionExample":
        return cls(
            kind=str(d["kind"]),
            state=d["state"],
            instructions=str(d["instructions"]),
            options=[[str(o[0]), "" if o[1] is None else str(o[1])] for o in d["options"]],
            label=int(d["label"]),
            task=str(d.get("task", "unknown")),
            weight=float(d.get("weight", 1.0)),
            instruction_variants=[str(v) for v in (d.get("instruction_variants") or [])],
            meta=dict(d.get("meta") or {}),
        )


# ---------------------------------------------------------------------------
# Jev request rows
# ---------------------------------------------------------------------------

def _criteria_pairs(kind: str, criteria: Any) -> list[list[str]]:
    if kind == "noul":
        crit = {**NOUL_DEFAULT_CRITERIA, **{str(k): v for k, v in (criteria or {}).items()}}
        return [[name, _text(crit[name])] for name in NOUL_OPTION_NAMES]
    if kind == "choice":
        if isinstance(criteria, dict):
            return [[str(k), _text(v)] for k, v in criteria.items()]
        if isinstance(criteria, list):
            return [[str(k), ""] for k in criteria]
        raise ValueError("choice criteria must be a mapping of option -> description or a list")
    if kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("score criteria must be a list of levels, lowest first")
        return [[str(i), _text(v)] for i, v in enumerate(criteria)]
    raise ValueError(f"unknown question type {kind!r}")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _gold_index(kind: str, options: list[list[str]], gold: Any, name: str) -> int:
    if kind == "noul":
        if isinstance(gold, bool):
            return int(gold)
        if isinstance(gold, str) and gold.lower() in NOUL_OPTION_NAMES:
            return NOUL_OPTION_NAMES.index(gold.lower())
        raise ValueError(f"noul answer for {name!r} must be a bool or 'true'/'false', got {gold!r}")
    if kind == "score":
        if isinstance(gold, bool) or not isinstance(gold, (int, str)):
            raise ValueError(f"score answer for {name!r} must be a level index or level text")
        if isinstance(gold, int):
            index = gold
        else:
            names = [o[1] for o in options]
            if gold not in names:
                raise ValueError(f"score answer {gold!r} for {name!r} is not one of the levels")
            index = names.index(gold)
        if not 0 <= index < len(options):
            raise ValueError(f"score answer {gold!r} for {name!r} is out of range")
        return index
    names = [o[0] for o in options]
    if gold not in names:
        raise ValueError(f"choice answer {gold!r} for {name!r} is not one of {names}")
    return names.index(gold)


def examples_from_jev_row(row: dict[str, Any], *, task: str = "jev") -> list[DecisionExample]:
    """Explode one Jev-shaped request with gold ``answers`` into DecisionExamples.

    Questions without a gold answer are skipped -- they carry no training signal.
    """
    questions = row.get("questions") or {}
    answers = row.get("answers") or {}
    out: list[DecisionExample] = []
    for name, q in questions.items():
        if name not in answers:
            continue
        kind = str(q.get("type", "")).strip().lower()
        options = _criteria_pairs(kind, q.get("criteria"))
        out.append(
            DecisionExample(
                kind=kind,
                state=row["state"],
                instructions=_text(q.get("instructions", "")),
                options=options,
                label=_gold_index(kind, options, answers[name], name),
                task=str(row.get("task") or q.get("task") or task),
                weight=float(row.get("weight", 1.0)),
                instruction_variants=[str(v) for v in (q.get("instruction_variants") or [])],
                meta=dict(row.get("meta") or {}),
            )
        )
    return out


def examples_from_dict(d: dict[str, Any]) -> list[DecisionExample]:
    """Dispatch on row shape: decision-row/v1 (has ``kind``) or a Jev request row."""
    if "kind" in d:
        return [DecisionExample.from_dict(d)]
    if "questions" in d and "state" in d:
        return examples_from_jev_row(d)
    raise ValueError("row is neither a decision row (kind/options/label) nor a Jev request row")


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------

def _open(path: str | Path, mode: str) -> IO[str]:
    path = str(path)
    if path.endswith(".gz"):
        return cast(IO[str], gzip.open(path, mode + "t", encoding="utf-8"))
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[DecisionExample]:
    with _open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield from examples_from_dict(json.loads(line))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc


def load_examples(paths: Iterable[str | Path]) -> list[DecisionExample]:
    out: list[DecisionExample] = []
    for p in paths:
        out.extend(read_jsonl(p))
    return out


def write_jsonl(path: str | Path, examples: Iterable[DecisionExample]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as fh:
        for ex in examples:
            fh.write(ex.to_json() + "\n")
            n += 1
    return n


def split_by_task(
    examples: list[DecisionExample], *, val_fraction: float, seed: int
) -> tuple[list[DecisionExample], list[DecisionExample]]:
    """Train/validation split stratified by task, so small tasks are not lost."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    rng = random.Random(seed)
    by_task: dict[str, list[DecisionExample]] = {}
    for ex in examples:
        by_task.setdefault(ex.task, []).append(ex)
    train: list[DecisionExample] = []
    val: list[DecisionExample] = []
    for task in sorted(by_task):
        items = list(by_task[task])
        rng.shuffle(items)
        n_val = int(round(len(items) * val_fraction)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
