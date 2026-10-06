"""Decision rows, prompt rendering and the collator (no model, no transformers).

Uses a whitespace/punctuation fake tokenizer with real character offsets, so
the option-position and truncation logic is exercised exactly as with a fast
HF tokenizer.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DECISION_DIR = REPO_ROOT / "Trainers" / "decision"
for p in (REPO_ROOT, DECISION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch = pytest.importorskip("torch")

from decision_core.collate import CollatorConfig, DecisionCollator, target_distribution  # noqa: E402
from decision_core.examples import (  # noqa: E402
    DecisionExample,
    examples_from_dict,
    examples_from_jev_row,
    read_jsonl,
    split_by_task,
    write_jsonl,
)
from decision_core.prompting import option_markers, render_prompt  # noqa: E402

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class FakeTokenizer:
    """Word/punctuation tokens with exact character offsets; ids from a growing vocab."""

    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<pad>"

    def __init__(self):
        self.vocab = {"<pad>": 0}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab))

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        for m in TOKEN_RE.finditer(text):
            ids.append(self._id(m.group()))
            offsets.append((m.start(), m.end()))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]


def choice_example(label=2, task="intent"):
    return DecisionExample(
        kind="choice",
        state="My card was charged twice.",
        instructions="Which team should handle this?",
        options=[["billing", "payments"], ["technical", "bugs"], ["fraud", "unauthorised use"],
                 ["sales", "new plans"]],
        label=label,
        task=task,
    )


def score_example(label=1):
    return DecisionExample(
        kind="score", state="Pretty good.", instructions="How positive?",
        options=[["0", "negative"], ["1", "mixed"], ["2", "positive"]], label=label, task="sentiment",
    )


# ---- examples ---------------------------------------------------------------

def test_strands_row_roundtrip(tmp_path):
    row = {"kind": "noul", "state": "x", "instructions": "Is it spam?",
           "options": [["false", "not spam"], ["true", "spam"]], "label": 1, "task": "spam",
           "weight": 1.0, "instruction_variants": ["Spam?"]}
    ex = examples_from_dict(row)[0]
    assert ex.label == 1 and ex.instruction_variants == ["Spam?"]
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [ex, choice_example()])
    back = list(read_jsonl(path))
    assert back[0] == ex and back[1].options[2][0] == "fraud"


def test_jev_row_explodes_questions_and_maps_gold():
    row = {
        "state": "Help! My payouts have been failing for 3 days.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "money", "technical": "bugs"}},
            "severity": {"type": "score", "instructions": "How severe?",
                         "criteria": ["low", "medium", "high"]},
            "unlabelled": {"type": "noul", "instructions": "ignored"},
        },
        "answers": {"urgent": True, "team": "technical", "severity": "high"},
    }
    exs = examples_from_jev_row(row)
    assert [e.kind for e in exs] == ["noul", "choice", "score"]
    assert exs[0].options[1][0] == "true" and exs[0].label == 1
    assert exs[1].label == 1
    assert exs[2].label == 2 and exs[2].options[2] == ["2", "high"]


@pytest.mark.parametrize("bad", [
    {"kind": "choice", "state": "s", "instructions": "i", "options": [["a", ""]], "label": 0},
    {"kind": "choice", "state": "s", "instructions": "i", "options": [["a", ""], ["b", ""]], "label": 5},
    {"kind": "noul", "state": "s", "instructions": "i", "options": [["true", ""], ["false", ""]], "label": 0},
    {"kind": "rank", "state": "s", "instructions": "i", "options": [["a", ""], ["b", ""]], "label": 0},
])
def test_invalid_rows_rejected(bad):
    with pytest.raises(ValueError):
        examples_from_dict(bad)


def test_jev_choice_gold_must_be_an_option():
    row = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i",
                                             "criteria": {"a": "", "b": ""}}},
           "answers": {"q": "c"}}
    with pytest.raises(ValueError):
        examples_from_jev_row(row)


def test_split_by_task_keeps_every_task():
    rows = [choice_example(task=t) for t in ["a"] * 40 + ["b"] * 10 + ["c"] * 3]
    train, val = split_by_task(rows, val_fraction=0.2, seed=0)
    assert len(train) + len(val) == len(rows)
    assert Counter(e.task for e in val) == {"a": 8, "b": 2, "c": 1}


# ---- prompting --------------------------------------------------------------

def test_render_layout_and_spans():
    ex = choice_example()
    r = render_prompt(ex.state, ex.kind, ex.instructions, ex.options, order=[2, 0, 3, 1])
    assert r.text.startswith("<state>\nMy card was charged twice.\n</state>\n<question type=\"choice\">")
    assert r.text.endswith("</options>\n</question>\n<answer>")
    assert r.slot_names == ("fraud", "billing", "sales", "technical")
    for marker, (a, b), name in zip(r.markers, r.option_spans, r.slot_names):
        line = r.text[a:b]
        assert line.startswith(f"{marker}. {name}") and "\n" not in line


def test_markers():
    assert option_markers("numbers", 3) == ["1", "2", "3"]
    assert option_markers("letters", 3) == ["A", "B", "C"]
    with pytest.raises(ValueError):
        option_markers("letters", 27)


# ---- collator ---------------------------------------------------------------

def test_shuffle_remaps_label_and_points_at_option_lines():
    tok = FakeTokenizer()
    coll = DecisionCollator(tok, CollatorConfig(max_length=512, seed=3), train=True)
    ex = choice_example(label=2)
    seen_slots = set()
    for _ in range(30):
        batch = coll([ex])
        order = batch["orders"][0]
        slot = int(batch["labels"][0])
        assert order[slot] == ex.label  # the gold option is rendered at the labelled slot
        assert batch["target"][0, slot] == 1.0
        seen_slots.add(slot)
        # each option index is the last token of that option's line
        inv = {v: k for k, v in tok.vocab.items()}
        ids = batch["input_ids"][0].tolist()
        for k, idx in enumerate(batch["option_index"][0].tolist()):
            name, desc = ex.options[order[k]]
            assert inv[ids[idx]] == desc.split()[-1]
        assert batch["answer_index"][0] == int(batch["attention_mask"][0].sum()) - 1
    assert len(seen_slots) > 1, "options were never shuffled"


def test_eval_collator_is_canonical():
    coll = DecisionCollator(FakeTokenizer(), CollatorConfig(), train=False)
    batch = coll([choice_example(label=3)])
    assert batch["orders"][0] == [0, 1, 2, 3] and int(batch["labels"][0]) == 3


def test_score_rows_only_reverse_and_smoothing_follows_levels():
    ex = score_example(label=1)
    dist = target_distribution(ex, [2, 1, 0], ordinal_smoothing=0.1)
    # level 1 sits in slot 1 either way; neighbours 0 and 2 share the 0.1
    assert dist == pytest.approx([0.05, 0.9, 0.05])
    edge = score_example(label=0)
    assert target_distribution(edge, [2, 1, 0], 0.1) == pytest.approx([0.0, 0.1, 0.9])
    coll = DecisionCollator(FakeTokenizer(), CollatorConfig(reverse_score_prob=0.5, seed=1), train=True)
    orders = {tuple(coll([ex])["orders"][0]) for _ in range(30)}
    assert orders <= {(0, 1, 2), (2, 1, 0)} and len(orders) == 2


def test_padding_and_ragged_option_counts():
    coll = DecisionCollator(FakeTokenizer(), CollatorConfig(), train=False)
    batch = coll([choice_example(), score_example()])
    assert batch["option_index"].shape == (2, 4)
    assert batch["option_index"][1, 3] == -1
    assert batch["n_options"].tolist() == [4, 3]
    assert batch["target"][1].sum() == pytest.approx(1.0)


def test_overlong_prompt_cuts_state_from_front():
    long_state = " ".join(f"w{i}" for i in range(500))
    ex = DecisionExample(kind="noul", state=long_state, instructions="Is it long?",
                         options=[["false", ""], ["true", ""]], label=1)
    tok = FakeTokenizer()
    full = DecisionCollator(tok, CollatorConfig(max_length=10**6), train=False)([ex])
    cut = DecisionCollator(tok, CollatorConfig(max_length=60), train=False)([ex])
    assert cut["input_ids"].shape[1] == 60
    assert cut["input_ids"][0].tolist() == full["input_ids"][0, -60:].tolist()
    shift = full["input_ids"].shape[1] - 60
    assert (cut["option_index"] == full["option_index"] - shift).all()
    with pytest.raises(ValueError):
        DecisionCollator(tok, CollatorConfig(max_length=5), train=False)([ex])


def test_instruction_variants_sampled_only_in_training():
    ex = choice_example()
    ex.instruction_variants = ["Route this ticket.", "Who owns this?"]
    tok = FakeTokenizer()
    train = DecisionCollator(tok, CollatorConfig(seed=0), train=True)
    seen = {train.instruction(ex) for _ in range(40)}
    assert seen == set(ex.all_instructions())
    assert DecisionCollator(tok, CollatorConfig(), train=False).instruction(ex) == ex.instructions


def test_config_loader_rejects_unknown_keys(tmp_path):
    from decision_core.config import load_run_config

    good = load_run_config(DECISION_DIR / "configs" / "config.yaml")
    assert good.model.readout == "pointer" and good.model.registry_name == "qwen35-2b-base"
    letters = load_run_config(DECISION_DIR / "configs" / "letter_logits.yaml")
    assert letters.model.readout == "letter_logits" and letters.prompt.marker_style == "letters"
    bad = tmp_path / "bad.yaml"
    bad.write_text(json.dumps({"model": {"readout": "pointer", "nope": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="nope"):
        load_run_config(bad)
    bad.write_text(json.dumps({"model": {"readout": "slot"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="readout"):
        load_run_config(bad)
    bad.write_text(json.dumps({"prompt": {"headers": {"noul": "x", "choice": "y"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="headers"):
        load_run_config(bad)


def test_configured_headers_render():
    ex = choice_example()
    r = render_prompt(ex.state, ex.kind, ex.instructions, ex.options,
                      headers={"noul": "N", "choice": "Route the ticket.", "score": "S"})
    assert "\nRoute the ticket.\nWhich team should handle this?\n" in r.text
    coll = DecisionCollator(FakeTokenizer(), CollatorConfig(headers={"noul": "N", "choice": "Route it.",
                                                                     "score": "S"}), train=False)
    assert "Route it." in coll.encode(ex).rendered.text
