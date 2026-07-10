"""Regression guard: pool-row metadata must survive into the steer record.

_run_one_pass and _run_batch (MechInterp/cli.py) used to return a record
built ONLY from the fields those passes compute themselves (row_key,
strength, active, answer_text, prompt_len). Any other field the recipe's
rows_path put on a row -- a project's class label, a gold answer's aliases,
anything a grader or a `mechinterp score-gates` group-by field needs -- was
silently dropped before the grader ever saw it (run_steer calls
`grader(rec)`, not `grader(row)`). That broke correctness grading (a grader
that matches answer_text against aliases got None back) and score-gates
grouping (no per-class field to group rows by) for any real recipe, even
though both the grading interface docs and the example project grader's
docstring already promised "any fields carried through from the input rows
pool" (see MechInterp/grading/interface.py and
experiments/common/graders/example_grader.py in the parent project).

This module reuses the tiny offline GPT-2 + word-level tokenizer fixtures
from test_batched_generation.py (same CPU-only, no-download setup) so it
exercises the real generation path rather than a stub, then checks the
returned record dict rather than the generated text.
"""

import json

from tests.mech_interp.test_batched_generation import (
    _build_controller,
    _build_tiny_model,
    _build_tiny_tokenizer,
    _render,
)

from MechInterp.cli import (
    _redact_record_fields,
    _run_batch,
    _run_one_pass,
    _write_checkpoint_record,
)
from MechInterp.config import GenerationContract

_LAYER_IDX = 0
_MAX_NEW_TOKENS = 4

# A pool row carrying the kind of project metadata a real recipe's rows_path
# puts on every row: a class/cell label a score-gates group-by reads, and the
# gold aliases a correctness grader reads, plus one arbitrary extra field to
# confirm the pass-through is not a hard-coded allowlist of two names.
_ROW = {
    "row_key": "r0",
    "_prompt": "w1 w2 w3",
    "_strength": 1.5,
    "_active": True,
    "cell": "should_flip",
    "aliases": ["w4", "w5"],
    "category_canon": "geography",
}


def _grader(rec: dict) -> dict:
    """Tiny fake grader: correct iff any gold alias appears in answer_text.

    Mirrors the real contract (MechInterp/grading/interface.py): a callable
    over the per-row output dict, returning fields to merge back in. Returns
    None for `correct` when aliases are missing entirely, matching the
    reported failure mode this test guards against.
    """
    aliases = rec.get("aliases")
    if aliases is None:
        return {"correct": None}
    text = str(rec.get("answer_text", ""))
    return {"correct": any(a in text for a in aliases)}


def _fresh_row():
    return dict(_ROW)


def test_run_one_pass_carries_pool_row_metadata_through():
    model = _build_tiny_model()
    tokenizer = _build_tiny_tokenizer()
    controller = _build_controller("erase_write")
    generation = GenerationContract(max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)

    rec = _run_one_pass(
        model, tokenizer, controller, _fresh_row(), generation, "anchor", _render
    )

    assert rec["cell"] == "should_flip"
    assert rec["aliases"] == ["w4", "w5"]
    assert rec["category_canon"] == "geography"


def test_run_batch_carries_pool_row_metadata_through():
    model = _build_tiny_model()
    tokenizer = _build_tiny_tokenizer()
    controller = _build_controller("erase_write")
    generation = GenerationContract(max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)

    [rec] = _run_batch(
        model, tokenizer, controller, [_fresh_row()], generation, "anchor", _render
    )

    assert rec["cell"] == "should_flip"
    assert rec["aliases"] == ["w4", "w5"]
    assert rec["category_canon"] == "geography"


def test_no_internal_underscore_fields_leak_into_the_record():
    model = _build_tiny_model()
    tokenizer = _build_tiny_tokenizer()
    controller = _build_controller("erase_write")
    generation = GenerationContract(max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)

    rec_one = _run_one_pass(
        model, tokenizer, controller, _fresh_row(), generation, "anchor", _render
    )
    [rec_batch] = _run_batch(
        model, tokenizer, controller, [_fresh_row()], generation, "anchor", _render
    )

    for rec in (rec_one, rec_batch):
        leaked = [k for k in rec if k.startswith("_")]
        assert not leaked, f"internal fields leaked into the output record: {leaked}"


def test_computed_keys_are_never_shadowed_by_a_same_named_pool_field():
    # A pool row that happens to carry a field named "answer_text" must not
    # clobber the generated text the pass itself computes.
    row = _fresh_row()
    row["answer_text"] = "SHOULD NOT SURVIVE"
    row["n_new_tokens"] = -1
    row["terminated_naturally"] = "spoofed"

    model = _build_tiny_model()
    tokenizer = _build_tiny_tokenizer()
    controller = _build_controller("erase_write")
    generation = GenerationContract(max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)

    rec = _run_one_pass(model, tokenizer, controller, row, generation, "anchor", _render)

    assert rec["answer_text"] != "SHOULD NOT SURVIVE"
    assert rec["n_new_tokens"] != -1
    assert rec["terminated_naturally"] != "spoofed"
    assert isinstance(rec["n_new_tokens"], int)
    assert isinstance(rec["terminated_naturally"], bool)


def test_grader_can_now_read_aliases_carried_through_the_record():
    model = _build_tiny_model()
    tokenizer = _build_tiny_tokenizer()
    controller = _build_controller("erase_write")
    generation = GenerationContract(max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)

    rec = _run_one_pass(
        model, tokenizer, controller, _fresh_row(), generation, "anchor", _render
    )
    rec.update(_grader(rec))

    # Before the fix, aliases was absent from rec and the grader's `correct`
    # came back None regardless of what the model generated -- exactly the
    # failure mode reported against the real project grader.
    assert rec["correct"] is not None
    assert isinstance(rec["correct"], bool)


def test_grader_returns_none_when_aliases_absent_documenting_the_old_failure():
    # Same grader, but on a row with no aliases field at all: this is what
    # EVERY row looked like at the grader before the fix, regardless of what
    # the pool row actually carried.
    grade = _grader({"answer_text": "w4 lives here"})
    assert grade["correct"] is None


def test_record_redaction_is_recursive_and_opt_in():
    rec = {
        "row_key": "r0",
        "answer_text": "generated",
        "aliases": ["gold"],
        "grade": {"answer_value": "decoded", "correct": True},
        "nested": [{"answer_text": "inner", "keep": 1}],
    }

    redacted = _redact_record_fields(rec, {"answer_text", "aliases", "answer_value"})

    assert redacted == {
        "row_key": "r0",
        "grade": {"correct": True},
        "nested": [{"keep": 1}],
    }
    assert _redact_record_fields(rec, set()) is rec


def test_checkpoint_writer_redacts_before_persisting(tmp_path):
    path = tmp_path / "rows.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        _write_checkpoint_record(
            handle,
            {
                "row_key": "r0",
                "answer_text": "generated",
                "grade": {"answer_value": "decoded", "correct": True},
            },
            {"answer_text", "answer_value"},
        )

    [record] = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert record == {"row_key": "r0", "grade": {"correct": True}}
