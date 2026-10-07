"""Train/eval contamination check (shared/contamination.py)."""

import json
from pathlib import Path

import pytest

from Evaluator.prompt_sets import PromptCase
from shared import contamination as cont


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _row(text: str, *, row_id: str = "ds:1") -> cont.TrainingRow:
    dataset, line = row_id.rsplit(":", 1)
    return cont.TrainingRow(
        row_id=row_id, dataset=dataset, line_number=int(line), row_format="messages",
        text=text, prompt_texts=[text],
    )


# -- normalization -------------------------------------------------------------

def test_normalize_text_nfkc_lowercase_punctuation_whitespace():
    assert cont.normalize_text("  Héllo,\tWORLD!!  ") == "héllo world"
    # NFKC folds the ligature and full-width digits; punctuation/underscores split words.
    assert cont.normalize_text("ﬁle_name/x.md １２") == "file name x md 12"
    assert cont.normalize_text("don't") == "don t"
    assert cont.normalize_text("") == ""


def test_tokenize_and_ngram_hashes():
    tokens = cont.tokenize("a b c d e")
    assert tokens == ["a", "b", "c", "d", "e"]
    assert len(cont.ngram_hashes(tokens, 3)) == 3
    assert cont.ngram_hashes(tokens, 6) == set()
    # Duplicate n-grams collapse.
    assert len(cont.ngram_hashes(["x"] * 10, 2)) == 1
    with pytest.raises(ValueError):
        cont.ngram_hashes(tokens, 0)


# -- containment math ----------------------------------------------------------

def test_containment_fraction_of_item_ngrams():
    item = cont.ngram_hashes(cont.tokenize("one two three four five six"), 2)  # 5 bigrams
    row = cont.ngram_hashes(cont.tokenize("zero one two three nine"), 2)  # shares 2
    assert cont.containment(item, row) == pytest.approx(2 / 5)
    assert cont.containment(item, item) == 1.0
    assert cont.containment(set(), row) == 0.0


def test_run_check_reports_max_containment_over_single_rows():
    cfg = cont.ContaminationConfig(ngram=3, threshold=0.5, min_item_tokens=3, top_k=10)
    eval_text = "alpha beta gamma delta epsilon zeta"  # 4 trigrams
    rows = [
        _row("alpha beta gamma delta and unrelated words", row_id="ds:1"),  # 2/4
        _row("noise gamma delta epsilon zeta noise", row_id="ds:2"),  # 2/4 + ...
        _row("completely different text here", row_id="ds:3"),
    ]
    items = [cont.EvalItem(source="s.yaml", item_id="t1", kind="prompt", text=eval_text)]
    report = cont.run_check(rows, items, cfg)

    item = report["items"][0]
    assert item["total_ngrams"] == 4
    assert item["max_containment"] == 0.5  # never 4/4: no single row holds all trigrams
    assert item["flagged"] is True
    assert report["summary"]["items_over_threshold"] == 1
    assert set(report["flagged_training_rows"]) == {"ds:1", "ds:2"}
    assert [p["train_row_id"] for p in report["top_pairs"]] == ["ds:1", "ds:2"]


# -- index -----------------------------------------------------------------------

def test_index_postings_and_restriction():
    index = cont.NgramIndex([2])
    index.add(0, "a b c")
    index.add(1, "b c d")
    ab, bc, cd = (cont.ngram_hashes(["a", "b"], 2).pop(), cont.ngram_hashes(["b", "c"], 2).pop(),
                  cont.ngram_hashes(["c", "d"], 2).pop())
    assert index.postings[bc] == [0, 1]
    assert index.postings[ab] == [0]
    assert index.match_counts({ab, bc, cd}) == {0: 2, 1: 2}

    restricted = cont.NgramIndex([2], restrict_to={bc})
    restricted.add(0, "a b c")
    restricted.add(1, "b c d")
    assert set(restricted.postings) == {bc}
    assert restricted.row_count == 2


def test_short_items_use_whole_item_gram_or_exact_only():
    cfg = cont.ContaminationConfig(ngram=8, threshold=0.5, min_item_tokens=3)
    rows = [_row("please archive the completed project files today", row_id="ds:1")]
    items = [
        cont.EvalItem(source="s", item_id="short", kind="prompt", text="Archive the completed project files"),
        cont.EvalItem(source="s", item_id="tiny", kind="prompt", text="Hi there"),
    ]
    report = cont.run_check(rows, items, cfg)
    by_id = {item["item_id"]: item for item in report["items"]}
    assert by_id["short"]["gram_size"] == 5
    assert by_id["short"]["max_containment"] == 1.0
    assert by_id["tiny"]["gram_size"] is None
    assert report["summary"]["too_short_items"] == 1


def test_exact_duplicate_prompts_after_normalization():
    cfg = cont.ContaminationConfig(ngram=8, min_item_tokens=50)
    row = cont.TrainingRow(
        row_id="ds:7", dataset="ds", line_number=7, row_format="messages",
        text="Move my meeting notes to the archive!\nsure", prompt_texts=["Move my meeting notes to the archive!"],
    )
    items = [cont.EvalItem(source="s", item_id="dup", kind="prompt", text="move my MEETING notes to the archive",
                           exact_text="move my MEETING notes to the archive")]
    report = cont.run_check([row], items, cfg)
    assert report["summary"]["exact_duplicate_prompts"] == 1
    assert report["items"][0]["exact_duplicate_row_ids"] == ["ds:7"]
    assert report["passed"] is False
    assert report["flagged_training_rows"]["ds:7"][0]["reason"] == "exact_duplicate_prompt"


# -- training row formats ----------------------------------------------------------

def test_detects_trainer_row_formats():
    assert cont.detect_row_format({"messages": [{"role": "user", "content": "x"}]}) == "messages"
    assert cont.detect_row_format({"conversations": [{"role": "user", "content": "x"}], "label": True}) == "messages+label"
    assert cont.detect_row_format({"prompt": "p", "completion": "c"}) == "prompt_completion"
    assert cont.detect_row_format({"prompt": [], "chosen": [], "rejected": []}) == "dpo"
    assert cont.detect_row_format({"prompt": [{"role": "user", "content": "x"}]}) == "prompt_only"
    raw = {"schema_version": "syntunia-sft-row/v1", "format": "raw_text", "text": "t", "split": "train"}
    assert cont.detect_row_format(raw) == "raw_text"
    with pytest.raises(ValueError):
        cont.detect_row_format({"text": "an arbitrary text column is not a trainer format"})


def test_build_training_row_renders_tool_calls_and_filters_roles():
    row = {
        "conversations": [
            {"role": "system", "content": "SYSTEM boilerplate"},
            {"role": "user", "content": "find notes"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "searchTool", "arguments": "{\"query\": \"notes\"}"}}]},
        ]
    }
    built = cont.build_training_row(row, dataset="ds", line_number=3, train_roles=["user", "assistant"])
    assert built.row_id == "ds:3"
    assert "SYSTEM" not in built.text
    assert "tool_call: searchTool" in built.text
    assert built.prompt_texts == ["find notes"]

    dpo = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "good answer"}],
        "rejected": [{"role": "assistant", "content": "bad answer"}],
    }
    built = cont.build_training_row(dpo, dataset="ds", line_number=1)
    assert "good answer" in built.text and "bad answer" in built.text


def test_load_training_rows_reports_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "x"}]}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        cont.load_training_rows(path)


# -- eval sources --------------------------------------------------------------------

def test_items_from_prompt_cases_prompt_and_reference_fields():
    case = PromptCase(
        case_id="c1",
        question="What is the capital?",
        metadata={"system": "You are terse.", "reference": {"answer": "Paris is the capital", "n": 3}},
    )
    items = cont.items_from_prompt_cases([case], source="set.json", reference_fields=["reference"])
    assert [(i.kind, i.text) for i in items] == [
        ("prompt", "What is the capital?"),
        ("reference", "Paris is the capital"),
    ]
    with_system = cont.items_from_prompt_cases([case], source="set.json", include_system=True)
    assert "You are terse." in with_system[0].text


def test_load_eval_source_uses_evaluator_scenario_loader(tmp_path):
    scenarios = tmp_path / "config" / "scenarios"
    scenarios.mkdir(parents=True)
    (scenarios / "mini.yaml").write_text(
        "name: mini\ntests:\n"
        "  - id: t1\n    question: Summarize the quarterly planning notes for the team\n    tags: [x]\n"
        "  - id: t2\n    messages:\n      - {role: user, content: first turn}\n"
        "      - {role: assistant, content: ok}\n      - {role: user, content: second turn}\n",
        encoding="utf-8",
    )
    items = cont.load_eval_source(scenarios / "mini.yaml")
    assert [(i.item_id, i.text) for i in items] == [
        ("t1", "Summarize the quarterly planning notes for the team"),
        ("t2", "first turn\nsecond turn"),
    ]


def test_load_text_source(tmp_path):
    path = _write_jsonl(tmp_path / "holdout.jsonl", [{"id": "a", "body": {"text": "hello world"}}])
    items = cont.load_text_source(path, text_field="body.text")
    assert items[0].item_id == "a" and items[0].text == "hello world" and items[0].kind == "text"
    with pytest.raises(ValueError, match="missing string field"):
        cont.load_text_source(path, text_field="nope")


# -- decontaminated output ------------------------------------------------------------

def test_write_decontaminated_drops_flagged_rows_and_writes_sidecar(tmp_path):
    rows = [{"messages": [{"role": "user", "content": f"row {i}"}]} for i in range(4)]
    source = _write_jsonl(tmp_path / "train.jsonl", rows)
    flagged = {"train.jsonl:2": [{"reason": "containment", "containment": 0.9, "eval_item_id": "e1"}]}
    out = tmp_path / "out" / "train.clean.jsonl"

    summary = cont.write_decontaminated(source, "train.jsonl", flagged, out)

    kept = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert kept == [rows[0], rows[2], rows[3]]
    sidecar = json.loads(Path(summary["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["removed_rows"] == 1 and sidecar["kept_rows"] == 3
    assert sidecar["removed"][0]["row_id"] == "train.jsonl:2"
    assert sidecar["removed"][0]["reasons"][0]["eval_item_id"] == "e1"


def test_dataset_labels_are_unique():
    labels = cont.dataset_labels([Path("a/x/train.jsonl"), Path("b/x/train.jsonl"), Path("c/other.jsonl")])
    assert labels == ["a/x/train.jsonl", "b/x/train.jsonl", "other.jsonl"]
