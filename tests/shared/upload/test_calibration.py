import hashlib
import json
from pathlib import Path

import pytest

from shared.upload.converters.calibration import (
    RENDER_CHAT_TEMPLATE,
    RENDER_PLAIN,
    CalibrationSpec,
    render_calibration_text,
)


class FakeChatTokenizer:
    chat_template = "{fake}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        return "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)


def _write_jsonl(path: Path, rows) -> Path:
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _chat_row(i: int, **extra):
    return {
        "messages": [
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        ],
        **extra,
    }


def test_renders_supported_row_shapes_with_chat_template(tmp_path):
    dataset = _write_jsonl(tmp_path / "d.jsonl", [
        _chat_row(0),
        {"conversations": [{"role": "user", "content": "conv q"}, {"role": "assistant", "content": "conv a"}]},
        {"prompt": "pc q", "completion": "pc a"},
    ])
    result = render_calibration_text(
        dataset, tmp_path / "out" / "cal.txt", tokenizer=FakeChatTokenizer(), max_rows=10
    )
    text = result.output_path.read_text(encoding="utf-8")

    assert result.render_mode == RENDER_CHAT_TEMPLATE
    assert result.rows_total == 3
    assert result.rows_used == 3
    assert result.rows_skipped == 0
    assert "<|user|>question 0<|end|><|assistant|>answer 0<|end|>" in text
    assert "<|user|>conv q<|end|>" in text
    assert "<|user|>pc q<|end|><|assistant|>pc a<|end|>" in text
    assert result.text_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert result.chars == len(text)


def test_tool_calls_are_rendered_through_shared_sanitizer(tmp_path):
    dataset = _write_jsonl(tmp_path / "d.jsonl", [{
        "messages": [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "search", "arguments": "{\"q\": \"x\"}"}}
            ]},
        ]
    }])
    result = render_calibration_text(dataset, tmp_path / "cal.txt", tokenizer=FakeChatTokenizer())
    assert "tool_call: search" in result.output_path.read_text(encoding="utf-8")


def test_plain_fallback_without_chat_template(tmp_path):
    dataset = _write_jsonl(tmp_path / "d.jsonl", [_chat_row(1)])
    result = render_calibration_text(dataset, tmp_path / "cal.txt", tokenizer=None)
    assert result.render_mode == RENDER_PLAIN
    assert result.output_path.read_text(encoding="utf-8") == "question 1\n\nanswer 1\n"


def test_tokenizer_without_template_falls_back_to_plain(tmp_path):
    class NoTemplate(FakeChatTokenizer):
        chat_template = None

    dataset = _write_jsonl(tmp_path / "d.jsonl", [_chat_row(1)])
    result = render_calibration_text(dataset, tmp_path / "cal.txt", tokenizer=NoTemplate())
    assert result.render_mode == RENDER_PLAIN


def test_unrenderable_rows_are_skipped_and_counted(tmp_path):
    dataset = _write_jsonl(tmp_path / "d.jsonl", [
        _chat_row(0),
        "{not json",
        json.dumps(["a", "list"]),
        {"unrelated": "shape"},
        _chat_row(1),
    ])
    result = render_calibration_text(dataset, tmp_path / "cal.txt", tokenizer=FakeChatTokenizer())
    assert result.rows_total == 5
    assert result.rows_used == 2
    assert result.rows_skipped == 3


def test_sampling_is_deterministic_and_seeded(tmp_path):
    dataset = _write_jsonl(tmp_path / "d.jsonl", [_chat_row(i) for i in range(50)])
    tok = FakeChatTokenizer()
    a = render_calibration_text(dataset, tmp_path / "a.txt", tokenizer=tok, max_rows=10, seed=7)
    b = render_calibration_text(dataset, tmp_path / "b.txt", tokenizer=tok, max_rows=10, seed=7)
    c = render_calibration_text(dataset, tmp_path / "c.txt", tokenizer=tok, max_rows=10, seed=8)

    assert a.rows_used == 10
    assert a.text_sha256 == b.text_sha256
    assert a.text_sha256 != c.text_sha256


def test_character_budget_stops_before_overflow(tmp_path):
    dataset = _write_jsonl(tmp_path / "d.jsonl", [_chat_row(i) for i in range(20)])
    one_row = len(FakeChatTokenizer().apply_chat_template(_chat_row(0)["messages"]))
    result = render_calibration_text(
        dataset, tmp_path / "cal.txt", tokenizer=FakeChatTokenizer(),
        max_rows=100, max_chars=one_row * 3 + 10,
    )
    assert result.rows_used == 3


def test_label_field_filters_undesirable_rows(tmp_path):
    dataset = _write_jsonl(tmp_path / "d.jsonl", [
        _chat_row(0, label=True),
        _chat_row(1, label=False),
        _chat_row(2),
    ])
    result = render_calibration_text(
        dataset, tmp_path / "cal.txt", tokenizer=FakeChatTokenizer(), label_field="label"
    )
    assert result.rows_used == 2
    assert result.rows_filtered == 1
    assert "answer 1" not in result.output_path.read_text(encoding="utf-8")

    unfiltered = render_calibration_text(dataset, tmp_path / "all.txt", tokenizer=FakeChatTokenizer())
    assert unfiltered.rows_used == 3


def test_raises_when_nothing_renders(tmp_path):
    dataset = _write_jsonl(tmp_path / "d.jsonl", [{"unrelated": 1}])
    with pytest.raises(ValueError, match="No calibration rows"):
        render_calibration_text(dataset, tmp_path / "cal.txt", tokenizer=None)


def test_manifest_keys_are_stable(tmp_path):
    dataset = _write_jsonl(tmp_path / "d.jsonl", [_chat_row(0)])
    result = render_calibration_text(
        dataset, tmp_path / "cal.txt", tokenizer=None, source="hf://datasets/o/r/train.jsonl"
    )
    assert set(result.to_manifest()) == {
        "source", "text_sha256", "render_mode", "rows_total", "rows_used",
        "rows_skipped", "rows_filtered", "chars", "seed", "max_rows",
        "max_chars", "label_field",
    }
    assert result.to_manifest()["source"] == "hf://datasets/o/r/train.jsonl"


def test_spec_validation():
    with pytest.raises(ValueError):
        CalibrationSpec(dataset_path=Path("x.jsonl"), max_rows=0)
    with pytest.raises(ValueError):
        CalibrationSpec(dataset_path=Path("x.jsonl"), chunks=0)
    assert CalibrationSpec(dataset_path=Path("x.jsonl"), chunks=None).source_label == "x.jsonl"
