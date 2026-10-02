"""Label fixes in SFT preprocessing (one test group per fixed bug).

* prompt_completion honours ``completion_only_loss`` (assistant_only_loss=False
  trains every token).
* The default render masks the tokens the template emits after the final
  assistant turn's end-of-turn token (e.g. ``<|im_end|>`` then ``\\n``).
* Rows with no supervised tokens left (truncation) are dropped with a logged
  count on every render path; the run fails above
  ``training.max_dropped_row_fraction``.
* Rows whose assistant-only mask stopped before the end of the prompt render are
  dropped under the same policy instead of training prompt tokens.

Offline fake tokenizers only; files go under ``scratch/``.
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

pytest.importorskip("datasets")

from datasets import Dataset  # noqa: E402

from shared.sft_preprocessing import (  # noqa: E402
    DEFAULT_MAX_DROPPED_ROW_FRACTION,
    cached_end_of_turn_tokens,
    materialize_sft_example,
)
from tests.trainers._trainer_import import load_trainer_module  # noqa: E402

preprocessing = load_trainer_module("sft", "preprocessing")


class _AddedToken:
    def __init__(self, content: str):
        self.content = content
        self.special = True


class ChatTokenizer:
    """ChatML-like offline tokenizer: char ids plus a few special tokens.

    ``eos_token`` defaults to ``<eos>``, which is NOT the template's end-of-turn
    token ``<|im_end|>`` (the case where eos and end-of-turn differ).
    """

    SPECIAL = {"<|im_start|>": 1000, "<|im_end|>": 1001, "<eos>": 1002}

    def __init__(self, *, eos_token: str = "<eos>", turn_end: str = "<|im_end|>\n",
                 diverge_early: bool = False):
        self.eos_token = eos_token
        self.eos_token_id = self.SPECIAL[eos_token]
        self.turn_end = turn_end
        self.diverge_early = diverge_early
        self.all_special_ids = list(self.SPECIAL.values())
        self.added_tokens_decoder = {i: _AddedToken(t) for t, i in self.SPECIAL.items()}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        header = "D2\n" if (self.diverge_early and add_generation_prompt) else "D1\n"
        text = header + "".join(
            f"<|im_start|>{m['role']}\n{m['content']}{self.turn_end}" for m in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        ids, position = [], 0
        while position < len(text):
            for token, token_id in self.SPECIAL.items():
                if text.startswith(token, position):
                    ids.append(token_id)
                    position += len(token)
                    break
            else:
                ids.append(ord(text[position]))
                position += 1
        return ids

    def decode(self, ids):
        reverse = {v: k for k, v in self.SPECIAL.items()}
        return "".join(reverse.get(i, chr(i)) for i in ids)


IM_END = ChatTokenizer.SPECIAL["<|im_end|>"]
EOS = ChatTokenizer.SPECIAL["<eos>"]


def _row(*turns, **extra):
    roles = ["user", "assistant"] * len(turns)
    return {"messages": [{"role": r, "content": c} for r, c in zip(roles, turns)], **extra}


def _trained(prepared):
    return [token for token, label in zip(prepared.input_ids, prepared.labels) if label != -100]


# ---------------------------------------------------------------------------
# prompt_completion honours completion_only_loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("assistant_only_loss", [True, False])
def test_prompt_completion_honours_completion_only_loss(assistant_only_loss):
    tokenizer = ChatTokenizer(eos_token="<|im_end|>")
    prepared = materialize_sft_example(
        tokenizer=tokenizer,
        record=_row("hi", "hello"),
        max_seq_length=512,
        assistant_only_loss=assistant_only_loss,
        prompt_render="prompt_completion",
    )
    if assistant_only_loss:
        assert prepared.loss_mask_mode == "assistant_only"
        assert tokenizer.decode(_trained(prepared)) == "hello<|im_end|>"
    else:
        assert prepared.loss_mask_mode == "full_sequence"
        assert prepared.labels == prepared.input_ids
        assert prepared.masked_prefix_length == 0


# ---------------------------------------------------------------------------
# prompt_completion closes with the template's end-of-turn token
# ---------------------------------------------------------------------------


def _prompt_completion(tokenizer, assistant_only_loss=True):
    return materialize_sft_example(
        tokenizer=tokenizer,
        record=_row("hi", "hello"),
        max_seq_length=512,
        assistant_only_loss=assistant_only_loss,
        prompt_render="prompt_completion",
    )


def test_prompt_completion_is_byte_identical_when_eos_is_end_of_turn():
    # eos == <|im_end|>: the historical construction was prompt render + raw
    # completion + eos_token_id; the template-derived terminal is the same id.
    tokenizer = ChatTokenizer(eos_token="<|im_end|>")
    prompt_ids = tokenizer.encode(
        tokenizer.apply_chat_template(_row("hi")["messages"][:1], add_generation_prompt=True)
    )
    historical_ids = prompt_ids + tokenizer.encode("hello") + [tokenizer.eos_token_id]
    historical_labels = [-100] * len(prompt_ids) + tokenizer.encode("hello") + [tokenizer.eos_token_id]

    prepared = _prompt_completion(tokenizer)
    assert prepared.input_ids == historical_ids
    assert prepared.labels == historical_labels


def test_prompt_completion_uses_end_of_turn_not_eos_when_they_differ():
    tokenizer = ChatTokenizer(eos_token="<eos>")
    prepared = _prompt_completion(tokenizer)
    assert prepared.input_ids[-1] == IM_END
    assert EOS not in prepared.input_ids
    assert tokenizer.decode(_trained(prepared)) == "hello<|im_end|>"


def test_prompt_completion_falls_back_to_eos_and_logs_once(capsys):
    tokenizer = ChatTokenizer(eos_token="<eos>", turn_end="\n")  # renders no terminator
    first = _prompt_completion(tokenizer)
    second = _prompt_completion(tokenizer)
    assert first.input_ids[-1] == EOS and second.input_ids[-1] == EOS
    out = capsys.readouterr().out
    assert out.count("closing completions with eos_token_id") == 1


# ---------------------------------------------------------------------------
# Default render: nothing after the final end-of-turn token is trained
# ---------------------------------------------------------------------------


def test_default_render_stops_labels_at_end_of_turn_when_eos_differs():
    tokenizer = ChatTokenizer()  # eos <eos> != end-of-turn <|im_end|>
    assert cached_end_of_turn_tokens(tokenizer).token_ids == [IM_END]
    prepared = materialize_sft_example(
        tokenizer=tokenizer, record=_row("hi", "hello"), max_seq_length=512,
        assistant_only_loss=True,
    )
    assert tokenizer.decode(_trained(prepared)) == "hello<|im_end|>"
    assert prepared.input_ids[-1] == ord("\n") and prepared.labels[-1] == -100


def test_default_render_uses_only_the_final_turn_terminator():
    tokenizer = ChatTokenizer()
    prepared = materialize_sft_example(
        tokenizer=tokenizer, record=_row("q1", "a1", "q2", "a2"), max_seq_length=512,
        assistant_only_loss=True,
    )
    assert tokenizer.decode(_trained(prepared)) == "a2<|im_end|>"

    # When truncation cuts the final turn's terminator, the earlier turn's
    # terminator (inside the prompt) must not be used to mask the kept reply.
    full = materialize_sft_example(
        tokenizer=tokenizer, record=_row("q1", "a1", "q2", "a long reply"),
        max_seq_length=10**6, assistant_only_loss=True,
    )
    cut = materialize_sft_example(
        tokenizer=tokenizer, record=_row("q1", "a1", "q2", "a long reply"),
        max_seq_length=len(full.input_ids) - 4, assistant_only_loss=True,
    )
    assert tokenizer.decode(_trained(cut)) == "a long rep"


def test_default_render_without_template_terminator_keeps_trailing_tokens():
    # The template closes turns with a bare newline: nothing derivable to stop
    # at, so labels are unchanged (the eos_token fallback is not rendered).
    tokenizer = ChatTokenizer(turn_end="\n")
    prepared = materialize_sft_example(
        tokenizer=tokenizer, record=_row("hi", "hello"), max_seq_length=512,
        assistant_only_loss=True,
    )
    assert cached_end_of_turn_tokens(tokenizer).source == "eos_token"
    assert tokenizer.decode(_trained(prepared)) == "hello\n"


def test_full_sequence_loss_is_unchanged():
    tokenizer = ChatTokenizer()
    prepared = materialize_sft_example(
        tokenizer=tokenizer, record=_row("hi", "hello"), max_seq_length=512,
        assistant_only_loss=False,
    )
    assert prepared.labels == prepared.input_ids


# ---------------------------------------------------------------------------
# Untrainable rows: dropped with a count, run fails above the threshold
# ---------------------------------------------------------------------------


def _rows_with_one_cut(n_good: int):
    # The first row's prompt alone exceeds max_seq_length=40, so truncation
    # removes every supervised token.
    return [_row("x" * 60, "y")] + [_row("hi", f"answer {i}") for i in range(n_good)]


@pytest.mark.parametrize("prompt_render", ["full_conversation", "prompt_completion"])
def test_unsupervised_rows_are_dropped_on_every_render_path(prompt_render, capsys):
    prepared = preprocessing.prepare_sft_dataset(
        Dataset.from_list(_rows_with_one_cut(9)),
        tokenizer=ChatTokenizer(eos_token="<|im_end|>"),
        max_seq_length=40,
        prompt_render=prompt_render,
        max_dropped_row_fraction=0.2,
    )
    assert len(prepared) == 9
    assert prepared.column_names == ["input_ids", "attention_mask", "labels"]
    assert all(any(label != -100 for label in row) for row in prepared["labels"])
    out = capsys.readouterr().out
    assert "dropped 1/10 rows (no_supervised_tokens=1, mask_prefix_mismatch=0)" in out


@pytest.mark.parametrize("prompt_render", ["full_conversation", "prompt_completion"])
def test_unsupervised_rows_above_threshold_fail_the_run(prompt_render):
    assert DEFAULT_MAX_DROPPED_ROW_FRACTION == 0.01
    with pytest.raises(preprocessing.DroppedRowsError, match="no_supervised_tokens=1"):
        preprocessing.prepare_sft_dataset(
            Dataset.from_list(_rows_with_one_cut(9)),
            tokenizer=ChatTokenizer(eos_token="<|im_end|>"),
            max_seq_length=40,
            prompt_render=prompt_render,
        )


def test_invalid_threshold_is_refused():
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        preprocessing.prepare_sft_dataset(
            Dataset.from_list([_row("hi", "hello")]),
            tokenizer=ChatTokenizer(), max_seq_length=64, max_dropped_row_fraction=1.5,
        )


def test_prefix_mismatch_rows_are_dropped_not_trained():
    tokenizer = ChatTokenizer(diverge_early=True)
    prepared = materialize_sft_example(
        tokenizer=tokenizer, record=_row("hi", "hello"), max_seq_length=512,
        assistant_only_loss=True,
    )
    assert prepared.drop_reason == "mask_prefix_mismatch"
    with pytest.raises(preprocessing.DroppedRowsError, match="mask_prefix_mismatch=2"):
        preprocessing.prepare_sft_dataset(
            Dataset.from_list([_row("hi", "hello"), _row("q", "a")]),
            tokenizer=tokenizer, max_seq_length=512,
        )
    kept = preprocessing.prepare_sft_dataset(
        Dataset.from_list([_row("hi", "hello"), _row("q", "a")]),
        tokenizer=tokenizer, max_seq_length=512, max_dropped_row_fraction=1.0,
    )
    assert len(kept) == 0


def test_threshold_is_a_declared_trainer_config_key():
    import importlib.util
    import yaml

    spec = importlib.util.spec_from_file_location(
        "label_fix_sft_config_loader", ROOT / "Trainers/sft/configs/config_loader.py"
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    assert loader.load_config().training.max_dropped_row_fraction == DEFAULT_MAX_DROPPED_ROW_FRACTION

    config = yaml.safe_load((ROOT / "Trainers/sft/configs/config.yaml").read_text())
    config["training"]["max_dropped_row_fraction"] = 0.05
    path = ROOT / "scratch" / "tests" / f"label_fix_{uuid.uuid4().hex[:8]}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        assert loader.load_config(str(path)).training.max_dropped_row_fraction == 0.05
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Grouped validation split stays aligned after drops
# ---------------------------------------------------------------------------


@pytest.fixture
def scratch_dir():
    path = ROOT / "scratch" / "tests" / f"label_fix_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_grouped_split_group_values_follow_dropped_rows(scratch_dir):
    data_loader = load_trainer_module("sft", "data_loader")

    tokenizer = ChatTokenizer(eos_token="<|im_end|>")
    rows = [_row("x" * 60, "drop me", group="A")]
    for group in "ABCDEFGH":
        rows += [_row("hi", f"g{group}1", group=group), _row("hi", f"g{group}2", group=group)]
    path = scratch_dir / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    train, validation = data_loader.load_and_prepare_tokenized_dataset(
        local_file=str(path), tokenizer=tokenizer, max_seq_length=40,
        split_dataset=True, test_size=0.25, validation_group_key="group",
        max_dropped_row_fraction=0.1,
    )
    assert len(train) + len(validation) == 16

    def groups(dataset):
        return {
            tokenizer.decode(_trained_ids(row))[1]
            for row in dataset
        }

    assert groups(train).isdisjoint(groups(validation))


def _trained_ids(row):
    return [token for token, label in zip(row["input_ids"], row["labels"]) if label != -100]
