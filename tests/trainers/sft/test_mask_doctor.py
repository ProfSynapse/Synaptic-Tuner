"""Tests for the SFT loss-mask doctor (``python tuner.py doctor sft-mask``).

The doctor runs rows through the real SFT preprocessing hop
(``preprocessing.materialize_sft_row``) and reads the diagnostic fields
``shared.sft_preprocessing.materialize_sft_example`` records. These tests use a
small offline ChatML-style tokenizer (character-level ids plus a few special
tokens, in the style of the other SFT preprocessing tests) and variants of its
template that break masking on purpose. Files go under ``scratch/``.
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

pytest.importorskip("datasets")

import mask_doctor  # noqa: E402
import preprocessing  # noqa: E402
from shared.sft_preprocessing import (  # noqa: E402
    derive_end_of_turn_tokens,
    materialize_sft_example,
)


class _AddedToken:
    def __init__(self, content: str, special: bool = True):
        self.content = content
        self.special = special


class FakeChatTokenizer:
    """Offline ChatML-like tokenizer with configurable template quirks."""

    SPECIAL = {"<|im_start|>": 1000, "<|im_end|>": 1001, "<eos>": 1002, "<bos>": 1003}

    def __init__(
        self,
        *,
        eos_token: str = "<|im_end|>",
        bos_prefix: str = "<bos>",
        diverge_early: bool = False,
        turn_end: str = "<|im_end|>\n",
    ):
        self.eos_token = eos_token
        self.eos_token_id = self.SPECIAL[eos_token]
        self.bos_token = "<bos>"
        self.bos_token_id = self.SPECIAL["<bos>"]
        self.bos_prefix = bos_prefix
        self.diverge_early = diverge_early
        self.turn_end = turn_end
        self.chat_template = "fake"
        self.all_special_ids = list(self.SPECIAL.values())
        self.added_tokens_decoder = {i: _AddedToken(t) for t, i in self.SPECIAL.items()}
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        self.calls.append(dict(kwargs))
        # Simulated date injection: the header differs between the prompt render
        # and the full render, so the two diverge a few tokens in.
        header = "Date: 02\n" if (self.diverge_early and add_generation_prompt) else "Date: 01\n"
        text = self.bos_prefix + header
        for message in messages:
            text += f"<|im_start|>{message['role']}\n{message['content']}{self.turn_end}"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
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


def _row(*turns):
    roles = ["user", "assistant"] * len(turns)
    return {"messages": [{"role": r, "content": c} for r, c in zip(roles, turns)]}


def _settings(**overrides):
    values = dict(
        model_name="fake",
        max_seq_length=512,
        loss_mask_mode="assistant_only",
        prompt_render="full_conversation",
        sample_size=0,
        preview_count=1,
    )
    values.update(overrides)
    return mask_doctor.MaskDoctorSettings(**values)


def _doctor_config():
    return mask_doctor.load_doctor_config()


def _diagnose(rows, tokenizer=None, **settings):
    return mask_doctor.diagnose_rows(
        list(enumerate(rows)),
        tokenizer=tokenizer or FakeChatTokenizer(),
        settings=_settings(**settings),
        doctor_config=_doctor_config(),
    )


def _check(report, name):
    return next(check for check in report["checks"] if check["name"] == name)


# ---------------------------------------------------------------------------
# Preprocessing diagnostics
# ---------------------------------------------------------------------------


def test_diagnostic_fields_on_clean_row():
    tokenizer = FakeChatTokenizer()
    prepared = materialize_sft_example(
        tokenizer=tokenizer, record=_row("hi", "hello"), max_seq_length=512, assistant_only_loss=True
    )
    prompt = tokenizer.encode(
        tokenizer.apply_chat_template(_row("hi")["messages"][:1], add_generation_prompt=True)
    )
    assert prepared.labels[: len(prompt)] == [-100] * len(prompt)
    # The reply through its end-of-turn token is trained; the template's
    # trailing newline after <|im_end|> is not.
    assert prepared.labels[len(prompt):-1] == prepared.input_ids[len(prompt):-1]
    assert prepared.input_ids[-2:] == [FakeChatTokenizer.SPECIAL["<|im_end|>"], ord("\n")]
    assert prepared.labels[-1] == -100
    assert prepared.drop_reason is None
    assert prepared.prompt_token_count == len(prompt)
    assert prepared.masked_prefix_length == len(prompt)
    assert prepared.mask_prefix_mismatch is False
    assert prepared.mask_fallback_reason is None
    assert prepared.untruncated_length == len(prepared.input_ids)


def test_diagnostic_fields_record_early_divergence_and_fallback():
    diverging = materialize_sft_example(
        tokenizer=FakeChatTokenizer(diverge_early=True),
        record=_row("hi", "hello"),
        max_seq_length=512,
        assistant_only_loss=True,
    )
    assert diverging.mask_prefix_mismatch is True
    assert diverging.masked_prefix_length == 1 + len("Date: 0")
    assert diverging.mask_divergence_expected_token == ord("2")
    assert diverging.drop_reason == "mask_prefix_mismatch"

    fallback = materialize_sft_example(
        tokenizer=FakeChatTokenizer(),
        record={"messages": [{"role": "user", "content": "only a question"}]},
        max_seq_length=512,
        assistant_only_loss=True,
    )
    assert fallback.loss_mask_mode == "full_sequence"
    assert fallback.mask_fallback_reason == "final_message_not_assistant"
    assert fallback.labels == fallback.input_ids


# ---------------------------------------------------------------------------
# End-of-turn derivation (shared with the trainer)
# ---------------------------------------------------------------------------


def test_end_of_turn_is_derived_from_the_template_not_eos():
    tokenizer = FakeChatTokenizer(eos_token="<eos>")
    spec = derive_end_of_turn_tokens(tokenizer)
    assert spec.token_ids == [FakeChatTokenizer.SPECIAL["<|im_end|>"]]
    assert spec.source == "chat_template"


def test_end_of_turn_falls_back_to_eos_when_template_emits_nothing():
    tokenizer = FakeChatTokenizer(eos_token="<eos>", turn_end="\n")
    spec = derive_end_of_turn_tokens(tokenizer)
    assert spec.token_ids == [FakeChatTokenizer.SPECIAL["<eos>"]]
    assert spec.source == "eos_token"


def test_end_of_turn_probe_receives_chat_template_kwargs():
    tokenizer = FakeChatTokenizer()
    derive_end_of_turn_tokens(tokenizer, chat_template_kwargs={"enable_thinking": False})
    assert tokenizer.calls[-1] == {"enable_thinking": False}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def test_clean_dataset_passes():
    report = _diagnose([_row("hi", "hello"), _row("what?", "this")])
    assert report["success"] is True
    assert all(check["status"] in ("ok", "info") for check in report["checks"])
    assert report["summary"]["rows_analyzed"] == 2


def test_early_divergence_is_reported_and_counted_as_a_trainer_drop():
    report = _diagnose([_row("hi", "hello")], tokenizer=FakeChatTokenizer(diverge_early=True))
    check = _check(report, "mask_prefix_mismatch")
    assert check["status"] == "warn" and check["count"] == 1 and check["rows"] == [0]
    divergence = report["mask_divergences"][0]
    assert divergence["input_token"] == "1"
    assert divergence["expected_prompt_token"] == "2"
    assert divergence["prompt_tokens_trained"] > 0
    dropped = _check(report, "dropped_rows")
    assert dropped["status"] == "fail" and dropped["by_reason"]["mask_prefix_mismatch"] == 1
    assert report["success"] is False


def test_zero_trained_tokens_counts_toward_the_dropped_row_threshold():
    # max_seq_length cuts inside the prompt: every kept position is masked.
    long_row = _row("a long question " * 4, "x")
    rows = [long_row] + [_row("hi", f"answer {index}") for index in range(9)]
    report = _diagnose(rows, max_seq_length=40)
    assert _check(report, "zero_trained_tokens")["count"] == 1
    dropped = _check(report, "dropped_rows")
    assert dropped["count"] == 1 and dropped["fraction"] == 0.1
    assert dropped["status"] == "fail"  # 10% > default 1%
    report = _diagnose(rows, max_seq_length=40, max_dropped_row_fraction=0.2)
    assert _check(report, "dropped_rows")["status"] == "warn"
    assert report["success"] is True


def test_unexpected_full_sequence_fallback_is_reported():
    rows = [{"messages": [{"role": "user", "content": "no answer"}]}]
    report = _diagnose(rows)
    assert _check(report, "full_sequence_fallback")["status"] == "fail"
    # Requested full-sequence loss is not a fallback.
    report = _diagnose(rows, loss_mask_mode="full_sequence")
    assert _check(report, "full_sequence_fallback")["count"] == 0


def test_prompt_completion_closes_with_template_end_of_turn_even_when_eos_differs():
    # The prompt_completion path closes the completion with the template's
    # end-of-turn token, so an eos that differs from it no longer matters.
    for eos_token in ("<eos>", "<|im_end|>"):
        report = _diagnose(
            [_row("hi", "hello")],
            tokenizer=FakeChatTokenizer(eos_token=eos_token),
            prompt_render="prompt_completion",
        )
        assert _check(report, "missing_end_of_turn")["count"] == 0
        assert report["success"] is True


def test_wrong_trained_terminator_is_a_hard_failure(monkeypatch):
    # Any trained span that ends with a token other than the derived end-of-turn
    # is reported (simulated here; the real preprocessing paths now close turns
    # with the template's terminator).
    from shared.sft_preprocessing import PreparedSFTExample

    tokenizer = FakeChatTokenizer()
    ids = tokenizer.encode("<|im_start|>assistant\nhello<eos>")

    def fake_row(row, **_kwargs):
        return PreparedSFTExample(
            input_ids=ids, attention_mask=[1] * len(ids), labels=[-100] * 3 + ids[3:],
            example_format="messages", loss_mask_mode="assistant_only",
            truncation_applied=False, untruncated_length=len(ids),
        )

    monkeypatch.setattr(mask_doctor, "materialize_sft_row", fake_row)
    report = _diagnose([_row("hi", "hello")], tokenizer=tokenizer)
    assert _check(report, "missing_end_of_turn")["status"] == "fail"
    assert report["end_of_turn_mismatches"] == [{"index": 0, "trained_span_ends_with": "<eos>"}]
    assert report["success"] is False


def test_doubled_bos_is_a_hard_failure():
    report = _diagnose([_row("hi", "hello")], tokenizer=FakeChatTokenizer(bos_prefix="<bos><bos>"))
    assert _check(report, "doubled_bos")["status"] == "fail"


def test_truncation_rate_and_lengths():
    rows = [_row("hi", "hello"), _row("q", "y" * 300)]
    report = _diagnose(rows, max_seq_length=100)
    truncation = _check(report, "truncation")
    assert truncation["count"] == 1 and truncation["rate"] == 0.5
    assert truncation["status"] == "warn"
    # The end-of-turn of the truncated row was cut: warning, not a hard failure.
    assert _check(report, "terminator_lost_to_truncation")["rows"] == [1]
    assert _check(report, "missing_end_of_turn")["count"] == 0
    lengths = report["lengths"]
    assert lengths["max"] > 100 and lengths["max_seq_length"] == 100
    assert lengths["p50"] <= lengths["p95"] <= lengths["max"]
    assert report["success"] is True


def test_multi_turn_untrained_earlier_assistant_turns_are_counted():
    report = _diagnose([_row("q1", "a1", "q2", "a2", "q3", "a3"), _row("q", "a")])
    check = _check(report, "earlier_assistant_turns_untrained")
    assert check["status"] == "info"
    assert check["count"] == 1 and check["untrained_turns"] == 2
    assert report["success"] is True


def test_row_error_is_reported_as_failure():
    report = _diagnose([{"prompt": None, "completion": None, "text": "x"}])
    assert _check(report, "row_error")["status"] == "fail"
    assert "ValueError" in report["row_errors"][0]["error"]


def test_preview_marks_masked_and_trained_tokens_and_divergence():
    report = _diagnose([_row("hi", "hello")], tokenizer=FakeChatTokenizer(diverge_early=True))
    preview = report["previews"][0]
    assert "MASK " in preview["text"] and "TRAIN" in preview["text"]
    assert "mask stopped at position" in preview["text"]
    assert any(token["trained"] for token in preview["tokens"])
    assert any(not token["trained"] for token in preview["tokens"])
    text = mask_doctor.format_report(report)
    assert "[WARN] mask_prefix_mismatch: 1" in text
    assert "[FAIL] dropped_rows: 1" in text
    assert "Result: FAIL" in text and "dropped_rows" in text


def test_materialize_sft_row_is_the_prepare_sft_dataset_path():
    from datasets import Dataset

    tokenizer = FakeChatTokenizer()
    rows = [_row("hi", "hello"), {"messages": [{"role": "user", "content": "x"}]}]
    prepared = preprocessing.prepare_sft_dataset(
        Dataset.from_list(rows), tokenizer=tokenizer, max_seq_length=512
    )
    assert prepared.column_names == ["input_ids", "attention_mask", "labels"]
    for index, row in enumerate(rows):
        single = preprocessing.materialize_sft_row(row, tokenizer=tokenizer, max_seq_length=512)
        assert prepared[index]["labels"] == single.labels


# ---------------------------------------------------------------------------
# Data loader + CLI handler (files under scratch/)
# ---------------------------------------------------------------------------


@pytest.fixture
def scratch_dir():
    path = ROOT / "scratch" / "tests" / f"mask_doctor_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_data_loader_refuses_prefix_mismatch_rows(scratch_dir, capsys):
    import data_loader

    dataset_path = _write_jsonl(scratch_dir / "rows.jsonl", [_row("hi", "hello"), _row("q", "a")])
    with pytest.raises(preprocessing.DroppedRowsError, match="mask_prefix_mismatch=2"):
        data_loader.load_and_prepare_tokenized_dataset(
            local_file=str(dataset_path),
            tokenizer=FakeChatTokenizer(diverge_early=True),
            max_seq_length=512,
        )
    assert "dropped 2/2 rows" in capsys.readouterr().out


def _handler_args(dataset: Path | None, **overrides) -> Namespace:
    values = dict(
        command="doctor",
        subcommand="sft-mask",
        json=True,
        doctor_fix=False,
        sft_config=None,
        model="fake-tokenizer",
        dataset_path=str(dataset) if dataset else None,
        max_seq_length=None,
        chat_template_kwargs=None,
        prompt_render=None,
        no_completion_only=False,
        sample_size=None,
        seed=None,
        preview_rows=None,
        preview_count=None,
        tokenizer_revision=None,
        trust_remote_code=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_handler_json_output_and_exit_codes(scratch_dir, capsys):
    from tuner.handlers.sft_mask_doctor_handler import SFTMaskDoctorHandler

    dataset_path = _write_jsonl(scratch_dir / "rows.jsonl", [_row("hi", "hello"), _row("q", "a")])

    clean = SFTMaskDoctorHandler(
        args=_handler_args(dataset_path, chat_template_kwargs='{"enable_thinking": false}'),
        tokenizer=FakeChatTokenizer(),
    )
    assert clean.handle() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["success"] is True and report["exit_code"] == 0
    assert report["settings"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert report["settings"]["loss_mask_mode"] == "assistant_only"

    broken = SFTMaskDoctorHandler(
        args=_handler_args(dataset_path, preview_rows="1"),
        tokenizer=FakeChatTokenizer(diverge_early=True),
    )
    assert broken.handle() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["exit_code"] == 1
    assert [preview["index"] for preview in report["previews"]] == [1]


def test_handler_reuses_trainer_config(scratch_dir, capsys):
    import yaml
    from tuner.handlers.sft_mask_doctor_handler import SFTMaskDoctorHandler

    dataset_path = _write_jsonl(scratch_dir / "rows.jsonl", [_row("q", "y" * 200)])
    config = yaml.safe_load((ROOT / "Trainers" / "sft" / "configs" / "config.yaml").read_text())
    config["model"]["model_name"] = "fake-from-config"
    config["training"]["max_seq_length"] = 64
    config["training"]["prompt_render"] = "prompt_completion"
    config["training"]["chat_template_kwargs"] = {"enable_thinking": False}
    config["dataset"]["local_file"] = str(dataset_path)
    config_path = scratch_dir / "sft_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    handler = SFTMaskDoctorHandler(
        args=_handler_args(None, model=None, sft_config=str(config_path)),
        tokenizer=FakeChatTokenizer(),
    )
    assert handler.handle() == 0
    settings = json.loads(capsys.readouterr().out)["settings"]
    assert settings["model_name"] == "fake-from-config"
    assert settings["max_seq_length"] == 64
    assert settings["prompt_render"] == "prompt_completion"
    assert settings["chat_template_kwargs"] == {"enable_thinking": False}
    assert settings["sources"] == [f"trainer config {config_path}"]

    # Explicit flags override the trainer config, as train_sft.py CLI flags do.
    handler = SFTMaskDoctorHandler(
        args=_handler_args(None, model="flag-model", sft_config=str(config_path), max_seq_length=128),
        tokenizer=FakeChatTokenizer(),
    )
    assert handler.handle() == 0
    settings = json.loads(capsys.readouterr().out)["settings"]
    assert settings["model_name"] == "flag-model" and settings["max_seq_length"] == 128


RAW_TEXT_ROW = {
    "schema_version": "syntunia-sft-row/v1",
    "format": "raw_text",
    "text": "plain pretraining-style text",
    "split": "train",
}


def test_dataset_contract_failure_is_reported(scratch_dir):
    dataset_path = _write_jsonl(scratch_dir / "raw.jsonl", [RAW_TEXT_ROW])
    # The trainer rejects raw_text rows without preassigned splits and with
    # assistant-only loss; the doctor runs the same contract check.
    report = mask_doctor.run_mask_doctor(
        tokenizer=FakeChatTokenizer(),
        settings=_settings(local_file=str(dataset_path), loss_mask_mode="full_sequence"),
        doctor_config=_doctor_config(),
    )
    contract = _check(report, "dataset_contract")
    assert contract["status"] == "fail"
    assert "use_preassigned_splits" in contract["error"]
    assert report["success"] is False
    assert "failed checks: dataset_contract" in mask_doctor.format_report(report)


def test_raw_text_rows_end_with_eos(scratch_dir):
    dataset_path = _write_jsonl(scratch_dir / "raw.jsonl", [RAW_TEXT_ROW])
    report = mask_doctor.run_mask_doctor(
        tokenizer=FakeChatTokenizer(eos_token="<eos>"),
        settings=_settings(
            local_file=str(dataset_path),
            loss_mask_mode="full_sequence",
            use_preassigned_splits=True,
        ),
        doctor_config=_doctor_config(),
    )
    assert report["success"] is True
    assert _check(report, "dataset_contract")["status"] == "ok"
    assert _check(report, "missing_end_of_turn")["count"] == 0


def test_authoritative_prompt_completion_truncation_is_a_row_error():
    row = {
        "schema_version": "syntunia-sft-row/v2",
        "format": "messages",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "y" * 300},
        ],
        "split": "train",
    }
    report = _diagnose([row], max_seq_length=50, prompt_render="prompt_completion")
    assert _check(report, "row_error")["status"] == "fail"
    assert "truncation is forbidden" in report["row_errors"][0]["error"]


def test_handler_setup_error_exit_code(capsys):
    from tuner.handlers.sft_mask_doctor_handler import SFTMaskDoctorHandler

    handler = SFTMaskDoctorHandler(
        args=_handler_args(ROOT / "scratch" / "does-not-exist.jsonl"),
        tokenizer=FakeChatTokenizer(),
    )
    assert handler.handle() == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "MASK_DOCTOR_SETUP"


def test_router_routes_doctor_sft_mask(monkeypatch):
    from tuner.cli import router
    from tuner.cli.parser import create_parser
    from tuner.handlers import sft_mask_doctor_handler

    seen = {}

    class _Stub:
        def __init__(self, args, context=None):
            seen["args"] = args
            seen["context"] = context

        def handle(self):
            return 7

    monkeypatch.setattr(sft_mask_doctor_handler, "SFTMaskDoctorHandler", _Stub)
    args = create_parser().parse_args(
        ["doctor", "sft-mask", "--dataset-path", "d.jsonl", "--model", "m", "--sample-size", "5",
         "--preview-rows", "1,2", "--chat-template-kwargs", "{}", "--prompt-render", "prompt_completion"]
    )
    assert router.route_command(args) == 7
    assert seen["args"].sample_size == 5 and seen["args"].max_seq_length is None
    assert seen["context"] is not None

    unknown = create_parser().parse_args(["doctor", "nonsense"])
    assert router.route_command(unknown) == 2


def test_doctor_config_refuses_unknown_keys(scratch_dir):
    from shared.training_utils import UnknownConfigKeysError

    config = mask_doctor.load_doctor_config()
    assert config.sample_size > 0

    bad = scratch_dir / "mask_doctor.yaml"
    bad.write_text("sample_sise: 10\nextra_end_of_turn_tokens: []\n", encoding="utf-8")
    with pytest.raises(UnknownConfigKeysError) as excinfo:
        mask_doctor.load_doctor_config(bad)
    assert excinfo.value.paths == ["sample_sise", "extra_end_of_turn_tokens"]
