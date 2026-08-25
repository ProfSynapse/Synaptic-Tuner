from __future__ import annotations

import pytest

from Trainers.sft.v1_runtime.contracts import MASK_CONTRACT_V1, RuntimeContractError
from Trainers.sft.v1_runtime.preprocessing import materialize_conversations_v1


class PrefixTokenizer:
    chat_template = "embedded"
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is False
        values = []
        for index, message in enumerate(messages, 1):
            values.extend((index * 10, index * 10 + len(message["content"])))
        return values


ROWS = [{"conversations": [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "answer"},
]}]


def materialize(*, scope="assistant_messages", length=8, rows=ROWS, tokenizer=None):
    return materialize_conversations_v1(
        rows,
        tokenizer=tokenizer or PrefixTokenizer(),
        max_seq_length=length,
        loss_scope=scope,
        masking_contract=MASK_CONTRACT_V1,
    )


def test_assistant_mask_and_padding_are_deterministic() -> None:
    row = materialize()[0]
    assert row.input_ids == (10, 11, 20, 25, 30, 36, 0, 0)
    assert row.attention_mask == (1, 1, 1, 1, 1, 1, 0, 0)
    assert row.labels == (-100, -100, -100, -100, 30, 36, -100, -100)


def test_full_sequence_supervises_only_nonpadding() -> None:
    row = materialize(scope="full_sequence")[0]
    assert row.labels == (10, 11, 20, 25, 30, 36, -100, -100)


def test_right_truncation_preserves_prefix() -> None:
    row = materialize(scope="full_sequence", length=3)[0]
    assert row.input_ids == (10, 11, 20)
    assert row.attention_mask == (1, 1, 1)


def test_right_truncation_that_removes_assistant_supervision_is_rejected() -> None:
    with pytest.raises(RuntimeContractError, match="removed every supervised token"):
        materialize(length=4)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"text": "not conversations"}],
        [{"conversations": []}],
        [{"conversations": [{"role": "user", "content": "only user"}]}],
        [{"conversations": [{"role": "alien", "content": "x"}, {"role": "assistant", "content": "y"}]}],
        [{"conversations": [{"role": "assistant", "content": "x", "extra": 1}]}],
    ],
)
def test_malformed_rows_are_rejected_without_dropping(rows) -> None:
    with pytest.raises(RuntimeContractError):
        materialize(rows=rows)


def test_missing_embedded_template_is_rejected() -> None:
    tokenizer = PrefixTokenizer()
    tokenizer.chat_template = ""
    with pytest.raises(RuntimeContractError, match="embedded chat template"):
        materialize(tokenizer=tokenizer)


def test_non_prefix_template_is_rejected() -> None:
    class BadTokenizer(PrefixTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [len(messages)]

    with pytest.raises(RuntimeContractError, match="conversation-prefix"):
        materialize(tokenizer=BadTokenizer())

