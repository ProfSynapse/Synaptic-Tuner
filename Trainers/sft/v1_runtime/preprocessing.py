"""Pure conversation materialization for the SFT v1 masking contract."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import MASK_CONTRACT_V1, RuntimeContractError

_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(frozen=True, slots=True)
class MaterializedConversation:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]


def _tokens(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[int, ...]:
    values = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeContractError("embedded chat template must return token ids")
    result = tuple(values)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in result):
        raise RuntimeContractError("embedded chat template returned an invalid token id")
    return result


def materialize_conversations_v1(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_seq_length: int,
    loss_scope: str,
    masking_contract: str,
) -> tuple[MaterializedConversation, ...]:
    if masking_contract != MASK_CONTRACT_V1:
        raise RuntimeContractError("unsupported masking contract")
    if loss_scope not in {"assistant_messages", "full_sequence"}:
        raise RuntimeContractError("unsupported loss scope")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise RuntimeContractError("dataset rows must be a non-empty sequence")
    if not isinstance(max_seq_length, int) or isinstance(max_seq_length, bool) or max_seq_length <= 0:
        raise RuntimeContractError("max sequence length must be positive")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise RuntimeContractError("tokenizer embedded chat template is required")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_id, int) or isinstance(pad_id, bool) or pad_id < 0:
        raise RuntimeContractError("tokenizer pad token id is required")

    output: list[MaterializedConversation] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"conversations"}:
            raise RuntimeContractError("row must contain exactly conversations")
        raw_messages = row["conversations"]
        if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)) or not raw_messages:
            raise RuntimeContractError("conversations must be non-empty")
        messages: list[dict[str, str]] = []
        assistant_seen = False
        for raw in raw_messages:
            if not isinstance(raw, Mapping) or set(raw) != {"role", "content"}:
                raise RuntimeContractError("message must contain exactly role and content")
            role, content = raw["role"], raw["content"]
            if role not in _ROLES or not isinstance(content, str):
                raise RuntimeContractError("message role or content is invalid")
            assistant_seen = assistant_seen or role == "assistant"
            messages.append({"role": role, "content": content})
        if not assistant_seen:
            raise RuntimeContractError("conversation requires an assistant message")

        previous: tuple[int, ...] = ()
        labels: list[int] = []
        for index, message in enumerate(messages):
            current = _tokens(tokenizer, messages[: index + 1])
            if current[: len(previous)] != previous:
                raise RuntimeContractError("chat template violates conversation-prefix-v1")
            appended = current[len(previous):]
            labels.extend(
                appended
                if loss_scope == "full_sequence" or message["role"] == "assistant"
                else [-100] * len(appended)
            )
            previous = current

        input_ids = list(previous[:max_seq_length])
        labels = labels[:max_seq_length]
        if not input_ids or not any(label != -100 for label in labels):
            raise RuntimeContractError("right truncation removed every supervised token")
        real = len(input_ids)
        padding = max_seq_length - real
        output.append(MaterializedConversation(
            tuple(input_ids + [pad_id] * padding),
            tuple([1] * real + [0] * padding),
            tuple(labels + [-100] * padding),
        ))
    return tuple(output)
