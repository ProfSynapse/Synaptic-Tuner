from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


LossMaskMode = Literal["full_sequence", "assistant_only"]
ExampleFormat = Literal["messages", "prompt_completion"]


@dataclass
class PreparedSFTExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    example_format: ExampleFormat
    loss_mask_mode: LossMaskMode
    truncation_applied: bool
    source_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_jsonl_line(line: str) -> str:
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()[:8]


def render_tool_call_content(tool_calls: list[dict[str, Any]]) -> str:
    """Render OpenAI-style tool calls into the repo's ChatML-style text format."""
    rendered_parts: list[str] = []
    for tool_call in tool_calls:
        function_payload = tool_call.get("function") or {}
        name = function_payload.get("name") or tool_call.get("name") or "unknown"
        arguments = function_payload.get("arguments", tool_call.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments_obj = json.loads(arguments)
            except json.JSONDecodeError:
                arguments_obj = arguments
        else:
            arguments_obj = arguments
        arguments_text = (
            json.dumps(arguments_obj, ensure_ascii=False, indent=2)
            if not isinstance(arguments_obj, str)
            else arguments_obj
        )
        rendered_parts.append(f"tool_call: {name}\narguments: {arguments_text}")
    return "\n\n".join(rendered_parts)


def sanitize_messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize nullable content and render tool calls into assistant text."""
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        content = normalized.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        tool_calls = normalized.get("tool_calls") or []
        if tool_calls:
            tool_content = render_tool_call_content(tool_calls)
            content = f"{content}\n\n{tool_content}".strip() if content else tool_content

        normalized["content"] = content
        normalized.pop("tool_calls", None)
        sanitized.append(normalized)
    return sanitized


def normalize_sft_messages(record: dict[str, Any]) -> tuple[list[dict[str, Any]], ExampleFormat]:
    """Convert repo-supported raw example shapes into canonical conversational messages."""
    if record.get("messages"):
        return list(record["messages"]), "messages"
    if record.get("conversations"):
        return list(record["conversations"]), "messages"

    prompt = record.get("prompt")
    completion = record.get("completion")
    if prompt is None or completion is None:
        raise ValueError("SFT example must provide messages/conversations or prompt/completion.")

    messages: list[dict[str, Any]] = []
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        messages.extend(prompt)
    elif isinstance(prompt, dict) and prompt.get("messages"):
        messages.extend(prompt["messages"])
    else:
        raise ValueError(f"Unsupported prompt shape for SFT preprocessing: {type(prompt)!r}")

    if isinstance(completion, str):
        messages.append({"role": "assistant", "content": completion})
    elif isinstance(completion, dict):
        messages.append(completion)
    elif isinstance(completion, list):
        messages.extend(completion)
    else:
        raise ValueError(f"Unsupported completion shape for SFT preprocessing: {type(completion)!r}")

    return messages, "prompt_completion"


def materialize_sft_example(
    *,
    tokenizer: Any,
    record: dict[str, Any],
    max_seq_length: int,
    assistant_only_loss: bool,
    source_hash: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> PreparedSFTExample:
    # chat_template_kwargs is forwarded verbatim into apply_chat_template (e.g.
    # {"enable_thinking": False} for thinking-capable models). Default None ⇒ empty
    # dict ⇒ byte-identical rendering for callers that pass nothing. HF tokenizers
    # forward unrecognized keys into the Jinja context and ignore them, so this is
    # safe for any chat template that does not reference the supplied keys.
    template_kwargs = chat_template_kwargs or {}

    messages, example_format = normalize_sft_messages(record)
    messages = sanitize_messages_for_chat_template(messages)

    if not messages:
        raise ValueError("Cannot materialize empty SFT conversation.")

    # Unwrap Processor → Tokenizer for multimodal models (Gemma 4, Qwen-VL, etc.)
    # Processors have apply_chat_template but lack encode(); the inner .tokenizer does.
    _encoder = getattr(tokenizer, "tokenizer", tokenizer)

    full_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    full_tokens = _encoder.encode(full_str, add_special_tokens=False)
    truncation_applied = len(full_tokens) > max_seq_length
    input_ids = list(full_tokens[:max_seq_length])
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    loss_mask_mode: LossMaskMode = "full_sequence"
    if assistant_only_loss and messages[-1].get("role") == "assistant":
        prompt_str = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        prompt_tokens = _encoder.encode(prompt_str, add_special_tokens=False)
        mask_len = min(len(prompt_tokens), len(labels))
        for idx in range(mask_len):
            if labels[idx] == prompt_tokens[idx]:
                labels[idx] = -100
            else:
                break
        loss_mask_mode = "assistant_only"

    return PreparedSFTExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        example_format=example_format,
        loss_mask_mode=loss_mask_mode,
        truncation_applied=truncation_applied,
        source_hash=source_hash,
    )
