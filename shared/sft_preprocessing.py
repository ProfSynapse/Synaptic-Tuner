from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


LossMaskMode = Literal["full_sequence", "assistant_only"]
ExampleFormat = Literal["messages", "prompt_completion", "raw_text"]

RAW_TEXT_SCHEMA_VERSION = "syntunia-sft-row/v1"
RAW_TEXT_FORMAT = "raw_text"
MESSAGES_SCHEMA_VERSION_V2 = "syntunia-sft-row/v2"
MESSAGES_FORMAT = "messages"


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


def render_tool_result_content(content: Any) -> str:
    """Render a tool-result message body into the repo's ChatML-style text format.

    Mirrors :func:`render_tool_call_content` (which renders assistant *requests*),
    producing the symmetric *result* side of a tool exchange. JSON-shaped bodies are
    pretty-printed; everything else is stringified verbatim. The ``tool_result:``
    label keeps multi-turn transcripts coherent and visually paired with the
    ``tool_call:`` label emitted for assistant tool calls.
    """
    if isinstance(content, str):
        body = content
    elif content is None:
        body = ""
    else:
        body = json.dumps(content, ensure_ascii=False, indent=2)
    return f"tool_result:\n{body}" if body else "tool_result:"


def sanitize_messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize nullable content and render tool calls/results into plain text.

    Behavior is split by role so multi-turn trajectories template coherently while
    existing single-turn rows stay byte-identical:

    * ``system`` / ``user`` / ``assistant`` — unchanged from the original contract:
      ``None`` content becomes ``""``, non-string content is JSON-encoded, and any
      assistant ``tool_calls`` are rendered into text via
      :func:`render_tool_call_content` then popped. Existing data uses only these
      roles, so their output is identical to before this function learned about
      tool-result messages.
    * ``tool`` — a tool-RESULT message (the environment's reply to an assistant
      tool call). The repo's house style renders tool exchanges into *text* rather
      than passing them structurally to the chat template, and most chat templates
      only understand system/user/assistant roles. To keep the render-to-text style
      consistent AND guarantee the transcript templates on any tokenizer, the result
      body is rendered via :func:`render_tool_result_content` and the message is
      re-tagged with ``role: "user"``. This matches how the rollout environment
      itself presents tool output (as a ``user`` turn carrying the execution
      result), so it is faithful to the source trace, not a lossy approximation.

    No existing single-turn row contains a ``tool`` role, so this branch is purely
    additive and cannot perturb the byte output of any current SFT example.
    """
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        role = normalized.get("role")

        if role == "tool":
            rendered = render_tool_result_content(normalized.get("content"))
            normalized["role"] = "user"
            normalized["content"] = rendered
            normalized.pop("tool_calls", None)
            sanitized.append(normalized)
            continue

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


def detect_sft_record_format(record: dict[str, Any]) -> ExampleFormat:
    """Identify the declared SFT row format without treating arbitrary text as authority."""
    schema_version = record.get("schema_version")
    declared_format = record.get("format")
    claims_raw_text = (
        schema_version == RAW_TEXT_SCHEMA_VERSION or declared_format == RAW_TEXT_FORMAT
    )
    if claims_raw_text:
        if schema_version != RAW_TEXT_SCHEMA_VERSION or declared_format != RAW_TEXT_FORMAT:
            raise ValueError(
                "raw_text SFT rows require both schema_version="
                f"{RAW_TEXT_SCHEMA_VERSION!r} and format={RAW_TEXT_FORMAT!r}."
            )
        if any(record.get(key) is not None for key in ("messages", "conversations", "prompt", "completion")):
            raise ValueError(
                "raw_text SFT rows cannot also declare messages/conversations or "
                "prompt/completion fields."
            )
        text = record.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("raw_text SFT rows require a non-empty string text field.")
        if record.get("split") not in {"train", "validation"}:
            raise ValueError(
                "raw_text SFT rows require split='train' or split='validation'."
            )
        return "raw_text"

    claims_authoritative_messages = (
        schema_version == MESSAGES_SCHEMA_VERSION_V2
        or declared_format == MESSAGES_FORMAT
    )
    if claims_authoritative_messages:
        if schema_version != MESSAGES_SCHEMA_VERSION_V2 or declared_format != MESSAGES_FORMAT:
            raise ValueError(
                "authoritative message SFT rows require both schema_version="
                f"{MESSAGES_SCHEMA_VERSION_V2!r} and format={MESSAGES_FORMAT!r}."
            )
        if any(record.get(key) is not None for key in ("conversations", "prompt", "completion", "text")):
            raise ValueError(
                "authoritative message SFT rows cannot also declare conversations, "
                "prompt/completion, or text fields."
            )
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("authoritative message SFT rows require exactly two messages.")
        for message, role in zip(messages, ("user", "assistant")):
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
                or not message["content"]
            ):
                raise ValueError(
                    "authoritative message SFT rows require user then assistant prose messages."
                )
        if record.get("split") not in {"train", "validation"}:
            raise ValueError(
                "authoritative message SFT rows require split='train' or split='validation'."
            )
        return "messages"

    if record.get("messages") or record.get("conversations"):
        return "messages"
    if record.get("prompt") is not None and record.get("completion") is not None:
        return "prompt_completion"
    # Preserve the established error and accepted shapes through the canonical
    # normalizer. In particular, an arbitrary ``text`` column is not authority.
    _, example_format = normalize_sft_messages(record)
    return example_format


def is_authoritative_preassigned_sft_record(record: dict[str, Any]) -> bool:
    """Return whether a row carries strict prepared-dataset split authority."""

    schema_version = record.get("schema_version")
    declared_format = record.get("format")
    if schema_version == RAW_TEXT_SCHEMA_VERSION or declared_format == RAW_TEXT_FORMAT:
        detect_sft_record_format(record)
        return True
    if schema_version == MESSAGES_SCHEMA_VERSION_V2 or declared_format == MESSAGES_FORMAT:
        detect_sft_record_format(record)
        return True
    return False


def materialize_sft_example(
    *,
    tokenizer: Any,
    record: dict[str, Any],
    max_seq_length: int,
    assistant_only_loss: bool,
    source_hash: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    prompt_render: str = "full_conversation",
) -> PreparedSFTExample:
    # chat_template_kwargs is forwarded verbatim into apply_chat_template (e.g.
    # {"enable_thinking": False} for thinking-capable models). Default None ⇒ empty
    # dict ⇒ byte-identical rendering for callers that pass nothing. HF tokenizers
    # forward unrecognized keys into the Jinja context and ignore them, so this is
    # safe for any chat template that does not reference the supplied keys.
    template_kwargs = chat_template_kwargs or {}

    example_format = detect_sft_record_format(record)
    authoritative_messages = (
        record.get("schema_version") == MESSAGES_SCHEMA_VERSION_V2
        and record.get("format") == MESSAGES_FORMAT
    )

    if example_format == "raw_text":
        if assistant_only_loss:
            raise ValueError(
                "raw_text SFT rows require full-sequence loss; assistant-only or "
                "completion-only loss is incompatible."
            )
        if prompt_render != "full_conversation":
            raise ValueError(
                "raw_text SFT rows bypass chat rendering and require "
                "prompt_render='full_conversation'."
            )

        _encoder = getattr(tokenizer, "tokenizer", tokenizer)
        terminal_id = getattr(_encoder, "eos_token_id", None)
        if terminal_id is None:
            terminal_id = getattr(tokenizer, "eos_token_id", None)
        if terminal_id is None:
            raise ValueError(
                "raw_text SFT rows require the tokenizer to define eos_token_id; "
                "the terminal is derived from the tokenizer and never hardcoded."
            )
        full_tokens = _encoder.encode(record["text"], add_special_tokens=False) + [terminal_id]
        truncation_applied = len(full_tokens) > max_seq_length
        input_ids = list(full_tokens[:max_seq_length])
        return PreparedSFTExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=list(input_ids),
            example_format="raw_text",
            loss_mask_mode="full_sequence",
            truncation_applied=truncation_applied,
            source_hash=source_hash,
        )

    messages, example_format = normalize_sft_messages(record)
    messages = sanitize_messages_for_chat_template(messages)

    if not messages:
        raise ValueError("Cannot materialize empty SFT conversation.")

    # Unwrap Processor → Tokenizer for multimodal models (Gemma 4, Qwen-VL, etc.)
    # Processors have apply_chat_template but lack encode(); the inner .tokenizer does.
    _encoder = getattr(tokenizer, "tokenizer", tokenizer)

    # prompt_render selects the render/masking strategy. The default
    # "full_conversation" path (below) renders the whole conversation with
    # add_generation_prompt=False and derives the assistant-only mask by a prefix
    # match. That prefix match breaks whenever a template renders the assistant
    # scaffold differently with vs. without add_generation_prompt (e.g. one fewer
    # newline around the header), so the masked boundary is not the generation
    # anchor. The "prompt_completion" branch instead builds input_ids from the
    # add_generation_prompt=True prompt render — so the prompt ends EXACTLY at the
    # generation anchor — followed by the raw completion plus a derived terminal,
    # masking the prompt segment to -100. It is gated strictly behind the
    # non-default flag AND an assistant final turn, so every existing caller is
    # byte-identical. template_kwargs are forwarded into the prompt-half render
    # identically to the full-conversation render, so no new divergence axis is
    # introduced; the completion half is encoded raw (no chat template).
    if prompt_render == "prompt_completion" and messages[-1].get("role") == "assistant":
        prompt_str = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        prompt_ids = _encoder.encode(prompt_str, add_special_tokens=False)

        completion_text = messages[-1].get("content")
        if not isinstance(completion_text, str):
            raise ValueError(
                "prompt_render='prompt_completion' requires the final assistant "
                "message content to be a string after sanitization, got "
                f"{type(completion_text).__name__}."
            )
        # Terminal is DERIVED from the tokenizer (never a hardcoded literal) so the
        # completion closes with the model's own end-of-turn id. Read from the
        # encoder whose vocabulary produced the ids, falling back to the outer
        # tokenizer (Processor wrappers proxy this); loud if neither defines it.
        terminal_id = getattr(_encoder, "eos_token_id", None)
        if terminal_id is None:
            terminal_id = getattr(tokenizer, "eos_token_id", None)
        if terminal_id is None:
            raise ValueError(
                "prompt_render='prompt_completion' requires the tokenizer to define "
                "eos_token_id (the completion terminal is derived from it, never "
                "hardcoded)."
            )
        completion_ids = (
            _encoder.encode(completion_text, add_special_tokens=False) + [terminal_id]
        )

        full_ids = prompt_ids + completion_ids
        if authoritative_messages and len(full_ids) > max_seq_length:
            raise ValueError(
                "authoritative message prompt_completion rows must fit fully within "
                "max_seq_length; target or context truncation is forbidden."
            )
        truncation_applied = len(full_ids) > max_seq_length
        input_ids = list(full_ids[:max_seq_length])
        attention_mask = [1] * len(input_ids)
        # Mask the prompt segment; every completion token (incl. the terminal)
        # carries a real label. Right-trim mirrors the full-conversation contract.
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
        if authoritative_messages and not any(label != -100 for label in labels):
            raise ValueError(
                "authoritative message prompt_completion rows require at least one "
                "supervised assistant token."
            )
        return PreparedSFTExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            example_format=example_format,
            loss_mask_mode="assistant_only",
            truncation_applied=truncation_applied,
            source_hash=source_hash,
        )

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
