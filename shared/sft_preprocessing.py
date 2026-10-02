from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import asdict, dataclass, field
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
    # Mask diagnostics. Descriptive only: training never reads these, and they
    # never alter input_ids/labels. They let the SFT mask doctor and the data
    # loader summary surface silent masking failures without re-implementing
    # the masking logic.
    #   untruncated_length: token count before max_seq_length truncation.
    #   prompt_token_count: tokens in the add_generation_prompt=True render of
    #       messages[:-1] (None when no prompt render was made).
    #   masked_prefix_length: leading positions whose label is -100.
    #   mask_prefix_mismatch: the full render diverged from the prompt render
    #       before the prompt render ended, so masking stopped early and the
    #       remaining prompt tokens carry real labels.
    #   mask_divergence_expected_token: the prompt-render token expected at the
    #       divergence index (input_ids holds the full-render token there).
    #   mask_fallback_reason: why assistant-only loss was requested but the
    #       row was materialized with full-sequence loss.
    #   drop_reason: why this row must not be trained (see DROP_REASONS);
    #       prepare_sft_dataset drops such rows and enforces the configured
    #       maximum dropped fraction. None = trainable.
    untruncated_length: int = 0
    prompt_token_count: int | None = None
    masked_prefix_length: int = 0
    mask_prefix_mismatch: bool = False
    mask_divergence_expected_token: int | None = None
    mask_fallback_reason: str | None = None
    drop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Rows that must not be trained. ``no_supervised_tokens``: every label is -100
# (typically truncation removed the whole target). ``mask_prefix_mismatch``: the
# full render diverged from the add_generation_prompt render before the prompt
# ended, so assistant-only masking would train prompt tokens.
DROP_NO_SUPERVISED_TOKENS = "no_supervised_tokens"
DROP_MASK_PREFIX_MISMATCH = "mask_prefix_mismatch"
DROP_REASONS = (DROP_NO_SUPERVISED_TOKENS, DROP_MASK_PREFIX_MISMATCH)
# Default for training.max_dropped_row_fraction: the run fails when more than
# this fraction of rows is dropped for the reasons above.
DEFAULT_MAX_DROPPED_ROW_FRACTION = 0.01

# Probe conversation used to read the chat template's assistant end-of-turn
# suffix. The assistant text only needs to be distinctive in the render.
END_OF_TURN_PROBE_USER = "sft end-of-turn probe question"
END_OF_TURN_PROBE_ASSISTANT = "sft end-of-turn probe answer"


@dataclass
class EndOfTurnSpec:
    """Token ids that close an assistant turn, derived from the chat template.

    ``source`` is ``chat_template`` (special tokens the template emits after the
    assistant content), ``chat_template_text`` (no special token; the last visible
    suffix token), ``eos_token`` (the template emits nothing after the content)
    or ``none`` (nothing derivable and no eos_token).
    """

    token_ids: list[int] = field(default_factory=list)
    source: str = "none"
    detail: str = ""

    @property
    def rendered_by_template(self) -> bool:
        return self.source in {"chat_template", "chat_template_text"}

    def to_dict(self, encoder: Any) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "tokens": [token_text(encoder, token_id) for token_id in self.token_ids],
            "source": self.source,
            "detail": self.detail,
        }


def encoder_of(tokenizer: Any) -> Any:
    """Unwrap Processor -> Tokenizer (processors lack encode/decode)."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def token_text(encoder: Any, token_id: int) -> str:
    try:
        return encoder.decode([token_id])
    except Exception:  # noqa: BLE001 - display only
        return f"<id:{token_id}>"


def special_token_ids(encoder: Any) -> set[int]:
    ids: set[int] = set(getattr(encoder, "all_special_ids", None) or [])
    added = getattr(encoder, "added_tokens_decoder", None) or {}
    for token_id, token in added.items():
        if getattr(token, "special", False):
            ids.add(int(token_id))
    return ids


def is_whitespace_token(encoder: Any, token_id: int) -> bool:
    return token_text(encoder, token_id).strip() == ""


def derive_end_of_turn_tokens(
    tokenizer: Any, *, chat_template_kwargs: dict[str, Any] | None = None
) -> EndOfTurnSpec:
    """Derive the assistant end-of-turn token(s) from the chat template itself.

    A fixed probe conversation is rendered with the run's template kwargs; the
    special tokens the template emits after the assistant content are its
    end-of-turn terminators. When it emits no special token there, the last
    visible suffix token is used; when it emits nothing, eos_token. No model
    family is hardcoded. Shared by SFT preprocessing and the SFT mask doctor.
    """
    encoder = encoder_of(tokenizer)
    eos_id = getattr(encoder, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)

    detail = ""
    try:
        render = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": END_OF_TURN_PROBE_USER},
                {"role": "assistant", "content": END_OF_TURN_PROBE_ASSISTANT},
            ],
            tokenize=False,
            add_generation_prompt=False,
            **(chat_template_kwargs or {}),
        )
    except Exception as exc:  # noqa: BLE001 - reported, then fall back to eos
        render = None
        detail = f"probe render failed: {exc}"

    if render is not None:
        position = render.rfind(END_OF_TURN_PROBE_ASSISTANT)
        if position < 0:
            detail = "probe assistant text not found in the template render"
        else:
            head = render[: position + len(END_OF_TURN_PROBE_ASSISTANT)]
            full_ids = encoder.encode(render, add_special_tokens=False)
            head_ids = encoder.encode(head, add_special_tokens=False)
            common = 0
            for left, right in zip(full_ids, head_ids):
                if left != right:
                    break
                common += 1
            suffix_ids = list(full_ids[common:])
            special = special_token_ids(encoder)
            special_suffix = [token_id for token_id in suffix_ids if token_id in special]
            visible_suffix = [
                token_id for token_id in suffix_ids if not is_whitespace_token(encoder, token_id)
            ]
            suffix_text = render[position + len(END_OF_TURN_PROBE_ASSISTANT):]
            if special_suffix:
                return EndOfTurnSpec(
                    list(dict.fromkeys(special_suffix)),
                    "chat_template",
                    f"special tokens after assistant content: {suffix_text!r}",
                )
            if visible_suffix:
                return EndOfTurnSpec(
                    [visible_suffix[-1]],
                    "chat_template_text",
                    f"no special token after assistant content: {suffix_text!r}",
                )
            detail = f"template emits no end-of-turn after assistant content: {suffix_text!r}"

    if eos_id is None:
        return EndOfTurnSpec([], "none", detail or "no end-of-turn token and no eos_token_id")
    return EndOfTurnSpec([int(eos_id)], "eos_token", detail)


_END_OF_TURN_CACHE: "weakref.WeakKeyDictionary[Any, dict[str, EndOfTurnSpec]]" = (
    weakref.WeakKeyDictionary()
)


def cached_end_of_turn_tokens(
    tokenizer: Any, chat_template_kwargs: dict[str, Any] | None = None
) -> EndOfTurnSpec:
    """Per-tokenizer memo of :func:`derive_end_of_turn_tokens` (one probe render)."""
    key = json.dumps(chat_template_kwargs or {}, sort_keys=True, default=repr)
    try:
        per_tokenizer = _END_OF_TURN_CACHE.setdefault(tokenizer, {})
    except TypeError:  # not weak-referenceable: derive every time
        return derive_end_of_turn_tokens(tokenizer, chat_template_kwargs=chat_template_kwargs)
    if key not in per_tokenizer:
        per_tokenizer[key] = derive_end_of_turn_tokens(
            tokenizer, chat_template_kwargs=chat_template_kwargs
        )
    return per_tokenizer[key]


_EOS_TERMINAL_FALLBACK_LOGGED: set[int] = set()


def prompt_completion_terminal_id(
    tokenizer: Any, chat_template_kwargs: dict[str, Any] | None = None
) -> int:
    """Token that closes a prompt_completion target: the template's end-of-turn.

    Derived by :func:`derive_end_of_turn_tokens` (never a hardcoded literal), so
    the completion ends exactly as the chat template ends an assistant turn. When
    the template renders no end-of-turn token, eos_token_id is used and the
    fallback is logged once per tokenizer/kwargs. Loud if neither exists.
    """
    end_of_turn = cached_end_of_turn_tokens(tokenizer, chat_template_kwargs)
    if end_of_turn.rendered_by_template:
        return end_of_turn.token_ids[0]
    encoder = encoder_of(tokenizer)
    eos_id = getattr(encoder, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError(
            "prompt_render='prompt_completion' requires a chat template that renders an "
            "end-of-turn token or a tokenizer that defines eos_token_id (the completion "
            "terminal is derived from them, never hardcoded)."
        )
    if id(end_of_turn) not in _EOS_TERMINAL_FALLBACK_LOGGED:
        _EOS_TERMINAL_FALLBACK_LOGGED.add(id(end_of_turn))
        print(
            "prompt_completion: the chat template renders no end-of-turn token after "
            f"assistant content ({end_of_turn.detail or end_of_turn.source}); closing "
            "completions with eos_token_id."
        )
    return int(eos_id)


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
            untruncated_length=len(full_tokens),
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
    # generation anchor — followed by the raw completion plus the template's
    # end-of-turn token (eos_token_id when the template renders none),
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
        terminal_id = prompt_completion_terminal_id(tokenizer, template_kwargs)
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
        # Mask the prompt segment only when assistant-only loss is requested;
        # with completion_only_loss=false every token is trained, as on the
        # full-conversation path. Right-trim mirrors the full-conversation contract.
        if assistant_only_loss:
            labels = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
        else:
            labels = list(input_ids)
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
            loss_mask_mode="assistant_only" if assistant_only_loss else "full_sequence",
            truncation_applied=truncation_applied,
            source_hash=source_hash,
            untruncated_length=len(full_ids),
            prompt_token_count=len(prompt_ids),
            masked_prefix_length=min(len(prompt_ids), len(labels)) if assistant_only_loss else 0,
            drop_reason=_no_supervision_reason(labels),
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
    prompt_token_count: int | None = None
    masked_prefix_length = 0
    mask_prefix_mismatch = False
    mask_divergence_expected_token: int | None = None
    mask_fallback_reason: str | None = None
    if assistant_only_loss and messages[-1].get("role") == "assistant":
        prompt_str = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        prompt_tokens = _encoder.encode(prompt_str, add_special_tokens=False)
        prompt_token_count = len(prompt_tokens)
        mask_len = min(len(prompt_tokens), len(labels))
        for idx in range(mask_len):
            if labels[idx] == prompt_tokens[idx]:
                labels[idx] = -100
                masked_prefix_length = idx + 1
            else:
                mask_prefix_mismatch = True
                mask_divergence_expected_token = prompt_tokens[idx]
                break
        loss_mask_mode = "assistant_only"
        # Tokens the template emits after the final assistant turn's end-of-turn
        # token (e.g. a trailing newline) are not part of the reply; mask them.
        # Search only the final turn (at/after the prompt render) of the
        # untruncated render, so an earlier turn's terminator is never used.
        end_of_turn = cached_end_of_turn_tokens(tokenizer, template_kwargs)
        if end_of_turn.rendered_by_template:
            for position in range(len(full_tokens) - 1, len(prompt_tokens) - 1, -1):
                if full_tokens[position] in end_of_turn.token_ids:
                    for trailing in range(position + 1, len(labels)):
                        labels[trailing] = -100
                    break
    elif assistant_only_loss:
        mask_fallback_reason = "final_message_not_assistant"

    drop_reason = (
        DROP_MASK_PREFIX_MISMATCH if mask_prefix_mismatch else _no_supervision_reason(labels)
    )

    return PreparedSFTExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        example_format=example_format,
        loss_mask_mode=loss_mask_mode,
        truncation_applied=truncation_applied,
        source_hash=source_hash,
        untruncated_length=len(full_tokens),
        prompt_token_count=prompt_token_count,
        masked_prefix_length=masked_prefix_length,
        mask_prefix_mismatch=mask_prefix_mismatch,
        mask_divergence_expected_token=mask_divergence_expected_token,
        mask_fallback_reason=mask_fallback_reason,
        drop_reason=drop_reason,
    )


def _no_supervision_reason(labels: list[int]) -> str | None:
    return None if any(label != -100 for label in labels) else DROP_NO_SUPERVISED_TOKENS
