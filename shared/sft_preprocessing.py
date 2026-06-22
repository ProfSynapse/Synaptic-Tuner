from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal


LossMaskMode = Literal["full_sequence", "completion_only", "assistant_only"]
ExampleFormat = Literal["messages", "prompt_completion"]

# What each loss-mask mode supervises:
#   full_sequence   – every token (no masking).
#   completion_only – ONLY the final assistant turn (the trailing completion).
#                     Implemented as a prompt-prefix diff: mask the longest
#                     common token prefix of render(messages[:-1]+gen_prompt)
#                     against the full render, train the remainder. Multi-turn
#                     intermediate assistant turns are MASKED (they sit inside
#                     the prefix). This was historically (mis)named
#                     "assistant_only".
#   assistant_only  – EVERY assistant turn (incl. intermediate tool-call turns);
#                     system/user/tool spans are masked. Required for multi-turn
#                     agentic tool-use SFT, where every <tool_call> turn must be
#                     in the loss. Implemented by scanning the rendered token
#                     stream for the ChatML <|im_start|>assistant\n … <|im_end|>
#                     spans (marker ids resolved from the tokenizer; fail-loud on
#                     a non-ChatML template).
_LOSS_MASK_MODES = ("full_sequence", "completion_only", "assistant_only")


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


ToolCallMode = Literal["render_text", "native"]


def _normalize_tool_call_arguments(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce each tool call's ``arguments`` into an OBJECT (mapping).

    Native rendering (Qwen3.5/Qwen3 ``<tool_call><function=…><parameter=…>``)
    requires ``arguments`` to be a mapping — the Jinja template calls ``.items()``
    on it and raises ``TypeError: Can only get item pairs from a mapping`` if it
    is a JSON STRING (as Codex emits) rather than an object (as Claude Code
    emits). We parse string arguments to dicts here so both sources render the
    same native structure. A string that does not parse to a dict is left as-is
    (the template will surface the failure loudly rather than silently dropping).
    """
    normalized_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        new_call = dict(call)
        fn = new_call.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    parsed = args
                if isinstance(parsed, dict):
                    fn["arguments"] = parsed
            new_call["function"] = fn
        normalized_calls.append(new_call)
    return normalized_calls


def sanitize_messages_for_chat_template(
    messages: list[dict[str, Any]],
    *,
    tool_call_mode: ToolCallMode = "render_text",
) -> list[dict[str, Any]]:
    """Normalize nullable content for the chat template.

    ``tool_call_mode`` controls how structured ``tool_calls`` are handled:

      "render_text" (default) – LEGACY prose fold: render the tool calls into the
        assistant TEXT and POP the structured key. Preserves the historical
        behavior byte-for-byte. Rows without a structured ``tool_calls`` field
        (every old prose dataset) are unaffected — the fold branch only fires
        ``if tool_calls``, which is empty/absent there.

      "native" – PASS the structured ``tool_calls`` THROUGH to the chat template
        so it renders native ``<tool_call>`` markup (verified against the stock
        Qwen3.5 template). ``arguments`` are coerced to objects so Codex
        JSON-string args and Claude object args render identically. Used by the
        agentic tool-trajectory rows where the <tool_call> tokens must be trained
        as native structure, not prose.
    """
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        content = normalized.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        tool_calls = normalized.get("tool_calls") or []
        if tool_calls and tool_call_mode == "render_text":
            tool_content = render_tool_call_content(tool_calls)
            content = f"{content}\n\n{tool_content}".strip() if content else tool_content
            normalized.pop("tool_calls", None)
        elif tool_calls:  # native: preserve structured tool_calls for the template
            normalized["tool_calls"] = _normalize_tool_call_arguments(tool_calls)

        normalized["content"] = content
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


def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    template_kwargs: dict[str, Any],
) -> str:
    """Single chat-template render path. Every render in this module funnels
    through here so the full-sequence render and the prefix renders the masks
    depend on stay byte-identical. We render with the STOCK template (no
    train/inference skew); empty-think handling is done later at the LABEL level
    (see _mask_assistant_only_spans), never by rewriting the rendered string."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )


def _render_token_len(
    tokenizer: Any,
    encoder: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    template_kwargs: dict[str, Any],
) -> int:
    """Token length of a prefix of ``messages`` rendered through the chat
    template. Used to locate per-turn token boundaries by diffing successive
    prefix renders — valid because ChatML-style templates are append-only
    (render(messages[:k]) is a token prefix of render(messages[:k+1]))."""
    if not messages and not add_generation_prompt:
        return 0
    rendered = _render_chat(
        tokenizer, messages,
        add_generation_prompt=add_generation_prompt,
        template_kwargs=template_kwargs,
    )
    return len(encoder.encode(rendered, add_special_tokens=False))


def _mask_completion_only(
    tokenizer: Any,
    encoder: Any,
    messages: list[dict[str, Any]],
    labels: list[int],
    *,
    template_kwargs: dict[str, Any],
) -> None:
    """Prompt-prefix mask: supervise ONLY the final assistant turn. Masks the
    longest common token prefix of render(messages[:-1]+gen_prompt) against the
    full render, in place. Multi-turn intermediate assistant turns stay masked
    (they sit inside the common prefix)."""
    if messages[-1].get("role") != "assistant":
        return
    prompt_str = _render_chat(
        tokenizer, messages[:-1],
        add_generation_prompt=True,
        template_kwargs=template_kwargs,
    )
    prompt_tokens = encoder.encode(prompt_str, add_special_tokens=False)
    mask_len = min(len(prompt_tokens), len(labels))
    for idx in range(mask_len):
        if labels[idx] == prompt_tokens[idx]:
            labels[idx] = -100
        else:
            break


def _resolve_chatml_marker_ids(encoder: Any) -> tuple[int, int, int, int]:
    """Resolve the ChatML structural token-ids (<|im_start|>, <|im_end|>,
    the `assistant` role word, and the `\\n` that closes the role header) FROM
    the tokenizer — never hardcoded. Fails loud on a non-ChatML template so the
    per-span assistant mask can never silently mis-supervise.

    Returns ``(im_start_id, im_end_id, assistant_id, newline_id)``.

    The assistant turn header renders as the 3-token sequence
    ``<|im_start|>`` ``assistant`` ``\\n`` (verified against the Qwen3.5 / Qwen3
    ChatML template); the turn body runs from there to the next ``<|im_end|>``.
    """
    def _single_id(text: str, label: str) -> int:
        ids = encoder.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"loss_mask_mode='assistant_only' requires a ChatML tokenizer, but "
                f"{label} ({text!r}) did not encode to a single token (got {ids}). "
                "Per-assistant-span masking is only implemented for ChatML-style "
                "templates (<|im_start|>role\\n … <|im_end|>); use loss_mask_mode="
                "'completion_only' for non-ChatML templates."
            )
        return ids[0]

    im_start_id = _single_id("<|im_start|>", "<|im_start|>")
    im_end_id = _single_id("<|im_end|>", "<|im_end|>")
    # The header is "<|im_start|>assistant\n": the role word + newline are
    # ordinary tokens. Derive them from the full header so we match exactly what
    # the template emits (the role word may tokenize differently in isolation).
    header_ids = encoder.encode("<|im_start|>assistant\n", add_special_tokens=False)
    if len(header_ids) < 3 or header_ids[0] != im_start_id:
        raise ValueError(
            "loss_mask_mode='assistant_only' requires a ChatML tokenizer whose "
            "assistant header renders as <|im_start|>assistant\\n; got header token "
            f"ids {header_ids}. Use loss_mask_mode='completion_only' instead."
        )
    assistant_id = header_ids[1]
    newline_id = header_ids[-1]
    return im_start_id, im_end_id, assistant_id, newline_id


# A leading EMPTY think block — "<think>" then only whitespace then "</think>".
# The Qwen3.5 / Qwen3 template injects "<think>\n\n</think>\n\n" UNCONDITIONALLY
# at the start of every assistant turn that lacks reasoning_content. Empty think
# carries no signal, so under assistant_only we MASK those tokens (training them
# teaches the "think about nothing" antipattern). A FILLED think block (real
# reasoning between the tags) does not match — its body is non-whitespace — and
# stays TRAINED. Anchored at the start so it only ever matches the leading block.
_EMPTY_THINK_PREFIX_RE = re.compile(r"\A\s*<think>\s*</think>\s*")


def _empty_think_prefix_token_len(encoder: Any, body_ids: list[int]) -> int:
    """Number of LEADING tokens of ``body_ids`` that constitute an empty
    ``<think></think>`` block (incl. the whitespace the template emits before and
    after it), or 0 if the body does not start with an empty think block.

    Works at the STRING level (tokenizer-agnostic — no hardcoded think-tag ids):
    decode the body, regex-match the leading empty-think run, then walk token
    boundaries to find the fewest leading tokens whose decode covers the matched
    char span. A filled think block has non-whitespace between the tags, so the
    whitespace-only regex does not match and this returns 0 (reasoning stays
    trained)."""
    if not body_ids:
        return 0
    body_str = encoder.decode(body_ids)
    m = _EMPTY_THINK_PREFIX_RE.match(body_str)
    if not m:
        return 0
    target_chars = m.end()
    # Grow the leading token run until its decode covers the matched char span.
    # Cumulative re-decode keeps this robust to multi-char/merged tokens.
    for k in range(1, len(body_ids) + 1):
        if len(encoder.decode(body_ids[:k])) >= target_chars:
            return k
    return len(body_ids)


def _mask_assistant_only_spans(
    encoder: Any,
    labels: list[int],
    input_ids: list[int],
) -> None:
    """True per-span assistant mask: supervise EVERY assistant turn (incl.
    intermediate tool-call turns), mask system/user/tool spans.

    Implemented by scanning the rendered token sequence for the ChatML assistant
    header ``<|im_start|>assistant\\n`` and unmasking the turn BODY up to and
    INCLUDING the closing ``<|im_end|>`` (so the model is supervised to emit the
    stop token). The role-header tokens themselves stay masked, mirroring
    completion_only's per-turn boundary (the header is fixed scaffolding the
    model never has to predict). Tool-result turns render under the ``user``
    role in the Qwen ChatML template (``<tool_response>…``), so a header whose
    role word is not ``assistant`` is skipped — tool outputs stay masked.

    EMPTY-THINK MASKING (unconditional): the stock Qwen template injects an empty
    ``<think></think>`` at the head of every traceless assistant turn. Within each
    supervised body we re-mask a LEADING empty-think block back to -100 — empty
    think is never worth training (it teaches "think about nothing"), while real
    reasoning (a filled think block) stays trained. This keeps the native format
    intact with zero train/inference skew (we render the stock template and only
    adjust LABELS).

    The marker ids are resolved from the tokenizer (fail-loud on non-ChatML), so
    nothing here is hardcoded. ``labels`` and ``input_ids`` are identical on
    entry; we mutate ``labels`` in place, copying back the real token id for
    supervised positions and leaving everything else at -100."""
    im_start_id, im_end_id, assistant_id, newline_id = _resolve_chatml_marker_ids(encoder)

    n = len(labels)
    for idx in range(n):
        labels[idx] = -100

    i = 0
    while i < n:
        # An assistant turn header is the 3-token run <|im_start|> assistant \n.
        if (
            input_ids[i] == im_start_id
            and i + 2 < n
            and input_ids[i + 1] == assistant_id
            and input_ids[i + 2] == newline_id
        ):
            body_start = i + 3  # first body token after the role header newline
            j = body_start
            while j < n and input_ids[j] != im_end_id:
                j += 1
            # Supervise the body + the closing <|im_end|> (j) if present.
            end = min(j + 1, n) if j < n else n
            for pos in range(body_start, end):
                labels[pos] = input_ids[pos]
            # Re-mask a leading EMPTY <think></think> block within this body.
            skip = _empty_think_prefix_token_len(encoder, input_ids[body_start:j])
            for pos in range(body_start, body_start + skip):
                labels[pos] = -100
            i = end
            continue
        i += 1


def materialize_sft_example(
    *,
    tokenizer: Any,
    record: dict[str, Any],
    max_seq_length: int,
    loss_mask_mode: LossMaskMode = "full_sequence",
    tool_call_mode: ToolCallMode = "render_text",
    source_hash: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> PreparedSFTExample:
    # chat_template_kwargs is forwarded verbatim into apply_chat_template (e.g.
    # {"enable_thinking": False} for thinking-capable models). Default None ⇒ empty
    # dict ⇒ byte-identical rendering for callers that pass nothing. HF tokenizers
    # forward unrecognized keys into the Jinja context and ignore them, so this is
    # safe for any chat template that does not reference the supplied keys.
    #
    # tool_call_mode controls structured-tool-call handling (see
    # sanitize_messages_for_chat_template). Default "render_text" folds tool calls
    # to prose (legacy, byte-identical for prose data); "native" passes structured
    # tool_calls through so the template renders native <tool_call> markup.
    #
    # We render with the STOCK chat template (no train/inference skew). Under
    # assistant_only, a leading EMPTY <think></think> block on traceless turns is
    # masked at the LABEL level (_mask_assistant_only_spans) so we never train it.
    if loss_mask_mode not in _LOSS_MASK_MODES:
        raise ValueError(
            f"Unsupported loss_mask_mode {loss_mask_mode!r}; "
            f"expected one of {_LOSS_MASK_MODES}."
        )
    template_kwargs = chat_template_kwargs or {}

    messages, example_format = normalize_sft_messages(record)
    messages = sanitize_messages_for_chat_template(messages, tool_call_mode=tool_call_mode)

    if not messages:
        raise ValueError("Cannot materialize empty SFT conversation.")

    # Unwrap Processor → Tokenizer for multimodal models (Gemma 4, Qwen-VL, etc.)
    # Processors have apply_chat_template but lack encode(); the inner .tokenizer does.
    _encoder = getattr(tokenizer, "tokenizer", tokenizer)

    full_str = _render_chat(
        tokenizer, messages,
        add_generation_prompt=False,
        template_kwargs=template_kwargs,
    )
    full_tokens = _encoder.encode(full_str, add_special_tokens=False)
    truncation_applied = len(full_tokens) > max_seq_length
    input_ids = list(full_tokens[:max_seq_length])
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    if loss_mask_mode == "completion_only":
        _mask_completion_only(
            tokenizer, _encoder, messages, labels,
            template_kwargs=template_kwargs,
        )
    elif loss_mask_mode == "assistant_only":
        _mask_assistant_only_spans(_encoder, labels, input_ids)
    # full_sequence: labels already == input_ids (train everything).

    return PreparedSFTExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        example_format=example_format,
        loss_mask_mode=loss_mask_mode,
        truncation_applied=truncation_applied,
        source_hash=source_hash,
    )
