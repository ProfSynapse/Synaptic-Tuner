"""
SFT loss-mask doctor: inspect how the real SFT preprocessing path masks a dataset.

Rows are materialized with :func:`preprocessing.materialize_sft_row`, the same
per-row hop ``prepare_sft_dataset`` uses during training, and loaded with
:func:`data_loader.load_raw_sft_dataset`, the trainer's own loader. Only a
tokenizer is needed (no model weights, no GPU). Nothing here re-implements
masking: the checks read the labels and the diagnostic fields that
``shared.sft_preprocessing.materialize_sft_example`` records on
``PreparedSFTExample``.

The dataset-level contract the trainer enforces before materializing rows
(:func:`preprocessing.validate_sft_dataset_contract`: format mixing, raw_text and
authoritative prepared-row rules) runs over the whole loaded dataset first.

Checks and severities:
    dataset_contract                 fail  the trainer would reject the dataset
    row_error                        fail  preprocessing raised for the row
    zero_trained_tokens              warn  every label is -100; the trainer
                                           drops the row
    mask_prefix_mismatch             warn  masking stopped before the end of the
                                           prompt render; the trainer drops the
                                           row instead of training prompt tokens
    dropped_rows                     fail  rows the trainer drops exceed
                                           training.max_dropped_row_fraction
                                           (the training run would fail)
    full_sequence_fallback           fail  assistant-only loss requested but the
                                           row ends without an assistant turn
                                           (configurable: fail_on_full_sequence_fallback)
    missing_end_of_turn              fail  trained span does not end with the
                                           end-of-turn token derived by
                                           shared.sft_preprocessing (the trainer
                                           uses the same derivation; eos_token
                                           for raw_text rows, which append it)
    doubled_bos                      fail  the sequence starts with BOS twice
    truncation                       warn  truncation rate above the threshold
    terminator_lost_to_truncation    warn  truncation cut the trained span's
                                           end-of-turn token
    earlier_assistant_turns_untrained info multi-turn rows where only the final
                                           assistant turn is trained
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from shared.sft_preprocessing import (
    DROP_NO_SUPERVISED_TOKENS,
    DROP_REASONS,
    derive_end_of_turn_tokens,
    encoder_of,
    is_whitespace_token,
    token_text,
)
from shared.training_utils import dict_to_dataclass, reject_unknown_config_keys

from data_loader import load_raw_sft_dataset
from preprocessing import (
    ASSISTANT_ONLY,
    dropped_row_fraction_exceeded,
    materialize_sft_row,
    normalize_sft_example,
    validate_sft_dataset_contract,
)

FAIL = "fail"
WARN = "warn"
INFO = "info"
OK = "ok"

DEFAULT_DOCTOR_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mask_doctor.yaml"

CHECK_ORDER = (
    (
        "dataset_contract",
        FAIL,
        "The dataset-level SFT row contract failed; the trainer rejects the dataset "
        "before materializing any row.",
    ),
    ("row_error", FAIL, "Preprocessing raised for these rows (training would crash)."),
    (
        "zero_trained_tokens",
        WARN,
        "No position carries a label (often truncation inside the prompt); the trainer "
        "drops the row.",
    ),
    (
        "mask_prefix_mismatch",
        WARN,
        "The full render diverged from the add_generation_prompt render before the prompt "
        "ended; the trainer drops the row instead of training prompt tokens.",
    ),
    (
        "dropped_rows",
        FAIL,
        "Rows the trainer drops exceed training.max_dropped_row_fraction, so the training "
        "run fails.",
    ),
    (
        "full_sequence_fallback",
        FAIL,
        "Assistant-only loss was requested but the row does not end with an assistant turn, "
        "so the whole sequence is trained.",
    ),
    (
        "missing_end_of_turn",
        FAIL,
        "The trained span does not end with the derived end-of-turn token, so the model "
        "does not learn to stop (or learns the wrong stop token).",
    ),
    ("doubled_bos", FAIL, "The sequence starts with two BOS tokens."),
    ("truncation", WARN, "Rows longer than max_seq_length are cut."),
    (
        "terminator_lost_to_truncation",
        WARN,
        "Truncation removed the end-of-turn token from the trained span.",
    ),
    (
        "earlier_assistant_turns_untrained",
        INFO,
        "Multi-turn rows: only the final assistant turn is trained; earlier assistant "
        "turns are masked as prompt.",
    ),
)


@dataclass
class MaskDoctorSettings:
    """Everything that decides how rows are materialized, resolved like the trainer."""

    model_name: str
    max_seq_length: int
    loss_mask_mode: str
    prompt_render: str = "full_conversation"
    chat_template_kwargs: dict[str, Any] | None = None
    local_file: str | None = None
    dataset_name: str | None = None
    dataset_file: str | None = None
    filter_desirable: bool = False
    # Dataset-contract inputs, read from the trainer config like train_sft.py does.
    assistant_only_loss_requested: bool = False
    aux_token_position: str | int | None = None
    use_preassigned_splits: bool = False
    max_dropped_row_fraction: float = 0.01
    sample_size: int = 200
    seed: int = 0
    preview_rows: list[int] | None = None
    preview_count: int = 2
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "loss_mask_mode": self.loss_mask_mode,
            "prompt_render": self.prompt_render,
            "chat_template_kwargs": self.chat_template_kwargs,
            "local_file": self.local_file,
            "dataset_name": None if self.local_file else self.dataset_name,
            "dataset_file": None if self.local_file else self.dataset_file,
            "filter_desirable": self.filter_desirable,
            "assistant_only_loss_requested": self.assistant_only_loss_requested,
            "aux_token_position": self.aux_token_position,
            "use_preassigned_splits": self.use_preassigned_splits,
            "max_dropped_row_fraction": self.max_dropped_row_fraction,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "preview_rows": self.preview_rows,
            "preview_count": self.preview_count,
            "sources": list(self.sources),
        }


@dataclass
class MaskDoctorConfig:
    """Schema of Trainers/sft/configs/mask_doctor.yaml (unknown keys are refused)."""

    sample_size: int = 200
    seed: int = 0
    preview_count: int = 2
    preview_segment_tokens: int = 32
    max_listed_rows: int = 20
    fail_on_full_sequence_fallback: bool = True
    truncation_warn_rate: float = 0.01


def load_doctor_config(path: str | Path | None = None) -> MaskDoctorConfig:
    """Load the doctor config through the repo's strict config-key checks."""
    config_path = Path(path) if path else DEFAULT_DOCTOR_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Mask doctor config must be a YAML mapping: {config_path}")
    reject_unknown_config_keys(MaskDoctorConfig, data, source=str(config_path))
    return dict_to_dataclass(MaskDoctorConfig, data)


# ---------------------------------------------------------------------------
# Row analysis
# ---------------------------------------------------------------------------


@dataclass
class RowResult:
    index: int
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    length: int = 0
    untruncated_length: int = 0
    trained_tokens: int = 0
    truncated: bool = False
    loss_mask_mode: str | None = None
    prompt_token_count: int | None = None
    masked_prefix_length: int = 0
    divergence: dict[str, Any] | None = None
    fallback_reason: str | None = None
    drop_reason: str | None = None
    end_of_turn_index: int | None = None
    unexpected_terminator: str | None = None
    untrained_assistant_turns: int = 0
    input_ids: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "error": self.error,
            "length": self.length,
            "untruncated_length": self.untruncated_length,
            "trained_tokens": self.trained_tokens,
            "truncated": self.truncated,
            "loss_mask_mode": self.loss_mask_mode,
            "prompt_token_count": self.prompt_token_count,
            "masked_prefix_length": self.masked_prefix_length,
            "divergence": self.divergence,
            "fallback_reason": self.fallback_reason,
            "drop_reason": self.drop_reason,
            "unexpected_terminator": self.unexpected_terminator,
            "untrained_assistant_turns": self.untrained_assistant_turns,
        }


def analyze_row(
    index: int,
    row: dict[str, Any],
    *,
    tokenizer: Any,
    settings: MaskDoctorSettings,
    end_of_turn: Any,
    fail_on_fallback: bool = True,
) -> RowResult:
    """Materialize one row through the trainer path and evaluate every check."""
    encoder = encoder_of(tokenizer)
    result = RowResult(index=index)
    try:
        # raw_text rows normalize to text only (no messages).
        messages = normalize_sft_example(row).get("messages") or []
        prepared = materialize_sft_row(
            row,
            tokenizer=tokenizer,
            max_seq_length=settings.max_seq_length,
            loss_mask_mode=settings.loss_mask_mode,
            chat_template_kwargs=settings.chat_template_kwargs,
            prompt_render=settings.prompt_render,
        )
    except Exception as exc:  # noqa: BLE001 - every row error is reported, not raised
        result.error = f"{type(exc).__name__}: {exc}"
        result.failures.append("row_error")
        return result

    labels = list(prepared.labels)
    input_ids = list(prepared.input_ids)
    trained = [position for position, label in enumerate(labels) if label != -100]

    result.input_ids = input_ids
    result.labels = labels
    result.length = len(input_ids)
    result.untruncated_length = prepared.untruncated_length
    result.trained_tokens = len(trained)
    result.truncated = prepared.truncation_applied
    result.loss_mask_mode = prepared.loss_mask_mode
    result.prompt_token_count = prepared.prompt_token_count
    result.masked_prefix_length = prepared.masked_prefix_length
    result.fallback_reason = prepared.mask_fallback_reason
    result.drop_reason = prepared.drop_reason

    if prepared.drop_reason == DROP_NO_SUPERVISED_TOKENS:
        result.warnings.append("zero_trained_tokens")

    if prepared.mask_prefix_mismatch:
        stop = prepared.masked_prefix_length
        expected = prepared.mask_divergence_expected_token
        result.divergence = {
            "position": stop,
            "prompt_token_count": prepared.prompt_token_count,
            "prompt_tokens_trained": (prepared.prompt_token_count or 0) - stop,
            "input_token": token_text(encoder, input_ids[stop]) if stop < len(input_ids) else None,
            "expected_prompt_token": token_text(encoder, expected) if expected is not None else None,
        }
        result.warnings.append("mask_prefix_mismatch")

    if prepared.mask_fallback_reason is not None:
        (result.failures if fail_on_fallback else result.warnings).append("full_sequence_fallback")

    if trained and prepared.drop_reason is None:
        # raw_text rows bypass the chat template and close with eos_token.
        accepted = end_of_turn.token_ids
        if prepared.example_format == "raw_text":
            eos_id = getattr(encoder, "eos_token_id", None)
            if eos_id is None:
                eos_id = getattr(tokenizer, "eos_token_id", None)
            accepted = [eos_id]
        position = trained[-1]
        while position > trained[0] and is_whitespace_token(encoder, input_ids[position]):
            position -= 1
        if input_ids[position] in accepted:
            result.end_of_turn_index = position
        elif prepared.truncation_applied and trained[-1] == len(input_ids) - 1:
            result.warnings.append("terminator_lost_to_truncation")
        else:
            result.unexpected_terminator = token_text(encoder, input_ids[position])
            result.failures.append("missing_end_of_turn")

    bos_id = getattr(encoder, "bos_token_id", None)
    if bos_id is not None and len(input_ids) >= 2 and input_ids[0] == bos_id == input_ids[1]:
        result.failures.append("doubled_bos")

    if result.truncated:
        result.warnings.append("truncation")

    if prepared.loss_mask_mode == ASSISTANT_ONLY:
        assistant_turns = sum(1 for message in messages if message.get("role") == "assistant")
        result.untrained_assistant_turns = max(assistant_turns - 1, 0)

    return result


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------


def _display(text: str) -> str:
    """Make whitespace visible on one terminal line."""
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _segments(labels: list[int]) -> list[tuple[bool, int, int]]:
    segments: list[tuple[bool, int, int]] = []
    start = 0
    for position in range(1, len(labels) + 1):
        if position == len(labels) or (labels[position] != -100) != (labels[start] != -100):
            segments.append((labels[start] != -100, start, position - 1))
            start = position
    return segments


def build_preview(result: RowResult, tokenizer: Any, segment_tokens: int) -> dict[str, Any]:
    """Token-by-token view of one row: which positions are masked and which trained."""
    encoder = encoder_of(tokenizer)
    tokens = [
        {
            "position": position,
            "id": token_id,
            "text": token_text(encoder, token_id),
            "trained": result.labels[position] != -100,
        }
        for position, token_id in enumerate(result.input_ids)
    ]
    lines = [
        f"--- row {result.index} | {result.length} tokens"
        f"{f' (untruncated {result.untruncated_length})' if result.truncated else ''}"
        f" | trained {result.trained_tokens} | loss {result.loss_mask_mode}"
        + (f" | FAIL: {', '.join(result.failures)}" if result.failures else "")
        + (f" | WARN: {', '.join(result.warnings)}" if result.warnings else "")
    ]
    if result.error:
        lines.append(f"  error: {result.error}")
    for trained, start, end in _segments(result.labels):
        marker = "TRAIN" if trained else "MASK "
        span = tokens[start : end + 1]
        if segment_tokens > 0 and len(span) > 2 * segment_tokens:
            shown = (
                "".join(f"[{_display(token['text'])}]" for token in span[:segment_tokens])
                + f" ... ({len(span) - 2 * segment_tokens} tokens) ... "
                + "".join(f"[{_display(token['text'])}]" for token in span[-segment_tokens:])
            )
        else:
            shown = "".join(f"[{_display(token['text'])}]" for token in span)
        lines.append(f"  {marker} {start}..{end}: {shown}")
    if result.divergence:
        lines.append(
            f"  ^ mask stopped at position {result.divergence['position']}: input token "
            f"[{_display(result.divergence['input_token'] or '')}] but the prompt render expected "
            f"[{_display(result.divergence['expected_prompt_token'] or '')}]; "
            f"{result.divergence['prompt_tokens_trained']} prompt token(s) trained "
            f"(prompt render = {result.divergence['prompt_token_count']} tokens)"
        )
    if result.end_of_turn_index is not None:
        lines.append(
            f"  end-of-turn [{_display(tokens[result.end_of_turn_index]['text'])}] "
            f"trained at position {result.end_of_turn_index}"
        )
    if result.unexpected_terminator is not None:
        lines.append(
            f"  ! trained span ends with [{_display(result.unexpected_terminator)}], "
            "not a derived end-of-turn token"
        )
    return {"index": result.index, "text": "\n".join(lines), "tokens": tokens}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(math.ceil(fraction * len(ordered)) - 1, 0)
    return ordered[min(rank, len(ordered) - 1)]


def select_indices(total: int, sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0 or sample_size >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), sample_size))


def diagnose_rows(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    tokenizer: Any,
    settings: MaskDoctorSettings,
    doctor_config: MaskDoctorConfig,
    dataset_rows: int | None = None,
    contract_error: str | None = None,
) -> dict[str, Any]:
    """Run every check over ``rows`` (``(dataset_index, row)`` pairs) and build the report.

    ``contract_error`` is the trainer's dataset-level contract failure, if any
    (see :func:`run_mask_doctor`); it is a hard failure of the whole dataset.
    """
    encoder = encoder_of(tokenizer)
    end_of_turn = derive_end_of_turn_tokens(
        tokenizer, chat_template_kwargs=settings.chat_template_kwargs
    )
    if not end_of_turn.token_ids:
        raise ValueError(
            "Cannot derive an end-of-turn token: the chat template emits none after "
            f"assistant content and the tokenizer defines no eos_token_id ({end_of_turn.detail})."
        )
    fail_on_fallback = bool(doctor_config.fail_on_full_sequence_fallback)
    max_listed = int(doctor_config.max_listed_rows)
    truncation_warn_rate = float(doctor_config.truncation_warn_rate)

    results = [
        analyze_row(
            index,
            row,
            tokenizer=tokenizer,
            settings=settings,
            end_of_turn=end_of_turn,
            fail_on_fallback=fail_on_fallback,
        )
        for index, row in rows
    ]
    analyzed = len(results)

    checks: list[dict[str, Any]] = []
    for name, severity, description in CHECK_ORDER:
        if name == "dataset_contract":
            checks.append(
                {
                    "name": name,
                    "status": FAIL if contract_error else OK,
                    "count": 1 if contract_error else 0,
                    "rows": [],
                    "description": description,
                    "error": contract_error,
                }
            )
            continue
        if name == "dropped_rows":
            hits = [result for result in results if result.drop_reason is not None]
            fraction = (len(hits) / analyzed) if analyzed else 0.0
            exceeded = dropped_row_fraction_exceeded(
                len(hits), analyzed, settings.max_dropped_row_fraction
            )
            checks.append(
                {
                    "name": name,
                    "status": FAIL if exceeded else (WARN if hits else OK),
                    "count": len(hits),
                    "rows": [result.index for result in hits[:max_listed]],
                    "description": description,
                    "fraction": round(fraction, 4),
                    "max_fraction": settings.max_dropped_row_fraction,
                    "by_reason": {
                        reason: sum(1 for result in hits if result.drop_reason == reason)
                        for reason in DROP_REASONS
                    },
                }
            )
            continue
        if name == "earlier_assistant_turns_untrained":
            hits = [result for result in results if result.untrained_assistant_turns > 0]
            extra = {"untrained_turns": sum(result.untrained_assistant_turns for result in hits)}
        else:
            hits = [result for result in results if name in result.failures or name in result.warnings]
            extra = {}
        if name == "full_sequence_fallback" and not fail_on_fallback:
            severity = WARN
        if name == "truncation":
            rate = (len(hits) / analyzed) if analyzed else 0.0
            extra = {"rate": round(rate, 4), "warn_rate": truncation_warn_rate}
            status = WARN if rate > truncation_warn_rate else (INFO if hits else OK)
        else:
            status = severity if hits else OK
        checks.append(
            {
                "name": name,
                "status": status,
                "count": len(hits),
                "rows": [result.index for result in hits[:max_listed]],
                "description": description,
                **extra,
            }
        )

    lengths = [result.untruncated_length for result in results if result.error is None]
    hard_failures = sorted(
        {result.index for result in results if any(check in result.failures for check in _fail_names(checks))}
    )

    preview_indices = settings.preview_rows
    if preview_indices is None:
        failing = [result.index for result in results if result.failures]
        others = [result.index for result in results if not result.failures]
        preview_indices = (failing + others)[: max(settings.preview_count, 0)]
    by_index = {result.index: result for result in results}
    segment_tokens = int(doctor_config.preview_segment_tokens)
    previews = [
        build_preview(by_index[index], tokenizer, segment_tokens)
        for index in preview_indices
        if index in by_index
    ]

    return {
        "success": not _fail_names(checks),
        "settings": settings.to_dict(),
        "tokenizer": {
            "eos_token": getattr(encoder, "eos_token", None),
            "eos_token_id": getattr(encoder, "eos_token_id", None),
            "bos_token": getattr(encoder, "bos_token", None),
            "bos_token_id": getattr(encoder, "bos_token_id", None),
            "end_of_turn": end_of_turn.to_dict(encoder),
        },
        "summary": {
            "dataset_rows": dataset_rows if dataset_rows is not None else analyzed,
            "rows_analyzed": analyzed,
            "rows_with_hard_failures": len(hard_failures),
            "trained_tokens_total": sum(result.trained_tokens for result in results),
        },
        "lengths": {
            "max_seq_length": settings.max_seq_length,
            "p50": _percentile(lengths, 0.50),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths) if lengths else None,
            "truncated_rows": sum(1 for result in results if result.truncated),
        },
        "checks": checks,
        "row_errors": [
            {"index": result.index, "error": result.error} for result in results if result.error
        ][:max_listed],
        "mask_divergences": [
            {"index": result.index, **result.divergence} for result in results if result.divergence
        ][:max_listed],
        "end_of_turn_mismatches": [
            {"index": result.index, "trained_span_ends_with": result.unexpected_terminator}
            for result in results
            if result.unexpected_terminator is not None
        ][:max_listed],
        "flagged_rows": [
            result.summary() for result in results if result.failures or result.warnings
        ][:max_listed],
        "previews": previews,
    }


def _fail_names(checks: list[dict[str, Any]]) -> set[str]:
    return {check["name"] for check in checks if check["status"] == FAIL}


def run_mask_doctor(
    *,
    tokenizer: Any,
    settings: MaskDoctorSettings,
    doctor_config: MaskDoctorConfig,
) -> dict[str, Any]:
    """Load the dataset with the trainer's loader, sample it, and diagnose the sample."""
    dataset = load_raw_sft_dataset(
        dataset_name=settings.dataset_name if not settings.local_file else None,
        data_files=settings.dataset_file if not settings.local_file else None,
        local_file=settings.local_file,
        filter_desirable=settings.filter_desirable,
    )
    try:
        validate_sft_dataset_contract(
            dataset,
            loss_mask_mode=settings.loss_mask_mode,
            prompt_render=settings.prompt_render,
            assistant_only_loss_requested=settings.assistant_only_loss_requested,
            aux_token_position=settings.aux_token_position,
            use_preassigned_splits=settings.use_preassigned_splits,
        )
        contract_error = None
    except ValueError as exc:
        contract_error = str(exc)
    total = len(dataset)
    indices = select_indices(total, settings.sample_size, settings.seed)
    for index in settings.preview_rows or []:
        if not 0 <= index < total:
            raise ValueError(f"--preview-rows index {index} is outside the dataset (0..{total - 1}).")
    indices = sorted(set(indices) | set(settings.preview_rows or []))
    rows = [(index, dataset[index]) for index in indices]
    return diagnose_rows(
        rows,
        tokenizer=tokenizer,
        settings=settings,
        doctor_config=doctor_config,
        dataset_rows=total,
        contract_error=contract_error,
    )


def format_report(report: dict[str, Any]) -> str:
    """Plain-text rendering of a mask doctor report for terminals."""
    settings = report["settings"]
    tokenizer = report["tokenizer"]
    summary = report["summary"]
    lengths = report["lengths"]
    end_of_turn = tokenizer["end_of_turn"]
    dataset = settings["local_file"] or (
        f"{settings['dataset_name']}" + (f" ({settings['dataset_file']})" if settings["dataset_file"] else "")
    )
    lines = [
        "SFT MASK DOCTOR",
        "=" * 60,
        f"Dataset:        {dataset}",
        f"Tokenizer:      {settings['model_name']}",
        f"Loss mask:      {settings['loss_mask_mode']} (prompt_render={settings['prompt_render']})",
        f"Template kwargs: {json.dumps(settings['chat_template_kwargs'] or {})}",
        f"Max seq length: {settings['max_seq_length']}",
        f"Settings from:  {', '.join(settings['sources']) or 'flags only'}",
        f"End-of-turn:    {', '.join(_display(token) for token in end_of_turn['tokens'])} "
        f"(source: {end_of_turn['source']})",
        f"EOS / BOS:      {tokenizer['eos_token']!r} / {tokenizer['bos_token']!r}",
        f"Rows analyzed:  {summary['rows_analyzed']} of {summary['dataset_rows']}",
        f"Token lengths:  p50={lengths['p50']} p95={lengths['p95']} max={lengths['max']} "
        f"(truncated rows: {lengths['truncated_rows']})",
        "",
        "Checks:",
    ]
    for check in report["checks"]:
        rows = check["rows"]
        row_text = f"  rows: {rows}{' ...' if check['count'] > len(rows) else ''}" if rows else ""
        extra = ""
        if check["name"] == "dataset_contract" and check.get("error"):
            extra = f" ({check['error']})"
        elif check["name"] == "dropped_rows":
            reasons = ", ".join(f"{key}={value}" for key, value in check["by_reason"].items())
            extra = (
                f" ({check['fraction']:.2%} of analyzed rows, maximum "
                f"{check['max_fraction']:.2%}; {reasons})"
            )
        elif check["name"] == "truncation":
            extra = f" (rate {check['rate']:.2%}, warn above {check['warn_rate']:.2%})"
        elif check["name"] == "earlier_assistant_turns_untrained" and check["count"]:
            extra = f" ({check['untrained_turns']} untrained assistant turns)"
        lines.append(f"  [{check['status'].upper():4}] {check['name']}: {check['count']}{extra}{row_text}")
    for divergence in report["mask_divergences"]:
        lines.append(
            f"  row {divergence['index']}: mask stopped at {divergence['position']} of "
            f"{divergence['prompt_token_count']} prompt tokens; input "
            f"[{_display(divergence['input_token'] or '')}] vs prompt "
            f"[{_display(divergence['expected_prompt_token'] or '')}]"
        )
    for mismatch in report["end_of_turn_mismatches"]:
        lines.append(
            f"  row {mismatch['index']}: trained span ends with "
            f"[{_display(mismatch['trained_span_ends_with'])}], expected one of "
            f"{', '.join(_display(token) for token in end_of_turn['tokens'])}"
        )
    for error in report["row_errors"]:
        lines.append(f"  row {error['index']}: {error['error']}")
    if report["previews"]:
        lines.extend(["", "Token previews ([token] per position; MASK = label -100, TRAIN = supervised):"])
        for preview in report["previews"]:
            lines.append(preview["text"])
    lines.extend(
        [
            "",
            "Result: "
            + (
                "PASS (no hard failures)"
                if report["success"]
                else f"FAIL ({summary['rows_with_hard_failures']} row(s) with hard failures; "
                + "failed checks: "
                + ", ".join(check["name"] for check in report["checks"] if check["status"] == FAIL)
                + ")"
            ),
        ]
    )
    return "\n".join(lines)
