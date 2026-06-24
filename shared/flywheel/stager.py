"""
shared/flywheel/stager.py

DatasetStager: assembles tagged examples into versioned JSONL training datasets.
Reads tagged logs from the catalog, formats them as ChatML training data
(matching existing Datasets/ conventions), writes versioned JSONL files,
and registers the flywheel cycle in RunRegistry.

Used by: orchestrator.py (pipeline stage), CLI (manual staging)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.validation import FilterStats, RolloutFilterSet

from .catalog import DatasetVersion, InferenceLogRecord, LogCatalog, LogFilter
from .config import FlywheelConfig
from .utils import read_log_content

logger = logging.getLogger(__name__)

# Default target set for a staging filter that omits ``applies_to``. Quality
# filters apply to the positive/GRPO targets only; ``kto_negative`` is excluded
# so hard negatives are never silently dropped unless a researcher explicitly
# lists ``kto_negative`` in ``applies_to`` (mirrors the projector convention).
_DEFAULT_FILTER_TARGETS = ("sft", "kto_positive", "grpo")


@dataclass
class StagingResult:
    """Summary of a staging run."""
    version_id: str = ""
    sft_count: int = 0
    kto_pos_count: int = 0
    kto_neg_count: int = 0
    grpo_count: int = 0
    total_records: int = 0
    file_paths: dict[str, str] = field(default_factory=dict)
    content_hash: str = ""
    run_id: str = ""


class DatasetStager:
    """Assembles tagged examples into versioned JSONL training datasets.

    Output structure:
        Datasets/flywheel/v003/
            sft_training.jsonl
            kto_training.jsonl
            grpo_training.jsonl
            manifest.json

    Args:
        catalog: LogCatalog instance
        config: FlywheelConfig
        datasets_dir: Base directory for staged datasets
    """

    def __init__(
        self,
        catalog: LogCatalog,
        config: FlywheelConfig,
        datasets_dir: Path | None = None,
    ) -> None:
        self._catalog = catalog
        self._config = config
        self._datasets_dir = Path(
            datasets_dir or config.datasets_dir,
        )
        # Build the declarative staging filter engine. Invalid specs raise
        # ValueError here (fail-closed at construction). An empty config yields
        # an empty set whose .apply() is a pass-all no-op.
        self._filter_set = RolloutFilterSet(
            filters=config.filters or [],
            default_targets=_DEFAULT_FILTER_TARGETS,
        )

    async def stage_dataset(
        self,
        filters: LogFilter | None = None,
    ) -> StagingResult:
        """Stage all tagged-but-unused logs into a new dataset version.

        1. Query catalog for tagged, unused logs
        2. Format each log into ChatML training format
        3. Write versioned JSONL files
        4. Compute content hash
        5. Create DatasetVersion in catalog
        6. Register flywheel_cycle RunRecord in RunRegistry
        7. Mark all used logs in catalog

        Args:
            filters: Optional additional filters

        Returns:
            StagingResult with version info and counts
        """
        result = StagingResult()

        # Query tagged, unused logs by type
        sft_logs = await self._catalog.find_logs(
            LogFilter(tag="sft", unused_only=True),
        )
        kto_logs = await self._catalog.find_logs(
            LogFilter(tag="kto", unused_only=True),
        )
        grpo_logs = await self._catalog.find_logs(
            LogFilter(tag="grpo", unused_only=True),
        )

        if not sft_logs and not kto_logs and not grpo_logs:
            logger.info("No tagged logs to stage")
            return result

        # Determine version
        version_id = self._next_version_id()
        version_dir = self._datasets_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        file_paths: dict[str, Path] = {}
        used_log_ids: list[str] = []

        # Accumulate per-target/per-filter drop stats for this staging run.
        # Fresh per run so re-staging never carries stale tallies.
        filter_stats = FilterStats()

        # Stage SFT examples
        if sft_logs:
            sft_path = version_dir / "sft_training.jsonl"
            sft_count = self._write_sft(sft_logs, sft_path, filter_stats)
            result.sft_count = sft_count
            file_paths["sft"] = sft_path
            used_log_ids.extend(r.log_id for r in sft_logs)

        # Stage KTO examples (SFT logs as positive, KTO logs as negative)
        if sft_logs or kto_logs:
            kto_path = version_dir / "kto_training.jsonl"
            pos, neg = self._write_kto(sft_logs, kto_logs, kto_path, filter_stats)
            result.kto_pos_count = pos
            result.kto_neg_count = neg
            file_paths["kto"] = kto_path
            used_log_ids.extend(r.log_id for r in kto_logs)

        # Stage GRPO examples
        if grpo_logs and self._config.grpo_enabled:
            grpo_path = version_dir / "grpo_training.jsonl"
            grpo_count = self._write_grpo(grpo_logs, grpo_path, filter_stats)
            result.grpo_count = grpo_count
            file_paths["grpo"] = grpo_path
            used_log_ids.extend(r.log_id for r in grpo_logs)

        # Compute content hash
        content_hash = self._compute_content_hash(file_paths)

        # Determine source model
        all_logs = sft_logs + kto_logs + grpo_logs
        source_model = all_logs[0].model_id if all_logs else "unknown"

        # Get parent version
        latest = await self._catalog.get_latest_dataset_version()
        parent = latest.version_id if latest else None

        # Create DatasetVersion
        record_counts = {
            "sft": result.sft_count,
            "kto_pos": result.kto_pos_count,
            "kto_neg": result.kto_neg_count,
            "grpo": result.grpo_count,
        }

        version = DatasetVersion(
            version_id=version_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_model_id=source_model,
            record_counts=record_counts,
            file_paths={k: str(v) for k, v in file_paths.items()},
            content_hash=content_hash,
            parent_version=parent,
            filter_criteria={
                "sft_threshold": self._config.sft_threshold,
                "kto_min_threshold": self._config.kto_min_threshold,
                "scoring_method": self._config.scoring_method,
                # Declarative staging filters applied to this version (empty when
                # no filters configured). Records what filtering was applied so a
                # staged version is self-describing.
                "staging_filters": [f.describe() for f in self._filter_set.filters],
                "staging_filter_stats": (
                    {} if filter_stats.is_empty else filter_stats.as_dict()
                ),
            },
        )

        await self._catalog.create_dataset_version(version)

        # Write manifest
        manifest_path = version_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(version.to_dict(), f, indent=2, ensure_ascii=False)

        # Mark logs as used (deduplicate IDs)
        unique_ids = list(set(used_log_ids))
        if unique_ids:
            await self._catalog.mark_used(unique_ids, version_id)

        # Register in RunRegistry
        run_id = self._register_flywheel_cycle(version)

        result.version_id = version_id
        result.total_records = (
            result.sft_count + result.kto_pos_count
            + result.kto_neg_count + result.grpo_count
        )
        result.file_paths = {k: str(v) for k, v in file_paths.items()}
        result.content_hash = content_hash
        result.run_id = run_id

        logger.info(
            "Staged dataset %s: sft=%d kto_pos=%d kto_neg=%d grpo=%d",
            version_id, result.sft_count, result.kto_pos_count,
            result.kto_neg_count, result.grpo_count,
        )
        if not self._filter_set.is_empty and not filter_stats.is_empty:
            logger.info(
                "Staging filter breakdown (per target): %s",
                filter_stats.as_dict(),
            )
        return result

    @staticmethod
    def _filter_view(record: InferenceLogRecord, content: dict) -> dict[str, Any]:
        """Build the dot-path namespace a staging filter evaluates against.

        Filters in the flywheel address an inference-log record, not an
        environment rollout, so the namespace differs from the projector.
        Exactly two layers are exposed:

        * Record-level fields at the TOP level — the real
          :class:`~shared.flywheel.catalog.InferenceLogRecord` dataclass fields
          (via ``dataclasses.asdict``). Notable keys a researcher can address:
          ``log_id``, ``timestamp``, ``model_id``, ``adapter_name``,
          ``temperature``, ``max_tokens``, ``tools_requested``,
          ``finish_reason``, ``prompt_tokens``, ``completion_tokens``,
          ``latency_ms``, ``fitness_score``, ``is_valid``, ``tag``,
          ``dataset_version``, ``source_file``, ``line_number``, ``tenant_id``.
          (``messages``, ``tools``, ``tool_calls``, ``response_content`` and the
          token-id capture fields also exist on the dataclass but, for records
          read from the catalog index, only the index-backed fields are
          populated — prefer the ``content.*`` view below for transcript data.)

        * The parsed log content under a ``content.`` prefix — e.g.
          ``content.messages``, ``content.response_content``,
          ``content.tool_calls``, ``content.completion_token_ids``. This is the
          authoritative source of the request/response transcript.

        So a researcher can write ``field: fitness_score`` or
        ``field: content.response_content``. Record fields take the top level;
        the raw content dict is nested under ``content`` (never merged, so a
        ``content`` key inside the log cannot shadow a record field).
        """
        from dataclasses import asdict

        view: dict[str, Any] = asdict(record)
        view["content"] = content
        return view

    def _passes_filters(
        self,
        record: InferenceLogRecord,
        content: dict,
        target: str,
        stats: FilterStats | None,
    ) -> bool:
        """Apply the configured filter set to ``record`` for one staging target.

        Returns True when the record passes (or when no filters are
        configured). Records the decision into ``stats`` for the per-target /
        per-filter breakdown. Mirrors ``_passes_filters`` in the SynthChat
        projector, but with the stager's targets and filter view.
        """
        if self._filter_set.is_empty:
            return True
        view = self._filter_view(record, content)
        decision = self._filter_set.apply(view, target)
        if stats is not None:
            stats.record(target, self._filter_set, decision)
        return decision.passed

    def _next_version_id(self) -> str:
        """Determine next version ID by scanning datasets directory."""
        if not self._datasets_dir.exists():
            return "v001"

        existing = sorted(
            d.name for d in self._datasets_dir.iterdir()
            if d.is_dir() and d.name.startswith("v")
        )
        if not existing:
            return "v001"

        last = existing[-1]
        try:
            num = int(last[1:])
            return f"v{num + 1:03d}"
        except ValueError:
            return "v001"

    def _write_sft(
        self,
        logs: list[InferenceLogRecord],
        path: Path,
        stats: FilterStats | None = None,
    ) -> int:
        """Write SFT training examples (label: true).

        Records that fail the configured staging filters for the ``"sft"``
        target are skipped, so the returned count is the post-filter count.
        """
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for record in logs:
                content = self._read_log_content(record)
                if not content:
                    continue
                if not self._passes_filters(record, content, "sft", stats):
                    continue
                example = self._format_sft_example(record, content)
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        return count

    def _write_kto(
        self,
        sft_logs: list[InferenceLogRecord],
        kto_logs: list[InferenceLogRecord],
        path: Path,
        stats: FilterStats | None = None,
    ) -> tuple[int, int]:
        """Write KTO training examples with interleaved positive/negative pairs.

        Per KTO_TRAINING_REFERENCE.md, KTO training requires interleaved
        true/false examples. We zip positives and negatives, writing
        alternating pairs. When one list is exhausted, remaining items
        from the longer list are appended at the end.

        Staging filters are applied per-target BEFORE the pos/neg example lists
        are built, so the zip_longest interleaving operates on already-filtered
        lists and stays intact. Positives (from ``sft_logs``) use the
        ``"kto_positive"`` target; negatives (from ``kto_logs``) use the
        ``"kto_negative"`` target — which is excluded from the default target
        set, so hard negatives are never dropped unless a filter explicitly
        opts in via ``applies_to: [kto_negative]``.
        """
        pos_examples: list[dict] = []
        neg_examples: list[dict] = []

        for record in sft_logs:
            content = self._read_log_content(record)
            if not content:
                continue
            if not self._passes_filters(record, content, "kto_positive", stats):
                continue
            pos_examples.append(self._format_kto_example(record, content, label=True))

        for record in kto_logs:
            content = self._read_log_content(record)
            if not content:
                continue
            if not self._passes_filters(record, content, "kto_negative", stats):
                continue
            neg_examples.append(self._format_kto_example(record, content, label=False))

        # Interleave: alternate positive/negative, then append remainder
        with open(path, "w", encoding="utf-8") as f:
            from itertools import zip_longest

            for pos, neg in zip_longest(pos_examples, neg_examples):
                if pos is not None:
                    f.write(json.dumps(pos, ensure_ascii=False) + "\n")
                if neg is not None:
                    f.write(json.dumps(neg, ensure_ascii=False) + "\n")

        return len(pos_examples), len(neg_examples)

    def _write_grpo(
        self,
        logs: list[InferenceLogRecord],
        path: Path,
        stats: FilterStats | None = None,
    ) -> int:
        """Write GRPO training examples (with reward signal).

        Records that fail the configured staging filters for the ``"grpo"``
        target are skipped, so the returned count is the post-filter count.
        """
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for record in logs:
                content = self._read_log_content(record)
                if not content:
                    continue
                if not self._passes_filters(record, content, "grpo", stats):
                    continue
                example = self._format_grpo_example(record, content)
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        return count

    def _format_sft_example(
        self, record: InferenceLogRecord, content: dict,
    ) -> dict:
        """Format a log as an SFT training example."""
        conversations = self._build_conversations(content)
        return {"conversations": conversations, "label": True}

    def _format_kto_example(
        self, record: InferenceLogRecord, content: dict, *, label: bool,
    ) -> dict:
        """Format a log as a KTO training example."""
        conversations = self._build_conversations(content)
        return {"conversations": conversations, "label": label}

    def _format_grpo_example(
        self, record: InferenceLogRecord, content: dict,
    ) -> dict:
        """Format a log as a GRPO training example.

        When the log carries token-faithful rollout capture (completion token
        ids + per-token logprobs, from a logprobs-enabled proxy), those fields
        are passed through alongside the conversational form so a downstream
        token-faithful / replay GRPO consumer can use the exact sampled tokens.
        Records without capture are unchanged.
        """
        conversations = self._build_conversations(content)
        reward = (record.fitness_score or 0.0) * self._config.grpo_reward_scale
        example: dict[str, Any] = {"conversations": conversations, "reward": reward}

        for key in ("prompt_token_ids", "completion_token_ids", "completion_logprobs"):
            value = content.get(key)
            if value is not None:
                example[key] = value
        return example

    @staticmethod
    def _build_conversations(content: dict) -> list[dict[str, str]]:
        """Build ChatML conversations from log content."""
        messages = content.get("messages", [])
        response = content.get("response_content", "")

        conversations: list[dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role in ("user", "assistant", "system"):
                conversations.append({"role": role, "content": text})

        # Append the assistant response
        if response:
            conversations.append({"role": "assistant", "content": response})

        return conversations

    @staticmethod
    def _read_log_content(record: InferenceLogRecord) -> dict | None:
        """Read full log content from the source JSONL file."""
        return read_log_content(record)

    def _compute_content_hash(self, file_paths: dict[str, Path]) -> str:
        """SHA-256 hash of all staged dataset files."""
        h = hashlib.sha256()
        for key in sorted(file_paths.keys()):
            path = file_paths[key]
            if path.exists():
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
        return h.hexdigest()

    def _register_flywheel_cycle(self, version: DatasetVersion) -> str:
        """Register this staging cycle in RunRegistry and return run_id.

        Uses flywheel_cycle_to_run_record() adapter from
        shared/experiment_tracking/adapters.py.
        """
        try:
            from shared.experiment_tracking.adapters import (
                flywheel_cycle_to_run_record,
            )
            from shared.experiment_tracking.registry import RunRegistry

            cycle_data = {
                "version_id": version.version_id,
                "record_counts": version.record_counts,
                "filter_criteria": version.filter_criteria,
                "source_model_id": version.source_model_id,
                "content_hash": version.content_hash,
            }

            record = flywheel_cycle_to_run_record(
                cycle_data,
                output_dir=str(self._datasets_dir / version.version_id),
            )

            registry = RunRegistry()
            run_id = registry.register_run(record)
            logger.info(
                "Flywheel cycle registered: %s (version %s)",
                run_id, version.version_id,
            )
            return run_id

        except Exception as exc:
            logger.error(
                "Failed to register flywheel cycle in RunRegistry: %s", exc,
            )
            return ""
