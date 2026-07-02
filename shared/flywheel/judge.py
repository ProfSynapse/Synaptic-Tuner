"""Structured judge adapter and metadata helpers for flywheel artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from shared.judge import JudgeConfig, JudgeResult, JudgeScore, JudgeService, RubricDef
from shared.llm import create_client
from shared.validation.fitness import FitnessResult

from .catalog import InferenceLogRecord
from .config import FlywheelConfig, FlywheelJudgeConfig


@dataclass
class FlywheelJudgeOutcome:
    """Structured flywheel judge result used by cleaner/tagger/stagers."""

    passed: bool
    verdict_rationale: str
    rubric_scores: list[dict[str, Any]]
    raw_output: dict[str, Any] | None = None
    error: str | None = None
    latency_s: float | None = None

    @property
    def score(self) -> float:
        if not self.rubric_scores:
            return 0.0
        return min(
            1.0,
            max(float(score.get("score", 0.0)) for score in self.rubric_scores),
        )

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "passed": self.passed,
            "verdict_rationale": self.verdict_rationale,
            "rubric_scores": self.rubric_scores,
        }
        if self.error:
            metadata["error"] = self.error
        if self.latency_s is not None:
            metadata["latency_s"] = self.latency_s
        return metadata


class FlywheelJudge:
    """Small adapter from flywheel records to shared.judge.JudgeService."""

    def __init__(
        self,
        config: FlywheelJudgeConfig,
        judge_service: JudgeService,
    ) -> None:
        self._config = config
        self._judge_service = judge_service
        self._rubric = _rubric_from_config(config.rubric)

    @classmethod
    def from_flywheel_config(
        cls,
        config: FlywheelConfig,
        judge_service: JudgeService | None = None,
    ) -> FlywheelJudge | None:
        if not config.judge.enabled:
            return None
        service = judge_service or JudgeService(
            llm_client=create_client(
                env_prefix=config.judge.env_prefix,
                config_defaults=config.judge.llm,
            ),
            judge_config=JudgeConfig(**config.judge.judge_config),
        )
        return cls(config.judge, service)

    def judge_record(
        self,
        record: InferenceLogRecord,
        content: dict[str, Any] | None,
        fitness: FitnessResult | None = None,
    ) -> FlywheelJudgeOutcome:
        prompt = self._render_prompt(record, content or {}, fitness)
        result = self._judge_service.judge(
            prompt=prompt,
            rubrics=[self._rubric],
            system_prompt=self._config.system_prompt,
        )
        return outcome_from_judge_result(result)

    def _render_prompt(
        self,
        record: InferenceLogRecord,
        content: dict[str, Any],
        fitness: FitnessResult | None,
    ) -> str:
        response_content = str(
            content.get("response_content", record.response_content) or ""
        )
        if self._config.max_response_chars > 0:
            response_content = response_content[: self._config.max_response_chars]

        values = {
            "rubric_prompt": self._rubric.judge_prompt,
            "log_id": record.log_id,
            "fitness_score": (
                fitness.score if fitness is not None else record.fitness_score
            ),
            "is_valid": fitness.is_valid if fitness is not None else record.is_valid,
            "errors": "; ".join(fitness.errors) if fitness else "; ".join(record.errors),
            "messages_json": json.dumps(
                content.get("messages", record.messages),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "response_content": response_content,
            "tool_calls_json": json.dumps(
                content.get("tool_calls", record.tool_calls),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        return self._config.prompt_template.format(**values)


def coerce_flywheel_judge(
    config: FlywheelConfig,
    judge: Any | None,
) -> FlywheelJudge | Any | None:
    """Normalize supported injected judge/client objects.

    Supported forms:
    - None: build from config only when config.judge.enabled is true.
    - object with judge_record(...): already a flywheel adapter/test double.
    - object with judge(...): shared.judge.JudgeService-compatible service.
    - object with structured_output(...): raw shared LLM client.
    """
    if judge is None:
        return FlywheelJudge.from_flywheel_config(config)
    if hasattr(judge, "judge_record"):
        return judge
    if hasattr(judge, "judge"):
        return FlywheelJudge(config.judge, judge)
    if hasattr(judge, "structured_output"):
        return FlywheelJudge(
            config.judge,
            JudgeService(
                llm_client=judge,
                judge_config=JudgeConfig(**config.judge.judge_config),
            ),
        )
    raise TypeError(
        "judge must expose judge_record(...), judge(...), or structured_output(...)"
    )


def outcome_from_judge_result(result: JudgeResult) -> FlywheelJudgeOutcome:
    rationale = result.error or _feedback_from_scores(result.scores) or (
        "structured judge returned no rationale"
    )
    return FlywheelJudgeOutcome(
        passed=result.passed,
        verdict_rationale=rationale,
        rubric_scores=[_score_to_dict(score) for score in result.scores],
        raw_output=result.raw_output,
        error=result.error,
        latency_s=result.latency_s,
    )


def judge_metadata_from_record(record: InferenceLogRecord) -> dict[str, Any] | None:
    return build_judge_metadata(
        verdict_rationale=record.verdict_rationale,
        rubric_scores=record.rubric_scores,
    )


def build_judge_metadata(
    *,
    verdict_rationale: str | None,
    rubric_scores: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not _has_structured_rubric_scores(rubric_scores):
        return None
    metadata: dict[str, Any] = {}
    if verdict_rationale is not None:
        metadata["verdict_rationale"] = verdict_rationale
    if rubric_scores is not None:
        metadata["rubric_scores"] = rubric_scores
    return metadata


def judge_metadata_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict):
        flywheel = metadata.get("flywheel") or {}
        if isinstance(flywheel, dict) and isinstance(flywheel.get("judge"), dict):
            return _structured_metadata_or_none(flywheel["judge"])
        if isinstance(metadata.get("judge"), dict):
            return _structured_metadata_or_none(metadata["judge"])
    return build_judge_metadata(
        verdict_rationale=row.get("verdict_rationale"),
        rubric_scores=row.get("rubric_scores"),
    )


def attach_flywheel_judge_metadata(
    row: dict[str, Any],
    record: InferenceLogRecord,
) -> dict[str, Any]:
    judge = judge_metadata_from_record(record)
    if not judge:
        return row
    metadata = dict(row.get("metadata") or {})
    flywheel = dict(metadata.get("flywheel") or {})
    flywheel["log_id"] = record.log_id
    flywheel["judge"] = judge
    metadata["flywheel"] = flywheel
    row["metadata"] = metadata
    return row


def _rubric_from_config(data: dict[str, Any]) -> RubricDef:
    return RubricDef(
        key=str(data.get("key") or "flywheel_quality"),
        name=str(data["name"]),
        description=str(data["description"]),
        scope=str(data.get("scope") or "response"),
        pass_threshold=float(data["pass_threshold"]),
        judge_prompt=str(data["judge_prompt"]),
        output_schema=dict(data["output_schema"]),
        improver_prompt=data.get("improver_prompt"),
        dimensions=data.get("dimensions"),
        weights_ratified=bool(data.get("weights_ratified", False)),
        quality_gate=data.get("quality_gate"),
    )


def _structured_metadata_or_none(value: dict[str, Any]) -> dict[str, Any] | None:
    if value.get("structured") is True:
        return dict(value)
    if _has_structured_rubric_scores(value.get("rubric_scores")):
        return dict(value)
    return None


def _has_structured_rubric_scores(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def _score_to_dict(score: JudgeScore) -> dict[str, Any]:
    return {
        "rubric_key": score.rubric_key,
        "rubric_name": score.rubric_name,
        "score": score.score,
        "passed": score.passed,
        "pass_threshold": score.pass_threshold,
        "feedback": score.feedback,
        "per_dimension": score.per_dimension,
        "quality_gated_score": score.quality_gated_score,
    }


def _feedback_from_scores(scores: list[JudgeScore]) -> str | None:
    feedback = [score.feedback for score in scores if score.feedback]
    if not feedback:
        return None
    return " ".join(feedback)
