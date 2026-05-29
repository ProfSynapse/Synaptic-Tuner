"""Evaluation adapters for prompt optimization scoring."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .service import PromptOptimizationError, PromptSubject


class EvaluatorScoringConfigError(PromptOptimizationError):
    """Raised for invalid Evaluator scoring configuration."""


@dataclass(frozen=True)
class PromptEvaluationScore:
    score: float
    metrics: dict[str, Any]
    passed: bool
    diagnostics: tuple[dict[str, Any], ...] = ()
    assertion_results: tuple[dict[str, Any], ...] = ()


class EvaluatorScoringAdapter:
    """Score prompt genomes through the config-first Evaluator runtime."""

    def __init__(
        self,
        *,
        evaluation_config: Mapping[str, Any],
        config_path: Path,
        repo_root: Path,
        score_floor: float,
    ) -> None:
        self.evaluation_config = dict(evaluation_config)
        self.config_path = config_path
        self.repo_root = repo_root
        self.score_floor = _clamp_score(score_floor)
        self.evaluator_config = _mapping(
            self.evaluation_config.get("evaluator") or self.evaluation_config,
            "evaluation.evaluator",
        )
        self.failure_policy = str(self.evaluator_config.get("failure_policy", "score_floor")).strip().lower()
        if self.failure_policy not in {"score_floor", "raise"}:
            raise EvaluatorScoringConfigError("evaluation.evaluator.failure_policy must be 'score_floor' or 'raise'.")
        self.objective = _mapping(self.evaluator_config.get("objective"), "evaluation.evaluator.objective")
        self.objective_metric = str(self.objective.get("metric") or "stats.normalized_score")
        try:
            self.pass_threshold = float(self.objective.get("pass_threshold", 1.0))
        except (TypeError, ValueError) as exc:
            raise EvaluatorScoringConfigError("evaluation.evaluator.objective.pass_threshold must be numeric.") from exc
        placement_config = self.evaluator_config.get("prompt_placement")
        placement_label = "evaluation.evaluator.prompt_placement"
        if placement_config is None and "prompt_injection" in self.evaluator_config:
            placement_config = self.evaluator_config.get("prompt_injection")
            placement_label = "evaluation.evaluator.prompt_injection"
        self.prompt_placement = _mapping(placement_config, placement_label)
        self.placement_mode = str(self.prompt_placement.get("mode") or "").strip()
        if self.placement_mode != "system_overlay":
            raise EvaluatorScoringConfigError(
                "evaluation.evaluator.prompt_placement.mode must be 'system_overlay' for Evaluator scoring."
            )
        self.overlay_template = str(
            self.prompt_placement.get("template")
            or self.prompt_placement.get("overlay_template")
            or "{candidate_prompt}\n\n{system}"
        )
        if "{candidate_prompt}" not in self.overlay_template:
            raise EvaluatorScoringConfigError(
                "evaluation.evaluator.prompt_placement.template must include {candidate_prompt}."
            )
        self.join_with = str(self.prompt_placement.get("join_with", "\n\n"))

    def score(self, genome_values: Mapping[str, str], subjects: Sequence[PromptSubject]) -> PromptEvaluationScore:
        return self._score(genome_values, subjects)

    def _score(self, genome_values: Mapping[str, str], subjects: Sequence[PromptSubject]) -> PromptEvaluationScore:
        from Evaluator.client_factory import create_client, create_settings
        from Evaluator.config_loader import ConfigLoader
        from Evaluator.reporting import aggregate_stats
        from Evaluator.runner import evaluate_cases

        config_dir = self._resolve_path(str(self.evaluator_config.get("config_dir", "Evaluator/config")))
        loader = ConfigLoader(config_dir)

        preset = self.evaluator_config.get("preset")
        scenario_files = self.evaluator_config.get("scenarios")
        if scenario_files is not None and not isinstance(scenario_files, list):
            raise EvaluatorScoringConfigError("evaluation.evaluator.scenarios must be a list when provided.")
        tag_filter = self.evaluator_config.get("tag_filter")
        exclude_tags = self.evaluator_config.get("exclude_tags")
        try:
            case_paths = scenario_files or loader.load_eval_run(str(preset) if preset else None).scenarios
            cases = loader.load_all_scenarios(
                case_paths,
                tag_filter=_string_list(tag_filter),
                exclude_tags=_string_list(exclude_tags),
            )
        except Exception as exc:
            raise EvaluatorScoringConfigError(f"Evaluator scoring config could not load cases: {exc}") from exc
        if not cases:
            raise EvaluatorScoringConfigError("Evaluator scoring loaded no cases.")

        candidate_prompt = self._candidate_prompt(genome_values, subjects)
        injected_cases = [self._inject_case(case, candidate_prompt) for case in cases]

        dry_run = bool(self.evaluator_config.get("dry_run", False))
        client = _DryRunClient()
        if not dry_run:
            model = _mapping(self.evaluator_config.get("model"), "evaluation.evaluator.model")
            if not model:
                run_config = loader.load_eval_run(str(preset) if preset else None)
                model = {
                    "backend": run_config.model_backend,
                    "name": run_config.model_name,
                    "temperature": run_config.temperature,
                    "max_tokens": run_config.max_tokens,
                    "seed": run_config.seed,
                }
            backend = str(model.get("backend") or "")
            model_name = str(model.get("name") or model.get("model") or "")
            if not backend or not model_name:
                raise EvaluatorScoringConfigError(
                    "evaluation.evaluator.model.backend and model.name are required when dry_run is false."
                )
            try:
                settings = create_settings(
                    backend=backend,
                    model=model_name,
                    host=model.get("host"),
                    port=model.get("port"),
                    temperature=float(model.get("temperature", 0.2)),
                    top_p=float(model.get("top_p", 0.9)),
                    max_tokens=int(model.get("max_tokens", 1024)),
                    seed=model.get("seed"),
                )
                client = create_client(
                    backend=backend,
                    settings=settings,
                    timeout=float(model.get("timeout", 60.0)),
                    retries=int(model.get("retries", 2)),
                )
            except Exception as exc:
                raise EvaluatorScoringConfigError(f"Evaluator scoring model config is invalid: {exc}") from exc

        try:
            max_workers = int(self.evaluator_config.get("max_workers", 4))
        except (TypeError, ValueError) as exc:
            raise EvaluatorScoringConfigError("evaluation.evaluator.max_workers must be an integer.") from exc

        try:
            records = evaluate_cases(
                injected_cases,
                client,
                dry_run=dry_run,
                parallel=bool(self.evaluator_config.get("parallel", False)),
                max_workers=max_workers,
            )
        except Exception as exc:
            if self.failure_policy == "raise":
                raise
            return self._runtime_failure_score(exc)
        stats = aggregate_stats(records)
        metric_context = {"stats": stats, "records": [_record_summary(record) for record in records]}
        raw_score = _resolve_metric(metric_context, self.objective_metric)
        normalized = _normalize_score(raw_score, self.objective)
        return PromptEvaluationScore(
            score=normalized,
            metrics={
                "normalized_score": normalized,
                "objective_metric": self.objective_metric,
                "objective_value": raw_score,
                "evaluator_stats": stats,
                "case_count": len(records),
            },
            passed=normalized >= self.pass_threshold,
            diagnostics=tuple(_record_diagnostics(records)),
        )

    def _runtime_failure_score(self, exc: Exception) -> PromptEvaluationScore:
        diagnostic = {
            "code": "EVALUATOR_SCORING_FAILED",
            "message": str(exc),
            "severity": "error",
        }
        return PromptEvaluationScore(
            score=self.score_floor,
            metrics={
                "normalized_score": self.score_floor,
                "objective_metric": self.objective_metric,
                "failure_policy": self.failure_policy,
            },
            passed=False,
            diagnostics=(diagnostic,),
        )

    def _candidate_prompt(self, genome_values: Mapping[str, str], subjects: Sequence[PromptSubject]) -> str:
        subject_id = self.prompt_placement.get("subject_id")
        if subject_id is not None:
            key = str(subject_id)
            if key not in genome_values:
                raise EvaluatorScoringConfigError(
                    f"evaluation.evaluator.prompt_placement.subject_id references unknown subject: {key}"
                )
            return str(genome_values[key])
        return self.join_with.join(str(genome_values[subject.id]) for subject in subjects)

    def _inject_case(self, case: Any, candidate_prompt: str) -> Any:
        injected = deepcopy(case)
        system = str(injected.metadata.get("system", ""))
        injected.metadata["system"] = self.overlay_template.format(
            candidate_prompt=candidate_prompt,
            system=system,
        )
        messages = injected.metadata.get("messages")
        if isinstance(messages, list) and messages:
            injected.metadata["messages"] = _overlay_system_message(
                messages,
                injected.metadata["system"],
            )
        return injected

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        candidates = [path] if path.is_absolute() else [self.repo_root / path, self.config_path.parent / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise EvaluatorScoringConfigError(f"Evaluator config path not found: {raw_path}")


class _DryRunClient:
    def chat(self, messages: list[dict[str, str]]) -> Any:
        raise RuntimeError("Dry-run Evaluator scoring should not call the backend client.")


def _overlay_system_message(messages: list[Any], system_prompt: str) -> list[Any]:
    updated = []
    replaced = False
    for message in messages:
        if isinstance(message, Mapping) and str(message.get("role", "")).strip() == "system":
            item = dict(message)
            item["content"] = system_prompt
            updated.append(item)
            replaced = True
        else:
            updated.append(deepcopy(message))
    if not replaced:
        updated.insert(0, {"role": "system", "content": system_prompt})
    return updated


def _resolve_metric(context: Mapping[str, Any], metric_path: str) -> float:
    current: Any = context
    for part in metric_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise EvaluatorScoringConfigError(f"Evaluator objective metric not found: {metric_path}")
    try:
        return float(current)
    except (TypeError, ValueError) as exc:
        raise EvaluatorScoringConfigError(f"Evaluator objective metric is not numeric: {metric_path}") from exc


def _normalize_score(value: float, objective: Mapping[str, Any]) -> float:
    if "min" in objective or "max" in objective:
        try:
            minimum = float(objective.get("min", 0.0))
            maximum = float(objective.get("max", 1.0))
        except (TypeError, ValueError) as exc:
            raise EvaluatorScoringConfigError("evaluation.evaluator.objective min/max must be numeric.") from exc
        if maximum <= minimum:
            raise EvaluatorScoringConfigError("evaluation.evaluator.objective.max must be greater than min.")
        value = (value - minimum) / (maximum - minimum)
    elif bool(objective.get("percentage", False)):
        value = value / 100.0
    return _clamp_score(value)


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _record_summary(record: Any) -> dict[str, Any]:
    scoring = record.scoring.to_dict() if getattr(record, "scoring", None) is not None else None
    return {
        "case_id": getattr(record.case, "case_id", ""),
        "status": record.status,
        "passed": record.passed,
        "error": record.error,
        "score": record.score,
        "scoring": scoring,
    }


def _record_diagnostics(records: Iterable[Any]) -> list[dict[str, Any]]:
    diagnostics = []
    for record in records:
        if getattr(record, "error", None):
            diagnostics.append(
                {
                    "code": "EVALUATOR_CASE_ERROR",
                    "case_id": getattr(record.case, "case_id", ""),
                    "message": str(record.error),
                    "severity": "error",
                }
            )
    return diagnostics


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise EvaluatorScoringConfigError("Evaluator tag filters must be lists or strings.")
    return [str(item) for item in value]


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EvaluatorScoringConfigError(f"{label} must be a mapping when provided.")
    return dict(value)
