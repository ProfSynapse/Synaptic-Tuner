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
        judge_validator = None
        if not dry_run:
            model = _mapping(self.evaluator_config.get("model"), "evaluation.evaluator.model")
            if not model:
                run_config = loader.load_eval_run(str(preset) if preset else None)
                model = {
                    "backend": run_config.model_backend,
                    "name": run_config.model_name,
                    "max_tokens": run_config.max_tokens,
                    "seed": run_config.seed,
                }
                # Temperature is OPT-IN. EvalRunConfig.temperature is now
                # Optional[float] (config_loader.py): None when the eval_run.yaml
                # omits it, the set value when present. Forward it ONLY when
                # explicitly set — None falls through to the omit-aware
                # model.get("temperature") logic below, so the request omits
                # temperature and gpt-5-family reasoning targets work. An
                # explicit value IS forwarded (preserves control for
                # non-reasoning backends); without this line an explicit
                # eval-run temperature would be a silent no-op.
                if run_config.temperature is not None:
                    model["temperature"] = run_config.temperature
            backend = str(model.get("backend") or "")
            model_name = str(model.get("name") or model.get("model") or "")
            if not backend or not model_name:
                raise EvaluatorScoringConfigError(
                    "evaluation.evaluator.model.backend and model.name are required when dry_run is false."
                )
            try:
                # Omit temperature by default: gpt-5-family reasoning targets
                # (e.g. gpt-5-nano) reject any temperature with HTTP 400, and
                # create_settings + the openai_responses provider already omit
                # the field when None. An explicitly-configured temperature is
                # still forwarded as a float (preserves control for non-reasoning
                # targets); only the default changes from 0.2 to omitted.
                _temp = model.get("temperature")
                temperature = float(_temp) if _temp is not None else None
                # Thread the canonical thinking_effort knob (upstream #98).
                # Accept the deprecated reasoning_effort key as an alias so
                # existing model configs keep working. DEFAULT IS NONE (effort
                # omitted unless the config sets it): per the #98 reconcile we no
                # longer force "minimal", which is unsupported on some judge/target
                # models (e.g. gpt-5.4-mini) and would HTTP-400. Models wanting
                # short outputs opt in explicitly (e.g. gpt-5-nano: minimal).
                _effort = model.get("thinking_effort", model.get("reasoning_effort"))
                thinking_effort = str(_effort) if _effort is not None else None
                # top_p is intentionally left as-is: the openai_responses adapter
                # and provider never forward top_p to the Responses API payload
                # (only model/input/max_output_tokens/store + conditional
                # temperature/reasoning), so it is dropped before the wire for
                # the reasoning backend and cannot trigger a 400. Backends that
                # do send top_p (lmstudio/llamacpp/mlc/unsloth) are not reasoning
                # models and 0.9 is the correct default for them.
                settings = create_settings(
                    backend=backend,
                    model=model_name,
                    host=model.get("host"),
                    port=model.get("port"),
                    temperature=temperature,
                    top_p=float(model.get("top_p", 0.9)),
                    max_tokens=int(model.get("max_tokens", 1024)),
                    seed=model.get("seed"),
                    thinking_effort=thinking_effort,
                )
                client = create_client(
                    backend=backend,
                    settings=settings,
                    timeout=float(model.get("timeout", 60.0)),
                    retries=int(model.get("retries", 2)),
                )
            except Exception as exc:
                raise EvaluatorScoringConfigError(f"Evaluator scoring model config is invalid: {exc}") from exc

            # Build the independent judge (if a judge block is configured) AFTER
            # the target client so we can enforce R1 model-independence against
            # the resolved target model_name. None when no judge block is present
            # (structural-only optimization). Dry-run never judges (the
            # _DryRunClient would refuse the call anyway).
            judge_validator = self._build_judge_validator(model_name)

        try:
            max_workers = int(self.evaluator_config.get("max_workers", 4))
        except (TypeError, ValueError) as exc:
            raise EvaluatorScoringConfigError("evaluation.evaluator.max_workers must be an integer.") from exc

        try:
            records = evaluate_cases(
                injected_cases,
                client,
                dry_run=dry_run,
                judge_validator=judge_validator,
                parallel=bool(self.evaluator_config.get("parallel", False)),
                max_workers=max_workers,
            )
        except Exception as exc:
            if self.failure_policy == "raise":
                raise
            return self._runtime_failure_score(exc)
        stats = aggregate_stats(records)
        metric_context = {"stats": stats, "records": [_record_summary(record) for record in records]}
        # When selecting on the judge gradient (stats.judge_normalized_score), a
        # candidate whose cases were ALL skipped/failed before the judge ran (e.g.
        # "and"-mode cheap-gate rejected every case, or the judge errored on all)
        # produces a None gradient. Treat that as the worst candidate (score_floor)
        # instead of crashing _resolve_metric on a non-numeric None — the optimizer
        # must keep ranking. Other metrics keep their strict numeric contract.
        if self._judge_metric_unresolved(stats):
            return self._runtime_failure_score(
                EvaluatorScoringConfigError(
                    "No case produced a judge gradient (judge_normalized_score is None): "
                    "candidate scored at the floor."
                )
            )
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

    def _judge_metric_unresolved(self, stats: Mapping[str, Any]) -> bool:
        """True when the objective targets the judge gradient but it is None.

        Only fires for the stats.judge_normalized_score objective (the judge
        gradient is None when no case was judged). Any other objective keeps the
        strict numeric contract enforced by _resolve_metric. Honours
        failure_policy='raise' by NOT short-circuiting (the caller proceeds to
        _resolve_metric, which raises on the None as before).
        """
        if self.objective_metric != "stats.judge_normalized_score":
            return False
        if self.failure_policy == "raise":
            return False
        return stats.get("judge_normalized_score") is None

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

    def _build_judge_validator(self, target_model_name: str) -> Any | None:
        """Construct a JudgeValidator from evaluation.evaluator.judge, or None.

        Mirrors the CLI judge-build (Evaluator/cli.py:684-757) so the optimizer's
        candidate-scoring path populates record.judge — without this, the adapter
        calls evaluate_cases with no judge_validator and record.judge is always
        None (the judge gradient is dormant on the optimizer path). The judge LLM
        client is built independently of the eval/target client, so the judge can
        (and for R1 anti-circularity, MUST) be a model distinct from the target.

        Returns None when no `judge` block is configured (structural-only
        optimization is still valid). Raises EvaluatorScoringConfigError on a
        misconfigured judge block — fail loud rather than silently scoring on a
        non-judge gradient when a judge was requested.

        R1 (judge-model independence) is enforced here: if the judge model equals
        the target model we refuse to build, because an identical judge cannot
        provide an independent gradient (the optimizer would select on the
        target grading itself — circular).
        """
        judge_block = self.evaluator_config.get("judge")
        if not judge_block:
            return None
        judge_cfg = _mapping(judge_block, "evaluation.evaluator.judge")

        try:
            from shared.llm import LLMConfig, create_client as create_llm_client
            from shared.judge import JudgeConfig, RubricLoader, InteractionLogger
            from Evaluator.judge_validator import JudgeValidator
        except ImportError as exc:  # pragma: no cover - environment guard
            raise EvaluatorScoringConfigError(f"Judge dependencies unavailable: {exc}") from exc

        provider = str(judge_cfg.get("provider") or "").strip()
        judge_model_name = str(judge_cfg.get("model") or "").strip()
        if not judge_model_name:
            raise EvaluatorScoringConfigError(
                "evaluation.evaluator.judge.model is required when a judge block is present."
            )

        # R1 anti-circularity: the judge must be independent of the target. A
        # judge identical to the target grades the target's own family — no
        # independent gradient — so we refuse at build time (fail loud).
        if judge_model_name == str(target_model_name).strip():
            raise EvaluatorScoringConfigError(
                "evaluation.evaluator.judge.model must differ from the target model "
                f"for R1 independence (both are '{judge_model_name}')."
            )

        # Judge LLM client: built from JUDGE-prefixed env, then overridden by the
        # explicit provider/model in the YAML (same precedence as cli.py). This is
        # a SEPARATE client from the target/eval client above — independent path.
        judge_llm_config = LLMConfig.from_env(env_prefix="JUDGE")
        if provider:
            judge_llm_config.provider = provider
        judge_llm_config.model = judge_model_name
        try:
            judge_llm_client = create_llm_client(config=judge_llm_config)
        except Exception as exc:
            raise EvaluatorScoringConfigError(f"Judge LLM client config is invalid: {exc}") from exc

        rubrics_dir = self._resolve_judge_rubrics_dir(str(judge_cfg.get("rubrics_dir") or ""))
        loader = RubricLoader(rubrics_dir)
        rubric_keys = _string_list(judge_cfg.get("rubrics")) or loader.list_available()
        if not rubric_keys:
            raise EvaluatorScoringConfigError(f"No judge rubrics found in {rubrics_dir}.")
        try:
            rubrics = loader.load_many(rubric_keys)
        except Exception as exc:
            raise EvaluatorScoringConfigError(f"Judge rubrics could not load: {exc}") from exc

        # Interaction logging is OFF on the optimizer path: candidate scoring runs
        # many judge calls per generation and KTO-style interaction capture is a
        # CLI-eval concern, not an optimization concern. Opt in via judge.log_interactions.
        interaction_logger = None
        if bool(judge_cfg.get("log_interactions", False)):
            from pathlib import Path as _Path

            interaction_logger = InteractionLogger(
                output_dir=_Path("Evaluator/interactions"),
                enabled=True,
                prefix="judge_opt",
            )

        # temperature/reasoning_effort default to JudgeConfig's omit-aware values
        # (temperature=None so gpt-5-family reasoning judges don't get a 400;
        # reasoning_effort="minimal"). Both overridable via the judge block.
        judge_config = JudgeConfig(
            temperature=judge_cfg.get("temperature"),
            max_tokens=int(judge_cfg.get("max_tokens", 2048)),
            reasoning_effort=judge_cfg.get("reasoning_effort", "minimal"),
        )
        return JudgeValidator(
            llm_client=judge_llm_client,
            rubrics=rubrics,
            judge_config=judge_config,
            interaction_logger=interaction_logger,
            default_judge_mode=str(judge_cfg.get("mode", "and")),
            eval_model=target_model_name,
            judge_model=judge_model_name,
        )

    def _resolve_judge_rubrics_dir(self, raw_path: str) -> Path:
        """Resolve the judge rubrics_dir, failing loud on an unsubstituted token.

        The active evolve configs carry rubrics_dir as "__SKILL_DIR__/configs/rubrics"
        and rely on the skill run.sh to sed-substitute the absolute path BEFORE
        invoking tuner.py (run.sh writes a substituted temp config). If the token
        survives to here, the YAML was run without that substitution (e.g. a raw
        `tuner.py prompt-optimize --prompt-opt-config generate_hashtags_evolve.yaml`):
        we refuse rather than search for a literal "__SKILL_DIR__" directory, and
        point the operator at the run.sh path.
        """
        if not raw_path:
            raise EvaluatorScoringConfigError(
                "evaluation.evaluator.judge.rubrics_dir is required when a judge block is present."
            )
        if "__SKILL_DIR__" in raw_path:
            raise EvaluatorScoringConfigError(
                "evaluation.evaluator.judge.rubrics_dir still contains the __SKILL_DIR__ "
                "placeholder. Run the eval via its skill run.sh (which substitutes the "
                "absolute path), or replace the token before invoking tuner.py directly. "
                f"Got: {raw_path}"
            )
        return self._resolve_path(raw_path)


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
