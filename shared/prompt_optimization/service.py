"""Config-first prompt optimization service.

It optimizes configured prompt text with YAML-declared operators and scores
candidates against fixture assertions declared in YAML. Operators are
deterministic unless the config explicitly selects an LLM-backed operator.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


class PromptOptimizationError(RuntimeError):
    """Raised when prompt optimization config or execution is invalid."""


@dataclass(frozen=True)
class PromptSubject:
    id: str
    source_path: str
    source_path_absolute: str
    source_path_repo_relative: str
    dotted_path: str
    original_value: Any
    prompt_text: str


@dataclass(frozen=True)
class PromptCandidate:
    id: str
    subject_id: str
    index: int
    operator: str
    prompt_text: str
    metadata: dict[str, Any]
    score: float = 0.0
    assertion_results: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PromptOptimizationResult:
    run_id: str
    output_dir: str
    best_candidate: dict[str, Any]
    candidate_count: int
    artifact_paths: dict[str, str]
    schema_version: int = 1
    strategy: str = "fixture"
    generation_count: int | None = None
    stop_reason: str | None = None
    best_score: float | None = None


@dataclass(frozen=True)
class PromptGenome:
    values: dict[str, str]


@dataclass(frozen=True)
class EvolutionCandidate:
    id: str
    generation: int
    index: int
    genome: PromptGenome
    parents: tuple[str, ...]
    operator: str
    metadata: dict[str, Any]
    score: float = 0.0
    metrics: dict[str, Any] | None = None
    passed: bool = False
    diagnostics: tuple[dict[str, Any], ...] = ()
    assertion_results: tuple[dict[str, Any], ...] = ()
    selected: bool = False


class PromptOptimizationService:
    """Run config-first prompt optimization from a YAML config."""

    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path.resolve()
        self.repo_root = _find_repo_root(self.config_path.parent)
        self._llm_clients: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> "PromptOptimizationService":
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise PromptOptimizationError(f"Prompt optimization config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        config = raw.get("prompt_optimization", raw)
        if not isinstance(config, dict):
            raise PromptOptimizationError("Prompt optimization config must be a YAML mapping.")
        if overrides:
            config = _deep_merge(config, overrides)
        return cls(config=config, config_path=config_path)

    def run(self) -> PromptOptimizationResult:
        if self._is_evolutionary_config():
            return self._run_evolutionary()
        return self._run_v1()

    def _is_evolutionary_config(self) -> bool:
        return int(self.config.get("schema_version", 1)) == 2 or str(self.config.get("strategy", "")).lower() == "evolutionary"

    def _run_v1(self) -> PromptOptimizationResult:
        run_id = str(self.config.get("run_id") or _default_run_id())
        seed = int(self.config.get("seed", 0))
        output_dir = self._resolve_output_dir(run_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        subjects = self._load_subjects()
        candidates = self._generate_candidates(subjects, seed)
        evaluated = [self._evaluate(candidate) for candidate in candidates]
        best = max(evaluated, key=lambda item: (item.score, -item.index, item.id))
        best_by_subject = self._best_candidates_by_subject(evaluated)
        operator_modes = sorted({candidate.metadata.get("source", "deterministic") for candidate in candidates})

        manifest = {
            "run_id": run_id,
            "config_path": str(self.config_path),
            "seed": seed,
            "subject_count": len(subjects),
            "candidate_count": len(evaluated),
            "best_candidate_id": best.id,
            "selected_candidate_ids": {
                subject_id: candidate.id
                for subject_id, candidate in best_by_subject.items()
            },
            "mode": "fixture",
            "operator_modes": operator_modes,
            "artifact_schema_version": 1,
        }
        overlays = self._build_overlays(best, best_by_subject, subjects)
        replay = {
            "run_id": run_id,
            "config": self.config,
            "subjects": [asdict(subject) for subject in subjects],
            "selected_candidate_id": best.id,
            "selected_candidate_ids": {
                subject_id: candidate.id
                for subject_id, candidate in best_by_subject.items()
            },
            "artifact_names": [
                "manifest.json",
                "candidate_history.jsonl",
                "best_candidate.json",
                "overlays.json",
                "replay.json",
            ],
        }
        best_payload = self._candidate_to_dict(best)
        best_payload["overlays"] = overlays

        artifact_paths = {
            "manifest": str(output_dir / "manifest.json"),
            "candidate_history": str(output_dir / "candidate_history.jsonl"),
            "best_candidate": str(output_dir / "best_candidate.json"),
            "overlays": str(output_dir / "overlays.json"),
            "replay": str(output_dir / "replay.json"),
        }
        _write_json(output_dir / "manifest.json", manifest)
        _write_jsonl(output_dir / "candidate_history.jsonl", [self._candidate_to_dict(c) for c in evaluated])
        _write_json(output_dir / "best_candidate.json", best_payload)
        _write_json(output_dir / "overlays.json", overlays)
        _write_json(output_dir / "replay.json", replay)

        return PromptOptimizationResult(
            run_id=run_id,
            output_dir=str(output_dir),
            best_candidate=best_payload,
            candidate_count=len(evaluated),
            artifact_paths=artifact_paths,
            schema_version=1,
            strategy="fixture",
            best_score=float(best.score),
        )

    def _run_evolutionary(self) -> PromptOptimizationResult:
        run_id = str(self.config.get("run_id") or _default_run_id())
        seed = int(self.config.get("seed", 0))
        rng = random.Random(seed)
        output_dir = self._resolve_output_dir(run_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        subjects = self._load_subjects()
        evaluation_mode = self._evolutionary_evaluation_mode()
        if evaluation_mode == "fixture_assertions":
            self._validate_evolutionary_fixture_evaluation(subjects)
            evaluator_adapter = None
        elif evaluation_mode == "evaluator":
            from .evaluators import EvaluatorScoringAdapter

            evaluator_adapter = EvaluatorScoringAdapter(
                evaluation_config=_mapping(self.config.get("evaluation"), "evaluation"),
                config_path=self.config_path,
                repo_root=self.repo_root,
                score_floor=float(self.config.get("score_floor", 0.0)),
            )
        else:
            raise PromptOptimizationError(
                "Schema v2 evolutionary evaluation.mode must be 'fixture_assertions' or 'evaluator'."
            )
        evolution = _mapping(self.config.get("evolution"), "evolution")
        population_size = int(self.config.get("population_size", evolution.get("population_size", 4)))
        max_generations = int(self.config.get("max_generations", evolution.get("max_generations", 3)))
        elite_count = int(self.config.get("elite_count", evolution.get("elite_count", 1)))
        mutation_rate = float(self.config.get("mutation_rate", evolution.get("mutation_rate", 0.7)))
        crossover_rate = float(self.config.get("crossover_rate", evolution.get("crossover_rate", 0.3)))
        if population_size < 1:
            raise PromptOptimizationError("population_size must be at least 1.")
        if max_generations < 1:
            raise PromptOptimizationError("max_generations must be at least 1.")
        elite_count = max(1, min(elite_count, population_size))

        stopping = _mapping(self.config.get("stopping"), "stopping")
        target_score = float(stopping.get("target_score", 1.0))
        max_stagnation = int(stopping.get("max_stagnation", max_generations))
        min_delta = float(stopping.get("min_delta", 0.0))
        score_floor = float(self.config.get("score_floor", 0.0))

        population = self._initial_population(subjects, population_size, rng)
        all_candidates: list[EvolutionCandidate] = []
        generation_rows: list[dict[str, Any]] = []
        lineage: dict[str, Any] = {"run_id": run_id, "candidates": {}}
        best: EvolutionCandidate | None = None
        stagnant_generations = 0
        stop_reason = "max_generations"

        for generation in range(max_generations):
            evaluated = [
                self._evaluate_genome(
                    candidate,
                    subjects,
                    score_floor=score_floor,
                    evaluator_adapter=evaluator_adapter,
                )
                for candidate in population
            ]
            ranked = sorted(evaluated, key=lambda item: (item.score, -item.index, item.id), reverse=True)
            elites = ranked[:elite_count]
            selected_ids = {candidate.id for candidate in elites}
            evaluated = [
                EvolutionCandidate(
                    id=candidate.id,
                    generation=candidate.generation,
                    index=candidate.index,
                    genome=candidate.genome,
                    parents=candidate.parents,
                    operator=candidate.operator,
                    metadata=candidate.metadata,
                    score=candidate.score,
                    metrics=candidate.metrics,
                    passed=candidate.passed,
                    diagnostics=candidate.diagnostics,
                    assertion_results=candidate.assertion_results,
                    selected=candidate.id in selected_ids,
                )
                for candidate in evaluated
            ]
            ranked = sorted(evaluated, key=lambda item: (item.score, -item.index, item.id), reverse=True)
            generation_best = ranked[0]
            previous_best_score = best.score if best is not None else None
            if best is None or (generation_best.score, -generation_best.index, generation_best.id) > (
                best.score,
                -best.index,
                best.id,
            ):
                best = generation_best

            improved = previous_best_score is None or generation_best.score > previous_best_score + min_delta
            stagnant_generations = 0 if improved else stagnant_generations + 1
            all_candidates.extend(evaluated)
            generation_rows.append(
                {
                    "generation": generation,
                    "candidate_count": len(evaluated),
                    "best_candidate_id": generation_best.id,
                    "best_score": generation_best.score,
                    "elite_candidate_ids": [candidate.id for candidate in ranked[:elite_count]],
                    "mean_score": sum(candidate.score for candidate in evaluated) / len(evaluated),
                }
            )
            for candidate in evaluated:
                lineage["candidates"][candidate.id] = {
                    "generation": candidate.generation,
                    "parents": list(candidate.parents),
                    "operator": candidate.operator,
                    "score": candidate.score,
                    "selected": candidate.selected,
                }

            if best.score >= target_score:
                stop_reason = "target_score"
                break
            if stagnant_generations >= max_stagnation:
                stop_reason = "stagnation"
                break
            if generation == max_generations - 1:
                stop_reason = "max_generations"
                break
            population = self._next_population(
                ranked=ranked,
                subjects=subjects,
                generation=generation + 1,
                population_size=population_size,
                elite_count=elite_count,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                rng=rng,
            )

        if best is None:
            raise PromptOptimizationError("Evolutionary prompt optimization produced no candidates.")

        best_by_subject = self._best_genome_by_subject(best, subjects)
        manifest = {
            "run_id": run_id,
            "config_path": str(self.config_path),
            "seed": seed,
            "subject_count": len(subjects),
            "candidate_count": len(all_candidates),
            "best_candidate_id": best.id,
            "selected_candidate_ids": {
                subject_id: best.id
                for subject_id in best.genome.values
            },
            "mode": evaluation_mode,
            "strategy": "evolutionary",
            "schema_version": 2,
            "artifact_schema_version": 2,
            "generation_count": len(generation_rows),
            "stop_reason": stop_reason,
            "best_score": best.score,
        }
        overlays = self._build_evolutionary_overlays(best, best_by_subject, subjects, len(generation_rows), stop_reason)
        best_payload = self._evolution_candidate_to_dict(best)
        best_payload["overlays"] = overlays
        replay = {
            "run_id": run_id,
            "config": self.config,
            "subjects": [asdict(subject) for subject in subjects],
            "selected_candidate_id": best.id,
            "selected_candidate_ids": manifest["selected_candidate_ids"],
            "generation_count": len(generation_rows),
            "stop_reason": stop_reason,
            "artifact_names": [
                "manifest.json",
                "candidate_history.jsonl",
                "best_candidate.json",
                "overlays.json",
                "replay.json",
                "generation_history.jsonl",
                "lineage.json",
            ],
        }
        artifact_paths = {
            "manifest": str(output_dir / "manifest.json"),
            "candidate_history": str(output_dir / "candidate_history.jsonl"),
            "best_candidate": str(output_dir / "best_candidate.json"),
            "overlays": str(output_dir / "overlays.json"),
            "replay": str(output_dir / "replay.json"),
            "generation_history": str(output_dir / "generation_history.jsonl"),
            "lineage": str(output_dir / "lineage.json"),
        }
        _write_json(output_dir / "manifest.json", manifest)
        _write_jsonl(output_dir / "candidate_history.jsonl", [self._evolution_candidate_to_dict(c) for c in all_candidates])
        _write_json(output_dir / "best_candidate.json", best_payload)
        _write_json(output_dir / "overlays.json", overlays)
        _write_json(output_dir / "replay.json", replay)
        _write_jsonl(output_dir / "generation_history.jsonl", generation_rows)
        _write_json(output_dir / "lineage.json", lineage)

        return PromptOptimizationResult(
            run_id=run_id,
            output_dir=str(output_dir),
            best_candidate=best_payload,
            candidate_count=len(all_candidates),
            artifact_paths=artifact_paths,
            schema_version=2,
            strategy="evolutionary",
            generation_count=len(generation_rows),
            stop_reason=stop_reason,
            best_score=float(best.score),
        )

    def _resolve_output_dir(self, run_id: str) -> Path:
        raw = self.config.get("output_dir", f".tracking/prompt_optimization/{run_id}")
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def _load_subjects(self) -> list[PromptSubject]:
        raw_subjects = self.config.get("subjects") or self.config.get("prompt_subjects")
        if not isinstance(raw_subjects, list) or not raw_subjects:
            raise PromptOptimizationError("Config requires a non-empty subjects list.")

        subjects: list[PromptSubject] = []
        for index, item in enumerate(raw_subjects):
            if not isinstance(item, dict):
                raise PromptOptimizationError(f"Subject #{index} must be a mapping.")
            subject_id = str(item.get("id") or f"subject_{index}")
            source_path = str(item.get("path") or item.get("file") or "")
            dotted_path = str(item.get("dotted_path") or item.get("path_in_file") or "")
            if not source_path or not dotted_path:
                raise PromptOptimizationError(f"Subject {subject_id!r} requires path and dotted_path.")
            resolved_source_path = self._resolve_input_path(source_path)
            loaded = _load_yaml(resolved_source_path)
            original = _resolve_dotted_path(loaded, dotted_path)
            subjects.append(
                PromptSubject(
                    id=subject_id,
                    source_path=source_path,
                    source_path_absolute=str(resolved_source_path),
                    source_path_repo_relative=_repo_relative_path(resolved_source_path, self.repo_root),
                    dotted_path=dotted_path,
                    original_value=original,
                    prompt_text=_prompt_text(original, item.get("join_with", "\n")),
                )
            )
        return subjects

    def _resolve_input_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([self.repo_root / path, self.config_path.parent / path])
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise PromptOptimizationError(f"Referenced YAML path not found: {raw_path}")

    def _generate_candidates(self, subjects: list[PromptSubject], seed: int) -> list[PromptCandidate]:
        operator_specs = self.config.get("operators")
        if not isinstance(operator_specs, list) or not operator_specs:
            raise PromptOptimizationError("Config requires a non-empty operators list.")
        per_subject = int(self.config.get("candidates_per_subject", self.config.get("candidate_count", 4)))
        include_baseline = bool(self.config.get("include_baseline", True))
        rng = random.Random(seed)

        candidates: list[PromptCandidate] = []
        for subject in subjects:
            if include_baseline:
                candidates.append(
                    PromptCandidate(
                        id=_candidate_id(subject.id, 0, "baseline", subject.prompt_text),
                        subject_id=subject.id,
                        index=len(candidates),
                        operator="baseline",
                        prompt_text=subject.prompt_text,
                        metadata={"source": "baseline"},
                    )
                )
            for local_index in range(per_subject):
                spec = rng.choice(operator_specs)
                candidate_text, metadata = self._apply_candidate_operator(subject, spec, rng)
                operator_name = str(spec.get("type") or spec.get("name") or "unknown")
                candidates.append(
                    PromptCandidate(
                        id=_candidate_id(subject.id, local_index + 1, operator_name, candidate_text),
                        subject_id=subject.id,
                        index=len(candidates),
                        operator=operator_name,
                        prompt_text=candidate_text,
                        metadata=metadata,
                    )
                )
        return candidates

    def _apply_candidate_operator(
        self,
        subject: PromptSubject,
        spec: dict[str, Any],
        rng: random.Random,
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(spec, dict):
            raise PromptOptimizationError("Each operator must be a mapping.")
        op_type = str(spec.get("type") or spec.get("name") or "")
        if op_type == "llm_rewrite":
            return self._apply_llm_rewrite(subject, spec)
        return _apply_operator(subject.prompt_text, spec, rng)

    def _apply_llm_rewrite(
        self,
        subject: PromptSubject,
        spec: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        llm_config = _deep_merge(_mapping(self.config.get("llm"), "llm"), _mapping(spec.get("llm"), "operator.llm"))
        provider = _required_text(llm_config, "provider", "llm")
        model = _required_text(llm_config, "model", "llm")
        prompt_template = _required_text(llm_config, "prompt_template", "llm")
        system_prompt = _required_text(llm_config, "system_prompt", "llm")
        temperature = float(llm_config.get("temperature", 0.2))
        max_tokens = int(llm_config.get("max_tokens", 1024))
        env_prefix = str(llm_config.get("env_prefix") or "PROMPT_OPT")

        prompt = prompt_template.format(
            prompt=subject.prompt_text,
            subject_id=subject.id,
            dotted_path=subject.dotted_path,
            source_path=subject.source_path,
        )
        client = self._get_llm_client(llm_config, provider, model, env_prefix)
        response = client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        rewritten = _clean_llm_prompt_response(response)
        if not rewritten:
            raise PromptOptimizationError("llm_rewrite returned an empty prompt.")
        return rewritten, {
            "operator": "llm_rewrite",
            "source": "llm",
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _get_llm_client(
        self,
        llm_config: dict[str, Any],
        provider: str,
        model: str,
        env_prefix: str,
    ) -> Any:
        client_config = dict(llm_config)
        client_config.pop("prompt_template", None)
        client_config.pop("system_prompt", None)
        key = json.dumps(
            {
                "provider": provider,
                "model": model,
                "env_prefix": env_prefix,
                "config": client_config,
            },
            sort_keys=True,
        )
        if key not in self._llm_clients:
            from shared.llm import create_client

            self._llm_clients[key] = create_client(
                provider=provider,
                model=model,
                env_prefix=env_prefix,
                config_defaults=client_config,
            )
        return self._llm_clients[key]

    def _evaluate(self, candidate: PromptCandidate) -> PromptCandidate:
        evaluation = self.config.get("evaluation", {})
        assertions = evaluation.get("assertions", []) if isinstance(evaluation, dict) else []
        if not isinstance(assertions, list):
            raise PromptOptimizationError("evaluation.assertions must be a list when provided.")
        total = 0.0
        results: list[dict[str, Any]] = []
        for assertion in assertions:
            result = _evaluate_assertion(candidate.prompt_text, assertion)
            total += result["score"]
            results.append(result)
        return PromptCandidate(
            id=candidate.id,
            subject_id=candidate.subject_id,
            index=candidate.index,
            operator=candidate.operator,
            prompt_text=candidate.prompt_text,
            metadata=candidate.metadata,
            score=total,
            assertion_results=tuple(results),
        )

    def _initial_population(
        self,
        subjects: list[PromptSubject],
        population_size: int,
        rng: random.Random,
    ) -> list[EvolutionCandidate]:
        baseline = PromptGenome({subject.id: subject.prompt_text for subject in subjects})
        population = [
            EvolutionCandidate(
                id=_genome_candidate_id(0, 0, "baseline", baseline.values),
                generation=0,
                index=0,
                genome=baseline,
                parents=(),
                operator="baseline",
                metadata={"source": "baseline"},
            )
        ]
        next_index = 1
        while len(population) < population_size:
            genome, metadata = self._mutate_genome(baseline, subjects, rng)
            population.append(
                EvolutionCandidate(
                    id=_genome_candidate_id(0, next_index, "mutation", genome.values),
                    generation=0,
                    index=next_index,
                    genome=genome,
                    parents=(population[0].id,),
                    operator="mutation",
                    metadata=metadata,
                )
            )
            next_index += 1
        return population

    def _next_population(
        self,
        *,
        ranked: list[EvolutionCandidate],
        subjects: list[PromptSubject],
        generation: int,
        population_size: int,
        elite_count: int,
        mutation_rate: float,
        crossover_rate: float,
        rng: random.Random,
    ) -> list[EvolutionCandidate]:
        next_population: list[EvolutionCandidate] = []
        for elite in ranked[:elite_count]:
            next_population.append(
                EvolutionCandidate(
                    id=_genome_candidate_id(generation, len(next_population), "elite", elite.genome.values),
                    generation=generation,
                    index=len(next_population),
                    genome=elite.genome,
                    parents=(elite.id,),
                    operator="elite",
                    metadata={"source": "elite"},
                )
            )

        selection = _mapping(self.config.get("selection"), "selection")
        parent_pool = ranked[: max(elite_count, min(len(ranked), int(selection.get("pool_size", len(ranked)))))]
        while len(next_population) < population_size:
            roll = rng.random()
            if len(parent_pool) >= 2 and roll < crossover_rate:
                first, second = rng.sample(parent_pool, 2)
                genome, metadata = self._crossover_genomes(first.genome, second.genome, rng)
                parents = (first.id, second.id)
                operator = "crossover"
            else:
                parent = rng.choice(parent_pool)
                if roll < crossover_rate + mutation_rate:
                    genome, metadata = self._mutate_genome(parent.genome, subjects, rng)
                    operator = "mutation"
                else:
                    genome = parent.genome
                    metadata = {"source": "copy"}
                    operator = "copy"
                parents = (parent.id,)
            next_population.append(
                EvolutionCandidate(
                    id=_genome_candidate_id(generation, len(next_population), operator, genome.values),
                    generation=generation,
                    index=len(next_population),
                    genome=genome,
                    parents=parents,
                    operator=operator,
                    metadata=metadata,
                )
            )
        return next_population

    def _mutate_genome(
        self,
        genome: PromptGenome,
        subjects: list[PromptSubject],
        rng: random.Random,
    ) -> tuple[PromptGenome, dict[str, Any]]:
        subject = rng.choice(subjects)
        operator_specs = self.config.get("operators")
        if not isinstance(operator_specs, list) or not operator_specs:
            raise PromptOptimizationError("Config requires a non-empty operators list.")
        spec = rng.choice(operator_specs)
        subject_for_candidate = PromptSubject(
            id=subject.id,
            source_path=subject.source_path,
            source_path_absolute=subject.source_path_absolute,
            source_path_repo_relative=subject.source_path_repo_relative,
            dotted_path=subject.dotted_path,
            original_value=subject.original_value,
            prompt_text=genome.values[subject.id],
        )
        candidate_text, metadata = self._apply_candidate_operator(subject_for_candidate, spec, rng)
        values = dict(genome.values)
        values[subject.id] = candidate_text
        metadata = dict(metadata)
        metadata["mutated_subject_id"] = subject.id
        return PromptGenome(values), metadata

    @staticmethod
    def _crossover_genomes(
        first: PromptGenome,
        second: PromptGenome,
        rng: random.Random,
    ) -> tuple[PromptGenome, dict[str, Any]]:
        values: dict[str, str] = {}
        second_subjects: list[str] = []
        for subject_id in sorted(first.values):
            use_second = rng.random() < 0.5
            values[subject_id] = second.values[subject_id] if use_second else first.values[subject_id]
            if use_second:
                second_subjects.append(subject_id)
        if len(first.values) > 1 and not second_subjects:
            subject_id = rng.choice(sorted(first.values))
            values[subject_id] = second.values[subject_id]
            second_subjects.append(subject_id)
        return PromptGenome(values), {"source": "crossover", "second_parent_subject_ids": second_subjects}

    def _evaluate_genome(
        self,
        candidate: EvolutionCandidate,
        subjects: list[PromptSubject],
        *,
        score_floor: float,
        evaluator_adapter: Any | None = None,
    ) -> EvolutionCandidate:
        diagnostics: list[dict[str, Any]] = []
        for subject in subjects:
            diagnostics.extend(_placeholder_diagnostics(candidate.genome.values[subject.id], self._required_placeholders(subject)))
        if diagnostics:
            return EvolutionCandidate(
                id=candidate.id,
                generation=candidate.generation,
                index=candidate.index,
                genome=candidate.genome,
                parents=candidate.parents,
                operator=candidate.operator,
                metadata=candidate.metadata,
                score=score_floor,
                metrics={"normalized_score": score_floor, "raw_score": 0.0, "max_score": 1.0},
                passed=False,
                diagnostics=tuple(diagnostics),
                assertion_results=(),
                selected=candidate.selected,
            )

        if evaluator_adapter is not None:
            evaluation_score = evaluator_adapter.score(candidate.genome.values, subjects)
            return EvolutionCandidate(
                id=candidate.id,
                generation=candidate.generation,
                index=candidate.index,
                genome=candidate.genome,
                parents=candidate.parents,
                operator=candidate.operator,
                metadata=candidate.metadata,
                score=evaluation_score.score,
                metrics=evaluation_score.metrics,
                passed=evaluation_score.passed,
                diagnostics=evaluation_score.diagnostics,
                assertion_results=evaluation_score.assertion_results,
                selected=candidate.selected,
            )

        evaluation = self.config.get("evaluation", {})
        assertions = evaluation.get("assertions", []) if isinstance(evaluation, dict) else []
        if not isinstance(assertions, list):
            raise PromptOptimizationError("evaluation.assertions must be a list when provided.")
        total = 0.0
        max_score = 0.0
        results: list[dict[str, Any]] = []
        joined_prompt = "\n\n".join(candidate.genome.values[subject.id] for subject in subjects)
        for assertion in assertions:
            if not isinstance(assertion, dict):
                raise PromptOptimizationError("Each assertion must be a mapping.")
            subject_id = assertion.get("subject_id")
            prompt_text = candidate.genome.values.get(str(subject_id), "") if subject_id else joined_prompt
            result = _evaluate_assertion(prompt_text, assertion)
            total += result["score"]
            max_score += float(result["weight"])
            results.append(result)
        normalized = 1.0 if max_score <= 0 else max(0.0, min(1.0, total / max_score))
        return EvolutionCandidate(
            id=candidate.id,
            generation=candidate.generation,
            index=candidate.index,
            genome=candidate.genome,
            parents=candidate.parents,
            operator=candidate.operator,
            metadata=candidate.metadata,
            score=normalized,
            metrics={"normalized_score": normalized, "raw_score": total, "max_score": max_score},
            passed=normalized >= 1.0,
            diagnostics=tuple(diagnostics),
            assertion_results=tuple(results),
            selected=candidate.selected,
        )

    def _validate_evolutionary_fixture_evaluation(self, subjects: list[PromptSubject]) -> None:
        evaluation = self.config.get("evaluation", {})
        assertions = evaluation.get("assertions", []) if isinstance(evaluation, dict) else []
        if not isinstance(assertions, list):
            raise PromptOptimizationError("evaluation.assertions must be a list when provided.")
        if not assertions:
            raise PromptOptimizationError(
                "Schema v2 evolutionary fixture evaluation requires at least one positive-weight assertion."
            )

        subject_ids = {subject.id for subject in subjects}
        for index, assertion in enumerate(assertions):
            if not isinstance(assertion, dict):
                raise PromptOptimizationError("Each assertion must be a mapping.")
            weight = float(assertion.get("weight", 1.0))
            if weight <= 0.0:
                raise PromptOptimizationError(
                    f"evaluation.assertions[{index}].weight must be strictly positive for schema v2 fixture evaluation."
                )
            subject_id = assertion.get("subject_id")
            if subject_id is not None and str(subject_id) not in subject_ids:
                raise PromptOptimizationError(
                    f"evaluation.assertions[{index}].subject_id references unknown subject: {subject_id}"
                )

    def _evolutionary_evaluation_mode(self) -> str:
        evaluation = self.config.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise PromptOptimizationError("evaluation must be a mapping when provided.")
        mode = str(evaluation.get("mode") or evaluation.get("type") or "").strip().lower()
        if not mode:
            return "fixture_assertions"
        if mode == "fixture":
            return "fixture_assertions"
        return mode

    def _required_placeholders(self, subject: PromptSubject) -> list[str]:
        raw_subjects = self.config.get("subjects") or self.config.get("prompt_subjects") or []
        placeholders: list[str] = []
        for item in raw_subjects:
            if isinstance(item, dict) and str(item.get("id")) == subject.id:
                value = item.get("required_placeholders") or item.get("placeholders") or []
                if isinstance(value, list):
                    placeholders.extend(str(part) for part in value)
        constraints = _mapping(self.config.get("constraints"), "constraints")
        value = constraints.get("required_placeholders", [])
        if isinstance(value, dict):
            placeholders.extend(str(part) for part in value.get(subject.id, []))
        return sorted(set(placeholders))

    @staticmethod
    def _best_genome_by_subject(
        best: EvolutionCandidate,
        subjects: Iterable[PromptSubject],
    ) -> dict[str, PromptCandidate]:
        return {
            subject.id: PromptCandidate(
                id=best.id,
                subject_id=subject.id,
                index=best.index,
                operator=best.operator,
                prompt_text=best.genome.values[subject.id],
                metadata={"genome_candidate_id": best.id},
                score=best.score,
            )
            for subject in subjects
        }

    def _build_evolutionary_overlays(
        self,
        best: EvolutionCandidate,
        best_by_subject: dict[str, PromptCandidate],
        subjects: Iterable[PromptSubject],
        generation_count: int,
        stop_reason: str,
    ) -> dict[str, Any]:
        overlays = self._build_overlays(
            PromptCandidate(
                id=best.id,
                subject_id="genome",
                index=best.index,
                operator=best.operator,
                prompt_text="",
                metadata={},
                score=best.score,
            ),
            best_by_subject,
            subjects,
        )
        overlays["version"] = 2
        overlays["strategy"] = "evolutionary"
        overlays["generation_count"] = generation_count
        overlays["stop_reason"] = stop_reason
        overlays["best_score"] = best.score
        for subject_id, subject_overlay in overlays["subjects"].items():
            subject_overlay["selected_genome_candidate_id"] = best.id
        return overlays

    @staticmethod
    def _evolution_candidate_to_dict(candidate: EvolutionCandidate) -> dict[str, Any]:
        return {
            "id": candidate.id,
            "candidate_id": candidate.id,
            "generation": candidate.generation,
            "index": candidate.index,
            "genome": candidate.genome.values,
            "parents": list(candidate.parents),
            "operator": candidate.operator,
            "metadata": candidate.metadata,
            "score": candidate.score,
            "metrics": candidate.metrics or {},
            "passed": candidate.passed,
            "diagnostics": list(candidate.diagnostics),
            "assertion_results": list(candidate.assertion_results),
            "selected": candidate.selected,
        }

    @staticmethod
    def _best_candidates_by_subject(
        evaluated: Iterable[PromptCandidate],
    ) -> dict[str, PromptCandidate]:
        best_by_subject: dict[str, PromptCandidate] = {}
        for candidate in evaluated:
            current = best_by_subject.get(candidate.subject_id)
            if current is None or (candidate.score, -candidate.index, candidate.id) > (
                current.score,
                -current.index,
                current.id,
            ):
                best_by_subject[candidate.subject_id] = candidate
        return best_by_subject

    def _build_overlays(
        self,
        best: PromptCandidate,
        best_by_subject: dict[str, PromptCandidate],
        subjects: Iterable[PromptSubject],
    ) -> dict[str, Any]:
        subject_by_id = {subject.id: subject for subject in subjects}
        return {
            "version": 1,
            "selected_candidate_id": best.id,
            "selected_candidate_ids": {
                subject_id: candidate.id
                for subject_id, candidate in best_by_subject.items()
            },
            "subjects": {
                subject_id: {
                    "source_path": subject.source_path,
                    "source_path_absolute": subject.source_path_absolute,
                    "source_path_repo_relative": subject.source_path_repo_relative,
                    "dotted_path": subject.dotted_path,
                    "selected_candidate_id": candidate.id,
                    "optimized_prompt": candidate.prompt_text,
                    "score": candidate.score,
                }
                for subject_id, candidate in best_by_subject.items()
                for subject in [subject_by_id[subject_id]]
            },
        }

    @staticmethod
    def _candidate_to_dict(candidate: PromptCandidate) -> dict[str, Any]:
        payload = asdict(candidate)
        payload["assertion_results"] = list(candidate.assertion_results)
        return payload


def _apply_operator(
    prompt_text: str,
    spec: dict[str, Any],
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(spec, dict):
        raise PromptOptimizationError("Each operator must be a mapping.")
    op_type = str(spec.get("type") or spec.get("name") or "")
    values = spec.get("values", [])
    value = rng.choice(values) if isinstance(values, list) and values else spec.get("value", "")

    if op_type == "append":
        text = prompt_text.rstrip() + "\n" + str(value).strip()
    elif op_type == "prepend":
        text = str(value).strip() + "\n" + prompt_text.lstrip()
    elif op_type == "replace":
        replacements = spec.get("replacements", [])
        if not isinstance(replacements, list) or not replacements:
            raise PromptOptimizationError("replace operator requires replacements list.")
        replacement = rng.choice(replacements)
        old = str(replacement.get("old", ""))
        new = str(replacement.get("new", ""))
        text = prompt_text.replace(old, new)
        value = {"old": old, "new": new}
    elif op_type == "remove_line_contains":
        needle = str(value)
        lines = [line for line in prompt_text.splitlines() if needle not in line]
        text = "\n".join(lines)
    else:
        raise PromptOptimizationError(f"Unsupported prompt operator: {op_type}")

    return text, {"operator": op_type, "value": value}


def _evaluate_assertion(prompt_text: str, assertion: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(assertion, dict):
        raise PromptOptimizationError("Each assertion must be a mapping.")
    assertion_type = str(assertion.get("type") or "")
    weight = float(assertion.get("weight", 1.0))
    label = str(assertion.get("id") or assertion_type)
    passed = False

    if assertion_type == "contains":
        passed = str(assertion.get("text", "")) in prompt_text
    elif assertion_type == "not_contains":
        passed = str(assertion.get("text", "")) not in prompt_text
    elif assertion_type == "regex":
        passed = re.search(str(assertion.get("pattern", "")), prompt_text, flags=re.MULTILINE) is not None
    elif assertion_type == "max_length":
        passed = len(prompt_text) <= int(assertion.get("value", 0))
    elif assertion_type == "min_length":
        passed = len(prompt_text) >= int(assertion.get("value", 0))
    else:
        raise PromptOptimizationError(f"Unsupported fixture assertion: {assertion_type}")

    return {
        "id": label,
        "type": assertion_type,
        "passed": passed,
        "weight": weight,
        "score": weight if passed else 0.0,
    }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PromptOptimizationError(f"{label} must be a mapping when provided.")
    return value


def _required_text(config: dict[str, Any], key: str, label: str) -> str:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise PromptOptimizationError(f"{label}.{key} is required for llm_rewrite.")
    return str(value)


def _clean_llm_prompt_response(response: Any) -> str:
    text = str(response or "").strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```text"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "tuner.py").exists():
            return candidate
    return Path.cwd().resolve()


def _repo_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return ""


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_dotted_path(data: Any, dotted_path: str) -> Any:
    current = data
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise PromptOptimizationError(f"Dotted path segment not found: {part}")
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise PromptOptimizationError(f"Invalid list segment in dotted path: {part}") from exc
        else:
            raise PromptOptimizationError(f"Cannot resolve {part!r} inside scalar value.")
    return current


def _prompt_text(value: Any, join_with: Any) -> str:
    if isinstance(value, list):
        return str(join_with).join(str(item) for item in value)
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def _candidate_id(subject_id: str, index: int, operator: str, prompt_text: str) -> str:
    digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]
    return f"{subject_id}-{index:03d}-{operator}-{digest}"


def _genome_candidate_id(generation: int, index: int, operator: str, genome: dict[str, str]) -> str:
    payload = json.dumps(genome, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"g{generation:03d}-{index:03d}-{operator}-{digest}"


def _placeholder_diagnostics(prompt_text: str, placeholders: Iterable[str]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for placeholder in placeholders:
        if placeholder not in prompt_text:
            diagnostics.append(
                {
                    "code": "MISSING_REQUIRED_PLACEHOLDER",
                    "message": f"Candidate prompt is missing required placeholder {placeholder!r}.",
                    "placeholder": placeholder,
                }
            )
    return diagnostics


def _default_run_id() -> str:
    return "prompt-opt-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
