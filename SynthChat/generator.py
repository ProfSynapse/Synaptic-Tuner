"""SynthChat Generator - Stage-by-stage dataset generation.

Location: SynthChat/generator.py
Purpose: Creates synthetic training examples from scenario configs with integrated quality improvement
Usage: Called by run.py for "generate" mode

Architecture:
    Generates examples stage-by-stage (system → user → assistant)
    Each stage is validated/improved before proceeding to next
    Uses scenarios/*.yaml for generation templates
    Integrates with engine.py for quality control
"""

import json
import re
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Mapping, Sequence
from dataclasses import dataclass
from copy import deepcopy

from .utils.yaml_loader import load_yaml
from .utils.docs_loader import DocFile
from .engine import ImprovementEngine
from .stage_gates import run_stage_gates
from .task_derivation import derive_task_spec
from shared.llm.factory import create_client
from shared.stage_judges import ConfigurableStageJudge

try:
    from shared.environments import EnvironmentValidator
except ImportError:
    EnvironmentValidator = None

try:
    from shared.agentic_judge import AgenticTurnJudge
    from shared.agentic_loop import AgenticModelResponse, run_environment_episode
    from Evaluator.schema_validator import ValidationResult, ValidatorIssue, validate_assistant_response
except ImportError:
    AgenticTurnJudge = None
    AgenticModelResponse = None
    run_environment_episode = None
    ValidationResult = None
    ValidatorIssue = None
    validate_assistant_response = None


@dataclass
class GenerationResult:
    """Result from generating a single example."""
    example: Dict
    scenario_key: str
    iterations: int
    success: bool
    stage_failures: List[str]  # Stages that failed to pass


class ScenarioLoader:
    """Loads and manages scenario configurations."""

    def __init__(self, scenarios_dir: Path):
        """
        Initialize scenario loader.

        Args:
            scenarios_dir: Directory containing scenario YAML files
        """
        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios: Dict[str, Dict] = {}
        self._load_all_scenarios()

    def _load_all_scenarios(self):
        """Load all scenario files from scenarios directory."""
        for yaml_file in self.scenarios_dir.glob("*.yaml"):
            data = load_yaml(yaml_file)
            if "scenarios" in data:
                self.scenarios.update(data["scenarios"])

    def get_scenario(self, key: str) -> Optional[Dict]:
        """Get scenario by key."""
        return self.scenarios.get(key)

    def list_scenarios(self) -> List[str]:
        """List all available scenario keys."""
        return list(self.scenarios.keys())

    def get_scenarios_by_type(self, scenario_type: str) -> Dict[str, Dict]:
        """Get all scenarios of a specific type (tool, behavioral, destructive)."""
        return {
            key: scenario
            for key, scenario in self.scenarios.items()
            if scenario.get("type") == scenario_type
        }


class SynthChatGenerator:
    """
    Main generator - creates synthetic examples from scenarios.

    Responsibility: Orchestrate stage-by-stage generation with quality control.
    Each stage (system → user → assistant) is generated then improved before proceeding.
    """

    def __init__(
        self,
        config_dir: Path,
        scenarios_dir: Path,
        rubrics_dir: Path,
        llm_client,  # Generation LLM from shared.llm
        engine: Optional[ImprovementEngine] = None,
        environment_validator: Optional["EnvironmentValidator"] = None,
        enable_stage_validation: bool = True,
        logger=None
    ):
        """
        Initialize generator.

        Args:
            config_dir: Path to config directory (settings.yaml, validation.yaml)
            scenarios_dir: Path to scenarios directory
            rubrics_dir: Path to rubrics directory
            llm_client: LLM client for generation (from shared.llm.create_client)
            engine: Improvement engine (optional, will create if None)
            environment_validator: Optional environment validator for runtime-backed
                tool execution checks.
            enable_stage_validation: Whether to validate each stage (default: True)
            logger: Logger instance (optional)
        """
        self.config_dir = Path(config_dir)
        self.llm_client = llm_client
        self.logger = logger
        self.enable_stage_validation = enable_stage_validation
        self.environment_validator = environment_validator
        self._llm_client_cache: Dict[Tuple[Any, ...], Any] = {}

        # Load scenario configurations
        self.scenario_loader = ScenarioLoader(scenarios_dir)

        # Initialize or use provided improvement engine
        if engine is None and enable_stage_validation:
            # Create engine for stage validation
            # Use validation.yaml (formerly scope_config.yaml)
            validation_config = self.config_dir / "validation.yaml"
            self.engine = ImprovementEngine(
                llm_client=llm_client,
                rubrics_dir=rubrics_dir,
                config_path=validation_config,
                logger=logger,
                enable_interactions=True
            )
        else:
            self.engine = engine

    def generate_batch(
        self,
        targets: Dict[str, Any],  # {scenario_key: count|target_spec}
        max_iterations: int = 3,
        randomize_params: bool = True,
        doc_context: Optional[DocFile] = None,
        on_result=None,
        shared_seed_spec: Optional[Dict[str, Any]] = None,
    ) -> List[GenerationResult]:
        """
        Generate a batch of examples from scenario targets.

        Args:
            targets: Dictionary mapping scenario keys to counts
            max_iterations: Max improvement iterations per stage
            randomize_params: Whether to randomize LLM parameters
            doc_context: Optional document context for template variables
                        Makes {doc_content} and {doc_path} available in prompts
            on_result: Optional callback invoked for each completed GenerationResult

        Returns:
            List of GenerationResult objects
        """
        results = []
        total = 0
        normalized_targets: List[Tuple[str, Dict[str, Any], Dict[str, int]]] = []

        shared_seed_targets = set((shared_seed_spec or {}).get("targets") or [])
        using_shared_seed = bool(shared_seed_spec)
        shared_seed_count = int((shared_seed_spec or {}).get("seed_count", 0) or 0)

        for scenario_key, raw_target in targets.items():
            scenario = self.scenario_loader.get_scenario(scenario_key)
            if not scenario:
                if self.logger:
                    self.logger.warning(f"Scenario not found: {scenario_key}")
                continue
            target_spec = _normalize_target_spec(raw_target)
            normalized_targets.append((scenario_key, scenario, target_spec))
            if using_shared_seed and (not shared_seed_targets or scenario_key in shared_seed_targets):
                total += shared_seed_count * target_spec["rollouts_per_seed"]
            else:
                total += target_spec["seed_count"] * target_spec["rollouts_per_seed"]

        current = 0

        if using_shared_seed:
            shared_scenario_key = str(shared_seed_spec.get("scenario") or "").strip()
            shared_scenario = self.scenario_loader.get_scenario(shared_scenario_key)
            if not shared_scenario:
                raise ValueError(f"Shared seed scenario not found: {shared_scenario_key}")

            applicable_targets = [
                item
                for item in normalized_targets
                if not shared_seed_targets or item[0] in shared_seed_targets
            ]
            local_targets = [
                item
                for item in normalized_targets
                if shared_seed_targets and item[0] not in shared_seed_targets
            ]

            for shared_seed_index in range(shared_seed_count):
                shared_seed_id = f"{shared_scenario_key}:shared_seed:{shared_seed_index + 1}"
                shared_seed_bundle = self.prepare_seed_bundle(
                    scenario_key=shared_scenario_key,
                    seed_id=shared_seed_id,
                    scenario=shared_scenario,
                    randomize_params=randomize_params,
                    doc_context=doc_context,
                )

                for scenario_key, scenario, target_spec in applicable_targets:
                    if target_spec["seed_count"] not in {0, 1} and self.logger:
                        self.logger.warning(
                            f"Scenario '{scenario_key}' uses shared seed mode; ignoring seed_count={target_spec['seed_count']} "
                            "and reusing the pack-global shared seed instead."
                        )

                    for rollout_index in range(target_spec["rollouts_per_seed"]):
                        current += 1
                        if self.logger:
                            self.logger.info(
                                f"Generating {current}/{total}: {scenario_key} "
                                f"(shared seed {shared_seed_index + 1}/{shared_seed_count}, "
                                f"rollout {rollout_index + 1}/{target_spec['rollouts_per_seed']})"
                            )

                        result = self.generate_single(
                            scenario_key,
                            scenario,
                            max_iterations,
                            randomize_params,
                            doc_context,
                            seed_bundle=shared_seed_bundle,
                            rollout_metadata={
                                "seed_id": shared_seed_id,
                                "seed_index": shared_seed_index,
                                "seed_count": shared_seed_count,
                                "rollout_index": rollout_index,
                                "rollouts_per_seed": target_spec["rollouts_per_seed"],
                                "shared_seed_source": shared_scenario_key,
                                "shared_across_scenarios": True,
                            },
                        )
                        results.append(result)
                        if on_result is not None:
                            on_result(result)

            for scenario_key, scenario, target_spec in local_targets:
                seed_count = target_spec["seed_count"]
                rollouts_per_seed = target_spec["rollouts_per_seed"]

                for seed_index in range(seed_count):
                    seed_id = f"{scenario_key}:seed:{seed_index + 1}"
                    seed_bundle = self.prepare_seed_bundle(
                        scenario_key=scenario_key,
                        seed_id=seed_id,
                        scenario=scenario,
                        randomize_params=randomize_params,
                        doc_context=doc_context,
                    )

                    for rollout_index in range(rollouts_per_seed):
                        current += 1
                        if self.logger:
                            self.logger.info(
                                f"Generating {current}/{total}: {scenario_key} "
                                f"(seed {seed_index + 1}/{seed_count}, rollout {rollout_index + 1}/{rollouts_per_seed})"
                            )

                        result = self.generate_single(
                            scenario_key,
                            scenario,
                            max_iterations,
                            randomize_params,
                            doc_context,
                            seed_bundle=seed_bundle,
                            rollout_metadata={
                                "seed_id": seed_id,
                                "seed_index": seed_index,
                                "seed_count": seed_count,
                                "rollout_index": rollout_index,
                                "rollouts_per_seed": rollouts_per_seed,
                            },
                        )
                        results.append(result)
                        if on_result is not None:
                            on_result(result)

            return results

        for scenario_key, scenario, target_spec in normalized_targets:
            seed_count = target_spec["seed_count"]
            rollouts_per_seed = target_spec["rollouts_per_seed"]

            for seed_index in range(seed_count):
                seed_id = f"{scenario_key}:seed:{seed_index + 1}"
                seed_bundle = self.prepare_seed_bundle(
                    scenario_key=scenario_key,
                    seed_id=seed_id,
                    scenario=scenario,
                    randomize_params=randomize_params,
                    doc_context=doc_context,
                )

                for rollout_index in range(rollouts_per_seed):
                    current += 1
                    if self.logger:
                        self.logger.info(
                            f"Generating {current}/{total}: {scenario_key} "
                            f"(seed {seed_index + 1}/{seed_count}, rollout {rollout_index + 1}/{rollouts_per_seed})"
                        )

                    result = self.generate_single(
                        scenario_key,
                        scenario,
                        max_iterations,
                        randomize_params,
                        doc_context,
                        seed_bundle=seed_bundle,
                        rollout_metadata={
                            "seed_id": seed_id,
                            "seed_index": seed_index,
                            "seed_count": seed_count,
                            "rollout_index": rollout_index,
                            "rollouts_per_seed": rollouts_per_seed,
                        },
                    )
                    results.append(result)
                    if on_result is not None:
                        on_result(result)

        return results

    def prepare_seed_bundle(
        self,
        *,
        scenario_key: Optional[str] = None,
        seed_id: Optional[str] = None,
        scenario: Dict[str, Any],
        randomize_params: bool,
        doc_context: Optional[DocFile] = None,
    ) -> Dict[str, Any]:
        """Prepare a reusable environment/system-context seed for one or more rollouts."""
        prompts = scenario.get("prompts", {})
        template_vars = {}
        if doc_context:
            template_vars["doc_content"] = doc_context.content
            template_vars["doc_path"] = doc_context.path

        def render_prompt(prompt: str) -> str:
            result = prompt
            for key, value in template_vars.items():
                result = result.replace(f"{{{key}}}", value)
            return result

        environment_mode = self._resolve_environment_mode(scenario)
        generated_environment = {}
        stage_reviews: Dict[str, Any] = {}
        seed_stage_failures: List[str] = []
        if environment_mode in {"generated", "hybrid"}:
            generation_cfg = scenario.get("environment_generation") or {}
            trace_label = ":".join(
                part for part in [scenario_key, "environment_generation", seed_id] if part
            ) or "environment_generation"
            if scenario_key:
                self._log_stage(
                    scenario_key,
                    "environment_generation",
                    "start",
                    extra=f"seed_id={seed_id}" if seed_id else "seed_prepare",
                )
            generated_environment = self._generate_environment_spec(
                scenario=scenario,
                render_prompt=render_prompt,
                randomize_params=randomize_params,
                trace_label=trace_label,
            )
            review = self._run_stage_review(
                stage_name="environment_generation",
                stage_config=generation_cfg,
                scenario_key=scenario_key or seed_id or "environment_generation",
                scenario=scenario,
                task_context={},
                payload=self._build_environment_generation_review_payload(
                    generated_environment=generated_environment,
                    seed_id=seed_id,
                ),
            )
            if review is not None:
                stage_reviews["environment_generation"] = review
                if review.get("passed") is False and review.get("enforce", True):
                    seed_stage_failures.append("environment_generation")
            if scenario_key:
                self._log_stage(
                    scenario_key,
                    "environment_generation",
                    "done",
                    extra=(
                        f"seed_id={seed_id} keys={sorted(generated_environment.keys())}"
                        if generated_environment
                        else f"seed_id={seed_id} empty"
                    ) if seed_id else (
                        f"keys={sorted(generated_environment.keys())}" if generated_environment else "empty"
                    ),
                )
        if generated_environment:
            template_vars["environment_json"] = json.dumps(generated_environment, indent=2)

        base_environment_config = scenario.get("environment")
        generated_environment_config = (
            generated_environment.get("environment") if environment_mode in {"generated", "hybrid"} else None
        )
        resolved_environment_config = _deep_merge_dicts(
            base_environment_config,
            generated_environment_config,
        )
        resolved_system_context = _deep_merge_dicts(
            scenario.get("system_context"),
            generated_environment.get("system_context"),
        )
        resolved_task_context = _make_json_safe(
            _deep_merge_dicts(
                scenario.get("task_context"),
                generated_environment.get("task_context"),
            )
        )
        return {
            "environment_mode": environment_mode,
            "generated_environment": generated_environment,
            "resolved_environment_config": resolved_environment_config,
            "resolved_system_context": resolved_system_context,
            "resolved_task_context": resolved_task_context,
            "template_vars": template_vars,
            "prompts": prompts,
            "stage_reviews": stage_reviews,
            "stage_failures": seed_stage_failures,
        }

    def generate_single(
        self,
        scenario_key: str,
        scenario: Dict,
        max_iterations: int,
        randomize_params: bool,
        doc_context: Optional[DocFile] = None,
        seed_bundle: Optional[Dict[str, Any]] = None,
        rollout_metadata: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        Generate a single example through stage-by-stage pipeline.

        Pipeline (all stages config-driven via scenario.rubrics):
            1. Generate system prompt → validate with rubrics.system_prompt
            2. Generate user request → validate with rubrics.user
            3. Generate thinking (if rubrics.thinking specified) → validate
            4. Generate assistant response → validate with rubrics.response

        Args:
            scenario_key: Scenario identifier
            scenario: Scenario configuration
            max_iterations: Max improvement iterations per stage
            randomize_params: Whether to randomize LLM parameters
            doc_context: Optional document context for template variables

        Returns:
            GenerationResult with final example and metrics
        """
        prompts = scenario.get("prompts", {})
        # Get all rubrics from scenario config - fully config-driven
        scenario_rubrics = scenario.get("rubrics", {})

        # Build template variables from doc context
        template_vars = {}
        if doc_context:
            template_vars["doc_content"] = doc_context.content
            template_vars["doc_path"] = doc_context.path

        def render_prompt(prompt: str) -> str:
            """Render template variables in prompt."""
            result = prompt
            for key, value in template_vars.items():
                result = result.replace(f"{{{key}}}", value)
            return result

        example = {"conversations": []}
        stage_failures = []
        total_iterations = 0
        stage_reviews: Dict[str, Any] = {}
        if seed_bundle is not None:
            environment_mode = seed_bundle.get("environment_mode", self._resolve_environment_mode(scenario))
            generated_environment = deepcopy(seed_bundle.get("generated_environment") or {})
            resolved_environment_config = _deep_merge_dicts(
                deepcopy(seed_bundle.get("resolved_environment_config")),
                scenario.get("environment"),
            )
            resolved_system_context = _deep_merge_dicts(
                deepcopy(seed_bundle.get("resolved_system_context")),
                scenario.get("system_context"),
            )
            resolved_task_context = _deep_merge_dicts(
                deepcopy(seed_bundle.get("resolved_task_context")),
                scenario.get("task_context"),
            )
            for key, value in (seed_bundle.get("template_vars") or {}).items():
                template_vars[key] = value
            stage_reviews.update(deepcopy(seed_bundle.get("stage_reviews") or {}))
            stage_failures.extend(list(seed_bundle.get("stage_failures") or []))
        else:
            environment_mode = self._resolve_environment_mode(scenario)
            generated_environment = {}
            if environment_mode in {"generated", "hybrid"}:
                self._log_stage(scenario_key, "environment_generation", "start")
                generation_cfg = scenario.get("environment_generation") or {}
                generated_environment = self._generate_environment_spec(
                    scenario=scenario,
                    render_prompt=render_prompt,
                    randomize_params=randomize_params,
                    trace_label=f"{scenario_key}:environment_generation",
                )
                review = self._run_stage_review(
                    stage_name="environment_generation",
                    stage_config=generation_cfg,
                    scenario_key=scenario_key,
                    scenario=scenario,
                    task_context={},
                    payload=self._build_environment_generation_review_payload(
                        generated_environment=generated_environment,
                    ),
                )
                if review is not None:
                    stage_reviews["environment_generation"] = review
                    if review.get("passed") is False and review.get("enforce", True):
                        stage_failures.append("environment_generation")
                self._log_stage(
                    scenario_key,
                    "environment_generation",
                    "done",
                    extra=f"keys={sorted(generated_environment.keys())}" if generated_environment else "empty",
                )
            if generated_environment:
                template_vars["environment_json"] = json.dumps(generated_environment, indent=2)

            base_environment_config = scenario.get("environment")
            generated_environment_config = (
                generated_environment.get("environment") if environment_mode in {"generated", "hybrid"} else None
            )
            resolved_environment_config = _deep_merge_dicts(
                base_environment_config,
                generated_environment_config,
            )
            resolved_system_context = _deep_merge_dicts(
                scenario.get("system_context"),
                generated_environment.get("system_context"),
            )
            resolved_task_context = _deep_merge_dicts(
                scenario.get("task_context"),
                generated_environment.get("task_context"),
            )

        derived_task_spec = derive_task_spec(
            scenario_key=scenario_key,
            scenario=scenario,
            environment_config=resolved_environment_config or {},
            existing_task_context=resolved_task_context,
        )
        resolved_task_context = _make_json_safe(derived_task_spec.task_context)
        template_vars.update(_task_context_template_vars(resolved_task_context))
        resolved_environment_config = _render_template_object(
            resolved_environment_config,
            template_vars,
            resolved_task_context,
        )
        resolved_system_context = _render_template_object(
            resolved_system_context,
            template_vars,
            resolved_task_context,
        )

        # Stage 1: Assistant's system prompt
        # Priority: assistant_system > system (legacy) > system: true (generate)
        assistant_system_prompt = prompts.get("assistant_system")
        system_enabled = scenario.get("system", True)
        system_template = scenario.get("system_template")

        if system_template == "mocked_workspace_vault":
            system_content = self._render_mocked_workspace_system_prompt(
                system_context=resolved_system_context or {},
                environment_config=resolved_environment_config or {},
                tool_schema=(self.environment_validator.tool_schema if self.environment_validator else None),
            )
            example["conversations"].append({
                "role": "system",
                "content": system_content
            })
            review = self._run_stage_review(
                stage_name="system_generation",
                stage_config=scenario.get("system_generation"),
                scenario_key=scenario_key,
                scenario=scenario,
                task_context=resolved_task_context or {},
                payload={
                    "text": system_content,
                    "system_text": system_content,
                    "generated_environment": generated_environment,
                    "system_context": resolved_system_context or {},
                    "environment_config": resolved_environment_config or {},
                },
            )
            if review is not None:
                stage_reviews["system_generation"] = review
                if review.get("passed") is False and review.get("enforce", True):
                    stage_failures.append("system_generation")

        elif assistant_system_prompt:
            # New style: assistant_system in prompts (static template with vars)
            system_content = render_prompt(assistant_system_prompt)
            example["conversations"].append({
                "role": "system",
                "content": system_content
            })
            review = self._run_stage_review(
                stage_name="system_generation",
                stage_config=scenario.get("system_generation"),
                scenario_key=scenario_key,
                scenario=scenario,
                task_context=resolved_task_context or {},
                payload={
                    "text": system_content,
                    "system_text": system_content,
                    "generated_environment": generated_environment,
                    "system_context": resolved_system_context or {},
                    "environment_config": resolved_environment_config or {},
                },
            )
            if review is not None:
                stage_reviews["system_generation"] = review
                if review.get("passed") is False and review.get("enforce", True):
                    stage_failures.append("system_generation")

        elif system_enabled and isinstance(system_enabled, bool):
            # Legacy: generate system prompt from "system" prompt
            system_prompt = render_prompt(prompts.get("system", ""))
            if system_prompt:
                self._log_stage(scenario_key, "system_prompt", "start")
                system_content = self._call_llm(
                    system_prompt,
                    randomize_params,
                    trace_label=f"{scenario_key}:system_prompt",
                )
                self._log_stage(scenario_key, "system_prompt", "done", extra=f"chars={len(system_content)}")

                example["conversations"].append({
                    "role": "system",
                    "content": system_content
                })
                review = self._run_stage_review(
                    stage_name="system_generation",
                    stage_config=scenario.get("system_generation"),
                    scenario_key=scenario_key,
                    scenario=scenario,
                    task_context=resolved_task_context or {},
                    payload={
                        "text": system_content,
                        "system_text": system_content,
                        "generated_environment": generated_environment,
                        "system_context": resolved_system_context or {},
                        "environment_config": resolved_environment_config or {},
                    },
                )
                if review is not None:
                    stage_reviews["system_generation"] = review
                    if review.get("passed") is False and review.get("enforce", True):
                        stage_failures.append("system_generation")

                # Validate/improve system stage (config-driven)
                system_rubrics = scenario_rubrics.get("system_prompt", [])
                if self.enable_stage_validation and system_rubrics:
                    improved, iterations, passed = self._improve_stage(
                        example,
                        stage="system_prompt",
                        rubrics=system_rubrics,
                        max_iterations=max_iterations
                    )
                    example = improved
                    total_iterations += iterations
                    if not passed:
                        stage_failures.append("system_prompt")

        elif isinstance(system_enabled, dict) and "template" in system_enabled:
            # Use fixed template file
            template_path = Path(system_enabled["template"])
            if template_path.exists():
                with open(template_path) as f:
                    system_content = render_prompt(f.read())
                example["conversations"].append({
                    "role": "system",
                    "content": system_content
                })
                review = self._run_stage_review(
                    stage_name="system_generation",
                    stage_config=scenario.get("system_generation"),
                    scenario_key=scenario_key,
                    scenario=scenario,
                    task_context=resolved_task_context or {},
                    payload={
                        "text": system_content,
                        "system_text": system_content,
                        "generated_environment": generated_environment,
                        "system_context": resolved_system_context or {},
                        "environment_config": resolved_environment_config or {},
                    },
                )
                if review is not None:
                    stage_reviews["system_generation"] = review
                    if review.get("passed") is False and review.get("enforce", True):
                        stage_failures.append("system_generation")

        system_message = next((msg for msg in example["conversations"] if msg.get("role") == "system"), None)
        if system_message is not None:
            review = self._run_stage_review(
                stage_name="system_generation",
                stage_config=scenario.get("system_generation"),
                scenario_key=scenario_key,
                scenario=scenario,
                task_context=resolved_task_context or {},
                payload={
                    "text": system_message.get("content") or "",
                    "system_text": system_message.get("content") or "",
                    "generated_environment": generated_environment,
                    "system_context": resolved_system_context or {},
                    "environment_config": resolved_environment_config or {},
                },
            )
            if review is not None:
                stage_reviews["system_generation"] = review
                if review.get("passed") is False and review.get("enforce", True) and "system_generation" not in stage_failures:
                    stage_failures.append("system_generation")

        # Stage 2: User request
        # Check for user_system (persona) to guide user generation
        user_system_prompt = prompts.get("user_system", "")
        if user_system_prompt:
            user_system_prompt = render_prompt(user_system_prompt)

        user_prompt = render_prompt(prompts.get("user", ""))

        # Build context for user generation
        user_context_parts = []
        if user_system_prompt:
            user_context_parts.append(f"Your persona:\n{user_system_prompt}")
        # Add assistant's system context if present
        existing_context = self._build_user_context(example)
        if existing_context:
            user_context_parts.append(existing_context)
        user_context_parts.append(
            "Return only a natural user request in plain text. Do not output JSON, tool calls, code fences, parameter blobs, or internal instructions."
        )
        user_context_parts.extend(_user_generation_style_instructions(scenario))

        user_context = "\n\n".join(user_context_parts)
        user_stage_cfg = scenario.get("user_generation") if isinstance(scenario.get("user_generation"), dict) else {}
        self._log_stage(scenario_key, "user", "start")
        user_content = self._call_llm(
            f"{user_context}\n\n{user_prompt}",
            randomize_params,
            trace_label=f"{scenario_key}:user",
            llm_clients=self._get_stage_llm_clients(user_stage_cfg),
            max_retries=int(user_stage_cfg.get("max_retries", 3) or 3),
        )
        self._log_stage(scenario_key, "user", "done", extra=f"chars={len(user_content)}")

        example["conversations"].append({
            "role": "user",
            "content": user_content
        })
        review = self._run_stage_review(
            stage_name="user_generation",
            stage_config=scenario.get("user_generation"),
            scenario_key=scenario_key,
            scenario=scenario,
            task_context=resolved_task_context or {},
            payload={
                "text": user_content,
                "user_text": user_content,
                "task_context": resolved_task_context or {},
                "allowed_tools": _resolve_allowed_tool_names(
                    scenario=scenario,
                    tool_schema=(self.environment_validator.tool_schema if self.environment_validator else None),
                ),
            },
        )
        _apply_stage_review_result(stage_failures, stage_reviews, "user_generation", review)

        # Validate/improve user stage (config-driven)
        user_rubrics = scenario_rubrics.get("user", [])
        if self.enable_stage_validation and user_rubrics:
            improved, iterations, passed = self._improve_stage(
                example,
                stage="user",
                rubrics=user_rubrics,
                max_iterations=max_iterations
            )
            example = improved
            total_iterations += iterations
            if not passed:
                stage_failures.append("user")

        latest_user_message = next(
            (msg for msg in reversed(example["conversations"]) if msg.get("role") == "user"),
            None,
        )
        if latest_user_message is not None:
            review = self._run_stage_review(
                stage_name="user_generation",
                stage_config=scenario.get("user_generation"),
                scenario_key=scenario_key,
                scenario=scenario,
                task_context=resolved_task_context or {},
                payload={
                    "text": latest_user_message.get("content") or "",
                    "user_text": latest_user_message.get("content") or "",
                    "task_context": resolved_task_context or {},
                    "allowed_tools": _resolve_allowed_tool_names(
                        scenario=scenario,
                        tool_schema=(self.environment_validator.tool_schema if self.environment_validator else None),
                    ),
                },
            )
            _apply_stage_review_result(stage_failures, stage_reviews, "user_generation", review)

        # Stage 3: Thinking (if rubrics.thinking specified)
        thinking_rubrics = scenario_rubrics.get("thinking", [])
        thinking_content = None

        if thinking_rubrics:
            # Generate thinking as separate stage
            thinking_prompt = render_prompt(prompts.get("thinking", ""))
            if thinking_prompt:
                thinking_context = self._build_assistant_context(example, scenario)
                self._log_stage(scenario_key, "thinking", "start")
                thinking_content = self._call_llm(
                    f"{thinking_context}\n\n{thinking_prompt}",
                    randomize_params,
                    trace_label=f"{scenario_key}:thinking",
                    llm_clients=self._get_stage_llm_clients(scenario.get("thinking_generation")),
                    max_retries=int(((scenario.get("thinking_generation") or {}).get("max_retries", 3)) or 3),
                )
                self._log_stage(
                    scenario_key,
                    "thinking",
                    "done",
                    extra=f"chars={len(thinking_content)}",
                )

                # Temporarily add thinking to example for validation
                # (will be combined with response later)
                temp_example = {**example, "thinking": thinking_content}

                if self.enable_stage_validation:
                    improved, iterations, passed = self._improve_stage(
                        temp_example,
                        stage="thinking",
                        rubrics=thinking_rubrics,
                        max_iterations=max_iterations
                    )
                    thinking_content = improved.get("thinking", thinking_content)
                    total_iterations += iterations
                    if not passed:
                        stage_failures.append("thinking")

        # Stage 4: Assistant response (text or tool)
        assistant_prompt = render_prompt(prompts.get("assistant", ""))
        loop_cfg = (
            resolved_environment_config.get("loop")
            if isinstance(resolved_environment_config, dict) and isinstance(resolved_environment_config.get("loop"), dict)
            else {}
        )
        use_agentic_loop = bool(
            self.environment_validator is not None
            and run_environment_episode is not None
            and validate_assistant_response is not None
            and loop_cfg.get("enabled")
        )

        if use_agentic_loop:
            assistant_msg, example, environment_trace = self._generate_agentic_episode(
                scenario_key=scenario_key,
                scenario=scenario,
                example=example,
                assistant_prompt=assistant_prompt,
                randomize_params=randomize_params,
                resolved_system_context=resolved_system_context or {},
                resolved_environment_config=resolved_environment_config or {},
                resolved_task_context=resolved_task_context or {},
                hard_requirements=derived_task_spec.hard_requirements,
                quality_rubric=derived_task_spec.quality_rubric,
                thinking_content=thinking_content,
                stage_failures=stage_failures,
            )
        else:
            assistant_context = self._build_assistant_context(example, scenario)

            # Include thinking in context if it was generated
            if thinking_content:
                assistant_context = f"{assistant_context}\n\nYour thinking:\n{thinking_content}"

            self._log_stage(scenario_key, "assistant", "start")
            assistant_content = self._generate_assistant_response(
                scenario=scenario,
                system_context=resolved_system_context or {},
                assistant_context=assistant_context,
                assistant_prompt=assistant_prompt,
                randomize_params=randomize_params,
                trace_label=f"{scenario_key}:assistant",
            )
            self._log_stage(scenario_key, "assistant", "done", extra=f"chars={len(assistant_content)}")

            # Parse assistant response for tool calls
            assistant_msg = self._parse_assistant_response(assistant_content, scenario)

            # If thinking was generated separately, prepend it to content
            if thinking_content and assistant_msg.get("tool_calls"):
                # For tool calls, thinking goes in content field
                assistant_msg["content"] = f"<thinking>{thinking_content}</thinking>"

            example["conversations"].append(assistant_msg)
            review = self._run_stage_review(
                stage_name="assistant_generation",
                stage_config=scenario.get("assistant_generation"),
                scenario_key=scenario_key,
                scenario=scenario,
                task_context=resolved_task_context or {},
                payload={
                    "assistant_response": assistant_msg,
                    "assistant_response_json": json.dumps(_make_json_safe(assistant_msg), ensure_ascii=False, indent=2),
                    "text": self._stringify_assistant_message(assistant_msg),
                    "task_context": resolved_task_context or {},
                },
            )
            if review is not None:
                stage_reviews["assistant_generation"] = review
                if review.get("passed") is False and review.get("enforce", True):
                    stage_failures.append("assistant_generation")

        # Validate/improve response stage (config-driven)
        response_rubrics = scenario_rubrics.get("response", [])
        if self.enable_stage_validation and response_rubrics:
            improved, iterations, passed = self._improve_stage(
                example,
                stage="response",
                rubrics=response_rubrics,
                max_iterations=max_iterations
            )
            example = improved
            total_iterations += iterations
            if not passed:
                stage_failures.append("response")

        # Add metadata
        environment_trace = locals().get("environment_trace")
        if self.environment_validator is not None and environment_trace is None:
            try:
                system_prompt_text = ""
                for msg in example["conversations"]:
                    if msg.get("role") == "system":
                        system_prompt_text = msg.get("content") or ""
                        break
                expected_tools = scenario.get("expected_tools")
                if not expected_tools and scenario.get("tool"):
                    expected_tools = [scenario.get("tool")]
                env_result = self.environment_validator.validate_response(
                    system_prompt=system_prompt_text,
                    response=assistant_msg,
                    environment_config=resolved_environment_config,
                    expected_tools=expected_tools,
                )
                environment_trace = env_result.to_dict()
                if not env_result.passed:
                    stage_failures.append("environment")
            except Exception as exc:
                stage_failures.append("environment")
                environment_trace = {
                    "passed": False,
                    "issues": [{"level": "error", "message": f"Environment validation failed: {exc}"}],
                    "executed_tools": [],
                }

        example["metadata"] = {
            "scenario": scenario_key,
            "category": scenario_key,
            "type": scenario.get("type", "unknown"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment_mode": environment_mode,
        }
        if rollout_metadata:
            example["metadata"]["environment_seed"] = dict(rollout_metadata)
        if generated_environment:
            example["metadata"]["generated_environment"] = generated_environment
        if resolved_environment_config:
            example["metadata"]["resolved_environment_config"] = _make_json_safe(
                resolved_environment_config
            )
        if resolved_task_context:
            example["metadata"]["task_context"] = resolved_task_context
        if derived_task_spec.hard_requirements:
            example["metadata"]["hard_requirements"] = derived_task_spec.hard_requirements
        if derived_task_spec.quality_rubric:
            example["metadata"]["quality_rubric"] = derived_task_spec.quality_rubric
        if derived_task_spec.derivation_summary:
            example["metadata"]["derivation_summary"] = derived_task_spec.derivation_summary
        if environment_trace is not None:
            example["metadata"]["environment"] = environment_trace
            if isinstance(environment_trace.get("judge_trace"), list):
                example["metadata"]["judge"] = {
                    "in_loop": True,
                    "trace": environment_trace.get("judge_trace", []),
                    "feedback_visible_to_model": bool(
                        ((scenario.get("judge") or {}).get("in_loop") or {}).get("feedback_visible_to_model", False)
                    ),
                }
        final_review = self._run_stage_review(
            stage_name="final",
            stage_config=scenario.get("final_judge"),
            scenario_key=scenario_key,
            scenario=scenario,
            task_context=resolved_task_context or {},
            payload={
                "user_text": user_content,
                "assistant_response": assistant_msg,
                "assistant_response_json": json.dumps(_make_json_safe(assistant_msg), ensure_ascii=False, indent=2),
                "environment_result": environment_trace or {},
                "environment_passed": bool((environment_trace or {}).get("passed")) if environment_trace is not None else None,
                "final_text_required": (environment_trace or {}).get("final_text_required") if isinstance(environment_trace, dict) else None,
                "final_text_satisfied": (environment_trace or {}).get("final_text_satisfied") if isinstance(environment_trace, dict) else None,
                "conversation_trace": example.get("conversation_trace") or [],
                "task_context": resolved_task_context or {},
                "hard_requirements": derived_task_spec.hard_requirements,
                "quality_rubric": derived_task_spec.quality_rubric,
            },
        )
        if final_review is not None:
            stage_reviews["final"] = final_review
            if final_review.get("passed") is False and final_review.get("enforce", True):
                stage_failures.append("final")
        if doc_context:
            example["metadata"]["source_doc"] = doc_context.path
        if stage_reviews:
            example["metadata"]["stage_reviews"] = stage_reviews
        example["metadata"]["labels"] = self._build_metadata_labels(
            scenario_key=scenario_key,
            scenario=scenario,
            environment_mode=environment_mode,
            stage_failures=stage_failures,
            environment_trace=environment_trace,
            generated_environment=generated_environment,
        )

        return GenerationResult(
            example=example,
            scenario_key=scenario_key,
            iterations=total_iterations,
            success=len(stage_failures) == 0,
            stage_failures=stage_failures
        )

    def _run_stage_review(
        self,
        *,
        stage_name: str,
        stage_config: Optional[Dict[str, Any]],
        scenario_key: str,
        scenario: Dict[str, Any],
        task_context: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(stage_config, dict):
            return None

        gates_cfg = stage_config.get("gates") if isinstance(stage_config.get("gates"), list) else []
        judge_cfg = stage_config.get("judge") if isinstance(stage_config.get("judge"), dict) else {}
        if not gates_cfg and not judge_cfg:
            return None

        review: Dict[str, Any] = {
            "stage": stage_name,
            "enforce": bool(stage_config.get("enforce", True)),
            "passed": True,
            "gates": [],
            "judge": None,
        }

        gate_results = run_stage_gates(gates_cfg, payload)
        if gate_results:
            review["gates"] = [result.to_dict() for result in gate_results]
            if any(not result.passed for result in gate_results):
                review["passed"] = False
            if self.logger:
                failed = sum(1 for result in gate_results if not result.passed)
                self.logger.info(f"[{scenario_key}] {stage_name} gates done (failed={failed}/{len(gate_results)})")

        judge_result = self._run_configured_stage_judge(
            stage_name=stage_name,
            judge_config=judge_cfg,
            scenario_key=scenario_key,
            scenario=scenario,
            task_context=task_context,
            payload=payload,
        )
        if judge_result is not None:
            review["judge"] = judge_result
            if not judge_result.get("passed", True):
                review["passed"] = False
            min_quality_score = judge_cfg.get("min_quality_score")
            if min_quality_score is not None:
                score = judge_result.get("score")
                if score is None or float(score) < float(min_quality_score):
                    review["passed"] = False
                    review["judge"]["below_min_quality_score"] = True
            if self.logger:
                self.logger.info(
                    f"[{scenario_key}] {stage_name} judge done "
                    f"(passed={judge_result.get('passed')} score={judge_result.get('score')})"
                )

        return review

    def _build_environment_generation_review_payload(
        self,
        *,
        generated_environment: Dict[str, Any],
        seed_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        environment = generated_environment.get("environment") if isinstance(generated_environment, dict) else {}
        fixture_snapshot = {}
        if isinstance(environment, dict):
            try:
                fixture = _merged_fixture_from_config(environment)
                fixture_snapshot = {
                    "directories": list(fixture.directories),
                    "files": [
                        {"path": path, "content": content}
                        for path, content in fixture.files.items()
                    ],
                }
            except Exception as exc:
                fixture_snapshot = {"error": str(exc)}
        payload = {
            "value": {
                "environment": {
                    "fixture": fixture_snapshot,
                    "assertions": list(environment.get("assertions") or []) if isinstance(environment, dict) else [],
                },
                "system_context": generated_environment.get("system_context") if isinstance(generated_environment, dict) else {},
                "task_context": generated_environment.get("task_context") if isinstance(generated_environment, dict) else {},
            },
            "generated_environment": generated_environment,
        }
        if seed_id:
            payload["seed_id"] = seed_id
        return payload

    def _get_stage_llm_clients(self, stage_config: Optional[Dict[str, Any]]) -> List[Any]:
        if not isinstance(stage_config, dict):
            return [self.llm_client]

        client_chain: List[Any] = []
        primary_spec = self._normalize_stage_llm_spec(stage_config)
        if primary_spec is None:
            client_chain.append(self.llm_client)
        else:
            client_chain.append(self._get_or_create_llm_client(primary_spec))

        fallback_specs = stage_config.get("fallback_models")
        if isinstance(fallback_specs, list):
            for raw_spec in fallback_specs:
                spec = self._normalize_stage_llm_spec(raw_spec)
                if spec is None:
                    continue
                client = self._get_or_create_llm_client(spec)
                if all(client is not existing for existing in client_chain):
                    client_chain.append(client)
        return client_chain

    def _normalize_stage_llm_spec(self, value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, str):
            model = value.strip()
            return {"model": model} if model else None
        if not isinstance(value, dict):
            return None

        model = str(value.get("model") or "").strip()
        provider = str(value.get("provider") or "").strip().lower()
        provider_routing = value.get("provider_routing")
        timeout_seconds = value.get("timeout_seconds")

        has_override = bool(model or provider or provider_routing is not None or timeout_seconds is not None)
        if not has_override:
            return None

        spec: Dict[str, Any] = {}
        if model:
            spec["model"] = model
        if provider:
            spec["provider"] = provider
        if provider_routing is not None:
            spec["provider_routing"] = provider_routing
        if timeout_seconds is not None:
            spec["timeout_seconds"] = timeout_seconds
        return spec

    def _get_or_create_llm_client(self, spec: Dict[str, Any]):
        base_provider = str(getattr(self.llm_client, "provider_name", "openrouter") or "openrouter").strip().lower()
        base_model = str(getattr(self.llm_client, "model_name", "") or "").strip()
        base_provider_routing = deepcopy(getattr(self.llm_client, "provider", None))
        base_timeout_seconds = getattr(self.llm_client, "timeout_seconds", None)

        provider = str(spec.get("provider") or base_provider).strip().lower()
        model = str(spec.get("model") or base_model).strip()
        provider_routing = deepcopy(spec.get("provider_routing", base_provider_routing))
        timeout_seconds = spec.get("timeout_seconds", base_timeout_seconds)

        cache_key = (
            provider,
            model,
            json.dumps(provider_routing, sort_keys=True) if provider_routing is not None else "",
            str(timeout_seconds) if timeout_seconds is not None else "",
        )
        cached = self._llm_client_cache.get(cache_key)
        if cached is not None:
            return cached

        if provider == base_provider and model == base_model:
            same_routing = provider_routing == base_provider_routing
            same_timeout = (
                timeout_seconds == base_timeout_seconds
                or (timeout_seconds is None and base_timeout_seconds is None)
            )
            if same_routing and same_timeout:
                self._llm_client_cache[cache_key] = self.llm_client
                return self.llm_client

        config_defaults = {
            "provider": provider,
            "model": model,
        }
        if provider_routing is not None:
            config_defaults["provider_routing"] = provider_routing
        if timeout_seconds is not None:
            config_defaults["timeout_seconds"] = timeout_seconds

        client = create_client(
            provider=provider,
            model=model,
            config_defaults=config_defaults,
        )
        self._llm_client_cache[cache_key] = client
        return client

    def _run_configured_stage_judge(
        self,
        *,
        stage_name: str,
        judge_config: Optional[Dict[str, Any]],
        scenario_key: str,
        scenario: Dict[str, Any],
        task_context: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(judge_config, dict) or not bool(judge_config.get("enabled")):
            return None
        prompt_template = str(judge_config.get("prompt") or "").strip()
        if not prompt_template:
            return None

        judge = ConfigurableStageJudge(
            llm_client=self.llm_client,
            llm_clients=self._get_stage_llm_clients(judge_config),
            prompt_template=prompt_template,
            system_prompt=judge_config.get("system"),
            output_schema=judge_config.get("output_schema"),
            temperature=float(judge_config.get("temperature", 0.2) or 0.2),
            max_tokens=judge_config.get("max_tokens"),
            max_retries=int(judge_config.get("max_retries", 3) or 3),
        )
        template_vars = self._build_stage_judge_template_vars(
            stage_name=stage_name,
            scenario_key=scenario_key,
            scenario=scenario,
            task_context=task_context,
            payload=payload,
        )
        return judge.judge(template_vars).to_dict()

    def _build_stage_judge_template_vars(
        self,
        *,
        stage_name: str,
        scenario_key: str,
        scenario: Dict[str, Any],
        task_context: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, str]:
        safe_payload = _make_json_safe(payload or {})
        safe_scenario = _make_json_safe(scenario or {})
        safe_task_context = _make_json_safe(task_context or {})
        return {
            "stage_name": stage_name,
            "scenario_key": scenario_key,
            "scenario_json": json.dumps(safe_scenario, ensure_ascii=False, indent=2),
            "task_context_json": json.dumps(safe_task_context, ensure_ascii=False, indent=2),
            "payload_json": json.dumps(safe_payload, ensure_ascii=False, indent=2),
            "value_json": json.dumps(safe_payload.get("value"), ensure_ascii=False, indent=2),
            "text": str(safe_payload.get("text") or ""),
            "system_text": str(safe_payload.get("system_text") or ""),
            "user_text": str(safe_payload.get("user_text") or ""),
            "assistant_response_json": (
                safe_payload.get("assistant_response_json")
                if isinstance(safe_payload.get("assistant_response_json"), str)
                else json.dumps(safe_payload.get("assistant_response"), ensure_ascii=False, indent=2)
            ),
            "environment_result_json": json.dumps(safe_payload.get("environment_result") or {}, ensure_ascii=False, indent=2),
            "conversation_trace_json": json.dumps(safe_payload.get("conversation_trace") or [], ensure_ascii=False, indent=2),
            "hard_requirements_json": json.dumps(safe_payload.get("hard_requirements") or [], ensure_ascii=False, indent=2),
            "quality_rubric_json": json.dumps(safe_payload.get("quality_rubric") or [], ensure_ascii=False, indent=2),
        }

    def _generate_agentic_episode(
        self,
        *,
        scenario_key: str,
        scenario: Dict[str, Any],
        example: Dict[str, Any],
        assistant_prompt: str,
        randomize_params: bool,
        resolved_system_context: Dict[str, Any],
        resolved_environment_config: Dict[str, Any],
        resolved_task_context: Dict[str, Any],
        hard_requirements: List[Dict[str, Any]],
        quality_rubric: List[str],
        thinking_content: Optional[str],
        stage_failures: List[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Generate a multi-turn agentic rollout using the shared episode runner."""
        loop_cfg = resolved_environment_config.get("loop") if isinstance(resolved_environment_config, dict) else {}
        max_turns = int(loop_cfg.get("max_turns", 6) or 6)
        max_tool_steps = int(loop_cfg.get("max_tool_steps", resolved_environment_config.get("max_steps", 0)) or 0)
        stop_on_text_response = bool(loop_cfg.get("stop_on_text_response", True))
        stop_on_environment_pass = bool(loop_cfg.get("stop_on_environment_pass", False))
        require_final_text_after_pass = bool(
            loop_cfg.get("require_final_text_after_pass", loop_cfg.get("require_final_text", False))
        )
        final_text_prompt = loop_cfg.get("final_text_prompt")
        continue_on_execution_error = bool(
            loop_cfg.get("continue_on_execution_error", str(loop_cfg.get("mode", "strict")).strip().lower() == "agentic")
        )
        stuck_repeat_limit = int(loop_cfg.get("stuck_repeat_limit", 2) or 2)
        no_progress_window = int(loop_cfg.get("no_progress_window", 3) or 3)
        tool_result_format = str(loop_cfg.get("tool_result_format", "json") or "json")
        judge_cfg = scenario.get("judge") if isinstance(scenario.get("judge"), dict) else {}
        in_loop_judge_cfg = judge_cfg.get("in_loop") if isinstance(judge_cfg.get("in_loop"), dict) else {}
        judge_feedback_visible_to_model = bool(in_loop_judge_cfg.get("feedback_visible_to_model", False))
        judge_stop_on_hard_failure = bool(in_loop_judge_cfg.get("stop_on_hard_failure", False))
        turn_judge = self._build_turn_judge(
            scenario_key=scenario_key,
            scenario=scenario,
            assistant_prompt=assistant_prompt,
            system_context=resolved_system_context,
            task_context=resolved_task_context,
            hard_requirements=hard_requirements,
            quality_rubric=quality_rubric,
            judge_config=in_loop_judge_cfg,
        )

        system_prompt_text = ""
        for msg in example["conversations"]:
            if msg.get("role") == "system":
                system_prompt_text = msg.get("content") or ""
                break

        session = self.environment_validator.start_session(
            system_prompt=system_prompt_text,
            environment_config=resolved_environment_config,
        )
        try:
            self._log_stage(scenario_key, "assistant_loop", "start")
            episode = run_environment_episode(
                initial_messages=example["conversations"],
                session=session,
                respond=lambda messages, turn_index: self._synthchat_loop_response(
                    scenario=scenario,
                    system_context=resolved_system_context,
                    messages=messages,
                    assistant_prompt=assistant_prompt,
                    randomize_params=randomize_params,
                    scenario_key=scenario_key,
                    turn_index=turn_index,
                    thinking_content=thinking_content,
                ),
                validate=self._validate_agentic_synthchat_response,
                max_turns=max_turns,
                max_tool_steps=max_tool_steps,
                stop_on_text_response=stop_on_text_response,
                stop_on_environment_pass=stop_on_environment_pass,
                continue_on_execution_error=continue_on_execution_error,
                stuck_repeat_limit=stuck_repeat_limit,
                no_progress_window=no_progress_window,
                tool_result_format=tool_result_format,
                expected_tools=scenario.get("expected_tools") or ([scenario.get("tool")] if scenario.get("tool") else None),
                require_expected_tools=bool(resolved_environment_config.get("require_expected_tools")),
                stringify_response=self._stringify_assistant_message,
                judge_turn=turn_judge,
                judge_feedback_visible_to_model=judge_feedback_visible_to_model,
                judge_stop_on_hard_failure=judge_stop_on_hard_failure,
                require_final_text_after_pass=require_final_text_after_pass,
                final_text_prompt=final_text_prompt,
            )
        finally:
            session.close()

        if not episode.environment_result.passed:
            stage_failures.append("environment")

        if episode.stop_reason in {
            "schema_validation_failed",
            "final_text_tool_calls_emitted",
            "final_text_missing",
            "judge_hard_failure",
            "judge_requested_stop",
        }:
            stage_failures.append("response")

        example["conversations"] = [dict(message) for message in episode.messages]
        example["conversation_trace"] = episode.conversation_trace
        final_response = episode.final_response if isinstance(episode.final_response, dict) else {"role": "assistant", "content": str(episode.final_response or "")}
        self._log_stage(
            scenario_key,
            "assistant_loop",
            "done",
            extra=f"turns={len(episode.turns)} stop={episode.stop_reason}",
        )
        environment_trace = episode.environment_result.to_dict()
        environment_trace["final_text_required"] = episode.final_text_required
        environment_trace["final_text_satisfied"] = episode.final_text_satisfied
        return final_response, example, {
            **environment_trace,
            "judge_trace": list(episode.judge_trace),
        }

    def _build_turn_judge(
        self,
        *,
        scenario_key: str,
        scenario: Dict[str, Any],
        assistant_prompt: str,
        system_context: Dict[str, Any],
        task_context: Dict[str, Any],
        hard_requirements: List[Dict[str, Any]],
        quality_rubric: List[str],
        judge_config: Dict[str, Any],
    ):
        if AgenticTurnJudge is None or not judge_config:
            return None
        if not bool(judge_config.get("enabled")):
            return None
        prompt_template = str(judge_config.get("prompt") or "").strip()
        if not prompt_template:
            return None
        judge = AgenticTurnJudge(
            llm_client=self.llm_client,
            llm_clients=self._get_stage_llm_clients(judge_config),
            prompt_template=prompt_template,
            system_prompt=judge_config.get("system"),
            output_schema=judge_config.get("output_schema"),
            temperature=float(judge_config.get("temperature", 0.2) or 0.2),
            max_tokens=judge_config.get("max_tokens"),
            max_retries=int(judge_config.get("max_retries", 3) or 3),
        )

        def run_judge(turn_payload: Dict[str, Any]):
            template_vars = self._build_turn_judge_template_vars(
                scenario_key=scenario_key,
                scenario=scenario,
                assistant_prompt=assistant_prompt,
                system_context=system_context,
                task_context=task_context,
                hard_requirements=hard_requirements,
                quality_rubric=quality_rubric,
                turn_payload=turn_payload,
            )
            result = judge.judge(template_vars)
            if self.logger:
                self.logger.info(
                    f"[{scenario_key}] turn_judge done "
                    f"(turn={turn_payload.get('turn_index')} passed={result.passed} "
                    f"hard_failure={result.hard_failure} stop={result.should_stop})"
                )
            return result

        return run_judge

    def _build_turn_judge_template_vars(
        self,
        *,
        scenario_key: str,
        scenario: Dict[str, Any],
        assistant_prompt: str,
        system_context: Dict[str, Any],
        task_context: Dict[str, Any],
        hard_requirements: List[Dict[str, Any]],
        quality_rubric: List[str],
        turn_payload: Dict[str, Any],
    ) -> Dict[str, str]:
        messages = turn_payload.get("messages") or []
        safe_scenario = _make_json_safe(scenario or {})
        safe_system_context = _make_json_safe(system_context or {})
        safe_task_context = _make_json_safe(task_context or {})
        safe_hard_requirements = _make_json_safe(hard_requirements or [])
        safe_quality_rubric = _make_json_safe(quality_rubric or [])
        safe_turn_payload = _make_json_safe(turn_payload or {})
        latest_user = ""
        for message in reversed(messages):
            if str(message.get("role", "")).strip() == "user":
                latest_user = str(message.get("content") or "")
                break
        return {
            "scenario_key": scenario_key,
            "assistant_prompt": assistant_prompt,
            "scenario_json": json.dumps(safe_scenario, ensure_ascii=False, indent=2),
            "system_context_json": json.dumps(safe_system_context, ensure_ascii=False, indent=2),
            "task_context_json": json.dumps(safe_task_context, ensure_ascii=False, indent=2),
            "hard_requirements_json": json.dumps(safe_hard_requirements, ensure_ascii=False, indent=2),
            "quality_rubric_json": json.dumps(safe_quality_rubric, ensure_ascii=False, indent=2),
            "messages_json": json.dumps(_make_json_safe(messages), ensure_ascii=False, indent=2),
            "latest_user_message": latest_user,
            "assistant_response_json": json.dumps(safe_turn_payload.get("response_message"), ensure_ascii=False, indent=2),
            "validation_json": json.dumps(safe_turn_payload.get("validation") or {}, ensure_ascii=False, indent=2),
            "environment_step_json": json.dumps(safe_turn_payload.get("environment_step") or {}, ensure_ascii=False, indent=2),
            "environment_preview_json": json.dumps(safe_turn_payload.get("environment_preview") or {}, ensure_ascii=False, indent=2),
            "tool_feedback": str(safe_turn_payload.get("tool_feedback") or ""),
            "turn_index": str(turn_payload.get("turn_index") or ""),
        }

    def _resolve_environment_mode(self, scenario: Dict[str, Any]) -> str:
        """Resolve how environment data should be sourced for this scenario."""
        configured_mode = str(scenario.get("environment_mode") or "").strip().lower()
        if configured_mode in {"provided", "generated", "hybrid"}:
            return configured_mode

        has_generation = bool(scenario.get("environment_generation") or scenario.get("prompts", {}).get("environment"))
        has_base_environment = bool(scenario.get("environment"))
        if has_generation and has_base_environment:
            return "hybrid"
        if has_generation:
            return "generated"
        return "provided"

    def _generate_environment_spec(
        self,
        scenario: Dict,
        render_prompt,
        randomize_params: bool,
        trace_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate structured environment/system-context data before other stages."""
        prompts = scenario.get("prompts", {})
        generation_cfg = scenario.get("environment_generation") or {}
        environment_prompt = generation_cfg.get("prompt") or prompts.get("environment")
        if not environment_prompt:
            return {}

        environment_system = generation_cfg.get("system") or prompts.get("environment_system")
        user_prompt = render_prompt(str(environment_prompt))
        system_prompt = render_prompt(str(environment_system)) if environment_system else None
        schema_name = str(generation_cfg.get("schema") or "").strip()
        max_tokens = generation_cfg.get("max_tokens")

        if schema_name == "canonical_environment":
            user_prompt = _build_canonical_environment_generation_prompt(user_prompt)
            parsed = self._call_llm_structured(
                prompt=user_prompt,
                schema=_build_canonical_environment_schema(),
                randomize=randomize_params,
                system_prompt=system_prompt,
                trace_label=trace_label,
                max_tokens=max_tokens,
                llm_clients=self._get_stage_llm_clients(generation_cfg),
                max_retries=int(generation_cfg.get("max_retries", 3) or 3),
            )
        else:
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(system_prompt)
            prompt_parts.append(user_prompt)
            raw = self._call_llm(
                "\n\n".join(part for part in prompt_parts if part),
                randomize_params,
                trace_label=trace_label,
                max_tokens=max_tokens,
                llm_clients=self._get_stage_llm_clients(generation_cfg),
                max_retries=int(generation_cfg.get("max_retries", 3) or 3),
            )
            parsed = self._parse_json_object(raw)

        if not isinstance(parsed, dict):
            return {}
        return self._normalize_generated_environment(parsed)

    def _generate_assistant_response(
        self,
        *,
        scenario: Dict[str, Any],
        system_context: Dict[str, Any],
        assistant_context: str,
        assistant_prompt: str,
        randomize_params: bool,
        trace_label: Optional[str] = None,
    ) -> str:
        """Generate assistant output, optionally with structured tool schema."""
        generation_cfg = scenario.get("assistant_generation")
        if not isinstance(generation_cfg, dict):
            generation_cfg = {}

        schema_name = str(generation_cfg.get("schema") or "").strip()
        max_tokens = generation_cfg.get("max_tokens")
        llm_clients = self._get_stage_llm_clients(generation_cfg)
        max_retries = int(generation_cfg.get("max_retries", 3) or 3)
        prompt = f"{assistant_context}\n\n{assistant_prompt}"
        if schema_name == "use_tools_response":
            wrapper_name = _tool_wrapper_name(self.environment_validator.tool_schema if self.environment_validator else None)
            allowed_tools = _resolve_allowed_tool_names(
                scenario=scenario,
                tool_schema=(self.environment_validator.tool_schema if self.environment_validator else None),
            )
            session_id, workspace_id = _resolve_context_defaults(
                system_context=system_context,
            )
            prompt = _build_use_tools_generation_prompt(
                base_prompt=prompt,
                wrapper_name=wrapper_name,
                allowed_tools=allowed_tools,
            )
            payload = self._call_llm_structured(
                prompt=prompt,
                schema=_build_use_tools_response_schema(
                    wrapper_name=wrapper_name,
                    allowed_tools=allowed_tools,
                    session_id=session_id,
                    workspace_id=workspace_id,
                ),
                randomize=randomize_params,
                trace_label=trace_label,
                max_tokens=max_tokens,
                llm_clients=llm_clients,
                max_retries=max_retries,
            )
            return json.dumps(payload)

        return self._call_llm(
            prompt,
            randomize_params,
            trace_label=trace_label,
            max_tokens=max_tokens,
            llm_clients=llm_clients,
            max_retries=max_retries,
        )

    def _build_metadata_labels(
        self,
        scenario_key: str,
        scenario: Dict[str, Any],
        environment_mode: str,
        stage_failures: List[str],
        environment_trace: Optional[Dict[str, Any]],
        generated_environment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build structured labels for downstream filtering and KTO/GRPO slicing."""
        tags = [str(tag).strip() for tag in scenario.get("tags", []) if str(tag).strip()]
        triggers = [str(trigger).strip() for trigger in scenario.get("triggers", []) if str(trigger).strip()]
        flat_labels = {
            f"scenario:{scenario_key}",
            f"type:{scenario.get('type', 'unknown')}",
            f"environment_mode:{environment_mode}",
        }

        tool_name = str(scenario.get("tool") or "").strip()
        if tool_name:
            flat_labels.add(f"tool:{tool_name}")

        for tag in tags:
            flat_labels.add(f"tag:{tag}")

        for trigger in triggers:
            slug = _slugify_label(trigger)
            if slug:
                flat_labels.add(f"trigger:{slug}")

        stage_failure_labels = []
        for stage in sorted({str(stage).strip() for stage in stage_failures if str(stage).strip()}):
            stage_failure_labels.append(stage)
            flat_labels.add(f"stage_failure:{stage}")

        issues = []
        issue_labels = set()
        executed_tools = []
        executed_tool_names = []
        tool_error_names = []
        environment_passed = None
        if isinstance(environment_trace, dict):
            environment_passed = bool(environment_trace.get("passed"))
            flat_labels.add("environment:present")
            flat_labels.add(f"environment_passed:{str(environment_passed).lower()}")

            for issue in environment_trace.get("issues", []) or []:
                if not isinstance(issue, dict):
                    continue
                issue_message = str(issue.get("message", "")).strip()
                issue_level = str(issue.get("level", "")).strip().lower() or "unknown"
                if issue_message:
                    issues.append({"level": issue_level, "message": issue_message})
                for label in _classify_environment_issue(issue_message):
                    issue_labels.add(label)
                    flat_labels.add(f"issue:{label}")

            for tool in environment_trace.get("executed_tools", []) or []:
                if not isinstance(tool, dict):
                    continue
                name = str(tool.get("name", "")).strip()
                if not name:
                    continue
                executed_tools.append(
                    {
                        "name": name,
                        "status": str(tool.get("status", "ok")).strip() or "ok",
                    }
                )
                executed_tool_names.append(name)
                flat_labels.add(f"executed_tool:{name}")
                status = str(tool.get("status", "ok")).strip().lower() or "ok"
                if status != "ok":
                    tool_error_names.append(name)
                    issue_labels.add("tool_runtime_error")
                    flat_labels.add("issue:tool_runtime_error")

        has_environment = environment_trace is not None
        if generated_environment:
            flat_labels.add("environment:generated_payload")

        if has_environment and environment_passed is True and not stage_failure_labels:
            flat_labels.add("kto_candidate:positive")
        elif has_environment and environment_passed is False:
            flat_labels.add("kto_candidate:negative")

        if "environment" in stage_failure_labels:
            flat_labels.add("failure_type:environment")
        if any(stage in stage_failure_labels for stage in ("response", "thinking")):
            flat_labels.add("failure_type:behavior")
        if any(stage in stage_failure_labels for stage in ("system_prompt", "user", "system_generation", "user_generation", "assistant_generation", "environment_generation", "final")):
            flat_labels.add("failure_type:generation")

        if any(label in issue_labels for label in {"missing_expected_tool", "retrieval_missing"}):
            flat_labels.add("behavior:retrieval_failure")
        if "frontmatter_missing" in issue_labels:
            flat_labels.add("behavior:structure_failure")
        if any(label in issue_labels for label in {"wrong_tool_called", "tool_runtime_error"}):
            flat_labels.add("behavior:tool_execution_failure")
        if "clarification_expected" in issue_labels:
            flat_labels.add("behavior:clarification_failure")

        return {
            "flat": sorted(flat_labels),
            "filter": {
                "scenario_key": scenario_key,
                "scenario_type": str(scenario.get("type", "unknown")),
                "tool_name": tool_name or None,
                "environment_mode": environment_mode,
                "has_environment": has_environment,
                "environment_passed": environment_passed,
                "generated_environment": bool(generated_environment),
                "stage_failures": stage_failure_labels,
                "issue_labels": sorted(issue_labels),
                "scenario_tags": tags,
                "scenario_triggers": triggers,
                "executed_tools": executed_tool_names,
                "executed_tool_details": executed_tools,
                "tool_errors": tool_error_names,
                "issues": issues,
                "kto_candidate_label": _derive_kto_candidate_label(
                    has_environment=has_environment,
                    environment_passed=environment_passed,
                    stage_failures=stage_failure_labels,
                    issue_labels=issue_labels,
                ),
            },
        }

    def _improve_stage(
        self,
        example: Dict,
        stage: str,
        rubrics: List[str],
        max_iterations: int
    ) -> Tuple[Dict, int, bool]:
        """
        Improve a single stage using the improvement engine.

        Args:
            example: Current example
            stage: Stage name (system_prompt, user, thinking, response)
            rubrics: List of rubric keys to apply
            max_iterations: Max improvement iterations

        Returns:
            Tuple of (improved_example, iterations_used, passed)
        """
        if not self.engine or not rubrics:
            return example, 0, True

        try:
            result = self.engine.run(
                example=example,
                rubric_keys=rubrics,
                max_iterations=max_iterations
            )
            return result.improved_example, result.iterations, result.passed
        except Exception as e:
            if self.logger:
                self.logger.error(f"Stage improvement failed: {e}")
            return example, 0, False

    def _log_stage(self, scenario_key: str, stage: str, event: str, extra: Optional[str] = None) -> None:
        """Emit lightweight stage progress logs when a logger is available."""
        if not self.logger:
            return
        message = f"[{scenario_key}] {stage} {event}"
        if extra:
            message = f"{message} ({extra})"
        self.logger.info(message)

    def _call_llm(
        self,
        prompt: str,
        randomize: bool = True,
        trace_label: Optional[str] = None,
        max_tokens: Optional[int] = None,
        llm_clients: Optional[Sequence[Any]] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Call LLM for generation.

        Args:
            prompt: Generation prompt
            randomize: Whether to randomize parameters

        Returns:
            Generated text
        """
        # Some providers occasionally return null/empty content; retry briefly
        # instead of crashing the whole generation job on a transient response.
        last_error = None
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = getattr(self.llm_client, "default_max_tokens", None)

        client_chain = list(llm_clients or [self.llm_client])
        for client_index, client in enumerate(client_chain):
            for attempt in range(1, max(1, int(max_retries or 1)) + 1):
                temperature = random.uniform(0.5, 0.9) if randomize else 0.7
                started_at = time.monotonic()
                if self.logger:
                    self.logger.info(
                        f"LLM chat start [{trace_label or 'unlabeled'}] "
                        f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} temp={temperature:.2f}"
                    )
                try:
                    chat_kwargs = {
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                    }
                    if resolved_max_tokens is not None:
                        chat_kwargs["max_tokens"] = resolved_max_tokens
                    response = client.chat(**chat_kwargs)
                except Exception as exc:  # pragma: no cover - provider-specific failures
                    last_error = exc
                    if self.logger:
                        self.logger.warning(
                            f"LLM chat failed [{trace_label or 'unlabeled'}] "
                            f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                            f"elapsed={time.monotonic() - started_at:.1f}s error={exc}"
                        )
                    continue

                if isinstance(response, str):
                    if response.strip():
                        if self.logger:
                            self.logger.info(
                                f"LLM chat success [{trace_label or 'unlabeled'}] "
                                f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                                f"elapsed={time.monotonic() - started_at:.1f}s chars={len(response)}"
                            )
                        return response
                elif response is not None:
                    response_text = str(response).strip()
                    if response_text:
                        if self.logger:
                            self.logger.info(
                                f"LLM chat success [{trace_label or 'unlabeled'}] "
                                f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                                f"elapsed={time.monotonic() - started_at:.1f}s chars={len(response_text)}"
                            )
                        return response_text

                last_error = ValueError("LLM returned an empty response")
                if self.logger:
                    self.logger.warning(
                        f"LLM chat empty [{trace_label or 'unlabeled'}] "
                        f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                        f"elapsed={time.monotonic() - started_at:.1f}s"
                    )

        if last_error is not None:
            raise last_error
        raise ValueError("LLM returned an empty response")

    def _call_llm_structured(
        self,
        *,
        prompt: str,
        schema: Dict[str, Any],
        randomize: bool = True,
        system_prompt: Optional[str] = None,
        trace_label: Optional[str] = None,
        max_tokens: Optional[int] = None,
        llm_clients: Optional[Sequence[Any]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Call structured output if available, retrying transient empty failures."""
        if not hasattr(self.llm_client, "structured_output"):
            raw = self._call_llm(
                f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                randomize=randomize,
                trace_label=trace_label,
                llm_clients=llm_clients,
                max_retries=max_retries,
            )
            parsed = self._parse_json_object(raw)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Structured generation requested but provider returned non-JSON output")

        last_error = None
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = getattr(self.llm_client, "default_max_tokens", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client_chain = list(llm_clients or [self.llm_client])
        for client_index, client in enumerate(client_chain):
            for attempt in range(1, max(1, int(max_retries or 1)) + 1):
                temperature = random.uniform(0.1, 0.4) if randomize else 0.2
                started_at = time.monotonic()
                if self.logger:
                    self.logger.info(
                        f"LLM structured start [{trace_label or schema.get('name', 'unlabeled')}] "
                        f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} temp={temperature:.2f}"
                    )
                try:
                    structured_kwargs = {
                        "messages": messages,
                        "schema": schema,
                        "temperature": temperature,
                    }
                    if resolved_max_tokens is not None:
                        structured_kwargs["max_tokens"] = resolved_max_tokens
                    payload = client.structured_output(**structured_kwargs)
                except Exception as exc:  # pragma: no cover - provider-specific failures
                    last_error = exc
                    if self.logger:
                        self.logger.warning(
                            f"LLM structured failed [{trace_label or schema.get('name', 'unlabeled')}] "
                            f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                            f"elapsed={time.monotonic() - started_at:.1f}s error={exc}"
                        )
                    continue

                if isinstance(payload, dict) and payload:
                    if self.logger:
                        self.logger.info(
                            f"LLM structured success [{trace_label or schema.get('name', 'unlabeled')}] "
                            f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                            f"elapsed={time.monotonic() - started_at:.1f}s keys={sorted(payload.keys())}"
                        )
                    return payload
                last_error = ValueError("LLM returned an empty structured response")
                if self.logger:
                    self.logger.warning(
                        f"LLM structured empty [{trace_label or schema.get('name', 'unlabeled')}] "
                        f"attempt={attempt} client={getattr(client, 'model_name', 'unknown')} "
                        f"elapsed={time.monotonic() - started_at:.1f}s"
                    )

        if last_error is not None:
            raise last_error
        raise ValueError("LLM returned an empty structured response")

    def _build_user_context(self, example: Dict) -> str:
        """Build context for user generation from current example."""
        # Include system message if present
        conversations = example.get("conversations", [])
        if conversations and conversations[0]["role"] == "system":
            return f"Given this workspace context:\n{conversations[0]['content']}"
        return ""

    def _build_assistant_context(self, example: Dict, scenario: Dict) -> str:
        """Build context for assistant generation from current example."""
        conversations = example.get("conversations", [])
        context_parts = []

        # System context (if present)
        for msg in conversations:
            if msg["role"] == "system":
                context_parts.append(f"Workspace context:\n{msg['content']}")
                break

        # User request (find the last user message)
        for msg in reversed(conversations):
            if msg["role"] == "user":
                context_parts.append(f"User request:\n{msg['content']}")
                break

        return "\n\n".join(context_parts)

    def _build_loop_assistant_context(self, messages: List[Mapping[str, Any]]) -> str:
        """Build transcript-aware assistant context for multi-turn rollouts."""
        parts: List[str] = []
        for message in messages:
            role = str(message.get("role", "")).strip() or "unknown"
            content = message.get("content")
            if isinstance(content, dict):
                content_str = json.dumps(content)
            else:
                content_str = str(content or "")
            if not content_str.strip():
                continue
            parts.append(f"{role.upper()}:\n{content_str}")
        return "\n\n".join(parts)

    def _synthchat_loop_response(
        self,
        *,
        scenario: Dict[str, Any],
        system_context: Dict[str, Any],
        messages: Sequence[Mapping[str, Any]],
        assistant_prompt: str,
        randomize_params: bool,
        scenario_key: str,
        turn_index: int,
        thinking_content: Optional[str],
    ) -> AgenticModelResponse:
        """Generate one assistant turn for a shared agentic episode."""
        assistant_context = self._build_loop_assistant_context(list(messages))
        if thinking_content:
            assistant_context = f"{assistant_context}\n\nYour prior thinking:\n{thinking_content}"

        trace_label = f"{scenario_key}:assistant_turn_{turn_index}"
        assistant_content = self._generate_assistant_response(
            scenario=scenario,
            system_context=system_context,
            assistant_context=assistant_context,
            assistant_prompt=assistant_prompt,
            randomize_params=randomize_params,
            trace_label=trace_label,
        )
        assistant_msg = self._parse_assistant_response(assistant_content, scenario)
        if thinking_content and assistant_msg.get("tool_calls"):
            assistant_msg["content"] = f"<thinking>{thinking_content}</thinking>"
        return AgenticModelResponse(
            message=assistant_msg,
            raw={"message": assistant_msg},
            latency_s=0.0,
        )

    def _stringify_assistant_message(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            content = response.get("content")
            tool_calls = response.get("tool_calls") or []
            parts: List[str] = []
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
            if tool_calls:
                parts.append(f"Tool calls: {tool_calls}")
            return "\n\n".join(parts).strip() or json.dumps(response)
        return str(response)

    def _validate_agentic_synthchat_response(self, message: Any):
        """Relax eval-only generator-format checks for synthetic loop rollouts."""
        result = validate_assistant_response(message, None)
        filtered_issues = []
        for issue in result.issues:
            message_text = str(issue.message)
            if "does not match generator format" in message_text:
                continue
            filtered_issues.append(issue)

        passed = all(str(issue.level).lower() != "error" for issue in filtered_issues)
        return ValidationResult(
            passed=passed,
            issues=[
                issue
                if isinstance(issue, ValidatorIssue)
                else ValidatorIssue(level=getattr(issue, "level", "ERROR"), message=getattr(issue, "message", str(issue)))
                for issue in filtered_issues
            ],
            tool_calls=result.tool_calls,
            context_validation=None,
        )

    def _parse_assistant_response(self, content: str, scenario: Dict) -> Dict:
        """
        Parse assistant response for tool calls and thinking.

        Handles four cases:
        1. Direct JSON: response is already {"content":..., "tool_calls":[...]}
        2. Text-only: content is text, no tool_calls
        3. Tool-only: content is null or empty, tool_calls present
        4. Thinking+tool: content contains <thinking>...</thinking>, tool_calls present

        Returns:
            Assistant message dict with role, content, and optional tool_calls
        """
        # Case 1: Check if response is already in final JSON format
        # This handles scenarios that ask the LLM to output direct JSON
        if content is None:
            content = ""

        content_stripped = content.strip()

        # Strip markdown code fences if present
        if content_stripped.startswith('```'):
            # Remove opening fence (```json or ```)
            first_newline = content_stripped.find('\n')
            if first_newline > 0:
                content_stripped = content_stripped[first_newline + 1:]
            # Remove closing fence
            if content_stripped.rstrip().endswith('```'):
                content_stripped = content_stripped.rstrip()[:-3].strip()

        if content_stripped.startswith('{') and '"tool_calls"' in content_stripped:
            try:
                # First attempt: try parsing as-is
                parsed = json.loads(content_stripped)
            except json.JSONDecodeError:
                # Second attempt: try to fix common JSON issues
                # Replace literal newlines inside strings with \n
                # This regex finds strings and escapes any literal newlines inside them
                fixed = content_stripped
                # Remove literal newlines between JSON tokens (keep only in string values)
                fixed = re.sub(r'\n\s*', ' ', fixed)
                try:
                    parsed = json.loads(fixed)
                except json.JSONDecodeError:
                    parsed = None

            if parsed and "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                    # Valid direct JSON format
                    message = {"role": "assistant"}
                    message["content"] = parsed.get("content")

                    # Normalize tool_calls - ensure arguments is stringified
                    tool_calls = []
                    for tc in parsed["tool_calls"]:
                        normalized_tc = {
                            "id": tc.get("id", f"call_{len(tool_calls)+1:04d}"),
                            "type": tc.get("type", "function"),
                            "function": {}
                        }
                        fn = tc.get("function", {})
                        normalized_tc["function"]["name"] = fn.get("name", "")

                        # Handle arguments - stringify if it's an object
                        args = fn.get("arguments", "{}")
                        if isinstance(args, dict):
                            normalized_tc["function"]["arguments"] = json.dumps(args)
                        else:
                            normalized_tc["function"]["arguments"] = args

                        tool_calls.append(normalized_tc)

                    message["tool_calls"] = tool_calls
                    return message
            if parsed and parsed.get("tool_calls") is None and isinstance(parsed.get("content"), str):
                return {
                    "role": "assistant",
                    "content": parsed.get("content"),
                }

        # Extract thinking block if present
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
        thinking_block = None
        if thinking_match:
            thinking_block = f"<thinking>{thinking_match.group(1)}</thinking>"
            # Remove thinking from content for tool detection
            content_without_thinking = content.replace(thinking_match.group(0), '').strip()
        else:
            content_without_thinking = content

        # Detect tool calls - look for patterns like:
        # "tool_name(args)" or "Use: tool_name" or JSON-like tool definitions
        tool_calls = []
        tool_pattern = r'(\w+Manager_\w+)\s*\((.*?)\)'
        tool_matches = re.finditer(tool_pattern, content_without_thinking)

        call_id_counter = 1
        for match in tool_matches:
            tool_name = match.group(1)
            tool_args = match.group(2).strip()

            # Try to parse arguments as JSON
            try:
                # Clean up args if needed
                if not tool_args.startswith('{'):
                    tool_args = '{' + tool_args + '}'
                # Validate it's JSON-like
                arguments_str = tool_args
            except:
                # Fallback to empty args
                arguments_str = "{}"

            tool_calls.append({
                "id": f"call_{call_id_counter:04d}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments_str
                }
            })
            call_id_counter += 1

        # If scenario specifies a tool but we didn't detect it, create from scenario
        if not tool_calls and scenario.get("type") == "tool":
            tool_name = scenario.get("tool", "")
            if tool_name:
                # Extract any JSON-like content from the response as arguments
                json_match = re.search(r'\{[^}]+\}', content_without_thinking)
                arguments_str = json_match.group(0) if json_match else "{}"

                tool_calls.append({
                    "id": "call_0001",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments_str
                    }
                })

        # Build final message
        message = {"role": "assistant"}

        if tool_calls:
            # Tool call case
            message["tool_calls"] = tool_calls
            # Content is thinking block only, or null if no thinking
            message["content"] = thinking_block if thinking_block else None
        else:
            # Text-only case
            message["content"] = content

        return message

    def _parse_json_object(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON object from model output, tolerating code fences."""
        candidate = (content or "").strip()
        if not candidate:
            return None

        if candidate.startswith("```"):
            first_newline = candidate.find("\n")
            if first_newline != -1:
                candidate = candidate[first_newline + 1:]
            candidate = candidate.rstrip()
            if candidate.endswith("```"):
                candidate = candidate[:-3].rstrip()

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                parsed = json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                return None

        return parsed if isinstance(parsed, dict) else None

    def _normalize_generated_environment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Accept a few config-driven shapes for generated environment payloads."""
        normalized = deepcopy(payload)
        environment = normalized.get("environment")

        if not isinstance(environment, dict):
            if any(key in normalized for key in ("fixture", "assertions", "allowed_tools", "max_steps", "execution")):
                environment = {
                    key: normalized.pop(key)
                    for key in ("fixture", "assertions", "allowed_tools", "max_steps", "execution")
                    if key in normalized
                }
            elif any(key in normalized for key in ("directories", "files", "notes", "folders")):
                environment = {
                    "fixture": {
                        key: normalized.pop(key)
                        for key in ("directories", "files", "notes", "folders")
                        if key in normalized
                    }
                }
            else:
                environment = {}

        if environment and not environment.get("fixture"):
            fixture = {
                key: normalized.pop(key)
                for key in ("directories", "files", "notes", "folders")
                if key in normalized
            }
            if fixture:
                environment["fixture"] = fixture

        normalized["environment"] = environment
        assertions = environment.get("assertions")
        if isinstance(assertions, list):
            environment["assertions"] = [self._normalize_generated_assertion(assertion) for assertion in assertions]
        normalized.setdefault("system_context", {})
        normalized.setdefault("task_context", {})
        return normalized

    def _normalize_generated_assertion(self, assertion: Any) -> Any:
        if not isinstance(assertion, dict):
            return assertion

        normalized = deepcopy(assertion)
        assertion_type = str(normalized.get("type") or "").strip()

        # Generated environments often use more natural field names than the
        # validator contract. Normalize those aliases here so environment seeds
        # stay strict downstream without having to tutor the model in the prompt.
        if assertion_type in {"file_contains", "file_not_contains"}:
            if "text" not in normalized and "content" in normalized:
                normalized["text"] = normalized.pop("content")
        elif assertion_type == "frontmatter_has_key":
            if "field" not in normalized and "key" in normalized:
                normalized["field"] = normalized.pop("key")
        elif assertion_type in {"frontmatter_field_equals", "frontmatter_field_contains"}:
            if "field" not in normalized and "key" in normalized:
                normalized["field"] = normalized.pop("key")
            if assertion_type == "frontmatter_field_equals" and "value" not in normalized and "content" in normalized:
                normalized["value"] = normalized.pop("content")
            if assertion_type == "frontmatter_field_contains" and "text" not in normalized and "content" in normalized:
                normalized["text"] = normalized.pop("content")

        return normalized

    def _render_mocked_workspace_system_prompt(
        self,
        system_context: Dict[str, Any],
        environment_config: Dict[str, Any],
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render production-style mocked workspace prompt from structured config."""
        session_id = str(system_context.get("session_id") or "session_eval_001")
        workspace_id = str(system_context.get("workspace_id") or "default")
        fixture = _merged_fixture_from_config(environment_config)

        available_workspaces = system_context.get("available_workspaces") or []
        available_prompts = system_context.get("available_prompts") or []
        selected_workspace = dict(system_context.get("selected_workspace") or {})
        note_contents = system_context.get("note_contents")
        note_paths = system_context.get("note_content_paths") or []
        extra_sections = system_context.get("extra_sections") or []

        selected_workspace.setdefault("id", workspace_id)
        selected_workspace.setdefault("name", "Current Workspace")

        matched_workspace = None
        for workspace in available_workspaces:
            if isinstance(workspace, dict) and workspace.get("id") == selected_workspace.get("id"):
                matched_workspace = workspace
                break

        context_payload = dict(selected_workspace.get("context") or {})
        context_payload.setdefault("id", selected_workspace.get("id"))
        context_payload.setdefault("name", selected_workspace.get("name"))
        if matched_workspace:
            context_payload.setdefault("description", matched_workspace.get("description"))
            context_payload.setdefault("rootFolder", matched_workspace.get("root_folder", ""))
        context_payload.setdefault("rootFolder", selected_workspace.get("root_folder", ""))

        workspace_structure = selected_workspace.get("workspace_structure") or _workspace_structure_from_fixture(fixture)
        recent_files = selected_workspace.get("recent_files") or []
        key_files = selected_workspace.get("key_files") or []
        workflows = selected_workspace.get("workflows") or []
        preferences = selected_workspace.get("preferences", "")
        sessions = selected_workspace.get("sessions") or []

        selected_workspace_json = json.dumps(
            {
                "context": context_payload,
                "workspaceStructure": workspace_structure,
                "recentFiles": recent_files,
                "keyFiles": key_files,
                "workflows": workflows,
                "preferences": preferences,
                "sessions": sessions,
            },
            indent=2,
        )

        if not note_contents:
            note_contents = _note_entries_from_fixture(fixture, note_paths=note_paths)

        sections = [
            _build_session_context_section(session_id, workspace_id),
            _build_wrapped_section("vault_structure", _vault_structure_text_from_fixture(fixture)),
            _build_wrapped_section("available_workspaces", _render_available_workspaces(available_workspaces)),
            _build_wrapped_section("available_prompts", _render_available_prompts(available_prompts)),
            _build_wrapped_section("available_tools", _render_available_tools(tool_schema)),
            _build_selected_workspace_section(
                selected_workspace.get("name") or context_payload.get("name") or "Current Workspace",
                selected_workspace.get("id") or context_payload.get("id") or workspace_id,
                selected_workspace_json,
            ),
            _build_wrapped_section("note_contents", _render_note_contents(note_contents)),
            _render_extra_sections(extra_sections),
            str(system_context.get("assistant_instructions", "")).strip(),
        ]
        return "\n\n".join(section for section in sections if section)


def _deep_merge_dicts(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(base, dict) and not isinstance(override, dict):
        return override if override is not None else base
    if not isinstance(base, dict):
        return deepcopy(override or {})
    if not isinstance(override, dict):
        return deepcopy(base)

    merged: Dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _task_context_template_vars(task_context: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(task_context, dict) or not task_context:
        return {}

    safe_task_context = _make_json_safe(task_context)
    template_vars: Dict[str, str] = {
        "task_context_json": json.dumps(safe_task_context, indent=2),
    }
    for key, value in safe_task_context.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        template_vars[f"task_{key_str}"] = rendered
    return template_vars


def _user_generation_style_instructions(scenario: Optional[Dict[str, Any]]) -> List[str]:
    task_family = scenario.get("task_family") if isinstance(scenario, dict) else None
    if not isinstance(task_family, dict):
        return []
    style = task_family.get("user_request_style")
    if not isinstance(style, dict):
        return []

    instructions: List[str] = []
    if style.get("vague_human_request"):
        instructions.append(
            "Write like a normal human user, not an operator reading internal paths or system metadata."
        )
    if style.get("require_request_form"):
        instructions.append(
            "Phrase the text as a request or question to the assistant, not as a status update, report, or completed action."
        )
    if style.get("allow_exact_paths") is False:
        instructions.append(
            "Do not mention exact file or folder paths in the user request."
        )
    if style.get("allow_exact_links") is False:
        instructions.append(
            "Do not include literal markdown links or exact linked file paths in the user request."
        )
    if style.get("avoid_exact_source_path") or style.get("avoid_exact_target_path"):
        instructions.append(
            "Avoid exact filesystem paths unless the scenario explicitly requires them."
        )
    if style.get("avoid_exact_source_path"):
        instructions.append(
            "Refer to the source file by a natural title, topic, or fuzzy description rather than its exact current path."
        )
    if style.get("avoid_exact_target_path"):
        instructions.append(
            "Refer to the destination or target location by a human folder description rather than an exact target path."
        )
    reference_mode = str(style.get("reference_mode") or "").strip().lower()
    if reference_mode == "title_only":
        instructions.append(
            "Prefer note titles, project names, or topic names over any internal path-like wording."
        )
    elif reference_mode == "folder_purpose":
        instructions.append(
            "Prefer describing folders by their purpose, such as logs, project notes, or meeting notes, rather than their exact names."
        )
    examples = style.get("examples")
    if isinstance(examples, list):
        cleaned_examples = [str(item).strip() for item in examples if str(item).strip()]
        if cleaned_examples:
            instructions.append("Use the following only as style examples. Do not copy them verbatim.")
            instructions.extend(f"- {example}" for example in cleaned_examples[:3])
    return instructions


def _apply_stage_review_result(
    stage_failures: List[str],
    stage_reviews: Dict[str, Any],
    stage_name: str,
    review: Optional[Dict[str, Any]],
) -> None:
    if review is None:
        return
    stage_reviews[stage_name] = review
    enforce = review.get("enforce", True)
    passed = review.get("passed")
    if passed is False and enforce:
        if stage_name not in stage_failures:
            stage_failures.append(stage_name)
        return
    if passed is True and stage_name in stage_failures:
        stage_failures.remove(stage_name)


def _render_template_object(value: Any, template_vars: Dict[str, str], task_context: Optional[Dict[str, Any]] = None) -> Any:
    if isinstance(value, dict):
        return {
            key: _render_template_object(item, template_vars, task_context)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_render_template_object(item, template_vars, task_context) for item in value]
    if isinstance(value, str):
        exact_match = re.fullmatch(r"\{task_([A-Za-z0-9_]+)\}", value.strip())
        if exact_match and isinstance(task_context, dict):
            raw_value = task_context.get(exact_match.group(1))
            if raw_value is not None:
                return deepcopy(raw_value)
        rendered = value
        for key, replacement in template_vars.items():
            rendered = rendered.replace(f"{{{key}}}", str(replacement))
        return rendered
    return deepcopy(value)


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _merged_fixture_from_config(environment_config: Optional[Dict[str, Any]]):
    fixture_config = {}
    if isinstance(environment_config, dict):
        fixture_config = environment_config.get("fixture") or {}
    from shared.environments.fixture_parser import EnvironmentFixture, merge_environment_fixture

    return merge_environment_fixture(EnvironmentFixture(), fixture_config)


def _workspace_structure_from_fixture(fixture) -> List[Dict[str, Any]]:
    directory_set = {""}
    for directory in fixture.directories:
        cleaned = _clean_path(directory)
        if not cleaned:
            continue
        parts = [part for part in cleaned.split("/") if part]
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else part
            directory_set.add(current)

    for file_path in fixture.files:
        cleaned = _clean_path(file_path)
        parent = Path(cleaned).parent
        current = ""
        for part in parent.parts:
            if part in {"", "."}:
                continue
            current = f"{current}/{part}" if current else part
            directory_set.add(current)

    child_map: Dict[str, set[str]] = {directory: set() for directory in directory_set}
    for directory in directory_set:
        if not directory:
            continue
        parent = str(Path(directory).parent)
        parent_key = "" if parent == "." else parent.replace("\\", "/")
        child_map.setdefault(parent_key, set()).add(f"{Path(directory).name}/")

    for file_path in fixture.files:
        cleaned = _clean_path(file_path)
        parent = str(Path(cleaned).parent)
        parent_key = "" if parent == "." else parent.replace("\\", "/")
        child_map.setdefault(parent_key, set()).add(Path(cleaned).name)

    entries: List[Dict[str, Any]] = []
    for directory in sorted(directory_set):
        entries.append(
            {
                "path": f"{directory}/" if directory else "",
                "type": "folder",
                "children": sorted(child_map.get(directory, set())),
            }
        )
    return entries


def _vault_structure_text_from_fixture(fixture) -> str:
    folders = sorted(
        str(entry.get("path", "")).strip()
        for entry in _workspace_structure_from_fixture(fixture)
        if isinstance(entry, dict) and str(entry.get("path", "")).strip()
    )
    files = sorted(_clean_path(path) for path in fixture.files if _clean_path(path))
    lines = ["Folders:"]
    lines.extend(f" - {path}" for path in folders)
    lines.append("")
    lines.append("Files:")
    lines.extend(f" - {path}" for path in files)
    return "\n".join(lines).strip()


def _render_available_workspaces(workspaces: List[Dict[str, Any]]) -> str:
    if not isinstance(workspaces, list) or not workspaces:
        return ""
    lines: List[str] = []
    for workspace in workspaces:
        if not isinstance(workspace, dict):
            continue
        lines.append(f'- {workspace.get("name", "Workspace")} (id: "{workspace.get("id", "")}")')
        description = workspace.get("description")
        if description:
            lines.append(f"  Description: {description}")
        root_folder = workspace.get("root_folder")
        if root_folder is not None:
            lines.append(f"  Root folder: {root_folder}")
        lines.append("")
    if lines:
        lines.append("Use memoryManager with loadWorkspace mode to get full workspace context.")
    return "\n".join(lines).strip()


def _render_available_prompts(prompts: List[Dict[str, Any]]) -> str:
    if not isinstance(prompts, list) or not prompts:
        return ""
    lines: List[str] = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        prompt_id = prompt.get("id")
        name = prompt.get("name", "Prompt")
        lines.append(f"- {prompt_id} - {name}" if prompt_id else f"- {name}")
        purpose = prompt.get("purpose") or prompt.get("description")
        if purpose:
            lines.append(f"  Purpose: {purpose}")
    return "\n".join(lines).strip()


def _tool_wrapper_name(tool_schema: Optional[Dict[str, Any]]) -> str:
    if not isinstance(tool_schema, dict):
        return "useTools"
    wrapper_cfg = tool_schema.get("tool_format") or {}
    return str(wrapper_cfg.get("wrapper") or "useTools").strip() or "useTools"


def _render_available_tools(tool_schema: Optional[Dict[str, Any]]) -> str:
    if not isinstance(tool_schema, dict):
        return ""

    wrapper_name = _tool_wrapper_name(tool_schema)
    lines: List[str] = [
        f"Use the `{wrapper_name}` wrapper for tool calls.",
        "Required wrapper context fields: sessionId, workspaceId, memory, goal.",
        "",
    ]

    tools = tool_schema.get("tools") or {}
    for agent in sorted(tools.keys()):
        agent_tools = tools.get(agent)
        if not isinstance(agent_tools, list) or not agent_tools:
            continue
        lines.append(f"{agent}:")
        for tool in agent_tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name") or "").strip()
            params = tool.get("params") or {}
            required = ", ".join(str(item) for item in params.get("required") or []) or "-"
            optional = ", ".join(str(item) for item in params.get("optional") or []) or "-"
            if tool_name:
                lines.append(f"- {tool_name}: required [{required}] optional [{optional}]")
        lines.append("")

    return "\n".join(lines).strip()


def _build_selected_workspace_section(name: str, workspace_id: str, payload: str) -> str:
    return "\n".join(
        [
            f'<selected_workspace name="{name}" id="{workspace_id}">',
            "This workspace is currently selected.",
            "",
            payload,
            "</selected_workspace>",
        ]
    )


def _build_session_context_section(session_id: str, workspace_id: str) -> str:
    return "\n".join(
        [
            "<session_context>",
            "IMPORTANT: When using tools, include these values in your tool call parameters:",
            "",
            f'- sessionId: "{session_id}"',
            f'- workspaceId: "{workspace_id}" (current workspace)',
            "",
            'Include these in the "context" parameter of your tool calls.',
            "</session_context>",
        ]
    )


def _build_wrapped_section(tag: str, content: str) -> str:
    clean = str(content or "").strip()
    if not clean:
        return ""
    return f"<{tag}>\n{clean}\n</{tag}>"


def _render_extra_sections(extra_sections: List[Dict[str, Any]]) -> str:
    rendered: List[str] = []
    for section in extra_sections:
        if not isinstance(section, dict):
            continue
        tag = str(section.get("tag", "")).strip()
        content = str(section.get("content", "")).strip()
        if tag and content:
            rendered.append(_build_wrapped_section(tag, content))
    return "\n\n".join(rendered)


def _note_entries_from_fixture(fixture, note_paths: Optional[List[str]] = None) -> List[Dict[str, str]]:
    selected_paths = {str(path).strip() for path in (note_paths or []) if str(path).strip()}
    entries: List[Dict[str, str]] = []
    for path, content in sorted(fixture.files.items()):
        if selected_paths and path not in selected_paths:
            continue
        entries.append({"path": path, "content": content})
    return entries


def _render_note_contents(note_entries: Any) -> str:
    if not note_entries:
        return ""
    lines: List[str] = []
    for entry in note_entries:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", "")).strip()
        content = str(entry.get("content", "")).rstrip()
        if not path:
            continue
        lines.append(f"- {path}")
        if content:
            for line in content.splitlines():
                lines.append(f"  {line}")
        else:
            lines.append("  ")
        lines.append("")
    return "\n".join(lines).strip()


def _scalar_schema() -> Dict[str, Any]:
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
            {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
            },
        ]
    }


def _assertion_schema() -> Dict[str, Any]:
    scalar = _scalar_schema()
    return {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "path_exists"},
                    "path": {"type": "string", "minLength": 1},
                },
                "required": ["type", "path"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "path_not_exists"},
                    "path": {"type": "string", "minLength": 1},
                },
                "required": ["type", "path"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "file_contains"},
                    "path": {"type": "string", "minLength": 1},
                    "text": {"type": "string"},
                },
                "required": ["type", "path", "text"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "file_not_contains"},
                    "path": {"type": "string", "minLength": 1},
                    "text": {"type": "string"},
                },
                "required": ["type", "path", "text"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "dir_contains"},
                    "path": {"type": "string"},
                    "item": {"type": "string", "minLength": 1},
                },
                "required": ["type", "path", "item"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "frontmatter_has_key"},
                    "path": {"type": "string", "minLength": 1},
                    "field": {"type": "string", "minLength": 1},
                },
                "required": ["type", "path", "field"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "frontmatter_field_equals"},
                    "path": {"type": "string", "minLength": 1},
                    "field": {"type": "string", "minLength": 1},
                    "value": scalar,
                },
                "required": ["type", "path", "field", "value"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"const": "frontmatter_field_contains"},
                    "path": {"type": "string", "minLength": 1},
                    "field": {"type": "string", "minLength": 1},
                    "value": scalar,
                },
                "required": ["type", "path", "field", "value"],
            },
        ]
    }


def _build_canonical_environment_generation_prompt(base_prompt: str) -> str:
    """Add a compact in-band contract for canonical environment generation."""
    contract_lines = [
        "Return one valid JSON object only.",
        "Top-level keys allowed: environment, system_context, task_context.",
        "environment may contain: fixture, assertions, allowed_tools, max_steps, loop, execution.",
        "fixture may contain: directories, files, notes, local_path, source.",
        "notes entries may contain: path, frontmatter, body.",
        "task_context should contain the hidden task anchors used to keep the environment, user request, and assertions aligned.",
        "Use only these assertion types:",
        "- path_exists",
        "- path_not_exists",
        "- file_contains",
        "- file_not_contains",
        "- dir_contains",
        "- frontmatter_has_key",
        "- frontmatter_field_equals",
        "- frontmatter_field_contains",
        "Do not add unsupported assertion types or extra top-level keys.",
        "Do not use markdown fences.",
    ]
    contract = "\n".join(contract_lines)
    prompt_text = str(base_prompt or "").strip()
    if not prompt_text:
        return contract
    return f"{contract}\n\nTask:\n{prompt_text}"


def _build_canonical_environment_schema() -> Dict[str, Any]:
    scalar = _scalar_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "environment": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fixture": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "directories": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                            "files": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "notes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "path": {"type": "string", "minLength": 1},
                                        "frontmatter": {
                                            "type": "object",
                                            "additionalProperties": scalar,
                                        },
                                        "body": {"type": "string"},
                                    },
                                    "required": ["path"],
                                },
                            },
                        },
                    },
                    "assertions": {
                        "type": "array",
                        "items": _assertion_schema(),
                    },
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "max_steps": {"type": "integer", "minimum": 1},
                },
                "required": ["fixture", "assertions"],
            },
            "system_context": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "assistant_instructions": {"type": "string"},
                    "available_workspaces": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "root_folder": {"type": "string"},
                            },
                        },
                    },
                    "available_prompts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "purpose": {"type": "string"},
                            },
                        },
                    },
                    "selected_workspace": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "root_folder": {"type": "string"},
                            "recent_files": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "key_files": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "preferences": {"type": "string"},
                        },
                    },
                },
                "additionalProperties": True,
            },
            "task_context": {
                "type": "object",
                "additionalProperties": scalar,
            },
        },
        "required": ["environment"],
    }


def _build_use_tools_response_schema(
    wrapper_name: str = "useTools",
    allowed_tools: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    allowed_tools = [tool for tool in (allowed_tools or []) if isinstance(tool, str) and "_" in tool]
    agent_enum = sorted({tool.split("_", 1)[0] for tool in allowed_tools})
    tool_enum = sorted({tool.split("_", 1)[1] for tool in allowed_tools})

    context_properties: Dict[str, Any] = {
        "sessionId": {"type": "string", "minLength": 1},
        "workspaceId": {"type": "string", "minLength": 1},
        "memory": {"type": "string", "minLength": 1},
        "goal": {"type": "string", "minLength": 1},
    }
    if session_id:
        context_properties["sessionId"] = {"const": session_id}
    if workspace_id:
        context_properties["workspaceId"] = {"const": workspace_id}

    call_properties: Dict[str, Any] = {
        "agent": {"type": "string", "minLength": 1},
        "tool": {"type": "string", "minLength": 1},
        "params": {
            "type": "object",
            "additionalProperties": True,
        },
    }
    if agent_enum:
        call_properties["agent"] = {"enum": agent_enum}
    if tool_enum:
        call_properties["tool"] = {"enum": tool_enum}

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
            "tool_calls": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "maxItems": 0,
                    },
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "type": {"const": "function"},
                                "function": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"const": wrapper_name},
                                        "arguments": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "context": {
                                                    "type": "object",
                                                    "additionalProperties": True,
                                                    "properties": context_properties,
                                                    "required": ["sessionId", "workspaceId", "memory", "goal"],
                                                },
                                                "calls": {
                                                    "type": "array",
                                                    "minItems": 1,
                                                    "items": {
                                                        "type": "object",
                                                        "additionalProperties": False,
                                                        "properties": call_properties,
                                                        "required": ["agent", "tool", "params"],
                                                    },
                                                },
                                                "strategy": {
                                                    "type": "string",
                                                    "enum": ["serial", "parallel"],
                                                },
                                            },
                                            "required": ["context", "calls"],
                                        },
                                    },
                                    "required": ["name", "arguments"],
                                },
                            },
                            "required": ["id", "type", "function"],
                        },
                    },
                ]
            },
        },
        "required": ["content", "tool_calls"],
    }


def _resolve_allowed_tool_names(
    *,
    scenario: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]],
) -> List[str]:
    configured = []
    for key in ("expected_tools", "acceptable_tools"):
        values = scenario.get(key) or []
        if isinstance(values, list):
            configured.extend(
                str(value).strip() for value in values
                if isinstance(value, str) and str(value).strip() and str(value).strip() != "TEXT_ONLY"
            )
    tool_name = str(scenario.get("tool") or "").strip()
    if tool_name:
        configured.append(tool_name)

    configured = sorted(dict.fromkeys(configured))
    if configured:
        return configured

    if not isinstance(tool_schema, dict):
        return []

    names: List[str] = []
    for agent, tools in (tool_schema.get("tools") or {}).items():
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name", "")).strip()
            if tool_name:
                names.append(f"{agent}_{tool_name}")
    return sorted(dict.fromkeys(names))


def _resolve_context_defaults(
    *,
    system_context: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(system_context, dict):
        return None, None

    session_id = system_context.get("session_id")
    workspace_id = system_context.get("workspace_id")
    if not workspace_id and isinstance(system_context.get("selected_workspace"), dict):
        workspace_id = system_context["selected_workspace"].get("id")

    session_id = str(session_id).strip() if isinstance(session_id, str) and str(session_id).strip() else None
    workspace_id = str(workspace_id).strip() if isinstance(workspace_id, str) and str(workspace_id).strip() else None
    return session_id, workspace_id


def _build_use_tools_generation_prompt(
    *,
    base_prompt: str,
    wrapper_name: str,
    allowed_tools: List[str],
) -> str:
    lines = [
        "Return a single JSON object only.",
        "Your job is to either call tools or respond via text.",
        f"If tools are needed, use exactly one tool_calls entry whose function.name is '{wrapper_name}'.",
        "If no tool call is needed, respond with normal text in content and set tool_calls to null or [].",
        "Inside function.arguments.calls, each item must use this exact shape:",
        '{"agent": "AgentName", "tool": "toolName", "params": {...}}',
        "Do not use dotted names like 'contentManager.read' for either agent or tool.",
        "Do not use nested wrappers like params.tool, params.parameters, or assistant as the agent name.",
        "Put the real tool arguments directly inside params.",
        "Use content as null when the response is tool-only.",
        "When the task is already complete, when clarification is needed, or when you are asked for a final confirmation, respond with text instead of calling tools.",
    ]
    if allowed_tools:
        formatted = ", ".join(allowed_tools)
        lines.append(f"Allowed concrete tools for this task: {formatted}.")
    lines.append("")
    lines.append(base_prompt)
    return "\n".join(lines)


def _clean_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").strip("/")


def _slugify_label(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return value.strip("_")


def _classify_environment_issue(message: str) -> List[str]:
    text = str(message or "").strip().lower()
    if not text:
        return []

    labels = set()

    if "expected tool(s) not executed" in text:
        labels.add("missing_expected_tool")
    if "no acceptable tool called" in text:
        labels.add("wrong_tool_called")
    if "front matter" in text or "yaml front matter" in text:
        labels.add("frontmatter_missing")
    if "expected path to exist" in text or "expected path to be absent" in text:
        labels.add("path_state_mismatch")
    if "does not contain expected text" in text or "contains forbidden text" in text:
        labels.add("content_mismatch")
    if "failed reading" in text:
        labels.add("read_failure")
    if "is a directory" in text or "file exists" in text:
        labels.add("path_type_error")
    if "strict schema" in text or "missing required args" in text:
        labels.add("schema_error")
    if "searchmanager_searchcontent" in text or "searchmanager_searchdirectory" in text:
        labels.add("retrieval_missing")
    if "clarification" in text:
        labels.add("clarification_expected")
    if "tool '" in text and "failed:" in text:
        labels.add("tool_runtime_error")

    return sorted(labels)


def _derive_kto_candidate_label(
    has_environment: bool,
    environment_passed: Optional[bool],
    stage_failures: List[str],
    issue_labels: set[str],
) -> Optional[bool]:
    if not has_environment:
        return None
    if environment_passed and not stage_failures:
        return True
    if environment_passed is False:
        noisy_labels = {"schema_error"}
        if issue_labels and issue_labels.issubset(noisy_labels):
            return None
        return False
    return None


def _normalize_target_spec(raw_target: Any) -> Dict[str, int]:
    """Normalize target count config into explicit seed/rollout counts."""
    if isinstance(raw_target, bool):
        raise ValueError("Boolean target specs are not supported")
    if isinstance(raw_target, int):
        if raw_target < 0:
            raise ValueError("Target counts must be non-negative")
        return {"seed_count": raw_target, "rollouts_per_seed": 1}
    if not isinstance(raw_target, dict):
        raise ValueError(f"Unsupported target spec: {raw_target!r}")

    count = raw_target.get("count")
    seed_count = raw_target.get("seed_count", count if count is not None else 1)
    rollouts_per_seed = raw_target.get("rollouts_per_seed", 1)

    seed_count = int(seed_count)
    rollouts_per_seed = int(rollouts_per_seed)
    if seed_count < 0 or rollouts_per_seed < 0:
        raise ValueError("Target specs must use non-negative integers")
    return {
        "seed_count": seed_count,
        "rollouts_per_seed": rollouts_per_seed,
    }


def _extract_shared_seed_spec(targets: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Split special shared-seed config from normal scenario targets."""
    if not isinstance(targets, dict):
        raise ValueError("Targets must be a dictionary")

    cleaned_targets = dict(targets)
    raw_spec = cleaned_targets.pop("_shared_seed", None)
    if raw_spec is None:
        return None, cleaned_targets
    if not isinstance(raw_spec, dict):
        raise ValueError("_shared_seed must be an object")

    scenario_key = str(raw_spec.get("scenario") or raw_spec.get("scenario_key") or "").strip()
    if not scenario_key:
        raise ValueError("_shared_seed requires a 'scenario' key")

    seed_count = int(raw_spec.get("seed_count", 1) or 1)
    if seed_count < 0:
        raise ValueError("_shared_seed.seed_count must be non-negative")

    raw_targets = raw_spec.get("targets") or raw_spec.get("scenarios") or []
    target_keys = [str(item).strip() for item in raw_targets if str(item).strip()]

    return (
        {
            "scenario": scenario_key,
            "seed_count": seed_count,
            "targets": target_keys,
        },
        cleaned_targets,
    )
