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
from .task_derivation import derive_task_spec
from .template_utils import (
    _deep_merge_dicts,
    _make_json_safe,
    _render_template_object,
    _task_context_template_vars,
    _user_generation_style_instructions,
    _clean_path,
)
from .targets import (
    _normalize_target_spec,
    _extract_shared_seed_spec,
    _apply_stage_review_result,
)
from .workspace.renderer import render_workspace_prompt
from .workspace.sections import _tool_wrapper_name
from .schemas.environment_schema import (
    _build_canonical_environment_schema,
    _build_canonical_environment_generation_prompt,
)
from .schemas.tool_response_schema import (
    build_tool_response_schema,
    build_tool_generation_prompt,
    _resolve_allowed_tool_names,
    _resolve_context_defaults,
    resolve_wrapper_name,
)
from .labeling import (
    build_metadata_labels,
    _slugify_label,
    _classify_environment_issue,
    _derive_kto_candidate_label,
)
from .config.format_resolver import (
    load_tool_call_formats,
    load_workspace_formats,
    load_label_mappings,
    resolve_tool_call_format,
    resolve_workspace_format,
)
from .parsing import (
    stringify_assistant_message,
    parse_assistant_response,
    parse_json_object,
    normalize_generated_environment,
    _normalize_generated_assertion,
)
from .llm.client_pool import LLMClientPool
from .llm.caller import call_llm, call_llm_structured
from .review import (
    run_stage_review,
    build_environment_generation_review_payload,
    run_configured_stage_judge,
    build_stage_judge_template_vars,
)
from .agentic.episode import (
    generate_agentic_episode,
    build_turn_judge,
    build_turn_judge_template_vars,
    synthchat_loop_response,
    validate_agentic_synthchat_response,
)
from shared.llm.factory import create_client  # noqa: F401 — re-exported for test monkeypatching

try:
    from shared.environments import EnvironmentValidator
except ImportError:
    EnvironmentValidator = None

try:
    from shared.agentic_loop import run_environment_episode
    from Evaluator.schema_validator import validate_assistant_response
except ImportError:
    run_environment_episode = None
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
        self.client_pool = LLMClientPool(llm_client, client_factory=create_client)
        self._llm_client_cache = self.client_pool._cache

        # Load scenario configurations
        self.scenario_loader = ScenarioLoader(scenarios_dir)

        # Load config-driven format registries (tool calls, workspace, labels)
        self._tool_call_formats = load_tool_call_formats()
        self._workspace_formats = load_workspace_formats()
        self._label_mappings = load_label_mappings()

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
            ws_format = resolve_workspace_format(scenario, self._workspace_formats)
            tcf = resolve_tool_call_format(scenario, self._tool_call_formats) if ws_format else None
            tool_schema = self.environment_validator.tool_schema if self.environment_validator else None
            # Apply tool_schema wrapper override to resolved tool_call_format
            if tcf and isinstance(tool_schema, dict):
                ts_wrapper = (tool_schema.get("tool_format") or {}).get("wrapper")
                if ts_wrapper and str(ts_wrapper).strip():
                    tcf = dict(tcf)
                    tcf["wrapper_name"] = str(ts_wrapper).strip()
            system_content = self._render_mocked_workspace_system_prompt(
                system_context=resolved_system_context or {},
                environment_config=resolved_environment_config or {},
                tool_schema=tool_schema,
                format_config=ws_format,
                tool_call_format=tcf,
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
        return run_stage_review(
            stage_name=stage_name,
            stage_config=stage_config,
            scenario_key=scenario_key,
            scenario=scenario,
            task_context=task_context,
            payload=payload,
            llm_client=self.llm_client,
            get_stage_llm_clients=self._get_stage_llm_clients,
            logger=self.logger,
        )

    def _build_environment_generation_review_payload(
        self,
        *,
        generated_environment: Dict[str, Any],
        seed_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return build_environment_generation_review_payload(
            generated_environment=generated_environment,
            seed_id=seed_id,
        )

    def _get_stage_llm_clients(self, stage_config: Optional[Dict[str, Any]]) -> List[Any]:
        return self.client_pool.get_stage_clients(stage_config)

    def _normalize_stage_llm_spec(self, value: Any) -> Optional[Dict[str, Any]]:
        return LLMClientPool.normalize_stage_spec(value)

    def _get_or_create_llm_client(self, spec: Dict[str, Any]):
        return self.client_pool.get_or_create(spec)

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
        return run_configured_stage_judge(
            stage_name=stage_name,
            judge_config=judge_config,
            scenario_key=scenario_key,
            scenario=scenario,
            task_context=task_context,
            payload=payload,
            llm_client=self.llm_client,
            get_stage_llm_clients=self._get_stage_llm_clients,
        )

    def _build_stage_judge_template_vars(
        self,
        *,
        stage_name: str,
        scenario_key: str,
        scenario: Dict[str, Any],
        task_context: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, str]:
        return build_stage_judge_template_vars(
            stage_name=stage_name,
            scenario_key=scenario_key,
            scenario=scenario,
            task_context=task_context,
            payload=payload,
        )

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
        return generate_agentic_episode(
            scenario_key=scenario_key,
            scenario=scenario,
            example=example,
            assistant_prompt=assistant_prompt,
            randomize_params=randomize_params,
            resolved_system_context=resolved_system_context,
            resolved_environment_config=resolved_environment_config,
            resolved_task_context=resolved_task_context,
            hard_requirements=hard_requirements,
            quality_rubric=quality_rubric,
            thinking_content=thinking_content,
            stage_failures=stage_failures,
            environment_validator=self.environment_validator,
            llm_client=self.llm_client,
            get_stage_llm_clients=self._get_stage_llm_clients,
            log_stage=self._log_stage,
            build_loop_assistant_context=self._build_loop_assistant_context,
            generate_assistant_response=self._generate_assistant_response,
            parse_response=self._parse_assistant_response,
            stringify_response=self._stringify_assistant_message,
            logger=self.logger,
        )

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
        return build_turn_judge(
            scenario_key=scenario_key,
            scenario=scenario,
            assistant_prompt=assistant_prompt,
            system_context=system_context,
            task_context=task_context,
            hard_requirements=hard_requirements,
            quality_rubric=quality_rubric,
            judge_config=judge_config,
            llm_client=self.llm_client,
            get_stage_llm_clients=self._get_stage_llm_clients,
            logger=self.logger,
        )

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
        return build_turn_judge_template_vars(
            scenario_key=scenario_key,
            scenario=scenario,
            assistant_prompt=assistant_prompt,
            system_context=system_context,
            task_context=task_context,
            hard_requirements=hard_requirements,
            quality_rubric=quality_rubric,
            turn_payload=turn_payload,
        )

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
            tool_schema = self.environment_validator.tool_schema if self.environment_validator else None
            tool_call_fmt = resolve_tool_call_format(scenario, self._tool_call_formats)
            # Check tool_schema for wrapper override (e.g. tool_schema.tool_format.wrapper)
            # This allows the environment validator's tool schema to set the wrapper name
            wrapper_name = resolve_wrapper_name(tool_call_fmt, tool_schema)
            if isinstance(tool_schema, dict):
                ts_wrapper = (tool_schema.get("tool_format") or {}).get("wrapper")
                if ts_wrapper and str(ts_wrapper).strip():
                    wrapper_name = str(ts_wrapper).strip()
            allowed_tools = _resolve_allowed_tool_names(
                scenario=scenario,
                tool_schema=tool_schema,
            )
            session_id, workspace_id = _resolve_context_defaults(
                system_context=system_context,
            )
            # Apply wrapper_name override to format config if needed
            fmt = tool_call_fmt
            if wrapper_name != fmt.get("wrapper_name"):
                fmt = dict(fmt)
                fmt["wrapper_name"] = wrapper_name

            prompt = build_tool_generation_prompt(
                format_config=fmt,
                base_prompt=prompt,
                allowed_tools=allowed_tools,
            )
            context_overrides = {}
            if session_id:
                context_overrides["sessionId"] = session_id
            if workspace_id:
                context_overrides["workspaceId"] = workspace_id

            payload = self._call_llm_structured(
                prompt=prompt,
                schema=build_tool_response_schema(
                    format_config=fmt,
                    allowed_tools=allowed_tools,
                    context_overrides=context_overrides,
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
        return build_metadata_labels(
            scenario_key=scenario_key,
            scenario=scenario,
            environment_mode=environment_mode,
            stage_failures=stage_failures,
            environment_trace=environment_trace,
            generated_environment=generated_environment,
            label_mappings=self._label_mappings,
        )

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
        """Call LLM for generation with retry and client-chain fallback."""
        return call_llm(
            prompt=prompt,
            default_client=self.llm_client,
            logger=self.logger,
            randomize=randomize,
            trace_label=trace_label,
            max_tokens=max_tokens,
            llm_clients=llm_clients,
            max_retries=max_retries,
        )

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
        return call_llm_structured(
            prompt=prompt,
            schema=schema,
            default_client=self.llm_client,
            logger=self.logger,
            randomize=randomize,
            system_prompt=system_prompt,
            trace_label=trace_label,
            max_tokens=max_tokens,
            llm_clients=llm_clients,
            max_retries=max_retries,
        )

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
    ):
        """Generate one assistant turn for a shared agentic episode."""
        return synthchat_loop_response(
            scenario=scenario,
            system_context=system_context,
            messages=messages,
            assistant_prompt=assistant_prompt,
            randomize_params=randomize_params,
            scenario_key=scenario_key,
            turn_index=turn_index,
            thinking_content=thinking_content,
            build_loop_assistant_context=self._build_loop_assistant_context,
            generate_assistant_response=self._generate_assistant_response,
            parse_response=self._parse_assistant_response,
        )

    def _stringify_assistant_message(self, response: Any) -> str:
        return stringify_assistant_message(response)

    def _validate_agentic_synthchat_response(self, message: Any):
        """Relax eval-only generator-format checks for synthetic loop rollouts."""
        return validate_agentic_synthchat_response(message)

    def _parse_assistant_response(self, content: str, scenario: Dict) -> Dict:
        """Parse assistant response for tool calls and thinking."""
        return parse_assistant_response(content, scenario)

    def _parse_json_object(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON object from model output, tolerating code fences."""
        return parse_json_object(content)

    def _normalize_generated_environment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Accept a few config-driven shapes for generated environment payloads."""
        return normalize_generated_environment(payload)

    def _normalize_generated_assertion(self, assertion: Any) -> Any:
        """Normalize field name aliases in a single assertion dict."""
        return _normalize_generated_assertion(assertion)

    def _render_mocked_workspace_system_prompt(
        self,
        system_context: Dict[str, Any],
        environment_config: Dict[str, Any],
        tool_schema: Optional[Dict[str, Any]],
        format_config: Dict[str, Any],
        tool_call_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render production-style mocked workspace prompt from structured config."""
        return render_workspace_prompt(
            system_context, environment_config, tool_schema,
            format_config=format_config, tool_call_format=tool_call_format,
        )


