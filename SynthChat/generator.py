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
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from shared.llm import create_client
from .utils.yaml_loader import load_yaml
from .engine import ImprovementEngine


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
            enable_stage_validation: Whether to validate each stage (default: True)
            logger: Logger instance (optional)
        """
        self.config_dir = Path(config_dir)
        self.llm_client = llm_client
        self.logger = logger
        self.enable_stage_validation = enable_stage_validation

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
        targets: Dict[str, int],  # {scenario_key: count}
        max_iterations: int = 3,
        randomize_params: bool = True
    ) -> List[GenerationResult]:
        """
        Generate a batch of examples from scenario targets.

        Args:
            targets: Dictionary mapping scenario keys to counts
            max_iterations: Max improvement iterations per stage
            randomize_params: Whether to randomize LLM parameters

        Returns:
            List of GenerationResult objects
        """
        results = []
        total = sum(targets.values())
        current = 0

        for scenario_key, count in targets.items():
            scenario = self.scenario_loader.get_scenario(scenario_key)
            if not scenario:
                if self.logger:
                    self.logger.warning(f"Scenario not found: {scenario_key}")
                continue

            for i in range(count):
                current += 1
                if self.logger:
                    self.logger.info(f"Generating {current}/{total}: {scenario_key}")

                # Generate single example
                result = self._generate_single(
                    scenario_key,
                    scenario,
                    max_iterations,
                    randomize_params
                )
                results.append(result)

        return results

    def _generate_single(
        self,
        scenario_key: str,
        scenario: Dict,
        max_iterations: int,
        randomize_params: bool
    ) -> GenerationResult:
        """
        Generate a single example through stage-by-stage pipeline.

        Pipeline:
            1. Generate system prompt (if enabled) → validate/improve
            2. Generate user request → validate/improve
            3. Generate assistant response → validate/improve

        Args:
            scenario_key: Scenario identifier
            scenario: Scenario configuration
            max_iterations: Max improvement iterations per stage
            randomize_params: Whether to randomize LLM parameters

        Returns:
            GenerationResult with final example and metrics
        """
        prompts = scenario.get("prompts", {})
        example = {"conversations": []}
        stage_failures = []
        total_iterations = 0

        # Stage 1: System prompt (if enabled)
        system_enabled = scenario.get("system", True)
        if system_enabled and isinstance(system_enabled, bool):
            # Generate system prompt
            system_prompt = prompts.get("system", "")
            system_content = self._call_llm(system_prompt, randomize_params)

            # Add to conversations
            example["conversations"].append({
                "role": "system",
                "content": system_content
            })

            # Validate/improve system stage
            if self.enable_stage_validation:
                improved, iterations, passed = self._improve_stage(
                    example,
                    stage="system_prompt",
                    rubrics=["system_prompt_format"],
                    max_iterations=max_iterations
                )
                example = improved
                total_iterations += iterations
                if not passed:
                    stage_failures.append("system_prompt")

        elif isinstance(system_enabled, dict) and "template" in system_enabled:
            # Use fixed template
            template_path = Path(system_enabled["template"])
            if template_path.exists():
                with open(template_path) as f:
                    system_content = f.read()
                example["conversations"].append({
                    "role": "system",
                    "content": system_content
                })

        # Stage 2: User request
        user_prompt = prompts.get("user", "")
        # Pass current example as context for user generation
        user_context = self._build_user_context(example)
        user_content = self._call_llm(f"{user_context}\n\n{user_prompt}", randomize_params)

        example["conversations"].append({
            "role": "user",
            "content": user_content
        })

        # Validate/improve user stage (if rubrics exist)
        if self.enable_stage_validation:
            improved, iterations, passed = self._improve_stage(
                example,
                stage="user",
                rubrics=[],  # User rubrics if any
                max_iterations=max_iterations
            )
            example = improved
            total_iterations += iterations
            if not passed and iterations > 0:  # Only mark failure if we tried
                stage_failures.append("user")

        # Stage 3: Assistant response
        assistant_prompt = prompts.get("assistant", "")
        assistant_context = self._build_assistant_context(example, scenario)
        assistant_content = self._call_llm(
            f"{assistant_context}\n\n{assistant_prompt}",
            randomize_params
        )

        # Parse assistant response for tool calls
        assistant_msg = self._parse_assistant_response(assistant_content, scenario)
        example["conversations"].append(assistant_msg)

        # Validate/improve assistant stage
        if self.enable_stage_validation:
            improved, iterations, passed = self._improve_stage(
                example,
                stage="response",  # or "thinking" if present
                rubrics=["thinking_quality", "tool_alignment", "response_quality"],
                max_iterations=max_iterations
            )
            example = improved
            total_iterations += iterations
            if not passed:
                stage_failures.append("response")

        # Add metadata
        example["metadata"] = {
            "category": scenario_key,
            "type": scenario.get("type", "unknown"),
            "generated_at": datetime.utcnow().isoformat()
        }

        return GenerationResult(
            example=example,
            scenario_key=scenario_key,
            iterations=total_iterations,
            success=len(stage_failures) == 0,
            stage_failures=stage_failures
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

    def _call_llm(self, prompt: str, randomize: bool = True) -> str:
        """
        Call LLM for generation.

        Args:
            prompt: Generation prompt
            randomize: Whether to randomize parameters

        Returns:
            Generated text
        """
        # TODO: Implement LLM call using shared.llm client
        # For now, placeholder
        if hasattr(self.llm_client, 'chat'):
            response = self.llm_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=random.uniform(0.5, 0.9) if randomize else 0.7
            )
            return response.choices[0].message.content
        return ""

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

        # System context
        if conversations and conversations[0]["role"] == "system":
            context_parts.append(f"Workspace context:\n{conversations[0]['content']}")

        # User request
        if len(conversations) > 1:
            context_parts.append(f"User request:\n{conversations[-1]['content']}")

        # Scenario hints
        scenario_type = scenario.get("type", "")
        if scenario_type == "tool":
            tool = scenario.get("tool", "")
            context_parts.append(f"Use tool: {tool}")

        return "\n\n".join(context_parts)

    def _parse_assistant_response(self, content: str, scenario: Dict) -> Dict:
        """
        Parse assistant response for tool calls and thinking.

        Returns:
            Assistant message dict with role, content, and optional tool_calls
        """
        # TODO: Implement proper parsing for:
        # - <thinking>...</thinking> extraction
        # - Tool call detection and formatting
        # For now, simple structure
        return {
            "role": "assistant",
            "content": content
        }
