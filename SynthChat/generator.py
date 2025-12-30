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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .utils.yaml_loader import load_yaml
from .utils.docs_loader import DocFile
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
        randomize_params: bool = True,
        doc_context: Optional[DocFile] = None
    ) -> List[GenerationResult]:
        """
        Generate a batch of examples from scenario targets.

        Args:
            targets: Dictionary mapping scenario keys to counts
            max_iterations: Max improvement iterations per stage
            randomize_params: Whether to randomize LLM parameters
            doc_context: Optional document context for template variables
                        Makes {doc_content} and {doc_path} available in prompts

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
                    randomize_params,
                    doc_context
                )
                results.append(result)

        return results

    def _generate_single(
        self,
        scenario_key: str,
        scenario: Dict,
        max_iterations: int,
        randomize_params: bool,
        doc_context: Optional[DocFile] = None
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

        # Stage 1: Assistant's system prompt
        # Priority: assistant_system > system (legacy) > system: true (generate)
        assistant_system_prompt = prompts.get("assistant_system")
        system_enabled = scenario.get("system", True)

        if assistant_system_prompt:
            # New style: assistant_system in prompts (static template with vars)
            system_content = render_prompt(assistant_system_prompt)
            example["conversations"].append({
                "role": "system",
                "content": system_content
            })

        elif system_enabled and isinstance(system_enabled, bool):
            # Legacy: generate system prompt from "system" prompt
            system_prompt = render_prompt(prompts.get("system", ""))
            if system_prompt:
                system_content = self._call_llm(system_prompt, randomize_params)

                example["conversations"].append({
                    "role": "system",
                    "content": system_content
                })

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

        user_context = "\n\n".join(user_context_parts)
        user_content = self._call_llm(f"{user_context}\n\n{user_prompt}", randomize_params)

        example["conversations"].append({
            "role": "user",
            "content": user_content
        })

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

        # Stage 3: Thinking (if rubrics.thinking specified)
        thinking_rubrics = scenario_rubrics.get("thinking", [])
        thinking_content = None

        if thinking_rubrics:
            # Generate thinking as separate stage
            thinking_prompt = render_prompt(prompts.get("thinking", ""))
            if thinking_prompt:
                thinking_context = self._build_assistant_context(example, scenario)
                thinking_content = self._call_llm(
                    f"{thinking_context}\n\n{thinking_prompt}",
                    randomize_params
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
        assistant_context = self._build_assistant_context(example, scenario)

        # Include thinking in context if it was generated
        if thinking_content:
            assistant_context = f"{assistant_context}\n\nYour thinking:\n{thinking_content}"

        assistant_content = self._call_llm(
            f"{assistant_context}\n\n{assistant_prompt}",
            randomize_params
        )

        # Parse assistant response for tool calls
        assistant_msg = self._parse_assistant_response(assistant_content, scenario)

        # If thinking was generated separately, prepend it to content
        if thinking_content and assistant_msg.get("tool_calls"):
            # For tool calls, thinking goes in content field
            assistant_msg["content"] = f"<thinking>{thinking_content}</thinking>"

        example["conversations"].append(assistant_msg)

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
        example["metadata"] = {
            "category": scenario_key,
            "type": scenario.get("type", "unknown"),
            "generated_at": datetime.utcnow().isoformat()
        }
        if doc_context:
            example["metadata"]["source_doc"] = doc_context.path

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
        # shared.llm.chat() returns str directly
        temperature = random.uniform(0.5, 0.9) if randomize else 0.7
        response = self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048
        )
        return response

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
