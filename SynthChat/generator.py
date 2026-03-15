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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from copy import deepcopy

from .utils.yaml_loader import load_yaml
from .utils.docs_loader import DocFile
from .engine import ImprovementEngine

try:
    from shared.environments import EnvironmentValidator
except ImportError:
    EnvironmentValidator = None


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
                result = self.generate_single(
                    scenario_key,
                    scenario,
                    max_iterations,
                    randomize_params,
                    doc_context
                )
                results.append(result)

        return results

    def generate_single(
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
        environment_mode = self._resolve_environment_mode(scenario)
        generated_environment = {}
        if environment_mode in {"generated", "hybrid"}:
            generated_environment = self._generate_environment_spec(
                scenario=scenario,
                render_prompt=render_prompt,
                randomize_params=randomize_params,
            )
        if generated_environment:
            template_vars["environment_json"] = json.dumps(generated_environment, indent=2)

        base_environment_config = scenario.get("environment") if environment_mode in {"provided", "hybrid"} else None
        generated_environment_config = generated_environment.get("environment") if environment_mode in {"generated", "hybrid"} else None
        resolved_environment_config = _deep_merge_dicts(
            base_environment_config,
            generated_environment_config,
        )
        resolved_system_context = _deep_merge_dicts(
            scenario.get("system_context"),
            generated_environment.get("system_context"),
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
            )
            example["conversations"].append({
                "role": "system",
                "content": system_content
            })

        elif assistant_system_prompt:
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
        environment_trace = None
        if self.environment_validator is not None:
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
            "category": scenario_key,
            "type": scenario.get("type", "unknown"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment_mode": environment_mode,
        }
        if generated_environment:
            example["metadata"]["generated_environment"] = generated_environment
        if environment_trace is not None:
            example["metadata"]["environment"] = environment_trace
        if doc_context:
            example["metadata"]["source_doc"] = doc_context.path
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
    ) -> Dict[str, Any]:
        """Generate structured environment/system-context data before other stages."""
        prompts = scenario.get("prompts", {})
        generation_cfg = scenario.get("environment_generation") or {}
        environment_prompt = generation_cfg.get("prompt") or prompts.get("environment")
        if not environment_prompt:
            return {}

        environment_system = generation_cfg.get("system") or prompts.get("environment_system")
        prompt_parts = []
        if environment_system:
            prompt_parts.append(render_prompt(str(environment_system)))
        prompt_parts.append(render_prompt(str(environment_prompt)))

        raw = self._call_llm("\n\n".join(part for part in prompt_parts if part), randomize_params)
        parsed = self._parse_json_object(raw)
        if not isinstance(parsed, dict):
            return {}
        return self._normalize_generated_environment(parsed)

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
        if any(stage in stage_failure_labels for stage in ("system_prompt", "user")):
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
        normalized.setdefault("system_context", {})
        return normalized

    def _render_mocked_workspace_system_prompt(
        self,
        system_context: Dict[str, Any],
        environment_config: Dict[str, Any],
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
