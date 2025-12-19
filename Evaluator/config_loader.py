"""Configuration loader for YAML-based evaluation scenarios.

Loads test scenarios from YAML files and converts them to PromptCase objects
that can be used with the existing evaluation runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .prompt_sets import PromptCase


@dataclass
class ScenarioConfig:
    """Loaded scenario configuration."""
    name: str
    description: str
    defaults: Dict[str, Any]
    tests: List[Dict[str, Any]]


@dataclass
class EvalRunConfig:
    """Configuration for an evaluation run."""
    name: str
    description: str
    scenarios: List[str]
    tag_filter: List[str] = field(default_factory=list)
    exclude_tags: List[str] = field(default_factory=list)
    model_backend: str = "lmstudio"
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: Optional[int] = None
    pass_threshold: float = 0.8


class ConfigLoader:
    """Loads YAML configurations and converts to evaluation objects."""

    def __init__(self, config_dir: Union[str, Path]):
        """Initialize with config directory path.

        Args:
            config_dir: Path to config directory (e.g., Evaluator/config)
        """
        self.config_dir = Path(config_dir)
        self._tool_schema: Optional[Dict] = None
        self._templates: Optional[Dict] = None

    @property
    def tool_schema(self) -> Dict[str, Any]:
        """Lazy load tool schema."""
        if self._tool_schema is None:
            self._tool_schema = self._load_yaml("tool_schema.yaml")
        return self._tool_schema

    @property
    def templates(self) -> Dict[str, Any]:
        """Lazy load system prompt templates."""
        if self._templates is None:
            self._templates = self._load_yaml("templates/system_prompts.yaml")
        return self._templates

    def _load_yaml(self, relative_path: str) -> Dict[str, Any]:
        """Load a YAML file relative to config directory."""
        path = self.config_dir / relative_path
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_eval_run(self, preset: Optional[str] = None) -> EvalRunConfig:
        """Load evaluation run configuration.

        Args:
            preset: Optional preset name (e.g., 'quick', 'full')

        Returns:
            EvalRunConfig with run settings
        """
        config = self._load_yaml("eval_run.yaml")
        run = config.get("run", {})

        # Apply preset if specified
        if preset:
            presets = config.get("presets", {})
            if preset in presets:
                preset_config = presets[preset]
                # Merge preset into run config
                for key, value in preset_config.items():
                    if key != "description":
                        if isinstance(value, dict) and key in run:
                            run[key] = {**run.get(key, {}), **value}
                        else:
                            run[key] = value

        model_config = run.get("model", {})
        inference = model_config.get("inference", {})

        return EvalRunConfig(
            name=run.get("name", "Evaluation"),
            description=run.get("description", ""),
            scenarios=run.get("scenarios", []),
            tag_filter=run.get("tag_filter", []),
            exclude_tags=run.get("exclude_tags", []),
            model_backend=model_config.get("backend", "lmstudio"),
            model_name=model_config.get("name", ""),
            temperature=inference.get("temperature", 0.7),
            max_tokens=inference.get("max_tokens", 2048),
            seed=inference.get("seed"),
            pass_threshold=run.get("scoring", {}).get("pass_threshold", 0.8),
        )

    def load_scenario(self, scenario_path: str) -> ScenarioConfig:
        """Load a scenario file.

        Args:
            scenario_path: Path relative to config/scenarios/

        Returns:
            ScenarioConfig with test definitions
        """
        full_path = self.config_dir / "scenarios" / scenario_path
        if not full_path.exists():
            # Try without scenarios/ prefix
            full_path = self.config_dir / scenario_path

        with open(full_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return ScenarioConfig(
            name=data.get("name", scenario_path),
            description=data.get("description", ""),
            defaults=data.get("defaults", {}),
            tests=data.get("tests", []),
        )

    def load_all_scenarios(
        self,
        scenario_paths: List[str],
        tag_filter: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
    ) -> List[PromptCase]:
        """Load multiple scenarios and convert to PromptCase objects.

        Args:
            scenario_paths: List of scenario file paths
            tag_filter: Only include tests with these tags (OR logic)
            exclude_tags: Exclude tests with these tags

        Returns:
            List of PromptCase objects ready for evaluation
        """
        cases: List[PromptCase] = []

        for scenario_path in scenario_paths:
            scenario = self.load_scenario(scenario_path)
            scenario_cases = self._scenario_to_cases(
                scenario, tag_filter, exclude_tags
            )
            cases.extend(scenario_cases)

        return cases

    def _scenario_to_cases(
        self,
        scenario: ScenarioConfig,
        tag_filter: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
    ) -> List[PromptCase]:
        """Convert scenario tests to PromptCase objects.

        Args:
            scenario: Loaded scenario config
            tag_filter: Only include tests with these tags
            exclude_tags: Exclude tests with these tags

        Returns:
            List of PromptCase objects
        """
        cases: List[PromptCase] = []
        defaults = scenario.defaults

        for test in scenario.tests:
            # Apply tag filters
            tags = test.get("tags", [])

            if exclude_tags and any(t in exclude_tags for t in tags):
                continue

            if tag_filter and not any(t in tag_filter for t in tags):
                continue

            case = self._test_to_case(test, defaults)
            cases.append(case)

        return cases

    def _test_to_case(
        self,
        test: Dict[str, Any],
        defaults: Dict[str, Any],
    ) -> PromptCase:
        """Convert a single test definition to a PromptCase.

        Args:
            test: Test definition from YAML
            defaults: Scenario defaults

        Returns:
            PromptCase object
        """
        expect = test.get("expect", {})

        # Build expected_tools from expect.tool
        expected_tools: List[str] = []
        if "tool" in expect:
            expected_tools = [expect["tool"]]

        # Build acceptable_tools from expect.acceptable
        acceptable_tools: List[str] = []
        if "acceptable" in expect:
            for option in expect["acceptable"]:
                if "tool" in option:
                    acceptable_tools.append(option["tool"])
                if "pseudo_tool" in option and option["pseudo_tool"] == "TEXT_ONLY":
                    acceptable_tools.append("TEXT_ONLY")

        # Also check first_tool_any_of for acceptable tools
        if "first_tool_any_of" in expect:
            acceptable_tools.extend(expect["first_tool_any_of"])

        # Build metadata
        metadata: Dict[str, Any] = {}

        # Add system prompt from template
        template_name = test.get("system_prompt_template") or defaults.get("system_prompt_template")
        if template_name:
            system_prompt = self._render_template(template_name, test)
            metadata["system"] = system_prompt

        # Add behavior expectations
        behaviors = test.get("behaviors", defaults.get("behaviors", []))
        if behaviors:
            metadata["behavior_expectations"] = behaviors

        # Add response type expectation
        response_type = expect.get("response_type") or defaults.get("response_type")
        if response_type:
            metadata["expected_response_type"] = response_type

        # Add params expectations for validation
        if "params_include" in expect:
            metadata["expected_params"] = expect["params_include"]

        # Add sequence expectations
        if "first_tool" in expect:
            metadata["first_tool"] = expect["first_tool"]
        if "first_tool_any_of" in expect:
            metadata["first_tool_any_of"] = expect["first_tool_any_of"]
        if "not_first" in expect:
            metadata["not_first"] = expect["not_first"]

        return PromptCase(
            case_id=test.get("id", ""),
            question=test.get("question", ""),
            tags=test.get("tags", []),
            expected_tools=expected_tools,
            acceptable_tools=list(set(acceptable_tools)),  # Dedupe
            metadata=metadata,
        )

    def _render_template(
        self,
        template_name: str,
        test: Dict[str, Any],
    ) -> str:
        """Render a system prompt template with variables.

        Args:
            template_name: Name of template from templates/system_prompts.yaml
            test: Test definition (may contain override values)

        Returns:
            Rendered system prompt string
        """
        templates = self.templates.get("templates", {})
        template_def = templates.get(template_name, {})

        if not template_def:
            return ""

        content = template_def.get("content", "")
        defaults = template_def.get("defaults", {})

        # Merge defaults with test-specific values
        variables = {**defaults}
        if "template_vars" in test:
            variables.update(test["template_vars"])

        # Handle auto-generated values
        if variables.get("available_tools") == "{{auto_from_tool_schema}}":
            variables["available_tools"] = self._generate_tools_list()

        # Substitute variables
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            if placeholder in content:
                content = content.replace(placeholder, str(value))

        return content

    def _generate_tools_list(self) -> str:
        """Generate tools list from tool_schema.yaml."""
        lines = ["Available tools:"]

        for agent, tools in self.tool_schema.get("tools", {}).items():
            lines.append(f"\n{agent}:")
            for tool in tools:
                name = tool.get("name", "")
                desc = tool.get("description", "")
                params = tool.get("params", {})
                required = params.get("required", [])

                param_str = ", ".join(required) if required else "none"
                lines.append(f"  - {name}: {desc} (params: {param_str})")

        return "\n".join(lines)

    def get_all_tags(self, scenario_paths: List[str]) -> List[str]:
        """Get all unique tags from scenarios.

        Args:
            scenario_paths: List of scenario file paths

        Returns:
            Sorted list of unique tags
        """
        tags: set = set()

        for scenario_path in scenario_paths:
            scenario = self.load_scenario(scenario_path)
            for test in scenario.tests:
                tags.update(test.get("tags", []))

        return sorted(tags)


def load_yaml_scenarios(
    config_dir: Union[str, Path],
    scenario_files: Optional[List[str]] = None,
    preset: Optional[str] = None,
    tag_filter: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
) -> List[PromptCase]:
    """Convenience function to load YAML scenarios.

    Args:
        config_dir: Path to config directory
        scenario_files: Specific scenario files to load (overrides eval_run.yaml)
        preset: Preset name from eval_run.yaml
        tag_filter: Only include tests with these tags
        exclude_tags: Exclude tests with these tags

    Returns:
        List of PromptCase objects
    """
    loader = ConfigLoader(config_dir)

    # Get scenario list from eval_run or use provided
    if scenario_files:
        scenarios = scenario_files
    else:
        run_config = loader.load_eval_run(preset)
        scenarios = run_config.scenarios
        # Apply filters from run config if not overridden
        if not tag_filter:
            tag_filter = run_config.tag_filter or None
        if not exclude_tags:
            exclude_tags = run_config.exclude_tags or None

    return loader.load_all_scenarios(scenarios, tag_filter, exclude_tags)
