"""Configuration loader for YAML-based evaluation scenarios.

Loads test scenarios from YAML files and converts them to PromptCase objects
that can be used with the existing evaluation runner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .prompt_sets import PromptCase
from shared.environments.fixture_parser import EnvironmentFixture, merge_environment_fixture


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
    parallel: bool = False
    max_workers: int = 4


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
            parallel=bool(run.get("execution", {}).get("parallel", False)),
            max_workers=int(run.get("execution", {}).get("max_workers", 4) or 4),
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

        Supports the assertion-driven YAML format. Correctness is configured
        through the test's ``correct`` block and evaluated generically by the
        runner.

        Args:
            test: Test definition from YAML
            defaults: Scenario defaults

        Returns:
            PromptCase object
        """
        # Build metadata
        metadata: Dict[str, Any] = {}
        if isinstance(test.get("correct"), dict):
            metadata["correct"] = test["correct"]
        if isinstance(test.get("messages"), list):
            metadata["messages"] = test["messages"]
        metadata["config_dir"] = str(self.config_dir)

        # Add system prompt - check direct field first, then template
        if test.get("system"):
            # Direct system prompt from migrated JSON
            metadata["system"] = test["system"]
        else:
            # Try template
            template_name = (
                test.get("system_template")
                or test.get("system_prompt_template")
                or defaults.get("system_template")
                or defaults.get("system_prompt_template")
            )
            if template_name:
                system_prompt = self._render_template(template_name, test, defaults)
                metadata["system"] = system_prompt

        # Add expected context for ID validation
        if test.get("expected_context"):
            metadata["expected_context"] = test["expected_context"]
        else:
            system_context = _deep_merge_dicts(defaults.get("system_context"), test.get("system_context"))
            inferred_context = _expected_context_from_system_context(system_context)
            if inferred_context:
                metadata["expected_context"] = inferred_context

        # Optional environment execution config
        # Allows runtime-backed validation in evaluator when enabled via CLI.
        environment_cfg = _deep_merge_dicts(defaults.get("environment"), test.get("environment"))
        if environment_cfg:
            metadata["environment"] = environment_cfg

        scoring_cfg = _deep_merge_dicts(defaults.get("scoring"), test.get("scoring"))
        if scoring_cfg:
            metadata["scoring"] = scoring_cfg

        return PromptCase(
            case_id=test.get("id", ""),
            question=test.get("question", _question_from_messages(test.get("messages", []))),
            tags=test.get("tags", []),
            metadata=metadata,
        )

    def _render_template(
        self,
        template_name: str,
        test: Dict[str, Any],
        scenario_defaults: Optional[Dict[str, Any]] = None,
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
        scenario_defaults = scenario_defaults or {}

        # Merge defaults with test-specific values
        variables = {**defaults}
        for key in ("template_vars", "system_vars"):
            if isinstance(scenario_defaults.get(key), dict):
                variables.update(scenario_defaults[key])
            if isinstance(test.get(key), dict):
                variables.update(test[key])

        system_context = _deep_merge_dicts(scenario_defaults.get("system_context"), test.get("system_context"))
        if system_context:
            variables.update(self._build_system_context_vars(system_context, test, scenario_defaults))

        # Handle auto-generated values
        if variables.get("available_tools") == "{{auto_from_tool_schema}}":
            variables["available_tools"] = self._generate_tools_list()

        # Substitute variables
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            if placeholder in content:
                content = content.replace(placeholder, str(value))

        return content

    def _build_system_context_vars(
        self,
        system_context: Dict[str, Any],
        test: Dict[str, Any],
        scenario_defaults: Dict[str, Any],
    ) -> Dict[str, str]:
        session_id = str(system_context.get("session_id") or "session_eval_001")
        workspace_id = str(system_context.get("workspace_id") or "default")
        environment_cfg = _deep_merge_dicts(scenario_defaults.get("environment"), test.get("environment"))
        fixture = _merged_fixture_from_config(environment_cfg)

        available_workspaces = system_context.get("available_workspaces") or []
        available_prompts = system_context.get("available_prompts") or []
        selected_workspace = dict(system_context.get("selected_workspace") or {})
        note_contents = system_context.get("note_contents")
        note_paths = system_context.get("note_content_paths") or []
        extra_sections = system_context.get("extra_sections") or []

        selected_workspace.setdefault("id", workspace_id)
        selected_workspace.setdefault("name", selected_workspace.get("name") or "Current Workspace")

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

        return {
            "session_id": session_id,
            "workspace_id": workspace_id,
            "session_context_section": _build_session_context_section(session_id, workspace_id),
            "vault_structure_section": _build_wrapped_section(
                "vault_structure",
                _vault_structure_text_from_fixture(fixture),
            ),
            "available_workspaces_section": _build_wrapped_section(
                "available_workspaces",
                _render_available_workspaces(available_workspaces),
            ),
            "available_prompts_section": _build_wrapped_section(
                "available_prompts",
                _render_available_prompts(available_prompts),
            ),
            "selected_workspace_section": _build_selected_workspace_section(
                selected_workspace.get("name") or context_payload.get("name") or "Current Workspace",
                selected_workspace.get("id") or context_payload.get("id") or workspace_id,
                selected_workspace_json,
            ),
            "note_contents_section": _build_wrapped_section(
                "note_contents",
                _render_note_contents(note_contents),
            ),
            "extra_sections": _render_extra_sections(extra_sections),
            "assistant_instructions": str(system_context.get("assistant_instructions", "")).strip(),
        }

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


def _deep_merge_dicts(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(base, dict) and not isinstance(override, dict):
        return override if override is not None else base
    if not isinstance(base, dict):
        return dict(override or {})
    if not isinstance(override, dict):
        return dict(base)

    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merged_fixture_from_config(environment_config: Optional[Dict[str, Any]]) -> EnvironmentFixture:
    fixture_config = {}
    if isinstance(environment_config, dict):
        fixture_config = environment_config.get("fixture") or {}
    return merge_environment_fixture(EnvironmentFixture(), fixture_config)


def _workspace_structure_from_fixture(fixture: EnvironmentFixture) -> List[Dict[str, Any]]:
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


def _vault_structure_text_from_fixture(fixture: EnvironmentFixture) -> str:
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
        lines.append("Use the memory load-workspace command to get full workspace context.")
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
            "IMPORTANT: When using tools, include these values in the tool-call context fields required by the active format:",
            "",
            f'- sessionId: "{session_id}"',
            f'- workspaceId: "{workspace_id}" (current workspace)',
            "",
            "Include these in the tool-call context fields required by the active format.",
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


def _note_entries_from_fixture(fixture: EnvironmentFixture, note_paths: Optional[List[str]] = None) -> List[Dict[str, str]]:
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


def _expected_context_from_system_context(system_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(system_context, dict):
        return None

    session_id = str(system_context.get("session_id", "")).strip()
    workspace_id = str(system_context.get("workspace_id", "")).strip()
    if not session_id or not workspace_id:
        return None

    workspace_ids = []
    for workspace in system_context.get("available_workspaces") or []:
        if isinstance(workspace, dict):
            wid = str(workspace.get("id", "")).strip()
            if wid:
                workspace_ids.append(wid)
    if workspace_id not in workspace_ids:
        workspace_ids.append(workspace_id)

    agent_ids = []
    for prompt in system_context.get("available_prompts") or []:
        if isinstance(prompt, dict):
            pid = str(prompt.get("id", "")).strip()
            if pid:
                agent_ids.append(pid)

    return {
        "session_id": session_id,
        "workspace_id": workspace_id,
        "workspace_ids": workspace_ids,
        "agent_ids": agent_ids,
    }


def _question_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        content = message.get("content")
        if content is not None:
            return str(content)
    return ""
