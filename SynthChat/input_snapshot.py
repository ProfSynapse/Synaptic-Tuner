"""Immutable admitted workflow documents; no provider or execution authority.

Snapshot validation is pure and must run before a host opens LLM clients or
sessions. Shared response-parser/tool-executor custody is a separate gate.
"""
from dataclasses import dataclass, field
import re

from .utils.yaml_loader import parse_workflow_yaml

_REF = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_SINGLE_REFS = {
    "validation": {"config:validation"},
    "privacy": {"config:privacy_profiles"},
    "registry": {"registry:tool_call_formats", "registry:workspace_formats", "registry:label_mappings"},
}


def reject_snapshot_client_override(config):
    """Use existing client-spec semantics; snapshot mode uses injected clients."""
    from .llm.client_pool import LLMClientPool
    if LLMClientPool.normalize_stage_spec(config) is not None:
        raise ValueError("LLM client overrides are unsupported with input snapshots")
    if isinstance(config, dict) and config.get("fallback_models"):
        raise ValueError("LLM fallback overrides are unsupported with input snapshots")


@dataclass(frozen=True)
class WorkflowInputSnapshot:
    documents: tuple[tuple[str, str, bytes], ...] = field(repr=False)

    def __post_init__(self):
        if type(self.documents) is not tuple or not 0 < len(self.documents) <= 128:
            raise ValueError("snapshot requires a bounded immutable document tuple")
        refs, total = set(), 0
        for entry in self.documents:
            if type(entry) is not tuple or len(entry) != 3:
                raise TypeError("snapshot entries must be exact role/ref/bytes tuples")
            role, ref, raw = entry
            if type(role) is not str or type(ref) is not str or _REF.fullmatch(ref) is None:
                raise ValueError("snapshot entries require logical role/reference names")
            if role in _SINGLE_REFS:
                valid = ref in _SINGLE_REFS[role]
            else:
                valid = role in {"scenario", "rubric"} and ref.startswith(role + ":") and bool(ref[len(role) + 1:])
            if not valid or ref in refs:
                raise ValueError("snapshot references must be unique and match workflow roles")
            if type(raw) is not bytes or not 0 < len(raw) <= 4 * 1024 * 1024:
                raise ValueError("snapshot document bytes exceed their bound")
            refs.add(ref)
            total += len(raw)
        if total > 16 * 1024 * 1024:
            raise ValueError("snapshot aggregate bytes exceed their bound")
        # Fail corrupt selected documents eagerly, without path fallbacks.
        for role, ref, _ in self.documents:
            self.document(role, ref)

    def has(self, role, ref):
        return any(item_role == role and item_ref == ref for item_role, item_ref, _ in self.documents)

    def document(self, role, ref):
        for item_role, item_ref, raw in self.documents:
            if (item_role, item_ref) == (role, ref):
                value = parse_workflow_yaml(raw)
                if type(value) is not dict or not value:
                    raise ValueError("snapshot document must contain a nonempty object")
                return value  # Fresh parse; never share mutable parser/cache state.
        raise ValueError(f"Required snapshot document missing: {role}/{ref}")

    def role_documents(self, role):
        return {ref.split(":", 1)[1]: self.document(role, ref)
                for item_role, ref, _ in self.documents if item_role == role}

    def validate_generation(self, scenario_refs, *, max_iterations=1, privacy_settings=None):
        """Pure pre-effect validation of currently supported snapshot scenarios."""
        from .config.format_resolver import parse_tool_call_formats, parse_workspace_formats, parse_label_mappings
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("Snapshot generation requires a positive iteration count")
        if max_iterations > 1:
            self.validate_improvement()
        from .config.privacy import parse_privacy_profiles, resolve_privacy_settings
        privacy = resolve_privacy_settings({"privacy_preprocess": privacy_settings or {}})
        if privacy.get("enabled"):
            profiles = parse_privacy_profiles(self.document("privacy", "config:privacy_profiles"), require_complete=True)
            if not profiles["profiles"].get(privacy.get("profile")):
                raise ValueError("Selected privacy profile is absent from snapshot")
        registries = {}
        for name, parser in (("tool_call_formats", parse_tool_call_formats),
                             ("workspace_formats", parse_workspace_formats), ("label_mappings", parse_label_mappings)):
            registries[name] = parser(self.document("registry", "registry:" + name), require_complete=True)
        from shared.validation.parsing.configured_formats import build_wrapper_specs
        build_wrapper_specs(registries["tool_call_formats"])
        if type(scenario_refs) is not tuple or not scenario_refs or any(type(ref) is not str or not ref for ref in scenario_refs) or len(set(scenario_refs)) != len(scenario_refs):
            raise ValueError("Snapshot generation requires explicit selected scenario references")
        scenarios = self.role_documents("scenario")
        if not scenarios:
            raise ValueError("Generation snapshot has no selected scenario documents")
        all_definitions = {}
        for document in scenarios.values():
            definitions = document.get("scenarios")
            if type(definitions) is not dict or not definitions:
                raise ValueError("Scenario document requires scenario definitions")
            for key, definition in definitions.items():
                if key not in scenario_refs:
                    continue
                if key in all_definitions:
                    raise ValueError("Snapshot scenario definition IDs conflict")
                all_definitions[key] = definition
        if any(ref not in all_definitions for ref in scenario_refs):
            raise ValueError("Selected scenario is absent from snapshot documents")
        for ref in scenario_refs:
            scenario = all_definitions[ref]
            if type(scenario) is not dict:
                raise ValueError("Scenario definitions must be objects")
            system = scenario.get("system")
            if isinstance(system, dict) and "template" in system:
                raise ValueError("File templates are unsupported with input snapshots")
            for name in ("system_generation", "user_generation", "thinking_generation",
                         "assistant_generation", "environment_generation", "final_judge"):
                stage = scenario.get(name)
                reject_snapshot_client_override(stage)
                if isinstance(stage, dict):
                    reject_snapshot_client_override(stage.get("judge"))
            judge = scenario.get("judge")
            if isinstance(judge, dict):
                reject_snapshot_client_override(judge.get("in_loop"))
            for selector, registry in (("tool_call_format", registries["tool_call_formats"]),
                                    ("workspace_format", registries["workspace_formats"])):
                named = scenario.get(selector)
                if isinstance(named, str) and not registry.get(named):
                    raise ValueError("Selected named format is absent from snapshot registry")
            if max_iterations > 1:
                rubric_groups = scenario.get("rubrics") or {}
                if type(rubric_groups) is not dict:
                    raise ValueError("Scenario rubric declarations must be grouped lists")
                requested = []
                for group in rubric_groups.values():
                    if type(group) is not list or any(type(ref) is not str or not ref for ref in group):
                        raise ValueError("Scenario rubric references must be lists of names")
                    requested.extend(group)
                if requested:
                    self.validate_improvement(tuple(dict.fromkeys(requested)))
            seed = scenario.get("seed_data") or {}
            if isinstance(seed, dict) and seed.get("preprocess_profile"):
                profiles = parse_privacy_profiles(self.document("privacy", "config:privacy_profiles"), require_complete=True)
                if not profiles["profiles"].get(seed["preprocess_profile"]):
                    raise ValueError("Selected privacy profile is absent from snapshot")
        return {ref: all_definitions[ref] for ref in scenario_refs}

    def validate_improvement(self, rubric_refs=None):
        """Pure validation of scope/rubric consistency before client creation."""
        from .config.config_loader import ConfigLoader
        config = ConfigLoader.from_mapping(self.document("validation", "config:validation")).load()
        rubrics = self.role_documents("rubric")
        if rubric_refs is not None:
            if (type(rubric_refs) is not tuple or not rubric_refs
                    or any(type(ref) is not str or not ref for ref in rubric_refs)
                    or len(set(rubric_refs)) != len(rubric_refs)):
                raise ValueError("Improvement requires explicit unique rubric references")
            if any(ref not in rubrics for ref in rubric_refs):
                raise ValueError("Selected rubric is absent from snapshot")
            rubrics = {ref: rubrics[ref] for ref in rubric_refs}
        for rubric in rubrics.values():
            scope = rubric.get("scope", "response")
            scopes = scope if isinstance(scope, list) else [scope]
            if not scopes or any(type(name) is not str or name not in config.scopes
                                 or name not in config.scope_processing_order for name in scopes):
                raise ValueError("Selected rubric scopes are absent from snapshot validation")
        return config
