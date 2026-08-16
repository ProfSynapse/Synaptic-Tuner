from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

from synaptic_tuner import __version__
from synaptic_tuner._version import __version__ as canonical_version
from synaptic_tuner.api.v1 import (
    CapabilityDescriptor,
    EventEnvelope,
    PathRef,
    PluginContext,
    ProjectContext,
    ResultEnvelope,
    SecretRef,
    SourceLock,
)
from tuner.project.plugins import (
    PLUGIN_ENTRY_POINT_GROUP,
    TRUST_NOTICE,
    PluginConflictError,
    PluginContractError,
    PluginLoadError,
    discover_plugins,
    plugin_entry_point_group,
    resolve_plugin,
)

KIND = "renderer"
API = "synaptic.plugin/renderer/v1"


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    loaded: bool = False
    group: str = plugin_entry_point_group(KIND, API)
    dist: object | None = None

    def load(self) -> object:
        self.loaded = True
        raise AssertionError("metadata discovery must not load entry points")


def manifest(target: str = "host_plugins.prompt:render") -> dict[str, object]:
    return {
        "plugins": {
            "bindings": {
                "example.prompt": {"kind": KIND, "api": API, "target": target}
            }
        }
    }


def test_manifest_binding_precedes_installed_entry_point_without_import() -> None:
    installed = FakeEntryPoint("example.prompt", "installed_plugin:render")

    binding = resolve_plugin(
        "plugin://example.prompt",
        kind=KIND,
        api=API,
        manifest=manifest(),
        entry_points=[installed],
    )

    assert binding.origin == "manifest"
    assert binding.target == "host_plugins.prompt:render"
    assert installed.loaded is False
    assert binding.to_dict()["trust_notice"] == TRUST_NOTICE


def test_installed_metadata_discovery_never_loads_plugin() -> None:
    installed = FakeEntryPoint("example.prompt", "expensive_gpu_plugin:render")

    binding = resolve_plugin(
        "plugin://example.prompt", kind=KIND, api=API, entry_points=[installed]
    )
    discovered = discover_plugins(kind=KIND, api=API, entry_points=[installed])

    assert binding.origin == "entry_point"
    assert discovered == (binding,)
    assert installed.loaded is False
    assert "expensive_gpu_plugin" not in sys.modules


def test_installed_contract_cannot_be_reinterpreted_by_the_caller() -> None:
    installed = FakeEntryPoint("example.prompt", "builtins:len")

    renderer = resolve_plugin(
        "plugin://example.prompt", kind=KIND, api=API, entry_points=[installed]
    )

    assert renderer.kind == KIND
    assert renderer.api == API
    assert installed.loaded is False
    with pytest.raises(PluginContractError, match="requested contract") as exc_info:
        resolve_plugin(
            "plugin://example.prompt",
            kind="grader",
            api="synaptic.plugin/grader/v1",
            entry_points=[installed],
        )
    assert exc_info.value.details["declared_contracts"] == [
        {
            "kind": KIND,
            "api": API,
            "group": plugin_entry_point_group(KIND, API),
        }
    ]
    assert installed.loaded is False


def test_installed_binding_without_versioned_contract_metadata_fails_closed() -> None:
    installed = FakeEntryPoint(
        "example.prompt", "builtins:len", group=PLUGIN_ENTRY_POINT_GROUP
    )

    with pytest.raises(PluginContractError, match="missing authoritative"):
        resolve_plugin(
            "plugin://example.prompt", kind=KIND, api=API, entry_points=[installed]
        )
    assert installed.loaded is False


@pytest.mark.parametrize(
    "group",
    [
        PLUGIN_ENTRY_POINT_GROUP,
        f"{PLUGIN_ENTRY_POINT_GROUP}.renderer.latest",
        f"{PLUGIN_ENTRY_POINT_GROUP}.renderer.v2",
    ],
)
def test_bulk_discovery_fails_closed_on_invalid_reserved_group(group: str) -> None:
    installed = FakeEntryPoint("invalid.prompt", "builtins:len", group=group)

    with pytest.raises(PluginContractError):
        discover_plugins(kind=KIND, api=API, entry_points=[installed])

    assert installed.loaded is False


def test_bulk_discovery_validates_other_contracts_then_filters_them() -> None:
    grader = FakeEntryPoint(
        "example.grader",
        "builtins:len",
        group=plugin_entry_point_group("grader", "synaptic.plugin/grader/v1"),
    )
    unrelated = FakeEntryPoint(
        "example.prompt", "builtins:len", group="some_other_project.plugins"
    )

    assert discover_plugins(
        kind=KIND, api=API, entry_points=[grader, unrelated]
    ) == ()
    assert grader.loaded is False
    assert unrelated.loaded is False


def test_installed_api_kind_must_match_requested_kind() -> None:
    with pytest.raises(PluginContractError, match="requested kind"):
        resolve_plugin(
            "plugin://example.prompt",
            kind="grader",
            api=API,
            entry_points=[],
        )


def test_explicit_legacy_target_is_a_compatibility_binding() -> None:
    binding = resolve_plugin(
        "legacy_renderer:render", kind=KIND, api=API, entry_points=[]
    )

    assert binding.origin == "legacy"
    assert binding.target == "legacy_renderer:render"


def test_manifest_contract_mismatch_is_fatal_before_execution() -> None:
    with pytest.raises(PluginContractError, match="requested contract"):
        resolve_plugin(
            "plugin://example.prompt",
            kind="grader",
            api=API,
            manifest=manifest(),
            entry_points=[],
        )


def test_manifest_name_cannot_fall_back_to_installed_plugin_of_another_kind() -> None:
    installed = FakeEntryPoint("example.prompt", "installed_plugin:render")
    discovered = discover_plugins(
        kind="grader",
        api="synaptic.plugin/grader/v1",
        manifest=manifest(),
        entry_points=[installed],
    )

    assert discovered == ()
    assert installed.loaded is False


def test_duplicate_installed_bindings_are_fatal() -> None:
    entry_points = [
        FakeEntryPoint("example.prompt", "first:render"),
        FakeEntryPoint("example.prompt", "second:render"),
    ]
    with pytest.raises(PluginConflictError, match="Duplicate"):
        discover_plugins(kind=KIND, api=API, entry_points=entry_points)


def test_trusted_load_is_explicit_and_wraps_import_failure() -> None:
    binding = resolve_plugin(
        "missing_trusted_plugin:render", kind=KIND, api=API, entry_points=[]
    )
    with pytest.raises(PluginLoadError, match="Trusted plug-in load failed"):
        binding.load_trusted()


def test_public_api_is_narrow_versioned_and_serializable(tmp_path) -> None:
    assert canonical_version == "1.1.0"
    assert __version__ == canonical_version
    assert PathRef.parse("project://data/train.jsonl").scheme == "project"
    assert SecretRef("env", "HF_TOKEN").to_dict() == {
        "provider": "env",
        "name": "HF_TOKEN",
    }
    context = ProjectContext.host(engine_root=tmp_path / "engine", project_root=tmp_path)
    plugin_context = PluginContext(
        project=context, name="example.prompt", kind=KIND, api=API
    )
    assert plugin_context.project == context

    capability = CapabilityDescriptor(
        id="example.run", summary="Run an example", command=("example", "run")
    )
    result = ResultEnvelope(True, "example.run", "run_1")
    event = EventEnvelope(
        "completed", "example.run", "run_1", 1, final=True, result=result
    )
    assert capability.to_dict()["schema_version"] == "synaptic-capability/v1"
    assert event.to_dict()["result"]["schema_version"] == "synaptic-result/v1"
    assert SourceLock.__module__ == "tuner.project.source_bundle"
