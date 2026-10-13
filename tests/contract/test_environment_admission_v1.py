"""C1 admits pinned inputs, never opens an environment or LLM client."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from synaptic_tuner.api.v1.data_facade import (
    DataMode, DataOperationError, DataPlan, DataRequest, DataScenarioTarget,
)
from synaptic_tuner.api.v1.environment_execution import (
    ENVIRONMENT_BINDING_SCHEMA, EnvironmentBindingV1, EnvironmentBoundsV1,
    EnvironmentDependencyV1, EnvironmentMode, EnvironmentSelectionV1,
)
from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1, ReferenceDataPortsV1
from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1


ROOT = Path(__file__).resolve().parents[2]
BOUNDS = EnvironmentBoundsV1(4, 5, 8, 10, 1024, 1024, 1024, 20)
SELECTION = EnvironmentSelectionV1("fixture-profile", EnvironmentMode.EXECUTE)


class HostEnvironment:
    def __init__(self, selection=SELECTION):
        self.selection = selection
        policy = {
            "backend_ref": "local-fixture", "backend_revision": "a" * 64,
            "scope_ref": "host-scope", "mode": selection.mode.value,
            "bounds": BOUNDS.to_dict(), "allowed_actions": ["read", "write"],
        }
        self.raw = {
            "tool-schema": json.dumps({"tools": {"configured-action": {}}}).encode(),
            "policy": json.dumps(policy).encode(),
            "fixture-schema": json.dumps({"type": "object", "additionalProperties": False,
                                          "properties": {"fixture": {"type": "object"}}}).encode(),
        }
        self.dependencies = tuple(EnvironmentDependencyV1(role, ref, hashlib.sha256(raw).hexdigest(), len(raw))
            for role, ref, raw in (("tool_schema", "tool-schema", self.raw["tool-schema"]),
                                  ("execution_policy", "policy", self.raw["policy"]),
                                  ("fixture_schema", "fixture-schema", self.raw["fixture-schema"])))
        self.resolve_calls = 0

    def resolve(self, selection, *, project_ref):
        self.resolve_calls += 1
        return EnvironmentBindingV1(ENVIRONMENT_BINDING_SCHEMA, self.selection, "local-fixture",
                                    "a" * 64, "host-scope", BOUNDS, ("read", "write"), self.dependencies)

    def read_dependency(self, ref, *, maximum_bytes):
        return self.raw[ref]


class NoClients:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise AssertionError("C1 must not create clients")


class Clock:
    def now(self):
        return "2026-10-13T12:00:00Z"


@pytest.fixture
def harness(tmp_path):
    config = tmp_path / "config"
    shutil.copytree(ROOT / "SynthChat/config", config)
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    scenario = scenarios / "selected.yaml"
    scenario.write_text("scenarios:\n  selected:\n    type: behavioral\n    environment: {fixture: {}}\n")
    clients = NoClients()
    host = HostEnvironment()
    ports = ReferenceDataPortsV1(config, scenarios, ROOT / "SynthChat/rubrics", clients, host)
    records = InMemoryDurableRecordStoreV1()
    operations = ReferenceDataOperationsV1(records=records, streams=InMemoryDurableStreamStoreV1(), clock=Clock(), ports=ports)
    request = DataRequest("request", "project", DataMode.GENERATE, "lmstudio", "model",
                          str(tmp_path / "host-output.jsonl"), (DataScenarioTarget("selected", 1),),
                          environment_selection=SELECTION)
    return operations, request, ports, host, clients, scenario


def test_opt_in_plan_is_versioned_bound_and_nonexecuting(harness):
    operations, request, _, _, clients, _ = harness
    plan = operations.plan(request)
    assert plan.schema_version == "synaptic-dataset-plan/v2"
    assert DataPlan.from_dict(plan.to_dict()) == plan
    schema = json.loads((ROOT / "schemas/synaptic-dataset-plan-v2.schema.json").read_text())
    Draft202012Validator(schema).validate(plan.to_dict())
    roles = {item.role for item in plan.environment_binding.dependencies}
    assert roles == {"execution_policy", "fixture_schema", "tool_schema", "scenario", "registry", "privacy"}
    preflight = operations.preflight(plan)
    assert not preflight.ready and "environment_unavailable" in preflight.diagnostic_codes
    with pytest.raises(DataOperationError, match="environment_unavailable"):
        operations.start(plan)
    assert clients.calls == 0


def test_simulation_is_also_disabled_until_an_adapter_is_qualified(harness):
    operations, request, ports, _, clients, _ = harness
    selection = replace(SELECTION, mode=EnvironmentMode.SIMULATE)
    host = HostEnvironment(selection)
    operations._ports = replace(ports, environments=host)
    plan = operations.plan(replace(request, environment_selection=selection))
    with pytest.raises(DataOperationError, match="environment_unavailable"):
        operations.start(plan)
    assert clients.calls == 0


@pytest.mark.parametrize("missing", ["selection", "port", "fixture-schema"])
def test_missing_requested_inputs_reject_before_clients(harness, missing):
    operations, request, ports, host, clients, _ = harness
    if missing == "selection":
        request = replace(request, environment_selection=None)
    elif missing == "port":
        operations._ports = replace(ports, environments=None)
    else:
        host.raw.pop(missing)
    with pytest.raises(DataOperationError):
        operations.plan(request)
    assert clients.calls == 0


@pytest.mark.parametrize("mutated", ["policy", "tool-schema", "fixture-schema", "privacy", "scenario", "duplicate"])
def test_dependency_mutation_and_duplicate_selected_id_reject_before_effects(harness, mutated):
    operations, request, ports, host, clients, scenario = harness
    plan = operations.plan(request)
    if mutated in host.raw:
        host.raw[mutated] += b" "
    elif mutated == "privacy":
        (ports.config_dir / "privacy_profiles.yaml").write_bytes(b"profiles: {}\n")
    elif mutated == "scenario":
        scenario.write_text(scenario.read_text() + "    extra: changed\n")
    else:
        (ports.scenarios_dir / "duplicate.yaml").write_bytes(scenario.read_bytes())
    with pytest.raises(DataOperationError):
        operations.start(plan)
    assert clients.calls == 0


@pytest.mark.parametrize("field,value", [("backend_ref", "cloud"), ("scope_ref", "other"),
                                        ("bounds", {"max_seconds": 99999}), ("credentials", {"token": "test"})])
def test_scenario_data_cannot_escalate_host_policy(harness, field, value):
    operations, request, _, _, clients, scenario = harness
    scenario.write_text(json.dumps({"scenarios": {"selected": {"environment": {field: value}}}}))
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(request)
    assert clients.calls == 0


def test_remote_schema_reference_is_rejected_without_resolution(harness):
    operations, request, _, host, clients, _ = harness
    raw = json.dumps({"type": "object", "additionalProperties": False,
                      "properties": {"fixture": {"$ref": "https://invalid.example/schema"}}}).encode()
    host.raw["fixture-schema"] = raw
    host.dependencies = tuple(replace(item, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
                              if item.role == "fixture_schema" else item for item in host.dependencies)
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(request)
    assert clients.calls == 0


def test_remote_tool_argument_schema_reference_is_rejected(harness):
    operations, request, _, host, clients, _ = harness
    raw = json.dumps({"tools": {"configured-action": {"arguments": {
        "$ref": "https://invalid.example/arguments"}}}}).encode()
    host.raw["tool-schema"] = raw
    host.dependencies = tuple(replace(item, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
                              if item.role == "tool_schema" else item for item in host.dependencies)
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(request)
    assert clients.calls == 0


@pytest.mark.parametrize("raw", [b"not: [valid", b"tools: {}\ntools: {}\n", b"x: .nan\n",
                                b"x: &recursive [*recursive]\n", b"{}", b"[]"])
def test_corrupt_or_empty_host_dependency_is_rejected(harness, raw):
    operations, request, _, host, clients, _ = harness
    host.raw["tool-schema"] = raw
    host.dependencies = tuple(replace(item, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
                              if item.role == "tool_schema" else item for item in host.dependencies)
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(request)
    assert clients.calls == 0


def test_improvement_binds_actual_dataset_validation_and_rubric_bytes(harness, tmp_path):
    operations, request, ports, _, clients, _ = harness
    rubrics = tmp_path / "rubrics"
    shutil.copytree(ports.rubrics_dir, rubrics)
    operations._ports = replace(ports, rubrics_dir=rubrics)
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "original"}]}\n')
    request = replace(request, mode=DataMode.IMPROVE, scenarios=(), input_dataset_ref=str(dataset),
                      rubric_refs=("response_quality",))
    plan = operations.plan(request)
    roles = {item.role for item in plan.environment_binding.dependencies}
    assert roles == {"execution_policy", "fixture_schema", "tool_schema", "validation", "rubric", "dataset"}
    rubric = rubrics / "response_quality.yaml"
    rubric.write_bytes(rubric.read_bytes() + b"\n# changed input bytes\n")
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.start(plan)
    assert clients.calls == 0


def test_unselected_scenario_bytes_are_not_committed_but_duplicates_are_checked(harness):
    operations, request, ports, _, _, _ = harness
    other = ports.scenarios_dir / "other.yaml"
    other.write_text("scenarios:\n  unrelated: {type: behavioral}\n")
    before = operations.plan(request)
    other.write_text("scenarios:\n  unrelated: {type: behavioral, prompt: changed}\n")
    after = operations.plan(request)
    assert after.plan_fingerprint == before.plan_fingerprint


def test_symlink_input_and_over_bound_scan_reject_before_clients(harness):
    operations, request, ports, _, clients, scenario = harness
    original = scenario.read_bytes()
    scenario.unlink()
    target = ports.config_dir / "scenario-copy.yaml"
    target.write_bytes(original)
    scenario.symlink_to(target)
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(request)
    scenario.unlink()
    scenario.write_bytes(original)
    for index in range(128):
        (ports.scenarios_dir / f"other-{index}.yaml").write_text("scenarios: {}\n")
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(request)
    assert clients.calls == 0


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, float("inf"), float("nan"), 10001])
def test_bounds_reject_nonexact_or_out_of_range_values(value):
    with pytest.raises(ValueError):
        replace(BOUNDS, max_rows=value)


def test_legacy_selection_omission_and_plan_bytes_remain_v1(harness):
    operations, request, _, _, _, scenario = harness
    scenario.write_text("scenarios:\n  selected: {type: behavioral}\n")
    request = replace(request, environment_selection=None)
    assert "environment_selection" not in request.to_dict()
    plan = operations.plan(request)
    assert plan.schema_version == "synaptic-dataset-plan/v1"
    assert set(plan.to_dict()) == {"schema_version", "request", "scenarios", "rows_requested", "source_digest"}
    assert DataPlan.from_dict(plan.to_dict()) == plan
    schema = json.loads((ROOT / "schemas/synaptic-dataset-plan-v1.schema.json").read_text())
    Draft202012Validator(schema).validate(plan.to_dict())


@pytest.mark.parametrize("declaration", ["environment_execution: disabled", "environment: {enabled: false}"])
def test_explicit_nonexecuting_environment_generation_needs_no_execution_port(harness, declaration):
    operations, request, ports, _, clients, scenario = harness
    operations._ports = replace(ports, environments=None)
    request = replace(request, environment_selection=None)
    scenario.write_text("scenarios:\n  selected:\n    type: behavioral\n    " + declaration +
                        "\n    environment_generation: {prompt: 'Generate bounded fixture data'}\n")
    plan = operations.plan(request)
    assert plan.environment_binding is None and plan.schema_version == "synaptic-dataset-plan/v1"
    assert clients.calls == 0


def test_unbound_file_template_is_rejected_without_opening_it(harness, tmp_path):
    operations, request, _, _, clients, scenario = harness
    path = tmp_path / "external-template.txt"
    path.write_text("Uncommitted template bytes")
    scenario.write_text(json.dumps({"scenarios": {"selected": {"system": {"template": str(path)}}}}))
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(request)
    assert clients.calls == 0


@pytest.mark.parametrize("value", [None, False, 0, "unknown", {}, []])
def test_malformed_execution_declaration_rejects_with_or_without_selection(harness, value):
    operations, request, _, _, clients, scenario = harness
    scenario.write_text(json.dumps({"scenarios": {"selected": {"environment_execution": value}}}))
    for selected in (request, replace(request, environment_selection=None)):
        with pytest.raises(DataOperationError, match="environment_invalid"):
            operations.plan(selected)
    assert clients.calls == 0


def test_required_execution_rejects_missing_or_simulated_selection(harness):
    operations, request, ports, _, clients, scenario = harness
    scenario.write_text("scenarios:\n  selected: {environment_execution: required}\n")
    with pytest.raises(DataOperationError, match="environment_required"):
        operations.plan(replace(request, environment_selection=None))
    selection = replace(SELECTION, mode=EnvironmentMode.SIMULATE)
    operations._ports = replace(ports, environments=HostEnvironment(selection))
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(replace(request, environment_selection=selection))
    assert clients.calls == 0


def test_explicit_contract_submodule_imports_no_engine_or_provider():
    subprocess.run([sys.executable, "-I", "-c", f"import sys;sys.path.insert(0,{str(ROOT)!r});"
        "import synaptic_tuner.api.v1.environment_execution;"
        "assert not any(n.split('.')[0] in {'tuner','SynthChat','modal','e2b','torch'} for n in sys.modules)"], check=True)
