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


def _forbid_workflow_paths(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Captured workflow consulted mutable inputs")
    for name in ("load_tool_call_formats", "load_workspace_formats", "load_label_mappings", "load_privacy_profiles", "load_yaml"):
        monkeypatch.setattr("SynthChat.generator." + name, forbidden)
    monkeypatch.setattr("shared.validation.parsing.configured_formats.get_configured_wrapper_specs", forbidden)
    monkeypatch.setattr("shared.environments.tool_executor.cli_command_catalog", forbidden)
    monkeypatch.setattr("synaptic_tuner.api.v1.reference.data._read_dataset_rows", forbidden)


def test_real_generation_consumes_capture_after_source_deletion(harness, monkeypatch):
    from shared.llm.metering import MeteredLLMClient
    from shared.llm.usage import UsageAccumulator
    from test_reference_data_v1 import FakeLocalClient

    operations, request, ports, _, clients, scenario = harness
    scenario.write_text('scenarios:\n  selected:\n    type: behavioral\n    environment_execution: required\n    prompts: {system: system, user: user, assistant: assistant}\n')
    plan = operations.plan(request)
    captured = operations._verify_plan(plan)
    shutil.rmtree(ports.config_dir)
    scenario.unlink()
    _forbid_workflow_paths(monkeypatch)
    inner = FakeLocalClient(['bound system', 'bound user', '{"tool_calls":[{"function":{"name":"direct_action","arguments":{"value":"bound"}}}]}'])
    client = MeteredLLMClient(inner, UsageAccumulator())
    rows = []
    operations._generate(plan, client, None, lambda ref, row, passed, writer: rows.append((row, passed)),
                         lambda: None, lambda *args: None, admitted_inputs=captured)
    assert inner.chat_calls == 3 and clients.calls == 0
    assert rows[0][1] is True
    assert rows[0][0]['conversations'][0]['content'] == 'bound system'
    assert rows[0][0]['conversations'][-1]['tool_calls'][0]['function']['name'] == 'direct_action'


def test_real_improvement_consumes_captured_dataset_and_rubrics(harness, tmp_path, monkeypatch):
    from shared.llm.metering import MeteredLLMClient
    from shared.llm.usage import UsageAccumulator
    from test_reference_data_v1 import FakeLocalClient

    operations, request, ports, _, clients, _ = harness
    rubrics = tmp_path / 'bound-rubrics'
    shutil.copytree(ports.rubrics_dir, rubrics)
    ports = replace(ports, rubrics_dir=rubrics)
    operations._ports = ports
    dataset = tmp_path / 'input.jsonl'
    example = {'conversations': [{'role': 'user', 'content': 'bound question'}, {'role': 'assistant', 'content': 'bound answer'}]}
    dataset.write_text(json.dumps(example) + '\n')
    request = replace(request, mode=DataMode.IMPROVE, scenarios=(), input_dataset_ref=str(dataset), rubric_refs=('data_quality',))
    plan = operations.plan(request)
    captured = operations._verify_plan(plan)
    dataset.unlink()
    shutil.rmtree(ports.config_dir)
    shutil.rmtree(rubrics)
    _forbid_workflow_paths(monkeypatch)
    inner = FakeLocalClient([], structured=[{'data_quality_score': .95}])
    client = MeteredLLMClient(inner, UsageAccumulator())
    rows = []
    operations._improve(plan, client, None, lambda ref, row, passed, writer: rows.append((row, passed)),
                        lambda: None, lambda *args: None, admitted_inputs=captured)
    assert rows == [(example, True)] and inner.structured == [] and clients.calls == 0


def test_private_capture_must_match_plan_before_workflow_construction(harness):
    operations, request, _, _, clients, _ = harness
    plan = operations.plan(request)
    captured = operations._verify_plan(plan)
    mismatched = replace(plan, source_digest='b' * 64)
    with pytest.raises(DataOperationError, match='environment_invalid'):
        operations._workflow_snapshot(mismatched, captured)
    assert clients.calls == 0


@pytest.mark.parametrize('helper', ['_generate', '_improve'])
def test_selected_private_helpers_refuse_missing_capture_without_path_fallback(harness, helper, monkeypatch):
    operations, request, _, _, clients, _ = harness
    plan = operations.plan(request)
    _forbid_workflow_paths(monkeypatch)
    with pytest.raises(DataOperationError, match='environment_invalid'):
        getattr(operations, helper)(plan, None, None, None, None, None)
    assert clients.calls == 0


def test_effective_snapshot_validation_rejects_stage_override_before_clients(harness):
    operations, request, _, _, clients, scenario = harness
    scenario.write_text('scenarios:\n  selected:\n    environment_execution: required\n    assistant_generation: {model: raw-provider-override}\n')
    plan = operations.plan(request)
    with pytest.raises(DataOperationError, match='environment_invalid'):
        operations.start(plan)
    assert clients.calls == 0


@pytest.mark.parametrize('invalid_format', [
    {'wrapper_name': 'custom', 'command_field': 'undeclared'},
    {'wrapper_name': 'custom', 'command_escapes': {'multi-character': 'x'}},
    {'wrapper_name': 'custom', 'argument_fields': 'malformed shape'},
    {'wrapper_name': 'custom', 'argument_fields': ['malformed shape']},
])
def test_effective_wrapper_normalization_rejects_before_client_factory(harness, monkeypatch, invalid_format):
    import yaml
    import SynthChat.config.format_resolver as formats

    operations, request, ports, _, clients, _ = harness
    # Registry discovery remains the normal byte-binding path, redirected to this
    # host-owned fixture so the checked-in global registry is never modified.
    (ports.config_dir / 'tool_call_formats.yaml').write_text(yaml.safe_dump({'formats': {'default': invalid_format}}))
    monkeypatch.setattr(formats, '_CONFIG_DIR', ports.config_dir)
    plan = operations.plan(request)
    with pytest.raises(DataOperationError, match='environment_invalid'):
        operations.preflight(plan)
    with pytest.raises(DataOperationError, match='environment_invalid'):
        operations.start(plan)
    assert clients.calls == 0


def test_malformed_captured_validation_shape_has_closed_diagnostic(harness):
    import yaml

    operations, request, ports, _, clients, _ = harness
    path = ports.config_dir / 'validation.yaml'
    config = yaml.safe_load(path.read_text())
    config['scopes']['response']['extraction'] = 'malformed shape'
    path.write_text(yaml.safe_dump(config))
    plan = operations.plan(replace(request, max_iterations=2))
    with pytest.raises(DataOperationError, match='environment_invalid'):
        operations.preflight(plan)
    with pytest.raises(DataOperationError, match='environment_invalid'):
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
    assert roles == {"execution_policy", "fixture_schema", "tool_schema", "validation", "rubric", "dataset", "registry"}
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


@pytest.mark.parametrize("declaration", ["environment_execution: disabled", "environment: {enabled: false}"])
@pytest.mark.parametrize("mode", list(EnvironmentMode))
def test_disabled_scenario_rejects_any_environment_selection(harness, declaration, mode):
    operations, request, ports, _, clients, scenario = harness
    scenario.write_text("scenarios:\n  selected:\n    " + declaration + "\n")
    selection = replace(SELECTION, mode=mode)
    operations._ports = replace(ports, environments=HostEnvironment(selection))
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(replace(request, environment_selection=selection))
    assert clients.calls == 0


def test_required_and_disabled_declarations_are_contradictory(harness):
    operations, request, _, _, clients, scenario = harness
    scenario.write_text("scenarios:\n  selected:\n    environment_execution: required\n    environment: {enabled: false}\n")
    for selected in (request, replace(request, environment_selection=None)):
        with pytest.raises(DataOperationError, match="environment_invalid"):
            operations.plan(selected)
    assert clients.calls == 0


def test_explicit_contract_submodule_imports_no_engine_or_provider():
    subprocess.run([sys.executable, "-I", "-c", f"import sys;sys.path.insert(0,{str(ROOT)!r});"
        "import synaptic_tuner.api.v1.environment_execution;"
        "assert not any(n.split('.')[0] in {'tuner','SynthChat','modal','e2b','torch'} for n in sys.modules)"], check=True)


def test_fresh_capture_keeps_same_read_bytes_after_sources_change_or_disappear(harness, monkeypatch):
    operations, request, ports, host, clients, scenario = harness
    plan = operations.plan(request)
    capture = operations._verify_plan(plan)
    assert capture is not None
    expected_scenario = scenario.read_bytes()
    expected_privacy = (ports.config_dir / "privacy_profiles.yaml").read_bytes()
    expected_policy = host.raw["policy"]
    original_documents = capture.documents
    scenario.unlink()
    (ports.config_dir / "privacy_profiles.yaml").write_bytes(b"profiles: {}\n")
    host.raw["policy"] = b"changed after capture"
    # Consumption is strictly in-memory; no hidden reopen or resolver read.
    import synaptic_tuner.api.v1.reference.environment as admission
    monkeypatch.setattr(admission, "_read", lambda path: pytest.fail("captured access reopened source"))
    assert capture.document_bytes("scenario", "scenario:selected") == expected_scenario
    assert capture.document_bytes("privacy", "config:privacy_profiles") == expected_privacy
    assert capture.document_bytes("execution_policy", "policy") == expected_policy
    assert capture.documents == original_documents
    workflow = capture.workflow_documents()
    assert {entry[0] for entry in workflow} == {"scenario", "registry", "privacy"}
    assert all(type(entry[2]) is bytes for entry in workflow)
    assert clients.calls == 0


def test_new_start_capture_must_match_plan_before_clients(harness):
    operations, request, _, _, clients, scenario = harness
    plan = operations.plan(request)
    first = operations._verify_plan(plan)
    scenario.write_bytes(scenario.read_bytes() + b"\n# changed exact bytes, same parsed scenario\n")
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations._verify_plan(plan)
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.start(plan)
    assert first.document_bytes("scenario", "scenario:selected") != scenario.read_bytes()
    assert clients.calls == 0


def test_capture_clones_host_binding_and_rejects_substituted_documents(harness):
    operations, request, _, host, _, _ = harness
    plan = operations.plan(request)
    capture = operations._verify_plan(plan)
    object.__setattr__(host.dependencies[0], "sha256", "f" * 64)
    assert capture.binding == plan.environment_binding
    capture.workflow_documents()
    with pytest.raises(DataOperationError, match="environment_invalid"):
        replace(capture, documents=capture.documents[:-1])
    changed = list(capture.documents)
    role, ref, raw = changed[0]
    changed[0] = (role, ref, raw + b" ")
    with pytest.raises(DataOperationError, match="environment_invalid"):
        replace(capture, documents=tuple(changed))


def test_improvement_input_bytes_stay_reference_owned(harness, tmp_path):
    operations, request, _, _, clients, _ = harness
    dataset = tmp_path / "input.jsonl"
    raw = b'{"messages": [{"role": "assistant", "content": "original"}]}\n'
    dataset.write_bytes(raw)
    request = replace(request, mode=DataMode.IMPROVE, scenarios=(), input_dataset_ref=str(dataset),
                      rubric_refs=("response_quality",))
    plan = operations.plan(request)
    capture = operations._verify_plan(plan)
    dataset.unlink()
    assert capture.document_bytes("dataset", "input:dataset") == raw
    assert {entry[0] for entry in capture.workflow_documents()} == {"validation", "rubric", "registry"}
    assert "documents" not in plan.to_dict()
    with pytest.raises(DataOperationError):
        operations.start(plan)
    assert clients.calls == 0


_BAD_WORKFLOW_YAML = (
    b"a: 1\na: 2\n",
    b"a: &cycle [*cycle]\n",
    b"x: .nan\n", b"x: .inf\n",
    b"x: " + b"[" * 40 + b"0" + b"]" * 40,
    b"x: [" + b"0," * 20000 + b"0]\n",
    b"x: !!binary YQ==\n", b"x: 2026-10-13\n",
    b"1: not-a-string-key\n", b"[]\n",
    b"base: &base {x: 1}\nuse: {<<: *base, x: 2}\n",
)


@pytest.mark.parametrize("raw", _BAD_WORKFLOW_YAML)
def test_shared_workflow_parser_and_admission_reject_same_unsafe_documents(harness, raw):
    from SynthChat.utils.yaml_loader import parse_workflow_yaml
    from synaptic_tuner.api.v1.reference.environment import _yaml
    with pytest.raises(ValueError):
        parse_workflow_yaml(raw)
    with pytest.raises(ValueError):
        _yaml(raw)
    operations, request, _, host, clients, _ = harness
    host.raw["tool-schema"] = raw
    host.dependencies = tuple(replace(item, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
                              if item.role == "tool_schema" else item for item in host.dependencies)
    with pytest.raises(DataOperationError, match="environment_invalid"):
        operations.plan(request)
    assert clients.calls == 0


def test_valid_merge_alias_fractional_and_encoded_yaml_have_same_meaning(harness):
    from SynthChat.utils.yaml_loader import parse_workflow_yaml
    from synaptic_tuner.api.v1.reference.environment import _yaml
    text = "base: &base {fraction: 0.125}\nmerged: {<<: *base, label: café}\n"
    expected = {"base": {"fraction": 0.125}, "merged": {"fraction": 0.125, "label": "café"}}
    for raw in (text.encode("utf-8"), text.encode("utf-16")):
        assert parse_workflow_yaml(raw) == _yaml(raw) == expected
    operations, request, _, host, clients, _ = harness
    raw = text.encode("utf-8")
    host.raw["tool-schema"] = raw
    host.dependencies = tuple(replace(item, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
                              if item.role == "tool_schema" else item for item in host.dependencies)
    capture = operations._verify_plan(operations.plan(request))
    assert parse_workflow_yaml(capture.document_bytes("tool_schema", "tool-schema")) == expected
    assert clients.calls == 0
