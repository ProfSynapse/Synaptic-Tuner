"""Reference Data family over in-memory stores and a fake local LLM client (api-facade slice 9).

Drives ``ReferenceDataOperationsV1`` through ``DataAPI`` on an ``APIHost``:
generate and improve runs over SynthChat with a scripted client, artifact
digests against the files on disk, the sidecar as ``dataset_metadata``, an
aborted run finishing ``partially_succeeded`` with its artifact, cooperative
cancellation at a row boundary finishing ``cancelled`` with its artifact, the
metered-backend refusal, the durable event budget, paging for ``list``,
``datasets`` and ``validate`` with ``cursor_invalid``, and the import gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1.data_facade import (
    DataAPI,
    DataListRequest,
    DataMode,
    DataOperationCode,
    DataOperationError,
    DataOutcome,
    DataPlan,
    DataRequest,
    DataRunRef,
    DataRunState,
    DataScenarioOutcome,
    DataScenarioTarget,
    DatasetListRequest,
    DatasetValidateRequest,
    ValidationFindingCode,
)
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.observations import (
    ObservationFamily,
    ObservationKind,
    ObservationStreamRef,
    ObservationsRequest,
)
from synaptic_tuner.api.v1.ports import StoragePartition
from synaptic_tuner.api.v1.reference import ReferenceDataPortsV1, build_data_operations, compose_reference_host
from synaptic_tuner.api.v1.reference.data import (
    DATA_DURABLE_EVENT_BUDGET,
    ReferenceDataOperationsV1,
    data_run_ref,
)
from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1
from shared.llm.usage import LLMCompletionV1, LLMStructuredV1


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "SynthChat" / "config"
RUBRICS_DIR = ROOT / "SynthChat" / "rubrics"
PROJECT = "acme"
SCENARIO_YAML = """scenarios:
  greeting:
    type: behavioral
    prompts:
      system: Write a system prompt.
      user: Write a user message.
      assistant: Write an assistant reply.
  farewell:
    type: behavioral
    prompts:
      system: Write a system prompt.
      user: Write a parting user message.
      assistant: Write a parting assistant reply.
"""
CHAT_CALLS_PER_ROW = 3


class Clock:
    def __init__(self) -> None:
        self.tick = 0

    def now(self) -> str:
        self.tick += 1
        return f"2026-09-17T12:{self.tick // 60:02d}:{self.tick % 60:02d}Z"


class FakeLocalClient:
    """Scripted ``shared.llm`` style client; ``on_call`` runs before every chat call."""

    def __init__(self, responses, *, structured=(), failure=None, on_call=None) -> None:
        self.responses = list(responses)
        self.structured = list(structured)
        self.failure = failure
        self.on_call = on_call
        self.chat_calls = 0
        self.default_max_tokens = None
        self.provider = None
        self.timeout_seconds = 60.0

    provider_name = "lmstudio"
    model_name = "fake-local"

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        if self.on_call is not None:
            self.on_call(self.chat_calls)
        self.chat_calls += 1
        if not self.responses:
            if self.failure is not None:
                raise self.failure
            raise AssertionError("scripted responses exhausted")
        return LLMCompletionV1(self.responses.pop(0))

    def structured_output(self, messages, schema, temperature=0.3, max_tokens=2048):
        return LLMStructuredV1(self.structured.pop(0))


class Factory:
    def __init__(self, client) -> None:
        self.client = client
        self.requests = []

    def create(self, *, backend, model):
        self.requests.append((backend, model))
        return self.client


class Harness:
    def __init__(self, tmp_path: Path, client, *, records=None, streams=None) -> None:
        self.root = tmp_path
        self.scenarios_dir = tmp_path / "scenarios"
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        (self.scenarios_dir / "behavioral.yaml").write_text(SCENARIO_YAML, encoding="utf-8")
        self.records = records or InMemoryDurableRecordStoreV1()
        self.streams = streams or InMemoryDurableStreamStoreV1()
        self.clock = Clock()
        self.factory = Factory(client)
        self.operations = build_data_operations(
            records=self.records, streams=self.streams, clock=self.clock,
            ports=ReferenceDataPortsV1(CONFIG_DIR, self.scenarios_dir, RUBRICS_DIR, self.factory),
        )
        self.host = APIHost(HostPorts(
            training=object(), runs=object(), artifacts=None, evaluation=None,
            chat=None, data=self.operations, pipelines=None, clock=self.clock,
        ))

    @property
    def api(self) -> DataAPI:
        return self.host.data

    def request(self, name: str = "run", *, backend: str = "lmstudio", **changes) -> DataRequest:
        values = {
            "request_id": f"request-{name}", "project_ref": PROJECT, "mode": DataMode.GENERATE,
            "backend": backend, "model": "fake-local", "output_ref": str(self.root / f"{name}.jsonl"),
            "scenarios": (DataScenarioTarget("greeting", 2), DataScenarioTarget("farewell", 1)),
        }
        values.update(changes)
        return DataRequest(**values)  # type: ignore[arg-type]

    def observations(self, run: DataRunRef):
        stream = ObservationStreamRef(ObservationFamily.DATA, run.project_ref, run.run_id)
        return self.api.observations(ObservationsRequest(stream, limit=100)).records

    def durable_events(self, run: DataRunRef) -> int:
        from synaptic_tuner.api.v1.reference.data import _run_key

        page = self.streams.read_page(
            partition=StoragePartition.DATA.value, stream_key=_run_key(run), after_sequence=None, limit=100,
        )
        return len(page.entries)


def _digest(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_artifacts_match_disk(outcome: DataOutcome, output: Path) -> None:
    by_role = {artifact.role: artifact for artifact in outcome.artifacts}
    assert set(by_role) == {"dataset_jsonl", "dataset_metadata"}
    assert (by_role["dataset_jsonl"].sha256, by_role["dataset_jsonl"].size_bytes) == _digest(output)
    sidecar = output.with_name(output.name + ".meta.json")
    assert (by_role["dataset_metadata"].sha256, by_role["dataset_metadata"].size_bytes) == _digest(sidecar)
    rows = _rows(output)
    assert rows and all(type(row) is dict and "_meta" not in row and "conversations" in row for row in rows)
    assert len(rows) == outcome.rows_written
    assert json.loads(sidecar.read_text(encoding="utf-8"))["rows_written"] == outcome.rows_written


def _no_synaptic_dir(*roots: Path) -> bool:
    return all(not any(path.name == ".synaptic" for path in root.rglob(".synaptic")) for root in roots)


# --- generate ------------------------------------------------------------------------


def test_generate_run_writes_one_homogeneous_artifact_and_stays_within_the_event_budget(tmp_path) -> None:
    client = FakeLocalClient([f"text {index}" for index in range(3 * CHAT_CALLS_PER_ROW)])
    harness = Harness(tmp_path, client)
    api = harness.api
    request = harness.request()
    cwd_synaptic = (Path.cwd() / ".synaptic").exists()

    plan = api.plan(request)
    assert type(plan) is DataPlan and plan.rows_requested == 3 and plan.scenarios == request.scenarios
    assert api.plan(request) == plan
    preflight = api.preflight(plan)
    assert preflight.ready and preflight.binds(plan)
    assert [item.operation for item in preflight.authorization] == ["data.start"]
    assert all(item.paid_effect is False for item in preflight.authorization)

    start = api.start(plan)
    assert start.accepted and start.run == data_run_ref(plan)
    assert harness.factory.requests == [("lmstudio", "fake-local")]
    assert client.chat_calls == 3 * CHAT_CALLS_PER_ROW

    outcome = api.show(start.run)
    assert outcome.state is DataRunState.SUCCEEDED and outcome.diagnostic_code is None
    assert (outcome.rows_requested, outcome.rows_written) == (3, 3)
    assert outcome.scenarios == (DataScenarioOutcome("farewell", 1, 1), DataScenarioOutcome("greeting", 2, 2))
    _assert_artifacts_match_disk(outcome, tmp_path / "run.jsonl")

    kinds = [(record.kind, record.sequence) for record in harness.observations(start.run)]
    assert [sequence for _, sequence in kinds] == list(range(len(kinds)))
    assert [kind for kind, _ in kinds].count(ObservationKind.DATA_ROW_WRITTEN) == 3
    assert [kind for kind, _ in kinds].count(ObservationKind.DATA_STAGE_GATE_EVALUATED) == 3
    assert [kind for kind, _ in kinds].count(ObservationKind.DATA_SCENARIO_COMPLETED) == 2
    payloads = [record.payload for record in harness.observations(start.run) if record.kind is ObservationKind.DATA_SCENARIO_COMPLETED]
    # scenarios execute in the plan's canonical (sorted) order
    assert [(item.scenario_ref, item.rows_written) for item in payloads] == [("farewell", 1), ("greeting", 2)]

    assert harness.durable_events(start.run) == 3 <= DATA_DURABLE_EVENT_BUDGET
    assert harness.records.read(partition=StoragePartition.EFFECTS.value, key="x") is None
    assert not harness.records.list_page(partition=StoragePartition.EFFECTS.value, prefix="", after_key=None, limit=10).records

    assert api.start(plan) == start
    assert client.chat_calls == 3 * CHAT_CALLS_PER_ROW
    page = api.list(DataListRequest(PROJECT))
    assert page.outcomes == (outcome,) and not page.truncated
    assert api.list(DataListRequest("other")).outcomes == ()
    with pytest.raises(DataOperationError) as captured:
        api.cancel(start.run, "too late")
    assert captured.value.code is DataOperationCode.CANCEL_INELIGIBLE
    assert _no_synaptic_dir(tmp_path) and (Path.cwd() / ".synaptic").exists() == cwd_synaptic


def test_backend_failure_after_some_rows_finishes_partially_succeeded_with_the_artifact(tmp_path) -> None:
    from shared.llm.exceptions import LLMConnectionError

    client = FakeLocalClient([f"text {index}" for index in range(CHAT_CALLS_PER_ROW)], failure=LLMConnectionError("boom"))
    harness = Harness(tmp_path, client)
    api = harness.api
    plan = api.plan(harness.request("partial"))
    start = api.start(plan)
    outcome = api.show(start.run)
    assert outcome.state is DataRunState.PARTIALLY_SUCCEEDED and outcome.state.terminal
    assert outcome.rows_written == 1 and outcome.diagnostic_code == "backend_unavailable"
    assert outcome.scenarios == (DataScenarioOutcome("farewell", 1, 1), DataScenarioOutcome("greeting", 2, 0))
    _assert_artifacts_match_disk(outcome, tmp_path / "partial.jsonl")
    assert "boom" not in json.dumps(outcome.to_dict())
    assert harness.durable_events(start.run) == 3


def test_backend_failure_before_any_row_finishes_failed_without_artifacts(tmp_path) -> None:
    from shared.llm.exceptions import LLMConnectionError

    harness = Harness(tmp_path, FakeLocalClient([], failure=LLMConnectionError("down")))
    start = harness.api.start(harness.api.plan(harness.request("empty")))
    outcome = harness.api.show(start.run)
    assert outcome.state is DataRunState.FAILED and outcome.rows_written == 0 and outcome.artifacts == ()
    assert outcome.diagnostic_code == "backend_unavailable"


def test_cooperative_cancel_at_a_row_boundary_finishes_cancelled_with_the_artifact(tmp_path) -> None:
    holder: dict[str, object] = {}

    def cancel_after_first_row(call_index: int) -> None:
        if call_index == CHAT_CALLS_PER_ROW:
            api: DataAPI = holder["api"]  # type: ignore[assignment]
            requested = api.cancel(holder["run"], "operator request")  # type: ignore[arg-type]
            assert requested.state is DataRunState.CANCEL_REQUESTED
            assert api.show(holder["run"]).state is DataRunState.CANCEL_REQUESTED  # type: ignore[arg-type]

    client = FakeLocalClient([f"text {index}" for index in range(3 * CHAT_CALLS_PER_ROW)], on_call=cancel_after_first_row)
    harness = Harness(tmp_path, client)
    plan = harness.api.plan(harness.request("cancelled"))
    holder.update(api=harness.api, run=data_run_ref(plan))
    start = harness.api.start(plan)
    outcome = harness.api.show(start.run)
    assert outcome.state is DataRunState.CANCELLED and outcome.state.terminal
    assert outcome.rows_written == 2 and outcome.diagnostic_code == "cancelled"
    assert client.chat_calls == 2 * CHAT_CALLS_PER_ROW
    _assert_artifacts_match_disk(outcome, tmp_path / "cancelled.jsonl")
    assert harness.durable_events(start.run) == 4 <= DATA_DURABLE_EVENT_BUDGET
    with pytest.raises(DataOperationError) as captured:
        harness.api.cancel(start.run, "again")
    assert captured.value.code is DataOperationCode.CANCEL_INELIGIBLE


def test_a_cold_composition_over_the_same_stores_cancels_a_running_run(tmp_path) -> None:
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    holder: dict[str, object] = {}

    def cancel_from_cold_host(call_index: int) -> None:
        if call_index == CHAT_CALLS_PER_ROW:
            cold = Harness(tmp_path / "cold", FakeLocalClient([]), records=records, streams=streams)
            assert cold.api.show(holder["run"]).state is DataRunState.RUNNING  # type: ignore[arg-type]
            cold.api.cancel(holder["run"], "cold host")  # type: ignore[arg-type]

    client = FakeLocalClient([f"text {index}" for index in range(3 * CHAT_CALLS_PER_ROW)], on_call=cancel_from_cold_host)
    warm = Harness(tmp_path / "warm", client, records=records, streams=streams)
    plan = warm.api.plan(warm.request("shared"))
    holder["run"] = data_run_ref(plan)
    start = warm.api.start(plan)
    outcome = warm.api.show(start.run)
    # the request lands mid-row; the in-flight row completes and the next boundary honours it
    assert outcome.state is DataRunState.CANCELLED and outcome.rows_written == 2
    cold = Harness(tmp_path / "cold2", FakeLocalClient([]), records=records, streams=streams)
    assert cold.api.show(start.run) == outcome
    assert cold.api.list(DataListRequest(PROJECT)).outcomes == (outcome,)


def test_show_and_cancel_refuse_a_missing_run(tmp_path) -> None:
    harness = Harness(tmp_path, FakeLocalClient([]))
    for verb in ("show", "cancel"):
        with pytest.raises(DataOperationError) as captured:
            getattr(harness.api, verb)(DataRunRef("data-missing", PROJECT), *(("why",) if verb == "cancel" else ()))
        assert captured.value.code is DataOperationCode.RUN_MISSING


# --- backends and plan refusals -----------------------------------------------------------


@pytest.mark.parametrize("backend", ["openrouter", "openai", "openai_responses", "anthropic"])
def test_metered_backends_are_refused_with_backend_unmetered(tmp_path, backend) -> None:
    harness = Harness(tmp_path, FakeLocalClient([]))
    plan = harness.api.plan(harness.request("paid", backend=backend))
    preflight = harness.api.preflight(plan)
    assert not preflight.ready and preflight.diagnostic_codes == ("backend_unmetered",)
    with pytest.raises(DataOperationError) as captured:
        harness.api.start(plan)
    assert captured.value.code is DataOperationCode.BACKEND_UNMETERED
    assert harness.factory.requests == [] and harness.api.list(DataListRequest(PROJECT)).outcomes == ()


def test_unknown_backend_is_refused_with_backend_unavailable(tmp_path) -> None:
    harness = Harness(tmp_path, FakeLocalClient([]))
    plan = harness.api.plan(harness.request("unknown", backend="mystery"))
    assert harness.api.preflight(plan).diagnostic_codes == ("backend_unavailable",)
    with pytest.raises(DataOperationError) as captured:
        harness.api.start(plan)
    assert captured.value.code is DataOperationCode.BACKEND_UNAVAILABLE


def test_plan_refuses_unknown_scenarios_relative_outputs_and_occupied_outputs(tmp_path) -> None:
    harness = Harness(tmp_path, FakeLocalClient([]))
    with pytest.raises(DataOperationError) as captured:
        harness.api.plan(harness.request("bad", scenarios=(DataScenarioTarget("missing", 1),)))
    assert captured.value.code is DataOperationCode.SCENARIO_INVALID
    with pytest.raises(DataOperationError) as captured:
        harness.api.plan(harness.request("bad", output_ref="relative/out.jsonl"))
    assert captured.value.code is DataOperationCode.MANIFEST_INVALID
    (tmp_path / "taken.jsonl").write_text('{"conversations": []}\n', encoding="utf-8")
    plan = harness.api.plan(harness.request("taken"))
    assert harness.api.preflight(plan).diagnostic_codes == ("output_conflict",)
    with pytest.raises(DataOperationError) as captured:
        harness.api.start(plan)
    assert captured.value.code is DataOperationCode.OUTPUT_CONFLICT
    forged = DataPlan(plan.schema_version, plan.request, plan.scenarios, plan.rows_requested, "f" * 64)
    with pytest.raises(DataOperationError) as captured:
        harness.api.start(forged)
    assert captured.value.code is DataOperationCode.MANIFEST_INVALID


# --- improve ---------------------------------------------------------------------------------


def test_improve_run_rewrites_the_input_dataset_as_one_artifact(tmp_path) -> None:
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        "".join(json.dumps({"conversations": [{"role": "user", "content": f"q{index}"}, {"role": "assistant", "content": f"a{index}"}]}) + "\n" for index in range(2)),
        encoding="utf-8",
    )
    client = FakeLocalClient([], structured=[{"data_quality_score": 0.95}] * 2)
    harness = Harness(tmp_path, client)
    request = harness.request(
        "improved", mode=DataMode.IMPROVE, scenarios=(), input_dataset_ref=str(seed), rubric_refs=("data_quality",),
    )
    plan = harness.api.plan(request)
    assert plan.scenarios == (DataScenarioTarget(str(seed), 2),) and plan.rows_requested == 2
    start = harness.api.start(plan)
    outcome = harness.api.show(start.run)
    assert outcome.state is DataRunState.SUCCEEDED and outcome.rows_written == 2
    _assert_artifacts_match_disk(outcome, tmp_path / "improved.jsonl")
    assert client.chat_calls == 0 and client.structured == []
    gates = [record.payload for record in harness.observations(start.run) if record.kind is ObservationKind.DATA_STAGE_GATE_EVALUATED]
    assert [(item.gate, item.passed) for item in gates] == [("improve", True), ("improve", True)]

    with pytest.raises(DataOperationError) as captured:
        harness.api.plan(harness.request(
            "nope", mode=DataMode.IMPROVE, scenarios=(), input_dataset_ref=str(tmp_path / "absent.jsonl"), rubric_refs=("data_quality",),
        ))
    assert captured.value.code is DataOperationCode.DATASET_MISSING
    with pytest.raises(DataOperationError) as captured:
        harness.api.plan(harness.request(
            "nope", mode=DataMode.IMPROVE, scenarios=(), input_dataset_ref=str(seed), rubric_refs=("no_such_rubric",),
        ))
    assert captured.value.code is DataOperationCode.MANIFEST_INVALID


# --- read-only dataset queries -------------------------------------------------------------


def test_datasets_pages_a_directory_with_a_settled_cursor(tmp_path) -> None:
    harness = Harness(tmp_path, FakeLocalClient([]))
    directory = tmp_path / "datasets"
    directory.mkdir()
    for name in ("c", "a", "b"):
        (directory / f"{name}.jsonl").write_text('{"conversations": []}\n' * (ord(name) - 96), encoding="utf-8")
    (directory / "a.jsonl.meta.json").write_text("{}\n", encoding="utf-8")
    (directory / "ignored.txt").write_text("x", encoding="utf-8")

    first = harness.api.datasets(DatasetListRequest(PROJECT, str(directory), limit=2))
    assert [Path(item.dataset_ref).name for item in first.datasets] == ["a.jsonl", "b.jsonl"]
    assert first.datasets[0].metadata_ref == str(directory / "a.jsonl.meta.json")
    assert first.datasets[1].metadata_ref is None
    assert first.datasets[1].size_bytes == len('{"conversations": []}\n') * 2
    assert first.truncated and first.next_cursor is not None
    second = harness.api.datasets(DatasetListRequest(PROJECT, str(directory), cursor=first.next_cursor, limit=2))
    assert [Path(item.dataset_ref).name for item in second.datasets] == ["c.jsonl"] and not second.truncated

    for cursor in ("garbage", "v1.", "v1.%%%"):
        with pytest.raises(DataOperationError) as captured:
            harness.api.datasets(DatasetListRequest(PROJECT, str(directory), cursor=cursor))
        assert captured.value.code is DataOperationCode.CURSOR_INVALID
    with pytest.raises(DataOperationError) as captured:
        harness.api.datasets(DatasetListRequest(PROJECT, str(tmp_path / "absent")))
    assert captured.value.code is DataOperationCode.DATASET_MISSING
    with pytest.raises(DataOperationError) as captured:
        harness.api.datasets(DatasetListRequest(PROJECT, "relative"))
    assert captured.value.code is DataOperationCode.MANIFEST_INVALID
    assert harness.api.list(DataListRequest(PROJECT)).outcomes == ()


def test_validate_reports_structural_findings_by_line_and_pages(tmp_path) -> None:
    harness = Harness(tmp_path, FakeLocalClient([]))
    dataset = tmp_path / "check.jsonl"
    dataset.write_text(
        "\n".join([
            json.dumps({"conversations": [{"role": "user", "content": "x"}]}),
            "not json",
            "[1]",
            json.dumps({"_meta": {"synthchat_version": "1.0.0"}}),
            json.dumps({"metadata": {}}),
            json.dumps({"conversations": []}),
            json.dumps({"conversations": [{"role": "assistant", "tool_calls": []}]}),
        ]) + "\n",
        encoding="utf-8",
    )
    first = harness.api.validate(DatasetValidateRequest(PROJECT, str(dataset), limit=4))
    assert (first.rows_checked, first.rows_valid) == (4, 1)
    assert [(item.line_number, item.code) for item in first.findings] == [
        (2, ValidationFindingCode.MALFORMED_JSON), (3, ValidationFindingCode.ROW_NOT_OBJECT),
        (4, ValidationFindingCode.LEGACY_METADATA_ROW),
    ]
    assert first.truncated
    second = harness.api.validate(DatasetValidateRequest(PROJECT, str(dataset), cursor=first.next_cursor, limit=4))
    assert (second.rows_checked, second.rows_valid) == (3, 1) and not second.truncated
    assert [(item.line_number, item.code) for item in second.findings] == [
        (5, ValidationFindingCode.CONVERSATIONS_MISSING), (6, ValidationFindingCode.CONVERSATIONS_INVALID),
    ]
    for cursor in ("garbage", "v1.", "v1.-1", "v1.abc"):
        with pytest.raises(DataOperationError) as captured:
            harness.api.validate(DatasetValidateRequest(PROJECT, str(dataset), cursor=cursor))
        assert captured.value.code is DataOperationCode.CURSOR_INVALID
    with pytest.raises(DataOperationError) as captured:
        harness.api.validate(DatasetValidateRequest(PROJECT, str(tmp_path / "absent.jsonl")))
    assert captured.value.code is DataOperationCode.DATASET_MISSING


# --- composition and import gates ----------------------------------------------------------


def test_reference_composition_accepts_data_ports_and_exports_them_lazily() -> None:
    import inspect

    from synaptic_tuner.api.v1 import reference

    assert "data" in inspect.signature(compose_reference_host).parameters
    assert reference.ReferenceDataPortsV1 is ReferenceDataPortsV1
    assert type(build_data_operations(
        records=InMemoryDurableRecordStoreV1(), streams=InMemoryDurableStreamStoreV1(), clock=Clock(),
        ports=ReferenceDataPortsV1(CONFIG_DIR, RUBRICS_DIR, RUBRICS_DIR, Factory(None)),
    )) is ReferenceDataOperationsV1
    with pytest.raises(ValueError, match="absolute"):
        ReferenceDataPortsV1(Path("relative"), RUBRICS_DIR, RUBRICS_DIR, Factory(None))
    with pytest.raises(TypeError, match="create"):
        ReferenceDataPortsV1(CONFIG_DIR, RUBRICS_DIR, RUBRICS_DIR, object())
    assert "data" not in v1._LAZY_MODULE_ATTRIBUTES.get("reference", {}) and "ReferenceDataPortsV1" not in v1.__all__


def test_reference_data_module_has_no_ambient_environment_or_process_exits() -> None:
    source = (ROOT / "synaptic_tuner/api/v1/reference/data.py").read_text(encoding="utf-8")
    facade = (ROOT / "synaptic_tuner/api/v1/data_facade.py").read_text(encoding="utf-8")
    for needle in (r"sys\.exit", "load_dotenv", r"os\.environ", "getenv", r"(?<![A-Za-z_])print\("):
        assert re.search(needle, source) is None and re.search(needle, facade) is None, needle
    assert "import SynthChat" not in facade and "from SynthChat" not in facade and "from tuner" not in facade


def test_package_import_loads_no_engine_synthchat_or_provider_modules_and_reference_data_no_synthchat() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1
blocked = ('tuner', 'sqlite3', 'modal', 'huggingface_hub', 'runpod', 'SynthChat')
first = sorted(n for n in sys.modules if n in blocked or n.startswith(tuple(b + '.' for b in blocked)))
import synaptic_tuner.api.v1.reference.data
later = ('sqlite3', 'modal', 'huggingface_hub', 'runpod', 'SynthChat')
second = sorted(n for n in sys.modules if n in later or n.startswith(tuple(b + '.' for b in later)))
print(json.dumps([first, second]))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert json.loads(completed.stdout) == [[], []]
