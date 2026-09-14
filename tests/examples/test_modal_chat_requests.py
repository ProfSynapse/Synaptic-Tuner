"""Actual rich-compiler bridge tests for the minimal Modal chat consumer."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from examples.modal_chat.requests import (
    ModalChatRequestError,
    ModalChatTrainingRequests,
)
from synaptic_tuner.api.v1.planning import (
    ProviderPlanRef,
    TrainingPlan,
    TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.project.context import ProjectContext
from tuner.training import TrainingService, default_recipe_registry
from tuner.training.contracts import CanonicalDocument
from tuner.training.coordinator_material import derive_coordinator_material

from tests.training.test_training_service import Resolver, _execution_source


class Catalog:
    def __init__(self):
        self.values = {}

    def resolve(self, key):
        return self.values.get(key)

    def publish_if_absent(self, key, value):
        if key in self.values:
            return False
        self.values[key] = value
        return True


class Identities:
    def __init__(self):
        self.calls = []

    def allocate(self, *, project_ref, request_digest):
        self.calls.append((project_ref, request_digest))
        return "request-a", TrainingRunRef("run-service", project_ref)


def _input() -> str:
    value = {
        "schema_version": "synaptic-training-input/v1",
        "method": "sft",
        "model": {
            "ref": "example/model",
            "revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
        },
        "dataset": {"ref": "project://data.jsonl"},
        "hyperparameters": {
            "schema_version": "synaptic-sft-hyperparameters/v1",
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0002,
            "duration": {"max_steps": 1, "num_epochs": None},
            "max_seq_length": 512,
            "seed": 42,
            "save_steps": 1,
            "save_total_limit": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules": ["k_proj", "q_proj", "v_proj"],
            "use_dora": False,
            "use_rslora": False,
            "init_lora_weights": True,
            "split_dataset": False,
        },
        "artifacts": {
            "required_kinds": ["final_model", "training_lineage"],
            "retain_checkpoints": True,
        },
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _case(tmp_path):
    project = tmp_path / "project"
    engine = project / "vendor" / "engine"
    engine.mkdir(parents=True)
    recipes = default_recipe_registry()
    service = TrainingService(
        context=ProjectContext.host(engine_root=engine, project_root=project),
        resolver=Resolver(_execution_source("vendor/engine")),
        recipes=recipes,
    )
    identities = Identities()
    requests, materials = Catalog(), Catalog()
    bridge = ModalChatTrainingRequests(
        service=service,
        recipes=recipes,
        project_ref="project-a",
        identities=identities,
        request_catalog=requests,
        material_catalog=materials,
    )
    return bridge, identities, requests, materials


def test_unicode_public_input_round_trips_through_rich_document(tmp_path):
    bridge, _, _, _ = _case(tmp_path)
    document = json.loads(_input())
    document["model"]["ref"] = "example/modèle"
    raw = TrainingInputV1.from_dict(document).canonical_json()
    assert "modèle" in raw
    request = bridge.load(raw)
    resolved = bridge.resolve(request)
    plan = TrainingPlan(
        "synaptic-training-plan/v2",
        TrainingPlanBasisV1.from_resolved(resolved),
        ProviderPlanRef("a" * 64),
    )
    assert bridge.for_plan(plan) == TrainingRunRef("run-service", "project-a")
    assert request.canonical_json == raw


def _material(bridge, request_json, *, project_ref="project-a", run_id="run-service"):
    rich = bridge._service.resolve(
        bridge._service.load(CanonicalDocument(request_json))
    )
    if rich.execution_source.run_id != run_id:
        source = replace(rich.execution_source, run_id=run_id)
        workload = bridge._recipes.resolve("sft").compile(
            resolved_config=rich.resolved_config,
            execution_source=source,
        )
        rich = replace(
            rich,
            execution_source=source,
            workload=CanonicalDocument(workload.canonical_bytes.decode("utf-8")),
        )
    return derive_coordinator_material(
        rich,
        bridge._recipes,
        request_id="request-a",
        project_ref=project_ref,
        run_id=run_id,
    ).canonical_bytes


def test_actual_service_compiler_bridge_retains_one_run_identity(tmp_path):
    bridge, identities, requests, materials = _case(tmp_path)
    request = bridge.load(_input())
    resolved = bridge.resolve(request)
    basis = TrainingPlanBasisV1.from_resolved(resolved)
    plan = TrainingPlan("synaptic-training-plan/v2", basis, ProviderPlanRef("a" * 64))

    assert resolved.request_id == request.request_id == "request-a"
    assert bridge.for_plan(plan) == TrainingRunRef("run-service", "project-a")
    assert len(identities.calls) == 1
    assert tuple(requests.values) == tuple(materials.values) == ("request-a",)
    retained = next(iter(materials.values.values()))
    assert b'"entrypoint":"Trainers/sft/runtime_v1.py"' in retained


def test_cached_resolution_never_calls_rich_service_again(tmp_path):
    bridge, _, _, _ = _case(tmp_path)
    request = bridge.load(_input())
    expected = bridge.resolve(request)

    class NoResolution:
        def resolve(self, *args, **kwargs):
            raise AssertionError("cached resolution called rich resolver")

    bridge._service._resolver = NoResolution()
    assert bridge.resolve(request) == expected


def test_repeat_load_invokes_identity_policy_again_and_accepts_same_binding(tmp_path):
    bridge, identities, _, _ = _case(tmp_path)
    first = bridge.load(_input())
    second = bridge.load(_input())
    assert second == first
    assert len(identities.calls) == 2


@pytest.mark.parametrize(
    "value",
    (
        '{"method":"sft"}',
        _input() + "\n",
        _input().replace('"method":"sft"', '"method":"foreign"'),
    ),
)
def test_loader_rejects_unknown_incomplete_or_noncanonical_input(tmp_path, value):
    bridge, identities, _, _ = _case(tmp_path)
    with pytest.raises(ModalChatRequestError):
        bridge.load(value)
    assert identities.calls == []


def test_changed_or_unknown_request_cannot_resolve(tmp_path):
    bridge, _, _, _ = _case(tmp_path)
    request = bridge.load(_input())
    from synaptic_tuner.api.v1.training_facade import TrainingRequest

    with pytest.raises(ModalChatRequestError):
        bridge.resolve(TrainingRequest(request.request_id, request.project_ref, "{}"))
    with pytest.raises(ModalChatRequestError):
        bridge.resolve(
            TrainingRequest("unknown", request.project_ref, request.canonical_json)
        )


def test_rich_resolution_must_bind_preallocated_run(tmp_path):
    bridge, _, _, _ = _case(tmp_path)
    request = bridge.load(_input())
    object.__setattr__(
        bridge._service._resolver.execution_source, "run_id", "other-run"
    )
    with pytest.raises(ModalChatRequestError):
        bridge.resolve(request)


@pytest.mark.parametrize("change", ("request", "project", "run"))
def test_cached_material_binds_complete_retained_identity(tmp_path, change):
    bridge, _, _, materials = _case(tmp_path)
    request = bridge.load(_input())
    if change == "request":
        changed = json.loads(_input())
        changed["model"]["revision"] = "other-model-revision"
        request_json = json.dumps(changed, sort_keys=True, separators=(",", ":"))
        payload = _material(bridge, request_json)
    elif change == "project":
        payload = _material(bridge, _input(), project_ref="project-b")
    else:
        payload = _material(bridge, _input(), run_id="run-other")
    materials.values[request.request_id] = payload
    with pytest.raises(ModalChatRequestError):
        bridge.resolve(request)


def test_altered_retained_record_cannot_resolve_or_supply_run_identity(tmp_path):
    bridge, _, requests, _ = _case(tmp_path)
    request = bridge.load(_input())
    resolved = bridge.resolve(request)
    record = json.loads(requests.values[request.request_id])
    record["request"]["request_id"] = "request-other"
    requests.values[request.request_id] = canonical_bytes(record)
    plan = TrainingPlan(
        "synaptic-training-plan/v2",
        TrainingPlanBasisV1.from_resolved(resolved),
        ProviderPlanRef("a" * 64),
    )
    with pytest.raises(ModalChatRequestError):
        bridge.resolve(request)
    with pytest.raises(ModalChatRequestError):
        bridge.for_plan(plan)
