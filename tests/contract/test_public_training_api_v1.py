"""Contract tests for the stable training API v1 facade."""

from __future__ import annotations

import dataclasses
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from synaptic_tuner.api.v1 import (
    ArtifactPolicy,
    CanonicalDocument,
    ExecutionGrant,
    GitCliLocalSourceInspector,
    AuthenticatedSourceEvidenceV1,
    ExecutionSourceV1,
    ProjectContext,
    ResolvedTrainingRequest,
    ResolvedTrainingComponents,
    ResourceSpec,
    RuntimeSpec,
    SourceLock,
    SourceLockBindingV1,
    TrainingAPI,
    TrainingInputV1,
    TrainingPlan,
    TrainingPreflight,
    TrainingRequest,
    TrainingRequestResolver,
    compile_training_plan_v1,
)


ROOT = Path(__file__).resolve().parents[2]


def _execution_source(
    *, commit: str = "a" * 40, configuration: dict[str, object] | None = None
) -> ExecutionSourceV1:
    project = {
        "url": "https://github.com/example/product.git",
        "commit": "9" * 40,
        "dirty": False,
        "pushed": False,
    }
    engine = {
        "url": "https://github.com/example/engine.git",
        "commit": commit,
        "dirty": False,
        "pushed": False,
        "submodule_path": "vendor/engine",
        "gitlink_commit": commit,
    }
    provisional = SourceLock.from_dict(
        {
            "schema_version": "synaptic-source-lock/v1",
            "run_id": "run-1",
            "created_at": "2026-08-25T12:00:00Z",
            "mode": "superproject",
            "sources": {"project": project, "engine": engine},
            "project": {},
            "configuration": configuration or {},
            "plugins": [],
            "inputs": [],
            "runtime": {},
            "outputs": {},
        }
    )
    roots = {
        "engine": "/workspace/engine", "project": "/workspace/project",
        "artifacts": "/workspace/run/run-1/artifacts", "state": "/workspace/run/run-1/state",
        "tracking": "/workspace/run/run-1/tracking", "cache": "/workspace/run/run-1/cache",
        "tmp": "/workspace/run/run-1/tmp",
    }
    return ExecutionSourceV1(
        run_id=provisional.run_id, created_at=provisional.created_at,
        project_source=provisional.project_source, engine_source=provisional.engine_source,
        engine_submodule_path="vendor/engine",
        source_evidence=AuthenticatedSourceEvidenceV1(
            project_url=project["url"], project_commit=project["commit"],
            engine_url=engine["url"], engine_commit=commit,
            engine_submodule_path="vendor/engine", gitlink_commit=commit,
            source_lock_binding=provisional.binding,
            issuer_ref="fake-verifier", evidence_ref="push-proof",
            audience_ref="project/run-1", challenge_nonce="source-nonce",
            verified_at="2026-08-25T12:01:00Z", expires_at="2026-08-25T12:10:00Z",
            key_ref="source-key", tag_base64="dGFn", attestation_digest="8" * 64,
        ),
        deployment_member_sha256="7" * 64, roots=roots,
        writable_capability_root="/workspace/run", python_implementation="cpython",
        python_version="3.12.7", python_executable="/usr/local/bin/python3.12",
        python_executable_digest="6" * 64, environment={
            "PATH": "/usr/local/bin", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
            "PYTHONPATH": roots["engine"], "SYNAPTIC_ENGINE_ROOT": roots["engine"],
            "SYNAPTIC_PROJECT_ROOT": roots["project"], "SYNAPTIC_ARTIFACT_ROOT": roots["artifacts"],
            "SYNAPTIC_STATE_ROOT": roots["state"], "SYNAPTIC_TRACKING_ROOT": roots["tracking"],
            "SYNAPTIC_CACHE_ROOT": roots["cache"], "SYNAPTIC_TMP_ROOT": roots["tmp"],
            "HF_HOME": roots["cache"] + "/huggingface",
            "TRANSFORMERS_CACHE": roots["cache"] + "/transformers", "WANDB_DISABLED": "true",
        },
        secret_requirements_digest="5" * 64, provider_runtime_requirements_digest="4" * 64,
    )


def _resolved(**changes: object) -> ResolvedTrainingRequest:
    values: dict[str, object] = {
        "request": TrainingRequest(CanonicalDocument.from_mapping({"method": "sft"})),
        "execution_source": _execution_source(),
        "execution_context": CanonicalDocument.from_mapping(
            {"schema_version": "synaptic-test-execution-context/v1"}
        ),
        "resolved_config": CanonicalDocument.from_mapping(
            {"model": "example/model", "learning_rate": 0.0002}
        ),
        "workload": CanonicalDocument.from_mapping(
            {"entrypoint": "train", "arguments": ["--max-steps", "1"]}
        ),
        "runtime": RuntimeSpec(
            image="example/trainer@sha256:" + "1" * 64,
            dependency_lock_digest="2" * 64,
            python_version="3.12.7",
        ),
        "resources": ResourceSpec(accelerator="a10g", accelerator_count=1),
        "artifact_policy": ArtifactPolicy(),
    }
    values.update(changes)
    return ResolvedTrainingRequest(**values)  # type: ignore[arg-type]


def _plan(**changes: object) -> TrainingPlan:
    resolved = _resolved()
    values: dict[str, object] = {
        "execution_source": resolved.execution_source,
        "execution_context": resolved.execution_context,
        "resolved_config": resolved.resolved_config,
        "workload": resolved.workload,
        "runtime": resolved.runtime,
        "resources": resolved.resources,
        "artifact_policy": resolved.artifact_policy,
    }
    values.update(changes)
    return TrainingPlan(**values)  # type: ignore[arg-type]


def _training_input() -> TrainingInputV1:
    return TrainingInputV1.from_dict(
        {
            "schema_version": "synaptic-training-input/v1",
            "method": "sft",
            "model": {
                "ref": "example/model",
                "revision": "model-revision",
                "tokenizer_revision": "tokenizer-revision",
            },
            "dataset": {"ref": "dataset://example/training"},
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
    )


def _planning_context(tmp_path: Path) -> ProjectContext:
    project = tmp_path / "product"
    engine = project / "vendor" / "engine"
    engine.mkdir(parents=True)
    return ProjectContext.host(engine_root=engine, project_root=project)


class PlanningResolver:
    def __init__(self) -> None:
        self.request: TrainingRequest | None = None
        self.context: ProjectContext | None = None
        self.resolve_calls = 0

    def resolve(self, request: TrainingRequest, *, context: ProjectContext):
        self.resolve_calls += 1
        self.request = request
        self.context = context
        return ResolvedTrainingComponents(
            execution_source=_execution_source(),
            execution_context=CanonicalDocument.from_mapping(
                {"schema_version": "synaptic-test-execution-context/v1"}
            ),
            resolved_config=CanonicalDocument.from_mapping(
                {
                    "schema_version": "synaptic-sft-config/v1",
                    "method": "sft",
                    "model": {
                        "ref": "example/model",
                        "revision": "c" * 40,
                        "tokenizer_revision": "d" * 40,
                        "load_in_4bit": False,
                    },
                    "dataset": {
                        "ref": "project://data.jsonl",
                        "revision": "e" * 40,
                    },
                    "sft": {"max_steps": 1},
                }
            ),
            runtime=RuntimeSpec(
                image="example/trainer@sha256:" + "1" * 64,
                dependency_lock_digest="2" * 64,
                python_version="3.12.7",
            ),
            resources=ResourceSpec(accelerator="a10g"),
        )


def test_training_api_has_only_the_accepted_verbs() -> None:
    verbs = {
        name
        for name, member in TrainingAPI.__dict__.items()
        if not name.startswith("_") and inspect.isfunction(member)
    }
    assert verbs == {
        "load", "resolve", "plan", "preflight", "start", "outcome", "reverify",
    }


def test_host_resolver_contract_is_public_and_structural() -> None:
    class Resolver:
        def resolve(self, request, *, context):  # pragma: no cover - contract only
            raise AssertionError

    assert isinstance(Resolver(), TrainingRequestResolver)
    assert dataclasses.is_dataclass(ResolvedTrainingComponents)
    assert callable(GitCliLocalSourceInspector().inspect)


def test_public_compile_training_plan_owns_the_complete_planning_pipeline(
    tmp_path: Path,
) -> None:
    training_input = _training_input()
    context = _planning_context(tmp_path)
    resolver = PlanningResolver()

    plan = compile_training_plan_v1(
        training_input=training_input,
        context=context,
        resolver=resolver,
    )

    assert type(plan) is TrainingPlan
    assert len(plan.fingerprint) == 64
    assert resolver.resolve_calls == 1
    assert resolver.context is context
    assert resolver.request is not None
    assert resolver.request.document == CanonicalDocument(
        training_input.canonical_json()
    )
    assert plan.workload.to_dict()["entrypoint"] == "Trainers/sft/runtime_v1.py"


def test_public_compile_training_plan_has_only_the_closed_keyword_contract() -> None:
    parameters = inspect.signature(compile_training_plan_v1).parameters
    assert tuple(parameters) == ("training_input", "context", "resolver")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )


def test_public_compile_training_plan_rejects_foreign_boundary_types(
    tmp_path: Path,
) -> None:
    training_input = _training_input()
    context = _planning_context(tmp_path)
    resolver = PlanningResolver()

    class TrainingInputSubclass(TrainingInputV1):
        __slots__ = ()

    foreign_input = TrainingInputSubclass(
        training_input.schema_version,
        training_input.method,
        training_input.model,
        training_input.dataset,
        training_input.hyperparameters,
        training_input.artifacts,
    )
    with pytest.raises(TypeError, match="exact TrainingInputV1"):
        compile_training_plan_v1(
            training_input=foreign_input,
            context=context,
            resolver=resolver,
        )

    class ProjectContextSubclass(ProjectContext):
        pass

    foreign_context = ProjectContextSubclass(
        **{
            field.name: getattr(context, field.name)
            for field in dataclasses.fields(ProjectContext)
        }
    )
    with pytest.raises(TypeError, match="exact ProjectContext"):
        compile_training_plan_v1(
            training_input=training_input,
            context=foreign_context,
            resolver=resolver,
        )

    class InvalidResolver:
        resolve = None

    with pytest.raises(TypeError, match="TrainingRequestResolver"):
        compile_training_plan_v1(
            training_input=training_input,
            context=context,
            resolver=InvalidResolver(),  # type: ignore[arg-type]
        )


def test_public_compile_training_plan_never_executes_dynamic_resolver_lookup(
    tmp_path: Path,
) -> None:
    dynamic_lookups = 0

    class DynamicResolver:
        def __getattr__(self, name: str):
            nonlocal dynamic_lookups
            dynamic_lookups += 1
            raise AssertionError(f"dynamic lookup executed for {name}")

    with pytest.raises(TypeError, match="TrainingRequestResolver"):
        compile_training_plan_v1(
            training_input=_training_input(),
            context=_planning_context(tmp_path),
            resolver=DynamicResolver(),  # type: ignore[arg-type]
        )
    assert dynamic_lookups == 0


def test_public_compile_training_plan_rejects_descriptors_without_execution(
    tmp_path: Path,
) -> None:
    descriptor_calls = 0

    class HostileDescriptor:
        def __get__(self, instance, owner):
            nonlocal descriptor_calls
            descriptor_calls += 1
            raise AssertionError("resolver descriptor executed")

    class DescriptorResolver:
        resolve = HostileDescriptor()

    with pytest.raises(TypeError, match="TrainingRequestResolver"):
        compile_training_plan_v1(
            training_input=_training_input(),
            context=_planning_context(tmp_path),
            resolver=DescriptorResolver(),  # type: ignore[arg-type]
        )
    assert descriptor_calls == 0


def test_public_compile_training_plan_rejects_callable_instance_attributes(
    tmp_path: Path,
) -> None:
    class InstanceAttributeResolver:
        pass

    resolver = InstanceAttributeResolver()
    resolver.resolve = lambda request, *, context: None
    with pytest.raises(TypeError, match="TrainingRequestResolver"):
        compile_training_plan_v1(
            training_input=_training_input(),
            context=_planning_context(tmp_path),
            resolver=resolver,  # type: ignore[arg-type]
        )


def test_public_compile_training_plan_binds_before_the_one_deliberate_call(
    tmp_path: Path,
) -> None:
    class LookupHostileResolver(PlanningResolver):
        def __getattribute__(self, name: str):
            if name == "resolve":
                raise AssertionError("caller-controlled resolve lookup executed")
            return object.__getattribute__(self, name)

        def resolve(self, request: TrainingRequest, *, context: ProjectContext):
            return super().resolve(request, context=context)

    resolver = LookupHostileResolver()
    plan = compile_training_plan_v1(
        training_input=_training_input(),
        context=_planning_context(tmp_path),
        resolver=resolver,
    )

    assert type(plan) is TrainingPlan
    assert resolver.resolve_calls == 1


def test_public_compile_training_plan_rejects_a_foreign_service_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tuner.training import TrainingService

    monkeypatch.setattr(TrainingService, "plan", lambda self, resolved: object())
    with pytest.raises(TypeError, match="exact TrainingPlan"):
        compile_training_plan_v1(
            training_input=_training_input(),
            context=_planning_context(tmp_path),
            resolver=PlanningResolver(),
        )


def test_public_compile_training_plan_eagerly_validates_the_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        TrainingPlan,
        "fingerprint",
        property(lambda self: "not-a-fingerprint"),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        compile_training_plan_v1(
            training_input=_training_input(),
            context=_planning_context(tmp_path),
            resolver=PlanningResolver(),
        )


def test_public_compile_training_plan_symbol_import_is_provider_light() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
before = set(sys.modules)
from synaptic_tuner.api.v1 import compile_training_plan_v1
after = set(sys.modules) - before
forbidden = sorted(name for name in after if name == 'tuner.training' or name.startswith(('tuner.training.', 'tuner.backends', 'tuner.execution.providers', 'docker', 'modal', 'huggingface_hub', 'runpod', 'sqlite3', 'torch', 'transformers', 'unsloth')))
print(json.dumps(forbidden))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_public_records_and_config_are_immutable() -> None:
    document = CanonicalDocument.from_mapping({"b": [2, 1], "a": {"x": True}})
    assert document.canonical_json == '{"a":{"x":true},"b":[2,1]}'

    with pytest.raises(dataclasses.FrozenInstanceError):
        document.canonical_json = "{}"  # type: ignore[misc]

    mutable = document.to_dict()
    mutable["new"] = "value"
    assert "new" not in document.to_dict()


def test_runtime_requires_exact_immutable_inputs() -> None:
    with pytest.raises(ValueError, match="pinned"):
        RuntimeSpec("example/trainer:latest", "2" * 64, "3.12.7")
    with pytest.raises(ValueError, match="dependency_lock_digest"):
        RuntimeSpec("example/trainer@sha256:" + "1" * 64, "floating", "3.12.7")


def test_public_planning_contract_rejects_a_provisional_source_lock() -> None:
    finalized = _execution_source()
    provisional = SourceLock(
        run_id=finalized.run_id,
        created_at=finalized.created_at,
        mode="superproject",
        project_source=finalized.project_source,
        engine_source=finalized.engine_source,
        project={},
        configuration={},
    )
    with pytest.raises(TypeError, match="ExecutionSourceV1"):
        _resolved(execution_source=provisional)


def test_source_lock_binding_is_public_and_exactly_embedded_in_execution_source() -> None:
    source = _execution_source()
    binding = source.source_evidence.source_lock_binding
    assert type(binding) is SourceLockBindingV1
    assert SourceLockBindingV1.from_dict(binding.to_dict()) == binding
    assert source.to_dict()["source_evidence"]["source_lock_binding"] == binding.to_dict()


@pytest.mark.parametrize(
    "key",
    [
        "training_input_digest",
        "training_contract_identity_digest",
        "training_source_sha256",
        "training_ingress_digest",
        "provider_policy_digest",
    ],
)
def test_each_host_provenance_key_changes_the_exact_training_plan_fingerprint(key: str) -> None:
    configuration={
        "training_input_digest":"1"*64,
        "training_contract_identity_digest":"2"*64,
        "training_source_sha256":"3"*64,
        "training_ingress_digest":"4"*64,
        "provider_policy_digest":"5"*64,
    }
    baseline_source=_execution_source(configuration=configuration)
    changed_configuration=dict(configuration)
    changed_configuration[key]="f"*64
    changed_source=_execution_source(configuration=changed_configuration)
    baseline_plan=_plan(execution_source=baseline_source)
    changed_plan=_plan(execution_source=changed_source)
    assert baseline_source.source_evidence.source_lock_binding != changed_source.source_evidence.source_lock_binding
    assert baseline_source.source_evidence.authenticated_payload != changed_source.source_evidence.authenticated_payload
    assert baseline_source.fingerprint != changed_source.fingerprint
    assert baseline_plan.fingerprint != changed_plan.fingerprint


@pytest.mark.parametrize(
    "replacement",
    [
        lambda plan: {"execution_source": _execution_source(commit="b" * 40)},
        lambda plan: {
            "execution_context": CanonicalDocument.from_mapping(
                {"schema_version": "synaptic-test-execution-context/v2"}
            )
        },
        lambda plan: {
            "resolved_config": CanonicalDocument.from_mapping({"learning_rate": 0.1})
        },
        lambda plan: {
            "workload": CanonicalDocument.from_mapping({"entrypoint": "different"})
        },
        lambda plan: {
            "runtime": replace(plan.runtime, dependency_lock_digest="3" * 64)
        },
        lambda plan: {"resources": replace(plan.resources, accelerator_count=2)},
        lambda plan: {
            "artifact_policy": ArtifactPolicy(required_kinds=("final_model",))
        },
    ],
)
def test_plan_fingerprint_binds_every_execution_input(replacement) -> None:
    plan = _plan()
    changed = replace(plan, **replacement(plan))

    assert plan.fingerprint == _plan().fingerprint
    assert changed.fingerprint != plan.fingerprint


def test_preflight_binds_the_exact_plan_before_start_delegation() -> None:
    plan = _plan()
    stale = _plan(resources=replace(plan.resources, accelerator_count=2))
    preflight = TrainingPreflight(
        plan_fingerprint=stale.fingerprint,
        ready=True,
        checked_at="2026-08-25T12:01:00Z",
        expires_at="2026-08-25T12:06:00Z",
    )

    class Operations:
        def start(self, plan, preflight, grant):  # pragma: no cover - must not run
            raise AssertionError("stale preflight crossed the facade")

    with pytest.raises(ValueError, match="exact training plan"):
        TrainingAPI(Operations()).start(plan, preflight, ExecutionGrant("grant-1"))


def test_failed_preflight_cannot_start() -> None:
    from synaptic_tuner.api.v1 import ErrorCode, ExecutionError

    plan = _plan()
    preflight = TrainingPreflight(
        plan_fingerprint=plan.fingerprint,
        ready=False,
        checked_at="2026-08-25T12:01:00Z",
        expires_at="2026-08-25T12:06:00Z",
        errors=(ExecutionError(ErrorCode.PREFLIGHT_FAILED, "not ready"),),
    )

    class Operations:
        def start(self, plan, preflight, grant):  # pragma: no cover - must not run
            raise AssertionError("failed preflight crossed the facade")

    with pytest.raises(ValueError, match="did not pass preflight"):
        TrainingAPI(Operations()).start(plan, preflight, ExecutionGrant("grant-1"))


def test_training_facade_delegates_each_typed_stage() -> None:
    document = CanonicalDocument.from_mapping({"method": "sft"})
    request = TrainingRequest(document)
    resolved = _resolved(request=request)
    plan = _plan()
    preflight = TrainingPreflight(
        plan_fingerprint=plan.fingerprint,
        ready=True,
        checked_at="2026-08-25T12:01:00Z",
        expires_at="2026-08-25T12:06:00Z",
    )
    calls: list[str] = []

    class Operations:
        def load(self, value):
            calls.append("load")
            assert value is document
            return request

        def resolve(self, value):
            calls.append("resolve")
            assert value is request
            return resolved

        def plan(self, value):
            calls.append("plan")
            assert value is resolved
            return plan

        def preflight(self, value):
            calls.append("preflight")
            assert value is plan
            return preflight

        def start(self, value, checked, grant):
            calls.append("start")
            assert (value, checked, grant.grant_ref) == (plan, preflight, "grant-1")
            return "submission"

        def outcome(self, value):
            calls.append("outcome")
            assert value == "submission"
            return "outcome"

    api = TrainingAPI(Operations())
    assert api.load(document) is request
    assert api.resolve(request) is resolved
    assert api.plan(resolved) is plan
    assert api.preflight(plan) is preflight
    assert api.start(plan, preflight, ExecutionGrant("grant-1")) == "submission"
    assert api.outcome("submission") == "outcome"  # type: ignore[arg-type]
    assert calls == ["load", "resolve", "plan", "preflight", "start", "outcome"]
