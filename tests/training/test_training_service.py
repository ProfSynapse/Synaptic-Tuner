from __future__ import annotations

from pathlib import Path

import pytest

from synaptic_tuner.api.v1.training import (
    CanonicalDocument,
    ResourceSpec,
    RuntimeSpec,
)
from tuner.project.context import ProjectContext
from tuner.project.execution_source import AuthenticatedSourceEvidenceV1, ExecutionSourceV1
from tuner.project.source_bundle import SourceLock
from tuner.training import (
    ResolvedTrainingComponents,
    TrainingResolutionError,
    TrainingService,
    default_recipe_registry,
)


def _execution_source(submodule_path: str) -> ExecutionSourceV1:
    project = {
        "url": "https://github.com/example/product.git",
        "commit": "a" * 40,
        "dirty": False,
        "pushed": True,
    }
    engine = {
        "url": "https://github.com/example/engine.git",
        "commit": "b" * 40,
        "dirty": False,
        "pushed": True,
        "submodule_path": submodule_path,
        "gitlink_commit": "b" * 40,
    }
    provisional = SourceLock.from_dict(
        {
            "schema_version": "synaptic-source-lock/v1",
            "run_id": "run-service",
            "created_at": "2026-08-25T12:00:00Z",
            "mode": "superproject",
            "sources": {"project": project, "engine": engine},
            "project": {},
            "configuration": {},
            "plugins": [],
            "inputs": [],
            "runtime": {},
            "outputs": {},
        }
    )
    roots = {
        "engine": "/workspace/engine", "project": "/workspace/project",
        "artifacts": "/workspace/run/run-service/artifacts",
        "state": "/workspace/run/run-service/state",
        "tracking": "/workspace/run/run-service/tracking",
        "cache": "/workspace/run/run-service/cache", "tmp": "/workspace/run/run-service/tmp",
    }
    return ExecutionSourceV1(
        run_id=provisional.run_id, created_at=provisional.created_at,
        project_source=provisional.project_source, engine_source=provisional.engine_source,
        engine_submodule_path=submodule_path,
        source_evidence=AuthenticatedSourceEvidenceV1(
            project_url=project["url"], project_commit=project["commit"],
            engine_url=engine["url"], engine_commit=engine["commit"],
            engine_submodule_path=submodule_path, gitlink_commit=engine["commit"],
            issuer_ref="fake-verifier", evidence_ref="push-proof",
            audience_ref="project/run-1", challenge_nonce="source-nonce",
            verified_at="2026-08-25T12:01:00Z", expires_at="2026-08-25T12:10:00Z",
            key_ref="source-key", tag_base64="dGFn", attestation_digest="9" * 64,
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


def _resolved_config() -> CanonicalDocument:
    return CanonicalDocument.from_mapping(
        {
            "schema_version": "synaptic-sft-config/v1",
            "method": "sft",
            "model": {
                "ref": "example/model",
                "revision": "c" * 40,
                "tokenizer_revision": "d" * 40,
                "load_in_4bit": False,
            },
            "dataset": {"ref": "project://data.jsonl", "revision": "e" * 40},
            "sft": {"max_steps": 1},
        }
    )


class Resolver:
    def __init__(self, execution_source: ExecutionSourceV1) -> None:
        self.execution_source = execution_source

    def resolve(self, request, *, context):
        return ResolvedTrainingComponents(
            execution_source=self.execution_source,
            execution_context=CanonicalDocument.from_mapping(
                {"schema_version": "synaptic-test-execution-context/v1"}
            ),
            resolved_config=_resolved_config(),
            runtime=RuntimeSpec(
                "example/trainer@sha256:" + "1" * 64,
                "2" * 64,
                "3.12.7",
            ),
            resources=ResourceSpec("gpu"),
        )


def test_service_accepts_an_arbitrary_locked_submodule_path(tmp_path: Path) -> None:
    project = tmp_path / "product"
    engine = project / "vendor" / "ml" / "training engine"
    engine.mkdir(parents=True)
    context = ProjectContext.host(engine_root=engine, project_root=project)
    service = TrainingService(
        context=context,
        resolver=Resolver(_execution_source("vendor/ml/training engine")),
        recipes=default_recipe_registry(),
    )
    request = service.load(CanonicalDocument.from_mapping({"method": "sft"}))
    resolved = service.resolve(request)
    plan = service.plan(resolved)

    assert plan.workload.to_dict()["execution_source"]["topology"][
        "engine_submodule_path"
    ] == "vendor/ml/training engine"
    assert plan.workload.to_dict()["entrypoint"] == "Trainers/sft/runtime_v1.py"


def test_service_rejects_context_source_topology_drift(tmp_path: Path) -> None:
    project = tmp_path / "product"
    engine = project / "actual" / "engine"
    engine.mkdir(parents=True)
    service = TrainingService(
        context=ProjectContext.host(engine_root=engine, project_root=project),
        resolver=Resolver(_execution_source("different/engine")),
        recipes=default_recipe_registry(),
    )
    request = service.load(CanonicalDocument.from_mapping({"method": "sft"}))

    with pytest.raises(TrainingResolutionError, match="submodule path"):
        service.resolve(request)
