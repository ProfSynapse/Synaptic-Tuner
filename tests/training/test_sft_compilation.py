from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
from jsonschema.validators import validator_for
from referencing import Registry, Resource

from synaptic_tuner.api.v1.training import CanonicalDocument
from tuner.project.execution_source import (
    AuthenticatedSourceEvidenceV1,
    ExecutionSourceV1,
)
from tuner.project.source_bundle import SourceLock
from tuner.training.methods.sft import SFTRecipe, compile_sft_workload


_ROOT = Path(__file__).parents[2]


def _workload_validator():
    schema = json.loads(
        (_ROOT / "schemas" / "synaptic-sft-workload-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    source_schema = json.loads(
        (_ROOT / "schemas" / "synaptic-execution-source-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator_type = validator_for(schema)
    validator_type.check_schema(schema)
    registry = Registry().with_resource(
        source_schema["$id"], Resource.from_contents(source_schema)
    )
    return validator_type(schema, registry=registry)


def _execution_source(
    *, engine_commit: str = "a" * 40, source_configuration_digest: str = "2" * 64
) -> ExecutionSourceV1:
    project = {
        "url": "https://github.com/example/product.git",
        "commit": "9" * 40,
        "dirty": False,
        "pushed": False,
    }
    engine = {
        "url": "https://github.com/example/training-engine.git",
        "commit": engine_commit,
        "dirty": False,
        "pushed": False,
        "submodule_path": "vendor/training-engine",
        "gitlink_commit": engine_commit,
    }
    provisional = SourceLock.from_dict(
        {
            "schema_version": "synaptic-source-lock/v1",
            "run_id": "run-1",
            "created_at": "2026-08-25T12:00:00Z",
            "mode": "superproject",
            "sources": {"project": project, "engine": engine},
            "project": {
                "manifest_uri": "project://synaptic.yaml",
                "manifest_sha256": "1" * 64,
                "engine_requires": "training-engine==1",
            },
            "configuration": {
                "resolved_uri": "project://resolved-config.json",
                "resolved_sha256": source_configuration_digest,
                "documents": [],
            },
            "plugins": [],
            "inputs": [],
            "runtime": {},
            "outputs": {},
        }
    )
    pushed = AuthenticatedSourceEvidenceV1(
        project_url=project["url"], project_commit=project["commit"],
        engine_url=engine["url"], engine_commit=engine_commit,
        engine_submodule_path="vendor/training-engine", gitlink_commit=engine_commit,
        source_lock_binding=provisional.binding,
        issuer_ref="fake-verifier", evidence_ref="push-proof",
        audience_ref="project/run-1", challenge_nonce="source-nonce",
        verified_at="2026-08-25T12:01:00Z", expires_at="2026-08-25T12:10:00Z",
        key_ref="source-key", tag_base64="dGFn",
        attestation_digest="8" * 64,
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
        engine_submodule_path="vendor/training-engine", source_evidence=pushed,
        deployment_member_sha256="7" * 64, roots=roots,
        writable_capability_root="/workspace/run", python_implementation="cpython",
        python_version="3.12.3", python_executable="/usr/local/bin/python3.12",
        python_executable_digest="6" * 64,
        environment={
            "PATH": "/usr/local/bin", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
            "PYTHONPATH": roots["engine"], "SYNAPTIC_ENGINE_ROOT": roots["engine"],
            "SYNAPTIC_PROJECT_ROOT": roots["project"], "SYNAPTIC_ARTIFACT_ROOT": roots["artifacts"],
            "SYNAPTIC_STATE_ROOT": roots["state"], "SYNAPTIC_TRACKING_ROOT": roots["tracking"],
            "SYNAPTIC_CACHE_ROOT": roots["cache"], "SYNAPTIC_TMP_ROOT": roots["tmp"],
            "HF_HOME": roots["cache"] + "/huggingface",
            "TRANSFORMERS_CACHE": roots["cache"] + "/transformers", "WANDB_DISABLED": "true",
        },
        secret_requirements_digest="5" * 64,
        provider_runtime_requirements_digest="4" * 64,
    )


def _config(*, model_revision: str = "b" * 40) -> CanonicalDocument:
    return CanonicalDocument.from_mapping(
        {
            "schema_version": "synaptic-sft-config/v1",
            "method": "sft",
            "model": {
                "ref": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "revision": model_revision,
                "tokenizer_revision": "c" * 40,
                "load_in_4bit": False,
            },
            "dataset": {
                "ref": "project://data/training.jsonl",
                "revision": "d" * 40,
                "content_digest": "e" * 64,
                "split": "train",
                "format": "configured_chat_rows/v2",
            },
            "sft": {
                "learning_rate": "0.0002",
                "max_steps": 1,
                "custom_row_adapter": "project_plugin.normalize/v3",
            },
        }
    )


def test_sft_workload_is_canonical_and_deterministic() -> None:
    first = compile_sft_workload(
        resolved_config=_config(), execution_source=_execution_source()
    )
    reordered = CanonicalDocument.from_mapping(
        dict(reversed(tuple(_config().to_dict().items())))
    )
    second = SFTRecipe().compile(
        resolved_config=reordered,
        execution_source=_execution_source(),
    )

    assert first.canonical_bytes == second.canonical_bytes
    assert first.fingerprint == second.fingerprint
    assert first.document["configuration"]["document"]["dataset"]["format"] == (
        "configured_chat_rows/v2"
    )
    requirements = first.document["runtime_requirements"]
    assert requirements["schema_version"] == "synaptic-sft-runtime-requirements/v1"
    assert requirements["trainer_projection_schema"] == "synaptic-sft-trainer-projection/v1"
    assert "python_executable" not in json.dumps(requirements)


def test_fingerprint_binds_model_and_source_revisions() -> None:
    baseline = compile_sft_workload(
        resolved_config=_config(), execution_source=_execution_source()
    )
    changed_model = compile_sft_workload(
        resolved_config=_config(model_revision="f" * 40),
        execution_source=_execution_source(),
    )
    changed_source = compile_sft_workload(
        resolved_config=_config(),
        execution_source=_execution_source(engine_commit="9" * 40),
    )
    changed_source_configuration = compile_sft_workload(
        resolved_config=_config(),
        execution_source=_execution_source(source_configuration_digest="3" * 64),
    )

    assert baseline.fingerprint != changed_model.fingerprint
    assert baseline.fingerprint != changed_source.fingerprint
    assert baseline.fingerprint != changed_source_configuration.fingerprint


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_resolved_sft_config_requires_explicit_boolean_quantization(value) -> None:
    config = _config().to_dict()
    if value is None:
        config["model"].pop("load_in_4bit")
    else:
        config["model"]["load_in_4bit"] = value
    with pytest.raises((TypeError, ValueError), match="load_in_4bit"):
        compile_sft_workload(
            resolved_config=CanonicalDocument.from_mapping(config),
            execution_source=_execution_source(),
        )


@pytest.mark.parametrize("value", [True, False, "gaussian", "loftq", "corda"])
def test_resolved_sft_config_accepts_semantic_lora_initializers(value) -> None:
    config = _config().to_dict()
    config["sft"]["init_lora_weights"] = value
    workload = compile_sft_workload(
        resolved_config=CanonicalDocument.from_mapping(config),
        execution_source=_execution_source(),
    )
    assert workload.document["configuration"]["document"]["sft"][
        "init_lora_weights"
    ] == value


@pytest.mark.parametrize("value", ["true", "false", "eva", 1, None])
def test_resolved_sft_config_rejects_noncanonical_lora_initializers(value) -> None:
    config = _config().to_dict()
    config["sft"]["init_lora_weights"] = value
    with pytest.raises((TypeError, ValueError), match="init_lora_weights"):
        compile_sft_workload(
            resolved_config=CanonicalDocument.from_mapping(config),
            execution_source=_execution_source(),
        )


@pytest.mark.parametrize(
    ("section", "field"),
    (("model", "revision"), ("model", "tokenizer_revision"), ("dataset", "revision")),
)
def test_all_external_resource_revisions_must_be_exact(section: str, field: str) -> None:
    value = _config().to_dict()
    value[section][field] = "main"
    with pytest.raises(ValueError, match="exact lowercase"):
        compile_sft_workload(
            resolved_config=CanonicalDocument.from_mapping(value),
            execution_source=_execution_source(),
        )


def test_compiled_workload_is_immutable() -> None:
    workload = compile_sft_workload(
        resolved_config=_config(), execution_source=_execution_source()
    )
    changed = workload.document
    changed["method"] = "kto"
    assert workload.document["method"] == "sft"
    with pytest.raises(ValueError, match="identity"):
        replace(workload, method="kto")


def test_workload_schema_accepts_each_required_artifact_role_in_any_order() -> None:
    document = compile_sft_workload(
        resolved_config=_config(), execution_source=_execution_source()
    ).document
    validator = _workload_validator()
    validator.validate(document)
    document["artifacts"]["requirements"].reverse()
    validator.validate(document)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "substituted"])
def test_workload_schema_rejects_nonexact_artifact_role_sets(mutation: str) -> None:
    document = compile_sft_workload(
        resolved_config=_config(), execution_source=_execution_source()
    ).document
    requirements = document["artifacts"]["requirements"]
    if mutation == "duplicate":
        requirements[-1] = dict(requirements[0])
    elif mutation == "missing":
        requirements.pop()
    else:
        requirements[-1] = {
            "role": "debug_dump",
            "minimum": 1,
            "maximum": 1,
        }

    assert list(_workload_validator().iter_errors(document))
