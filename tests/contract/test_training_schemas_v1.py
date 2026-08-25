"""Strict and disjoint training/admin-job JSON Schema v1 gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

try:
    import jsonschema
except ImportError:  # pragma: no cover - depends on the test environment
    jsonschema = None


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCHEMA_PATH = REPO_ROOT / "schemas" / "synaptic-training-v1.json"
JOB_SCHEMA_PATH = REPO_ROOT / "schemas" / "synaptic-job-v1.json"
HEX_A = "a" * 64
HEX_B = "b" * 64
IMAGE_DIGEST = f"sha256:{'c' * 64}"


def _schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(path: Path):
    if jsonschema is None:
        pytest.skip("jsonschema is not installed; validator-dependent contract case skipped")
    schema = _schema(path)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _training_document() -> dict:
    return {
        "schema_version": "synaptic.training/v1",
        "kind": "training",
        "method": "sft",
        "model": {"identifier": "org/model", "trust_remote_code": False},
        "dataset": {"identifier": "org/data", "split": "train"},
        "parameters": {"max_steps": 5, "learning_rate": 0.0002},
        "adapter": {"mode": "lora", "rank": 16, "alpha": 32},
        "execution": {
            "provider": "modal",
            "runtime_image": "registry.example/trainer@" + IMAGE_DIGEST,
            "runtime_image_digest": IMAGE_DIGEST,
            "dependency_lock_digest": HEX_B,
            "allowed_secret_refs": ["secret://hf/token"],
        },
        "artifacts": {
            "retain_checkpoints": True,
            "required_artifacts": ["training_lineage", "final_model"],
        },
    }


def _job_document() -> dict:
    return {
        "schema_version": "synaptic.job/v1",
        "kind": "job",
        "name": "admin-diagnostic",
        "source": {
            "repository": "https://example.invalid/repo.git",
            "commit": HEX_A,
            "content_digest": HEX_B,
        },
        "command": {
            "entrypoint_profile_ref": "entrypoint-profile://approved/python-module-v1",
            "arguments": [{"parameter_ref": "parameter://training/config"}],
        },
        "execution": {
            "provider": "modal",
            "image": "registry.example/approved@" + IMAGE_DIGEST,
            "image_digest": IMAGE_DIGEST,
            "cpu": 2,
            "memory_mb": 2048,
            "disk_mb": 4096,
            "process_limit": 32,
            "timeout_seconds": 300,
            "network_policy": "none",
            "secret_refs": [],
        },
        "artifacts": {"required_artifacts": ["result.json"]},
    }


def test_schema_metadata_has_exact_disjoint_identity() -> None:
    training = _schema(TRAINING_SCHEMA_PATH)
    job = _schema(JOB_SCHEMA_PATH)
    assert training["properties"]["schema_version"] == {"const": "synaptic.training/v1"}
    assert training["properties"]["kind"] == {"const": "training"}
    assert job["properties"]["schema_version"] == {"const": "synaptic.job/v1"}
    assert job["properties"]["kind"] == {"const": "job"}
    assert training["$id"] != job["$id"]


def test_canonical_documents_validate_only_against_their_own_schema() -> None:
    training_validator = _validator(TRAINING_SCHEMA_PATH)
    job_validator = _validator(JOB_SCHEMA_PATH)
    training = _training_document()
    job = _job_document()
    training_validator.validate(training)
    job_validator.validate(job)
    assert not job_validator.is_valid(training)
    assert not training_validator.is_valid(job)


@pytest.mark.parametrize(
    "path, value",
    [
        (("unknown",), True),
        (("model", "unknown"), True),
        (("parameters", "unknown"), 1),
        (("execution", "unknown"), "value"),
        (("artifacts", "unknown"), True),
    ],
)
def test_training_schema_rejects_unknown_fields_at_every_level(path, value) -> None:
    document = _training_document()
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not _validator(TRAINING_SCHEMA_PATH).is_valid(document)


@pytest.mark.parametrize(
    "path, value",
    [
        (("shell",), "bash"),
        (("setup",), ["pip install anything"]),
        (("run",), {"steps": ["python train.py"]}),
        (("execution", "command"), "python train.py"),
        (("execution", "allowed_secret_refs"), ["hf_live_literal_secret"]),
    ],
)
def test_training_schema_rejects_shell_setup_steps_and_literal_secrets(path, value) -> None:
    document = _training_document()
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not _validator(TRAINING_SCHEMA_PATH).is_valid(document)


def test_administrative_job_schema_is_explicitly_disabled_by_default() -> None:
    schema = _schema(JOB_SCHEMA_PATH)
    assert schema.get("x-synaptic-enabled-by-default") is False


@pytest.mark.parametrize(
    "command",
    [
        {"entrypoint_profile_ref": "/bin/bash", "arguments": []},
        {"entrypoint_profile_ref": "entrypoint-profile://approved/BASH", "arguments": []},
        {"entrypoint_profile_ref": "entrypoint-profile://approved/env-bash", "arguments": []},
        {"entrypoint_profile_ref": "entrypoint-profile://approved/python", "arguments": ["literal"]},
        {"entrypoint_profile_ref": "entrypoint-profile://approved/python", "arguments": ["hf_live_secret"]},
        {"entrypoint_profile_ref": "entrypoint-profile://approved/python", "arguments": [{"value": "literal"}]},
        {"entrypoint_profile_ref": "entrypoint-profile://approved/python", "arguments": [{"parameter_ref": "bad"}]},
        {"entrypoint_profile_ref": "entrypoint-profile://approved/python", "arguments": [{"parameter_ref": "parameter://x", "secret_ref": "secret://y"}]},
        {"argv": ["/bin/bash", "-c", "python approved.py"]},
    ],
)
def test_administrative_job_schema_accepts_only_profile_and_typed_reference_arguments(command):
    document = _job_document()
    document["command"] = command
    assert not _validator(JOB_SCHEMA_PATH).is_valid(document)


@pytest.mark.parametrize(
    "document_factory, schema_path, mutation",
    [
        (_training_document, TRAINING_SCHEMA_PATH, ("execution", "runtime_image_digest", "sha256:short")),
        (_training_document, TRAINING_SCHEMA_PATH, ("execution", "dependency_lock_digest", "not-a-sha")),
        (_job_document, JOB_SCHEMA_PATH, ("source", "commit", "branch-name")),
        (_job_document, JOB_SCHEMA_PATH, ("execution", "image_digest", "sha256:short")),
    ],
)
def test_digest_fields_require_canonical_sha256(document_factory, schema_path, mutation) -> None:
    document = copy.deepcopy(document_factory())
    section, field, value = mutation
    document[section][field] = value
    assert not _validator(schema_path).is_valid(document)

