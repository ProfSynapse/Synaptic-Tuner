import pytest

from tuner.execution.providers.modal.coordinator_bundle import (
    _object,
    _workload_object,
)
from tuner.training.recipes import CompiledWorkload, canonical_json_bytes


def _workload(learning_rate=0.0002):
    return {
        "schema_version": "synaptic-sft-workload/v1",
        "method": "sft",
        "entrypoint": "Trainers/sft/runtime_v1.py",
        "learning_rate": learning_rate,
    }


def test_workload_boundary_accepts_canonical_finite_float_and_reconstructs_type():
    payload = canonical_json_bytes(_workload())
    document = _workload_object(payload)
    rebuilt = CompiledWorkload(
        document["method"], document["schema_version"], document["entrypoint"], payload
    )
    assert rebuilt.document["learning_rate"] == 0.0002
    assert rebuilt.canonical_bytes == payload


def test_general_bundle_object_boundary_remains_integer_only():
    with pytest.raises(ValueError, match="non-integer number"):
        _object(canonical_json_bytes(_workload()), "non-workload")


@pytest.mark.parametrize(
    "payload",
    [
        b'{"entrypoint":"Trainers/sft/runtime_v1.py","learning_rate":NaN,"method":"sft","schema_version":"synaptic-sft-workload/v1"}',
        b'{"entrypoint":"Trainers/sft/runtime_v1.py","learning_rate":0.0002,"learning_rate":0.0003,"method":"sft","schema_version":"synaptic-sft-workload/v1"}',
        canonical_json_bytes(_workload()) + b"\n",
    ],
)
def test_workload_boundary_rejects_nonfinite_duplicate_or_noncanonical_json(payload):
    with pytest.raises(ValueError):
        _workload_object(payload)


def test_workload_boundary_keeps_secret_rejection():
    document = _workload()
    document["api_token"] = "not-allowed"
    with pytest.raises(ValueError, match="forbidden secret field"):
        _workload_object(canonical_json_bytes(document))
