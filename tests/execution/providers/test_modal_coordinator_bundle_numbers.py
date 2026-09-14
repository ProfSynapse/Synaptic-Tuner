import pytest

from tuner.execution.providers.modal.coordinator_bundle import (
    _object,
    parse_workload_object,
)
from tuner.training.recipes import (
    CompiledWorkload,
    MAX_WORKLOAD_BYTES,
    canonical_json_bytes,
)


def _workload(learning_rate=0.0002):
    return {
        "schema_version": "synaptic-sft-workload/v1",
        "method": "sft",
        "entrypoint": "Trainers/sft/runtime_v1.py",
        "learning_rate": learning_rate,
    }


def test_workload_boundary_accepts_canonical_finite_float_and_reconstructs_type():
    payload = canonical_json_bytes(_workload())
    document = parse_workload_object(payload)
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
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":1e999}',
        b'{"x":0.05,"x":0.05}',
        b'{"entrypoint":"Trainers/sft/runtime_v1.py","learning_rate":0.0002,"learning_rate":0.0003,"method":"sft","schema_version":"synaptic-sft-workload/v1"}',
        canonical_json_bytes(_workload()) + b"\n",
    ],
)
def test_workload_boundary_rejects_nonfinite_duplicate_or_noncanonical_json(payload):
    with pytest.raises(ValueError):
        parse_workload_object(payload)


def test_workload_boundary_keeps_secret_rejection():
    document = _workload()
    document["api_token"] = "not-allowed"
    with pytest.raises(ValueError, match="forbidden secret field"):
        parse_workload_object(canonical_json_bytes(document))


@pytest.mark.parametrize(
    "payload", [b"", "{}", bytearray(b"{}"), b" " * (MAX_WORKLOAD_BYTES + 1)]
)
def test_workload_boundary_requires_bounded_exact_bytes(payload):
    with pytest.raises(ValueError, match="bounded nonempty bytes"):
        parse_workload_object(payload)
