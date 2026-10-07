"""Regression tests for engine-owned rich compilation contracts."""

from dataclasses import replace

import pytest

from tuner.training.contracts import (
    AcceleratorDeviceRequestV1,
    ArtifactPolicy,
    CanonicalDocument,
    ResourceSpec,
    RuntimeSpec,
    TrainingPlan,
    TrainingRequestResolver,
    _safe_training_request_resolver,
)
from tests.training.test_training_service import _execution_source


def _plan() -> TrainingPlan:
    document = CanonicalDocument.from_mapping
    return TrainingPlan(
        execution_source=_execution_source("vendor/engine"),
        execution_context=document({"schema_version": "context/v1"}),
        resolved_config=document({"method": "sft", "value": 1}),
        workload=document({"schema_version": "workload/v1"}),
        runtime=RuntimeSpec("registry/image@sha256:" + "1" * 64, "2" * 64, "3.12.7"),
        resources=ResourceSpec("a100", 1, 3600),
        artifact_policy=ArtifactPolicy(("training_lineage", "final_model"), True),
    )


def test_canonical_document_and_plan_fingerprint_are_stable() -> None:
    assert CanonicalDocument('{"b":2,"a":1}').canonical_json == '{"a":1,"b":2}'
    plan = _plan()
    assert plan.fingerprint == _plan().fingerprint
    assert replace(plan, resources=ResourceSpec("a100", 2, 3600)).fingerprint != plan.fingerprint


def test_extracted_plan_preserves_the_existing_digest_domain_exactly() -> None:
    # Measured against the pre-cutover rich implementation at f00cef35,
    # then independently matched to the moved definition on 2026-09-09.
    # The removed public implementation is not a compatibility dependency.
    assert _plan().fingerprint == (
        "00669ed117fa78411000f73296b7800ee7b217e8e0877f6a4953f2ffc8c96f1c"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RuntimeSpec("image:latest", "2" * 64, "3.12.7"),
        lambda: RuntimeSpec("registry/image@sha256:" + "1" * 64, "X" * 64, "3.12.7"),
        lambda: ResourceSpec("gpu", True, 1),
        lambda: ArtifactPolicy(("final_model", "final_model"), True),
        lambda: AcceleratorDeviceRequestV1("cuda", (1, 0), ("sm80",)),
    ],
)
def test_contracts_preserve_closed_validation(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_accelerator_digest_is_canonical() -> None:
    value = AcceleratorDeviceRequestV1("cuda", (0,), ("sm80",))
    assert len(value.accelerator_device_request_digest) == 64
    assert value.to_dict() == {"kind": "cuda", "device_indices": [0], "capabilities": ["sm80"]}


def test_static_resolver_binding_ignores_dynamic_attribute_interception() -> None:
    class Resolver:
        def __init__(self) -> None:
            self.lookups = 0

        def __getattribute__(self, name):
            if name == "resolve":
                object.__setattr__(self, "lookups", object.__getattribute__(self, "lookups") + 1)
                raise AssertionError("dynamic lookup crossed resolver boundary")
            return object.__getattribute__(self, name)

        def resolve(self, request, *, context):
            return request, context

    resolver = Resolver()
    safe = _safe_training_request_resolver(resolver)
    assert isinstance(safe, TrainingRequestResolver)
    assert safe.resolve("request", context="context") == ("request", "context")
    assert resolver.lookups == 0


def test_static_resolver_rejects_callable_instance_member() -> None:
    class Resolver:
        resolve = object()

    with pytest.raises(TypeError, match="TrainingRequestResolver"):
        _safe_training_request_resolver(Resolver())
