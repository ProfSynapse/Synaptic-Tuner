from __future__ import annotations

import hashlib
import json

import pytest

from tuner.execution.contracts import EffectIdentity, EffectKind, ExecutionScope
from tuner.execution.operation import ModalStageTargetV1, OperationBindingV1


D = "d" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _members() -> dict[str, bytes]:
    closure = _canonical({"closure_digest": "8" * 64})
    plan = {
        "run_id": "run-1", "effect_id": "effect-1", "effect_key": "run-1",
        "provider": "modal", "account_ref": "acct", "namespace_ref": "env",
        "artifact_slot_ref": "slot", "resource_digest": D,
        "quote_digest": D, "secret_requirements_digest": D,
        "worker_closure_manifest_sha256": _sha(closure),
        "worker_closure_digest": "8" * 64,
    }
    return {
        "artifact-contract.json": _canonical({"schema_version": "artifacts/v1"}),
        "deployment.json": _canonical({"attestation_digest": D}),
        "execution-source.json": _canonical({"run_id": "run-1"}),
        "invocation-intent.json": _canonical({
            "run_id": "run-1", "invocation_nonce": "nonce",
            "interpreter": "/python", "argv": ["/python"], "cwd": "/tmp",
            "environment_digest": D,
        }),
        "log-terminal-policy.json": _canonical({"schema_version": "logs/v1"}),
        "plan.json": _canonical(plan),
        "workload.json": _canonical({"schema_version": "workload/v1"}),
        "worker-closure-manifest.json": closure,
    }


def _operation(members: dict[str, bytes]) -> OperationBindingV1:
    return OperationBindingV1.from_predecessors(
        project_ref="project", grant_ref="grant",
        effect=EffectIdentity(
            "effect-1", "run-1", EffectKind.SUBMIT,
            ExecutionScope("modal", "acct", "env"),
        ),
        stage_target=ModalStageTargetV1(
            "slot", "control", "artifacts", "operations/effect-1/output", 1, "key"
        ),
        member_documents=members,
    )


def test_operation_derives_and_round_trips_eighth_closure_predecessor() -> None:
    members = _members()
    operation = _operation(members)
    assert operation.worker_closure_manifest_digest == _sha(
        members["worker-closure-manifest.json"]
    )
    assert OperationBindingV1.from_dict(operation.to_dict()) == operation


def test_operation_requires_exact_eight_predecessors() -> None:
    members = _members()
    members.pop("worker-closure-manifest.json")
    with pytest.raises(ValueError, match="exact eight predecessor"):
        _operation(members)


def test_operation_rejects_closure_substitution_not_bound_by_plan() -> None:
    members = _members()
    members["worker-closure-manifest.json"] = _canonical({"closure_digest": "9" * 64})
    with pytest.raises(ValueError, match="one operation identity"):
        _operation(members)


def test_bound_closure_change_changes_operation_identity() -> None:
    original = _members()
    changed = dict(original)
    changed_closure = _canonical({"closure_digest": "9" * 64})
    changed["worker-closure-manifest.json"] = changed_closure
    plan = json.loads(changed["plan.json"])
    plan["worker_closure_manifest_sha256"] = _sha(changed_closure)
    plan["worker_closure_digest"] = "9" * 64
    changed["plan.json"] = _canonical(plan)
    assert _operation(original).digest != _operation(changed).digest
