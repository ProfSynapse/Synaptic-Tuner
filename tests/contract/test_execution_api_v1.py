"""Frozen private-staging execution and lifecycle contract gates."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import get_type_hints

import pytest

import synaptic_tuner._next_api_v1 as next_api_v1
from synaptic_tuner._next_api_v1 import execution, jobs
from synaptic_tuner._next_api_v1.execution import (
    AccessContext,
    ArtifactRef,
    ArtifactVerificationState,
    AuthorizationRequirement,
    CancelResult,
    ExecutionGrant,
    LogCursor,
    LogEntry,
    LogPage,
    ProviderDescriptor,
    RunRef,
    RunState,
    RunStatus,
)
from synaptic_tuner._next_api_v1.jobs import JobAPI


EXPECTED_EXECUTION_EXPORTS = {
    "AccessContext",
    "ArtifactRef",
    "ArtifactVerificationState",
    "AuthorizationRequirement",
    "CancelResult",
    "ExecutionGrant",
    "LogCursor",
    "LogEntry",
    "LogPage",
    "ProviderDescriptor",
    "RunRef",
    "RunState",
    "RunStatus",
}


def _run() -> RunRef:
    return RunRef(run_id="run-001", project_ref="project://alpha")


def test_public_vocabulary_is_exact_and_contains_no_legacy_cloud_training() -> None:
    assert set(execution.__all__) == EXPECTED_EXECUTION_EXPORTS
    assert jobs.__all__ == ["JobAPI"]
    assert not any(name.startswith("CloudTraining") for name in next_api_v1.__all__)
    assert not any(name.startswith("CloudTraining") for name in dir(next_api_v1))


@pytest.mark.parametrize("module", [execution, jobs])
def test_execution_and_job_contract_modules_are_stdlib_import_light(module) -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert roots <= {"__future__", "dataclasses", "datetime", "enum", "math", "typing"}


def test_job_api_is_lifecycle_only_and_requires_access_plus_owned_run() -> None:
    assert [name for name in JobAPI.__dict__ if not name.startswith("_")] == [
        "list",
        "show",
        "logs",
        "cancel",
    ]
    for name in ("show", "logs", "cancel"):
        signature = inspect.signature(getattr(JobAPI, name))
        assert list(signature.parameters)[:3] == ["self", "access", "run"]
        hints = get_type_hints(getattr(JobAPI, name))
        assert hints["access"] is AccessContext
        assert hints["run"] is RunRef
    list_hints = get_type_hints(JobAPI.list)
    assert list_hints["access"] is AccessContext
    assert inspect.signature(JobAPI.list).parameters["cursor"].kind is inspect.Parameter.KEYWORD_ONLY


def test_execution_grant_is_an_opaque_single_field_handle() -> None:
    assert [item.name for item in fields(ExecutionGrant)] == ["grant_ref"]
    assert ExecutionGrant("grant://opaque").to_dict() == {"grant_ref": "grant://opaque"}


def test_lifecycle_values_are_frozen_deeply_normalized_and_json_safe() -> None:
    run = _run()
    entry = LogEntry(0, "2026-08-25T12:00:00-04:00", "INFO", "started", "ok")
    page = LogPage(run=run, entries=[entry], next_cursor=LogCursor("next"))
    provider = ProviderDescriptor(
        provider="MODAL",
        display_name="Modal",
        supported_methods=["SFT", "KTO"],
        available=True,
    )
    assert page.entries == (entry,)
    assert provider.supported_methods == ("sft", "kto")
    assert json.loads(json.dumps(page.to_dict())) == page.to_dict()
    assert page.to_dict()["entries"][0]["timestamp"] == "2026-08-25T16:00:00Z"
    with pytest.raises(FrozenInstanceError):
        run.run_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "constructor, kwargs",
    [
        (ArtifactRef, {"artifact_id": "a", "run": "run", "kind": "model"}),
        (
            RunStatus,
            {
                "run": "run",
                "state": RunState.RUNNING,
                "artifact_state": ArtifactVerificationState.PENDING,
                "updated_at": "2026-08-25T12:00:00Z",
            },
        ),
        (LogPage, {"run": _run(), "entries": ["not-an-entry"]}),
        (CancelResult, {"run": _run(), "state": "cancelled", "accepted": True}),
    ],
)
def test_lifecycle_values_reject_invalid_nested_types(constructor, kwargs) -> None:
    with pytest.raises(TypeError):
        constructor(**kwargs)


def test_authorization_requirement_has_stable_json_safe_serialization() -> None:
    requirement = AuthorizationRequirement(
        operation="training.start",
        paid_effect=True,
        maximum_cost_minor_units=25,
        currency="usd",
    )
    assert requirement.to_dict() == {
        "operation": "training.start",
        "paid_effect": True,
        "maximum_cost_minor_units": 25,
        "currency": "USD",
    }

