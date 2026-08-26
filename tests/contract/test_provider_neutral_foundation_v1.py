"""B1 proofs for the staged provider-neutral contract foundation."""

from __future__ import annotations

import ast
import inspect
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from synaptic_tuner.api.v1.artifacts_facade import (
    ArtifactsAPI, PublicationRef, PublicationResult, PublicationState,
)
from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1, ProviderPlanRef, ResolvedTrainingRequest, TrainingPlan,
    TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, TrainingRunState, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunOutcome, RunsAPI
from synaptic_tuner.api.v1.training_facade import (
    AuthorizationRequirement, TrainingAPI, TrainingPreflight, TrainingStart,
)


ROOT = Path(__file__).resolve().parents[2]
DIGESTS = tuple(character * 64 for character in "123456789abcdef")


def _resolved(**changes: object) -> ResolvedTrainingRequest:
    values = dict(
        schema_version="synaptic-resolved-training-request/v1", request_id="request-1",
        project_ref="project-1", source_digest=DIGESTS[0],
        resolved_config_digest=DIGESTS[1], workload_digest=DIGESTS[2],
        runtime_digest=DIGESTS[3], artifact_policy_digest=DIGESTS[4],
    )
    values.update(changes)
    return ResolvedTrainingRequest(**values)  # type: ignore[arg-type]


def _descriptor(**changes: object) -> ProviderDescriptor:
    values = dict(
        schema_version="synaptic-provider-descriptor/v1", provider_id="docker",
        display_name="Local Docker", implementation_version="1.0.0",
        capabilities=ProviderCapabilities(True, True, True, True, True, False),
    )
    values.update(changes)
    return ProviderDescriptor(**values)  # type: ignore[arg-type]


def _plan() -> tuple[TrainingPlanBasisV1, ProviderPlanContextV1, TrainingPlan]:
    basis = TrainingPlanBasisV1.from_resolved(_resolved())
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", ProviderRef("docker", "local-default"),
        basis.basis_digest, _descriptor().descriptor_digest, DIGESTS[5],
    )
    plan = TrainingPlan(
        "synaptic-training-plan/v2", basis,
        ProviderPlanRef(context.provider_context_digest),
    )
    return basis, context, plan


def _verbs(api_type: type) -> set[str]:
    return {
        name for name, member in api_type.__dict__.items()
        if not name.startswith("_") and inspect.isfunction(member)
    }


def test_semantic_api_ownership_is_exact() -> None:
    assert _verbs(TrainingAPI) == {"load", "resolve", "plan", "preflight", "start"}
    assert _verbs(RunsAPI) == {
        "list", "show", "outcome", "logs", "cancel", "reconcile", "verify",
        "reverify", "artifacts",
    }
    assert _verbs(ArtifactsAPI) == {"destinations", "publications", "publish", "verify"}
    assert not hasattr(TrainingAPI, "outcome")
    assert not hasattr(TrainingAPI, "publish")
    assert not hasattr(ArtifactsAPI, "reverify")


@pytest.mark.parametrize(
    ("schema_name", "document"),
    [
        ("synaptic-provider-descriptor-v1.schema.json", lambda: _descriptor().to_dict()),
        ("synaptic-resolved-training-request-v1.schema.json", lambda: _resolved().to_dict()),
        ("synaptic-provider-plan-context-v1.schema.json", lambda: _plan()[1].to_dict()),
        ("synaptic-training-plan-v2.schema.json", lambda: _plan()[2].to_dict()),
        (
            "synaptic-run-outcome-v1.schema.json",
            lambda: RunOutcome(
                "synaptic-run-outcome/v1", TrainingRunRef("run-1", "project-1"),
                TrainingRunState.SUCCEEDED,
            ).to_dict(),
        ),
        (
            "synaptic-publication-result-v1.schema.json",
            lambda: PublicationResult(
                "synaptic-publication-result/v1",
                PublicationRef("publication-1", "local/default"),
                PublicationState.VERIFIED,
            ).to_dict(),
        ),
        (
            "synaptic-training-preflight-v1.schema.json",
            lambda: TrainingPreflight(
                _plan()[2].plan_fingerprint, True, "2026-08-26T12:00:00Z",
                "2026-08-26T12:05:00Z",
                (AuthorizationRequirement("training.start", True, 125.0, "USD"),),
            ).to_dict(),
        ),
    ],
)
def test_closed_schemas_accept_their_canonical_python_documents(schema_name, document) -> None:
    schema = json.loads((ROOT / "schemas" / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(document(), schema)


def test_pre_b1_formal_exports_are_frozen_exactly() -> None:
    import synaptic_tuner.api.v1 as api

    baseline = json.loads(
        (ROOT / "tests/contract/fixtures/api_v1_formal_exports_pre_b1.json").read_text(
            encoding="utf-8"
        )
    )
    assert api.__all__ == baseline
    assert "EventCode" not in api.__all__
    assert "EventCode" in dir(api)
    assert api.EventCode.__name__ == "EventCode"
    assert "ProviderDescriptor" not in api.__all__
    with pytest.raises(AttributeError):
        getattr(api, "DefinitelyNotAPublicAttribute")

    namespace: dict[str, object] = {}
    exec("from synaptic_tuner.api.v1 import *", namespace)
    assert sorted(name for name in namespace if not name.startswith("__")) == sorted(baseline)


class _Clock:
    def __init__(self, value: str) -> None:
        self.value = value

    def now(self) -> str:
        return self.value


def _preflight(plan: TrainingPlan, **changes: object) -> TrainingPreflight:
    values: dict[str, object] = {
        "plan_fingerprint": plan.plan_fingerprint,
        "ready": True,
        "checked_at": "2026-08-26T12:00:00Z",
        "expires_at": "2026-08-26T12:05:00Z",
        "authorization": (AuthorizationRequirement("training.start", True, 125, "USD"),),
        "diagnostic_codes": (),
    }
    values.update(changes)
    return TrainingPreflight(**values)  # type: ignore[arg-type]


class _TrainingOperations:
    def __init__(self, preflight: TrainingPreflight) -> None:
        self.preflight_result = preflight
        self.start_calls = 0

    def preflight(self, plan):
        return self.preflight_result

    def start(self, plan, preflight):
        self.start_calls += 1
        return TrainingStart(TrainingRunRef("run-1", plan.basis.project_ref), True)


def test_preflight_result_and_start_reject_cross_plan_reuse() -> None:
    basis, _, plan_a = _plan()
    plan_b = replace(plan_a, basis=replace(basis, workload_digest=DIGESTS[9]))
    preflight_a = _preflight(plan_a)
    operations = _TrainingOperations(preflight_a)
    api = TrainingAPI(operations, clock=_Clock("2026-08-26T12:01:00Z"))

    with pytest.raises(ValueError, match="exact training plan"):
        api.preflight(plan_b)
    with pytest.raises(ValueError, match="exact training plan"):
        api.start(plan_b, preflight_a)
    assert operations.start_calls == 0


def test_plan_mutation_after_preflight_fails_closed() -> None:
    basis, _, plan = _plan()
    preflight = _preflight(plan)
    mutated = replace(plan, basis=replace(basis, runtime_digest=DIGESTS[10]))
    operations = _TrainingOperations(preflight)
    api = TrainingAPI(operations, clock=_Clock("2026-08-26T12:01:00Z"))
    with pytest.raises(ValueError, match="exact training plan"):
        api.start(mutated, preflight)
    assert operations.start_calls == 0


def test_malformed_and_not_ready_preflights_fail_closed() -> None:
    _, _, plan = _plan()
    with pytest.raises(ValueError, match="plan_fingerprint"):
        _preflight(plan, plan_fingerprint="not-a-digest")
    not_ready = _preflight(plan, ready=False, diagnostic_codes=("capacity_unavailable",))
    operations = _TrainingOperations(not_ready)
    api = TrainingAPI(operations, clock=_Clock("2026-08-26T12:01:00Z"))
    with pytest.raises(ValueError, match="did not pass preflight"):
        api.start(plan, not_ready)
    assert operations.start_calls == 0


def test_preflight_expiry_is_checked_on_return_and_rechecked_before_start() -> None:
    _, _, plan = _plan()
    preflight = _preflight(plan)
    operations = _TrainingOperations(preflight)
    clock = _Clock("2026-08-26T12:01:00Z")
    api = TrainingAPI(operations, clock=clock)
    assert api.preflight(plan) is preflight

    clock.value = "2026-08-26T12:05:00Z"
    with pytest.raises(ValueError, match="expired"):
        api.start(plan, preflight)
    assert operations.start_calls == 0

    with pytest.raises(ValueError, match="expired"):
        api.preflight(plan)


@pytest.mark.parametrize(
    ("checked_at", "expires_at"),
    [
        ("2026-08-26T12:00:00", "2026-08-26T12:05:00Z"),
        ("2026-08-26T12:00:00Z", "2026-08-26T12:00:00Z"),
        ("malformed", "2026-08-26T12:05:00Z"),
    ],
)
def test_preflight_rejects_malformed_or_non_increasing_validity_windows(checked_at, expires_at) -> None:
    _, _, plan = _plan()
    with pytest.raises(ValueError):
        _preflight(plan, checked_at=checked_at, expires_at=expires_at)


def test_generic_authorization_quote_is_closed_canonical_and_round_trips() -> None:
    _, _, plan = _plan()
    preflight = _preflight(
        plan,
        authorization=(AuthorizationRequirement("training.start", True, 125.0, "USD"),),
    )
    assert preflight.authorization[0].maximum_cost_minor_units == 125
    document = preflight.to_dict()
    schema = json.loads(
        (ROOT / "schemas/synaptic-training-preflight-v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(document, schema, format_checker=jsonschema.FormatChecker())
    assert TrainingPreflight.from_dict(document) == preflight
    with pytest.raises(ValueError, match="supplied together"):
        AuthorizationRequirement("training.start", True, 125, None)
    with pytest.raises(ValueError, match="uppercase"):
        AuthorizationRequirement("training.start", True, 125, "usd")


def test_authorization_order_is_canonical_in_memory_and_on_wire() -> None:
    _, _, plan = _plan()
    publish = AuthorizationRequirement("artifact.publish", False)
    start = AuthorizationRequirement("training.start", True, 125, "USD")
    reverse = _preflight(plan, authorization=(start, publish))
    forward = _preflight(plan, authorization=(publish, start))

    assert tuple(item.operation for item in reverse.authorization) == (
        "artifact.publish", "training.start"
    )
    assert reverse == forward
    assert reverse.to_dict() == forward.to_dict()
    reverse_bytes = json.dumps(
        reverse.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    forward_bytes = json.dumps(
        forward.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert reverse_bytes == forward_bytes
    assert TrainingPreflight.from_dict(reverse.to_dict()) == reverse

    schema = json.loads(
        (ROOT / "schemas/synaptic-training-preflight-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(
        reverse.to_dict(), schema, format_checker=jsonschema.FormatChecker()
    )


@pytest.mark.parametrize(
    "currency",
    ["ÄBC", "usd", "Usd", "U\tD", " USD", "USD ", "U1D", "US", "USDD"],
)
def test_currency_python_and_schema_reject_the_same_non_ascii_or_noncanonical_values(
    currency: str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        AuthorizationRequirement("training.start", True, 125, currency)

    _, _, plan = _plan()
    document = _preflight(plan).to_dict()
    document["authorization"]["training.start"]["currency"] = currency  # type: ignore[index]
    schema = json.loads(
        (ROOT / "schemas/synaptic-training-preflight-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, schema, format_checker=jsonschema.FormatChecker())


@pytest.mark.parametrize("value", ["", " leading", "trailing ", "tab\tinside", "del\x7finside"])
def test_python_and_schema_reject_noncanonical_text(value: str) -> None:
    schema = json.loads(
        (ROOT / "schemas/synaptic-provider-descriptor-v1.schema.json").read_text(encoding="utf-8")
    )
    document = _descriptor().to_dict()
    document["provider_id"] = value
    with pytest.raises((ValueError, TypeError)):
        _descriptor(provider_id=value)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, schema)


@pytest.mark.parametrize("value", [False, True, 1.5, math.nan, math.inf, -math.inf])
def test_artifact_size_rejects_noncanonical_integers(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        VerifiedArtifact("model", DIGESTS[6], value)


def test_integral_float_normalizes_to_json_integer_and_matches_schema() -> None:
    artifact = VerifiedArtifact("model", DIGESTS[6], 7.0)  # type: ignore[arg-type]
    outcome = RunOutcome(
        "synaptic-run-outcome/v1", TrainingRunRef("run-1", "project-1"),
        TrainingRunState.SUCCEEDED, (artifact,),
    )
    document = outcome.to_dict()
    assert document["artifacts"] == {"model": {"sha256": DIGESTS[6], "size_bytes": 7}}
    assert isinstance(document["artifacts"]["model"]["size_bytes"], int)  # type: ignore[index]
    schema = json.loads(
        (ROOT / "schemas/synaptic-run-outcome-v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(document, schema)
    assert RunOutcome.from_dict(document) == outcome


def test_role_keyed_artifacts_are_closed_unique_and_deterministic() -> None:
    run = TrainingRunRef("run-1", "project-1")
    first = VerifiedArtifact("z-role", DIGESTS[6], 7)
    second = VerifiedArtifact("a-role", DIGESTS[7], 8)
    outcome = RunOutcome("synaptic-run-outcome/v1", run, TrainingRunState.SUCCEEDED, (first, second))
    assert list(outcome.to_dict()["artifacts"]) == ["a-role", "z-role"]  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        RunOutcome("synaptic-run-outcome/v1", run, TrainingRunState.SUCCEEDED, (first, first))
    malformed = outcome.to_dict()
    malformed["artifacts"]["a-role"]["unknown"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown fields"):
        RunOutcome.from_dict(malformed)


def test_publication_artifacts_use_the_same_role_keyed_canonical_shape() -> None:
    result = PublicationResult(
        "synaptic-publication-result/v1",
        PublicationRef("publication-1", "local/default"),
        PublicationState.VERIFIED,
        (VerifiedArtifact("model", DIGESTS[6], 7.0),),  # type: ignore[arg-type]
    )
    document = result.to_dict()
    schema = json.loads(
        (ROOT / "schemas/synaptic-publication-result-v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(document, schema)
    assert PublicationResult.from_dict(document) == result


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("request_id", "request-2"), ("project_ref", "project-2"),
        ("source_digest", DIGESTS[7]), ("resolved_config_digest", DIGESTS[8]),
        ("workload_digest", DIGESTS[9]), ("runtime_digest", DIGESTS[10]),
        ("artifact_policy_digest", DIGESTS[11]),
    ],
)
def test_basis_digest_binds_every_basis_input(field: str, replacement: str) -> None:
    basis = TrainingPlanBasisV1.from_resolved(_resolved())
    assert replace(basis, **{field: replacement}).basis_digest != basis.basis_digest


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provider", ProviderRef("docker", "another-profile")),
        ("basis_digest", DIGESTS[7]), ("descriptor_digest", DIGESTS[8]),
        ("profile_digest", DIGESTS[9]),
    ],
)
def test_provider_context_digest_binds_only_context_inputs(field, replacement) -> None:
    basis, context, _ = _plan()
    assert replace(context, **{field: replacement}).provider_context_digest != context.provider_context_digest


def test_plan_fingerprint_binds_context_ref_and_excludes_descendants() -> None:
    basis, context, plan = _plan()
    changed = replace(context, profile_digest=DIGESTS[10])
    assert replace(plan, basis=replace(basis, workload_digest=DIGESTS[9])).plan_fingerprint != plan.plan_fingerprint
    assert replace(plan, provider_plan=ProviderPlanRef(changed.provider_context_digest)).plan_fingerprint != plan.plan_fingerprint
    serialized = json.dumps(plan.to_dict(), sort_keys=True)
    for forbidden in ("plan_fingerprint", "preparation_digest", "effect", "grant", "command", "receipt", "job_ref"):
        assert forbidden not in serialized


def test_closed_provider_contracts_round_trip_and_reject_unknown_fields() -> None:
    descriptor = _descriptor()
    assert ProviderDescriptor.from_dict(descriptor.to_dict()) == descriptor
    document = descriptor.to_dict()
    document["modal_app_name"] = "forbidden"
    with pytest.raises(ValueError, match="unknown fields"):
        ProviderDescriptor.from_dict(document)


def test_foundation_imports_load_no_internal_provider_or_persistence_modules() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1.providers, synaptic_tuner.api.v1.planning
import synaptic_tuner.api.v1.results, synaptic_tuner.api.v1.training_facade
import synaptic_tuner.api.v1.runs_facade, synaptic_tuner.api.v1.artifacts_facade
import synaptic_tuner.host.v1
print(json.dumps(sorted(n for n in sys.modules if n == 'tuner' or n.startswith(('tuner.', 'modal', 'sqlite3')))))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []


def test_foundation_source_has_no_forbidden_imports() -> None:
    paths = [
        ROOT / "synaptic_tuner/api/v1" / name
        for name in ("_contract.py", "providers.py", "planning.py", "results.py", "training_facade.py", "runs_facade.py", "artifacts_facade.py")
    ] + [ROOT / "synaptic_tuner/host/v1/ports.py"]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imports.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        assert imports.isdisjoint({"tuner", "modal", "sqlite3"})
