"""The Modal preparation adapter uses real generic contracts, without a cloud."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from synaptic_tuner.api.v1.planning import (
    ProviderPlanRef, ResolvedTrainingRequest, TrainingPlan, TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.coordinator_v1.coordinator import (
    CoordinatorCodeV1, CoordinatorErrorV1, TrainingCoordinatorV1,
)
from tuner.execution.coordinator_v1.model import WorkflowPhaseV1
from tuner.execution.foundation_v2.canonical import (
    FoundationError, canonical_bytes,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.config import ModalProviderProfileV1
from tuner.execution.providers.modal.coordinator_adapter import ModalPreparationAdapter


ROOT = Path(__file__).resolve().parents[3]
CHANGED = "f" * 64


class Clock:
    def now_iso(self):
        return "2026-08-27T00:00:00Z"


def inputs():
    return dict(
        profile=ModalProviderProfileV1.from_mapping(yaml.safe_load(
            (ROOT / "examples/host-project/providers/modal-a10-v1.yaml").read_text()
        )),
        binding=ModalClientBinding("account-a", "workspace-a", "environment-a", "client-a", "1.5.4"),
        resolved=ResolvedTrainingRequest(
            "synaptic-resolved-training-request/v1", "request-a", "project-a",
            *(str(i) * 64 for i in range(1, 6)),
        ),
        runtime_environment={"LANG": "C.UTF-8"}, quote_digest="6" * 64,
        timeout_seconds=900, clock=Clock(),
    )


def composed(**changes):
    values = inputs() | changes
    adapter = ModalPreparationAdapter(**values)
    provider = ProviderRef("modal", values["profile"].profile)
    context = adapter.context(values["resolved"], provider)
    plan = TrainingPlan(
        "synaptic-training-plan/v2", TrainingPlanBasisV1.from_resolved(values["resolved"]),
        ProviderPlanRef(context.provider_context_digest),
    )
    return adapter, context, plan, adapter.resolve(provider, context)


def test_planning_keeps_modal_objects_out_of_generic_documents():
    adapter, context, plan, binding = composed()
    preparation = adapter.prepare(plan, TrainingRunRef("run-a", "project-a"), binding)
    assert context.provider.provider_id == "modal"
    assert adapter.describe(context.provider).descriptor_digest == context.descriptor_digest
    assert preparation.execution_binding_digest == binding.binding_digest
    assert preparation.source_digest == plan.basis.source_digest
    for document in (plan.to_dict(), context.to_dict(), preparation.to_dict()):
        raw = canonical_bytes(document)
        for forbidden in (b"workspace-a", b"environment-a", b"synaptic-training-v1", b"HF_TOKEN", b"volume"):
            assert forbidden not in raw


def test_configuration_is_detached_from_caller_mutation():
    values = inputs()
    adapter = ModalPreparationAdapter(**values)
    provider = ProviderRef("modal", values["profile"].profile)
    expected = adapter.context(values["resolved"], provider)
    values["runtime_environment"]["LANG"] = "changed"
    object.__setattr__(values["profile"], "control_volume_ref", "changed")
    object.__setattr__(values["binding"], "workspace_ref", "changed")
    assert adapter.context(values["resolved"], provider) == expected
    object.__setattr__(expected.provider, "profile_ref", "changed")
    assert adapter.context(values["resolved"], provider).provider == provider


@pytest.mark.parametrize("name", ["HF_TOKEN", "MODAL_TOKEN_SECRET", "CUSTOM_PASSWORD"])
def test_runtime_environment_reuses_existing_secret_refusal(name):
    with pytest.raises(ValueError, match="named Modal Secrets"):
        composed(runtime_environment={name: "synthetic-test-value"})


@pytest.mark.parametrize("field", ["source_digest", "resolved_config_digest", "workload_digest", "runtime_digest", "artifact_policy_digest"])
def test_changed_resolved_input_cannot_use_existing_context(field):
    adapter, context, _, _ = composed()
    changed = replace(inputs()["resolved"], **{field: CHANGED})
    with pytest.raises(FoundationError):
        adapter.context(changed, context.provider)


@pytest.mark.parametrize("field", ["basis_digest", "profile_digest", "descriptor_digest"])
def test_resolver_rejects_context_substitution(field):
    adapter, context, _, _ = composed()
    with pytest.raises(FoundationError):
        adapter.resolve(context.provider, replace(context, **{field: CHANGED}))


@pytest.mark.parametrize("provider", [ProviderRef("docker", "local"), ProviderRef("modal", "other")])
def test_other_provider_or_profile_is_not_a_fallback(provider):
    adapter, context, _, _ = composed()
    with pytest.raises(FoundationError):
        adapter.describe(provider)
    with pytest.raises(FoundationError):
        adapter.resolve(provider, context)


@pytest.mark.parametrize("field", [
    "provider_descriptor_digest", "profile_digest", "reconciliation_adapter_digest",
    "resource_digest", "quote_digest", "secret_requirements_digest",
])
def test_preparation_rejects_execution_binding_substitution(field):
    adapter, _, plan, binding = composed()
    with pytest.raises(FoundationError):
        adapter.prepare(plan, TrainingRunRef("run-a", "project-a"), replace(binding, **{field: CHANGED}))


@pytest.mark.parametrize("field", [
    "plan_fingerprint", "source_digest", "workload_digest", "runtime_digest",
    "resource_digest", "artifact_contract_digest", "quote_digest",
    "secret_requirements_digest", "execution_binding_digest",
])
def test_payload_rejects_rebound_preparation(field):
    adapter, _, plan, binding = composed()
    preparation = adapter.prepare(plan, TrainingRunRef("run-a", "project-a"), binding)
    document = preparation.to_dict() | {field: CHANGED}
    changed = CanonicalPreparationV2.parse(canonical_bytes(document))
    with pytest.raises(FoundationError):
        adapter.payload(changed, EffectKind.STAGE)


def test_preparation_rejects_other_project_and_changed_plan():
    adapter, _, plan, binding = composed()
    with pytest.raises(FoundationError):
        adapter.prepare(plan, TrainingRunRef("run-a", "other-project"), binding)
    with pytest.raises(FoundationError):
        adapter.prepare(replace(plan, basis=replace(plan.basis, source_digest=CHANGED)),
                        TrainingRunRef("run-a", "project-a"), binding)


@pytest.mark.parametrize("field", ["account_ref", "workspace_ref", "environment_ref", "client_ref"])
def test_scope_and_session_changes_change_the_profile_binding(field):
    _, context, _, binding = composed()
    _, changed_context, _, changed_binding = composed(
        binding=replace(inputs()["binding"], **{field: "different"}),
    )
    assert context.profile_digest != changed_context.profile_digest
    assert binding.binding_digest != changed_binding.binding_digest


def test_namespace_tuple_encoding_does_not_alias():
    original = inputs()["binding"]
    _, _, _, first = composed(binding=replace(original, workspace_ref="a/b", environment_ref="c"))
    _, _, _, second = composed(binding=replace(original, workspace_ref="a", environment_ref="b/c"))
    assert first.scope.namespace_ref != second.scope.namespace_ref


def test_quote_and_resources_are_bound_without_claiming_live_authority():
    adapter, context, plan, binding = composed()
    for changes in ({"quote_digest": CHANGED}, {"timeout_seconds": 901}):
        _, changed_context, _, changed_binding = composed(**changes)
        assert changed_context.profile_digest != context.profile_digest
        assert changed_binding.binding_digest != binding.binding_digest
    report = adapter.preflight(plan)
    assert not report.ready and report.binds(plan)
    assert report.authorization == ()
    assert report.diagnostic_codes == ("modal_coordinator_execution_unavailable",)
    assert not any(adapter.describe(context.provider).capabilities.to_dict().values())


def coordinator_harness(monkeypatch):
    # Reuse the existing generic coordinator's synthetic Foundation backend.
    # Only these test executors supply success; there is no Modal client here.
    from tests.execution.coordinator_v1 import test_start_reconcile_service as generic
    adapter, context, plan, binding = composed()
    for name, value in {
        "PROVIDER": context.provider, "SCOPE": binding.scope, "CONTEXT": context,
        "PLAN": plan, "DESC": adapter.describe(context.provider),
        "RUN": TrainingRunRef("run-a", "project-a"),
    }.items():
        monkeypatch.setattr(generic, name, value)
    harness = generic.Harness()
    harness.executor.descriptor = binding.executor_descriptor
    harness.service = TrainingCoordinatorV1(
        adapter, generic.PlanningStore(), harness.workflows, harness.preparations,
        harness.execution_grants, harness.reconciliation_grants,
        adapter, adapter, harness.authorization, harness.foundation,
        harness.authenticator, generic.Clock(), generic.Identity(),
    )
    return generic, harness, adapter, plan


def test_real_coordinator_consumes_adapter_and_retains_one_shot_lineage(monkeypatch):
    generic, harness, adapter, plan = coordinator_harness(monkeypatch)
    result = harness.service.start(plan, generic.preflight())
    assert result.phase is WorkflowPhaseV1.QUEUED
    assert harness.executor.calls == ["stage", "submit"]
    preparation = harness.preparations.get(result.preparation_digest)
    for effect in (result.stage, result.submit):
        command = parse_exact_command(effect.canonical_command_bytes)
        assert command.preparation == preparation
        assert command.preparation.provider.provider_id == "modal"
        assert command.payload.input_digest == plan.basis.workload_digest
    assert harness.service.start(plan, generic.preflight()) == result
    assert harness.executor.calls == ["stage", "submit"]
    assert harness.authorization.effect_issues == 2


def test_unavailable_production_preflight_stops_before_grants_and_effects(monkeypatch):
    _, harness, adapter, plan = coordinator_harness(monkeypatch)
    with pytest.raises(CoordinatorErrorV1) as caught:
        harness.service.start(plan, adapter.preflight(plan))
    assert caught.value.code is CoordinatorCodeV1.PREFLIGHT_INVALID
    assert harness.authorization.effect_issues == 0
    assert harness.executor.calls == []


def test_adapter_import_does_not_load_sdk_host_or_old_modal_lifecycle():
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import tuner.execution.providers.modal.coordinator_adapter
for prefix in ('modal', 'huggingface_hub', 'sqlite3', 'synaptic_host',
               'tuner.execution.providers.modal.training'):
    assert not any(n == prefix or n.startswith(prefix + '.') for n in sys.modules), prefix
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
