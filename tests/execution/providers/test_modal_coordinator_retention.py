"""Provider-free tests for consumer-owned Modal retention delegation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from synaptic_tuner.api.v1.results import TrainingRunRef

from tests.execution.providers.modal_coordinator_fixtures import real_launch_bundle_case
from tests.execution.providers.test_modal_coordinator_adapter import coordinator_harness
from tests.execution.providers.test_modal_coordinator_bundle import _fixture
from tests.execution.providers.test_modal_coordinator_launch import Authenticator
from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.references import ProviderStageRefV1
from tuner.execution.providers.modal.coordinator_retention import (
    ModalFoundationRetentionDelegate,
    ModalRetainedPreparation,
)
from tuner.execution.providers.modal.coordinator_staging import modal_stage_provider_ref


ROOT = Path(__file__).resolve().parents[3]


class Catalog:
    def __init__(self):
        self.values = {}
        self.publishes = 0

    def resolve(self, key):
        return self.values.get(key)

    def publish_if_absent(self, key, value):
        self.publishes += 1
        self.values.setdefault(key, value)


class Inputs:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def resolve(self, digest):
        self.calls += 1
        return self.value


class RetainedBindingAuthority:
    """Test authority that trusts only bytes present in the consumer catalog."""

    def __init__(self, catalog):
        self.catalog = catalog

    def authenticate(self, value):
        return any(
            type(candidate) is type(value)
            and candidate.canonical_bytes == value.canonical_bytes
            for candidate in self.catalog.values.values()
        )


class Foundation:
    def __init__(self, value):
        self.value = value
        self.executes = 0
        self.fail_execute = False

    def get(self, effect_id):
        return self.value.get(effect_id)

    def execute(self, command_bytes, grant, *, now_epoch):
        self.executes += 1
        if self.fail_execute:
            raise RuntimeError("simulated post-publication crash")
        return self.value[parse_exact_command(command_bytes).operation.effect.effect_id]

    def reconcile(self, *args, **kwargs):
        return (args, kwargs)

    def recover_orphan(self, *args, **kwargs):
        return (args, kwargs)


class Authority:
    def __init__(self, value):
        self.value = value
        self.assess_count = self.sign_count = 0
        self.accept_all = False

    def authenticate_grant(self, *args):
        return self.value.authenticate_grant(*args)

    def authenticate(self, value):
        return self.value.authenticate(value)

    def assess(self, value):
        self.assess_count += 1
        return self.value.assess(value)

    def sign(self, *args):
        self.sign_count += 1
        return self.value.sign(*args)

    def verify(self, *args):
        return True if self.accept_all else self.value.verify(*args)


def _case(monkeypatch):
    case = real_launch_bundle_case(monkeypatch)
    envelope = case["envelope"]
    stage = parse_exact_command(envelope.stage_material.binding.command_bytes)
    submit = parse_exact_command(envelope.submit_binding.command_bytes)
    records = {
        stage.operation.effect.effect_id: case["harness"].repository.get(stage.operation.effect.effect_id),
        submit.operation.effect.effect_id: case["harness"].repository.get(submit.operation.effect.effect_id),
    }
    members = {member.name: member.content for member in case["bundle"].members}
    retained = ModalRetainedPreparation(
        envelope.submit_binding.preparation_snapshot,
        envelope.submit_binding.deployment_bytes,
        case["bundle"].material.canonical_bytes, case["recipes"],
        members["log-terminal-policy.json"],
        members["worker-closure-manifest.json"],
        envelope.stage_material.control_volume_id,
        envelope.stage_material.artifact_volume_id,
        envelope.stage_material.key_ref,
    )
    foundation = Foundation(records)
    bindings, stages, launches = Catalog(), Catalog(), Catalog()
    inputs = Inputs(retained)
    assessment = Authority(case["harness"].foundation)
    signer = Authority(case["authenticator"])
    binding_authority = RetainedBindingAuthority(bindings)

    def wrapper():
        return ModalFoundationRetentionDelegate(
            foundation,
            foundation_authenticator=case["harness"].authenticator,
            assessment_authority=assessment,
            binding_authority=binding_authority, stage_authority=signer,
            launch_authority=signer, binding_catalog=bindings,
            stage_catalog=stages, launch_catalog=launches,
            retained_inputs=inputs,
        )
    return wrapper, foundation, inputs, bindings, stages, launches, assessment, signer, records, stage, submit


def test_retains_real_stage_and_launch_then_restart_reuses_exact_values(monkeypatch):
    (wrapper, foundation, _, bindings, stages, launches, assessments, signer,
     records, stage, submit) = _case(monkeypatch)
    first = wrapper()
    assert first.execute(stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
                         now_epoch=10) == records[stage.operation.effect.effect_id]
    assert first.execute(submit.canonical_bytes, records[submit.operation.effect.effect_id].grant,
                         now_epoch=11) == records[submit.operation.effect.effect_id]
    counts = (assessments.assess_count, signer.sign_count,
              bindings.publishes, stages.publishes, launches.publishes)
    second = wrapper()
    second.execute(stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
                   now_epoch=12)
    second.execute(submit.canonical_bytes, records[submit.operation.effect.effect_id].grant,
                   now_epoch=13)
    assert (assessments.assess_count, signer.sign_count,
            bindings.publishes, stages.publishes, launches.publishes) == counts
    assert foundation.executes == 4


def test_fresh_generic_start_publishes_before_catalog_authority_authenticates(monkeypatch):
    template, material, recipes, policy, closure = _fixture()
    import tests.execution.providers.test_modal_coordinator_adapter as adapter_tests
    original_inputs = adapter_tests.inputs
    monkeypatch.setattr(
        adapter_tests, "inputs",
        lambda: original_inputs() | {"resolved": material.planning_request},
    )
    generic, harness, adapter, plan = coordinator_harness(monkeypatch)
    monkeypatch.setattr(
        generic, "RUN", TrainingRunRef(material.run_id, material.planning_request.project_ref),
    )
    bindings, stages, launches = Catalog(), Catalog(), Catalog()
    retained = ModalRetainedPreparation(
        adapter.snapshot(), template.deployment_bytes, material.canonical_bytes,
        recipes, policy, closure, "control-id", "artifact-id", "stage-key",
    )
    signer = Authenticator()
    original_foundation = harness.foundation
    wrapper = ModalFoundationRetentionDelegate(
        original_foundation,
        foundation_authenticator=harness.authenticator,
        assessment_authority=original_foundation,
        binding_authority=RetainedBindingAuthority(bindings),
        stage_authority=signer, launch_authority=signer,
        binding_catalog=bindings, stage_catalog=stages, launch_catalog=launches,
        retained_inputs=Inputs(retained),
    )
    harness.foundation = wrapper
    harness.service = TrainingCoordinatorV1(
        adapter, generic.PlanningStore(), harness.workflows, harness.preparations,
        harness.execution_grants, harness.reconciliation_grants,
        adapter, adapter, harness.authorization, wrapper,
        harness.authenticator, generic.Clock(), generic.Identity(),
    )
    execute_once = harness.executor.execute_once

    def exact_stage_result(payload, request):
        result = execute_once(payload, request)
        if request.effect_kind == "stage":
            retained_stage = stages.resolve(request.command_digest)
            result = replace(result, stage_ref=ProviderStageRefV1(
                result.stage_ref.provider_id, result.stage_ref.profile_ref,
                result.stage_ref.account_ref, result.stage_ref.namespace_ref,
                modal_stage_provider_ref(retained_stage),
            ))
        return result

    harness.executor.execute_once = exact_stage_result
    workflow = harness.service.start(plan, generic.preflight())
    assert workflow.stage is not None and workflow.submit is not None
    assert bindings.publishes == 2 and stages.publishes == 1 and launches.publishes == 1
    assert harness.executor.calls == ["stage", "submit"]


def test_wrong_exact_grant_fails_before_retention_or_foundation(monkeypatch):
    wrapper, foundation, inputs, bindings, stages, launches, _, _, records, stage, submit = _case(monkeypatch)
    with pytest.raises(ValueError, match="grant authentication"):
        wrapper().execute(
            stage.canonical_bytes, records[submit.operation.effect.effect_id].grant,
            now_epoch=1,
        )
    assert (inputs.calls, bindings.publishes, stages.publishes,
            launches.publishes, foundation.executes) == (0, 0, 0, 0, 0)


def test_retained_stage_type_conflict_prevents_foundation_execution(monkeypatch):
    wrapper, foundation, _, _, stages, _, _, _, records, stage, _ = _case(monkeypatch)
    stages.values[stage.digest] = object()
    with pytest.raises(ValueError, match="stage"):
        wrapper().execute(
            stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
            now_epoch=1,
        )
    assert foundation.executes == 0


def test_corrupt_binding_readback_prevents_foundation_execution(monkeypatch):
    wrapper, foundation, _, bindings, _, _, _, _, records, stage, _ = _case(monkeypatch)
    bindings.publish_if_absent = lambda key, value: bindings.values.setdefault(key, object())
    with pytest.raises(ValueError, match="retained Modal value"):
        wrapper().execute(
            stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
            now_epoch=1,
        )
    assert foundation.executes == 0


def test_concurrent_different_valid_stage_winner_fails_this_attempt(monkeypatch):
    wrapper, foundation, _, _, stages, _, _, signer, records, stage, _ = _case(monkeypatch)
    signer.accept_all = True
    stages.publish_if_absent = lambda key, value: stages.values.setdefault(
        key, replace(value, claim_tag=b"different-valid-tag"),
    )
    with pytest.raises(ValueError, match="stage conflict"):
        wrapper().execute(
            stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
            now_epoch=1,
        )
    assert foundation.executes == 0


@pytest.mark.parametrize("kind", ["stage", "launch"])
def test_retained_authenticated_envelope_bad_mac_prevents_execution(monkeypatch, kind):
    (wrapper, foundation, _, _, stages, launches, _, _, records,
     stage, submit) = _case(monkeypatch)
    value = wrapper()
    value.execute(stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
                  now_epoch=1)
    if kind == "launch":
        value.execute(submit.canonical_bytes, records[submit.operation.effect.effect_id].grant,
                      now_epoch=2)
        launches.values[submit.digest] = replace(
            launches.values[submit.digest], claim_tag=b"invalid",
        )
        command = submit
    else:
        stages.values[stage.digest] = replace(
            stages.values[stage.digest], claim_tag=b"invalid",
        )
        command = stage
    foundation.executes = 0
    with pytest.raises(ValueError, match="authentication|conflict"):
        wrapper().execute(
            command.canonical_bytes, records[command.operation.effect.effect_id].grant,
            now_epoch=3,
        )
    assert foundation.executes == 0


def test_submit_reuses_retained_envelope_assessment_without_authority_calls(monkeypatch):
    (wrapper, foundation, _, _, stages, launches, assessments, signer,
     records, stage, submit) = _case(monkeypatch)
    case_wrapper = wrapper()
    case_wrapper.execute(stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
                         now_epoch=1)
    case_wrapper.execute(submit.canonical_bytes, records[submit.operation.effect.effect_id].grant,
                         now_epoch=2)
    prior = (assessments.assess_count, signer.sign_count)
    assert launches.resolve(submit.digest) is not None and stages.resolve(stage.digest) is not None
    wrapper().execute(submit.canonical_bytes, records[submit.operation.effect.effect_id].grant,
                      now_epoch=3)
    assert (assessments.assess_count, signer.sign_count) == prior
    assert foundation.executes == 3


def test_fresh_wrapper_reuses_stage_published_before_delegate_failure(monkeypatch):
    wrapper, foundation, _, _, stages, _, _, signer, records, stage, _ = _case(monkeypatch)
    foundation.fail_execute = True
    with pytest.raises(RuntimeError, match="post-publication"):
        wrapper().execute(
            stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
            now_epoch=1,
        )
    assert stages.resolve(stage.digest) is not None
    signed = signer.sign_count
    foundation.fail_execute = False
    wrapper().execute(
        stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
        now_epoch=2,
    )
    assert signer.sign_count == signed


def test_fresh_wrapper_reuses_launch_published_before_delegate_failure(monkeypatch):
    (wrapper, foundation, _, _, _, launches, assessments, signer,
     records, stage, submit) = _case(monkeypatch)
    first = wrapper()
    first.execute(
        stage.canonical_bytes, records[stage.operation.effect.effect_id].grant,
        now_epoch=1,
    )
    foundation.fail_execute = True
    with pytest.raises(RuntimeError, match="post-publication"):
        first.execute(
            submit.canonical_bytes, records[submit.operation.effect.effect_id].grant,
            now_epoch=2,
        )
    retained = launches.resolve(submit.digest)
    assert retained is not None
    prior = (assessments.assess_count, signer.sign_count, foundation.executes)
    foundation.fail_execute = False
    wrapper().execute(
        submit.canonical_bytes, records[submit.operation.effect.effect_id].grant,
        now_epoch=3,
    )
    assert launches.resolve(submit.digest) is retained
    assert (assessments.assess_count, signer.sign_count) == prior[:2]
    # One failed delegate entry and one idempotent retry; retention itself
    # never invokes the provider executor outside the unchanged Foundation.
    assert foundation.executes == prior[2] + 1


def test_nonexecution_methods_are_exact_delegations(monkeypatch):
    wrapper, foundation, *_ = _case(monkeypatch)
    value = wrapper()
    assert value.get("missing") is None
    grant = object()
    assert value.reconcile(b"x", grant, now_epoch=4) == (
        (b"x", grant), {"now_epoch": 4, "continuation": None},
    )
    # Compare shape rather than object identity for the deliberately opaque fake.
    args, kwargs = value.recover_orphan("effect-a", now_epoch=5)
    assert args == ("effect-a",) and kwargs == {"now_epoch": 5}


def test_import_is_provider_sdk_and_legacy_lifecycle_free():
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import tuner.execution.providers.modal.coordinator_retention
assert 'modal' not in sys.modules
assert 'tuner.execution.providers.modal.training' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode()
