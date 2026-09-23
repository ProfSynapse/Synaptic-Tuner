from __future__ import annotations

import io

import pytest

from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement
from tuner.execution.foundation_v2.commands import StageCommandV2, SubmitCommandV2
from tuner.execution.providers.modal.packaged_dispatch import parse_modal_packaged_dispatch
from tuner.training.contracts import RetainedTrainingInputStreamLease
from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1

from tests.execution.providers.test_modal_packaged_binding import _binding
from tests.execution.providers.test_modal_packaged_dispatch import (
    Auth, PREPARED_PAYLOAD, _case,
)


class _Catalog:
    def __init__(self, encode, decode):
        self.values, self.encode, self.decode = {}, encode, decode

    def publish_if_absent(self, key, value):
        raw = self.encode(value)
        if key in self.values:
            if self.values[key] == raw:
                return False
            raise ValueError("conflicting catalog entry")
        self.values[key] = raw
        return True

    def resolve(self, key):
        raw = self.values.get(key)
        return None if raw is None else self.decode(raw)


class _Attempts:
    def __init__(self):
        self.claimed = {}

    def claim(self, key, evidence):
        if key in self.claimed:
            raise ValueError("attempt already claimed")
        self.claimed[key] = evidence

    def resolve(self, key):
        return self.claimed.get(key)


class _Storage:
    def __init__(self):
        self.attempts = _Attempts()
        self.catalogs = {}

    def catalog(self, key, *, encode, decode):
        catalog = _Catalog(encode, decode)
        self.catalogs[key] = catalog
        return catalog


class _Source:
    def __init__(self, identity):
        self.identity = identity

    def open_lease(self):
        return RetainedTrainingInputStreamLease(self.identity, io.BytesIO(PREPARED_PAYLOAD))


def test_one_shot_claim_precedes_stage_and_submit_sources_and_rejects_response_loss():
    stage_binding = _binding(PREPARED_PAYLOAD)
    submit_binding, receipt, workload, policy = _case()
    assert type(stage_binding.command) is StageCommandV2
    assert type(submit_binding.command) is SubmitCommandV2
    assert stage_binding.execution_binding == submit_binding.execution_binding
    storage, signer = _Storage(), Auth()
    effects = ModalPackagedHostEffectsV1(
        storage=storage, runtime_release=stage_binding.runtime_release,
        provider_binding=stage_binding.provider_binding,
        provider_facts=stage_binding.provider_facts,
        execution_binding=stage_binding.execution_binding,
        retained_source=_Source(receipt_identity(stage_binding)),
        workload_bytes=workload, artifact_policy=policy,
        signer=signer, key_ref="dispatch-key", maximum_cost_minor_units=100,
    )
    requirements = (AuthorizationRequirement("training.start", True, 100, "USD"),)
    grant = effects.authorize(requirements)
    stage = stage_binding.command
    effects.bind(grant, operation=stage, requirements=requirements)
    assert effects.bindings.resolve(stage.digest) == stage_binding
    assert effects.authenticate(stage_binding)
    assert effects.stages.resolve(stage.digest).execution_binding_digest == stage_binding.execution_binding.binding_digest
    effects.stage_receipts.publish_if_absent(stage.digest, receipt)
    submit = submit_binding.command
    effects.bind(grant, operation=submit, requirements=requirements)
    payload = effects.dispatches.resolve(submit.digest)
    assert parse_modal_packaged_dispatch(payload, signer).submit_command_bytes == submit.canonical_bytes
    assert submit.digest in storage.attempts.claimed
    with pytest.raises(ValueError, match="already claimed"):
        effects.bind(grant, operation=submit, requirements=requirements)
    assert effects.dispatches.resolve(submit.digest) is None


def receipt_identity(binding):
    from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
    return PreparedTrainingInputIdentity.from_dict(binding.execution_binding.to_dict()["prepared_input"])
