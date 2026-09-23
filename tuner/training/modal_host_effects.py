"""Consumer-owned one-shot effect material for the standalone Modal host.

The generic coordinator and Foundation still own lifecycle authority.  This
module only retains exact packaged transport inputs and makes an irreversible
local attempt claim before either provider effect can run.
"""

from __future__ import annotations

import json
from threading import RLock

from synaptic_tuner.api.v1.execution import ExecutionGrant
from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.execution.foundation_v2.commands import StageCommandV2, SubmitCommandV2
from tuner.execution.providers.modal.packaged_binding import ModalPackagedCommandBinding
from tuner.execution.providers.modal.packaged_dispatch import build_modal_packaged_dispatch
from tuner.execution.providers.modal.packaged_staging import (
    ModalPackagedStageReceipt, prepare_modal_packaged_stage,
)


_BINDING_FIELDS = (
    "command_bytes", "runtime_release_bytes", "provider_binding_bytes",
    "provider_facts_bytes", "execution_binding_bytes",
)


def _encode_binding(value: ModalPackagedCommandBinding) -> bytes:
    if type(value) is not ModalPackagedCommandBinding:
        raise TypeError("exact packaged binding required")
    return json.dumps(
        {name: getattr(value, name).hex() for name in _BINDING_FIELDS},
        sort_keys=True, separators=(",", ":"),
    ).encode("ascii")


def _decode_binding(payload: bytes) -> ModalPackagedCommandBinding:
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate packaged binding field")
            result[key] = value
        return result
    fields = json.loads(payload.decode("ascii"), object_pairs_hook=unique)
    if set(fields) != set(_BINDING_FIELDS) or any(type(fields[name]) is not str for name in _BINDING_FIELDS):
        raise ValueError("packaged binding catalog is invalid")
    result = ModalPackagedCommandBinding(*(bytes.fromhex(fields[name]) for name in _BINDING_FIELDS))
    if _encode_binding(result) != payload:
        raise ValueError("packaged binding catalog is not canonical")
    return result


def _decode_receipt(payload: bytes) -> ModalPackagedStageReceipt:
    return ModalPackagedStageReceipt.from_dict(parse_canonical_object(payload, name="stage receipt"))


def _encode_call(value: str) -> bytes:
    if type(value) is not str or not value:
        raise TypeError("exact provider call reference required")
    return canonical_bytes({"provider_job_ref": value})


def _decode_call(payload: bytes) -> str:
    value = parse_canonical_object(payload, name="provider call")
    if set(value) != {"provider_job_ref"} or type(value["provider_job_ref"]) is not str:
        raise ValueError("provider call catalog is invalid")
    return value["provider_job_ref"]


def _canonical(value: object) -> bytes:
    projection = getattr(value, "canonical_bytes")
    result = projection() if callable(projection) else projection
    if type(result) is not bytes:
        raise TypeError("packaged canonical bytes required")
    return result


class _OneUseSource:
    def __init__(self) -> None:
        self._items: dict[str, object] = {}
        self._lock = RLock()

    def publish(self, key: str, value: object) -> None:
        with self._lock:
            if key in self._items:
                raise ValueError("packaged source already retained")
            self._items[key] = value

    def resolve(self, command_digest: str) -> object | None:
        with self._lock:
            return self._items.pop(command_digest, None)


class ModalPackagedHostEffectsV1:
    """Exact catalogs/sources plus the host grant binding boundary.

    ``storage`` is a private consumer-owned SQLite one-shot/catalog store.  A
    fresh command cannot borrow an old claim: duplicate claims always fail,
    even when the prior provider response was lost.
    """

    def __init__(self, *, storage: object, runtime_release: object,
                 provider_binding: object, provider_facts: object,
                 execution_binding: object, retained_source: object,
                 workload_bytes: bytes, artifact_policy: object,
                 signer: object, key_ref: str, maximum_cost_minor_units: int,
                 currency: str = "USD") -> None:
        if (type(maximum_cost_minor_units) is not int or maximum_cost_minor_units < 1
                or type(workload_bytes) is not bytes or not workload_bytes):
            raise ValueError("bounded packaged host inputs required")
        self.bindings = storage.catalog(
            "packaged-bindings-v1", encode=_encode_binding, decode=_decode_binding,
        )
        self.stage_receipts = storage.catalog(
            "packaged-stage-receipts-v1",
            encode=lambda value: value.canonical_bytes, decode=_decode_receipt,
        )
        self.calls = storage.catalog(
            "packaged-calls-v1", encode=_encode_call, decode=_decode_call,
        )
        self.stages, self.dispatches = _OneUseSource(), _OneUseSource()
        self._storage = storage
        self._release = runtime_release
        self._provider = provider_binding
        self._facts = provider_facts
        self._execution = execution_binding
        self._source = retained_source
        self._workload = bytes(workload_bytes)
        self._policy = artifact_policy
        self._signer, self._key_ref = signer, key_ref
        self._cost, self._currency = maximum_cost_minor_units, currency
        self._stage_effects: dict[str, str] = {}
        self._grant = ExecutionGrant("modal-packaged-training-v1")

    def authorize(self, requirements: tuple[object, ...]) -> ExecutionGrant:
        if (len(requirements) != 1
                or requirements[0].operation != "training.start"
                or requirements[0].paid_effect is not True
                or requirements[0].maximum_cost_minor_units != self._cost
                or requirements[0].currency != self._currency):
            raise ValueError("Modal paid grant differs from approved bound")
        return self._grant

    def authenticate(self, binding: ModalPackagedCommandBinding) -> bool:
        try:
            if type(binding) is not ModalPackagedCommandBinding:
                return False
            retained = self.bindings.resolve(binding.command_digest)
            return (retained == binding
                    and binding.runtime_release == self._release
                    and binding.provider_binding == self._provider
                    and binding.provider_facts == self._facts
                    and binding.execution_binding == self._execution
                    and self._storage.attempts.resolve(binding.command_digest) is not None)
        except Exception:
            return False

    def bind(self, grant: ExecutionGrant, *, operation: object,
             requirements: tuple[object, ...]) -> object:
        if grant != self.authorize(requirements) or type(operation) not in (StageCommandV2, SubmitCommandV2):
            raise ValueError("unsupported packaged Modal operation")
        binding = ModalPackagedCommandBinding(
            operation.canonical_bytes, _canonical(self._release),
            _canonical(self._provider), _canonical(self._facts),
            _canonical(self._execution),
        )
        if binding.command_digest != operation.digest:
            raise ValueError("packaged command digest mismatch")
        stage_receipt = None
        if type(operation) is SubmitCommandV2:
            stage_digest = self._stage_effects.get(operation.stage_predecessor.stage_effect_id)
            stage_receipt = None if stage_digest is None else self.stage_receipts.resolve(stage_digest)
            if stage_receipt is None:
                raise ValueError("authenticated stage receipt is unavailable")
        # Retain the exact binding first; a failed claim never permits an effect.
        self.bindings.publish_if_absent(operation.digest, binding)
        self._storage.attempts.claim(
            operation.digest,
            canonical_bytes({
                "schema_version": "synaptic-modal-packaged-host-attempt/v1",
                "command_digest": operation.digest,
                "binding_digest": binding.authenticated_binding_digest,
            }),
        )
        if type(operation) is StageCommandV2:
            self.stages.publish(operation.digest, prepare_modal_packaged_stage(
                self._execution, self._source.open_lease(),
                stage_effect_id=operation.operation.effect.effect_id,
                artifact_volume_id=self._facts.artifact_volume_id,
            ))
            self._stage_effects[operation.operation.effect.effect_id] = operation.digest
        else:
            self.dispatches.publish(operation.digest, build_modal_packaged_dispatch(
                binding, stage_receipt, self._workload, self._policy,
                self._signer, key_ref=self._key_ref,
            ))
        return binding
