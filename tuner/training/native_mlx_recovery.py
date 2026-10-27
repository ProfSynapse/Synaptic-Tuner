"""Lookup and quiescence through retained native evidence; never redispatch."""

from dataclasses import dataclass
import hashlib

from tuner.execution.coordinator_v1.foundation import QuiescenceRecoveryRequestV1
from tuner.execution.foundation_v2.canonical import domain_digest
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.executors import mint_resolved_adapter
from tuner.execution.foundation_v2.observations import ObservationDisposition, ProviderObservationV1
from tuner.execution.foundation_v2.reconciliation import ReconciliationTargetV1
from tuner.execution.foundation_v2.references import ProviderStageRefV1
from tuner.execution.foundation_v2.repository import DispatchState, EffectRecordV2
from tuner.runtime.native_mlx_material import canonical, strict_json


@dataclass(frozen=True)
class _NativeQuiescenceProof:
    canonical_bytes: bytes

    def __post_init__(self):
        if type(self.canonical_bytes) is not bytes or not 0 < len(self.canonical_bytes) <= 8192:
            raise ValueError("native quiescence proof bytes are invalid")

    @property
    def proof_digest(self):
        return domain_digest("synaptic-native-quiescence-proof/v1", self.canonical_bytes)


class NativeLifecycleRecovery:
    capabilities = ("lookup",)

    def __init__(self, family):
        self.family = family
        self.descriptor = family.planning.adapter_descriptor
        for name in ("provider_id", "profile_ref", "account_ref", "namespace_ref"):
            setattr(self, name, getattr(family, name))

    def resolve(self, request):
        return mint_resolved_adapter(request, self)

    def _material(self, command):
        if command.operation.effect.kind.value == "cancel":
            _, owned, material = self.family._cancellation(command.digest)
        else:
            material, owned = self.family._bound_material(command.digest)
        if owned.canonical_bytes != command.canonical_bytes:
            raise ValueError("native recovery command differs")
        return material

    def _stage(self, command):
        stored = self.family.records.read(partition="effects", key="native-stage/" + command.digest)
        if stored is None or not self.family.authority.authenticate_stage_ack(stored.canonical, command.digest):
            raise ValueError("native stage acknowledgement is unavailable")
        return {"kind": "stage", "ack_digest": hashlib.sha256(stored.canonical).hexdigest()}

    def witness(self, command):
        self.family.planning.require_lifecycle()
        material = self._material(command)
        if command.operation.effect.kind.value == "stage":
            return self._stage(command)
        self.family._outcome(material)
        terminal = self.family.records.read(partition="effects", key="native-terminal/" + material.invocation.digest)
        return {"kind": "terminal", "invocation_digest": material.invocation.digest,
                "terminal_digest": hashlib.sha256(terminal.canonical).hexdigest()}

    def lookup(self, target, preparation):
        if type(target) is not ReconciliationTargetV1:
            raise ValueError("native recovery target differs")
        self.family.planning.require_lifecycle()
        command = parse_exact_command(target.command_bytes)
        material = self._material(command)
        if (target.command_digest != command.digest or target.effect_id != command.operation.effect.effect_id
                or preparation.canonical_bytes != command.preparation.canonical_bytes):
            raise ValueError("native recovery target binding differs")
        disposition, references = ObservationDisposition.INDETERMINATE, {}
        kind = command.operation.effect.kind.value
        try:
            if kind == "stage":
                self._stage(command)
                references["stage_ref"] = ProviderStageRefV1(
                    self.provider_id, self.profile_ref, self.account_ref, self.namespace_ref, "native-stage-" + command.digest)
            elif kind == "submit":
                state = self.family.transport.observe(material.invocation)
                if state["state"] not in ("running", "terminated"):
                    raise ValueError("native original dispatch is unconfirmed")
                if state["state"] == "terminated": self.family._outcome(material)
                references["provider_run"] = self.family._run_ref(material)
            else:
                ack = self.family.records.read(partition="effects", key="native-cancellation/" + command.digest)
                if ack is None or not self.family.authority.authenticate_cancellation_ack(
                        ack.canonical, command.digest, material.invocation.digest):
                    # A terminal exact target already satisfies stop. This does
                    # not claim the lost request caused the terminal outcome.
                    self.family._outcome(material)
                references["cancellation"] = self.family._cancel_ref(command)
            disposition = ObservationDisposition.FOUND
        except Exception:
            references = {}  # Missing evidence never becomes absence/replay.
        return ProviderObservationV1(command.operation.effect.effect_id, command.digest,
            command.executor.digest, disposition, target.resolution_digest, target.ownership_epoch, **references)

    @staticmethod
    def _request(record):
        command = parse_exact_command(record.command_bytes)
        preparation = command.preparation
        return QuiescenceRecoveryRequestV1(
            "synaptic-quiescence-recovery-request/v1", command.operation.effect.effect_id, command.digest,
            domain_digest("synaptic-foundation-command-bytes/v1", record.command_bytes), command.executor.digest,
            record.record_digest, record.dispatch_source_digest, record.dispatch.value,
            record.grant.authenticated_grant_digest, preparation.provider.provider_id, preparation.provider.profile_ref,
            preparation.scope.account_ref, preparation.scope.namespace_ref, record.attempt_count, record.dispatch_epoch)

    def obtain(self, request, *, now_epoch):
        if type(request) is not QuiescenceRecoveryRequestV1 or type(now_epoch) is not int or now_epoch < 1:
            raise ValueError("native quiescence request differs")
        command_material = self.family.catalog.resolve(request.command_digest)
        command = parse_exact_command(command_material.command_bytes)
        if (request.effect_id != command.operation.effect.effect_id
                or request.command_bytes_digest != domain_digest("synaptic-foundation-command-bytes/v1", command.canonical_bytes)
                or request.executor_digest != command.executor.digest
                or (request.provider_id, request.profile_ref, request.account_ref, request.namespace_ref)
                != (self.provider_id, self.profile_ref, self.account_ref, self.namespace_ref)):
            raise ValueError("native quiescence scope differs")
        witness = self.witness(command)
        return _NativeQuiescenceProof(self.family.authority.recovery_evidence({
            "schema": "synaptic-native-quiescence-proof/v1", "request": strict_json(request.canonical_bytes),
            "witness": witness, "issued_epoch": now_epoch}))

    def verify_quiescence(self, proof, record, *, now_epoch):
        try:
            if (type(proof) is not _NativeQuiescenceProof or type(record) is not EffectRecordV2
                    or type(now_epoch) is not int or now_epoch < 1):
                return False
            content = self.family.authority.authenticate_recovery_evidence(proof.canonical_bytes)
            request = self._request(record)
            return (set(content) == {"schema", "request", "witness", "issued_epoch"}
                    and content["schema"] == "synaptic-native-quiescence-proof/v1"
                    and type(content["issued_epoch"]) is int and 1 <= content["issued_epoch"] <= now_epoch
                    and canonical(content["request"]) == request.canonical_bytes
                    and canonical(content["witness"]) == canonical(self.witness(parse_exact_command(record.command_bytes))))
        except Exception:
            return False

    def verify_finality(self, *args, **kwargs):
        return False  # No native absence proof or resubmission authority.

    def retain(self, core):
        return _NativeFoundation(core, self)


class _NativeFoundation:
    def __init__(self, core, recovery):
        self.core, self.recovery = core, recovery

    def get(self, effect_id):
        record = self.core.get(effect_id)
        if record is not None and record.dispatch is DispatchState.OWNED_IN_FLIGHT:
            command = parse_exact_command(record.command_bytes)
            material = self.recovery._material(command)
            active = self.recovery.family.transport._live.get(material.invocation.attempt)
            if active is None:
                try:
                    # Independently verified terminal/read-only stage evidence,
                    # never a persisted PID, permits closing the old dispatch.
                    self.recovery.witness(command)
                    self.core._repo.orphan(effect_id)
                except Exception:
                    pass
                record = self.core.get(effect_id)
        return record

    def assess(self, record): return self.core.assess(record)
    def authenticate(self, value): return self.core.authenticate(value)
    def execute(self, command_bytes, grant, *, now_epoch): return self.core.execute(command_bytes, grant, now_epoch=now_epoch)
    def reconcile(self, command_bytes, grant, *, now_epoch, continuation=None):
        return self.core.reconcile(command_bytes, grant, now_epoch=now_epoch, continuation=continuation)
    def recover_orphan(self, effect_id, *, now_epoch): return self.core.recover_orphan(effect_id, now_epoch=now_epoch)
