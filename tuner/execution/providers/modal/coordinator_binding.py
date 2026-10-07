"""Immutable complete-command content for a consumer-owned authenticated catalog.

Construction proves configuration equivalence, not provider authentication,
quote freshness, source availability, or permission to execute. The catalog's
authority must authenticate the complete canonical_bytes before either adapter
uses this value. No key, signing service, database, or registry lives here.
"""

from dataclasses import dataclass

from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest, parse_canonical_object
from tuner.execution.foundation_v2.commands import parse_exact_command

from .binding import ModalClientBinding
from .coordinator_adapter import ModalPreparationAdapter
from .resolution import VerifiedModalDeploymentIdentityV1


class _NoPreflightClock:
    def now_epoch(self):
        raise RuntimeError("command content reconstruction cannot perform preflight")

    def now_iso(self):
        raise RuntimeError("command content reconstruction cannot perform preflight")


@dataclass(frozen=True, slots=True)
class ModalCommandBinding:
    command_bytes: bytes
    preparation_snapshot: bytes
    deployment_bytes: bytes

    def __post_init__(self):
        for value in (self.command_bytes, self.preparation_snapshot, self.deployment_bytes):
            if type(value) is not bytes:
                raise TypeError("exact immutable canonical bytes required")
        adapter = ModalPreparationAdapter.restore(
            self.preparation_snapshot, clock=_NoPreflightClock(),
        )
        command = parse_exact_command(self.command_bytes)
        _, execution, plan = adapter._snapshot()
        expected = adapter.prepare(
            plan, TrainingRunRef(command.preparation.run_id, command.preparation.project_ref),
            execution,
        )
        if (command.preparation != expected or command.executor != execution.executor_descriptor
                or command.payload != adapter.payload(expected, command.operation.effect.kind)):
            raise ValueError("command differs from retained Modal preparation")
        document = parse_canonical_object(self.preparation_snapshot, name="preparation snapshot")
        if self.deployment.selection.to_dict() != document["configuration"]["selection"]:
            raise ValueError("deployment differs from retained Modal configuration")

    @property
    def deployment(self):
        return VerifiedModalDeploymentIdentityV1.from_dict(
            parse_canonical_object(self.deployment_bytes, name="deployment")
        )

    @property
    def client_binding(self):
        selection = self.deployment.selection
        return ModalClientBinding(
            selection.account_ref, selection.workspace_ref, selection.environment_ref,
            selection.client_ref, selection.sdk_version,
        )

    @property
    def command_digest(self):
        return parse_exact_command(self.command_bytes).digest

    @property
    def provider_id(self):
        return parse_exact_command(self.command_bytes).preparation.provider.provider_id

    @property
    def profile_ref(self):
        return parse_exact_command(self.command_bytes).preparation.provider.profile_ref

    @property
    def account_ref(self):
        return parse_exact_command(self.command_bytes).preparation.scope.account_ref

    @property
    def namespace_ref(self):
        return parse_exact_command(self.command_bytes).preparation.scope.namespace_ref

    @property
    def canonical_bytes(self):
        return canonical_bytes({
            "command": parse_canonical_object(self.command_bytes, name="command"),
            "preparation_snapshot": parse_canonical_object(self.preparation_snapshot, name="snapshot"),
            "deployment": parse_canonical_object(self.deployment_bytes, name="deployment"),
        })

    @property
    def authenticated_binding_digest(self):
        # Catalog protocol name: this identifies content; it is NOT authentication.
        return domain_digest("synaptic-modal-command-binding/v1", self.canonical_bytes)


__all__: list[str] = []
