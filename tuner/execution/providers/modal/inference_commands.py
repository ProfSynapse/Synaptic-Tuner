"""Canonical content binding for one prepared Modal chat command."""

from __future__ import annotations

import json

from tuner.execution.foundation_v2.canonical import (
    MAX_CANONICAL_BYTES,
    domain_digest,
)
from tuner.execution.foundation_v2.commands import (
    StageCommandV2,
    SubmitCommandV2,
    parse_exact_command,
)
from tuner.execution.foundation_v2.identities import EffectKind, derive_effect

from .inference_preparation import _validate_preparation_snapshot

_SCHEMA = "synaptic-modal-inference-command-binding/v1"
_EMPTY_ENVELOPE = {
    "schema_version": _SCHEMA,
    "command": {},
    "preparation_snapshot": {},
}
_EMPTY_ENVELOPE_BYTES = json.dumps(
    _EMPTY_ENVELOPE,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
_MAX_ENVELOPE_BYTES = 2 * MAX_CANONICAL_BYTES + len(_EMPTY_ENVELOPE_BYTES) - 4


class ModalInferenceCommandBinding:
    """Self-consistent chat command content; authentication remains external."""

    __slots__ = ("_command_bytes", "_preparation_snapshot")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("ModalInferenceCommandBinding is final")

    def __init__(self, command_bytes: bytes, preparation_snapshot: bytes) -> None:
        if type(command_bytes) is not bytes or type(preparation_snapshot) is not bytes:
            raise TypeError("exact command and preparation bytes are required")
        object.__setattr__(self, "_command_bytes", bytes(command_bytes))
        object.__setattr__(self, "_preparation_snapshot", bytes(preparation_snapshot))
        self.canonical_bytes

    def __setattr__(self, name, value):
        raise AttributeError("Modal inference command binding is immutable")

    @property
    def command_bytes(self) -> bytes:
        self._validate()
        return self._command_bytes

    @property
    def preparation_snapshot(self) -> bytes:
        self._validate()
        return self._preparation_snapshot

    def _validate(self):
        if (
            type(self._command_bytes) is not bytes
            or type(self._preparation_snapshot) is not bytes
        ):
            raise TypeError("exact immutable binding bytes are required")
        command = parse_exact_command(self._command_bytes)
        snapshot = _validate_preparation_snapshot(self._preparation_snapshot)
        if (
            type(command) not in (StageCommandV2, SubmitCommandV2)
            or command.preparation.to_dict() != snapshot["preparation"]
            or command.executor.to_dict() != snapshot["executor"]
            or command.executor.executor_id != "modal-chat-executor"
            or command.payload.provider_id != command.preparation.provider.provider_id
            or command.payload.input_digest != command.preparation.workload_digest
        ):
            raise ValueError("command binding changed")
        if type(command) is SubmitCommandV2:
            predecessor = command.stage_predecessor
            preparation = command.preparation
            stage_effect = derive_effect(preparation, EffectKind.STAGE)
            if (
                predecessor.provider_id,
                predecessor.profile_ref,
                predecessor.account_ref,
                predecessor.namespace_ref,
                predecessor.project_ref,
                predecessor.run_id,
                predecessor.plan_fingerprint,
                predecessor.preparation_digest,
                predecessor.workload_digest,
                predecessor.stage_effect_id,
            ) != (
                preparation.provider.provider_id,
                preparation.provider.profile_ref,
                preparation.scope.account_ref,
                preparation.scope.namespace_ref,
                preparation.project_ref,
                preparation.run_id,
                preparation.plan_fingerprint,
                preparation.preparation_digest,
                preparation.workload_digest,
                stage_effect.effect_id,
            ):
                raise ValueError("submit predecessor changed")
        return command, snapshot

    @property
    def command_digest(self) -> str:
        return self._validate()[0].digest

    @property
    def binding_digest(self) -> str:
        return domain_digest(_SCHEMA, self.canonical_bytes)

    @property
    def canonical_bytes(self) -> bytes:
        command, snapshot = self._validate()
        document = {
            "schema_version": _SCHEMA,
            "command": command.to_dict(),
            "preparation_snapshot": snapshot,
        }
        encoded = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if not encoded or len(encoded) > _MAX_ENVELOPE_BYTES:
            raise ValueError("Modal inference command binding exceeds its bound")
        return encoded


__all__: list[str] = []
