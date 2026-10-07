"""Authenticate exact chat command content in a consumer-owned catalog.

These helpers neither dispatch Foundation commands nor issue grants. A retained
binding is content, not proof of current source admission, an inspected runtime,
or permission to serve. The consumer supplies durable publish-if-absent semantics
and an authority that authenticates the complete binding's canonical bytes.
"""

from __future__ import annotations

from typing import Protocol

from tuner.execution.foundation_v2.canonical import digest_text

from .inference_commands import ModalInferenceCommandBinding


class ModalInferenceRetentionError(RuntimeError):
    """Closed, non-secret failure to authenticate or retain exact chat content."""


class _CommandCatalog(Protocol):
    def resolve(self, command_digest: str) -> object | None: ...

    def publish_if_absent(
        self, command_digest: str, binding: ModalInferenceCommandBinding
    ) -> None: ...


class _BindingAuthority(Protocol):
    def authenticate(self, binding: ModalInferenceCommandBinding) -> bool: ...


def _fields(binding: ModalInferenceCommandBinding) -> tuple[bytes, bytes]:
    if type(binding) is not ModalInferenceCommandBinding:
        raise TypeError("exact chat command binding required")
    values = (binding.command_bytes, binding.preparation_snapshot)
    if any(type(value) is not bytes for value in values):
        raise TypeError("exact binding byte snapshots required")
    return values


def _copy(binding, watched):
    snapshot = _fields(binding)
    copied = ModalInferenceCommandBinding(*snapshot)
    watched.extend(((binding, snapshot), (copied, snapshot)))
    return copied


def _guard(watched):
    for binding, snapshot in watched:
        if _fields(binding) != snapshot:
            raise ValueError("chat binding changed during retention")


def _authenticate(binding, authority, watched):
    # The collaborator receives a freshly validated owned copy. Guard every
    # previously observed object, including the caller's instance, afterwards.
    authenticated = authority.authenticate(binding)
    _guard(watched)
    if authenticated is not True:
        raise ValueError("chat binding authentication denied")


def _admit(value, key, authority, watched):
    binding = _copy(value, watched)
    if binding.command_digest != key:
        raise ValueError("chat catalog key differs from command")
    _authenticate(binding, authority, watched)
    return binding


def load_modal_chat_command(
    command_digest: str, *, catalog: _CommandCatalog, authority: _BindingAuthority
) -> ModalInferenceCommandBinding:
    """Read and authenticate exact retained content without creating anything."""
    try:
        if type(command_digest) is not str:
            raise TypeError("exact command digest required")
        digest_text(command_digest, "command_digest")
        value = catalog.resolve(command_digest)
        return _admit(value, command_digest, authority, [])
    except Exception:
        raise ModalInferenceRetentionError(
            "modal_inference_retention_invalid"
        ) from None


def retain_modal_chat_command(
    binding: ModalInferenceCommandBinding,
    *,
    catalog: _CommandCatalog,
    authority: _BindingAuthority,
) -> ModalInferenceCommandBinding:
    """Publish only if absent, then authenticate and compare the retained bytes.

    A publication exception or missing/conflicting reread is a failure, never an
    overwrite or retry. The caller may later load the exact key to reconcile
    storage uncertainty; these helpers cannot allocate a provider resource.
    """
    try:
        watched = []
        candidate = _copy(binding, watched)
        key = candidate.command_digest
        expected = candidate.canonical_bytes
        _authenticate(candidate, authority, watched)
        value = catalog.resolve(key)
        _guard(watched)
        if value is None:
            catalog.publish_if_absent(key, candidate)
            _guard(watched)
            value = catalog.resolve(key)
            _guard(watched)
        retained = _admit(value, key, authority, watched)
        if retained.canonical_bytes != expected:
            raise ValueError("chat catalog publication conflict")
        _guard(watched)
        return retained
    except Exception:
        raise ModalInferenceRetentionError(
            "modal_inference_retention_invalid"
        ) from None


__all__: list[str] = []
