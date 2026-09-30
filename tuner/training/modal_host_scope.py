"""Explicit packaged-host Modal credential and billing scope."""

from __future__ import annotations

import time

from tuner.execution.foundation_v2.canonical import safe_ref
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.runtime_build import _bounded


class ModalHostScopeUnavailable(RuntimeError):
    """Fixed diagnostic with no credential/provider text."""


def observe_modal_host_scope(*, sdk: object, client: object,
                             binding: ModalClientBinding) -> tuple[str, str, str, str]:
    try:
        _bounded(
            lambda: _read_scope(sdk, client, binding.workspace_ref,
                                binding.environment_ref),
            deadline=time.monotonic() + 30, code="modal_host_scope_unavailable",
        )
        return (binding.account_ref, binding.workspace_ref,
                binding.environment_ref, binding.client_ref)
    except Exception:
        raise ModalHostScopeUnavailable("modal_host_scope_unavailable") from None


def open_modal_host_scope(*, sdk: object, profile: str, environment_name: str,
                          client_ref: str = "train-config-host") -> tuple[object, ModalClientBinding]:
    """Use an explicitly named local profile and existing scoped environment."""
    try:
        if getattr(sdk, "__version__", None) != "1.5.4":
            raise ValueError
        profile = safe_ref(profile, "modal_profile")
        environment_name = safe_ref(environment_name, "environment_name")
        client_ref = safe_ref(client_ref, "client_ref")
        from modal.config import config

        token_id = config.get("token_id", profile=profile, use_env=False)
        token_secret = config.get("token_secret", profile=profile, use_env=False)
        if any(type(value) is not str or not value.strip()
               for value in (token_id, token_secret)):
            raise ValueError
        client = sdk.Client.from_credentials(token_id, token_secret)
        workspace_ref = _bounded(
            lambda: _read_scope(sdk, client, None, environment_name),
            deadline=time.monotonic() + 30, code="modal_host_scope_unavailable",
        )
        workspace_ref = safe_ref(workspace_ref, "workspace")
        return client, ModalClientBinding(
            workspace_ref, workspace_ref, environment_name, client_ref, "1.5.4",
        )
    except Exception:
        raise ModalHostScopeUnavailable("modal_host_scope_unavailable") from None


def _read_scope(sdk: object, client: object, workspace_name: str | None,
                environment_name: str) -> str:
    workspace = sdk.Workspace.from_context(client=client)
    workspace.hydrate(client)
    environment = sdk.Environment.from_name(
        environment_name, create_if_missing=False, client=client,
    )
    environment.hydrate(client)
    if (getattr(workspace, "is_hydrated", False) is not True
            or (workspace_name is not None and getattr(workspace, "name", None) != workspace_name)
            or getattr(environment, "is_hydrated", False) is not True
            or getattr(environment, "name", None) != environment_name):
        raise ValueError
    return workspace.name
