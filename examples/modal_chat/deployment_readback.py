"""Pinned-SDK, scoped, read-only current deployment metadata for the consumer.

Floating Function handles omit definition IDs. App generation and exact layout
are observed instead; this does not pin a later invocation to an old version.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from tuner.execution.foundation_v2.canonical import safe_ref

_TIMEOUT_SECONDS = 30


class ModalDeploymentReadbackError(RuntimeError):
    """Closed failure without provider response or exception text."""


@dataclass(frozen=True, slots=True)
class CurrentModalFunction:
    function_id: str
    name: str
    app_id: str
    web_url: str
    definition_id: str


@dataclass(frozen=True, slots=True)
class CurrentModalDeployment:
    app_id: str
    generation: int
    deployed: bool
    function_ids: tuple[tuple[str, str], ...]
    class_ids: tuple[tuple[str, str], ...]
    functions: tuple[CurrentModalFunction, ...]
    previous_app_id: str = ""


async def _read(client, app_name, environment_name, function_name):
    from modal.exception import NotFoundError
    from modal_proto import api_pb2

    request = api_pb2.AppGetByDeploymentNameRequest(
        name=app_name, environment_name=environment_name
    )
    try:
        before = await client.stub.AppGetByDeploymentName(request)
    except NotFoundError:
        # Caller independently rechecks the existing environment in its scope.
        return None
    state = before.lifecycle.app_state
    if (
        before.environment_name != environment_name
        or state not in (api_pb2.APP_STATE_DEPLOYED, api_pb2.APP_STATE_STOPPED)
        or type(before.lifecycle.version) is not int
        or before.lifecycle.version < 1
    ):
        raise ValueError
    if state == api_pb2.APP_STATE_STOPPED:
        if before.app_id or not before.previous_app_id:
            raise ValueError
        previous_app_id = safe_ref(before.previous_app_id, "previous_app_id")
        app_id, layout_app_id, deployed = "", previous_app_id, False
    else:
        if not before.app_id:
            raise ValueError
        app_id = safe_ref(before.app_id, "app_id")
        previous_app_id = (
            safe_ref(before.previous_app_id, "previous_app_id")
            if before.previous_app_id
            else ""
        )
        layout_app_id, deployed = app_id, True
    response = await client.stub.AppGetLayout(
        api_pb2.AppGetLayoutRequest(app_id=layout_app_id)
    )
    after = await client.stub.AppGetByDeploymentName(request)
    if after != before:
        raise ValueError
    layout = response.app_layout
    if (
        len(layout.objects) > 4096
        or len(layout.function_ids) > 256
        or len(layout.class_ids) > 256
    ):
        raise ValueError
    functions = tuple(
        sorted(
            (safe_ref(name, "function_name"), safe_ref(identifier, "function_id"))
            for name, identifier in layout.function_ids.items()
        )
    )
    classes = tuple(
        sorted(
            (safe_ref(name, "class_name"), safe_ref(identifier, "class_id"))
            for name, identifier in layout.class_ids.items()
        )
    )
    selected = layout.function_ids.get(function_name)
    function_objects = [identifier for _, identifier in functions]
    class_objects = [identifier for _, identifier in classes]
    if (
        len(set(function_objects)) != len(function_objects)
        or len(set(class_objects)) != len(class_objects)
        or set(function_objects).intersection(class_objects)
    ):
        raise ValueError
    observed = []
    # Do not read unrelated objects' metadata, source, arguments or secrets.
    for item in layout.objects:
        if selected is None or item.object_id != selected:
            continue
        metadata = item.function_handle_metadata
        if metadata.function_name != function_name or metadata.app_id != layout_app_id:
            raise ValueError
        definition = metadata.definition_id
        if definition:
            safe_ref(definition, "definition_id")
        # Retain visibility only, never the provider URL.
        observed.append(
            CurrentModalFunction(
                safe_ref(item.object_id, "function_id"),
                safe_ref(metadata.function_name, "function_name"),
                safe_ref(metadata.app_id, "app_id"),
                "PUBLIC" if metadata.web_url else "",
                definition,
            )
        )
    if selected is not None and len(observed) != 1:
        raise ValueError
    return CurrentModalDeployment(
        app_id,
        before.lifecycle.version,
        deployed,
        functions,
        classes,
        tuple(observed),
        previous_app_id,
    )


async def _bounded_read(client, app_name, environment_name, function_name):
    return await asyncio.wait_for(
        _read(client, app_name, environment_name, function_name),
        timeout=_TIMEOUT_SECONDS,
    )


def read_current_deployment(*, sdk, client, app_name, environment_name, function_name):
    """Use the supplied SDK client's loop; no new client or ambient credentials."""
    try:
        if getattr(sdk, "__version__", None) != "1.5.4" or client is None:
            raise ValueError
        for label, value in (
            ("app_name", app_name),
            ("environment_name", environment_name),
            ("function_name", function_name),
        ):
            safe_ref(value, label)
        # This pinned internal bridge translates the public Client to its
        # existing async client on Modal's own loop, preserving authentication.
        from modal._utils.async_utils import synchronizer

        return synchronizer.create_blocking(_bounded_read)(
            client, app_name, environment_name, function_name
        )
    except Exception:
        raise ModalDeploymentReadbackError(
            "modal_chat_deployment_readback_failed"
        ) from None
