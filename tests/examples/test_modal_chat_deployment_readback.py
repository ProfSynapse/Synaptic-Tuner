"""Exact scoped RPC boundaries; no provider or credential access."""

import asyncio
from types import SimpleNamespace

import pytest

from examples.modal_chat import deployment_readback as reader


def case(
    *, change=None, missing=False, public=False, metadata_error=None, stopped=False
):
    pytest.importorskip("modal")
    from modal_proto import api_pb2 as pb

    scoped = pb.AppGetByDeploymentNameResponse(
        app_id="" if stopped else "ap-owned",
        previous_app_id="ap-previous" if stopped else "",
        environment_name="isolated",
        lifecycle=pb.AppLifecycle(
            app_state=pb.APP_STATE_STOPPED if stopped else pb.APP_STATE_DEPLOYED,
            version=7,
        ),
    )
    after = pb.AppGetByDeploymentNameResponse()
    after.CopyFrom(scoped)
    if change == "version":
        after.lifecycle.version = 8
    elif change == "environment":
        after.environment_name = "other"
    elif change == "app":
        after.app_id = "ap-other"
    elif change == "stopped":
        after.lifecycle.app_state = pb.APP_STATE_STOPPED
    elif change == "deployed":
        after.lifecycle.app_state = pb.APP_STATE_DEPLOYED
    elif change == "previous_app":
        after.previous_app_id = "ap-other"
    layout = pb.AppLayout(
        function_ids=(
            {"historical-worker": "fu-old"}
            if stopped
            else ({} if missing else {"worker": "fu-owned"})
        )
    )
    if not missing and not stopped:
        item = layout.objects.add(object_id="fu-owned")
        item.function_handle_metadata.CopyFrom(
            pb.FunctionHandleMetadata(
                function_name="worker",
                app_id="ap-owned",
                definition_id="",
                web_url="https://private-value.invalid" if public else "",
            )
        )
    if metadata_error == "name":
        layout.objects[0].function_handle_metadata.function_name = "other"
    elif metadata_error == "app":
        layout.objects[0].function_handle_metadata.app_id = "ap-other"
    elif metadata_error == "duplicate":
        duplicate = layout.objects.add()
        duplicate.CopyFrom(layout.objects[0])
    elif metadata_error == "missing":
        del layout.objects[:]
    elif metadata_error == "alias":
        layout.function_ids["other"] = "fu-owned"
    elif metadata_error == "class_overlap":
        layout.class_ids["other"] = "fu-owned"
    calls = []

    class Stub:
        async def AppGetByDeploymentName(self, request):
            calls.append(("scope", request.name, request.environment_name))
            return scoped if len(calls) == 1 else after

        async def AppGetLayout(self, request):
            calls.append(("layout", request.app_id))
            return pb.AppGetLayoutResponse(app_layout=layout)

    return SimpleNamespace(stub=Stub()), calls


def test_reads_only_exact_app_and_brackets_layout_with_current_generation():
    client, calls = case()
    observed = asyncio.run(reader._bounded_read(client, "app", "isolated", "worker"))
    assert observed == reader.CurrentModalDeployment(
        "ap-owned",
        7,
        True,
        (("worker", "fu-owned"),),
        (),
        (reader.CurrentModalFunction("fu-owned", "worker", "ap-owned", "", ""),),
    )
    assert calls == [
        ("scope", "app", "isolated"),
        ("layout", "ap-owned"),
        ("scope", "app", "isolated"),
    ]


def test_stopped_app_brackets_historical_layout_and_preserves_previous_identity():
    client, calls = case(stopped=True)
    observed = asyncio.run(reader._bounded_read(client, "app", "isolated", "worker"))
    assert observed == reader.CurrentModalDeployment(
        "", 7, False, (("historical-worker", "fu-old"),), (), (), "ap-previous"
    )
    assert calls == [
        ("scope", "app", "isolated"),
        ("layout", "ap-previous"),
        ("scope", "app", "isolated"),
    ]


@pytest.mark.parametrize(
    "change", ["version", "environment", "app", "deployed", "previous_app"]
)
def test_stopped_app_drift_is_rejected(change):
    client, _ = case(stopped=True, change=change)
    with pytest.raises(ValueError):
        asyncio.run(reader._bounded_read(client, "app", "isolated", "worker"))


@pytest.mark.parametrize(
    "app_id,previous_app_id",
    [("ap-current", "ap-previous"), ("", "")],
)
def test_stopped_app_requires_empty_current_and_safe_previous_identity(
    app_id, previous_app_id
):
    pytest.importorskip("modal")
    from modal_proto import api_pb2 as pb

    response = pb.AppGetByDeploymentNameResponse(
        app_id=app_id,
        previous_app_id=previous_app_id,
        environment_name="isolated",
        lifecycle=pb.AppLifecycle(app_state=pb.APP_STATE_STOPPED, version=7),
    )

    class Stub:
        async def AppGetByDeploymentName(self, request):
            return response

        async def AppGetLayout(self, request):
            return pb.AppGetLayoutResponse(app_layout=pb.AppLayout())

    with pytest.raises(ValueError):
        asyncio.run(
            reader._bounded_read(
                SimpleNamespace(stub=Stub()), "app", "isolated", "worker"
            )
        )


def test_transitional_and_disabled_states_fail_closed_before_layout():
    pytest.importorskip("modal")
    from modal_proto import api_pb2 as pb

    rejected = (
        pb.APP_STATE_UNSPECIFIED,
        pb.APP_STATE_INITIALIZING,
        pb.APP_STATE_EPHEMERAL,
        pb.APP_STATE_DETACHED,
        pb.APP_STATE_DETACHED_DISCONNECTED,
        pb.APP_STATE_DISABLED,
        pb.APP_STATE_STOPPING,
        pb.APP_STATE_DERIVED,
    )
    for state in rejected:
        response = pb.AppGetByDeploymentNameResponse(
            app_id="ap-owned",
            environment_name="isolated",
            lifecycle=pb.AppLifecycle(app_state=state, version=7),
        )

        class Stub:
            async def AppGetByDeploymentName(self, request):
                return response

            async def AppGetLayout(self, request):
                raise AssertionError("inadmissible state layout must not be read")

        with pytest.raises(ValueError):
            asyncio.run(
                reader._bounded_read(
                    SimpleNamespace(stub=Stub()), "app", "isolated", "worker"
                )
            )


@pytest.mark.parametrize("change", ["version", "environment", "app", "stopped"])
def test_changes_during_read_rejected(change):
    client, _ = case(change=change)
    with pytest.raises(ValueError):
        asyncio.run(reader._bounded_read(client, "app", "isolated", "worker"))


def test_missing_function_not_fabricated_and_public_url_not_retained():
    client, _ = case(missing=True)
    observed = asyncio.run(reader._bounded_read(client, "app", "isolated", "worker"))
    assert observed.function_ids == () and observed.functions == ()
    client, _ = case(public=True)
    observed = asyncio.run(reader._bounded_read(client, "app", "isolated", "worker"))
    assert observed.functions[0].web_url == "PUBLIC"


@pytest.mark.parametrize(
    "metadata_error", ["name", "app", "duplicate", "missing", "alias", "class_overlap"]
)
def test_inconsistent_or_ambiguous_selected_metadata_rejected(metadata_error):
    client, _ = case(metadata_error=metadata_error)
    with pytest.raises(ValueError):
        asyncio.run(reader._bounded_read(client, "app", "isolated", "worker"))


def test_timeout_is_bounded_and_does_not_become_absence(monkeypatch):
    async def slow(*args):
        await asyncio.sleep(100)

    monkeypatch.setattr(reader, "_read", slow)
    monkeypatch.setattr(reader, "_TIMEOUT_SECONDS", 0.001)
    with pytest.raises(TimeoutError):
        asyncio.run(reader._bounded_read(object(), "app", "isolated", "worker"))


def test_sync_bridge_preserves_explicit_client_and_closes_error(monkeypatch):
    pytest.importorskip("modal")
    client = object()
    observed = []

    async def read(selected, *args):
        observed.append(selected)
        raise ValueError("private-provider-value")

    monkeypatch.setattr(reader, "_read", read)
    with pytest.raises(reader.ModalDeploymentReadbackError) as result:
        reader.read_current_deployment(
            sdk=SimpleNamespace(__version__="1.5.4"),
            client=client,
            app_name="app",
            environment_name="isolated",
            function_name="worker",
        )
    assert observed == [client]
    assert str(result.value) == "modal_chat_deployment_readback_failed"


def test_only_exact_not_found_is_absence():
    pytest.importorskip("modal")
    from modal.exception import NotFoundError

    class Stub:
        async def AppGetByDeploymentName(self, request):
            raise NotFoundError("private-provider-value")

    assert (
        asyncio.run(
            reader._bounded_read(
                SimpleNamespace(stub=Stub()), "app", "isolated", "worker"
            )
        )
        is None
    )
