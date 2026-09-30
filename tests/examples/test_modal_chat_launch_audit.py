"""Focused denial-order audit for the standalone launcher."""

from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from examples.modal_chat import launch
from tuner.execution.providers.modal import coordinator_adapter
from tuner.execution.providers.modal.coordinator_adapter import ModalPreparationAdapter


def test_quote_binding_matches_modal_preparation_adapter() -> None:
    from tests.execution.providers.test_modal_coordinator_adapter import inputs

    values = inputs()
    adapter = ModalPreparationAdapter(**values)
    _, expected, _ = adapter._snapshot()
    actual = launch._training_quote_binding(
        values["profile"],
        values["binding"],
        values["runtime_environment"],
        values["timeout_seconds"],
    )

    assert actual == {
        "provider_id": expected.provider.provider_id,
        "profile_ref": expected.provider.profile_ref,
        "account_ref": expected.scope.account_ref,
        "namespace_ref": expected.scope.namespace_ref,
        "resource_digest": expected.resource_digest,
        "timeout_seconds": values["timeout_seconds"],
    }


def test_quote_local_input_failure_precedes_credentials_and_provider_access(
    monkeypatch, capsys
):
    def reject(*args, **kwargs):
        raise launch.ModalChatLauncherError("modal_chat_source_invalid")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid local quote inputs reached credentials or provider access")

    monkeypatch.setattr(launch, "check_inputs", reject)
    monkeypatch.setattr(launch, "_client", forbidden)
    monkeypatch.setattr(launch, "quote_training", forbidden)
    assert (
        launch.main(
            [
                "--project-root",
                "/consumer",
                "--configuration",
                "configuration/smoke.json",
                "--mode",
                "quote-training",
                "--modal-profile",
                "selected",
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    document = json.loads(output)
    assert document["phase"] == "INPUTS"
    assert document["authorizing"] is False
    assert "modal_chat_source_invalid" not in output


def test_quote_training_reads_scope_and_rates_without_effectful_surfaces(
    monkeypatch,
):
    from tests.examples.test_modal_chat_deployment import _profile

    client = object()
    calls = []

    class Billing:
        def rates(self):
            calls.append(("rates",))
            return {
                "gpu_hour_cost_a10g": Decimal("1.10000"),
                "cpu_hour_cost": Decimal("0.04730"),
                "mem_gib_hour_cost": Decimal("0.00800"),
            }

    class Workspace:
        name = "workspace-a"
        object_id = "ws-a"
        is_hydrated = False
        billing = Billing()

        def hydrate(self, supplied_client):
            assert supplied_client is client
            self.is_hydrated = True
            calls.append(("workspace", "hydrate"))

    class Environment:
        name = "environment-a"
        object_id = "en-a"
        is_hydrated = False

        def hydrate(self, supplied_client):
            assert supplied_client is client
            self.is_hydrated = True
            calls.append(("environment", "hydrate"))

    class SDK:
        __version__ = "1.5.4"

        class Workspace:
            @staticmethod
            def from_context(*, client):
                calls.append(("workspace", "from_context"))
                return Workspace()

        class Environment:
            @staticmethod
            def from_name(name, *, create_if_missing, client):
                calls.append(("environment", "from_name", name, create_if_missing))
                assert create_if_missing is False
                return Environment()

    def forbidden(*args, **kwargs):
        pytest.fail("quote crossed an effectful or stateful boundary")

    for name in (
        "_check_launcher_python",
        "_model_token",
        "_private_directory",
        "ModalChatStorage",
        "ModalChatProvisioner",
        "ModalChatOwnedDeployment",
        "submit_training_once",
        "execute",
    ):
        monkeypatch.setattr(launch, name, forbidden)
    monkeypatch.setattr(launch, "_client", lambda sdk, profile: client)
    monkeypatch.setitem(sys.modules, "modal", SDK)
    settings = SimpleNamespace(
        environment_name="environment-a",
        profile=_profile(),
        runtime_environment={
            "LANG": "C.UTF-8",
            "PATH": "/opt/conda/bin:/usr/bin:/bin",
        },
        training_timeout_seconds=3600,
        maximum_training_cost_minor_units=1,
        canonical_bytes=b'{"fixture":"quote"}',
    )
    source = SimpleNamespace(
        project_source=SimpleNamespace(commit="a" * 40),
        engine_source=SimpleNamespace(commit="b" * 40),
    )

    result = json.loads(
        launch.quote_training(settings, source, profile="selected-profile")
    )

    assert result["schema_version"] == (
        "synaptic-modal-chat-training-quote-observation/v1"
    )
    assert result["authorizing"] is False
    assert result["within_configured_operator_maximum"] is False
    assert result["calculation"]["gpu_only_timeout_estimate_minor_units"] == 110
    assert result["calculation"]["excluded_billing_dimensions"] == [
        "build",
        "cpu",
        "memory",
        "storage",
        "usage-beyond-timeout",
    ]
    assert any(call == ("rates",) for call in calls)
    assert all(
        call[-1] is False
        for call in calls
        if call[:2] == ("environment", "from_name")
    )


@pytest.mark.parametrize(
    ("observe", "artifact_streaming"),
    ((False, False), (False, True), (True, False)),
)
def test_public_chat_capability_denial_precedes_credentials_storage_and_cloud(
    monkeypatch, observe, artifact_streaming
):
    # Interpreter denial is covered separately; isolate the next admission gate.
    monkeypatch.setattr(launch, "_check_launcher_python", lambda: None)
    monkeypatch.setattr(
        coordinator_adapter,
        "_descriptor",
        lambda: SimpleNamespace(
            capabilities=SimpleNamespace(
                observe=observe, artifact_streaming=artifact_streaming
            )
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("capability denial crossed a pre-cloud boundary")

    monkeypatch.setattr(launch, "_private_directory", forbidden)
    monkeypatch.setattr(launch, "_model_token", forbidden)
    monkeypatch.setattr(launch, "_client", forbidden)
    with pytest.raises(
        launch.ModalChatLauncherError,
        match="modal_chat_public_capabilities_unqualified",
    ):
        launch.execute(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            mode="train-chat",
            profile="selected",
            hf_token_env_file=None,
            emit=forbidden,
        )


def test_public_chat_exact_read_capabilities_pass_the_precloud_gate(
    monkeypatch,
):
    monkeypatch.setattr(launch, "_check_launcher_python", lambda: None)
    monkeypatch.setattr(
        coordinator_adapter,
        "_descriptor",
        lambda: SimpleNamespace(
            capabilities=SimpleNamespace(observe=True, artifact_streaming=True)
        ),
    )

    class GatePassed(RuntimeError):
        pass

    def reached_local_storage(*args, **kwargs):
        raise GatePassed

    monkeypatch.setattr(launch, "_private_directory", reached_local_storage)
    monkeypatch.setitem(
        sys.modules,
        "examples.modal_chat.training",
        SimpleNamespace(
            ModalChatRunIdentity=object,
            compose_modal_training_host=lambda **kwargs: None,
        ),
    )
    with pytest.raises(GatePassed):
        launch.execute(
            SimpleNamespace(),
            SimpleNamespace(state_root=Path("/unused")),
            SimpleNamespace(attempt_ref="fixture"),
            SimpleNamespace(),
            mode="train-chat",
            profile="selected",
            hf_token_env_file=None,
            emit=lambda value: None,
        )


def test_settings_or_source_rejection_precedes_execute(monkeypatch, capsys):
    def rejected(*args, **kwargs):
        raise launch.ModalChatLauncherError("modal_chat_source_invalid")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid local inputs reached execution")

    monkeypatch.setattr(launch, "check_inputs", rejected)
    monkeypatch.setattr(launch, "execute", forbidden)
    assert (
        launch.main(
            [
                "--project-root",
                "/consumer",
                "--configuration",
                "configuration/smoke.json",
                "--mode",
                "train-chat",
                "--modal-profile",
                "selected",
            ]
        )
        == 1
    )
    result = capsys.readouterr()
    assert result.err == ""
    assert "FAILED" in result.out


def test_parser_denies_unknown_execution_mode_before_input_or_cloud_work(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        launch,
        "check_inputs",
        lambda *args, **kwargs: pytest.fail("invalid mode reached input inspection"),
    )
    monkeypatch.setattr(
        launch,
        "execute",
        lambda *args, **kwargs: pytest.fail("invalid mode reached execution"),
    )
    assert (
        launch.main(
            [
                "--project-root",
                "/consumer",
                "--configuration",
                "configuration/smoke.json",
                "--mode",
                "unqualified-chat",
            ]
        )
        == 1
    )
    result = capsys.readouterr()
    assert result.err == ""
    assert "unqualified-chat" not in result.out


def test_execute_itself_denies_unknown_mode_before_storage_or_cloud(monkeypatch):

    def forbidden(*args, **kwargs):
        pytest.fail("unknown mode crossed the execution boundary")

    monkeypatch.setattr(launch, "_private_directory", forbidden)
    with pytest.raises(
        launch.ModalChatLauncherError, match="modal_chat_arguments_invalid"
    ):
        launch.execute(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            mode="unknown",
            profile="selected",
            hf_token_env_file=None,
            emit=forbidden,
        )
