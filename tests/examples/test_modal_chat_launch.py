"""Provider-free launcher ordering, confidentiality, polling and cleanup tests."""

from types import SimpleNamespace
from collections import namedtuple
import json
from pathlib import Path

import pytest

from examples.modal_chat import launch
from examples.modal_chat.storage import ModalChatStorage
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.providers.modal.facade import ModalFunctionCallState


@pytest.mark.parametrize(
    "implementation,version,release,accepted",
    [
        ("cpython", (3, 11, 14), "final", True),
        ("cpython", (3, 12, 9), "final", False),
        ("cpython", (3, 11, 13), "final", False),
        ("pypy", (3, 11, 14), "final", False),
        ("cpython", (3, 11, 14), "candidate", False),
    ],
)
def test_launcher_python_matches_packaged_training_pin(
    monkeypatch, implementation, version, release, accepted
):
    Version = namedtuple("Version", "major minor micro releaselevel serial")
    monkeypatch.setattr(
        launch,
        "sys",
        SimpleNamespace(
            implementation=SimpleNamespace(name=implementation),
            version_info=Version(*version, release, 0),
        ),
    )
    if accepted:
        launch._check_launcher_python()
    else:
        with pytest.raises(launch.ModalChatLauncherError, match="python_mismatch"):
            launch._check_launcher_python()


@pytest.mark.parametrize("mode", ["qualify-training", "train-chat"])
def test_python_rejection_precedes_credentials_state_and_provisioning(
    monkeypatch, mode
):
    def reject():
        raise launch.ModalChatLauncherError("modal_chat_launcher_python_mismatch")

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible interpreter reached credentials or state")

    monkeypatch.setattr(launch, "_check_launcher_python", reject)
    for name in (
        "_model_token",
        "_client",
        "_private_directory",
        "ModalChatStorage",
        "ModalChatProvisioner",
    ):
        monkeypatch.setattr(launch, name, forbidden)
    with pytest.raises(launch.ModalChatLauncherError, match="python_mismatch"):
        launch.execute(
            None,
            None,
            None,
            None,
            mode=mode,
            profile="selected",
            hf_token_env_file=None,
            emit=forbidden,
        )


def test_check_is_default_and_never_constructs_credentials_or_executes(
    monkeypatch, capsys
):
    monkeypatch.setattr(launch, "check_inputs", lambda *args, **kwargs: (1, 2, 3, 4))

    def forbidden(*args, **kwargs):
        pytest.fail("check mode entered execution")

    monkeypatch.setattr(launch, "execute", forbidden)
    assert (
        launch.main(
            [
                "--project-root",
                "/consumer",
                "--configuration",
                "configuration/smoke.json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "LOCAL_INPUTS_CHECKED"


def test_quote_training_dispatches_without_entering_paid_execute(monkeypatch, capsys):
    values = (object(), object(), object(), object())
    monkeypatch.setattr(launch, "check_inputs", lambda *args, **kwargs: values)

    def quote(settings, source, *, profile):
        assert (settings, source, profile) == (values[2], values[3], "selected")
        return b'{"authorizing":false,"status":"TRAINING_QUOTE_OBSERVED"}'

    monkeypatch.setattr(launch, "quote_training", quote)
    monkeypatch.setattr(
        launch,
        "execute",
        lambda *args, **kwargs: pytest.fail("quote entered paid execution"),
    )
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
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "authorizing": False,
        "status": "TRAINING_QUOTE_OBSERVED",
    }


def test_quote_training_rejects_irrelevant_model_token_file(monkeypatch, capsys):
    monkeypatch.setattr(
        launch, "check_inputs", lambda *args, **kwargs: (1, 2, 3, 4)
    )
    monkeypatch.setattr(
        launch,
        "quote_training",
        lambda *args, **kwargs: pytest.fail("invalid quote arguments reached rates"),
    )
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
                "--hf-token-env-file",
                "/private/token.env",
            ]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert result["phase"] == "INPUTS"
    assert result["authorizing"] is False


def test_quote_training_failure_reports_closed_quote_phase(monkeypatch, capsys):
    monkeypatch.setattr(
        launch, "check_inputs", lambda *args, **kwargs: (1, 2, 3, 4)
    )

    def fail(*args, **kwargs):
        raise RuntimeError("private-provider-rate-message")

    monkeypatch.setattr(launch, "quote_training", fail)
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
    result = json.loads(output)
    assert result["phase"] == "QUOTE_TRAINING"
    assert result["authorizing"] is False
    assert result["retry_authorized"] is False
    assert "private-provider-rate-message" not in output


def test_bad_arguments_never_echo_supplied_values(capsys):
    assert launch.main(["--not-a-real-flag", "private-argument-value"]) == 1
    output = capsys.readouterr()
    assert "private-argument-value" not in output.out + output.err
    assert json.loads(output.out)["retry_authorized"] is False


def test_main_reports_closed_chat_phase_and_suppressed_context(monkeypatch, capsys):
    monkeypatch.setattr(launch, "check_inputs", lambda *args, **kwargs: (1, 2, 3, 4))

    def fail(*args, emit, **kwargs):
        emit("CHAT_OPEN")
        try:
            raise ValueError("private-inner-value")
        except ValueError:
            raise launch.ModalChatLauncherError("private-outer-value") from None

    monkeypatch.setattr(launch, "execute", fail)
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
    output = capsys.readouterr()
    failure = json.loads(output.out.splitlines()[-1])
    assert failure["phase"] == "CHAT_OPEN"
    assert failure["authorizing"] is False
    assert failure["retry_authorized"] is False
    assert [item["exception_class"] for item in failure["exception_chain"]] == [
        "OTHER",
        "ValueError",
    ]
    assert "private-" not in output.out + output.err


@pytest.mark.parametrize("primary_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_cleanup_guard_preserves_primary_and_records_secondary(
    monkeypatch, primary_type
):
    primary = primary_type("private-primary")
    cleanup_calls, saved, emitted = [], [], []

    def fail_cleanup(*args):
        cleanup_calls.append(args)
        raise RuntimeError("private-cleanup")

    monkeypatch.setattr(launch, "_confirm_chat_cleanup", fail_cleanup)
    monkeypatch.setattr(launch, "_record", lambda *args: saved.append(args))
    with pytest.raises(primary_type) as caught:
        with launch._chat_cleanup_guard(
            "graph", "storage", "session", emit=emitted.append
        ):
            raise primary
    assert caught.value is primary
    assert cleanup_calls == [("graph", "storage", "session")]
    assert len(saved) == 1
    assert saved[0][1:3] == ("launch-chat-cleanup-failures", "session")
    assert saved[0][3]["phase"] == "CHAT_CLEANUP"
    assert saved[0][3]["authorizing"] is False
    assert "private-" not in json.dumps(saved[0][3])
    assert emitted == []


def test_cleanup_guard_propagates_primary_cleanup_failure(monkeypatch):
    error = RuntimeError("private-cleanup")
    emitted = []

    def fail_cleanup(*args):
        raise error

    monkeypatch.setattr(launch, "_confirm_chat_cleanup", fail_cleanup)
    with pytest.raises(RuntimeError) as caught:
        with launch._chat_cleanup_guard(None, None, "session", emit=emitted.append):
            pass
    assert caught.value is error
    assert emitted == ["CHAT_CLEANUP"]


def test_cleanup_storage_failure_cannot_replace_original_error(monkeypatch):
    primary = ValueError("private-primary")

    def fail(*args):
        raise RuntimeError("private-secondary")

    monkeypatch.setattr(launch, "_confirm_chat_cleanup", fail)
    monkeypatch.setattr(launch, "_record", fail)
    with pytest.raises(ValueError) as caught:
        with launch._chat_cleanup_guard(None, None, "session", emit=lambda value: None):
            raise primary
    assert caught.value is primary


def test_verified_snapshot_preserves_immutable_queued_ownership(monkeypatch, tmp_path):
    from tests.examples.test_modal_chat_qualification import _fixture
    from examples.modal_chat.qualification import qualify_modal_chat_run
    from tuner.execution.coordinator_v1.model import WorkflowPhaseV1

    host, store, reader, storage, run = _fixture(monkeypatch, tmp_path)
    with storage:
        queued = launch._snapshot_training(host, storage, run)
        ownership = storage.catalog("launch-run-ownership", encode=bytes, decode=bytes)
        original = ownership.resolve(run.run_id)
        qualify_modal_chat_run(host=host, storage=storage, run=run)
        verified = launch._snapshot_workflow(host, storage, run)
        assert verified.phase is WorkflowPhaseV1.VERIFIED
        assert verified.record_digest != queued.record_digest
        assert ownership.resolve(run.run_id) == original
        workflows = storage.catalog("launch-workflows", encode=bytes, decode=bytes)
        assert json.loads(workflows.resolve(queued.record_digest)) == queued.to_dict()
        assert (
            json.loads(workflows.resolve(verified.record_digest)) == verified.to_dict()
        )


def test_selected_modal_profile_disables_environment_override(monkeypatch):
    pytest.importorskip("modal")
    from modal.config import config

    calls = []

    def get(name, **kwargs):
        calls.append((name, kwargs))
        return "fixture-id" if name == "token_id" else "fixture-secret"

    monkeypatch.setattr(config, "get", get)
    returned = object()

    class Client:
        @staticmethod
        def from_credentials(token_id, token_secret):
            assert (token_id, token_secret) == ("fixture-id", "fixture-secret")
            return returned

    sdk = SimpleNamespace(__version__="1.5.4", Client=Client)
    assert launch._client(sdk, "selected") is returned
    assert calls == [
        ("token_id", {"profile": "selected", "use_env": False}),
        ("token_secret", {"profile": "selected", "use_env": False}),
    ]


def test_private_output_creation_does_not_chmod_existing_shared_directory(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    with pytest.raises(launch.ModalChatLauncherError):
        launch._private_directory(shared)
    assert shared.stat().st_mode & 0o777 == 0o755
    private = launch._private_directory(tmp_path / "new" / "private")
    assert private.stat().st_mode & 0o777 == 0o700


def test_polling_uses_only_exact_known_call_pending_hint(monkeypatch):
    states = iter([ModalFunctionCallState.PENDING, ModalFunctionCallState.UNKNOWN])
    calls, sleeps, emitted = [], [], []

    class Facade:
        def observe_known_call_pending(self, reference):
            calls.append(reference)
            return next(states)

    workflow = SimpleNamespace(
        provider_run_ref=SimpleNamespace(
            reference=SimpleNamespace(provider_job_ref="fc-exact")
        )
    )
    monkeypatch.setattr(launch, "_snapshot_training", lambda *args: workflow)
    monkeypatch.setattr(launch.time, "sleep", sleeps.append)
    launch._wait_training(
        None,
        SimpleNamespace(facade=lambda: Facade()),
        None,
        TrainingRunRef("run", "project"),
        timeout_seconds=300,
        emit=emitted.append,
    )
    assert calls == ["fc-exact", "fc-exact"]
    assert sleeps == [15]
    assert emitted == ["TRAINING_PENDING"]


def _graph(*, sandboxes=(), close_error=False, submitted=True, unknown=False):
    class Ownership:
        submit_command_digest = "a" * 64

        def __init__(self, sandbox):
            self.sandbox = sandbox

        def close(self):
            if close_error:
                raise RuntimeError("private-provider-message")

    owned = [Ownership(item) for item in sandboxes]
    if unknown:
        owned.append(Ownership(None))
    return SimpleNamespace(
        runtime=SimpleNamespace(owned_lease=None),
        pending_ownership=lambda: tuple(owned),
        catalog=SimpleNamespace(
            submit_digests=lambda: ("a" * 64,) if submitted else ()
        ),
    )


class Sandbox:
    def __init__(self, object_id, returncode, fail=False):
        self.object_id, self.returncode, self.fail = object_id, returncode, fail

    def poll(self):
        if self.fail:
            raise RuntimeError("private-provider-message")
        return self.returncode


@pytest.mark.parametrize("unknown,fail", [(True, False), (False, True)])
def test_cleanup_uncertainty_is_saved_even_if_close_or_poll_raises(
    tmp_path, unknown, fail, capsys
):
    private = launch._private_directory(tmp_path / "private")
    with ModalChatStorage(private / "state.sqlite3", "consumer") as storage:
        graph = _graph(
            sandboxes=(Sandbox("sb-first", 0, fail), Sandbox("sb-second", 0)),
            close_error=True,
            unknown=unknown,
        )
        with pytest.raises(launch.ModalChatLauncherError, match="cleanup_unconfirmed"):
            launch._confirm_chat_cleanup(graph, storage, "chat")
        result = json.loads(
            storage.catalog("launch-chat-cleanup", encode=bytes, decode=bytes).resolve(
                "chat"
            )
        )
        assert result["provider_shutdown_proof"] is False
        assert result["unresolved_creation_or_cleanup"] is True
        assert result["known_instances"][1]["provider_shutdown_proof"] is True
    assert "private-provider-message" not in capsys.readouterr().out


def test_cleanup_requires_exact_stopped_readback(tmp_path):
    private = launch._private_directory(tmp_path / "private")
    with ModalChatStorage(private / "state.sqlite3", "consumer") as storage:
        launch._confirm_chat_cleanup(
            _graph(sandboxes=(Sandbox("sb-exact", 0),)), storage, "chat"
        )
        result = json.loads(
            storage.catalog("launch-chat-cleanup", encode=bytes, decode=bytes).resolve(
                "chat"
            )
        )
        assert result["provider_shutdown_proof"] is True
        assert result["known_instances"][0]["sandbox_id"] == "sb-exact"


def test_preallocation_failure_does_not_invent_cleanup_ownership(tmp_path):
    private = launch._private_directory(tmp_path / "private")
    with ModalChatStorage(private / "state.sqlite3", "consumer") as storage:
        launch._confirm_chat_cleanup(_graph(submitted=False), storage, "chat")
        result = json.loads(
            storage.catalog("launch-chat-cleanup", encode=bytes, decode=bytes).resolve(
                "chat"
            )
        )
        assert result["provider_shutdown_proof"] is False
        assert result["submit_command_retained"] is False
        assert result["unresolved_creation_or_cleanup"] is False


def test_poll_without_stopped_result_does_not_claim_cleanup(tmp_path):
    private = launch._private_directory(tmp_path / "private")
    with ModalChatStorage(private / "state.sqlite3", "consumer") as storage:
        with pytest.raises(launch.ModalChatLauncherError, match="cleanup_unconfirmed"):
            launch._confirm_chat_cleanup(
                _graph(sandboxes=(Sandbox("sb-live", None),)), storage, "chat"
            )


def test_integrated_training_and_qualification_are_importable():
    from examples.modal_chat.training import compose_modal_training_host
    from examples.modal_chat.qualification import qualify_modal_chat_run

    assert callable(compose_modal_training_host)
    assert callable(qualify_modal_chat_run)
