"""Credential-free executable-boundary tests; no Modal or ML runtime."""

from contextlib import contextmanager
from dataclasses import replace
import io
import json
import threading
from types import SimpleNamespace

import pytest

from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from Evaluator.protocols import BackendResponse
from tuner.inference.run_chat import PreparedModelIdentity
from tuner.execution.providers.modal.inference_bootstrap import ModalChatWorkerSession
from tests.execution.providers.test_modal_inference_preparation import _document
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal import inference_entrypoint as entry
from tuner.execution.providers.modal.inference_channel import (
    decode_modal_chat_frame,
    encode_modal_chat_frame,
)
from tuner.execution.providers.modal.inference_preparation import (
    ModalInferencePreparationConfig,
)
from tuner.execution.providers.modal.inference_wire import ModalChatWorkerExpectation


def _start():
    source = SimpleNamespace(
        artifact_volume_id="source-id",
        command_binding=SimpleNamespace(
            client_binding=SimpleNamespace(
                account_ref="account",
                workspace_ref="workspace",
                environment_ref="environment",
                client_ref="client",
                sdk_version="1.5.4",
            )
        ),
    )
    document = _document(
        source,
        secrets=[
            {
                "name": "chat-runtime",
                "required_keys": ["CHAT_EVIDENCE_KEY", "HF_TOKEN"],
            }
        ],
    )
    expectation = ModalChatWorkerExpectation(
        configuration_bytes=ModalInferencePreparationConfig.build(
            document
        ).canonical_bytes,
        submit_command_digest="a" * 64,
        executor_version="0.1.0",
        issuer_ref="issuer",
        audience_ref="audience",
        key_ref="chat-evidence-key",
        challenge_nonce="challenge",
        artifact_root="/mount/artifacts",
        control_root="/mount/control",
        cache_root="/mount/cache",
    )
    return entry.ModalChatStart(
        expectation, b"synthetic-not-signed", "CHAT_EVIDENCE_KEY", "HF_TOKEN"
    )


def test_start_round_trips_exact_frozen_nonsecret_bytes():
    start = _start()
    payload = entry.encode_modal_chat_start(start)
    assert entry.decode_modal_chat_start(payload) == start
    assert payload.endswith(b"\n") and b"CHAT_EVIDENCE_KEY" in payload


@pytest.mark.parametrize(
    "attack",
    (
        "unknown",
        "duplicate",
        "whitespace",
        "base64",
        "large",
        "truncated",
        "wrong_type",
    ),
)
def test_start_rejects_malformed_frames(attack):
    payload = entry.encode_modal_chat_start(_start())
    document = json.loads(payload)
    if attack == "unknown":
        document["extra"] = "invalid"
        payload = entry._canonical(document)
    elif attack == "duplicate":
        payload = b'{"model_token_key":"HF_TOKEN",' + payload[1:]
    elif attack == "whitespace":
        payload = b" " + payload
    elif attack == "base64":
        document["argument_bytes"] = "!!!"
        payload = entry._canonical(document)
    elif attack == "large":
        payload = b"x" * (entry._MAX_START_BYTES + 1)
    elif attack == "truncated":
        payload = payload[:-1]
    else:
        payload = bytearray(payload)
    with pytest.raises(
        entry.ModalChatEntrypointError, match="^modal_chat_start_invalid$"
    ):
        entry.decode_modal_chat_start(payload)


@pytest.mark.parametrize(
    "changes",
    (
        {"evidence_environment_key": "MODAL_TOKEN_SECRET"},
        {"evidence_environment_key": "HF_TOKEN"},
        {"model_token_key": None},
        {"model_token_key": "UNDECLARED_TOKEN"},
        {"argument_bytes": b""},
    ),
)
def test_start_rejects_undeclared_or_colliding_credentials(changes):
    with pytest.raises(ValueError):
        entry.encode_modal_chat_start(replace(_start(), **changes))


def test_private_image_directories_cannot_be_a_mount():
    start = _start()
    expectation = replace(
        start.expectation, artifact_root="/workspace/modal-chat/model"
    )
    with pytest.raises(ValueError):
        entry.encode_modal_chat_start(replace(start, expectation=expectation))


def test_open_but_silent_initial_stdin_is_bounded(monkeypatch):
    release = threading.Event()

    class Silent:
        def readline(self, maximum):
            release.wait(1)
            return b""

    monkeypatch.setattr(entry, "_START_WAIT_SECONDS", 0.01)
    try:
        with pytest.raises(entry.ModalChatEntrypointError, match="unavailable"):
            entry._read_start(Silent())
    finally:
        release.set()


@pytest.mark.parametrize("control", (KeyboardInterrupt, SystemExit))
def test_initial_read_preserves_control_flow(control):
    error = control()

    class Interrupted:
        def readline(self, maximum):
            raise error

    with pytest.raises(control) as caught:
        entry._read_start(Interrupted())
    assert caught.value is error


def test_invalid_signed_launch_never_reads_preparation_token_or_starts(monkeypatch):
    start = _start()
    touched = []

    class ForbiddenEnvironment(dict):
        def get(self, key, default=None):
            touched.append(key)
            raise AssertionError("environment must not be read")

    monkeypatch.setattr(entry.os, "environ", ForbiddenEnvironment())
    with pytest.raises(Exception):
        entry.run_modal_chat_entrypoint(
            io.BytesIO(entry.encode_modal_chat_start(start)), io.BytesIO()
        )
    assert touched == []


def test_entrypoint_wires_exact_bootstrap_session_and_credential_free_child_environment(
    monkeypatch,
):
    start = _start()
    config = ModalInferencePreparationConfig.parse(
        start.expectation.configuration_bytes
    )
    calls = []

    class Admission:
        claim = canonical_bytes({"session_id": "session-a"})
        configuration = config

    def admit(*args, **kwargs):
        calls.append("admit")
        return Admission()

    monkeypatch.setattr(entry, "admit_modal_chat_launch", admit)
    monkeypatch.setattr(
        entry.os,
        "environ",
        {
            "HF_TOKEN": "synthetic-test-only",
            "MODAL_TOKEN_ID": "synthetic-test-only",
            "CHAT_EVIDENCE_KEY": "synthetic-test-only",
            "PYTHONPATH": "/not-forwarded",
            "PATH": "/operator/not-forwarded",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )

    class Backend:
        def chat(self, messages):
            calls.append(tuple(dict(item) for item in messages))
            return BackendResponse("hello", {}, 0.0)

    class Lease:
        cleanup_pending = False

        def close(self, **kwargs):
            calls.append("close")
            return True

    captured = {}

    @contextmanager
    def bootstrap(argument, **kwargs):
        captured.update(kwargs)
        assert calls == ["admit"]
        session = ChatSession(Backend(), Lease(), ChatSessionPolicy(1, 5, 5, 3, 4096))
        try:
            yield ModalChatWorkerSession(
                session,
                PreparedModelIdentity("fixture/model", "a" * 40, "a" * 40, "full"),
            )
        finally:
            session.close()
            session.wait_closed(1)

    monkeypatch.setattr(entry, "open_modal_chat_worker", bootstrap)
    import hashlib

    binding = {
        "schema_version": "synaptic-modal-chat-channel/v1",
        "session_id": "session-a",
        "launch_digest": hashlib.sha256(start.argument_bytes).hexdigest(),
    }
    request = encode_modal_chat_frame(
        binding | {"kind": "chat", "request_id": 1, "content": "hi"}, 8192
    )
    stop = encode_modal_chat_frame(binding | {"kind": "stop", "request_id": 2}, 8192)
    output = io.BytesIO()
    entry.run_modal_chat_entrypoint(
        io.BytesIO(entry.encode_modal_chat_start(start) + request + stop), output
    )
    frames = [
        decode_modal_chat_frame(line, 65536)
        for line in output.getvalue().splitlines(keepends=True)
    ]
    assert [item["kind"] for item in frames] == ["ready", "chat", "closed"]
    assert frames[1]["content"] == "hello"
    assert captured["environment"] == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert captured["destination"] == entry._DESTINATION
    assert captured["cwd"] == entry._PRIVATE_ROOT
    assert calls == ["admit", ({"role": "user", "content": "hi"},), "close"]


@pytest.mark.parametrize("error_type", (ValueError, SystemExit))
def test_cli_suppresses_setup_output_and_exception_details(monkeypatch, error_type):
    output, errors = io.BytesIO(), io.BytesIO()
    monkeypatch.setattr(entry.sys, "stdin", io.TextIOWrapper(io.BytesIO()))
    monkeypatch.setattr(
        entry.sys, "stdout", io.TextIOWrapper(output, write_through=True)
    )
    monkeypatch.setattr(
        entry.sys, "stderr", io.TextIOWrapper(errors, write_through=True)
    )

    def failed(*args):
        print("untrusted diagnostic")
        print("untrusted diagnostic", file=entry.sys.stderr)
        raise error_type("untrusted diagnostic")

    monkeypatch.setattr(entry, "run_modal_chat_entrypoint", failed)
    assert entry.main([]) == 125
    assert output.getvalue() == errors.getvalue() == b""


def test_cli_rejects_all_arguments_without_reading_stdin():
    assert entry.main(["--credential-value"]) == 124


def test_real_hmac_launch_still_denies_missing_runtime_lock_before_preparation(
    monkeypatch, tmp_path
):
    import base64
    from tests.execution.providers import (
        test_modal_inference_preparation as preparation_cases,
    )
    from tests.execution.providers import (
        test_modal_inference_launch_integration as launch_cases,
    )
    from tests.execution.providers.modal_inference_worker_fixtures import (
        mounted_launch_case,
    )
    from tuner.execution.providers.modal import inference_bootstrap as bootstrap
    from tuner.execution.providers.modal.runtime import EnvironmentHmacAuthenticator

    # Synthetic test key only. All launch/claim bytes below are signed through
    # the actual environment-backed HMAC implementation used by the executable.
    monkeypatch.setenv("CHAT_EVIDENCE_KEY", base64.b64encode(b"x" * 32).decode("ascii"))
    original_document = preparation_cases._document

    def document(source, **changes):
        return original_document(
            source,
            **(
                changes
                | {
                    "secrets": [
                        {
                            "name": "chat-runtime",
                            "required_keys": ["CHAT_EVIDENCE_KEY", "HF_TOKEN"],
                        }
                    ]
                }
            ),
        )

    monkeypatch.setattr(preparation_cases, "_document", document)
    monkeypatch.setattr(
        launch_cases,
        "LaunchAuthenticator",
        lambda: EnvironmentHmacAuthenticator(
            environment_key="CHAT_EVIDENCE_KEY",
            key_ref="chat-evidence-key",
        ),
    )
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="full")
    monkeypatch.setattr(entry, "_UTCClock", lambda: case.kwargs["clock"])
    touched = []

    def prepare(*args, **kwargs):
        touched.append(True)
        raise AssertionError("preparation must not run")

    monkeypatch.setattr(bootstrap, "prepare_modal_chat_worker", prepare)
    start = entry.ModalChatStart(
        case.kwargs["expectation"],
        case.envelope.argument_bytes,
        "CHAT_EVIDENCE_KEY",
        "HF_TOKEN",
    )
    with pytest.raises(
        bootstrap.ModalInferenceBootstrapError,
        match="^modal_inference_bootstrap_invalid$",
    ):
        entry.run_modal_chat_entrypoint(
            io.BytesIO(entry.encode_modal_chat_start(start)), io.BytesIO()
        )
    assert touched == []
