from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time
import threading

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "scripts" / "capture_modal_inference_runtime.py"
SPEC = importlib.util.spec_from_file_location("capture_modal_inference_runtime", PATH)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)

IMAGE = "docker.io/vllm/vllm-openai@sha256:" + "a" * 64
COMMIT = "b" * 40


def _candidate() -> bytes:
    return (
        json.dumps(
            {
                "distributions": {"vllm": "0.17.1"},
                "operator_selection": {"image": IMAGE, "source_commit": COMMIT},
                "python": {
                    "executable": "/usr/bin/python3",
                    "executable_sha256": "c" * 64,
                    "implementation": "cpython",
                    "version": "3.12.9",
                },
                "requirements": {
                    "modal": {"present": False, "version": None},
                    "vllm": {"present": True, "version": "0.17.1"},
                },
                "schema_version": "synaptic-modal-inference-runtime-inspection-candidate/v1",
                "status": "CANDIDATE_ONLY",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


class Image:
    def __init__(self, calls):
        self.calls = calls

    def entrypoint(self, value):
        self.calls.append(("entrypoint", value))
        return self

    def add_local_file(self, local, remote, *, copy):
        self.calls.append(("file", Path(local).name, remote, copy))
        return self


class Sandbox:
    object_id = "sb-123"
    returncode = 0

    def __init__(self, raw):
        self.stdout = [raw[:7], raw[7:]]
        self.terminate_calls = []
        self.poll_calls = 0

    def wait(self, *, raise_on_termination):
        assert raise_on_termination is False
        return 0

    def terminate(self, *, wait):
        self.terminate_calls.append(wait)

    def poll(self):
        self.poll_calls += 1
        return 0


def _sdk(raw=None, *, create_error=None):
    calls = []
    sandbox = Sandbox(_candidate() if raw is None else raw)

    class App:
        @staticmethod
        def lookup(name, **kwargs):
            calls.append(("lookup", name, kwargs))
            return object()

    class ImageFactory:
        @staticmethod
        def from_registry(value):
            calls.append(("image", value))
            return Image(calls)

    class SandboxFactory:
        @staticmethod
        def create(*args, **kwargs):
            calls.append(("create", args, kwargs))
            if create_error is not None:
                raise create_error
            return sandbox

    return (
        SimpleNamespace(
            __version__="1.5.4", App=App, Image=ImageFactory, Sandbox=SandboxFactory
        ),
        sandbox,
        calls,
    )


def test_capture_uses_existing_app_cpu_bounds_and_closes_exact_sandbox():
    sdk, sandbox, calls = _sdk()
    result = capture.capture_modal_inference_runtime(
        sdk=sdk,
        client=object(),
        app_name="existing-app",
        environment_name="production",
        image=IMAGE,
        source_commit=COMMIT,
    )
    value = json.loads(result.canonical_bytes)
    assert value["candidate"]["requirements"]["modal"]["present"] is False
    assert (
        value["inspection_script_sha256"]
        == capture._script_source(capture._script_path())[1]
    )
    assert value["operator_selection_only"] is True
    lookup = calls[0]
    assert lookup[2]["create_if_missing"] is False
    create_call = next(call for call in calls if call[0] == "create")
    options = create_call[2]
    assert options["cpu"] == 1.0
    assert options["memory"] == 2048
    assert options["timeout"] == 300
    assert options["idle_timeout"] == 300
    assert options["block_network"] is True
    assert (
        "gpu" not in options and "volumes" not in options and "secrets" not in options
    )
    assert sandbox.terminate_calls == [False]
    assert sandbox.poll_calls == 1


def test_create_ambiguity_has_no_retry_or_resource_adoption():
    sdk, _, calls = _sdk(create_error=RuntimeError("private"))
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError) as caught:
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing-app",
            environment_name="production",
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert caught.value.cleanup_lease is None
    assert caught.value.sandbox_id is None
    assert sum(call[0] == "create" for call in calls) == 1
    assert "private" not in str(caught.value)


def test_output_failure_retains_exact_cleanup_lease():
    sdk, sandbox, _ = _sdk(raw=b"not-json")
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError) as caught:
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing-app",
            environment_name="production",
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert caught.value.cleanup_lease is sandbox
    assert caught.value.sandbox_id == "sb-123"
    assert caught.value.cleanup_requested is True
    assert caught.value.cleanup_confirmed is True


@pytest.mark.parametrize("version", ["1.5.3", None])
def test_sdk_version_denied_before_lookup(version):
    sdk, _, calls = _sdk()
    sdk.__version__ = version
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError):
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing-app",
            environment_name="production",
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert calls == []


def test_unbounded_output_denied_and_cleanup_requested():
    sdk, sandbox, _ = _sdk(raw=b"x" * (capture._MAX_CAPTURE_BYTES + 1))
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError):
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing-app",
            environment_name="production",
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert sandbox.terminate_calls == [False]


def test_late_create_is_retained_and_terminated_without_retry(monkeypatch):
    sdk, sandbox, calls = _sdk()
    original = sdk.Sandbox.create

    def delayed(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    sdk.Sandbox.create = delayed
    monkeypatch.setattr(capture, "_HOST_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError) as caught:
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing-app",
            environment_name="production",
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert caught.value.reason_code == "sandbox_create_ambiguous"
    assert type(caught.value.cleanup_lease).__name__ == "_PendingOperation"
    time.sleep(0.1)
    assert sandbox.terminate_calls == [False]
    assert sum(call[0] == "create" for call in calls) == 1


def test_pending_completion_then_abandon_claims_cleanup_once():
    completed = threading.Event()
    cleaned = []
    value = object()
    pending = capture._PendingOperation(
        lambda: (completed.set(), value)[1], lambda item: cleaned.append(item)
    )
    assert completed.wait(1)
    pending.worker.join(1)
    pending.abandon()
    pending.abandon()
    assert cleaned == [value]


def test_pending_abandon_then_completion_claims_cleanup_once():
    release = threading.Event()
    cleaned = []
    value = object()
    pending = capture._PendingOperation(
        lambda: (release.wait(), value)[1], lambda item: cleaned.append(item)
    )
    pending.abandon()
    release.set()
    pending.worker.join(1)
    assert cleaned == [value]


def test_control_interruption_abandons_pending_operation(monkeypatch):
    release = threading.Event()
    cleaned = []
    original_get = capture.queue.Queue.get

    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(capture.queue.Queue, "get", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        capture._bounded_call(
            lambda: (release.wait(), "sandbox")[1],
            deadline=time.monotonic() + 1,
            timeout_code="timeout",
            late_result=lambda item: cleaned.append(item),
        )
    pending = caught.value.cleanup_lease
    monkeypatch.setattr(capture.queue.Queue, "get", original_get)
    release.set()
    pending.worker.join(1)
    assert cleaned == ["sandbox"]


def test_control_interrupt_between_create_return_and_local_assignment_retains_sandbox(
    monkeypatch,
):
    sdk, sandbox, calls = _sdk()
    original = capture._bounded_call

    def interrupt_after_return(operation, **kwargs):
        value = original(operation, **kwargs)
        if kwargs.get("ownership") is not None:
            raise KeyboardInterrupt()
        return value

    monkeypatch.setattr(capture, "_bounded_call", interrupt_after_return)
    with pytest.raises(KeyboardInterrupt) as caught:
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing-app",
            environment_name="production",
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert caught.value.cleanup_lease is sandbox
    assert caught.value.sandbox_id is None
    assert caught.value.cleanup_requested is True
    assert caught.value.cleanup_confirmed is True
    assert sandbox.terminate_calls == [False]
    assert sum(call[0] == "create" for call in calls) == 1


def test_control_interrupt_after_create_worker_start_abandons_for_late_cleanup(
    monkeypatch,
):
    sdk, sandbox, calls = _sdk()
    release = threading.Event()
    original_create = sdk.Sandbox.create
    original_start = capture.threading.Thread.start
    starts = 0

    def delayed_create(*args, **kwargs):
        release.wait()
        return original_create(*args, **kwargs)

    def start_then_interrupt(self):
        nonlocal starts
        result = original_start(self)
        if self.name == "modal-runtime-capture":
            starts += 1
            if starts == 2:
                raise KeyboardInterrupt()
        return result

    sdk.Sandbox.create = delayed_create
    monkeypatch.setattr(capture.threading.Thread, "start", start_then_interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing-app",
            environment_name="production",
            image=IMAGE,
            source_commit=COMMIT,
        )
    pending = caught.value.cleanup_lease
    assert type(pending).__name__ == "_PendingOperation"
    release.set()
    pending.worker.join(1)
    assert sandbox.terminate_calls == [False]
    assert sum(call[0] == "create" for call in calls) == 1


def test_duplicate_json_key_is_denied_and_cleanup_runs():
    raw = (
        b'{"status":"CANDIDATE_ONLY","status":"CANDIDATE_ONLY",'
        b'"operator_selection":{},"requirements":{},"distributions":{},'
        b'"python":{},"schema_version":"x"}'
    )
    sdk, sandbox, _ = _sdk(raw=raw)
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError):
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing-app",
            environment_name="production",
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert sandbox.terminate_calls == [False]


def test_cli_missing_credentials_is_closed(monkeypatch, capsys):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    assert (
        capture.main(
            [
                "--app",
                "a",
                "--environment",
                "e",
                "--image",
                IMAGE,
                "--source-commit",
                COMMIT,
            ]
        )
        == 125
    )
    assert "MODAL_TOKEN" not in capsys.readouterr().err


def test_cli_help_is_normal_success(capsys):
    assert capture.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "INTERRUPTED" not in captured.err


@pytest.mark.parametrize(
    "error,code", [(SystemExit("private"), 125), (KeyboardInterrupt(), 130)]
)
def test_cli_control_flow_has_fixed_closed_status(monkeypatch, capsys, error, code):
    monkeypatch.setattr(
        capture,
        "build_parser",
        lambda: SimpleNamespace(parse_args=lambda argv: (_ for _ in ()).throw(error)),
    )
    assert capture.main([]) == code
    captured = capsys.readouterr()
    assert "private" not in captured.err
    assert json.loads(captured.err)["status"] == "INTERRUPTED"
