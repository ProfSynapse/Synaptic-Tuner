from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
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


def _diagnostic() -> bytes:
    return (
        json.dumps(
            {
                "distributions": [
                    {
                        "name": "demo",
                        "version": "1.0",
                        "metadata_path": "/opt/site/demo-1.dist-info",
                    },
                    {
                        "name": "demo",
                        "version": "2.0",
                        "metadata_path": "/usr/site/demo-2.dist-info",
                    },
                ],
                "operator_selection": {"image": IMAGE, "source_commit": COMMIT},
                "schema_version": "synaptic-modal-inference-distribution-diagnostic/v1",
                "status": "DIAGNOSTIC_ONLY",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def test_diagnostic_capture_is_opt_in_and_never_candidate_admission():
    sdk, sandbox, calls = _sdk(raw=_diagnostic())
    result = capture.capture_modal_inference_runtime(
        sdk=sdk,
        client=object(),
        app_name="existing-app",
        environment_name="isolated",
        image=IMAGE,
        source_commit=COMMIT,
        diagnose_distributions=True,
    )
    value = json.loads(result.canonical_bytes)
    assert "candidate" not in value
    assert value["diagnostic"]["status"] == "DIAGNOSTIC_ONLY"
    assert (
        value["schema_version"]
        == "synaptic-modal-inference-distribution-diagnostic-capture/v1"
    )
    create = next(call for call in calls if call[0] == "create")
    assert create[1][-1] == "--diagnose-distributions"
    assert create[2]["block_network"] is True
    assert sandbox.terminate_calls == [False]
    assert sandbox.poll_calls == 1
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError):
        capture._parse_candidate(_diagnostic(), image=IMAGE, source_commit=COMMIT)
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError):
        capture._parse_diagnostic(_candidate(), image=IMAGE, source_commit=COMMIT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "private/name"),
        ("version", "private\nversion"),
        ("metadata_path", "relative/private"),
        ("metadata_path", "/opt/../private"),
        ("metadata_path", "/opt/private\npath"),
    ],
)
def test_diagnostic_parser_rejects_unvalidated_metadata(field, value):
    body = json.loads(_diagnostic())
    body["distributions"][0][field] = value
    raw = (
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError) as caught:
        capture._parse_diagnostic(raw, image=IMAGE, source_commit=COMMIT)
    assert value not in str(caught.value)


def test_stopped_diagnostic_read_never_admits_candidate():
    sandbox = _ReadSandbox(returncode=0, stdout=_diagnostic(), stderr=b"")
    sdk, calls = _read_sdk(sandbox)
    raw = capture.read_modal_inference_runtime_sandbox(
        sdk=sdk,
        client=object(),
        sandbox_id="sb-exact",
        image=IMAGE,
        source_commit=COMMIT,
        diagnose_distributions=True,
    )
    report = json.loads(raw)
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert "candidate" not in report
    assert report["diagnostic"] == json.loads(_diagnostic())


class Image:
    def __init__(self, calls):
        self.calls = calls

    def entrypoint(self, value):
        self.calls.append(("entrypoint", value))
        return self

    def add_local_file(self, local, remote, *, copy):
        self.calls.append(("file", Path(local).name, remote, copy))
        return self

    def run_commands(self, *commands):
        self.calls.append(("build_commands", commands))
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


def test_isolated_capture_builds_exact_helper_and_invokes_exact_python():
    body = json.loads(_candidate())
    body["python"]["executable"] = capture._ISOLATED_PYTHON
    raw = (
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )
    sdk, sandbox, calls = _sdk(raw=raw)
    result = capture.capture_modal_inference_runtime(
        sdk=sdk,
        client=object(),
        app_name="existing",
        environment_name="isolated",
        image=IMAGE,
        source_commit=COMMIT,
        isolated_python=True,
    )
    create = next(call for call in calls if call[0] == "create")
    assert create[1][:2] == (capture._ISOLATED_PYTHON, "-I")
    assert ("build_commands", ("python3 " + capture._REMOTE_PREPARER,)) in calls
    report = json.loads(result.canonical_bytes)
    assert report["python_preparation"]["qualification"] == "CANDIDATE_ONLY"
    assert (
        report["python_preparation"]["script_sha256"]
        == capture._script_source(
            capture._script_path().with_name("prepare_modal_inference_python.py")
        )[1]
    )
    assert sandbox.terminate_calls == [False]


def test_isolated_capture_rejects_wrong_interpreter_and_cleans_up():
    sdk, sandbox, _ = _sdk()
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError):
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing",
            environment_name="isolated",
            image=IMAGE,
            source_commit=COMMIT,
            isolated_python=True,
        )
    assert sandbox.terminate_calls == [False]


def test_modal_additions_require_isolation_before_provider_reads():
    sdk, _, calls = _sdk()
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError):
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing",
            environment_name="isolated",
            image=IMAGE,
            source_commit=COMMIT,
            modal_additions=True,
        )
    assert calls == []


def test_modal_additions_are_hashed_no_deps_and_checked():
    body = json.loads(_candidate())
    body["python"]["executable"] = capture._ISOLATED_PYTHON
    body["requirements"]["modal"] = {"present": True, "version": "1.5.4"}
    body["distributions"]["modal"] = "1.5.4"
    raw = (
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )
    sdk, sandbox, calls = _sdk(raw=raw)
    report = json.loads(
        capture.capture_modal_inference_runtime(
            sdk=sdk,
            client=object(),
            app_name="existing",
            environment_name="isolated",
            image=IMAGE,
            source_commit=COMMIT,
            isolated_python=True,
            modal_additions=True,
        ).canonical_bytes
    )
    commands = [
        command for call in calls if call[0] == "build_commands" for command in call[1]
    ]
    install = next(command for command in commands if " install " in command)
    assert "--no-deps --require-hashes --only-binary=:all:" in install
    assert capture._ISOLATED_PYTHON + " -I -m pip --isolated check" in commands
    assert (
        report["modal_additions_sha256"]
        == capture._script_source(ROOT / "requirements/modal-inference-additions.lock")[
            1
        ]
    )
    assert sandbox.terminate_calls == [False]


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


def _cli_args(*extra):
    return [
        "--app",
        "a",
        "--environment",
        "e",
        "--image",
        IMAGE,
        "--source-commit",
        COMMIT,
        *extra,
    ]


def _fake_modal(monkeypatch, *, values=None, error=None):
    calls = []

    class Config:
        @staticmethod
        def get(key, *, profile=None, use_env=True):
            calls.append(("get", key, profile, use_env))
            if error is not None:
                raise error
            return values[key]

    class Client:
        @staticmethod
        def from_credentials(token_id, token_secret):
            calls.append(("client", token_id, token_secret))
            return object()

    module = SimpleNamespace(Client=Client)
    config_module = ModuleType("modal.config")
    config_module.config = Config
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setitem(sys.modules, "modal.config", config_module)
    monkeypatch.setattr(capture.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        capture,
        "capture_modal_inference_runtime",
        lambda **kwargs: SimpleNamespace(canonical_bytes=b"{}"),
    )
    return calls


def test_cli_explicit_profile_ignores_environment_and_disables_env_fallback(
    monkeypatch, capsys
):
    monkeypatch.setenv("MODAL_TOKEN_ID", "environment-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "environment-secret")
    calls = _fake_modal(
        monkeypatch,
        values={"token_id": "profile-id", "token_secret": "profile-secret"},
    )
    assert capture.main(_cli_args("--modal-profile", "synaptic-labs")) == 0
    assert calls == [
        ("get", "token_id", "synaptic-labs", False),
        ("get", "token_secret", "synaptic-labs", False),
        ("client", "profile-id", "profile-secret"),
    ]
    assert capsys.readouterr().err == ""


def test_cli_environment_auth_never_reads_profile(monkeypatch, capsys):
    monkeypatch.setenv("MODAL_TOKEN_ID", "environment-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "environment-secret")
    calls = _fake_modal(monkeypatch, error=AssertionError("profile read"))
    assert capture.main(_cli_args()) == 0
    assert calls == [("client", "environment-id", "environment-secret")]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "values",
    (
        {"token_id": None, "token_secret": "secret"},
        {"token_id": "", "token_secret": "secret"},
        {"token_id": "id", "token_secret": "   "},
    ),
)
def test_cli_profile_missing_or_blank_pair_fails_before_capture(
    monkeypatch, capsys, values
):
    calls = _fake_modal(monkeypatch, values=values)
    invoked = []
    monkeypatch.setattr(
        capture,
        "capture_modal_inference_runtime",
        lambda **kwargs: invoked.append(kwargs),
    )
    assert capture.main(_cli_args("--modal-profile", "synaptic-labs")) == 125
    assert invoked == []
    assert not any(call[0] == "client" for call in calls)
    assert "secret" not in capsys.readouterr().err


def test_cli_profile_lookup_diagnostics_and_values_are_closed(monkeypatch, capsys):
    monkeypatch.setenv("MODAL_TOKEN_ID", "environment-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "environment-secret")
    calls = _fake_modal(monkeypatch, error=RuntimeError("private-profile-value"))
    assert capture.main(_cli_args("--modal-profile", "synaptic-labs")) == 125
    output = capsys.readouterr()
    assert output.out == ""
    assert "private-profile-value" not in output.err
    assert "environment-secret" not in output.err
    assert calls == [("get", "token_id", "synaptic-labs", False)]


@pytest.mark.parametrize("profile", ("", " bad", "bad/name", "a" * 65))
def test_cli_profile_name_is_rejected_before_auth_resolution(
    monkeypatch, capsys, profile
):
    calls = _fake_modal(
        monkeypatch, values={"token_id": "id", "token_secret": "secret"}
    )
    assert capture.main(_cli_args("--modal-profile", profile)) == 125
    assert calls == []
    assert "secret" not in capsys.readouterr().err


class _ReadSandbox:
    def __init__(self, sandbox_id="sb-exact", *, returncode=0, stdout=None, stderr=b""):
        self.object_id = sandbox_id
        self._returncode = returncode
        self.stdout = [_candidate() if stdout is None else stdout]
        self.stderr = [stderr] if stderr else []

    def poll(self):
        return self._returncode


def _read_sdk(sandbox):
    calls = []

    class SandboxFactory:
        @staticmethod
        def from_id(sandbox_id, client=None):
            calls.append(("from_id", sandbox_id, client))
            return sandbox

    return SimpleNamespace(__version__="1.5.4", Sandbox=SandboxFactory), calls


def test_read_sandbox_uses_exact_stopped_handle_without_mutation_or_discovery():
    sandbox = _ReadSandbox()
    sdk, calls = _read_sdk(sandbox)
    client = object()
    raw = capture.read_modal_inference_runtime_sandbox(
        sdk=sdk, client=client, sandbox_id="sb-exact", image=IMAGE, source_commit=COMMIT
    )
    assert json.loads(raw)["candidate"]["status"] == "CANDIDATE_ONLY"
    assert calls == [("from_id", "sb-exact", client)]
    assert not any(hasattr(sandbox, name) for name in ("terminate", "write", "create"))


def test_read_sandbox_reports_only_strict_remote_reason():
    error = (
        json.dumps(
            {
                "reason_code": "PYTHON_INVALID",
                "schema_version": "synaptic-modal-inference-runtime-inspection-error/v1",
                "status": "FAILED",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    sandbox = _ReadSandbox(returncode=125, stdout=b"", stderr=error)
    sdk, _ = _read_sdk(sandbox)
    raw = capture.read_modal_inference_runtime_sandbox(
        sdk=sdk,
        client=object(),
        sandbox_id="sb-exact",
        image=IMAGE,
        source_commit=COMMIT,
    )
    assert json.loads(raw) == {
        "reason_code": "PYTHON_INVALID",
        "returncode": 125,
        "sandbox_id": "sb-exact",
        "schema_version": "synaptic-modal-inference-runtime-read/v1",
        "status": "REMOTE_FAILED",
    }


@pytest.mark.parametrize(
    "reason",
    (
        "DISTRIBUTION_COUNT_LIMIT",
        "DISTRIBUTION_ENUMERATION_FAILED",
        "DISTRIBUTION_IDENTITY_DUPLICATE",
        "DISTRIBUTION_IDENTITY_UNPROVEN",
        "DISTRIBUTION_METADATA_READ_FAILED",
        "DISTRIBUTION_NAME_INVALID",
        "DISTRIBUTION_PHYSICAL_DUPLICATE",
        "DISTRIBUTION_PHYSICAL_METADATA_MISMATCH",
        "DISTRIBUTION_VERSION_INVALID",
    ),
)
def test_read_sandbox_accepts_closed_distribution_reason(reason):
    error = (
        json.dumps(
            {
                "reason_code": reason,
                "schema_version": "synaptic-modal-inference-runtime-inspection-error/v1",
                "status": "FAILED",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    sandbox = _ReadSandbox(returncode=125, stdout=b"", stderr=error)
    sdk, _ = _read_sdk(sandbox)
    raw = capture.read_modal_inference_runtime_sandbox(
        sdk=sdk,
        client=object(),
        sandbox_id="sb-exact",
        image=IMAGE,
        source_commit=COMMIT,
    )
    assert json.loads(raw) == {
        "reason_code": reason,
        "returncode": 125,
        "sandbox_id": "sb-exact",
        "schema_version": "synaptic-modal-inference-runtime-read/v1",
        "status": "REMOTE_FAILED",
    }


@pytest.mark.parametrize(
    "stderr", (b"private traceback\n", b'{"reason_code":"SECRET"}\n')
)
def test_read_sandbox_malformed_remote_log_is_closed_and_not_leaked(stderr):
    sandbox = _ReadSandbox(returncode=125, stdout=b"", stderr=stderr)
    sdk, _ = _read_sdk(sandbox)
    raw = capture.read_modal_inference_runtime_sandbox(
        sdk=sdk,
        client=object(),
        sandbox_id="sb-exact",
        image=IMAGE,
        source_commit=COMMIT,
    )
    assert json.loads(raw)["reason_code"] == "UNCLASSIFIED_OUTPUT"
    assert b"private" not in raw


def test_read_sandbox_invalid_success_output_preserves_known_result():
    sandbox = _ReadSandbox(returncode=0, stdout=b"private invalid\n")
    sdk, _ = _read_sdk(sandbox)
    raw = capture.read_modal_inference_runtime_sandbox(
        sdk=sdk,
        client=object(),
        sandbox_id="sb-exact",
        image=IMAGE,
        source_commit=COMMIT,
    )
    assert json.loads(raw) == {
        "reason_code": "OUTPUT_INVALID",
        "returncode": 0,
        "sandbox_id": "sb-exact",
        "schema_version": "synaptic-modal-inference-runtime-read/v1",
        "status": "REMOTE_FAILED",
    }
    assert b"private" not in raw


def test_read_sandbox_requires_stopped_before_stream_access():
    sandbox = _ReadSandbox(returncode=None)

    class Unreadable:
        def __iter__(self):
            raise AssertionError("stdout read")

    sandbox.stdout = Unreadable()
    sdk, _ = _read_sdk(sandbox)
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError) as caught:
        capture.read_modal_inference_runtime_sandbox(
            sdk=sdk,
            client=object(),
            sandbox_id="sb-exact",
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert str(caught.value) == "sandbox_not_stopped"


@pytest.mark.parametrize(
    "sandbox_id", ("", "sb-bad/slash", "xx-exact", "sb-" + "a" * 65)
)
def test_read_sandbox_rejects_malformed_exact_id_before_sdk(sandbox_id):
    sdk, calls = _read_sdk(_ReadSandbox())
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError):
        capture.read_modal_inference_runtime_sandbox(
            sdk=sdk,
            client=object(),
            sandbox_id=sandbox_id,
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert calls == []


def test_read_sandbox_aggregate_deadline_bounds_blocked_poll(monkeypatch):
    sandbox = _ReadSandbox()
    gate = threading.Event()
    sandbox.poll = gate.wait
    sdk, _ = _read_sdk(sandbox)
    monkeypatch.setattr(capture, "_READ_TIMEOUT_SECONDS", 0.01)
    started = time.monotonic()
    with pytest.raises(capture.ModalInferenceRuntimeCaptureError) as caught:
        capture.read_modal_inference_runtime_sandbox(
            sdk=sdk,
            client=object(),
            sandbox_id="sb-exact",
            image=IMAGE,
            source_commit=COMMIT,
        )
    assert str(caught.value) == "sandbox_read_timeout"
    assert time.monotonic() - started < 0.5


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
