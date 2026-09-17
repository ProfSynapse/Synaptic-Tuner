"""Provider-free tests for the standalone Modal chat Sandbox launcher."""

import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from examples.model_chat import modal_launch

HAPPY_STDOUT = "\n".join(
    [
        '{"status":"CHAT_INPUTS_CHECKED","training_required":false}',
        '{"status":"CHAT_SAVED_AND_CLOSED","training_required":false}',
        "--- chat-result.jsonl",
        '{"status":"CLAIMED","configuration_sha256":"ab","training_required":false}',
        '{"status":"REPLY_SAVED","response":"private reply text","training_required":false}',
        '{"status":"CHAT_CONTEXT_CLOSED","provider_shutdown_proof":false}',
        "",
    ]
)


class AlreadyExistsError(Exception):
    pass


class FakeSandbox:
    object_id = "sb-fake1"

    def __init__(self, calls, *, stdout, stderr, returncode, terminate_error, poll_result, poll_error):
        self.calls = calls
        self.stdout = SimpleNamespace(read=lambda: stdout)
        self.stderr = SimpleNamespace(read=lambda: stderr)
        self.returncode = returncode
        self.terminate_error = terminate_error
        self.poll_result = poll_result
        self.poll_error = poll_error

    def wait(self, *, raise_on_termination):
        self.calls.append(("wait", raise_on_termination))

    def terminate(self, *, wait):
        self.calls.append(("terminate", wait))
        if self.terminate_error is not None:
            raise self.terminate_error

    def poll(self):
        self.calls.append(("poll", self.object_id))
        if self.poll_error is not None:
            raise self.poll_error
        return self.poll_result


def _sdk(
    *,
    stdout=HAPPY_STDOUT,
    stderr="",
    returncode=0,
    create_error=None,
    terminate_error=None,
    poll_result=0,
    poll_error=None,
    version="1.5.4",
    tokens=None,
):
    calls = []
    sandbox = FakeSandbox(
        calls,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        terminate_error=terminate_error,
        poll_result=poll_result,
        poll_error=poll_error,
    )

    class Builder:
        def __init__(self, steps):
            self.steps = steps

        def entrypoint(self, commands):
            self.steps.append(("entrypoint", list(commands)))
            return self

        def add_local_dir(self, local_path, remote_path, *, copy, ignore):
            self.steps.append(("dir", Path(local_path), remote_path, copy, tuple(ignore)))
            return self

        def add_local_file(self, local_path, remote_path):
            self.steps.append(("file", Path(local_path), remote_path))
            return self

    class Image:
        @staticmethod
        def from_registry(reference):
            builder = Builder([("registry", reference)])
            calls.append(("image", builder.steps))
            return builder

    class App:
        @staticmethod
        def lookup(name, **kwargs):
            calls.append(("lookup", name, kwargs))
            return SimpleNamespace(name=name)

    class Sandbox:
        @staticmethod
        def create(*args, **kwargs):
            calls.append(("create", args, kwargs))
            if create_error is not None:
                raise create_error
            return sandbox

        @staticmethod
        def from_id(sandbox_id, *, client):
            calls.append(("from_id", sandbox_id, client))
            return sandbox

    class Config:
        @staticmethod
        def get(key, *, profile=None, use_env=True):
            calls.append(("get", key, profile, use_env))
            return (tokens or {}).get(key)

    class Client:
        @staticmethod
        def from_credentials(token_id, token_secret):
            calls.append(("client", token_id, token_secret))
            return SimpleNamespace(kind="profile-client")

    sdk = SimpleNamespace(
        __version__=version,
        App=App,
        Image=Image,
        Sandbox=Sandbox,
        Client=Client,
        config=SimpleNamespace(config=Config),
        exception=SimpleNamespace(AlreadyExistsError=AlreadyExistsError),
    )
    return sdk, sandbox, calls


def _chat_document():
    return {
        "model": "owner/model",
        "revision": "a" * 40,
        "adapter_path": None,
        "prompt": "hello",
        "max_tokens": 48,
        "lifetime_seconds": 120,
    }


def _provider_document():
    return {
        "schema_version": "synaptic-model-chat-modal-provider/v1",
        "environment_name": "env-1",
        "app_name": "app-1",
        "gpu": "A10",
        "cpu_millicores": 4000,
        "memory_mb": 16384,
        "startup_margin_seconds": 300,
    }


@pytest.fixture
def inputs(tmp_path):
    chat = tmp_path / "chat.json"
    chat.write_text(json.dumps(_chat_document()))
    provider = tmp_path / "provider.json"
    provider.write_text(json.dumps(_provider_document()))
    output = tmp_path / "private"
    output.mkdir(mode=0o700)
    return SimpleNamespace(chat=chat, provider=provider, output=output, root=tmp_path)


def _argv(inputs, *extra, output=None):
    return [
        "--configuration",
        str(inputs.chat),
        "--provider",
        str(inputs.provider),
        "--attempt",
        "attempt-1",
        "--modal-profile",
        "selected",
        "--output-directory",
        str(inputs.output if output is None else output),
        *extra,
    ]


def _lines(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def _forbid_import(monkeypatch):
    def forbidden(name):
        pytest.fail("check mode imported " + name)

    monkeypatch.setattr(modal_launch.importlib, "import_module", forbidden)


def test_check_mode_needs_no_sdk_and_validates_both_files(inputs, monkeypatch, capsys):
    _forbid_import(monkeypatch)
    argv = ["--configuration", str(inputs.chat), "--provider", str(inputs.provider), "--check"]
    assert modal_launch.main(argv) == 0
    assert capsys.readouterr().out == '{"status":"MODAL_CHAT_INPUTS_CHECKED"}\n'
    assert list(inputs.output.iterdir()) == []


@pytest.mark.parametrize("broken", ["chat", "provider"])
def test_check_mode_rejects_bad_files(inputs, monkeypatch, capsys, broken):
    _forbid_import(monkeypatch)
    getattr(inputs, broken).write_text("{")
    argv = ["--configuration", str(inputs.chat), "--provider", str(inputs.provider), "--check"]
    assert modal_launch.main(argv) == 1
    (line,) = _lines(capsys)
    assert line["status"] == "MODAL_CHAT_FAILED" and line["retry_authorized"] is False
    assert "{" not in json.dumps(line.get("reason"))


def test_check_mode_reads_the_lock_digest(inputs, monkeypatch, capsys):
    _forbid_import(monkeypatch)
    lock = inputs.root / "lock.json"
    lock.write_text(json.dumps({"base_registry_reference": "docker.io/x/y:latest"}))
    monkeypatch.setattr(modal_launch, "LOCK_PATH", lock)
    argv = ["--configuration", str(inputs.chat), "--provider", str(inputs.provider), "--check"]
    assert modal_launch.main(argv) == 1
    assert _lines(capsys)[0]["reason"] == "modal_chat_image_reference_invalid"


def _mutate(document, key, value):
    if value is Ellipsis:
        del document[key]
    else:
        document[key] = value
    return document


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", "synaptic-model-chat-modal-provider/v2"),
        ("environment_name", ...),
        ("environment_name", ""),
        ("environment_name", "-leading"),
        ("app_name", "a/b"),
        ("gpu", "A10 G"),
        ("gpu", 10),
        ("cpu_millicores", 4000.0),
        ("cpu_millicores", 0),
        ("memory_mb", "16384"),
        ("startup_margin_seconds", 0),
        ("startup_margin_seconds", 601),
        ("startup_margin_seconds", True),
        ("extra", 1),
    ],
)
def test_provider_schema_is_strict(inputs, key, value):
    inputs.provider.write_text(json.dumps(_mutate(_provider_document(), key, value)))
    with pytest.raises(modal_launch.ModalChatLaunchError, match="modal_chat_provider_invalid"):
        modal_launch.load_provider(inputs.provider)


def test_provider_rejects_duplicate_keys_and_constants(inputs):
    inputs.provider.write_text(
        json.dumps(_provider_document())[:-1] + ',"gpu":"A10"}'
    )
    with pytest.raises(modal_launch.ModalChatLaunchError, match="modal_chat_provider_invalid"):
        modal_launch.load_provider(inputs.provider)
    inputs.provider.write_text(json.dumps(_provider_document()).replace("4000", "NaN"))
    with pytest.raises(modal_launch.ModalChatLaunchError, match="modal_chat_provider_invalid"):
        modal_launch.load_provider(inputs.provider)


def test_checked_in_smoke_provider_parses():
    provider = modal_launch.load_provider(
        modal_launch.ENGINE / "examples/model_chat/modal-smoke-provider.json"
    )
    assert provider == modal_launch.ModalChatProvider(
        "synaptic-smoke-v1", "synaptic-model-chat-v1", "A10", 4000, 16384, 300
    )


@pytest.mark.parametrize("name", ["", "a b", "a/b", "x" * 65, "ü"])
def test_attempt_name_is_validated_before_any_sdk_call(inputs, name, capsys):
    sdk, _, calls = _sdk()
    argv = _argv(inputs)
    argv[argv.index("--attempt") + 1] = name
    assert modal_launch.main(argv, sdk=sdk) == 1
    assert _lines(capsys)[0]["reason"] in {"modal_chat_attempt_invalid", "modal_chat_arguments_invalid"}
    assert calls == [] and list(inputs.output.iterdir()) == []


def test_output_directory_must_be_private_canonical_and_owned(inputs, tmp_path, capsys):
    sdk, _, calls = _sdk()
    group_readable = tmp_path / "shared"
    group_readable.mkdir(mode=0o750)
    link = tmp_path / "link"
    link.symlink_to(inputs.output)
    for bad in (group_readable, link, tmp_path / "missing", Path("relative")):
        assert modal_launch.main(_argv(inputs, output=bad), sdk=sdk) == 1
        assert _lines(capsys)[0]["reason"] == "modal_chat_output_invalid"
    assert calls == [] and list(inputs.output.iterdir()) == []


def test_sdk_version_is_pinned_before_any_provider_call(inputs, capsys):
    sdk, _, calls = _sdk(version="1.5.3")
    assert modal_launch.main(_argv(inputs), sdk=sdk) == 1
    assert _lines(capsys)[0]["reason"] == "modal_chat_sdk_invalid"
    assert calls == [] and list(inputs.output.iterdir()) == []


def test_image_is_composed_from_lock_digest_with_four_mounts(inputs):
    sdk, _, calls = _sdk()
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 0
    (steps,) = [entry[1] for entry in calls if entry[0] == "image"]
    reference = modal_launch.image_reference()
    assert "@sha256:" in reference
    ignore = tuple(modal_launch.MOUNT_IGNORE)
    engine = modal_launch.ENGINE
    assert steps == [
        ("registry", reference),
        ("entrypoint", []),
        ("dir", engine / "Evaluator", "/engine/Evaluator", False, ignore),
        ("dir", engine / "tuner", "/engine/tuner", False, ignore),
        ("dir", engine / "synaptic_tuner", "/engine/synaptic_tuner", False, ignore),
        ("file", engine / "scripts/chat_model.py", "/engine/scripts/chat_model.py"),
        ("file", inputs.chat, "/engine/chat.json"),
    ]
    assert {"**/__pycache__", "**/*.pyc", "**/tests", "**/Datasets/**"} <= set(ignore)
    assert engine not in [step[1] for step in steps if step[0] == "dir"]


def test_sandbox_create_kwargs_bound_lifetime_and_carry_no_secrets(inputs):
    sdk, _, calls = _sdk()
    client = object()
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=client) == 0
    (lookup,) = [entry for entry in calls if entry[0] == "lookup"]
    assert lookup == (
        "lookup",
        "app-1",
        {"environment_name": "env-1", "create_if_missing": True, "client": client},
    )
    (create,) = [entry for entry in calls if entry[0] == "create"]
    assert create[1] == ("bash", "-lc", modal_launch.CONTAINER_SCRIPT)
    kwargs = create[2]
    assert kwargs["timeout"] == 120 + 300
    assert kwargs["idle_timeout"] == kwargs["timeout"]
    assert kwargs["gpu"] == "A10"
    assert kwargs["cpu"] == 4.0
    assert kwargs["memory"] == 16384
    assert kwargs["name"] == "attempt-1"
    assert kwargs["client"] is client
    assert kwargs["app"].name == "app-1"
    assert not {"secrets", "volumes", "env", "environment_name", "block_network"} & set(kwargs)
    assert "mkdir -m 700 /root/chat-attempt" in modal_launch.CONTAINER_SCRIPT
    assert "--- chat-result.jsonl" in modal_launch.CONTAINER_SCRIPT


def test_existing_sandbox_name_refuses_and_creates_nothing(inputs, capsys):
    sdk, sandbox, calls = _sdk(create_error=AlreadyExistsError("name taken"))
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 1
    assert _lines(capsys) == [
        {"status": "MODAL_CHAT_ATTEMPT_EXISTS", "retry_authorized": False}
    ]
    assert [entry[0] for entry in calls] == ["lookup", "image", "create"]
    assert list(inputs.output.iterdir()) == []


def test_attempt_record_is_written_before_wait(inputs):
    sdk, sandbox, calls = _sdk()
    seen = {}

    def wait(*, raise_on_termination):
        seen["attempt"] = json.loads((inputs.output / "attempt.json").read_text())
        seen["raise_on_termination"] = raise_on_termination

    sandbox.wait = wait
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 0
    assert seen["raise_on_termination"] is False
    record = seen["attempt"]
    assert record["sandbox_id"] == "sb-fake1"
    assert record["attempt"] == "attempt-1"
    assert (record["app_name"], record["environment_name"]) == ("app-1", "env-1")
    assert record["image_reference"] == modal_launch.image_reference()
    assert (record["timeout_seconds"], record["idle_timeout_seconds"]) == (420, 420)
    assert record["created_at_utc"].endswith("Z")
    assert stat.S_IMODE((inputs.output / "attempt.json").stat().st_mode) == 0o600


def test_happy_path_saves_three_records_and_exits_zero_with_proof(inputs, capsys):
    sdk, sandbox, calls = _sdk(stderr="vllm noise")
    client = object()
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=client) == 0
    out = capsys.readouterr().out
    assert "private reply text" not in out
    assert [json.loads(line) for line in out.splitlines()] == [
        {"status": "MODAL_CHAT_SANDBOX_CREATED", "sandbox_id": "sb-fake1"},
        {"status": "MODAL_CHAT_SANDBOX_FINISHED", "returncode": 0},
        {"status": "MODAL_CHAT_REPLY_SAVED"},
        {"status": "MODAL_CHAT_SANDBOX_STOPPED", "provider_shutdown_proof": True},
    ]
    result = inputs.output / "chat-result.jsonl"
    records = [json.loads(line) for line in result.read_text().splitlines()]
    assert [record["status"] for record in records] == list(modal_launch.RESULT_STATUSES)
    assert records[1]["response"] == "private reply text"
    assert (inputs.output / "sandbox-stderr.log").read_text() == "vllm noise"
    assert (inputs.output / "sandbox-stdout.log").read_text() == HAPPY_STDOUT
    for name in ("attempt.json", "sandbox-stdout.log", "sandbox-stderr.log", "chat-result.jsonl", "shutdown.json"):
        assert stat.S_IMODE((inputs.output / name).stat().st_mode) == 0o600
    shutdown = json.loads((inputs.output / "shutdown.json").read_text())
    assert shutdown["provider_shutdown_proof"] is True
    assert shutdown["poll_result"] == 0 and shutdown["returncode"] == 0
    assert shutdown["complete"] is True and shutdown["reason"] is None
    assert shutdown["command_statuses"] == ["CHAT_INPUTS_CHECKED", "CHAT_SAVED_AND_CLOSED"]
    assert [entry for entry in calls if entry[0] in {"wait", "terminate", "from_id", "poll"}] == [
        ("wait", False),
        ("terminate", True),
        ("from_id", "sb-fake1", client),
        ("poll", "sb-fake1"),
    ]


def test_nonzero_returncode_fails_closed_and_retains_logs(inputs, capsys):
    stdout = HAPPY_STDOUT.replace("CHAT_SAVED_AND_CLOSED", "CHAT_FAILED").split("REPLY_SAVED")[0]
    stdout = stdout.rsplit("\n", 1)[0] + "\n"
    sdk, _, _ = _sdk(stdout=stdout, stderr="--- chat-result.jsonl was not written", returncode=1)
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 1
    assert [line["status"] for line in _lines(capsys)] == [
        "MODAL_CHAT_SANDBOX_CREATED",
        "MODAL_CHAT_SANDBOX_FINISHED",
        "MODAL_CHAT_FAILED",
        "MODAL_CHAT_SANDBOX_STOPPED",
    ]
    assert (inputs.output / "sandbox-stdout.log").read_text() == stdout
    assert (inputs.output / "sandbox-stderr.log").read_text().endswith("was not written")
    shutdown = json.loads((inputs.output / "shutdown.json").read_text())
    assert shutdown["reason"] == "modal_chat_command_failed"
    assert shutdown["record_statuses"] == ["CLAIMED"]
    assert shutdown["provider_shutdown_proof"] is True


def test_missing_marker_is_a_closed_failure_not_a_crash(inputs, capsys):
    sdk, _, _ = _sdk(stdout='{"status":"CHAT_INPUTS_CHECKED","training_required":false}\n')
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 1
    lines = _lines(capsys)
    assert lines[2] == {
        "status": "MODAL_CHAT_FAILED",
        "reason": "modal_chat_result_unparsable",
        "retry_authorized": False,
    }
    assert not (inputs.output / "chat-result.jsonl").exists()
    assert (inputs.output / "sandbox-stdout.log").exists()


def test_unparsable_record_after_marker_fails_closed(inputs, capsys):
    sdk, _, _ = _sdk(stdout=HAPPY_STDOUT + "not json\n")
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 1
    assert _lines(capsys)[2]["reason"] == "modal_chat_result_unparsable"
    assert not (inputs.output / "chat-result.jsonl").exists()


def test_terminate_raising_records_class_and_still_polls(inputs, capsys):
    class NotFoundError(Exception):
        pass

    sdk, _, calls = _sdk(terminate_error=NotFoundError("private message"))
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 0
    out = capsys.readouterr().out
    assert "private message" not in out
    assert ("poll", "sb-fake1") in calls
    shutdown = json.loads((inputs.output / "shutdown.json").read_text())
    assert shutdown["terminate_exception_class"] == "NotFoundError"
    assert shutdown["provider_shutdown_proof"] is True


def test_poll_none_means_no_shutdown_proof_and_nonzero_exit(inputs, capsys):
    sdk, _, _ = _sdk(poll_result=None)
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 1
    lines = _lines(capsys)
    assert lines[2] == {"status": "MODAL_CHAT_REPLY_SAVED"}
    assert lines[3] == {"status": "MODAL_CHAT_SANDBOX_STOPPED", "provider_shutdown_proof": False}
    shutdown = json.loads((inputs.output / "shutdown.json").read_text())
    assert shutdown["provider_shutdown_proof"] is False and shutdown["poll_result"] is None
    assert (inputs.output / "chat-result.jsonl").exists()


def test_poll_raising_records_class_without_proof(inputs, capsys):
    sdk, _, _ = _sdk(poll_error=RuntimeError("private provider message"))
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 1
    assert "private provider message" not in capsys.readouterr().out
    shutdown = json.loads((inputs.output / "shutdown.json").read_text())
    assert shutdown["poll_exception_class"] == "RuntimeError"
    assert shutdown["provider_shutdown_proof"] is False


def test_second_run_against_same_directory_refuses_before_sdk(inputs, capsys):
    sdk, _, calls = _sdk()
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 0
    capsys.readouterr()
    before = sorted(path.name for path in inputs.output.iterdir())
    del calls[:]
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 1
    assert _lines(capsys)[0]["reason"] == "modal_chat_output_claimed"
    assert calls == []
    assert sorted(path.name for path in inputs.output.iterdir()) == before


def test_credentials_come_only_from_the_named_profile(inputs, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "environment-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "environment-secret")
    sdk, _, calls = _sdk(tokens={"token_id": "profile-id", "token_secret": "profile-secret"})
    assert modal_launch.main(_argv(inputs), sdk=sdk) == 0
    assert calls[:3] == [
        ("get", "token_id", "selected", False),
        ("get", "token_secret", "selected", False),
        ("client", "profile-id", "profile-secret"),
    ]
    (create,) = [entry for entry in calls if entry[0] == "create"]
    assert create[2]["client"].kind == "profile-client"
    assert "environment-id" not in json.dumps([str(entry) for entry in calls])


def test_missing_profile_credentials_refuse_before_lookup(inputs, capsys):
    sdk, _, calls = _sdk(tokens={"token_id": "profile-id", "token_secret": ""})
    assert modal_launch.main(_argv(inputs), sdk=sdk) == 1
    assert _lines(capsys)[0]["reason"] == "modal_chat_credentials_missing"
    assert [entry[0] for entry in calls] == ["get", "get"]
    assert list(inputs.output.iterdir()) == []


def test_effectful_mode_requires_attempt_profile_and_output(inputs, capsys):
    sdk, _, calls = _sdk()
    argv = ["--configuration", str(inputs.chat), "--provider", str(inputs.provider)]
    assert modal_launch.main(argv, sdk=sdk) == 1
    assert _lines(capsys)[0]["reason"] == "modal_chat_arguments_invalid"
    assert calls == []


def test_stdout_logs_are_bounded_to_one_mebibyte(inputs):
    sdk, _, _ = _sdk(stdout="x" * (2 * 1024 * 1024) + "\n" + HAPPY_STDOUT)
    assert modal_launch.main(_argv(inputs), sdk=sdk, client=object()) == 1
    assert os.path.getsize(inputs.output / "sandbox-stdout.log") == 1024 * 1024
