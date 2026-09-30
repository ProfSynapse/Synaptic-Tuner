import json

from examples.modal_chat.diagnostics import modal_chat_failure_diagnostic


def _raise_at(filename, kind=ValueError):
    namespace = {"kind": kind}
    exec(
        compile('def fail():\n raise kind("secret-message")\n', filename, "exec"),
        namespace,
    )
    namespace["fail"]()


def _capture(filename, kind=ValueError):
    try:
        _raise_at(filename, kind)
    except BaseException as error:
        return error
    raise AssertionError


def test_diagnostic_is_closed_canonical_and_uses_exact_relative_allowlist():
    error = _capture("/checkout/tuner/execution/providers/modal/inference_run_chat.py")
    payload = modal_chat_failure_diagnostic(phase="CHAT_OPEN", error=error)
    document = json.loads(payload)

    assert not payload.endswith(b"\n")
    assert set(document) == {
        "schema_version",
        "status",
        "phase",
        "retry_authorized",
        "authorizing",
        "exception_class",
        "exception_chain",
    }
    assert document["phase"] == "CHAT_OPEN"
    assert document["retry_authorized"] is False
    assert document["authorizing"] is False
    assert document["exception_class"] == "ValueError"
    assert document["exception_chain"][0]["locations"][-1] == {
        "path": "tuner/execution/providers/modal/inference_run_chat.py",
        "line": 2,
    }
    assert b"secret-message" not in payload
    assert b"/checkout/" not in payload


def test_quote_training_is_an_admitted_non_authorizing_phase():
    error = RuntimeError("private-rate-detail")
    payload = modal_chat_failure_diagnostic(phase="QUOTE_TRAINING", error=error)
    document = json.loads(payload)

    assert document["phase"] == "QUOTE_TRAINING"
    assert document["authorizing"] is False
    assert document["retry_authorized"] is False
    assert b"private-rate-detail" not in payload


def test_diagnostic_rejects_same_basename_and_unknown_exception_details():
    class PrivateFailure(Exception):
        pass

    error = _capture("/tmp/attacker/inference_run_chat.py", PrivateFailure)
    document = json.loads(
        modal_chat_failure_diagnostic(phase="CHAT_COMPOSITION", error=error)
    )
    assert document["exception_class"] == "OTHER"
    assert all(
        location["path"] != "tuner/execution/providers/modal/inference_run_chat.py"
        for item in document["exception_chain"]
        for location in item["locations"]
    )
    assert b"PrivateFailure" not in modal_chat_failure_diagnostic(
        phase="CHAT_COMPOSITION", error=error
    )

    secret_path = "/checkout/examples/modal_chat/hf_abcdefghijklmnopqrstuvwxyz.py"
    secret_error = _capture(secret_path)
    secret_payload = modal_chat_failure_diagnostic(
        phase="hf_abcdefghijklmnopqrstuvwxyz", error=secret_error
    )
    assert b"hf_abcdefghijklmnopqrstuvwxyz" not in secret_payload
    assert json.loads(secret_payload)["phase"] == "UNKNOWN"


def test_diagnostic_includes_suppressed_context_without_its_message():
    try:
        try:
            _raise_at("/checkout/examples/modal_chat/chat.py")
        except ValueError:
            raise RuntimeError("outer-secret") from None
    except RuntimeError as error:
        document = json.loads(
            modal_chat_failure_diagnostic(phase="CHAT_OPEN", error=error)
        )
    assert [item["exception_class"] for item in document["exception_chain"]] == [
        "RuntimeError",
        "ValueError",
    ]
    assert document["exception_chain"][1]["locations"][-1]["path"] == (
        "examples/modal_chat/chat.py"
    )


def test_diagnostic_bounds_exception_chain_and_invalid_phase():
    error = _capture("/checkout/modal/_functions.py")
    current = error
    for _ in range(10):
        try:
            raise RuntimeError("hidden") from current
        except RuntimeError as next_error:
            current = next_error
    document = json.loads(
        modal_chat_failure_diagnostic(phase="\N{SNOWMAN}", error=current)
    )
    assert document["phase"] == "UNKNOWN"
    assert len(document["exception_chain"]) == 8
    assert sum(len(item["locations"]) for item in document["exception_chain"]) <= 32


def test_diagnostic_admits_reviewed_coordinator_path_without_creating_authority():
    error = _capture("/checkout/tuner/execution/coordinator_v1/operations.py")
    first = modal_chat_failure_diagnostic(phase="CHAT_OPEN", error=error)
    second = modal_chat_failure_diagnostic(phase="CHAT_OPEN", error=error)
    document = json.loads(first)

    assert first == second
    assert document["exception_chain"][0]["locations"][-1]["path"] == (
        "tuner/execution/coordinator_v1/operations.py"
    )
    assert document["authorizing"] is False
    assert document["retry_authorized"] is False


def test_diagnostic_keeps_deepest_known_failure_and_handles_chain_cycle():
    deepest = _capture("/checkout/examples/modal_chat/chat.py")
    current = deepest
    for _ in range(7):
        try:
            raise RuntimeError("hidden") from current
        except RuntimeError as next_error:
            current = next_error
    document = json.loads(
        modal_chat_failure_diagnostic(phase="CHAT_OPEN", error=current)
    )
    assert len(document["exception_chain"]) == 8
    assert document["exception_chain"][-1]["exception_class"] == "ValueError"
    assert document["exception_chain"][-1]["locations"][-1]["path"] == (
        "examples/modal_chat/chat.py"
    )

    deepest.__context__ = current
    cycled = json.loads(modal_chat_failure_diagnostic(phase="CHAT_OPEN", error=current))
    assert len(cycled["exception_chain"]) == 8


def test_diagnostic_bounds_visits_through_unknown_traceback_paths():
    from examples.modal_chat import diagnostics

    class UnknownCode:
        co_filename = "/unlisted/private.py"

    class Frame:
        f_code = UnknownCode()

    class Trace:
        tb_frame = Frame()
        tb_lineno = 1
        visits = 0

        @property
        def tb_next(self):
            self.visits += 1
            assert self.visits <= diagnostics._MAX_VISITED_FRAMES
            return self

    class Error:
        __traceback__ = Trace()
        __cause__ = None
        __context__ = None

    error = Error()
    chain = diagnostics._exception_chain(error)
    assert chain == [{"exception_class": "OTHER", "locations": []}]
    assert error.__traceback__.visits == diagnostics._MAX_VISITED_FRAMES
