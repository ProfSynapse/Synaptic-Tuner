"""Import-light host client for one newly allocated Modal chat Sandbox.

The trusted caller must pass the exact handle returned by its authenticated
Sandbox creation.  Object identity checks prevent later substitution; they do
not prove that the handle was newly created or authorize the allocation.
"""

from __future__ import annotations

import hashlib
from inspect import getattr_static
import math
from queue import Empty, Queue
import threading
import time
from types import MappingProxyType
from typing import Mapping, Sequence

from Evaluator.protocols import BackendResponse
from tuner.inference.run_chat import PreparedModelIdentity

from .inference_channel import decode_modal_chat_frame, encode_modal_chat_frame

_SCHEMA = "synaptic-modal-chat-channel/v1"
_CLOSED = "modal_inference_channel_client_invalid"
_MAPPING_PROXY = type(MappingProxyType({}))
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60


class ModalInferenceChannelClientError(RuntimeError):
    """Closed, non-secret host channel failure."""


def _number(
    value: object,
    name: str,
    *,
    positive: bool = True,
    maximum: float = _MAX_TIMEOUT_SECONDS,
) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an exact finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be an exact finite number") from None
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be an exact finite number")
    if not positive and result < 0:
        raise ValueError(f"{name} must be an exact finite number")
    if result > maximum:
        raise ValueError(f"{name} must be an exact finite number")
    return result


def _now() -> float:
    value = time.monotonic()
    if type(value) not in (int, float):
        raise ValueError("channel monotonic clock is invalid")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("channel monotonic clock is invalid") from None
    if not math.isfinite(result):
        raise ValueError("channel monotonic clock is invalid")
    return result


def _frame(session_id: str, launch_digest: str, kind: str, **values) -> dict:
    return {
        "schema_version": _SCHEMA,
        "kind": kind,
        "session_id": session_id,
        "launch_digest": launch_digest,
        **values,
    }


class ModalInferenceChannelClient:
    """Sequential BackendClient and exact-owner Sandbox cleanup lease."""

    def __init__(
        self,
        sandbox,
        *,
        session_id: str,
        argument_bytes: bytes,
        deadline: float,
        startup_timeout_seconds: float,
        request_timeout_seconds: float,
        max_request_bytes: int,
        max_response_bytes: int,
    ) -> None:
        self._lock = threading.Lock()
        self._termination_lock = threading.Lock()
        self._termination_done = threading.Event()
        self._termination_thread: threading.Thread | None = None
        self._termination_result = False
        self._termination_error: BaseException | None = None
        self._healthy = False
        self._closed = False
        self._history: tuple[tuple[str, str], ...] = ()
        self._next_request_id = 1
        self._sandbox = sandbox
        self._stdin = None
        self._stdout = None
        self._stdout_iterator = None
        try:
            self._capture_sandbox(sandbox)
            if type(argument_bytes) is not bytes or not argument_bytes:
                raise TypeError("exact nonempty launch argument required")
            now = _now()
            self._deadline = _number(
                deadline,
                "channel deadline",
                maximum=now + _MAX_TIMEOUT_SECONDS,
            )
            if self._deadline <= now:
                raise ValueError("channel deadline must be in the future")
            self._startup_timeout = _number(startup_timeout_seconds, "startup timeout")
            self._request_timeout = _number(request_timeout_seconds, "request timeout")
            self._request_maximum = self._maximum(max_request_bytes)
            self._response_maximum = self._maximum(max_response_bytes)
            self._launch_digest = hashlib.sha256(argument_bytes).hexdigest()
            # Validate the binding with the sole shared codec before touching
            # any stream property.
            encode_modal_chat_frame(
                _frame(
                    session_id,
                    self._launch_digest,
                    "stop",
                    request_id=1,
                ),
                self._request_maximum,
            )
            self._session_id = session_id
            self._bind_sandbox(sandbox)
            ready = self._read_frame(
                min(self._deadline, _now() + self._startup_timeout)
            )
            if (
                ready.get("kind") != "ready"
                or ready.get("session_id") != session_id
                or ready.get("launch_digest") != self._launch_digest
            ):
                raise ValueError("channel ready binding differs")
            model = ready.get("model")
            if type(model) is not dict or set(model) != {
                "model_ref",
                "model_revision",
                "tokenizer_revision",
                "model_kind",
            }:
                raise ValueError("channel ready model is invalid")
            checked_model = PreparedModelIdentity(
                model["model_ref"],
                model["model_revision"],
                model["tokenizer_revision"],
                model["model_kind"],
            )
            self._model = (
                checked_model.model_ref,
                checked_model.model_revision,
                checked_model.tokenizer_revision,
                checked_model.model_kind,
            )
            self._healthy = True
        except (KeyboardInterrupt, SystemExit) as error:
            self._healthy = False
            self._constructor_cleanup(error)
            raise
        except Exception:
            self._healthy = False
            error = ModalInferenceChannelClientError(_CLOSED)
            self._constructor_cleanup(error)
            raise error from None

    @staticmethod
    def _maximum(value: object) -> int:
        if type(value) is not int or not 1 <= value <= 1024 * 1024:
            raise ValueError("channel frame bound is invalid")
        return value

    def _bind_sandbox(self, sandbox) -> None:
        sandbox_type = type(sandbox)
        for name in ("stdin", "stdout"):
            if getattr_static(sandbox_type, name, None) is None:
                raise TypeError("Sandbox handle is invalid")
        object_id = sandbox.object_id
        if object_id != self._object_id:
            raise TypeError("Sandbox object identity is invalid")
        stdin = sandbox.stdin
        stdout = sandbox.stdout
        if not callable(getattr_static(type(stdin), "write", None)) or not callable(
            getattr_static(type(stdin), "drain", None)
        ):
            raise TypeError("Sandbox input stream is invalid")
        if not callable(getattr_static(type(stdout), "__iter__", None)):
            raise TypeError("Sandbox output stream is invalid")
        self._stdin = stdin
        self._stdout = stdout
        self._stdout_iterator = iter(stdout)

    def _capture_sandbox(self, sandbox) -> None:
        sandbox_type = type(sandbox)
        if getattr_static(sandbox_type, "object_id", None) is None or not callable(
            getattr_static(sandbox_type, "terminate", None)
        ):
            raise TypeError("Sandbox ownership surface is invalid")
        object_id = sandbox.object_id
        if type(object_id) is not str or not object_id:
            raise TypeError("Sandbox object identity is invalid")
        self._object_id = object_id

    def _remaining(self, operation_deadline: float) -> float:
        now = _now()
        remaining = min(self._deadline, operation_deadline) - now
        if remaining <= 0:
            raise TimeoutError("channel deadline expired")
        return remaining

    def _operation(self, function, operation_deadline: float):
        result: Queue[object] = Queue(maxsize=1)
        timeout = self._remaining(operation_deadline)

        def invoke() -> None:
            try:
                value = function()
            except BaseException as error:
                value = error
            result.put(value)

        worker = threading.Thread(
            target=invoke, name="modal-chat-host-channel", daemon=True
        )
        worker.start()
        try:
            value = result.get(timeout=timeout)
        except Empty:
            raise TimeoutError("channel operation timed out") from None
        self._remaining(operation_deadline)
        if isinstance(value, BaseException):
            raise value
        return value

    def _read_frame(self, operation_deadline: float) -> dict[str, object]:
        iterator = self._stdout_iterator
        if iterator is None:
            raise ValueError("channel output is unavailable")
        line = self._operation(lambda: next(iterator), operation_deadline)
        if type(line) is not str or len(line) > self._response_maximum:
            raise ValueError("channel output frame is invalid")
        try:
            payload = line.encode("utf-8")
        except UnicodeError:
            raise ValueError("channel output frame is invalid") from None
        return decode_modal_chat_frame(payload, self._response_maximum)

    def _write_frame(self, frame: dict[str, object], operation_deadline: float) -> None:
        payload = encode_modal_chat_frame(frame, self._request_maximum)
        stream = self._stdin
        if stream is None:
            raise ValueError("channel input is unavailable")

        def write() -> None:
            result = stream.write(payload)
            if result is not None:
                raise OSError("Sandbox input returned an invalid result")
            drained = stream.drain()
            if drained is not None:
                raise OSError("Sandbox input drain returned an invalid result")

        self._operation(write, operation_deadline)

    @staticmethod
    def _messages(messages: Sequence[Mapping[str, str]]) -> tuple[tuple[str, str], ...]:
        if type(messages) is not tuple:
            raise TypeError("channel messages must be an exact tuple")
        result = []
        for message in messages:
            if type(message) not in (dict, _MAPPING_PROXY) or set(message) != {
                "role",
                "content",
            }:
                raise TypeError("channel message is invalid")
            role = message["role"]
            content = message["content"]
            if type(role) is not str or type(content) is not str:
                raise TypeError("channel message is invalid")
            result.append((role, content))
        return tuple(result)

    def chat(self, messages: Sequence[Mapping[str, str]]) -> BackendResponse:
        if not self._lock.acquire(blocking=False):
            raise ModalInferenceChannelClientError(_CLOSED)
        try:
            if not self._healthy or self._closed:
                raise ModalInferenceChannelClientError(_CLOSED)
            checked = self._messages(messages)
            if (
                len(checked) != len(self._history) + 1
                or checked[:-1] != self._history
                or checked[-1][0] != "user"
                or not checked[-1][1]
            ):
                raise ModalInferenceChannelClientError(_CLOSED)
            request_id = self._next_request_id
            operation_deadline = min(self._deadline, _now() + self._request_timeout)
            request = _frame(
                self._session_id,
                self._launch_digest,
                "chat",
                request_id=request_id,
                content=checked[-1][1],
            )
            started = _now()
            self._write_frame(request, operation_deadline)
            if not self._healthy or self._closed:
                raise ModalInferenceChannelClientError(_CLOSED)
            response = self._read_frame(operation_deadline)
            if not self._healthy or self._closed:
                raise ModalInferenceChannelClientError(_CLOSED)
            if response == _frame(
                self._session_id,
                self._launch_digest,
                "error",
                request_id=request_id,
                code="modal_chat_channel_error",
            ):
                self._healthy = False
                raise ModalInferenceChannelClientError(_CLOSED)
            closed = _frame(
                self._session_id,
                self._launch_digest,
                "closed",
                request_id=request_id,
            )
            if response == closed:
                self._healthy = False
                self._closed = True
                raise ModalInferenceChannelClientError(_CLOSED)
            expected = _frame(
                self._session_id,
                self._launch_digest,
                "chat",
                request_id=request_id,
                content=response.get("content"),
            )
            if response != expected or type(response.get("content")) is not str:
                self._healthy = False
                raise ModalInferenceChannelClientError(_CLOSED)
            latency = _now() - started
            if latency < 0:
                self._healthy = False
                raise ModalInferenceChannelClientError(_CLOSED)
            content = response["content"]
            self._history = checked + (("assistant", content),)
            self._next_request_id += 1
            return BackendResponse(
                message=content, raw=dict(response), latency_s=latency
            )
        except (KeyboardInterrupt, SystemExit):
            self._healthy = False
            raise
        except ModalInferenceChannelClientError:
            raise
        except Exception:
            self._healthy = False
            raise ModalInferenceChannelClientError(_CLOSED) from None
        finally:
            self._lock.release()

    @property
    def cleanup_pending(self) -> bool:
        return not self._termination_result

    @property
    def model(self) -> PreparedModelIdentity:
        return PreparedModelIdentity(*self._model)

    def _start_termination(self) -> None:
        with self._termination_lock:
            if self._termination_thread is not None:
                return

            def terminate() -> None:
                try:
                    if self._sandbox.object_id != self._object_id:
                        return
                    result = self._sandbox.terminate(wait=True)
                    if type(result) is not int:
                        return
                    self._termination_result = True
                except BaseException as error:
                    self._termination_error = error
                    return
                finally:
                    self._termination_done.set()

            thread = threading.Thread(
                target=terminate, name="modal-chat-sandbox-termination", daemon=True
            )
            self._termination_thread = thread
            try:
                thread.start()
            except BaseException:
                self._termination_done.set()
                raise

    def close(self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0) -> bool:
        term = _number(term_timeout, "stop timeout", positive=False)
        kill = _number(kill_timeout, "termination timeout", positive=False)
        control_error: BaseException | None = None
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            try:
                if self._healthy and not self._closed:
                    request_id = self._next_request_id
                    deadline = _now() + term
                    try:
                        self._write_frame(
                            _frame(
                                self._session_id,
                                self._launch_digest,
                                "stop",
                                request_id=request_id,
                            ),
                            deadline,
                        )
                        response = self._read_frame(deadline)
                        expected = _frame(
                            self._session_id,
                            self._launch_digest,
                            "closed",
                            request_id=request_id,
                        )
                        if response != expected:
                            raise ValueError("channel close response differs")
                    except (KeyboardInterrupt, SystemExit) as error:
                        control_error = error
                        self._healthy = False
                    except BaseException:
                        self._healthy = False
                    self._closed = True
            finally:
                self._lock.release()
        else:
            self._healthy = False
            self._closed = True
        try:
            self._start_termination()
        except (KeyboardInterrupt, SystemExit):
            if control_error is not None:
                raise control_error
            raise
        except BaseException:
            if control_error is not None:
                raise control_error
            return False
        self._termination_done.wait(kill)
        if control_error is not None:
            raise control_error
        if isinstance(self._termination_error, (KeyboardInterrupt, SystemExit)):
            raise self._termination_error
        return self._termination_result

    def _constructor_cleanup(self, error: BaseException | None = None) -> None:
        try:
            self.close(term_timeout=0.0, kill_timeout=0.0)
        except BaseException:
            pass
        # Completed cleanup still owns the exact handle. Preserve that owner so
        # callers never mistake fast termination for missing cleanup authority.
        if error is not None:
            error.cleanup_lease = self  # type: ignore[attr-defined]


__all__: list[str] = []
