"""Bounded stdio protocol for one admitted Modal chat worker."""

from __future__ import annotations

import hashlib
import json
from queue import Empty, Full, Queue
import re
import sys
import threading
from typing import BinaryIO

from Evaluator.chat_session import ChatSession
from tuner.inference.run_chat import PreparedModelIdentity

_SCHEMA = "synaptic-modal-chat-channel/v1"
_ERROR = "modal_chat_channel_error"
_MAX_FRAME_BYTES = 1024 * 1024
_POLL_SECONDS = 0.05
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_FIELDS = {
    "ready": frozenset(
        {"schema_version", "kind", "session_id", "launch_digest", "model"}
    ),
    "chat": frozenset(
        {
            "schema_version",
            "kind",
            "session_id",
            "launch_digest",
            "request_id",
            "content",
        }
    ),
    "stop": frozenset(
        {
            "schema_version",
            "kind",
            "session_id",
            "launch_digest",
            "request_id",
        }
    ),
    "error": frozenset(
        {
            "schema_version",
            "kind",
            "session_id",
            "launch_digest",
            "request_id",
            "code",
        }
    ),
    "closed": frozenset(
        {
            "schema_version",
            "kind",
            "session_id",
            "launch_digest",
            "request_id",
        }
    ),
}


class ModalInferenceChannelError(RuntimeError):
    """Closed, non-secret channel failure."""


def _maximum(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_FRAME_BYTES:
        raise ValueError("modal chat frame bound is invalid")
    return value


def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate frame field")
        result[key] = value
    return result


def _validate_frame(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("modal chat frame is not an exact object")
    kind = value.get("kind")
    if type(kind) is not str or kind not in _FIELDS or set(value) != _FIELDS[kind]:
        raise ValueError("modal chat frame shape is invalid")
    if value["schema_version"] != _SCHEMA:
        raise ValueError("modal chat frame schema is invalid")
    session_id = value["session_id"]
    launch_digest = value["launch_digest"]
    if type(session_id) is not str or _REF.fullmatch(session_id) is None:
        raise ValueError("modal chat session binding is invalid")
    if type(launch_digest) is not str or _DIGEST.fullmatch(launch_digest) is None:
        raise ValueError("modal chat launch binding is invalid")
    if kind != "ready":
        request_id = value["request_id"]
        if type(request_id) is not int or not 1 <= request_id <= 2**63 - 1:
            raise ValueError("modal chat request identity is invalid")
    else:
        model = value["model"]
        if type(model) is not dict or set(model) != {
            "model_ref",
            "model_revision",
            "tokenizer_revision",
            "model_kind",
        }:
            raise ValueError("modal chat model identity is invalid")
        PreparedModelIdentity(**model)
    if kind == "chat" and type(value["content"]) is not str:
        raise ValueError("modal chat content is invalid")
    if kind == "error" and value["code"] != _ERROR:
        raise ValueError("modal chat error code is invalid")
    return value


def encode_modal_chat_frame(value: dict[str, object], maximum_bytes: int) -> bytes:
    """Encode one exact protocol frame; this is framing, not authentication."""
    maximum = _maximum(maximum_bytes)
    checked = _validate_frame(value)
    try:
        payload = (
            json.dumps(
                checked,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, OverflowError):
        raise ValueError("modal chat frame encoding failed") from None
    if len(payload) > maximum:
        raise ValueError("modal chat frame exceeds its bound")
    return payload


def decode_modal_chat_frame(payload: bytes, maximum_bytes: int) -> dict[str, object]:
    """Decode one complete canonical protocol frame."""
    maximum = _maximum(maximum_bytes)
    if (
        type(payload) is not bytes
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
        or not 1 < len(payload) <= maximum
    ):
        raise ValueError("modal chat frame is incomplete or exceeds its bound")
    try:
        value = json.loads(
            payload[:-1].decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            parse_float=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
        checked = _validate_frame(value)
        if encode_modal_chat_frame(checked, maximum) != payload:
            raise ValueError("modal chat frame is not canonical")
        return checked
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeError, OverflowError):
        raise ValueError("modal chat frame is invalid") from None


def _bound(session_id: str, launch_digest: str, kind: str, **values) -> dict:
    return {
        "schema_version": _SCHEMA,
        "kind": kind,
        "session_id": session_id,
        "launch_digest": launch_digest,
        **values,
    }


def _write_sync(stream: BinaryIO, frame: dict[str, object], maximum: int) -> None:
    payload = encode_modal_chat_frame(frame, maximum)
    written = stream.write(payload)
    if written is not None and (type(written) is not int or written != len(payload)):
        raise OSError("partial modal chat frame write")
    stream.flush()


def _write(
    session: ChatSession,
    stream: BinaryIO,
    frame: dict[str, object],
    maximum: int,
) -> bool:
    results: Queue[BaseException | None] = Queue(maxsize=1)

    def write() -> None:
        try:
            _write_sync(stream, frame, maximum)
            result = None
        except BaseException as error:
            result = error
        results.put(result)

    writer = threading.Thread(
        target=write,
        name="modal-chat-channel-writer",
        daemon=True,
    )
    writer.start()
    while True:
        try:
            result = results.get(timeout=_POLL_SECONDS)
        except Empty:
            if session.state.closed:
                return False
            continue
        if result is None:
            return True
        raise result


def _reader(
    stream: BinaryIO,
    maximum: int,
    results: Queue[object],
    advance: threading.Event,
    stopped: threading.Event,
) -> None:
    while not stopped.is_set():
        try:
            result: object = stream.readline(maximum + 1)
        except BaseException as error:
            result = error
        while not stopped.is_set():
            try:
                results.put(result, timeout=_POLL_SECONDS)
                break
            except Full:
                continue
        if not isinstance(result, (bytes, bytearray, memoryview)) or not result:
            return
        while not stopped.is_set():
            if advance.wait(_POLL_SECONDS):
                advance.clear()
                break


def serve_modal_chat_channel(
    session: ChatSession,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    *,
    model: PreparedModelIdentity,
    session_id: str,
    argument_bytes: bytes,
    max_request_bytes: int,
    max_response_bytes: int,
) -> None:
    """Serve one sequential chat session through bounded binary stdio frames."""
    stopped = threading.Event()
    active_failure = False
    try:
        if type(session) is not ChatSession:
            raise TypeError("exact ChatSession required")
        if type(model) is not PreparedModelIdentity:
            raise TypeError("exact prepared model identity required")
        identity = PreparedModelIdentity(
            model.model_ref,
            model.model_revision,
            model.tokenizer_revision,
            model.model_kind,
        )
        if type(argument_bytes) is not bytes or not argument_bytes:
            raise TypeError("exact nonempty launch argument required")
        request_maximum = _maximum(max_request_bytes)
        response_maximum = _maximum(max_response_bytes)
        launch_digest = hashlib.sha256(argument_bytes).hexdigest()
        binding = _bound(
            session_id,
            launch_digest,
            "ready",
            model={
                name: getattr(identity, name) for name in identity.__dataclass_fields__
            },
        )
        encode_modal_chat_frame(binding, response_maximum)
        for stream, method in (
            (input_stream, "readline"),
            (output_stream, "write"),
            (output_stream, "flush"),
        ):
            if not callable(getattr(type(stream), method, None)):
                raise TypeError("channel stream is invalid")
        if not _write(session, output_stream, binding, response_maximum):
            return
        results: Queue[object] = Queue(maxsize=1)
        advance = threading.Event()
        reader = threading.Thread(
            target=_reader,
            args=(input_stream, request_maximum, results, advance, stopped),
            name="modal-chat-channel-reader",
            daemon=True,
        )
        reader.start()
        expected = 1
        while True:
            state = session.state
            if type(state.closed) is not bool or state.closed:
                _write(
                    session,
                    output_stream,
                    _bound(
                        session_id,
                        launch_digest,
                        "closed",
                        request_id=expected,
                    ),
                    response_maximum,
                )
                return
            try:
                raw = results.get(timeout=_POLL_SECONDS)
            except Empty:
                continue
            if isinstance(raw, (KeyboardInterrupt, SystemExit)):
                raise raw
            if isinstance(raw, BaseException) or type(raw) is not bytes:
                raise ValueError("channel input failed")
            if not raw:
                _write(
                    session,
                    output_stream,
                    _bound(
                        session_id,
                        launch_digest,
                        "closed",
                        request_id=expected,
                    ),
                    response_maximum,
                )
                return
            try:
                request = decode_modal_chat_frame(raw, request_maximum)
                if (
                    request["kind"] not in {"chat", "stop"}
                    or request["session_id"] != session_id
                    or request["launch_digest"] != launch_digest
                    or request["request_id"] != expected
                ):
                    raise ValueError("chat request differs from channel")
                if request["kind"] == "stop":
                    _write(
                        session,
                        output_stream,
                        _bound(
                            session_id,
                            launch_digest,
                            "closed",
                            request_id=expected,
                        ),
                        response_maximum,
                    )
                    return
                response = session.chat(request["content"])
                _write(
                    session,
                    output_stream,
                    _bound(
                        session_id,
                        launch_digest,
                        "chat",
                        request_id=expected,
                        content=response.message,
                    ),
                    response_maximum,
                )
                expected += 1
                advance.set()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _write(
                    session,
                    output_stream,
                    _bound(
                        session_id,
                        launch_digest,
                        "error",
                        request_id=expected,
                        code=_ERROR,
                    ),
                    response_maximum,
                )
                return
    except (KeyboardInterrupt, SystemExit):
        active_failure = True
        raise
    except Exception:
        active_failure = True
        raise ModalInferenceChannelError(_ERROR) from None
    finally:
        stopped.set()
        try:
            session.close()
        except (KeyboardInterrupt, SystemExit):
            if not active_failure:
                raise
        except BaseException:
            if not active_failure:
                raise ModalInferenceChannelError(_ERROR) from None


__all__: list[str] = []
