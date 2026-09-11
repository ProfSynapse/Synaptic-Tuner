"""Executable private stdio entrypoint for one authenticated Modal chat.

The trusted SDK adapter supplies the initial frame over its newly created
Sandbox's authenticated stdin. This frame is configuration transport, not a
grant: the concrete bootstrap still authenticates the signed launch and locked
runtime. The image must pre-create the private directories below, and the SDK
adapter must authenticate and mount the exact selected Volumes independently.
"""

from __future__ import annotations

import base64
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import sys
import threading
from typing import BinaryIO

from tuner.execution.foundation_v2.canonical import parse_canonical_object

from .inference_bootstrap import open_modal_chat_worker
from .inference_channel import serve_modal_chat_channel
from .inference_model import ModalPinnedModelPreparer
from .inference_preparation import ModalInferencePreparationConfig
from .inference_wire import ModalChatWorkerExpectation, admit_modal_chat_launch
from .runtime import EnvironmentHmacAuthenticator

_SCHEMA = "synaptic-modal-chat-start/v1"
_MAX_START_BYTES = 256 * 1024
_MAX_ARGUMENT_BYTES = 96 * 1024
_START_WAIT_SECONDS = 30.0
_KEY = re.compile(r"[A-Z][A-Z0-9_]{0,127}")
_PRIVATE_ROOT = Path("/workspace/modal-chat")
_DESTINATION = _PRIVATE_ROOT / "model"
_BASE = _PRIVATE_ROOT / "base"
_SCRATCH = _PRIVATE_ROOT / "scratch"
_CUDA_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
)


class ModalChatEntrypointError(RuntimeError):
    """Closed startup or execution failure; never contains provider details."""


@dataclass(frozen=True, slots=True)
class ModalChatStart:
    expectation: ModalChatWorkerExpectation
    argument_bytes: bytes
    evidence_environment_key: str
    model_token_key: str | None


class _UTCClock:
    def now_iso(self) -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate start field")
        result[key] = value
    return result


def _validate(start: ModalChatStart) -> None:
    if (
        type(start) is not ModalChatStart
        or type(start.expectation) is not ModalChatWorkerExpectation
    ):
        raise ValueError("exact chat start required")
    expectation = start.expectation
    ModalChatWorkerExpectation(
        **{
            name: getattr(expectation, name)
            for name in expectation.__dataclass_fields__
        }
    )
    if (
        type(start.argument_bytes) is not bytes
        or not 0 < len(start.argument_bytes) <= _MAX_ARGUMENT_BYTES
    ):
        raise ValueError("launch argument exceeds its bound")
    evidence = start.evidence_environment_key
    model = start.model_token_key
    for key in (evidence,) if model is None else (evidence, model):
        if (
            type(key) is not str
            or _KEY.fullmatch(key) is None
            or key.startswith("MODAL_")
        ):
            raise ValueError("invalid declared key name")
    if evidence == model:
        raise ValueError("preparation and evidence keys must differ")
    configuration = ModalInferencePreparationConfig.parse(
        expectation.configuration_bytes
    ).document
    declared = [
        key for secret in configuration["secrets"] for key in secret["required_keys"]
    ]
    expected = {evidence} if model is None else {evidence, model}
    if len(declared) != len(expected) or set(declared) != expected:
        raise ValueError("start keys differ from authenticated secret requirements")
    for root in (
        expectation.artifact_root,
        expectation.control_root,
        expectation.cache_root,
    ):
        mount = Path(root)
        if (
            mount == _PRIVATE_ROOT
            or mount in _PRIVATE_ROOT.parents
            or _PRIVATE_ROOT in mount.parents
        ):
            raise ValueError("mount overlaps private image directories")


def encode_modal_chat_start(start: ModalChatStart) -> bytes:
    """Freeze a bounded non-secret startup frame for authenticated SDK stdin."""
    _validate(start)
    expectation = {
        name: getattr(start.expectation, name)
        for name in start.expectation.__dataclass_fields__
    }
    expectation["configuration_bytes"] = base64.b64encode(
        expectation["configuration_bytes"]
    ).decode("ascii")
    payload = _canonical(
        {
            "schema_version": _SCHEMA,
            "expectation": expectation,
            "argument_bytes": base64.b64encode(start.argument_bytes).decode("ascii"),
            "evidence_environment_key": start.evidence_environment_key,
            "model_token_key": start.model_token_key,
        }
    )
    if len(payload) > _MAX_START_BYTES:
        raise ValueError("chat start exceeds its bound")
    return payload


def decode_modal_chat_start(payload: bytes) -> ModalChatStart:
    """Parse transport only; signed launch admission remains mandatory."""
    try:
        if type(payload) is not bytes or not 0 < len(payload) <= _MAX_START_BYTES:
            raise ValueError("invalid start bytes")
        document = json.loads(payload.decode("ascii"), object_pairs_hook=_pairs)
        if (
            type(document) is not dict
            or set(document)
            != {
                "schema_version",
                "expectation",
                "argument_bytes",
                "evidence_environment_key",
                "model_token_key",
            }
            or document["schema_version"] != _SCHEMA
        ):
            raise ValueError("invalid start shape")
        expectation = document["expectation"]
        if type(expectation) is not dict or set(expectation) != set(
            ModalChatWorkerExpectation.__dataclass_fields__
        ):
            raise ValueError("invalid expectation shape")
        expectation = dict(expectation)
        for value in (expectation["configuration_bytes"], document["argument_bytes"]):
            if type(value) is not str:
                raise ValueError("exact encoded bytes required")
        expectation["configuration_bytes"] = base64.b64decode(
            expectation["configuration_bytes"], validate=True
        )
        result = ModalChatStart(
            ModalChatWorkerExpectation(**expectation),
            base64.b64decode(document["argument_bytes"], validate=True),
            document["evidence_environment_key"],
            document["model_token_key"],
        )
        if encode_modal_chat_start(result) != payload:
            raise ValueError("noncanonical start frame")
        return result
    except Exception:
        raise ModalChatEntrypointError("modal_chat_start_invalid") from None


def _read_start(stream: BinaryIO) -> bytes:
    result: Queue[object] = Queue(maxsize=1)

    def read() -> None:
        try:
            value = stream.readline(_MAX_START_BYTES + 1)
        except BaseException as error:
            value = error
        result.put(value)

    threading.Thread(target=read, name="modal-chat-start-reader", daemon=True).start()
    try:
        value = result.get(timeout=_START_WAIT_SECONDS)
    except Empty:
        raise ModalChatEntrypointError("modal_chat_start_unavailable") from None
    if isinstance(value, (KeyboardInterrupt, SystemExit)):
        raise value
    if type(value) is not bytes or not value:
        raise ModalChatEntrypointError("modal_chat_start_unavailable")
    return value


def run_modal_chat_entrypoint(input_stream: BinaryIO, output_stream: BinaryIO) -> None:
    """Run exactly one session; no ambient client, mkdir, or public endpoint."""
    start = decode_modal_chat_start(_read_start(input_stream))
    verifier = EnvironmentHmacAuthenticator(
        environment_key=start.evidence_environment_key,
        key_ref=start.expectation.key_ref,
    )
    clock = _UTCClock()
    # Do not even read the preparation credential until the launch is admitted.
    admission = admit_modal_chat_launch(
        start.argument_bytes,
        expectation=start.expectation,
        verifier=verifier,
        clock=clock,
    )
    claim = parse_canonical_object(admission.claim, name="chat launch claim")
    policy = admission.configuration.document["policy"]
    token = (
        None if start.model_token_key is None else os.environ.get(start.model_token_key)
    )
    preparer = ModalPinnedModelPreparer(
        persistent_root=Path(start.expectation.cache_root),
        destination_root=_BASE,
        scratch_root=_SCRATCH,
        token=token,
    )
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
    for name in _CUDA_ENVIRONMENT:
        value = os.environ.get(name)
        if value:
            environment[name] = value
    with open_modal_chat_worker(
        start.argument_bytes,
        expectation=start.expectation,
        verifier=verifier,
        clock=clock,
        destination=_DESTINATION,
        cwd=_PRIVATE_ROOT,
        environment=environment,
        preparer=preparer,
    ) as prepared:
        serve_modal_chat_channel(
            prepared.session,
            input_stream,
            output_stream,
            model=prepared.model,
            session_id=claim["session_id"],
            argument_bytes=start.argument_bytes,
            max_request_bytes=policy["max_request_bytes"],
            max_response_bytes=policy["max_response_bytes"],
        )


def main(argv: list[str] | None = None) -> int:
    """Keep SDK/model diagnostics away from protocol stdout and user logs."""
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        return 124
    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    try:
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                run_modal_chat_entrypoint(input_stream, output_stream)
        return 0
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        # SystemExit can contain text that the interpreter would otherwise
        # print after the redirection context has unwound.
        return 125
    except Exception:
        # The host observes failure/EOF; never serialize exception text.
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
