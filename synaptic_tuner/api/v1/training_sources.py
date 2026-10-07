"""Provider-neutral contracts for host-side training-input preparation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import BinaryIO, Protocol, runtime_checkable

from ._contract import PreparedTrainingInputIdentity, required_text
from .training_facade import TrainingRequest


_MAX_TRAINING_BYTES = 64 * 1024
_MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def _canonical_training_json(value: str) -> str:
    if type(value) is not str:
        raise TypeError("canonical_training_json must be an exact string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise ValueError("canonical_training_json is invalid") from None
    if not encoded or len(encoded) > _MAX_TRAINING_BYTES:
        raise ValueError("canonical_training_json is outside its byte limit")
    from .training_input import TrainingInputV1

    parsed = TrainingInputV1.from_json(value)
    if parsed.canonical_json() != value:
        raise ValueError("canonical_training_json must be exact canonical JSON")
    return value


@dataclass(frozen=True, slots=True)
class LocalTrainingInputPathV1:
    """Host-only path to a source admitted for preparation."""

    path: Path

    def __post_init__(self) -> None:
        if type(self.path) is not type(Path()) or not self.path.is_absolute():
            raise TypeError("local training input path must be an absolute Path")

    def __copy__(self):
        raise TypeError("host-only training sources are not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("host-only training sources are not copyable")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("host-only training sources are not serializable")


def _snapshot_upload(stream: BinaryIO, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    failed = False
    try:
        while size <= maximum_bytes:
            chunk = stream.read(min(1024 * 1024, maximum_bytes + 1 - size))
            if type(chunk) is not bytes:
                raise TypeError("training input upload must yield exact bytes")
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        payload = b"".join(chunks)
        if not payload:
            raise ValueError("training input upload must not be empty")
        if len(payload) > maximum_bytes:
            raise ValueError("training input upload exceeded its byte limit")
        return payload
    except BaseException:
        failed = True
        raise
    finally:
        try:
            stream.close()
        except BaseException:
            if not failed:
                raise ValueError("training input upload could not be detached") from None


class OneUseTrainingInputUploadV1:
    """Immutable owned snapshot of one caller upload, consumable exactly once."""

    __slots__ = ("_payload", "_state", "_lock")

    def __init__(self, stream: BinaryIO, *, maximum_bytes: int = _MAX_UPLOAD_BYTES) -> None:
        if not callable(getattr(stream, "read", None)) or not callable(
            getattr(stream, "close", None)
        ):
            raise TypeError("upload requires a binary stream")
        if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= _MAX_UPLOAD_BYTES:
            raise ValueError("maximum_bytes is outside the supported range")
        self._payload: bytes | None = _snapshot_upload(stream, maximum_bytes)
        self._state = "available"
        self._lock = Lock()

    def take_stream(self) -> BinaryIO:
        with self._lock:
            if self._state != "available" or self._payload is None:
                raise ValueError("training input upload was already consumed")
            payload = self._payload
            self._payload = None
            self._state = "transferred"
            return io.BytesIO(payload)

    def close(self) -> None:
        with self._lock:
            self._payload = None
            self._state = "closed"

    def __copy__(self):
        raise TypeError("training input uploads are not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("training input uploads are not copyable")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("training input uploads are not serializable")


TrainingInputSourceV1 = LocalTrainingInputPathV1 | OneUseTrainingInputUploadV1


@runtime_checkable
class TrainingNormalizerConfigV1(Protocol):
    @property
    def normalizer_ref(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TrainingPreparationConfigV1:
    """Canonical request plus one closed, normalizer-specific host config."""

    request_id: str
    project_ref: str
    canonical_training_json: str
    normalizer_config: TrainingNormalizerConfigV1

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", required_text(self.request_id, "request_id"))
        object.__setattr__(self, "project_ref", required_text(self.project_ref, "project_ref"))
        object.__setattr__(
            self, "canonical_training_json", _canonical_training_json(self.canonical_training_json)
        )
        if not isinstance(self.normalizer_config, TrainingNormalizerConfigV1):
            raise TypeError("normalizer_config must implement TrainingNormalizerConfigV1")
        required_text(self.normalizer_config.normalizer_ref, "normalizer_ref")


@dataclass(frozen=True, slots=True)
class PreparedTrainingInputV1:
    """Strict canonical prepared identity and training request."""

    identity: PreparedTrainingInputIdentity
    request: TrainingRequest

    def __post_init__(self) -> None:
        if type(self.identity) is not PreparedTrainingInputIdentity:
            raise TypeError("identity must be exact PreparedTrainingInputIdentity")
        if type(self.request) is not TrainingRequest:
            raise TypeError("request must be exact TrainingRequest")
        from .training_input import TrainingInputV1

        parsed = TrainingInputV1.from_json(self.request.canonical_json)
        if parsed.canonical_json() != self.request.canonical_json:
            raise ValueError("prepared request must contain exact canonical JSON")
        if parsed.dataset.ref != self.identity.ref:
            raise ValueError("prepared request does not bind the prepared identity")

    def to_dict(self) -> dict[str, object]:
        from .training_input import TrainingInputV1

        parsed = TrainingInputV1.from_json(self.request.canonical_json)
        return {
            "identity": self.identity.to_dict(),
            "request": {
                "request_id": self.request.request_id,
                "project_ref": self.request.project_ref,
                "training_input": parsed.to_dict(),
            },
        }


@runtime_checkable
class RetainedPreparedTrainingInputSourceV1(Protocol):
    @property
    def identity(self) -> PreparedTrainingInputIdentity: ...

    def open_lease(self) -> object: ...


@dataclass(frozen=True, slots=True)
class PreparedTrainingInputResultV1:
    """Prepared public value plus explicitly host-only staging authority."""

    prepared: PreparedTrainingInputV1
    retained_source: RetainedPreparedTrainingInputSourceV1

    def __post_init__(self) -> None:
        if type(self.prepared) is not PreparedTrainingInputV1:
            raise TypeError("prepared must be exact PreparedTrainingInputV1")
        if not isinstance(self.retained_source, RetainedPreparedTrainingInputSourceV1):
            raise TypeError("retained_source must implement the prepared source port")
        if self.retained_source.identity != self.prepared.identity:
            raise ValueError("retained source does not bind the prepared identity")

    def __copy__(self):
        raise TypeError("host-only prepared retention is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("host-only prepared retention is not copyable")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("host-only prepared retention is not serializable")


__all__ = [
    "LocalTrainingInputPathV1",
    "OneUseTrainingInputUploadV1",
    "PreparedTrainingInputIdentity",
    "PreparedTrainingInputResultV1",
    "PreparedTrainingInputV1",
    "RetainedPreparedTrainingInputSourceV1",
    "TrainingInputSourceV1",
    "TrainingNormalizerConfigV1",
    "TrainingPreparationConfigV1",
]
