"""Local prepared-publication adapter for the training input source port."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from tuner.training.contracts import (
    PreparedTrainingInputIdentity,
    RetainedTrainingInputStreamLease,
)

from .context_messages import ROW_SCHEMA_VERSION_V2
from .models import ROW_SCHEMA_VERSION
from .publication import snapshot_prepared_dataset_v1, snapshot_prepared_dataset_v2


class LocalPreparedTrainingInputSource:
    """Reopen and reverify one private prepared publication for each fresh lease."""

    __slots__ = ("_path", "_identity")

    def __init__(self, path: Path, identity: PreparedTrainingInputIdentity) -> None:
        if type(path) is not type(Path()) or not path.is_absolute():
            raise TypeError("prepared publication path must be an absolute Path")
        if type(identity) is not PreparedTrainingInputIdentity:
            raise TypeError("exact prepared training input identity required")
        self._path = path
        self._identity = identity

    @property
    def identity(self) -> PreparedTrainingInputIdentity:
        return self._identity

    def open_lease(self) -> RetainedTrainingInputStreamLease:
        if self._identity.format == ROW_SCHEMA_VERSION:
            verified, content = snapshot_prepared_dataset_v1(self._path)
        elif self._identity.format == ROW_SCHEMA_VERSION_V2:
            verified, content = snapshot_prepared_dataset_v2(self._path)
        else:  # constructor identity is generic; this adapter is intentionally narrow.
            raise ValueError("prepared publication format is unsupported")
        observed = verified.semantic_identity
        current = PreparedTrainingInputIdentity(
            ref=f"prepared://sha256/{observed.dataset_digest}",
            revision=observed.dataset_digest,
            content_digest=observed.dataset_sha256,
            size_bytes=observed.dataset_bytes,
            format=self._identity.format,
        )
        if current != self._identity:
            raise ValueError("prepared publication changed before staging")
        return RetainedTrainingInputStreamLease(current, BytesIO(content))


__all__ = ["LocalPreparedTrainingInputSource"]
