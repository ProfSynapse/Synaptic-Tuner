"""Modal-local implementation of generic pinned-model preparation."""

from __future__ import annotations

import os
import re
from pathlib import Path

from tuner.inference.retrieved_model import _open_root, _platform
from tuner.inference.serving_target import (
    VerifiedPinnedModelSnapshot,
    capture_pinned_model_snapshot,
)

from .model_snapshot import prepare_model_snapshot

__all__: list[str] = []

_PINNED_REVISION = re.compile(r"[0-9a-f]{40}")


class ModalPinnedModelPreparer:
    """Prepare one authenticated upstream snapshot in private execution roots."""

    __slots__ = ("_persistent_root", "_destination_root", "_scratch_root", "_token")

    def __init__(
        self,
        *,
        persistent_root: Path,
        destination_root: Path,
        scratch_root: Path,
        token: str | None,
    ) -> None:
        roots = (persistent_root, destination_root, scratch_root)
        if any(not isinstance(root, Path) or not root.is_absolute() for root in roots):
            raise TypeError("model preparation roots must be exact absolute Paths")
        if len(set(roots)) != len(roots):
            raise ValueError("model preparation roots must be distinct")
        if token is not None and type(token) is not str:
            raise TypeError("model preparation token must be a string or None")
        self._persistent_root = persistent_root
        self._destination_root = destination_root
        self._scratch_root = scratch_root
        self._token = token

    def __repr__(self) -> str:
        return f"{type(self).__name__}(roots=<private>, token=<redacted>)"

    def prepare(
        self, *, model_ref: str, revision: str
    ) -> VerifiedPinnedModelSnapshot:
        if type(model_ref) is not str or not model_ref:
            raise TypeError("model_ref must be a non-empty string")
        if type(revision) is not str or _PINNED_REVISION.fullmatch(revision) is None:
            raise ValueError("Modal model revision must be an exact 40-hex commit")
        root_identities = _root_identities(
            self._persistent_root, self._destination_root, self._scratch_root
        )
        destination = prepare_model_snapshot(
            model_ref=model_ref,
            revision=revision,
            token=self._token,
            persistent_root=self._persistent_root,
            destination_root=self._destination_root,
            scratch_root=self._scratch_root,
        )
        if not isinstance(destination, Path) or not destination.is_absolute():
            raise ValueError("model preparer returned a noncanonical path")
        try:
            relative = destination.relative_to(self._destination_root)
        except ValueError:
            raise ValueError("model preparer returned a path outside its private root") from None
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("model preparer returned an invalid snapshot path")
        if root_identities != _root_identities(
            self._persistent_root, self._destination_root, self._scratch_root
        ):
            raise ValueError("model preparation root identity changed")
        try:
            result = capture_pinned_model_snapshot(
                model_ref=model_ref,
                revision=revision,
                root=self._destination_root,
                snapshot=relative.as_posix(),
            )
        except Exception:
            raise ValueError("model snapshot inventory capture failed") from None
        if type(result) is not VerifiedPinnedModelSnapshot:
            raise TypeError("snapshot capture returned the wrong type")
        if (
            result.model_ref != model_ref
            or result.revision != revision
            or result.root != self._destination_root
            or result.snapshot != relative.as_posix()
        ):
            raise ValueError("snapshot capture changed model identity")
        return result


def _root_identities(*roots: Path) -> tuple[tuple[int, int], ...]:
    _platform()
    values = []
    for root in roots:
        descriptor = _open_root(root)
        try:
            info = os.fstat(descriptor)
            values.append((info.st_dev, info.st_ino))
        finally:
            os.close(descriptor)
    if len(set(values)) != len(values):
        raise ValueError("model preparation roots share a physical directory")
    return tuple(values)
