"""Modal-local implementation of generic pinned-model preparation."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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

    def prepare(self, *, model_ref: str, revision: str) -> VerifiedPinnedModelSnapshot:
        if type(model_ref) is not str or not model_ref:
            raise TypeError("model_ref must be a non-empty string")
        if type(revision) is not str or _PINNED_REVISION.fullmatch(revision) is None:
            raise ValueError("Modal model revision must be an exact 40-hex commit")
        roots = (self._persistent_root, self._destination_root, self._scratch_root)
        with _retain_roots(*roots) as retained:
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
                raise ValueError(
                    "model preparer returned a path outside its private root"
                ) from None
            if not relative.parts or any(
                part in {"", ".", ".."} for part in relative.parts
            ):
                raise ValueError("model preparer returned an invalid snapshot path")
            _verify_retained_roots(roots, retained)
            try:
                result = capture_pinned_model_snapshot(
                    model_ref=model_ref,
                    revision=revision,
                    root=self._destination_root,
                    snapshot=relative.as_posix(),
                )
            except Exception:
                raise ValueError("model snapshot inventory capture failed") from None
            _verify_retained_roots(roots, retained)
            if type(result) is not VerifiedPinnedModelSnapshot:
                raise TypeError("snapshot capture returned the wrong type")
            if (
                result.model_ref != model_ref
                or result.revision != revision
                or result.root != self._destination_root
                or result.root_identity != retained[1][1]
                or result.snapshot != relative.as_posix()
            ):
                raise ValueError("snapshot capture changed model identity")
            return result


@contextmanager
def _retain_roots(
    *roots: Path,
) -> Iterator[tuple[tuple[int, tuple[int, int]], ...]]:
    _platform()
    retained: list[tuple[int, tuple[int, int]]] = []
    try:
        for root in roots:
            descriptor = _open_root(root)
            try:
                info = os.fstat(descriptor)
            except BaseException:
                os.close(descriptor)
                raise
            retained.append((descriptor, (info.st_dev, info.st_ino)))
        identities = tuple(identity for _, identity in retained)
        if len(set(identities)) != len(identities):
            raise ValueError("model preparation roots share a physical directory")
        yield tuple(retained)
    finally:
        for descriptor, _ in reversed(retained):
            os.close(descriptor)


def _verify_retained_roots(
    roots: tuple[Path, ...], retained: tuple[tuple[int, tuple[int, int]], ...]
) -> None:
    if len(roots) != len(retained):  # pragma: no cover - internal invariant
        raise ValueError("model preparation root identity changed")
    for root, (descriptor, identity) in zip(roots, retained, strict=True):
        current_descriptor = None
        try:
            retained_info = os.fstat(descriptor)
            if (retained_info.st_dev, retained_info.st_ino) != identity:
                raise ValueError("model preparation root identity changed")
            current_descriptor = _open_root(root)
            current_info = os.fstat(current_descriptor)
            if (current_info.st_dev, current_info.st_ino) != identity:
                raise ValueError("model preparation root identity changed")
        finally:
            if current_descriptor is not None:
                os.close(current_descriptor)
