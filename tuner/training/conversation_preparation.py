"""Conversation adapters for existing host-owned preparation extension ports."""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from synaptic_tuner.api.v1.training_sources import LocalTrainingInputPathV1, PreparedTrainingInputIdentity
from tuner.dataset_prep.conversation_messages import (
    MAX_EXECUTABLE_BYTES, ROW_SCHEMA, ConversationNormalizerConfigV1,
    build_conversation_publication, publish_conversation_dataset,
    read_source_members, snapshot_conversation_dataset,
)
from tuner.training.contracts import RetainedTrainingInputStreamLease
from .input_preparation import (
    NormalizedTrainingInputV1, PreparedInputVerificationV1,
    _stable_directory_identity,
)


class ConversationTrainingInputNormalizerV1:
    """Read declared members beneath caller-supplied stable source roots."""
    def __init__(self, *, allowed_source_roots: tuple[Path, ...]):
        if type(allowed_source_roots) is not tuple or not allowed_source_roots:
            raise TypeError("explicit conversation source roots required")
        self._roots = []
        for root in allowed_source_roots:
            if type(root) is not type(Path()) or not root.is_absolute():
                raise TypeError("conversation source root must be an absolute Path")
            self._roots.append((root, _stable_directory_identity(root, "conversation source root")))

    def prepare(self, source, *, config, output_root):
        if type(source) is not LocalTrainingInputPathV1 or type(config) is not ConversationNormalizerConfigV1:
            raise TypeError("exact local conversation source and config required")
        candidate = Path(os.path.abspath(source.path))
        admitted = False
        for root, identity in self._roots:
            if _stable_directory_identity(root, "conversation source root") != identity:
                raise ValueError("conversation source root changed")
            if candidate == root or root in candidate.parents:
                admitted = True
        if not admitted:
            raise ValueError("conversation source is outside authorized roots")
        dataset, metadata = read_source_members(candidate, config)
        prepared = build_conversation_publication(config, dataset, metadata)
        result = publish_conversation_dataset(output_root, prepared)
        return NormalizedTrainingInputV1(result.path, ROW_SCHEMA)


class _ConversationPreparedSource:
    __slots__ = ("_path", "_identity")

    def __init__(self, path, identity):
        if type(path) is not type(Path()) or not path.is_absolute():
            raise TypeError("conversation publication path must be absolute")
        self._path, self._identity = path, identity

    @property
    def identity(self):
        return self._identity

    def open_lease(self):
        verified, raw = snapshot_conversation_dataset(self._path)
        current = _prepared_identity(verified)
        if current != self._identity or len(raw) > MAX_EXECUTABLE_BYTES:
            raise ValueError("conversation prepared input changed or exceeds 2MiB executable limit")
        return RetainedTrainingInputStreamLease(current, BytesIO(raw))

    def __reduce_ex__(self, _protocol):
        raise TypeError("host-only conversation sources are not serializable")


def _prepared_identity(verified):
    semantic = verified.semantic_identity
    return PreparedTrainingInputIdentity(
        ref="prepared://sha256/" + semantic.dataset_digest, revision=semantic.dataset_digest,
        content_digest=semantic.dataset_sha256, size_bytes=semantic.dataset_bytes, format=ROW_SCHEMA,
    )


class ConversationPreparedInputFormatVerifierV1:
    """Independent publication validation; runtime format compatibility remains closed."""
    @property
    def format(self):
        return ROW_SCHEMA

    def verify(self, path):
        verified, snapshot = snapshot_conversation_dataset(path)
        identity = _prepared_identity(verified)
        if identity.size_bytes > MAX_EXECUTABLE_BYTES:
            raise ValueError("conversation input exceeds 2MiB executable limit")
        return PreparedInputVerificationV1(
            verified.path.resolve(strict=True), identity.revision, identity.content_digest,
            identity.size_bytes, snapshot,
        )

    def source(self, path, identity):
        if type(identity) is not PreparedTrainingInputIdentity or identity.format != ROW_SCHEMA:
            raise ValueError("conversation identity differs from verifier")
        return _ConversationPreparedSource(path, identity)
