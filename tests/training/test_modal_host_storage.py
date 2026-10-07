from __future__ import annotations

import os

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.training.modal_host_storage import (
    ModalHostAttemptAlreadyClaimed, ModalHostStorageV1,
)


pytestmark = pytest.mark.skipif(os.name != "posix", reason="host journal is POSIX-only")


def test_packaged_host_journal_retains_exact_catalog_and_prevents_replay(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    evidence = canonical_bytes({"schema_version": "test/v1", "intent": "one"})
    database = private / "attempt.sqlite3"
    with ModalHostStorageV1(database, "project") as journal:
        journal.attempts.claim("one", evidence)
        assert journal.attempts.resolve("one") == evidence
        catalog = journal.catalog("calls", encode=lambda value: value.encode(),
                                  decode=lambda value: value.decode())
        assert catalog.publish_if_absent("command", "fc-one") is True
        assert catalog.publish_if_absent("command", "fc-one") is False
        assert catalog.resolve("command") == "fc-one"
        with pytest.raises(ModalHostAttemptAlreadyClaimed):
            journal.attempts.claim("one", evidence)
    with ModalHostStorageV1(database, "project") as reopened:
        assert reopened.attempts.resolve("one") == evidence
        with pytest.raises(ModalHostAttemptAlreadyClaimed):
            reopened.attempts.claim("one", evidence)


def test_packaged_host_journal_rejects_world_readable_root(tmp_path):
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(Exception, match="modal_host_storage_invalid"):
        ModalHostStorageV1(public / "attempt.sqlite3", "project")
