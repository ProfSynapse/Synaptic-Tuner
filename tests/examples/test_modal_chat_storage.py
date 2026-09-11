from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys

import pytest

_ROOT = Path(__file__).parents[2]
_MODULE = _ROOT / "examples/modal_chat/storage.py"
_SPEC = importlib.util.spec_from_file_location("modal_chat_example_storage", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
storage = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = storage
_SPEC.loader.exec_module(storage)


def _private(tmp_path: Path) -> Path:
    os.chmod(tmp_path, 0o700)
    return tmp_path / "consumer.sqlite3"


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_attempt_claim_is_permanent_and_diagnostic_after_restart(
    tmp_path: Path,
) -> None:
    path = _private(tmp_path)
    evidence = _canonical(
        {
            "schema_version": "example-modal-chat-attempt/v1",
            "attempt_ref": "attempt-1",
            "provider_ref": "modal-production",
            "request_digest": "a" * 64,
        }
    )
    with storage.ModalChatStorage(path, "consumer-namespace") as first:
        claimed = first.attempts.claim("attempt-1", evidence)
        assert claimed.canonical_evidence == evidence

    with storage.ModalChatStorage(path, "consumer-namespace") as restarted:
        assert restarted.attempts.resolve("attempt-1") == claimed
        with pytest.raises(storage.AttemptAlreadyClaimed):
            restarted.attempts.claim("attempt-1", evidence)
        with pytest.raises(storage.AttemptAlreadyClaimed):
            restarted.attempts.claim(
                "attempt-1", _canonical({"different": "commitment"})
            )


def test_catalog_is_exact_publish_if_absent_and_restart_safe(tmp_path: Path) -> None:
    path = _private(tmp_path)
    with storage.ModalChatStorage(path, "consumer-namespace") as first:
        catalog = first.catalog(
            "chat-results", encode=lambda value: value, decode=bytes
        )
        assert catalog.publish_if_absent("result-1", b'{"result":"closed"}') is True
        assert catalog.publish_if_absent("result-1", b'{"result":"closed"}') is False
        with pytest.raises(storage.ModalChatStorageError):
            catalog.publish_if_absent("result-1", b'{"result":"other"}')

    with storage.ModalChatStorage(path, "consumer-namespace") as restarted:
        catalog = restarted.catalog(
            "chat-results", encode=lambda value: value, decode=bytes
        )
        assert catalog.resolve("result-1") == b'{"result":"closed"}'


def test_catalog_names_and_consumer_namespaces_are_exactly_isolated(
    tmp_path: Path,
) -> None:
    path = _private(tmp_path)
    codec = {"encode": lambda value: value, "decode": bytes}
    with storage.ModalChatStorage(path, "namespace-a") as first:
        assert first.catalog("events", **codec).publish_if_absent("same", b"event")
        assert first.catalog("results", **codec).resolve("same") is None
    with storage.ModalChatStorage(path, "namespace-b") as second:
        assert second.catalog("events", **codec).resolve("same") is None


def test_process_lifetime_lock_refuses_second_owner(tmp_path: Path) -> None:
    path = _private(tmp_path)
    first = storage.ModalChatStorage(path, "namespace-a")
    try:
        with pytest.raises(storage.ModalChatStorageError):
            storage.ModalChatStorage(path, "namespace-a")
    finally:
        first.close()


def test_database_and_lock_are_private(tmp_path: Path) -> None:
    path = _private(tmp_path)
    with storage.ModalChatStorage(path, "namespace-a"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.with_name(path.name + ".lock").stat().st_mode) == 0o600


def test_tampered_catalog_bytes_are_rejected(tmp_path: Path) -> None:
    path = _private(tmp_path)
    with storage.ModalChatStorage(path, "namespace-a") as owner:
        catalog = owner.catalog("results", encode=lambda value: value, decode=bytes)
        catalog.publish_if_absent("result-1", b"original")
    connection = sqlite3.connect(path)
    connection.execute("UPDATE catalog_items SET payload=?", (b"tampered",))
    connection.commit()
    connection.close()
    with storage.ModalChatStorage(path, "namespace-a") as restarted:
        catalog = restarted.catalog("results", encode=lambda value: value, decode=bytes)
        with pytest.raises(storage.ModalChatStorageError):
            catalog.resolve("result-1")


def test_refuses_permissive_or_linked_local_parent(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o755)
    with pytest.raises(storage.ModalChatStorageError):
        storage.ModalChatStorage(tmp_path / "consumer.sqlite3", "namespace-a")
    os.chmod(tmp_path, 0o700)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(private, target_is_directory=True)
    with pytest.raises(storage.ModalChatStorageError):
        storage.ModalChatStorage(linked / "consumer.sqlite3", "namespace-a")


@pytest.mark.parametrize("target", ["database", "lock"])
def test_refuses_hardlinked_database_or_lock(tmp_path: Path, target: str) -> None:
    path = _private(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"")
    os.chmod(unrelated, 0o600)
    destination = path if target == "database" else path.with_name(path.name + ".lock")
    os.link(unrelated, destination)
    before_mode = stat.S_IMODE(unrelated.stat().st_mode)

    with pytest.raises(storage.ModalChatStorageError):
        storage.ModalChatStorage(path, "namespace-a")
    assert stat.S_IMODE(unrelated.stat().st_mode) == before_mode


def test_use_after_close_is_a_closed_storage_failure(tmp_path: Path) -> None:
    path = _private(tmp_path)
    owner = storage.ModalChatStorage(path, "namespace-a")
    catalog = owner.catalog("results", encode=lambda value: value, decode=bytes)
    attempts = owner.attempts
    owner.close()

    with pytest.raises(storage.ModalChatStorageError, match="closed"):
        catalog.resolve("result-1")
    with pytest.raises(storage.ModalChatStorageError, match="closed"):
        attempts.resolve("attempt-1")


def test_malformed_existing_schema_and_identifier_fail_closed(tmp_path: Path) -> None:
    path = _private(tmp_path)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE catalog_items(wrong TEXT)")
    connection.commit()
    connection.close()
    os.chmod(path, 0o600)
    with storage.ModalChatStorage(path, "namespace-a") as owner:
        catalog = owner.catalog("results", encode=lambda value: value, decode=bytes)
        with pytest.raises(
            storage.ModalChatStorageError, match="^modal_chat_storage_invalid$"
        ):
            catalog.resolve("result-1")
        hostile = "secret-value-with spaces"
        with pytest.raises(storage.ModalChatStorageError) as error:
            catalog.resolve(hostile)
        assert hostile not in str(error.value)


def test_close_failure_preserves_active_body_exception() -> None:
    class BadConnection:
        def close(self):
            raise sqlite3.OperationalError("closed badly")

    owner = object.__new__(storage.ModalChatStorage)
    owner._connection = BadConnection()
    owner._lock_fd = os.open("/dev/null", os.O_RDONLY)
    try:
        raise LookupError("body failure")
    except LookupError:
        owner.__exit__(*sys.exc_info())
