"""Public host ports and the in-memory reference stores (api-facade slice 2)."""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1 import ports
from synaptic_tuner.api.v1.ports import (
    STORAGE_PARTITIONS, ClockPort, DurableRecordStorePort, DurableStreamStorePort,
    GrantAuthorityPort, SecretResolverPort, StoragePartition, StoredPageV1, StoredRecordV1,
    StoredStreamEntryV1, StoredStreamPageV1, require_partition,
)
from synaptic_tuner.api.v1.reference.stores import (
    InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1,
)


ROOT = Path(__file__).resolve().parents[2]
PARTITION = "workflow"
STREAM = "observation"

EXPECTED_PARTITIONS = frozenset({
    "workflow", "plan", "plan_context", "preparation", "execution_grant",
    "reconciliation_grant", "publication", "evaluation", "data", "pipeline",
    "chat_session", "observation", "effects", "authorization",
})
PORT_NAMES = (
    "ClockPort", "SecretResolverPort", "GrantAuthorityPort",
    "DurableRecordStorePort", "DurableStreamStorePort", "StoragePartition",
    "StoredRecordV1", "StoredPageV1", "StoredStreamEntryV1", "StoredStreamPageV1",
)


def _methods(protocol: type) -> frozenset[str]:
    return frozenset(
        name for name, value in vars(protocol).items()
        if inspect.isfunction(value) and not name.startswith("_")
    )


# --- closed vocabulary and protocol shape ------------------------------------

def test_partition_vocabulary_is_closed() -> None:
    assert frozenset(member.value for member in StoragePartition) == EXPECTED_PARTITIONS
    assert STORAGE_PARTITIONS == EXPECTED_PARTITIONS
    assert len(StoragePartition) == 14
    for value in EXPECTED_PARTITIONS:
        assert require_partition(value) == value
    for unknown in ("lifecycle", "Workflow", " workflow", "", "observation\n"):
        with pytest.raises((ValueError, TypeError)):
            require_partition(unknown)
    with pytest.raises(TypeError):
        require_partition(StoragePartition.WORKFLOW.value.encode())
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    with pytest.raises(ValueError, match="unknown storage partition"):
        records.create(partition="lifecycle", key="k", canonical=b"{}")
    with pytest.raises(ValueError, match="unknown storage partition"):
        records.list_page(partition="lifecycle", prefix="", after_key=None, limit=1)
    with pytest.raises(ValueError, match="unknown storage partition"):
        streams.append(partition="lifecycle", stream_key="s", sequence=0, canonical=b"{}")
    with pytest.raises(ValueError, match="unknown storage partition"):
        streams.read_page(partition="lifecycle", stream_key="s", after_sequence=None, limit=1)


def test_host_port_protocols_expose_exact_verb_sets() -> None:
    assert _methods(ClockPort) == {"now", "now_epoch"}
    assert _methods(SecretResolverPort) == {"resolve"}
    assert _methods(GrantAuthorityPort) == {"authorize", "bind"}
    assert _methods(DurableRecordStorePort) == {"create", "read", "compare_and_swap", "put_if_absent", "list_page"}
    assert _methods(DurableStreamStorePort) == {"append", "read_page"}
    assert _methods(InMemoryDurableRecordStoreV1) >= _methods(DurableRecordStorePort)
    assert _methods(InMemoryDurableStreamStoreV1) >= _methods(DurableStreamStorePort)
    for port, implementation in (
        (DurableRecordStorePort, InMemoryDurableRecordStoreV1),
        (DurableStreamStorePort, InMemoryDurableStreamStoreV1),
    ):
        for name in _methods(port):
            assert inspect.signature(getattr(port, name)) == inspect.signature(getattr(implementation, name)), name


def test_ports_are_formal_exports_and_reference_is_not() -> None:
    assert v1._LAZY_MODULE_ATTRIBUTES["ports"] == set(PORT_NAMES)
    assert set(PORT_NAMES) <= set(v1._FORMAL_EXPORTS)
    assert set(ports.__all__) == set(PORT_NAMES)
    for name in PORT_NAMES:
        assert getattr(v1, name) is getattr(ports, name)
    assert "reference" not in v1._LAZY_MODULE_ATTRIBUTES
    assert not any(name.startswith("compose_") or "InMemoryDurable" in name for name in v1._FORMAL_EXPORTS)


def test_superseded_host_v1_package_is_gone() -> None:
    assert "host" not in {entry.name for entry in (ROOT / "synaptic_tuner").iterdir()}
    assert importlib.util.find_spec(".host", "synaptic_tuner") is None


# --- record shapes ------------------------------------------------------------

def test_stored_record_rejects_bad_fields() -> None:
    record = StoredRecordV1("wf/1", 1, b"{}")
    assert (record.key, record.revision, record.canonical) == ("wf/1", 1, b"{}")
    with pytest.raises(ValueError):
        StoredRecordV1("", 1, b"{}")
    with pytest.raises(ValueError):
        StoredRecordV1("wf/1", 0, b"{}")
    with pytest.raises(ValueError):
        StoredRecordV1("wf/1", True, b"{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        StoredRecordV1("wf/1", 1, "{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        StoredRecordV1("wf/1", 1, bytearray(b"{}"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        StoredRecordV1("wf/1", 1, b"")


def test_stored_pages_keep_next_cursor_and_truncated_in_agreement() -> None:
    first, second = StoredRecordV1("a", 1, b"1"), StoredRecordV1("b", 1, b"2")
    assert StoredPageV1((first, second), "b", True).truncated
    assert StoredPageV1((first, second)).next_cursor is None
    with pytest.raises(ValueError, match="matrix"):
        StoredPageV1((first,), "a", False)
    with pytest.raises(ValueError, match="matrix"):
        StoredPageV1((first,), None, True)
    with pytest.raises(ValueError, match="last record key"):
        StoredPageV1((first, second), "a", True)
    with pytest.raises(ValueError, match="strictly increasing"):
        StoredPageV1((second, first))
    with pytest.raises(ValueError, match="strictly increasing"):
        StoredPageV1((first, first))
    with pytest.raises(TypeError):
        StoredPageV1([first])  # type: ignore[arg-type]

    e0, e1 = StoredStreamEntryV1(0, b"1"), StoredStreamEntryV1(7, b"2")
    assert StoredStreamPageV1((e0, e1), 7, True).truncated
    with pytest.raises(ValueError, match="matrix"):
        StoredStreamPageV1((e0,), None, True)
    with pytest.raises(ValueError, match="last entry sequence"):
        StoredStreamPageV1((e0, e1), 0, True)
    with pytest.raises(ValueError, match="strictly increasing"):
        StoredStreamPageV1((e1, e0))
    with pytest.raises(ValueError, match="strictly increasing"):
        StoredStreamPageV1((e0, e0))
    with pytest.raises(ValueError, match="a truncated page"):
        StoredStreamPageV1((), 0, True)


# --- record store -------------------------------------------------------------

def test_compare_and_swap_rejects_a_stale_revision() -> None:
    store = InMemoryDurableRecordStoreV1()
    assert store.create(partition=PARTITION, key="wf/1", canonical=b'{"v":1}') is True
    assert store.read(partition=PARTITION, key="wf/1") == StoredRecordV1("wf/1", 1, b'{"v":1}')

    assert store.compare_and_swap(partition=PARTITION, key="wf/1", expected_revision=1, canonical=b'{"v":2}') is True
    assert store.read(partition=PARTITION, key="wf/1") == StoredRecordV1("wf/1", 2, b'{"v":2}')

    # A writer still holding revision 1 loses and changes nothing.
    assert store.compare_and_swap(partition=PARTITION, key="wf/1", expected_revision=1, canonical=b'{"v":3}') is False
    assert store.read(partition=PARTITION, key="wf/1") == StoredRecordV1("wf/1", 2, b'{"v":2}')
    # A future revision and an absent key also lose.
    assert store.compare_and_swap(partition=PARTITION, key="wf/1", expected_revision=3, canonical=b'{"v":3}') is False
    assert store.compare_and_swap(partition=PARTITION, key="wf/2", expected_revision=1, canonical=b'{"v":3}') is False
    assert store.read(partition=PARTITION, key="wf/2") is None
    with pytest.raises(ValueError):
        store.compare_and_swap(partition=PARTITION, key="wf/1", expected_revision=0, canonical=b'{"v":3}')


def test_create_is_first_claim_and_partitions_are_isolated() -> None:
    store = InMemoryDurableRecordStoreV1()
    assert store.create(partition=PARTITION, key="k", canonical=b"1") is True
    assert store.create(partition=PARTITION, key="k", canonical=b"1") is False
    assert store.create(partition=PARTITION, key="k", canonical=b"2") is False
    assert store.read(partition=PARTITION, key="k") == StoredRecordV1("k", 1, b"1")
    assert store.read(partition="plan", key="k") is None
    assert store.create(partition="plan", key="k", canonical=b"2") is True
    assert store.read(partition=PARTITION, key="k").canonical == b"1"


def test_put_if_absent_is_idempotent_and_never_rewrites() -> None:
    store = InMemoryDurableRecordStoreV1()
    assert store.put_if_absent(partition=PARTITION, key="k", canonical=b"1") is True
    assert store.put_if_absent(partition=PARTITION, key="k", canonical=b"1") is True
    assert store.put_if_absent(partition=PARTITION, key="k", canonical=b"1") is True
    assert store.read(partition=PARTITION, key="k") == StoredRecordV1("k", 1, b"1")
    # Different bytes at the same key are refused and the record is untouched.
    assert store.put_if_absent(partition=PARTITION, key="k", canonical=b"2") is False
    assert store.read(partition=PARTITION, key="k") == StoredRecordV1("k", 1, b"1")
    # Once the record has advanced, even identical bytes are no longer "absent".
    assert store.compare_and_swap(partition=PARTITION, key="k", expected_revision=1, canonical=b"1") is True
    assert store.put_if_absent(partition=PARTITION, key="k", canonical=b"1") is False
    assert store.read(partition=PARTITION, key="k").revision == 2


def test_list_page_honours_prefix_limit_and_cursor() -> None:
    store = InMemoryDurableRecordStoreV1()
    for key in ("run/3", "run/1", "plan/9", "run/2", "run/10"):
        assert store.create(partition=PARTITION, key=key, canonical=key.encode())

    page = store.list_page(partition=PARTITION, prefix="run/", after_key=None, limit=2)
    assert [item.key for item in page.records] == ["run/1", "run/10"]
    assert page.truncated is True and page.next_cursor == "run/10"

    page = store.list_page(partition=PARTITION, prefix="run/", after_key=page.next_cursor, limit=2)
    assert [item.key for item in page.records] == ["run/2", "run/3"]
    assert page.truncated is False and page.next_cursor is None

    page = store.list_page(partition=PARTITION, prefix="run/", after_key="run/3", limit=2)
    assert page.records == () and page.truncated is False

    everything = store.list_page(partition=PARTITION, prefix="", after_key=None, limit=1000)
    assert [item.key for item in everything.records] == ["plan/9", "run/1", "run/10", "run/2", "run/3"]
    assert store.list_page(partition="plan", prefix="", after_key=None, limit=5).records == ()
    with pytest.raises(ValueError):
        store.list_page(partition=PARTITION, prefix="", after_key=None, limit=0)
    with pytest.raises(ValueError):
        store.list_page(partition=PARTITION, prefix="", after_key=None, limit=1001)
    with pytest.raises(TypeError):
        store.list_page(partition=PARTITION, prefix=None, after_key=None, limit=1)  # type: ignore[arg-type]


# --- stream store -------------------------------------------------------------

def test_stream_append_rejects_a_non_monotone_sequence() -> None:
    store = InMemoryDurableStreamStoreV1()
    assert store.append(partition=STREAM, stream_key="ev/1", sequence=0, canonical=b"a") is True
    assert store.append(partition=STREAM, stream_key="ev/1", sequence=5, canonical=b"b") is True
    assert store.append(partition=STREAM, stream_key="ev/1", sequence=5, canonical=b"b") is False
    assert store.append(partition=STREAM, stream_key="ev/1", sequence=4, canonical=b"c") is False
    assert store.append(partition=STREAM, stream_key="ev/1", sequence=0, canonical=b"d") is False
    page = store.read_page(partition=STREAM, stream_key="ev/1", after_sequence=None, limit=10)
    assert page.entries == (StoredStreamEntryV1(0, b"a"), StoredStreamEntryV1(5, b"b"))
    # Other streams are independent and may start anywhere.
    assert store.append(partition=STREAM, stream_key="ev/2", sequence=3, canonical=b"z") is True
    assert store.append(partition="chat_session", stream_key="ev/1", sequence=0, canonical=b"z") is True
    with pytest.raises(ValueError):
        store.append(partition=STREAM, stream_key="ev/1", sequence=-1, canonical=b"e")
    with pytest.raises(ValueError):
        store.append(partition=STREAM, stream_key="ev/1", sequence=True, canonical=b"e")  # type: ignore[arg-type]


def test_stream_read_page_honours_limit_and_cursor() -> None:
    store = InMemoryDurableStreamStoreV1()
    for sequence in (1, 2, 4, 8, 16):
        assert store.append(partition=STREAM, stream_key="s", sequence=sequence, canonical=str(sequence).encode())

    page = store.read_page(partition=STREAM, stream_key="s", after_sequence=None, limit=2)
    assert [item.sequence for item in page.entries] == [1, 2]
    assert page.truncated is True and page.next_cursor == 2

    page = store.read_page(partition=STREAM, stream_key="s", after_sequence=page.next_cursor, limit=2)
    assert [item.sequence for item in page.entries] == [4, 8]
    assert page.truncated is True and page.next_cursor == 8

    page = store.read_page(partition=STREAM, stream_key="s", after_sequence=page.next_cursor, limit=2)
    assert [item.sequence for item in page.entries] == [16]
    assert page.truncated is False and page.next_cursor is None

    assert store.read_page(partition=STREAM, stream_key="s", after_sequence=16, limit=2).entries == ()
    assert store.read_page(partition=STREAM, stream_key="missing", after_sequence=None, limit=2).entries == ()
    with pytest.raises(ValueError):
        store.read_page(partition=STREAM, stream_key="s", after_sequence=None, limit=0)


# --- import closure -----------------------------------------------------------

def test_reference_stores_import_no_sqlite3_provider_or_engine_modules() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1.ports
import synaptic_tuner.api.v1.reference
import synaptic_tuner.api.v1.reference.stores
print(json.dumps(sorted(n for n in sys.modules if n in ('tuner', 'sqlite3', 'modal', 'huggingface_hub', 'runpod') or n.startswith(('tuner.', 'modal.', 'sqlite3.', 'huggingface_hub.', 'runpod.')))))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []
