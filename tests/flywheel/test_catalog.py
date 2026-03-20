"""Tests for shared.flywheel.catalog — data structures and SQLiteLogCatalog."""
from __future__ import annotations

import json

import pytest
import pytest_asyncio

from shared.flywheel.catalog import (
    DatasetVersion,
    InferenceLogRecord,
    LogFilter,
    SQLiteLogCatalog,
    create_catalog,
)


# ---------------------------------------------------------------------------
# InferenceLogRecord
# ---------------------------------------------------------------------------

class TestInferenceLogRecord:
    """InferenceLogRecord creation and serialization."""

    def test_default_fields(self):
        r = InferenceLogRecord(log_id="abc", timestamp="2026-01-01T00:00:00Z", model_id="m1")
        assert r.log_id == "abc"
        assert r.tools_requested is False
        assert r.tool_calls == []
        assert r.fitness_score is None
        assert r.tag is None
        assert r.errors == []

    def test_to_json_roundtrip(self):
        r = InferenceLogRecord(
            log_id="test-1",
            timestamp="2026-01-01T00:00:00Z",
            model_id="gpt-test",
            tools_requested=True,
            tool_calls=[{"function": {"name": "search"}}],
            fitness_score=0.85,
        )
        json_str = r.to_json()
        data = json.loads(json_str)
        assert data["log_id"] == "test-1"
        assert data["tools_requested"] is True
        assert len(data["tool_calls"]) == 1
        assert data["fitness_score"] == 0.85

    def test_from_dict_ignores_unknown_fields(self):
        data = {
            "log_id": "x",
            "timestamp": "2026-01-01",
            "model_id": "m",
            "future_field": "ignored",
        }
        r = InferenceLogRecord.from_dict(data)
        assert r.log_id == "x"
        assert not hasattr(r, "future_field")


class TestDatasetVersion:
    """DatasetVersion creation and serialization."""

    def test_to_dict_roundtrip(self):
        v = DatasetVersion(
            version_id="v001",
            created_at="2026-01-01T00:00:00Z",
            source_model_id="model-a",
            record_counts={"sft": 100, "kto_pos": 50},
            file_paths={"sft": "path/to/sft.jsonl"},
            content_hash="abc123",
        )
        d = v.to_dict()
        v2 = DatasetVersion.from_dict(d)
        assert v2.version_id == "v001"
        assert v2.record_counts["sft"] == 100
        assert v2.content_hash == "abc123"

    def test_from_dict_ignores_unknown(self):
        data = {
            "version_id": "v1",
            "created_at": "now",
            "source_model_id": "m",
            "record_counts": {},
            "file_paths": {},
            "content_hash": "",
            "extra": "nope",
        }
        v = DatasetVersion.from_dict(data)
        assert v.version_id == "v1"


# ---------------------------------------------------------------------------
# SQLiteLogCatalog
# ---------------------------------------------------------------------------

def _make_record(log_id: str, **kwargs) -> InferenceLogRecord:
    """Helper to create a test record with sensible defaults."""
    defaults = dict(
        timestamp="2026-01-15T12:00:00Z",
        model_id="test-model",
        source_file="test.jsonl",
        line_number=0,
    )
    defaults.update(kwargs)
    return InferenceLogRecord(log_id=log_id, **defaults)


@pytest_asyncio.fixture
async def catalog(tmp_path) -> SQLiteLogCatalog:
    """Create and initialize an SQLiteLogCatalog with a temp DB."""
    cat = SQLiteLogCatalog(tmp_path / "test.db")
    await cat.initialize()
    yield cat
    await cat.close()


@pytest.mark.asyncio
class TestSQLiteLogCatalog:
    """SQLiteLogCatalog CRUD operations."""

    async def test_insert_and_find(self, catalog):
        record = _make_record("log-1", tool_calls=[{"fn": "test"}], tools_requested=True)
        await catalog.insert_log(record)

        results = await catalog.find_logs(LogFilter())
        assert len(results) == 1
        assert results[0].log_id == "log-1"
        assert results[0].tools_requested is True

    async def test_insert_duplicate_is_ignored(self, catalog):
        record = _make_record("dup-1")
        await catalog.insert_log(record)
        await catalog.insert_log(record)

        count = await catalog.count_logs(LogFilter())
        assert count == 1

    async def test_batch_insert(self, catalog):
        records = [_make_record(f"batch-{i}") for i in range(5)]
        inserted = await catalog.insert_logs_batch(records)
        assert inserted == 5

        count = await catalog.count_logs(LogFilter())
        assert count == 5

    async def test_find_with_tag_filter(self, catalog):
        r1 = _make_record("t1")
        r2 = _make_record("t2")
        await catalog.insert_log(r1)
        await catalog.insert_log(r2)

        await catalog.update_tag("t1", "sft", "rule")
        await catalog.update_tag("t2", "kto", "rule")

        sft = await catalog.find_logs(LogFilter(tag="sft"))
        assert len(sft) == 1
        assert sft[0].log_id == "t1"

    async def test_find_with_tag_list_filter(self, catalog):
        for i, tag in enumerate(["sft", "kto", "discard"]):
            r = _make_record(f"tl-{i}")
            await catalog.insert_log(r)
            await catalog.update_tag(f"tl-{i}", tag, "rule")

        results = await catalog.find_logs(LogFilter(tag=["sft", "kto"]))
        assert len(results) == 2

    async def test_find_with_score_filter(self, catalog):
        r = _make_record("s1")
        await catalog.insert_log(r)
        await catalog.update_score("s1", 0.85, True, [])

        high = await catalog.find_logs(LogFilter(min_score=0.8))
        assert len(high) == 1

        low = await catalog.find_logs(LogFilter(max_score=0.5))
        assert len(low) == 0

    async def test_unscored_only_filter(self, catalog):
        r1 = _make_record("us1")
        r2 = _make_record("us2")
        await catalog.insert_log(r1)
        await catalog.insert_log(r2)
        await catalog.update_score("us1", 0.9, True, [])

        unscored = await catalog.find_logs(LogFilter(unscored_only=True))
        assert len(unscored) == 1
        assert unscored[0].log_id == "us2"

    async def test_untagged_only_filter(self, catalog):
        r1 = _make_record("ut1")
        r2 = _make_record("ut2")
        await catalog.insert_log(r1)
        await catalog.insert_log(r2)
        await catalog.update_tag("ut1", "sft", "rule")

        untagged = await catalog.find_logs(LogFilter(untagged_only=True))
        assert len(untagged) == 1
        assert untagged[0].log_id == "ut2"

    async def test_mark_used(self, catalog):
        r1 = _make_record("mu1")
        r2 = _make_record("mu2")
        await catalog.insert_log(r1)
        await catalog.insert_log(r2)

        await catalog.mark_used(["mu1"], "v001")

        unused = await catalog.find_logs(LogFilter(unused_only=True))
        assert len(unused) == 1
        assert unused[0].log_id == "mu2"

    async def test_mark_used_empty_list_is_noop(self, catalog):
        await catalog.mark_used([], "v001")  # Should not raise

    async def test_create_and_get_dataset_version(self, catalog):
        version = DatasetVersion(
            version_id="v001",
            created_at="2026-01-15T12:00:00Z",
            source_model_id="test-model",
            record_counts={"sft": 50},
            file_paths={"sft": "/path/to/sft.jsonl"},
            content_hash="hash123",
        )
        vid = await catalog.create_dataset_version(version)
        assert vid == "v001"

        retrieved = await catalog.get_dataset_version("v001")
        assert retrieved is not None
        assert retrieved.version_id == "v001"
        assert retrieved.record_counts == {"sft": 50}

    async def test_get_nonexistent_version_returns_none(self, catalog):
        result = await catalog.get_dataset_version("nonexistent")
        assert result is None

    async def test_get_latest_dataset_version(self, catalog):
        v1 = DatasetVersion(
            version_id="v001", created_at="2026-01-01T00:00:00Z",
            source_model_id="m", record_counts={}, file_paths={},
            content_hash="",
        )
        v2 = DatasetVersion(
            version_id="v002", created_at="2026-01-02T00:00:00Z",
            source_model_id="m", record_counts={}, file_paths={},
            content_hash="",
        )
        await catalog.create_dataset_version(v1)
        await catalog.create_dataset_version(v2)

        latest = await catalog.get_latest_dataset_version()
        assert latest is not None
        assert latest.version_id == "v002"

    async def test_get_latest_version_empty_returns_none(self, catalog):
        latest = await catalog.get_latest_dataset_version()
        assert latest is None

    async def test_count_logs(self, catalog):
        for i in range(3):
            await catalog.insert_log(_make_record(f"c-{i}"))

        count = await catalog.count_logs(LogFilter())
        assert count == 3

    async def test_limit_filter(self, catalog):
        for i in range(10):
            await catalog.insert_log(_make_record(f"lim-{i}"))

        results = await catalog.find_logs(LogFilter(limit=3))
        assert len(results) == 3

    async def test_has_tool_calls_filter(self, catalog):
        r1 = _make_record("tc1", tool_calls=[{"fn": "x"}])
        r2 = _make_record("tc2", tool_calls=[])
        await catalog.insert_log(r1)
        await catalog.insert_log(r2)

        with_tools = await catalog.find_logs(LogFilter(has_tool_calls=True))
        assert len(with_tools) == 1
        assert with_tools[0].log_id == "tc1"


# ---------------------------------------------------------------------------
# create_catalog factory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCreateCatalog:
    """create_catalog() factory function."""

    async def test_sqlite_backend(self, tmp_path):
        cat = await create_catalog("sqlite", path=tmp_path / "factory.db")
        try:
            assert cat is not None
            await cat.insert_log(_make_record("factory-1"))
            count = await cat.count_logs(LogFilter())
            assert count == 1
        finally:
            await cat.close()

    async def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown catalog backend"):
            await create_catalog("redis")

    async def test_postgres_without_url_raises(self):
        with pytest.raises(ValueError, match="requires 'url'"):
            await create_catalog("postgres")

    async def test_postgres_without_tenant_raises(self):
        with pytest.raises(ValueError, match="requires 'tenant_id'"):
            await create_catalog("postgres", url="postgres://localhost/db")
