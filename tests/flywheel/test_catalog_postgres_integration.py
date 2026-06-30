"""Env-gated Postgres integration tests for the flywheel log catalog."""
from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from shared.flywheel.catalog import InferenceLogRecord, LogFilter, PostgresLogCatalog


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("FLYWHEEL_POSTGRES_DSN")
        or not os.getenv("FLYWHEEL_POSTGRES_TENANT"),
        reason=(
            "requires FLYWHEEL_POSTGRES_DSN and FLYWHEEL_POSTGRES_TENANT "
            "for opt-in Postgres catalog integration"
        ),
    ),
]


def _record(log_id: str, *, model_id: str = "catalog-pg-test") -> InferenceLogRecord:
    return InferenceLogRecord(
        log_id=log_id,
        timestamp="2026-06-30T12:00:00Z",
        model_id=model_id,
        adapter_name="test-adapter",
        tools_requested=True,
        tool_calls=[{"function": {"name": "lookup"}}],
        source_file="postgres-integration.jsonl",
        line_number=1,
    )


async def _assert_verdict_schema(catalog: PostgresLogCatalog, tenant_id: str) -> None:
    async with catalog._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = 'inference_logs'
              AND column_name IN (
                  'verdict_rationale',
                  'rubric_scores',
                  'tenant_id'
              )
            """,
            f"tenant_{tenant_id}",
        )

    columns = {row["column_name"]: dict(row) for row in rows}
    assert columns["verdict_rationale"]["data_type"] == "text"
    assert columns["rubric_scores"]["udt_name"] == "jsonb"
    assert columns["tenant_id"]["data_type"] == "text"


@pytest.fixture
def postgres_env() -> tuple[str, str]:
    asyncpg = pytest.importorskip("asyncpg")
    assert asyncpg is not None
    return (
        os.environ["FLYWHEEL_POSTGRES_DSN"],
        os.environ["FLYWHEEL_POSTGRES_TENANT"],
    )


@pytest.fixture
def tenant_ids(postgres_env) -> tuple[str, str]:
    _, tenant = postgres_env
    if not PostgresLogCatalog._TENANT_ID_RE.match(tenant):
        pytest.skip("FLYWHEEL_POSTGRES_TENANT must match ^[a-zA-Z0-9_]+$")
    suffix = uuid.uuid4().hex[:12]
    return f"{tenant}_catalog_it_{suffix}", f"{tenant}_catalog_iso_{suffix}"


@pytest_asyncio.fixture
async def cleanup_tenant_schemas(postgres_env, tenant_ids):
    asyncpg = pytest.importorskip("asyncpg")
    dsn, _ = postgres_env
    yield
    conn = await asyncpg.connect(dsn)
    try:
        for tenant_id in tenant_ids:
            await conn.execute(f"DROP SCHEMA IF EXISTS tenant_{tenant_id} CASCADE")
    finally:
        await conn.close()


async def test_postgres_catalog_persists_verdict_rubric_and_isolates_tenants(
    postgres_env, tenant_ids, cleanup_tenant_schemas
):
    dsn, _ = postgres_env
    tenant_id, isolated_tenant_id = tenant_ids
    catalog = PostgresLogCatalog(dsn, tenant_id, pool_size=1)
    isolated_catalog = PostgresLogCatalog(dsn, isolated_tenant_id, pool_size=1)
    log_id = f"pg-catalog-{uuid.uuid4().hex}"
    rubric_scores = [
        {
            "rubric_key": "tool_quality",
            "rubric_name": "Tool Quality",
            "score": 0.91,
            "passed": True,
            "pass_threshold": 0.8,
            "feedback": "The response used the requested lookup tool.",
            "per_dimension": [
                {
                    "key": "correctness",
                    "name": "Correctness",
                    "weight": 0.7,
                    "reasoning": "The tool call matches the user request.",
                    "score": 0.93,
                }
            ],
            "quality_gated_score": 0.9,
        }
    ]

    try:
        await catalog.initialize()
        await isolated_catalog.initialize()
        await _assert_verdict_schema(catalog, tenant_id)

        await catalog.insert_log(_record(log_id))
        await catalog.update_score(
            log_id,
            0.91,
            True,
            [],
            verdict_rationale="The answer used the requested lookup tool.",
            rubric_scores=rubric_scores,
        )

        results = await catalog.find_logs(LogFilter(model_id="catalog-pg-test"))

        assert [record.log_id for record in results] == [log_id]
        assert results[0].fitness_score == 0.91
        assert results[0].is_valid is True
        assert (
            results[0].verdict_rationale
            == "The answer used the requested lookup tool."
        )
        assert results[0].rubric_scores == rubric_scores
        assert results[0].tools_requested is True
        assert results[0].tool_calls == [{}]

        await catalog.mark_used([log_id], "dataset-pg-it")

        used = await catalog.find_logs(LogFilter(dataset_version="dataset-pg-it"))
        unused = await catalog.find_logs(LogFilter(unused_only=True))
        isolated_results = await isolated_catalog.find_logs(LogFilter())

        assert [record.log_id for record in used] == [log_id]
        assert unused == []
        assert isolated_results == []
    finally:
        await isolated_catalog.close()
        await catalog.close()
