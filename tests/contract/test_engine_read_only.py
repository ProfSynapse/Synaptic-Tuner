"""Engine-managed path writes must never target host or engine source."""

from __future__ import annotations

from pathlib import Path

import pytest

from tuner.project.context import ProjectContext
from tuner.project.errors import WriteAccessError
from tuner.project.path_refs import resolve_path


def _embedded_context(tmp_path: Path) -> ProjectContext:
    host = (tmp_path / "host source").resolve()
    engine = host / "third party" / "engine source"
    engine.mkdir(parents=True)
    return ProjectContext.host(
        engine_root=engine,
        project_root=host,
        invocation_cwd=tmp_path,
    )


@pytest.mark.parametrize(
    "reference",
    [
        "engine://generated.json",
        "project://configs/generated.json",
        "config://generated.json",
    ],
)
def test_engine_managed_writes_reject_source_roots(
    tmp_path: Path, reference: str
) -> None:
    context = _embedded_context(tmp_path)
    declaring_file = context.project_root / "configs" / "job.yaml"

    with pytest.raises(WriteAccessError) as exc_info:
        resolve_path(
            reference,
            context,
            declaring_file=declaring_file,
            access="write",
        )

    assert exc_info.value.code == "PROJECT_WRITE_DENIED"


@pytest.mark.parametrize(
    ("reference", "root_name"),
    [
        ("artifact://runs/one", "artifact_root"),
        ("state://runs/one", "state_root"),
        ("tracking://runs/one", "tracking_root"),
        ("cache://models/one", "cache_root"),
        ("tmp://staging/one", "tmp_root"),
    ],
)
def test_engine_managed_writes_select_only_declared_writable_roots(
    tmp_path: Path, reference: str, root_name: str
) -> None:
    context = _embedded_context(tmp_path)

    resolved = resolve_path(reference, context, access="write")

    root = getattr(context, root_name)
    assert resolved.is_relative_to(root)
    assert not resolved.is_relative_to(context.engine_root)
    assert resolved != context.project_root


def test_source_reads_remain_available_without_weakening_write_policy(
    tmp_path: Path,
) -> None:
    context = _embedded_context(tmp_path)

    engine_input = resolve_path("engine://schemas/input.json", context)
    project_input = resolve_path("project://data/input.jsonl", context)

    assert engine_input == context.engine_root / "schemas" / "input.json"
    assert project_input == context.project_root / "data" / "input.jsonl"

