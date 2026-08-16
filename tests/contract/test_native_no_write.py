"""Native execution detects source mutation and confines managed writes."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from tuner.project.context import discover_project_context
from tuner.project.path_refs import resolve_path


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "host-project"


@pytest.mark.skip(
    reason=(
        "Nodes D/F activation gate: a context-aware representative built-in "
        "native dry-run must exist and route all managed output to .synaptic"
    )
)
def test_deferred_nodes_d_f_builtin_dry_run_preserves_host_and_engine_sources() -> None:
    """Activate with a real built-in dry run after the D/F runtime seam lands."""
    pytest.fail("Nodes D/F must replace this deferred gate with a built-in dry-run assertion")


def _source_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and ".synaptic" not in path.relative_to(root).parts
    }


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _initialize_host_repository(host: Path) -> None:
    _git("init", "--quiet", cwd=host)
    _git("add", ".", cwd=host)
    _git(
        "-c",
        "user.name=Contract Test",
        "-c",
        "user.email=contract@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
        cwd=host,
    )


def _porcelain(host: Path) -> str:
    return _git("status", "--short", "--untracked-files=all", cwd=host).stdout


def test_context_and_managed_outputs_leave_native_source_trees_unchanged(
    tmp_path: Path,
) -> None:
    host = tmp_path / "native host with spaces"
    shutil.copytree(FIXTURE_ROOT, host)
    engine = host / "dependencies" / "nonstandard engine location"
    before_host = _source_snapshot(host)
    before_engine = _source_snapshot(engine)

    context = discover_project_context(
        engine_root=engine,
        explicit_project_root=host,
        invocation_cwd=tmp_path,
        environment={},
    )
    for reference in (
        "artifact://dry-run/result.json",
        "state://dry-run/state.json",
        "tracking://dry-run/source-lock.json",
        "cache://dry-run/model.bin",
        "tmp://dry-run/staging.json",
    ):
        destination = resolve_path(reference, context, access="write")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("contract output\n", encoding="utf-8")

    assert _source_snapshot(host) == before_host
    assert _source_snapshot(engine) == before_engine
    assert {path.parent.name for path in context.writable_roots} == {".synaptic"}


def test_git_detection_ignores_runtime_roots_but_reports_source_mutation(
    tmp_path: Path,
) -> None:
    host = tmp_path / "git host with spaces"
    shutil.copytree(FIXTURE_ROOT, host)
    _initialize_host_repository(host)
    assert _porcelain(host) == ""

    runtime_file = host / ".synaptic" / "tracking" / "run.json"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("{}\n", encoding="utf-8")
    assert _porcelain(host) == ""

    source_file = host / "configs" / "training config.yaml"
    source_file.write_text(
        source_file.read_text(encoding="utf-8") + "# accidental mutation\n",
        encoding="utf-8",
    )
    assert "configs/training config.yaml" in _porcelain(host).replace("\\", "/")
