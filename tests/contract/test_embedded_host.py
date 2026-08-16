"""Contracts for running the engine from a nested, generic host project."""

from __future__ import annotations

import shutil
from pathlib import Path

from tuner.project.context import discover_project_context
from tuner.project.manifest import load_project_manifest
from tuner.project.path_refs import resolve_path


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "host-project"
FROZEN_FIXTURE_MEMBERS = {
    ".gitignore",
    "configs/training config.yaml",
    "data/private dataset.jsonl",
    "plugins/host plugin.py",
    "synaptic.yaml",
    "dependencies/nonstandard engine location/ENGINE_FIXTURE.txt",
}


def _copy_host(tmp_path: Path) -> Path:
    host = tmp_path / "consumer workspace with spaces"
    shutil.copytree(FIXTURE_ROOT, host)
    return host


def test_frozen_host_fixture_has_no_runtime_owned_directories() -> None:
    members = {
        path.relative_to(FIXTURE_ROOT).as_posix()
        for path in FIXTURE_ROOT.rglob("*")
        if path.is_file()
    }

    assert members == FROZEN_FIXTURE_MEMBERS
    assert not (FIXTURE_ROOT / ".synaptic").exists()


def test_fixture_manifest_selects_declared_host_roots_without_creating_them(
    tmp_path: Path,
) -> None:
    host = _copy_host(tmp_path)
    engine = host / "dependencies" / "nonstandard engine location"

    manifest = load_project_manifest(host / "synaptic.yaml")
    context = manifest.create_context(engine_root=engine, invocation_cwd=tmp_path)

    assert manifest.project_id == "contract-host-project"
    assert (engine / "ENGINE_FIXTURE.txt").is_file()
    assert context.config_root == (host / "configs").resolve()
    assert context.writable_roots == tuple(
        (host / ".synaptic" / name).resolve()
        for name in ("artifacts", "state", "tracking", "cache", "tmp")
    )
    assert not (host / ".synaptic").exists()


def test_config_discovers_host_around_nonstandard_nested_engine(
    tmp_path: Path,
) -> None:
    host = _copy_host(tmp_path)
    engine = host / "dependencies" / "nonstandard engine location"
    config = host / "configs" / "training config.yaml"
    unrelated_cwd = tmp_path / "unrelated invocation directory"
    unrelated_cwd.mkdir()

    context = discover_project_context(
        engine_root=engine,
        primary_config=config,
        invocation_cwd=unrelated_cwd,
        environment={},
    )

    assert context.mode == "host"
    assert context.path_mode == "project_v1"
    assert context.project_root == host.resolve()
    assert context.engine_root == engine.resolve()
    assert context.invocation_cwd == unrelated_cwd.resolve()
    assert context.manifest_path == (host / "synaptic.yaml").resolve()
    assert context.artifact_root == (host / ".synaptic" / "artifacts").resolve()
    assert context.state_root == (host / ".synaptic" / "state").resolve()
    assert context.tracking_root == (host / ".synaptic" / "tracking").resolve()
    assert context.cache_root == (host / ".synaptic" / "cache").resolve()
    assert context.tmp_root == (host / ".synaptic" / "tmp").resolve()
    assert not (host / ".synaptic").exists()


def test_paths_with_spaces_resolve_from_their_declaring_document(
    tmp_path: Path,
) -> None:
    host = _copy_host(tmp_path)
    engine = host / "dependencies" / "nonstandard engine location"
    config = host / "configs" / "training config.yaml"
    context = discover_project_context(
        engine_root=engine,
        explicit_project_root=host,
        invocation_cwd=tmp_path,
        environment={},
    )

    dataset = resolve_path(
        "../data/private dataset.jsonl",
        context,
        declaring_file=config,
    )
    output = resolve_path("artifact://contract run", context, access="write")

    assert dataset == (host / "data" / "private dataset.jsonl").resolve()
    assert dataset.is_file()
    assert output == (host / ".synaptic" / "artifacts" / "contract run").resolve()
