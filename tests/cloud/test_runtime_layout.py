import os
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from tuner.cloud.runtime_layout import (
    RUNTIME_LAYOUT_SCHEMA,
    CloudRuntimeLayout,
    RuntimeMount,
    build_runtime_layout,
)
from tuner.project.context import ProjectContext
from tuner.project.errors import SourceLockError


def _context(tmp_path: Path) -> ProjectContext:
    engine = tmp_path / "host" / "deps" / "engine"
    project = tmp_path / "host"
    engine.mkdir(parents=True)
    return ProjectContext.host(engine_root=engine, project_root=project)


def test_runtime_layout_separates_read_only_sources_and_writable_roots(tmp_path: Path) -> None:
    context = _context(tmp_path)
    layout = build_runtime_layout(context)
    assert layout.schema_version == RUNTIME_LAYOUT_SCHEMA
    assert layout.engine.target == PurePosixPath("/workspace/engine")
    assert layout.project.target == PurePosixPath("/workspace/project")
    assert layout.engine.read_only and layout.project.read_only
    assert tuple(layout.writable_by_name) == ("artifacts", "state", "tracking", "cache", "tmp")
    assert all(not mount.read_only for mount in layout.writable)
    assert layout.writable_by_name["artifacts"].source == context.artifact_root


def test_runtime_layout_serialization_is_provider_neutral(tmp_path: Path) -> None:
    payload = build_runtime_layout(_context(tmp_path)).to_dict()
    assert payload["schema_version"] == "synaptic-runtime-layout/v1"
    assert {root["target"] for root in payload["source_roots"]} == {
        "/workspace/engine",
        "/workspace/project",
    }
    assert {root["name"] for root in payload["writable_roots"]} == {
        "artifacts",
        "state",
        "tracking",
        "cache",
        "tmp",
    }


def test_runtime_layout_rejects_writable_source_roots(tmp_path: Path) -> None:
    source = tmp_path / "source"
    with pytest.raises(SourceLockError, match="read-only"):
        CloudRuntimeLayout(
            engine=RuntimeMount("engine", source, PurePosixPath("/workspace/engine"), False),
            project=RuntimeMount("project", source, PurePosixPath("/workspace/project"), True),
            writable=tuple(
                RuntimeMount(name, tmp_path / name, PurePosixPath("/workspace") / name, False)
                for name in ("artifacts", "state", "tracking", "cache", "tmp")
            ),
        )


def test_runtime_layout_rejects_overlapping_targets(tmp_path: Path) -> None:
    with pytest.raises(SourceLockError, match="overlap"):
        CloudRuntimeLayout(
            engine=RuntimeMount("engine", tmp_path / "engine", PurePosixPath("/workspace/engine"), True),
            project=RuntimeMount("project", tmp_path / "project", PurePosixPath("/workspace/project"), True),
            writable=(
                RuntimeMount("artifacts", tmp_path / "a", PurePosixPath("/workspace/project/artifacts"), False),
                RuntimeMount("state", tmp_path / "s", PurePosixPath("/workspace/state"), False),
                RuntimeMount("tracking", tmp_path / "t", PurePosixPath("/workspace/tracking"), False),
                RuntimeMount("cache", tmp_path / "c", PurePosixPath("/workspace/cache"), False),
                RuntimeMount("tmp", tmp_path / "x", PurePosixPath("/workspace/tmp"), False),
            ),
        )


def test_runtime_layout_rejects_duplicate_canonical_targets(tmp_path: Path) -> None:
    with pytest.raises(SourceLockError, match="distinct canonical"):
        CloudRuntimeLayout(
            engine=RuntimeMount(
                "engine", tmp_path / "engine", PurePosixPath("/workspace/source"), True
            ),
            project=RuntimeMount(
                "project", tmp_path / "project", PurePosixPath("/workspace/source"), True
            ),
            writable=tuple(
                RuntimeMount(
                    name, tmp_path / name, PurePosixPath("/runtime") / name, False
                )
                for name in ("artifacts", "state", "tracking", "cache", "tmp")
            ),
        )


@pytest.mark.parametrize(
    "workspace_root",
    (
        "/workspace/./project",
        "/workspace/staging/../project",
        "/workspace//project",
        "/workspace/project/",
    ),
)
def test_runtime_layout_rejects_noncanonical_workspace_root(
    tmp_path: Path, workspace_root: str
) -> None:
    with pytest.raises(SourceLockError, match="canonical POSIX"):
        build_runtime_layout(_context(tmp_path), workspace_root=workspace_root)


def test_runtime_layout_rejects_semantic_target_alias_before_distinctness(
    tmp_path: Path,
) -> None:
    with pytest.raises(SourceLockError, match="canonical POSIX"):
        CloudRuntimeLayout(
            engine=RuntimeMount(
                "engine",
                tmp_path / "engine",
                PurePosixPath("/workspace/x/../project"),
                True,
            ),
            project=RuntimeMount(
                "project", tmp_path / "project", PurePosixPath("/workspace/project"), True
            ),
            writable=tuple(
                RuntimeMount(
                    name, tmp_path / name, PurePosixPath("/runtime") / name, False
                )
                for name in ("artifacts", "state", "tracking", "cache", "tmp")
            ),
        )


@pytest.mark.parametrize(
    "target",
    ("/workspace/./engine", "/workspace//engine", "/workspace/engine/"),
)
def test_runtime_mount_rejects_noncanonical_target_spelling(
    tmp_path: Path, target: str
) -> None:
    with pytest.raises(SourceLockError, match="canonical POSIX"):
        RuntimeMount("engine", tmp_path / "engine", target, True)  # type: ignore[arg-type]


@pytest.mark.parametrize("workspace_root", ("/", "/workspace/Training Ω/模型"))
def test_runtime_layout_preserves_canonical_root_spaces_and_unicode(
    tmp_path: Path, workspace_root: str
) -> None:
    layout = build_runtime_layout(_context(tmp_path), workspace_root=workspace_root)
    workspace = PurePosixPath(workspace_root)
    assert layout.engine.target == workspace / "engine"
    assert layout.project.target == workspace / "project"


def test_runtime_layout_requires_absolute_workspace(tmp_path: Path) -> None:
    with pytest.raises(SourceLockError, match="absolute"):
        build_runtime_layout(_context(tmp_path), workspace_root=PurePosixPath("relative"))


def test_standalone_legacy_context_projects_contract8_writable_carveouts_without_writes(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "engine"
    engine.mkdir()
    before = set(engine.rglob("*"))
    layout = build_runtime_layout(ProjectContext.standalone(engine_root=engine))
    assert layout.engine.read_only and layout.project.read_only
    assert layout.engine.source == engine.resolve() == layout.project.source
    assert {
        name: mount.source for name, mount in layout.writable_by_name.items()
    } == {
        name: (engine / ".synaptic" / name).resolve()
        for name in ("artifacts", "state", "tracking", "cache", "tmp")
    }
    assert all(not mount.read_only for mount in layout.writable)
    assert set(engine.rglob("*")) == before
    assert not (engine / ".synaptic").exists()


@pytest.mark.parametrize("source_role", ["engine", "project"])
def test_runtime_layout_rejects_exact_source_alias(tmp_path: Path, source_role: str) -> None:
    context = _context(tmp_path)
    aliased = getattr(context, f"{source_role}_root")
    with pytest.raises(SourceLockError, match="aliases|contains"):
        build_runtime_layout(replace(context, artifact_root=aliased))


def test_runtime_layout_rejects_writable_parent_of_sources(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(SourceLockError, match="contains"):
        build_runtime_layout(replace(context, artifact_root=tmp_path))


def test_runtime_layout_rejects_engine_descendant(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(SourceLockError, match="inside the engine"):
        build_runtime_layout(replace(context, artifact_root=context.engine_root / "outputs"))


def test_runtime_layout_rejects_undeclared_project_descendant(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(SourceLockError, match="approved project/.synaptic"):
        build_runtime_layout(replace(context, artifact_root=context.project_root / "other" / "outputs"))


def test_runtime_layout_rejects_misnamed_synaptic_carveout(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(SourceLockError, match="approved project/.synaptic"):
        build_runtime_layout(
            replace(context, artifact_root=context.project_root / ".synaptic" / "not-artifacts")
        )


def test_runtime_layout_rejects_writable_root_overlap(tmp_path: Path) -> None:
    context = _context(tmp_path)
    external = tmp_path / "runtime"
    context = replace(
        context,
        artifact_root=(external / "artifacts").resolve(),
        state_root=(external / "artifacts" / "state").resolve(),
        tracking_root=(external / "tracking").resolve(),
        cache_root=(external / "cache").resolve(),
        tmp_root=(external / "tmp").resolve(),
    )
    with pytest.raises(SourceLockError, match="Writable roots.*overlap"):
        build_runtime_layout(context)


def test_runtime_layout_allows_disjoint_external_writable_roots(tmp_path: Path) -> None:
    context = _context(tmp_path)
    external = tmp_path / "runtime"
    context = replace(
        context,
        artifact_root=(external / "artifacts").resolve(),
        state_root=(external / "state").resolve(),
        tracking_root=(external / "tracking").resolve(),
        cache_root=(external / "cache").resolve(),
        tmp_root=(external / "tmp").resolve(),
    )
    assert build_runtime_layout(context).writable_by_name["artifacts"].source == (
        external / "artifacts"
    ).resolve()


def test_runtime_layout_resolves_symlink_alias_to_source(tmp_path: Path) -> None:
    context = _context(tmp_path)
    alias = tmp_path / "engine-alias"
    try:
        os.symlink(context.engine_root, alias, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlink/junction creation unavailable: {exc}")
    with pytest.raises(SourceLockError, match="aliases|contains"):
        build_runtime_layout(replace(context, artifact_root=alias.absolute()))


def test_runtime_layout_resolves_symlink_descendant_into_engine(tmp_path: Path) -> None:
    context = _context(tmp_path)
    engine_output = context.engine_root / "generated"
    engine_output.mkdir()
    alias = tmp_path / "output-alias"
    try:
        os.symlink(engine_output, alias, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlink/junction creation unavailable: {exc}")
    with pytest.raises(SourceLockError, match="inside the engine"):
        build_runtime_layout(replace(context, artifact_root=alias.absolute()))


def test_runtime_layout_rejects_redirected_synaptic_component_as_junction_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuner.cloud.runtime_layout as layout_module

    context = _context(tmp_path)
    synaptic = context.project_root / ".synaptic"
    synaptic.mkdir()
    original = layout_module._is_redirect
    monkeypatch.setattr(
        layout_module,
        "_is_redirect",
        lambda path: path == synaptic or original(path),
    )
    with pytest.raises(SourceLockError, match="symlink or junction"):
        build_runtime_layout(context)


def test_standalone_runtime_rejects_redirected_child_without_creating_other_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuner.cloud.runtime_layout as layout_module

    engine = tmp_path / "engine"
    engine.mkdir()
    context = ProjectContext.standalone(engine_root=engine)
    artifacts = engine / ".synaptic" / "artifacts"
    artifacts.mkdir(parents=True)
    original = layout_module._is_redirect
    monkeypatch.setattr(
        layout_module,
        "_is_redirect",
        lambda path: path == artifacts or original(path),
    )
    before = set(engine.rglob("*"))
    with pytest.raises(SourceLockError, match="symlink or junction"):
        build_runtime_layout(context)
    assert set(engine.rglob("*")) == before
    assert not any(
        (engine / ".synaptic" / name).exists()
        for name in ("state", "tracking", "cache", "tmp")
    )
