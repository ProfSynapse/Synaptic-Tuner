from pathlib import Path

import pytest

from tuner.project.context import ProjectContext, discover_project_context
from tuner.project.errors import ProjectRootAmbiguousError


def test_host_defaults_keep_mutable_roots_in_host_project(tmp_path: Path) -> None:
    project = tmp_path / "host"
    engine = project / "vendor" / "synaptic-tuner"
    engine.mkdir(parents=True)
    context = ProjectContext.host(engine_root=engine, project_root=project)

    assert context.mode == "host"
    assert context.path_mode == "project_v1"
    assert context.artifact_root == project / ".synaptic" / "artifacts"
    assert context.tracking_root == project / ".synaptic" / "tracking"
    assert all(not root.is_relative_to(engine) for root in context.writable_roots)


def test_explicit_project_root_precedes_environment(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"
    context = discover_project_context(
        engine_root=tmp_path / "engine",
        explicit_project_root=explicit,
        environment={"SYNAPTIC_PROJECT_ROOT": str(environment)},
    )
    assert context.project_root == explicit.resolve()


def test_config_and_cwd_manifest_disagreement_is_fatal(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "synaptic.yaml").write_text("schema_version: synaptic-project/v1\n")
    (right / "synaptic.yaml").write_text("schema_version: synaptic-project/v1\n")
    config = left / "experiment.yaml"
    config.touch()

    with pytest.raises(ProjectRootAmbiguousError) as error:
        discover_project_context(
            engine_root=tmp_path / "engine",
            primary_config=config,
            invocation_cwd=right,
            environment={},
            superproject_root=None,
        )
    assert error.value.code == "PROJECT_ROOT_AMBIGUOUS"


def test_standalone_fallback_preserves_legacy_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tuner.project.context.find_git_superproject", lambda _: None)
    context = discover_project_context(
        engine_root=tmp_path / "engine",
        invocation_cwd=tmp_path,
        environment={},
    )
    assert context.project_root == (tmp_path / "engine").resolve()
    assert context.mode == "standalone"
    assert context.path_mode == "legacy"
