from pathlib import Path

import pytest

from tuner.project.context import ProjectContext
from tuner.project.errors import ExternalPathError, PathEscapeError, WriteAccessError
from tuner.project.path_refs import PathRef


@pytest.fixture
def context(tmp_path: Path) -> ProjectContext:
    host = tmp_path / "host"
    engine = host / "vendor" / "engine"
    engine.mkdir(parents=True)
    return ProjectContext.host(engine_root=engine, project_root=host, invocation_cwd=tmp_path)


def test_relative_config_path_uses_declaring_document(context: ProjectContext) -> None:
    document = context.project_root / "experiments" / "nested" / "job.yaml"
    assert PathRef.parse("../data/train.jsonl").resolve(
        context, declaring_file=document
    ) == context.project_root / "experiments" / "data" / "train.jsonl"


def test_included_documents_keep_their_own_declaring_directory(context: ProjectContext) -> None:
    one = context.project_root / "a" / "config.yaml"
    two = context.project_root / "b" / "config.yaml"
    assert PathRef.parse("rows.jsonl").resolve(context, declaring_file=one) != PathRef.parse(
        "rows.jsonl"
    ).resolve(context, declaring_file=two)


def test_uri_escape_is_rejected(context: ProjectContext) -> None:
    with pytest.raises(PathEscapeError):
        PathRef.parse("artifact://../host-source.txt").resolve(context)


def test_outputs_are_limited_to_writable_roots(context: ProjectContext) -> None:
    assert PathRef.parse("artifact://runs/one").resolve(
        context, access="write"
    ) == context.artifact_root / "runs" / "one"
    with pytest.raises(WriteAccessError):
        PathRef.parse("project://experiments/result.json").resolve(context, access="write")
    with pytest.raises(WriteAccessError):
        PathRef.parse("engine://result.json").resolve(context, access="write")


def test_external_paths_fail_closed_in_cloud(context: ProjectContext, tmp_path: Path) -> None:
    external = (tmp_path / "input.jsonl").resolve().as_uri()
    assert PathRef.parse(external).resolve(context) == tmp_path / "input.jsonl"
    with pytest.raises(ExternalPathError):
        PathRef.parse(external).resolve(context, cloud=True)


def test_deny_policy_rejects_local_file_and_bare_absolute_paths(
    context: ProjectContext, tmp_path: Path
) -> None:
    absolute = (tmp_path / "external.jsonl").resolve()
    with pytest.raises(ExternalPathError):
        PathRef.parse(absolute.as_uri()).resolve(context, external_paths="deny")
    with pytest.raises(ExternalPathError):
        PathRef.parse(str(absolute)).resolve(context, external_paths="deny")


def test_allow_policy_authorizes_external_reads_but_never_writes(
    context: ProjectContext, tmp_path: Path
) -> None:
    absolute = (tmp_path / "external.jsonl").resolve()
    assert PathRef.parse(str(absolute)).resolve(
        context, external_paths="allow"
    ) == absolute
    with pytest.raises(WriteAccessError):
        PathRef.parse(str(absolute)).resolve(
            context, external_paths="allow", access="write"
        )
    with pytest.raises(ExternalPathError):
        PathRef.parse(str(absolute)).resolve(
            context, external_paths="allow", cloud=True
        )


def test_windows_drive_is_not_parsed_as_uri_scheme() -> None:
    reference = PathRef.parse(r"C:\research\experiment.yaml")
    assert reference.scheme is None


def test_symlink_escape_is_rejected(context: ProjectContext, tmp_path: Path) -> None:
    link = context.artifact_root / "outside"
    link.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are unavailable in this test environment")
    with pytest.raises(PathEscapeError):
        PathRef.parse("artifact://outside/result.json").resolve(context, access="write")
