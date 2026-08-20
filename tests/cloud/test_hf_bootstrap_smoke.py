from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tuner.cloud import hf_bootstrap_smoke as smoke


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT_ENV = {
    "GIT_AUTHOR_NAME": "Synaptic Smoke",
    "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
    "GIT_COMMITTER_NAME": "Synaptic Smoke",
    "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
    "GIT_AUTHOR_DATE": "2026-08-20T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-08-20T00:00:00+00:00",
}


def _git(repository: Path, *arguments: str) -> str:
    environment = dict(os.environ)
    environment.update(COMMIT_ENV)
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repository(root: Path) -> tuple[Path, str]:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "README.md").write_text("fixed bootstrap smoke\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-q", "-m", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def _freeze(root: Path) -> None:
    for directory, _directories, files in os.walk(root):
        for name in files:
            os.chmod(Path(directory) / name, 0o444)
        os.chmod(directory, 0o555)


def _thaw(root: Path) -> None:
    if not root.exists():
        return
    for directory, directories, files in os.walk(root):
        os.chmod(directory, 0o755)
        for name in directories:
            os.chmod(Path(directory) / name, 0o755)
        for name in files:
            os.chmod(Path(directory) / name, 0o644)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _layout(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Path], str]:
    checkout = tmp_path / "checkout"
    physical, commit = _repository(checkout / "project")
    bootstrap = {
        "schema_version": "synaptic-bootstrap-result/v1",
        "project_root": str(physical),
        "engine_root": str(physical),
        "project_commit": commit,
        "engine_commit": commit,
    }
    result_path = checkout / ".synaptic-bootstrap-result.json"
    result_path.write_bytes(_canonical(bootstrap))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if os.name == "nt":
        project = engine = physical
    else:
        project = workspace / "project"
        engine = workspace / "engine"
        os.symlink(os.path.relpath(physical, project.parent), project, target_is_directory=True)
        os.symlink(os.path.relpath(physical, engine.parent), engine, target_is_directory=True)
    writable = {name: workspace / name for name in ("artifacts", "state", "tracking", "cache", "tmp")}
    for root in writable.values():
        root.mkdir()
    _freeze(physical)
    return result_path, project, engine, writable, commit


def _run(tmp_path: Path) -> dict[str, object]:
    result_path, project, engine, writable, _commit = _layout(tmp_path)
    try:
        return smoke.run_bootstrap_smoke(
            bootstrap_result_path=result_path,
            workspace_root=next(iter(writable.values())).parent,
            project_root=project,
            engine_root=engine,
            writable_roots=writable,
        )
    finally:
        _thaw(result_path.parent / "project")


def test_fixed_workload_is_canonical_bounded_and_non_extensible() -> None:
    document = json.loads(smoke.canonical_workload_bytes())
    assert document == {
        "schema_version": "synaptic-hf-bootstrap-smoke-workload/v1",
        "kind": "bootstrap_verification",
        "runtime": {"image": "python:3.12"},
        "hardware": {"flavor": "cpu-basic"},
        "limits": {
            "provider_timeout_seconds": 600,
            "cancel_after_seconds": 720,
            "outer_observation_seconds": 900,
            "projected_compute_usd": "0.01",
            "hard_total_usd": "2.00",
        },
        "network": {"ports": [], "ssh": False},
        "retries": 0,
        "effects": {"training": False, "publication": False},
    }
    assert len(smoke.canonical_workload_bytes()) < 4096
    assert smoke.WORKLOAD_SHA256 == "0d1d3454d079ea994a1e3a24b59b772bd4adb40cb441e00cc5801faf5d220841"
    assert smoke.workload_sha256() == hashlib.sha256(smoke.canonical_workload_bytes()).hexdigest()
    with pytest.raises(TypeError):
        smoke.WORKLOAD["kind"] = "training"  # type: ignore[index]
    with pytest.raises(TypeError):
        smoke.WORKLOAD["effects"]["training"] = True  # type: ignore[index]
    assert smoke.main(["--config", "anything.json"]) == 2


def test_remote_entrypoint_imports_only_the_standard_library() -> None:
    source = (REPO_ROOT / "tuner/cloud/hf_bootstrap_smoke.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )
    assert imports <= {
        "hashlib", "json", "os", "re", "stat", "subprocess", "sys", "tempfile", "pathlib",
        "types", "typing",
    }
    assert not ({"huggingface_hub", "torch", "transformers", "tuner"} & imports)


def test_smoke_verifies_bootstrap_sources_and_writable_layout(tmp_path: Path) -> None:
    result = _run(tmp_path)
    raw = smoke.canonical_result_bytes(result)
    document = json.loads(raw)
    assert document["success"] is True
    assert document["workload"] == {
        "kind": "bootstrap_verification",
        "sha256": smoke.WORKLOAD_SHA256,
    }
    assert document["sources"]["project"]["identity"] == "project://"
    assert document["sources"]["engine"]["identity"] == "engine://"
    assert document["writable_roots"] == ["artifacts", "state", "tracking", "cache", "tmp"]
    digest = document.pop("result_sha256")
    assert digest == hashlib.sha256(_canonical(document)).hexdigest()
    rendered = raw.decode("ascii").lower()
    assert all(word not in rendered for word in ("token", "url", "raw_log", "stdout", "stderr"))
    assert str(tmp_path).lower().replace("\\", "/") not in rendered.replace("\\", "/")


def test_smoke_result_is_deterministic_across_clean_roots(tmp_path: Path) -> None:
    first = _run(tmp_path / "one")
    second = _run(tmp_path / "two")
    assert smoke.canonical_result_bytes(first) == smoke.canonical_result_bytes(second)


def test_smoke_rejects_commit_drift(tmp_path: Path) -> None:
    result_path, project, engine, writable, _commit = _layout(tmp_path)
    document = json.loads(result_path.read_text(encoding="ascii"))
    document["project_commit"] = "a" * 40
    result_path.write_bytes(_canonical(document))
    try:
        with pytest.raises(smoke.BootstrapSmokeError, match="commit identity does not match"):
            smoke.run_bootstrap_smoke(
                bootstrap_result_path=result_path,
                workspace_root=next(iter(writable.values())).parent,
                project_root=project,
                engine_root=engine,
                writable_roots=writable,
            )
    finally:
        _thaw(result_path.parent / "project")


def test_smoke_rejects_writable_source_member(tmp_path: Path) -> None:
    result_path, project, engine, writable, _commit = _layout(tmp_path)
    source = result_path.parent / "project"
    os.chmod(source / "README.md", 0o644)
    try:
        with pytest.raises(smoke.BootstrapSmokeError, match="remains writable"):
            smoke.run_bootstrap_smoke(
                bootstrap_result_path=result_path,
                workspace_root=next(iter(writable.values())).parent,
                project_root=project,
                engine_root=engine,
                writable_roots=writable,
            )
    finally:
        _thaw(source)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_smoke_rejects_source_symlink_to_external_writable_tree(tmp_path: Path) -> None:
    result_path, project, engine, writable, _commit = _layout(tmp_path)
    source = result_path.parent / "project"
    external = tmp_path / "external-writable"
    external.mkdir()
    (external / "mutable.txt").write_text("must not be reached\n", encoding="utf-8")
    _thaw(source)
    link = source / "external-link"
    os.symlink(external, link, target_is_directory=True)
    _freeze(source)
    try:
        with pytest.raises(smoke.BootstrapSmokeError, match="links or reparse"):
            smoke.run_bootstrap_smoke(
                bootstrap_result_path=result_path,
                workspace_root=next(iter(writable.values())).parent,
                project_root=project,
                engine_root=engine,
                writable_roots=writable,
            )
        assert (external / "mutable.txt").read_text(encoding="utf-8") == "must not be reached\n"
        assert external.stat().st_mode & 0o222
    finally:
        _thaw(source)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction/reparse semantics")
def test_smoke_rejects_real_windows_junction_to_external_writable_tree(tmp_path: Path) -> None:
    result_path, project, engine, writable, _commit = _layout(tmp_path)
    source = result_path.parent / "project"
    external = tmp_path / "external-writable"
    external.mkdir()
    (external / "mutable.txt").write_text("must not be reached\n", encoding="utf-8")
    _thaw(source)
    _freeze(source)
    junction = source / "external-junction"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        _thaw(source)
        pytest.skip("Windows junction creation is unavailable in this environment")
    try:
        with pytest.raises(smoke.BootstrapSmokeError, match="links or reparse"):
            smoke.run_bootstrap_smoke(
                bootstrap_result_path=result_path,
                workspace_root=next(iter(writable.values())).parent,
                project_root=project,
                engine_root=engine,
                writable_roots=writable,
            )
        assert (external / "mutable.txt").read_text(encoding="utf-8") == "must not be reached\n"
    finally:
        if os.path.lexists(junction):
            os.rmdir(junction)
        _thaw(source)


@pytest.mark.skipif(os.name == "nt", reason="POSIX broken-symlink semantics")
def test_source_permission_walk_never_follows_link_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    os.symlink(tmp_path / "absent-target", source / "broken-link")
    os.chmod(source, 0o555)
    calls: list[bool] = []
    original_walk = smoke.os.walk

    def recording_walk(*args: object, **kwargs: object):
        calls.append(kwargs.get("followlinks") is False)
        yield from original_walk(*args, **kwargs)

    monkeypatch.setattr(smoke.os, "walk", recording_walk)
    try:
        with pytest.raises(smoke.BootstrapSmokeError, match="links or reparse"):
            smoke._verify_source_read_only(source)
        assert calls == [True]
    finally:
        os.chmod(source, 0o755)


def test_smoke_rejects_writable_root_inside_source(tmp_path: Path) -> None:
    result_path, project, engine, writable, _commit = _layout(tmp_path)
    source = result_path.parent / "project"
    try:
        writable["artifacts"] = source
        with pytest.raises(smoke.BootstrapSmokeError, match="canonical layout|overlaps source"):
            smoke.run_bootstrap_smoke(
                bootstrap_result_path=result_path,
                workspace_root=next(iter(writable.values())).parent,
                project_root=project,
                engine_root=engine,
                writable_roots=writable,
            )
    finally:
        _thaw(source)


def test_canonical_result_rejects_extension_and_digest_tampering(tmp_path: Path) -> None:
    result = _run(tmp_path)
    extended = dict(result)
    extended["raw_logs"] = "not allowed"
    with pytest.raises(smoke.BootstrapSmokeError, match="shape"):
        smoke.canonical_result_bytes(extended)
    tampered = dict(result)
    tampered["result_sha256"] = "0" * 64
    with pytest.raises(smoke.BootstrapSmokeError, match="digest"):
        smoke.canonical_result_bytes(tampered)


def test_smoke_module_has_no_arbitrary_execution_or_training_surface() -> None:
    public = set(smoke.__all__)
    assert not public & {"command", "run_command", "train", "publish", "submit", "configure"}
    source = (REPO_ROOT / "tuner/cloud/hf_bootstrap_smoke.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "eval(" not in source
    assert "exec(" not in source
    assert "run_job" not in source
    assert "huggingface_hub" not in source
