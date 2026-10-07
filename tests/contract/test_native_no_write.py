"""Native execution detects source mutation and confines managed writes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tuner.cli.main import main
from tuner.handlers.local_run_handler import LocalRunHandler
from tuner.project.context import discover_project_context
from tuner.project.path_refs import resolve_path


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "host-project"


def test_builtin_native_dry_run_preserves_host_and_engine_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Built-in native planning is source-clean; full-authority plugins are excluded."""
    host = tmp_path / "native dry run host with spaces"
    shutil.copytree(FIXTURE_ROOT, host)
    engine = host / "dependencies" / "nonstandard engine location"
    trainer = engine / "Trainers" / "sft" / "train_sft.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# fixture trainer\n", encoding="utf-8")
    (engine / "shared").mkdir()
    (engine / "tuner").mkdir()

    config = host / "configs" / "native dry run.yaml"
    config.write_text(
        """name: native-contract
provider: local_docker
job:
  transfer: bind
  persist: true
run:
  method: sft
  dry_run: true
dataset:
  local_file: ../data/private dataset.jsonl
training:
  max_steps: 1
artifacts:
  run_timestamp: contract
""",
        encoding="utf-8",
    )
    dataset = (host / "data" / "private dataset.jsonl").resolve()
    dataset_before = dataset.read_bytes()
    before_host = _source_snapshot(host)
    before_engine = _source_snapshot(engine)

    captured: dict[str, object] = {}
    original_compile = LocalRunHandler._compile
    original_subprocess_run = subprocess.run

    def capture_compile(
        handler: LocalRunHandler, config_path: Path, payload: dict[str, object]
    ) -> dict[str, object]:
        plan = original_compile(handler, config_path, payload)
        captured["context"] = handler.context
        captured["plan"] = plan
        return plan

    def reject_execution(*_args: object, **_kwargs: object) -> None:
        pytest.fail("CLI dry-run attempted Docker execution")

    def reject_docker_subprocess(
        args: list[str], *subprocess_args: object, **subprocess_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if args and args[0] == "docker":
            pytest.fail(f"CLI dry-run attempted Docker subprocess: {args}")
        return original_subprocess_run(args, *subprocess_args, **subprocess_kwargs)

    monkeypatch.setenv("SYNAPTIC_ENGINE_ROOT", str(engine))
    monkeypatch.setattr(LocalRunHandler, "_compile", capture_compile)
    monkeypatch.setattr(LocalRunHandler, "_pull_image", reject_execution)
    monkeypatch.setattr(LocalRunHandler, "_execute_copy_mode", reject_execution)
    monkeypatch.setattr(LocalRunHandler, "_execute_bind_mode", reject_execution)
    monkeypatch.setattr(LocalRunHandler, "_execute_persistent_bind_mode", reject_execution)
    monkeypatch.setattr(
        "tuner.handlers.local_run_handler.subprocess.run", reject_docker_subprocess
    )

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "local-run",
                "--project-root",
                str(host),
                "--job-config",
                str(config),
                "--dry-run",
                "--json",
            ]
        )

    assert exit_info.value.code == 0
    response = json.loads(capsys.readouterr().out)
    assert response["success"] is True
    context = captured["context"]
    plan = captured["plan"]

    assert context.mode == "host"
    assert context.project_root == host.resolve()
    assert "--dry-run" in plan["command"]
    local_file = plan["command"][plan["command"].index("--local-file") + 1]
    assert local_file == "/workspace/project/data/private dataset.jsonl"

    writable_roots = {root.resolve() for root in context.writable_roots}
    writable_mounts = {
        Path(mount["host"]).resolve()
        for mount in plan["runtime_mounts"]
        if mount["mode"] == "rw"
    }
    assert writable_mounts == writable_roots
    assert all(_is_strictly_within(root, host / ".synaptic") for root in writable_mounts)
    assert _is_strictly_within(Path(plan["host_artifact_path"]), context.artifact_root)

    assert not _is_within(dataset, engine)
    assert all(not _is_within(dataset, root) for root in writable_roots)
    assert dataset.read_bytes() == dataset_before
    assert _source_snapshot(host) == before_host
    assert _source_snapshot(engine) == before_engine


def _source_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and ".synaptic" not in path.relative_to(root).parts
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _is_strictly_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    return resolved_path != resolved_root and _is_within(resolved_path, resolved_root)


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
