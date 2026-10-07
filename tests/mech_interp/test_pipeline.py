"""CPU tests for config-first MechInterp pipelines."""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from MechInterp.config import SteerCellConfig
from MechInterp.pipeline import (
    PipelineConfig,
    build_pipeline_plan,
    load_pipeline_config,
    select_stages,
    run_local_pipeline,
)
from tuner.project import ProjectContext


def _cell(path):
    data = {
        "surface": {"rows_path": "rows.jsonl"},
        "readouts": [{"name": "axis", "path": "direction.json"}],
        "law": {"kind": "additive", "readout": "axis"},
        "arms": [{"name": "baseline", "strength": 0.0}],
        "execution": {"output_path": "rows_out.jsonl", "render_fn": "render:prompt"},
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _pipeline(tmp_path):
    cell = tmp_path / "cell.yaml"
    _cell(cell)
    return PipelineConfig(
        name="smoke",
        model="Tiny/Model",
        model_revision="rev123",
        runtime={"python": "python", "pythonpath": ["plugins"]},
        stages=[
            {"name": "extract", "kind": "mechinterp.extract", "config": "extract.yaml"},
            {"name": "fit", "kind": "mechinterp.probe-fit", "config": "probe.yaml"},
            {"name": "steer", "kind": "mechinterp.steer", "config": str(cell)},
            {
                "name": "gates",
                "kind": "mechinterp.score-gates",
                "gates_config": "gates.yaml",
                "rows_path": "rows_out.jsonl",
            },
        ],
    )


def test_pipeline_config_loads(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "model": "Tiny/Model",
                "stages": [
                    {
                        "name": "say",
                        "kind": "command",
                        "command": ["python", "-c", "print('ok')"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_pipeline_config(path)
    assert cfg.name == "demo"
    assert cfg.stages[0].kind == "command"


def test_command_string_requires_shell_true():
    with pytest.raises(ValueError):
        PipelineConfig(
            name="bad",
            stages=[{"name": "say", "kind": "command", "command": "echo bad"}],
        )


def test_select_stages_supports_only_from_and_skip(tmp_path):
    cfg = _pipeline(tmp_path)
    assert [s.name for s in select_stages(cfg, only_step="fit")] == ["fit"]
    assert [s.name for s in select_stages(cfg, from_step="fit")] == ["fit", "steer", "gates"]
    assert [s.name for s in select_stages(cfg, skip_steps=["fit"])] == ["extract", "steer", "gates"]
    with pytest.raises(ValueError):
        select_stages(cfg, only_step="missing")


def test_plan_compiles_mechinterp_commands_and_config_render_fn(tmp_path):
    cfg = _pipeline(tmp_path)
    plan = build_pipeline_plan(
        cfg,
        repo_root=tmp_path,
        provider="local",
        gpu_ack=True,
        force=True,
    )
    commands = {stage["name"]: stage["command"] for stage in plan["stages"]}
    assert commands["extract"][-1] == "--i-know-this-runs-on-gpu"
    assert "--model-revision" in commands["extract"]
    assert "rev123" in commands["steer"]
    assert "--render-fn" in commands["steer"]
    assert "render:prompt" in commands["steer"]
    assert "--force-full-run" in commands["steer"]
    assert commands["gates"][-1] == "arm"


def test_steer_execution_render_fn_parses():
    cfg = SteerCellConfig(
        surface={"rows_path": "rows.jsonl"},
        readouts=[{"name": "axis", "path": "direction.json"}],
        law={"kind": "additive", "readout": "axis"},
        arms=[{"name": "baseline", "strength": 0.0}],
        execution={"output_path": "rows_out.jsonl", "render_fn": "render:prompt"},
    )
    assert cfg.execution.render_fn == "render:prompt"


def test_host_pipeline_splits_engine_executable_from_project_workdir(tmp_path):
    engine = Path(__file__).resolve().parents[2]
    project = tmp_path / "host"
    recipe = project / "experiments" / "pipeline.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "name": "host-dry-run",
                "runtime": {"workdir": "project://."},
                "stages": [
                    {
                        "name": "fit",
                        "kind": "mechinterp.probe-fit",
                        "config": "config://probe.yaml",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    context = ProjectContext.host(engine_root=engine, project_root=project)

    cfg = load_pipeline_config(recipe, context=context)
    plan = build_pipeline_plan(
        cfg,
        repo_root=engine,
        context=context,
        provider="local",
    )

    assert Path(cfg.runtime.workdir) == project
    assert Path(cfg.stages[0].config) == recipe.parent / "probe.yaml"
    assert Path(plan["stages"][0]["command"][1]) == engine / "tuner.py"
    assert Path(plan["stages"][0]["workdir"]) == project


def test_reference_host_dry_run_from_unrelated_cwd_is_source_clean(tmp_path):
    engine = Path(__file__).resolve().parents[2]
    project = tmp_path / "reference host"
    unrelated = tmp_path / "unrelated cwd"
    recipe = project / "experiments" / "pipeline.yaml"
    recipe.parent.mkdir(parents=True)
    unrelated.mkdir()
    shutil.copyfile(engine / "examples" / "host-project" / "synaptic.yaml", project / "synaptic.yaml")
    recipe.write_text(
        yaml.safe_dump(
            {
                "name": "reference-host",
                "runtime": {"workdir": "project://."},
                "artifacts": {
                    "checkpoint_paths": ["artifact://mechinterp/checkpoint.jsonl"]
                },
                "stages": [
                    {
                        "name": "inspect",
                        "kind": "command",
                        "command": [sys.executable, "-c", "print('not executed')"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    before = {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(engine / "tuner.py"),
            "mechinterp",
            "run",
            "--config",
            str(recipe),
            "--project-root",
            str(project),
            "--dry-run",
            "--json",
        ],
        cwd=unrelated,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert Path(payload["data"]["runtime"]["workdir"]) == project
    after = {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (project / ".synaptic").exists()


def test_local_pipeline_propagates_one_host_context_to_every_stage(tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    project = tmp_path / "host"
    engine.mkdir()
    project.mkdir()
    context = ProjectContext.host(engine_root=engine, project_root=project)
    cfg = PipelineConfig(
        name="context-propagation",
        runtime={"workdir": str(project)},
        stages=[
            {"name": "one", "kind": "command", "command": ["one"]},
            {"name": "two", "kind": "command", "command": ["two"]},
        ],
    )
    calls = []

    class Result:
        returncode = 0

    def fake_run(cmd, *, cwd, env, shell=False):
        calls.append((cmd, Path(cwd), dict(env), shell))
        return Result()

    monkeypatch.setattr("MechInterp.pipeline.subprocess.run", fake_run)

    assert run_local_pipeline(cfg, repo_root=engine, context=context) == 0
    assert len(calls) == 2
    assert {call[1] for call in calls} == {project}
    assert {
        (call[2]["SYNAPTIC_ENGINE_ROOT"], call[2]["SYNAPTIC_PROJECT_ROOT"])
        for call in calls
    } == {(str(engine), str(project))}
