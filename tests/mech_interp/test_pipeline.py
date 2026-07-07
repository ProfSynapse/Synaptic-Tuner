"""CPU tests for config-first MechInterp pipelines."""

from __future__ import annotations

import yaml
import pytest

from MechInterp.config import SteerCellConfig
from MechInterp.pipeline import (
    PipelineConfig,
    build_pipeline_plan,
    load_pipeline_config,
    select_stages,
)


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
