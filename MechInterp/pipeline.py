"""Config-first MechInterp pipeline runner.

The pipeline layer keeps experiment orchestration declarative: a YAML file names
the stages and their recipe files, while CLI flags provide only late-bound
runtime facts such as provider, exact commit, and operator approval.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


StageKind = Literal[
    "command",
    "mechinterp.extract",
    "mechinterp.probe-fit",
    "mechinterp.steer",
    "mechinterp.score-gates",
]


class RuntimeConfig(BaseModel):
    provider: Literal["local", "modal"] = "local"
    python: str = Field(default_factory=lambda: sys.executable)
    workdir: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    pythonpath: list[str] = Field(default_factory=list)


class RepoConfig(BaseModel):
    url: Optional[str] = None
    branch: Optional[str] = None
    commit: Optional[str] = None


class ModalConfig(BaseModel):
    app_name: Optional[str] = None
    image: str = "vllm/vllm-openai:latest"
    gpu: str = "A10G"
    timeout_hours: float = 3.0
    checkpoint_interval_sec: int = 120
    volume_name: str = "mechinterp-pipeline"
    mount_path: str = "/vol/mechinterp"
    pip: list[str] = Field(default_factory=lambda: ["pyyaml", "pydantic"])
    apt: list[str] = Field(default_factory=lambda: ["git"])


class ArtifactConfig(BaseModel):
    checkpoint_paths: list[str] = Field(default_factory=list)


class StageConfig(BaseModel):
    name: str = Field(..., min_length=1)
    kind: StageKind
    config: Optional[str] = None
    model: Optional[str] = None
    adapter: Optional[str] = None
    render_fn: Optional[str] = None
    gates_config: Optional[str] = None
    rows_path: Optional[str] = None
    arm_field: str = "arm"
    command: Optional[str | list[str]] = None
    shell: bool = False
    workdir: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _required_fields(self) -> "StageConfig":
        if self.kind == "command" and self.command is None:
            raise ValueError(f"stage {self.name}: command is required")
        if (
            self.kind == "command"
            and isinstance(self.command, str)
            and not self.shell
        ):
            raise ValueError(
                f"stage {self.name}: string commands require shell: true; "
                "use a list argv for shell-free execution"
            )
        if self.kind in {"mechinterp.extract", "mechinterp.probe-fit", "mechinterp.steer"} and not self.config:
            raise ValueError(f"stage {self.name}: config is required")
        if self.kind == "mechinterp.score-gates" and (
            not self.gates_config or not self.rows_path
        ):
            raise ValueError(
                f"stage {self.name}: gates_config and rows_path are required"
            )
        return self


class PipelineConfig(BaseModel):
    schema_version: str = "mechinterp-pipeline/v1"
    name: str = Field(..., min_length=1)
    model: Optional[str] = None
    adapter: Optional[str] = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    repo: RepoConfig = Field(default_factory=RepoConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    stages: list[StageConfig] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_stage_names(self) -> "PipelineConfig":
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("stage names must be unique")
        return self


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"MechInterp pipeline config must be a YAML object: {path}")
    return PipelineConfig(**data)


def select_stages(
    cfg: PipelineConfig,
    *,
    only_step: str | None = None,
    from_step: str | None = None,
    skip_steps: list[str] | None = None,
) -> list[StageConfig]:
    skip = set(skip_steps or [])
    stages = list(cfg.stages)
    names = [stage.name for stage in stages]
    for requested in [only_step, from_step, *skip]:
        if requested and requested not in names:
            raise ValueError(f"unknown pipeline stage: {requested}")
    if only_step:
        stages = [stage for stage in stages if stage.name == only_step]
    elif from_step:
        start = names.index(from_step)
        stages = stages[start:]
    if skip:
        stages = [stage for stage in stages if stage.name not in skip]
    return stages


def _stage_model(cfg: PipelineConfig, stage: StageConfig) -> str:
    model = stage.model or cfg.model
    if not model:
        raise ValueError(f"stage {stage.name}: model is required")
    return model


def _stage_adapter(cfg: PipelineConfig, stage: StageConfig) -> str | None:
    return stage.adapter if stage.adapter is not None else cfg.adapter


def _repo_path(repo_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _steer_render_fn(stage: StageConfig, repo_root: Path) -> str | None:
    if stage.render_fn:
        return stage.render_fn
    if not stage.config:
        return None
    try:
        from MechInterp.config import load_steer_config

        return load_steer_config(_repo_path(repo_root, stage.config)).execution.render_fn
    except Exception:
        return None


def compile_stage_command(
    cfg: PipelineConfig,
    stage: StageConfig,
    *,
    repo_root: Path,
    gpu_ack: bool = False,
    force: bool = False,
) -> list[str] | str:
    python = cfg.runtime.python or sys.executable
    tuner = str(repo_root / "tuner.py")
    if stage.kind == "command":
        if isinstance(stage.command, list):
            return [str(item) for item in stage.command]
        return str(stage.command)

    if stage.kind == "mechinterp.extract":
        cmd = [
            python,
            tuner,
            "mechinterp",
            "extract",
            "--mi-config",
            str(stage.config),
            "--model",
            _stage_model(cfg, stage),
        ]
        adapter = _stage_adapter(cfg, stage)
        if adapter:
            cmd.extend(["--adapter", adapter])
        if gpu_ack:
            cmd.append("--i-know-this-runs-on-gpu")
        return cmd

    if stage.kind == "mechinterp.probe-fit":
        return [
            python,
            tuner,
            "mechinterp",
            "probe-fit",
            "--mi-config",
            str(stage.config),
        ]

    if stage.kind == "mechinterp.steer":
        cmd = [
            python,
            tuner,
            "mechinterp",
            "steer",
            "--mi-config",
            str(stage.config),
            "--model",
            _stage_model(cfg, stage),
        ]
        adapter = _stage_adapter(cfg, stage)
        render_fn = _steer_render_fn(stage, repo_root)
        if adapter:
            cmd.extend(["--adapter", adapter])
        if render_fn:
            cmd.extend(["--render-fn", render_fn])
        if gpu_ack:
            cmd.append("--i-know-this-runs-on-gpu")
        if force:
            cmd.append("--force-full-run")
        return cmd

    if stage.kind == "mechinterp.score-gates":
        return [
            python,
            tuner,
            "mechinterp",
            "score-gates",
            "--gates-config",
            str(stage.gates_config),
            "--rows-path",
            str(stage.rows_path),
            "--arm-field",
            stage.arm_field,
        ]

    raise ValueError(f"unsupported stage kind: {stage.kind}")


def build_pipeline_plan(
    cfg: PipelineConfig,
    *,
    repo_root: Path,
    provider: str,
    only_step: str | None = None,
    from_step: str | None = None,
    skip_steps: list[str] | None = None,
    gpu_ack: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    stages = select_stages(
        cfg, only_step=only_step, from_step=from_step, skip_steps=skip_steps
    )
    return {
        "name": cfg.name,
        "schema_version": cfg.schema_version,
        "provider": provider,
        "model": cfg.model,
        "adapter": cfg.adapter,
        "runtime": cfg.runtime.model_dump(),
        "modal": cfg.modal.model_dump() if provider == "modal" else None,
        "artifacts": cfg.artifacts.model_dump(),
        "stages": [
            {
                "name": stage.name,
                "kind": stage.kind,
                "workdir": stage.workdir or cfg.runtime.workdir,
                "command": compile_stage_command(
                    cfg,
                    stage,
                    repo_root=repo_root,
                    gpu_ack=gpu_ack,
                    force=force,
                ),
                "shell": stage.shell,
            }
            for stage in stages
        ],
    }


def _env_for_stage(cfg: PipelineConfig, stage: StageConfig, repo_root: Path) -> dict[str, str]:
    env = {**os.environ, **cfg.runtime.env, **stage.env}
    paths = [str(repo_root / p) if not Path(p).is_absolute() else p for p in cfg.runtime.pythonpath]
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def run_local_pipeline(
    cfg: PipelineConfig,
    *,
    repo_root: Path,
    only_step: str | None = None,
    from_step: str | None = None,
    skip_steps: list[str] | None = None,
    gpu_ack: bool = False,
    force: bool = False,
) -> int:
    for stage in select_stages(
        cfg, only_step=only_step, from_step=from_step, skip_steps=skip_steps
    ):
        cmd = compile_stage_command(
            cfg, stage, repo_root=repo_root, gpu_ack=gpu_ack, force=force
        )
        cwd = repo_root / (stage.workdir or cfg.runtime.workdir)
        env = _env_for_stage(cfg, stage, repo_root)
        print(f"[mechinterp-run] stage={stage.name} kind={stage.kind}")
        if isinstance(cmd, list):
            result = subprocess.run(cmd, cwd=cwd, env=env)
        else:
            result = subprocess.run(cmd, cwd=cwd, env=env, shell=stage.shell)
        if result.returncode != 0:
            return result.returncode
    return 0
