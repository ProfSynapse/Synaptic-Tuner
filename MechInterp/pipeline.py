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

from tuner.project import ProjectContext, resolve_path


StageKind = Literal[
    "command",
    "mechinterp.extract",
    "mechinterp.probe-fit",
    "mechinterp.steer",
    "mechinterp.dose-calibrate",
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
    model_revision: Optional[str] = None
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
        if self.kind in {
            "mechinterp.extract",
            "mechinterp.probe-fit",
            "mechinterp.steer",
            "mechinterp.dose-calibrate",
        } and not self.config:
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
    model_revision: Optional[str] = None
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


def load_pipeline_config(
    path: str | Path, *, context: ProjectContext | None = None
) -> PipelineConfig:
    declaring_file = Path(path).resolve()
    with open(declaring_file, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"MechInterp pipeline config must be a YAML object: {path}")
    cfg = PipelineConfig(**data)
    if context is None:
        return cfg

    def resolved(value: str, *, access: str = "read") -> str:
        return str(
            resolve_path(
                value,
                context,
                declaring_file=declaring_file,
                access=access,
            )
        )

    runtime = cfg.runtime.model_copy(
        update={
            "workdir": resolved(cfg.runtime.workdir),
            "pythonpath": [resolved(item) for item in cfg.runtime.pythonpath],
        }
    )
    stages = []
    for stage in cfg.stages:
        updates: dict[str, object] = {}
        for field_name in ("config", "gates_config", "rows_path", "workdir"):
            value = getattr(stage, field_name)
            if value:
                updates[field_name] = resolved(value)
        stages.append(stage.model_copy(update=updates))
    artifacts = cfg.artifacts.model_copy(
        update={
            "checkpoint_paths": [
                resolved(item, access="write") for item in cfg.artifacts.checkpoint_paths
            ]
        }
    )
    return cfg.model_copy(
        update={"runtime": runtime, "stages": stages, "artifacts": artifacts}
    )


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


def _stage_model_revision(cfg: PipelineConfig, stage: StageConfig) -> str | None:
    return stage.model_revision if stage.model_revision is not None else cfg.model_revision


def _stage_adapter(cfg: PipelineConfig, stage: StageConfig) -> str | None:
    return stage.adapter if stage.adapter is not None else cfg.adapter


def _repo_path(repo_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _execution_render_fn(
    stage: StageConfig,
    repo_root: Path,
    context: ProjectContext | None = None,
) -> str | None:
    if stage.render_fn:
        return stage.render_fn
    if not stage.config:
        return None
    try:
        from MechInterp.config import (
            load_dose_calibration_config,
            load_steer_config,
        )

        if stage.kind == "mechinterp.dose-calibrate":
            return load_dose_calibration_config(
                _repo_path(repo_root, stage.config), context=context
            ).execution.render_fn
        return load_steer_config(
            _repo_path(repo_root, stage.config), context=context
        ).execution.render_fn
    except Exception:
        if context is not None:
            raise
        return None


def compile_stage_command(
    cfg: PipelineConfig,
    stage: StageConfig,
    *,
    repo_root: Path,
    context: ProjectContext | None = None,
    gpu_ack: bool = False,
    force: bool = False,
) -> list[str] | str:
    python = cfg.runtime.python or sys.executable
    engine_root = context.engine_root if context is not None else repo_root
    tuner = str(engine_root / "tuner.py")
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
        revision = _stage_model_revision(cfg, stage)
        if revision:
            cmd.extend(["--model-revision", revision])
        adapter = _stage_adapter(cfg, stage)
        if adapter:
            cmd.extend(["--adapter", adapter])
        if gpu_ack:
            cmd.append("--i-know-this-runs-on-gpu")
        return cmd

    if stage.kind == "mechinterp.probe-fit":
        cmd = [
            python,
            tuner,
            "mechinterp",
            "probe-fit",
            "--mi-config",
            str(stage.config),
        ]
        return cmd

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
        revision = _stage_model_revision(cfg, stage)
        if revision:
            cmd.extend(["--model-revision", revision])
        adapter = _stage_adapter(cfg, stage)
        render_fn = _execution_render_fn(stage, repo_root, context)
        if adapter:
            cmd.extend(["--adapter", adapter])
        if render_fn:
            cmd.extend(["--render-fn", render_fn])
        if gpu_ack:
            cmd.append("--i-know-this-runs-on-gpu")
        if force:
            cmd.append("--force-full-run")
        return cmd

    if stage.kind == "mechinterp.dose-calibrate":
        cmd = [
            python,
            tuner,
            "mechinterp",
            "dose-calibrate",
            "--mi-config",
            str(stage.config),
            "--model",
            _stage_model(cfg, stage),
        ]
        adapter = _stage_adapter(cfg, stage)
        render_fn = _execution_render_fn(stage, repo_root, context)
        if adapter:
            cmd.extend(["--adapter", adapter])
        if render_fn:
            cmd.extend(["--render-fn", render_fn])
        if gpu_ack:
            cmd.append("--i-know-this-runs-on-gpu")
        return cmd

    if stage.kind == "mechinterp.score-gates":
        cmd = [
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
        return cmd

    raise ValueError(f"unsupported stage kind: {stage.kind}")


def build_pipeline_plan(
    cfg: PipelineConfig,
    *,
    repo_root: Path,
    context: ProjectContext | None = None,
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
        "model_revision": cfg.model_revision,
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
                    context=context,
                    gpu_ack=gpu_ack,
                    force=force,
                ),
                "shell": stage.shell,
            }
            for stage in stages
        ],
    }


def _env_for_stage(
    cfg: PipelineConfig,
    stage: StageConfig,
    repo_root: Path,
    context: ProjectContext | None = None,
) -> dict[str, str]:
    env = {**os.environ, **cfg.runtime.env, **stage.env}
    paths = [str(repo_root / p) if not Path(p).is_absolute() else p for p in cfg.runtime.pythonpath]
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if context is not None:
        env["SYNAPTIC_ENGINE_ROOT"] = str(context.engine_root)
        if context.mode == "host":
            env["SYNAPTIC_PROJECT_ROOT"] = str(context.project_root)
    return env


def run_local_pipeline(
    cfg: PipelineConfig,
    *,
    repo_root: Path,
    context: ProjectContext | None = None,
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
            cfg,
            stage,
            repo_root=repo_root,
            context=context,
            gpu_ack=gpu_ack,
            force=force,
        )
        workdir = Path(stage.workdir or cfg.runtime.workdir)
        project_root = context.project_root if context is not None else repo_root
        cwd = workdir if workdir.is_absolute() else project_root / workdir
        env = _env_for_stage(cfg, stage, repo_root, context)
        print(f"[mechinterp-run] stage={stage.name} kind={stage.kind}")
        if isinstance(cmd, list):
            result = subprocess.run(cmd, cwd=cwd, env=env)
        else:
            result = subprocess.run(cmd, cwd=cwd, env=env, shell=stage.shell)
        if result.returncode != 0:
            return result.returncode
    return 0
