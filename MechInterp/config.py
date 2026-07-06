"""
Pydantic v2 configuration models for MechInterp cells.

Loaded from YAML and validated at parse time. The steer cell config is the
six-block declarative model:

  surface   where the rows come from, the generation contract, the seed, and an
            optional expected-config sha for reproducibility pinning.
  readouts  the frozen direction files the cell reads and writes along.
  law       the intervention law and its shared parameters.
  arms      named strength overrides, including the baseline no-op, a seeded
            permuted control, and dose-ladder arms.
  execution lane-agnostic run controls: output path, resume behavior, grader.
  smoke     a small pre-run with readback tolerances; the full arms refuse to run
            until the smoke has recorded a pass, unless explicitly overridden.

The extraction and probe-fit configs are lighter; each drives one verb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------


class GenerationContract(BaseModel):
    """How completions are produced. Greedy by default for reproducibility."""

    max_new_tokens: int = 96
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0


class ReadoutRef(BaseModel):
    """A frozen direction file the cell reads or writes along."""

    name: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1, description="Path to a frozen direction JSON")


# --------------------------------------------------------------------------
# Steer cell
# --------------------------------------------------------------------------


class SurfaceConfig(BaseModel):
    """Where rows come from and the reproducibility contract."""

    rows_path: str = Field(..., min_length=1, description="JSONL of input rows")
    generation: GenerationContract = Field(default_factory=GenerationContract)
    seed: int = 0
    expected_config_sha: Optional[str] = Field(
        default=None,
        description="If set, the run aborts unless the computed config sha matches.",
    )


class LawConfig(BaseModel):
    """The intervention law and its shared parameters."""

    kind: Literal["additive", "erase_write"] = "additive"
    readout: str = Field(..., description="Name of the readout to intervene along")
    layer: Optional[int] = Field(
        default=None,
        description="Override the layer; defaults to the readout's frozen layer.",
    )
    position: Literal["anchor", "anchor_onward", "final", "answer_window"] = "anchor"
    generation_mode: Literal["anchor", "gen_stream"] = "anchor"


class ArmConfig(BaseModel):
    """A named arm: which rows are active and at what strength.

    selection choices (exactly one of):
      strength      apply a fixed strength to every row (baseline uses 0).
      score_field + threshold + strength   activate rows whose selection score
                    passes the threshold.
      flag_field    activate rows whose named boolean field is true.
      permuted_control  seeded count-matched control of another arm.
    """

    name: str = Field(..., min_length=1)
    strength: float = 0.0
    score_field: Optional[str] = None
    threshold: Optional[float] = None
    flag_field: Optional[str] = None
    permuted_control_of: Optional[str] = None
    control_seed: Optional[int] = None

    @model_validator(mode="after")
    def _one_selection(self) -> "ArmConfig":
        if self.score_field is not None and self.threshold is None:
            raise ValueError(f"arm {self.name}: score_field requires threshold")
        if self.permuted_control_of is not None and self.control_seed is None:
            raise ValueError(
                f"arm {self.name}: permuted_control_of requires control_seed"
            )
        return self


class SmokeConfig(BaseModel):
    """Small pre-run with readback tolerances that gates the full arms."""

    n_rows: int = Field(default=8, ge=1)
    write_rel_tol: float = Field(
        default=0.05,
        description="Allowed relative error on the commanded projection.",
    )
    write_abs_floor: float = Field(
        default=0.5, description="Absolute error floor for the write check."
    )
    offtarget_tol: float = Field(
        default=1e-3, description="Max allowed movement of inactive rows."
    )


class ExecutionConfig(BaseModel):
    """Lane-agnostic run controls."""

    output_path: str = Field(..., min_length=1, description="Per-row output JSONL")
    resume: bool = True
    grader: Optional[str] = Field(
        default=None, description="Grader spec 'module.path:callable'"
    )


class SteerCellConfig(BaseModel):
    """The full six-block steer cell."""

    surface: SurfaceConfig
    readouts: list[ReadoutRef] = Field(..., min_length=1)
    law: LawConfig
    arms: list[ArmConfig] = Field(..., min_length=1)
    execution: ExecutionConfig
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)

    @model_validator(mode="after")
    def _readout_exists(self) -> "SteerCellConfig":
        names = {r.name for r in self.readouts}
        if self.law.readout not in names:
            raise ValueError(
                f"law.readout {self.law.readout!r} is not a declared readout"
            )
        for arm in self.arms:
            if arm.permuted_control_of and arm.permuted_control_of not in {
                a.name for a in self.arms
            }:
                raise ValueError(
                    f"arm {arm.name}: permuted_control_of references unknown arm "
                    f"{arm.permuted_control_of!r}"
                )
        return self


# --------------------------------------------------------------------------
# Extraction and probe-fit
# --------------------------------------------------------------------------


class ExtractConfig(BaseModel):
    rows_path: str = Field(..., min_length=1)
    output_dir: str = Field(..., min_length=1)
    families: list[str] = Field(default_factory=lambda: ["anchor", "answer_end"])
    every_k: int = 4
    layers: Optional[list[int]] = None
    max_new_tokens: int = 48
    render_fn: str = Field(
        ..., description="Callable spec 'module.path:callable' returning a prompt"
    )
    content_end_fn: str = Field(
        ..., description="Callable spec resolving the last content-token index"
    )


class ProbeFitConfig(BaseModel):
    activations_path: str = Field(
        ..., min_length=1, description="Directory of extracted safetensors + manifest"
    )
    labels_path: str = Field(..., min_length=1, description="JSONL with row_key + label")
    position_family: str = "anchor"
    n_components: int = 128
    n_splits: int = 5
    seed: int = 0
    solver: str = "saga"
    tol: float = 1e-3
    C: float = 1.0
    output_direction: str = Field(..., min_length=1, description="Frozen direction JSON")
    normalize: bool = True


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_steer_config(path: str | Path) -> SteerCellConfig:
    return SteerCellConfig(**_load_yaml(path))


def load_extract_config(path: str | Path) -> ExtractConfig:
    return ExtractConfig(**_load_yaml(path))


def load_probe_fit_config(path: str | Path) -> ProbeFitConfig:
    return ProbeFitConfig(**_load_yaml(path))
