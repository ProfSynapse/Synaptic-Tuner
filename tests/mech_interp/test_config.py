"""CPU tests for MechInterp recipe parsing and validation."""

from pathlib import Path

import pytest
import yaml

from tuner.project import ProjectContext
from tuner.project.errors import WriteAccessError

from MechInterp.config import (
    SteerCellConfig,
    load_dose_calibration_config,
    load_steer_config,
    load_extract_config,
    load_probe_fit_config,
)


def _minimal_steer_dict():
    return {
        "surface": {"rows_path": "rows.jsonl", "seed": 7},
        "readouts": [{"name": "axis", "path": "d.json"}],
        "law": {"kind": "erase_write", "readout": "axis", "position": "anchor"},
        "arms": [
            {"name": "baseline", "strength": 0.0},
            {"name": "primary", "strength": 2.0, "score_field": "s", "threshold": 1.0},
        ],
        "execution": {"output_path": "out/rows.jsonl"},
    }


def test_steer_config_parses_and_defaults():
    cfg = SteerCellConfig(**_minimal_steer_dict())
    assert cfg.law.readout == "axis"
    assert cfg.smoke.n_rows == 8  # default
    assert cfg.surface.generation.do_sample is False  # greedy default


def test_law_readout_must_be_declared():
    d = _minimal_steer_dict()
    d["law"]["readout"] = "missing"
    with pytest.raises(ValueError):
        SteerCellConfig(**d)


def test_score_field_requires_threshold():
    d = _minimal_steer_dict()
    d["arms"][1] = {"name": "primary", "strength": 2.0, "score_field": "s"}
    with pytest.raises(ValueError):
        SteerCellConfig(**d)


def test_permuted_control_requires_seed():
    d = _minimal_steer_dict()
    d["arms"].append({"name": "control", "permuted_control_of": "primary"})
    with pytest.raises(ValueError):
        SteerCellConfig(**d)


def test_permuted_control_references_known_arm():
    d = _minimal_steer_dict()
    d["arms"].append(
        {"name": "control", "permuted_control_of": "ghost", "control_seed": 1}
    )
    with pytest.raises(ValueError):
        SteerCellConfig(**d)


def test_gain_field_mutually_exclusive_with_score_field():
    d = _minimal_steer_dict()
    d["arms"][1] = {
        "name": "primary", "strength": 1.0, "score_field": "s", "threshold": 1.0,
        "gain_field": "prop_z",
    }
    with pytest.raises(ValueError):
        SteerCellConfig(**d)


def test_gain_field_mutually_exclusive_with_flag_field():
    d = _minimal_steer_dict()
    d["arms"][1] = {"name": "primary", "strength": 1.0, "flag_field": "f", "gain_field": "prop_z"}
    with pytest.raises(ValueError):
        SteerCellConfig(**d)


def test_gain_field_mutually_exclusive_with_permuted_control_of():
    d = _minimal_steer_dict()
    d["arms"][1] = {
        "name": "primary", "strength": 1.0, "gain_field": "prop_z",
        "permuted_control_of": "baseline", "control_seed": 1,
    }
    with pytest.raises(ValueError):
        SteerCellConfig(**d)


def test_gain_clip_requires_gain_field():
    d = _minimal_steer_dict()
    d["arms"][1] = {"name": "primary", "strength": 1.0, "gain_clip": 2.0}
    with pytest.raises(ValueError):
        SteerCellConfig(**d)


def test_gain_field_arm_parses_with_clip_and_force_active():
    d = _minimal_steer_dict()
    d["arms"][1] = {
        "name": "coupled", "strength": 1.0, "gain_field": "prop_z", "gain_clip": 2.0,
    }
    d["arms"].append({"name": "ablate", "strength": 0.0, "force_active": True})
    cfg = SteerCellConfig(**d)
    assert cfg.arms[1].gain_field == "prop_z"
    assert cfg.arms[1].gain_clip == 2.0
    assert cfg.arms[2].force_active is True
    # defaults: force_active is False unless set
    assert cfg.arms[0].force_active is False


def test_permuted_control_of_gain_arm_parses():
    d = _minimal_steer_dict()
    d["arms"][1] = {"name": "coupled", "strength": 1.0, "gain_field": "prop_z"}
    d["arms"].append(
        {"name": "permuted", "strength": 1.0, "permuted_control_of": "coupled", "control_seed": 7}
    )
    cfg = SteerCellConfig(**d)
    assert cfg.arms[2].permuted_control_of == "coupled"


def test_bundled_templates_parse(tmp_path):
    """The shipped example recipes must parse against the schema."""
    from pathlib import Path

    tdir = Path(__file__).resolve().parents[2] / "MechInterp" / "configs" / "templates"
    steer = load_steer_config(tdir / "steer_cell.yaml")
    assert any(a.permuted_control_of == "primary" for a in steer.arms)
    extract = load_extract_config(tdir / "extract.yaml")
    assert extract.families
    probe = load_probe_fit_config(tdir / "probe_fit.yaml")
    assert probe.n_components > 0
    dose = load_dose_calibration_config(tdir / "dose_calibration.yaml")
    assert dose.calibration.doses


def test_load_steer_config_roundtrip(tmp_path):
    p = tmp_path / "cell.yaml"
    p.write_text(yaml.safe_dump(_minimal_steer_dict()))
    cfg = load_steer_config(p)
    assert cfg.arms[1].threshold == 1.0


def test_host_config_resolves_declaring_document_and_uri_paths(tmp_path):
    engine = tmp_path / "engine"
    project = tmp_path / "host"
    recipe = project / "experiments" / "cell.yaml"
    recipe.parent.mkdir(parents=True)
    (recipe.parent / "rows.jsonl").write_text("{}\n", encoding="utf-8")
    (project / "directions").mkdir()
    (project / "directions" / "d.json").write_text("{}", encoding="utf-8")
    data = _minimal_steer_dict()
    data["readouts"][0]["path"] = "project://directions/d.json"
    data["execution"]["output_path"] = "artifact://mechinterp/rows.jsonl"
    recipe.write_text(yaml.safe_dump(data), encoding="utf-8")
    context = ProjectContext.host(engine_root=engine, project_root=project)

    cfg = load_steer_config(recipe, context=context)

    assert Path(cfg.surface.rows_path) == recipe.parent / "rows.jsonl"
    assert Path(cfg.readouts[0].path) == project / "directions" / "d.json"
    assert Path(cfg.execution.output_path) == context.artifact_root / "mechinterp" / "rows.jsonl"


def test_host_config_rejects_source_tree_output(tmp_path):
    project = tmp_path / "host"
    recipe = project / "experiments" / "cell.yaml"
    recipe.parent.mkdir(parents=True)
    data = _minimal_steer_dict()
    data["execution"]["output_path"] = "rows-out.jsonl"
    recipe.write_text(yaml.safe_dump(data), encoding="utf-8")
    context = ProjectContext.host(engine_root=tmp_path / "engine", project_root=project)

    with pytest.raises(WriteAccessError):
        load_steer_config(recipe, context=context)
