"""CPU expectation tests for generic MechInterp dose calibration."""

from __future__ import annotations

import json

import pytest
import yaml

from MechInterp import cell as cell_mod
from MechInterp import config as config_mod
from MechInterp.pipeline import PipelineConfig, build_pipeline_plan


def _required_config_api():
    names = [
        "DoseCalibrationConfig",
        "DoseCalibrationSelection",
        "DoseCalibrationBlock",
        "DoseCalibrationExecution",
        "load_dose_calibration_config",
    ]
    missing = [name for name in names if not hasattr(config_mod, name)]
    assert not missing, f"missing dose calibration config API: {missing}"
    return config_mod.DoseCalibrationConfig, config_mod.load_dose_calibration_config


def _required_cell_api():
    names = [
        "dose_completed_keys",
        "dose_pending_rows",
        "summarize_dose_calibration",
        "write_dose_manifest",
    ]
    missing = [name for name in names if not hasattr(cell_mod, name)]
    assert not missing, f"missing dose calibration cell API: {missing}"
    return {name: getattr(cell_mod, name) for name in names}


def _minimal_dose_dict(tmp_path):
    return {
        "surface": {
            "rows_path": str(tmp_path / "rows.jsonl"),
            "generation": {"max_new_tokens": 8, "do_sample": False},
            "seed": 7,
        },
        "readouts": [
            {"name": "axis", "path": str(tmp_path / "axis.json")},
            {"name": "other", "path": str(tmp_path / "other.json")},
        ],
        "law": {
            "kind": "additive",
            "readout": "axis",
            "position": "anchor",
            "generation_mode": "anchor",
        },
        "calibration": {
            "doses": [0.0, 0.5, 1.0],
            "dose_kind": "strength",
            "selection": {},
        },
        "execution": {
            "output_path": str(tmp_path / "dose_rows.jsonl"),
            "summary_path": str(tmp_path / "dose_summary.json"),
            "render_fn": "tests.fixtures:render",
            "grader": "tests.fixtures:grade",
        },
    }


def _rows():
    return [
        {"row_key": "a", "question": "A?"},
        {"row_key": "b", "question": "B?"},
    ]


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_dose_config_parses_defaults_and_loader(tmp_path):
    DoseCalibrationConfig, load_dose_calibration_config = _required_config_api()
    data = _minimal_dose_dict(tmp_path)
    cfg = DoseCalibrationConfig(**data)

    assert cfg.law.readout == "axis"
    assert cfg.calibration.doses == [0.0, 0.5, 1.0]
    assert cfg.calibration.dose_kind == "strength"
    assert cfg.execution.resume is True
    assert cfg.execution.batch_size == 1
    assert cfg.execution.redact_fields == []

    path = tmp_path / "dose.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    loaded = load_dose_calibration_config(path)
    assert loaded.execution.output_path == str(tmp_path / "dose_rows.jsonl")


def test_dose_config_accepts_setpoint_dose_kind(tmp_path):
    DoseCalibrationConfig, _ = _required_config_api()
    data = _minimal_dose_dict(tmp_path)
    data["calibration"]["dose_kind"] = "setpoint"
    cfg = DoseCalibrationConfig(**data)
    assert cfg.calibration.dose_kind == "setpoint"


def test_dose_config_accepts_checkpoint_redaction_fields(tmp_path):
    DoseCalibrationConfig, _ = _required_config_api()
    data = _minimal_dose_dict(tmp_path)
    data["execution"]["redact_fields"] = ["answer_text", "aliases"]
    cfg = DoseCalibrationConfig(**data)
    assert cfg.execution.redact_fields == ["answer_text", "aliases"]


def test_dose_law_readout_must_be_declared(tmp_path):
    DoseCalibrationConfig, _ = _required_config_api()
    data = _minimal_dose_dict(tmp_path)
    data["law"]["readout"] = "missing"
    with pytest.raises(ValueError):
        DoseCalibrationConfig(**data)


def test_dose_completed_keys_are_readout_dose_row_triples(tmp_path):
    api = _required_cell_api()
    out = tmp_path / "dose_rows.jsonl"
    _write_jsonl(
        out,
        [
            {"readout": "axis", "dose": 0.5, "row_key": "a"},
            {"readout": "axis", "dose": 1.0, "row_key": "a"},
            {"readout": "other", "dose": 0.5, "row_key": "a"},
        ],
    )

    assert api["dose_completed_keys"](out) == {
        ("axis", "0.5", "a"),
        ("axis", "1", "a"),
        ("other", "0.5", "a"),
    }


def test_dose_pending_rows_skip_only_completed_readout_dose_row_keys(tmp_path):
    DoseCalibrationConfig, _ = _required_config_api()
    api = _required_cell_api()
    cfg = DoseCalibrationConfig(**_minimal_dose_dict(tmp_path))
    out = tmp_path / "dose_rows.jsonl"
    _write_jsonl(out, [{"readout": "axis", "dose": 0.5, "row_key": "a"}])

    selection = cfg.calibration.selection
    pending_axis_half = api["dose_pending_rows"](
        _rows(),
        out,
        resume=True,
        readout="axis",
        dose=0.5,
        strength=0.5,
        selection=selection,
    )
    pending_axis_one = api["dose_pending_rows"](
        _rows(),
        out,
        resume=True,
        readout="axis",
        dose=1.0,
        strength=1.0,
        selection=selection,
    )
    pending_other_half = api["dose_pending_rows"](
        _rows(),
        out,
        resume=True,
        readout="other",
        dose=0.5,
        strength=0.5,
        selection=selection,
    )

    assert {r["row_key"] for r in pending_axis_half} == {"b"}
    assert {r["row_key"] for r in pending_axis_one} == {"a", "b"}
    assert {r["row_key"] for r in pending_other_half} == {"a", "b"}


def test_dose_pending_rows_no_resume_runs_all_combinations(tmp_path):
    DoseCalibrationConfig, _ = _required_config_api()
    api = _required_cell_api()
    cfg = DoseCalibrationConfig(**_minimal_dose_dict(tmp_path))
    out = tmp_path / "dose_rows.jsonl"
    _write_jsonl(out, [{"readout": "axis", "dose": 0.5, "row_key": "a"}])

    pending = api["dose_pending_rows"](
        _rows(),
        out,
        resume=False,
        readout="axis",
        dose=0.5,
        strength=0.5,
        selection=cfg.calibration.selection,
    )

    assert {r["row_key"] for r in pending} == {"a", "b"}


def test_dose_summary_aggregates_boolean_grader_fields_by_readout_and_dose(tmp_path):
    api = _required_cell_api()
    out = tmp_path / "dose_rows.jsonl"
    _write_jsonl(
        out,
        [
            {
                "readout": "axis",
                "dose": 0.5,
                "row_key": "a",
                "active": True,
                "correct": True,
                "refusal": False,
            },
            {
                "readout": "axis",
                "dose": 0.5,
                "row_key": "b",
                "active": True,
                "correct": False,
                "refusal": False,
            },
            {
                "readout": "axis",
                "dose": 1.0,
                "row_key": "a",
                "active": True,
                "correct": True,
                "refusal": True,
            },
            {
                "readout": "other",
                "dose": 0.5,
                "row_key": "a",
                "active": True,
                "correct": True,
                "refusal": False,
            },
        ],
    )

    summary = api["summarize_dose_calibration"](out)

    groups = {
        (group["readout"], group["dose"]): group for group in summary["groups"]
    }

    assert summary["n_records"] == 4
    assert groups[("axis", 0.5)]["n"] == 2
    assert groups[("axis", 0.5)]["bool_metrics"]["correct"]["true"] == 1
    assert groups[("axis", 0.5)]["bool_metrics"]["correct"]["rate"] == pytest.approx(0.5)
    assert groups[("axis", 0.5)]["bool_metrics"]["refusal"]["true"] == 0
    assert groups[("axis", 1.0)]["bool_metrics"]["refusal"]["rate"] == pytest.approx(1.0)
    assert groups[("other", 0.5)]["bool_metrics"]["correct"]["rate"] == pytest.approx(1.0)


def test_write_dose_manifest_records_config_and_summary_paths(tmp_path):
    DoseCalibrationConfig, _ = _required_config_api()
    api = _required_cell_api()
    cfg = DoseCalibrationConfig(**_minimal_dose_dict(tmp_path))
    summary = {"axis": {"0.5": {"n": 2}}}

    manifest_path = api["write_dose_manifest"](
        cfg.execution.output_path,
        cfg,
        config_sha="abc123",
        run_summaries=summary,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config_sha"] == "abc123"
    assert manifest["calibration"]["doses"] == [0.0, 0.5, 1.0]
    assert manifest["runs"] == summary


def test_pipeline_compiles_dose_calibrate_with_gpu_ack(tmp_path):
    cfg_path = tmp_path / "dose.yaml"
    cfg_path.write_text(yaml.safe_dump(_minimal_dose_dict(tmp_path)), encoding="utf-8")
    cfg = PipelineConfig(
        name="dose",
        model="Tiny/Model",
        runtime={"python": "python"},
        stages=[
            {
                "name": "dose",
                "kind": "mechinterp.dose-calibrate",
                "config": str(cfg_path),
            }
        ],
    )

    plan = build_pipeline_plan(cfg, repo_root=tmp_path, provider="local", gpu_ack=True)
    cmd = plan["stages"][0]["command"]

    assert cmd[:4] == ["python", str(tmp_path / "tuner.py"), "mechinterp", "dose-calibrate"]
    assert "--mi-config" in cmd
    assert str(cfg_path) in cmd
    assert cmd[cmd.index("--model") + 1] == "Tiny/Model"
    assert "--i-know-this-runs-on-gpu" in cmd
