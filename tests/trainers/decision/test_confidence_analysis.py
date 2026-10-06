"""Confidence analysis on a tiny random Llama (CPU, no downloads).

Covers config loading, task-stratified FIT/CAL/TEST splits, ablated twins,
hidden-state capture at <answer>, every confidence arm, the export / external
direction options and the CLI end to end.
Needs transformers + peft + sklearn (runs in the training image).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DECISION_DIR = REPO_ROOT / "Trainers" / "decision"
TEST_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, DECISION_DIR, TEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from decision_core.confidence_analysis import (  # noqa: E402
    DirectionSpec,
    ablate,
    export_arrays,
    load_analysis_config,
    load_direction,
    score_direction,
    score_directions,
    split_fit_cal_test,
    stratified_sample,
)
from decision_core.examples import DecisionExample, write_jsonl  # noqa: E402


def rows(n: int, tasks=("a", "b", "c")) -> list[DecisionExample]:
    out = []
    for i in range(n):
        out.append(DecisionExample("choice", f"state {i}", "pick", [["x", ""], ["y", ""], ["z", ""]],
                                   i % 3, task=tasks[i % len(tasks)]))
    return out


def test_confidence_analysis_configs_load():
    for name in ("confidence_analysis_pointer.yaml", "confidence_analysis_letter_logits.yaml"):
        cfg = load_analysis_config(DECISION_DIR / "configs" / "experiments" / name)
        assert cfg.checkpoint.startswith("latest:")
        assert cfg.data.fit_fraction + cfg.data.cal_fraction < 1


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(json.dumps({"checkpoint": "x", "data": {"files": ["f"], "nope": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="nope"):
        load_analysis_config(p)


def _write_cfg(tmp_path, **extra) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(json.dumps({"checkpoint": "x", "data": {"files": ["f"]}, **extra}), encoding="utf-8")
    return p


def test_export_and_directions_default_off(tmp_path):
    cfg = load_analysis_config(_write_cfg(tmp_path))
    assert cfg.export.states is False and cfg.export.per_row is True and cfg.export.layers is None
    assert cfg.export.cal_rows is False and cfg.export.fit_rows is False
    assert cfg.directions == []
    cfg = load_analysis_config(_write_cfg(
        tmp_path, capture={"layers": [0, 3, 5]}, export={"states": True, "layers": [3]},
        directions=[{"name": "d1", "path": "a.json"}, {"name": "d2", "path": "/abs/b.json", "layer": 5}]))
    assert cfg.export.states and cfg.export.layers == [3]
    assert cfg.directions == [DirectionSpec("d1", "a.json", None), DirectionSpec("d2", "/abs/b.json", 5)]
    assert cfg.to_dict()["directions"][1] == {"name": "d2", "path": "/abs/b.json", "layer": 5}
    cfg = load_analysis_config(_write_cfg(tmp_path, export={"cal_rows": True, "fit_rows": True}))
    assert cfg.export.cal_rows is True and cfg.export.fit_rows is True
    assert cfg.to_dict()["export"]["cal_rows"] is True


@pytest.mark.parametrize("extra, match", [
    ({"export": {"nope": 1}}, "nope"),
    ({"export": {"states": "yes"}}, "true/false"),
    ({"export": {"cal_rows": "yes"}}, r"export\.cal_rows must be true/false"),
    ({"export": {"fit_rows": 1}}, r"export\.fit_rows must be true/false"),
    ({"export": {"test_rows": True}}, "test_rows"),
    ({"export": {"layers": [-1]}}, "export.layers"),
    ({"export": {"layers": []}}, "export.layers"),
    ({"capture": {"layers": [0, 1]}, "export": {"layers": [2]}}, r"\[2\] are not in capture.layers"),
    ({"directions": {"name": "d"}}, "must be a list"),
    ({"directions": ["d.json"]}, "must be a mapping"),
    ({"directions": [{"name": "d", "path": "p", "scale": 2}]}, "scale"),
    ({"directions": [{"path": "p"}]}, "name"),
    ({"directions": [{"name": "a b", "path": "p"}]}, "name"),
    ({"directions": [{"name": "d"}]}, "needs a path"),
    ({"directions": [{"name": "d", "path": "p", "layer": -2}]}, "hidden-state index"),
    ({"directions": [{"name": "d", "path": "p", "layer": True}]}, "hidden-state index"),
    ({"directions": [{"name": "d", "path": "p"}, {"name": "d", "path": "q"}]}, "duplicate"),
    ({"capture": {"layers": [0, 1]}, "directions": [{"name": "d", "path": "p", "layer": 4}]},
     "not in capture.layers"),
])
def test_export_and_directions_validation(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        load_analysis_config(_write_cfg(tmp_path, **extra))


def test_frozen_direction_load_score_and_checks(tmp_path):
    pytest.importorskip("sklearn")
    from MechInterp.probe.fit import freeze_direction

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 16))
    y = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
    path = tmp_path / "dir.json"
    freeze_direction(X, y, layer=3, out_path=path, n_components=8, seed=0)
    rec = load_direction(DirectionSpec("d", str(path)), path, hidden_size=16)
    assert rec["resolved_layer"] == 3
    # The documented rule reproduces the score freeze_direction computed its sigma / class stats on.
    s = score_direction(rec, X)
    assert s.std() == pytest.approx(rec["sigma"], rel=1e-9)
    assert s[y == 1].mean() == pytest.approx(rec["calibration"]["positive_mean"], rel=1e-9)
    assert s[y == 0].mean() == pytest.approx(rec["calibration"]["negative_mean"], rel=1e-9)
    # ...and ranks rows exactly like the mu-centred projection on the unit vector.
    proj = (X - rec["mu_np"]) @ np.asarray(rec["vector"], dtype=np.float64)
    np.testing.assert_allclose(s, rec["raw_norm"] * proj + (s - rec["raw_norm"] * proj).mean(), atol=1e-8)
    assert load_direction(DirectionSpec("d", str(path), layer=1), path)["resolved_layer"] == 1
    with pytest.raises(ValueError, match="hidden size 32"):
        load_direction(DirectionSpec("d", str(path)), path, hidden_size=32)
    with pytest.raises(FileNotFoundError):
        load_direction(DirectionSpec("d", "missing.json"), tmp_path / "missing.json")
    states = {0: X.astype(np.float16), 3: X.astype(np.float16)}
    assert score_directions([(DirectionSpec("d", str(path)), rec)], states)["d"].shape == (60,)
    with pytest.raises(ValueError, match="was not captured"):
        score_directions([(DirectionSpec("d", str(path)), {**rec, "resolved_layer": 5})], states)
    with pytest.raises(ValueError, match="hidden size 8"):
        score_directions([(DirectionSpec("d", str(path)), rec)], {3: np.zeros((4, 8), np.float16)})
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["schema_version"] = "other/v0"
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_direction(DirectionSpec("d", "bad.json"), tmp_path / "bad.json")


def test_export_arrays_layers_and_row_index():
    states = {0: np.ones((5, 4), np.float32), 2: np.zeros((5, 4), np.float16)}
    arrays = export_arrays(states, None, 5)
    assert set(arrays) == {"L0", "L2", "row_index"} and arrays["L0"].dtype == np.float16
    assert arrays["row_index"].tolist() == [0, 1, 2, 3, 4]
    assert set(export_arrays(states, [2], 5)) == {"L2", "row_index"}
    with pytest.raises(ValueError, match="not captured"):
        export_arrays(states, [1], 5)


def test_splits_are_disjoint_stratified_and_sum():
    data = rows(300)
    sample = stratified_sample(data, 150, seed=0)
    assert len(sample) == 150 and {r.task for r in sample} == {"a", "b", "c"}
    fit, cal, test = split_fit_cal_test(sample, 0.4, 0.2, seed=0)
    assert len(fit) + len(cal) + len(test) == 150
    ids = [id(r) for r in fit + cal + test]
    assert len(ids) == len(set(ids))
    assert len(fit) == pytest.approx(60, abs=3) and len(cal) == pytest.approx(30, abs=3)


def test_ablated_twin_keeps_question_and_options():
    ex = rows(1)[0]
    twin = ablate(ex, "(nothing)")
    assert twin.state == "(nothing)" and twin.options == ex.options and twin.label == ex.label
    assert twin.instructions == ex.instructions and ex.state == "state 0"


@pytest.mark.skipif(sys.platform == "win32", reason="shared.training_capacity needs `resource`")
def test_analyze_end_to_end_on_tiny_model(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("sklearn")
    from shared import env_bootstrap
    from test_decision_model_tiny import LORA_TARGETS, build_base, build_tokenizer, make_examples

    import analyze_confidence
    from decision_core.modeling import DecisionModel
    from decision_core.model_config import DecisionModelConfig

    monkeypatch.setattr(env_bootstrap, "init_trainer_env", lambda **_: None)
    base = tmp_path / "base"
    tok = build_tokenizer(base)
    build_base(tok, base)
    cfg = DecisionModelConfig(hf_id=str(base), lora_targets=LORA_TARGETS, readout="pointer", pointer_dim=32,
                              max_length=256, torch_dtype="float32", lora_r=4, lora_alpha=8)
    ckpt_root = tmp_path / "runs"
    DecisionModel.from_base(cfg).save(ckpt_root / "20261004_000000" / "final_model")
    data = tmp_path / "heldout.jsonl"
    examples = make_examples(240, seed=3)
    for i, ex in enumerate(examples):  # prior-knowledge labels, as an external labeling protocol supplies them
        ex.meta = {"knowledge": ("known", "unknown", "ambiguous")[i % 3], "s_pop": i}
    write_jsonl(data, examples)
    # An external frozen direction, fit on random 64-d "states" (the tiny model's hidden size).
    from MechInterp.probe.fit import freeze_direction

    rng = np.random.default_rng(1)
    rand_X = rng.normal(size=(40, 64))
    direction_path = tmp_path / "directions" / "random.json"
    freeze_direction(rand_X, (rand_X[:, 0] > 0).astype(int), layer=1, out_path=direction_path, n_components=8)
    config = {
        "checkpoint": f"latest:{ckpt_root}",
        "data": {"files": [str(data)], "max_rows": 240, "fit_fraction": 0.4, "cal_fraction": 0.2},
        "capture": {"layers": "all", "batch_size": 16},
        "probe": {"n_components": 8, "n_splits": 3, "max_iter": 300},
        "conformal": {"alphas": [0.2], "min_rows_per_kind": 10},
        "bootstrap": {"n_boot": 50},
        "output": {"output_root": str(tmp_path / "out")},
        "export": {"states": True, "layers": [0, 2], "per_row": True, "cal_rows": True},
        "directions": [{"name": "random", "path": str(direction_path)},
                       {"name": "random_last", "path": str(direction_path), "layer": 2}],
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    assert analyze_confidence.main(["--config", str(cfg_path), "--run-timestamp", "unit"]) == 0
    report = json.loads((tmp_path / "out" / "unit" / "confidence_report.json").read_text(encoding="utf-8"))
    assert report["abstention_rate"] == 0.0
    assert set(report["arms"]) == {"R0_raw", "R1_calibrated", "P_dial", "S_stacked"}
    assert str(report["probe_dial"]["best_layer"]) in report["probe_dial"]["auroc_by_layer"]  # JSON keys
    assert len(report["probe_dial"]["auroc_by_layer"]) == 3       # embeddings + 2 layers
    assert "permuted_label_cv_auroc" in report["probe_dial"]
    assert set(report["conformal"]["alpha_0.2"]) >= {"test_real", "test_ablated", "qhat_by_kind"}
    h = report["humility_ablated"]
    assert 0.0 <= h["mean_chance"] <= h["mean_max_prob"] <= 1.0
    assert 0.0 <= report["probe_gate"]["test_auroc_real_vs_ablated"] <= 1.0
    k = report["knowledge"]
    assert set(k["groups"]) == {"known", "unknown", "ambiguous"}
    assert 0.0 <= k["readout_auroc_known_vs_unknown"] <= 1.0
    assert "ku_probe_minus_readout" in k and len(k["by_popularity_quartile"]) == 4
    rows_out = (tmp_path / "out" / "unit" / "test_rows.jsonl").read_text(encoding="utf-8").splitlines()
    n_test = report["n_rows"]["test"]
    assert len(rows_out) == n_test
    assert {"conf_r1", "p_dial", "p_stack", "ablated_max_prob"} <= set(json.loads(rows_out[0]))

    # export.per_row fields
    recs = [json.loads(line) for line in rows_out]
    assert [r["row_index"] for r in recs] == list(range(n_test))
    assert {r["split"] for r in recs} == {"test"}
    for r in recs:
        assert len(r["probs_r0"]) == len(r["probs_r1"]) == r["n_options"]
        assert sum(r["probs_r1"]) == pytest.approx(1.0, abs=1e-6)
        assert max(r["probs_r0"]) == pytest.approx(r["conf_r0"]) and max(r["probs_r1"]) == pytest.approx(r["conf_r1"])
        assert r["p_dial"] == pytest.approx(1 / (1 + np.exp(-r["dial_score"])), abs=1e-6)
        assert r["p_stack"] == pytest.approx(1 / (1 + np.exp(-r["stack_score"])), abs=1e-6)
        assert isinstance(r["ku_probe_score"], float)   # the known-vs-unknown probe was fit
        assert set(r["direction_scores"]) == {"random", "random_last"}

    # export.cal_rows -> cal_rows.jsonl: same schema, CAL-relative row_index; fit_rows stays off
    run_dir = tmp_path / "out" / "unit"
    assert not (run_dir / "fit_rows.jsonl").exists()
    cal = [json.loads(line) for line in (run_dir / "cal_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    n_cal = report["n_rows"]["cal"]
    assert n_cal > 0 and len(cal) == n_cal
    assert [r["row_index"] for r in cal] == list(range(n_cal)) and {r["split"] for r in cal} == {"cal"}
    assert set(cal[0]) == set(recs[0])
    for r in cal:
        assert len(r["probs_r0"]) == len(r["probs_r1"]) == r["n_options"]
        assert sum(r["probs_r1"]) == pytest.approx(1.0, abs=1e-6)
        assert max(r["probs_r1"]) == pytest.approx(r["conf_r1"])
        assert r["p_dial"] == pytest.approx(1 / (1 + np.exp(-r["dial_score"])), abs=1e-6)
        assert r["p_stack"] == pytest.approx(1 / (1 + np.exp(-r["stack_score"])), abs=1e-6)
        assert isinstance(r["ku_probe_score"], float)
        assert set(r["direction_scores"]) == {"random", "random_last"}
    # CAL R1 uses the CAL-fit per-kind temperatures reported for TEST
    t = report["temperatures_cal"]
    for r in cal:
        lg = np.log(np.asarray(r["probs_r0"]))
        expect = np.exp(lg / t.get(r["kind"], 1.0))
        np.testing.assert_allclose(r["probs_r1"], expect / expect.sum(), rtol=1e-5, atol=1e-6)

    # export.states -> test_states.npz
    npz = np.load(tmp_path / "out" / "unit" / "test_states.npz")
    assert set(npz.files) == {"L0", "L2", "row_index"}
    assert npz["L0"].shape == npz["L2"].shape == (n_test, 64) and npz["L2"].dtype == np.float16
    assert npz["row_index"].tolist() == list(range(n_test))

    # directions report block; per-row scores follow the direction's own scoring rule
    d = report["directions"]
    assert set(d) == {"random", "random_last"}
    assert d["random"]["layer"] == d["random"]["source_layer"] == 1 and d["random_last"]["layer"] == 2
    for block in d.values():
        assert 0.0 <= block["auroc_correct"] <= 1.0 and block["n"] == n_test
        assert {"diff", "ci95", "n_boot_used"} <= set(block["auroc_correct_minus_r1"])
        assert block["by_correctness"]["correct"]["n"] + block["by_correctness"]["wrong"]["n"] == n_test
        assert 0.0 <= block["auroc_known_vs_unknown"] <= 1.0
        assert "diff" in block["auroc_known_vs_unknown_minus_r1"]
        assert set(block["by_knowledge"]) == {"known", "unknown", "ambiguous"}
        assert block["score_rule"].startswith("logistic_decision_value")
    rec = load_direction(DirectionSpec("random_last", str(direction_path), 2), direction_path)
    np.testing.assert_allclose([r["direction_scores"]["random_last"] for r in recs],
                               score_direction(rec, npz["L2"]), rtol=1e-6, atol=1e-6)
