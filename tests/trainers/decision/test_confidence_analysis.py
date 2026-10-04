"""Confidence analysis on a tiny random Llama (CPU, no downloads).

Covers config loading, task-stratified FIT/CAL/TEST splits, ablated twins,
hidden-state capture at <answer>, every confidence arm and the CLI end to end.
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

from decision_core.confidence_analysis import (  # noqa: E402
    ablate,
    load_analysis_config,
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
    config = {
        "checkpoint": f"latest:{ckpt_root}",
        "data": {"files": [str(data)], "max_rows": 240, "fit_fraction": 0.4, "cal_fraction": 0.2},
        "capture": {"layers": "all", "batch_size": 16},
        "probe": {"n_components": 8, "n_splits": 3, "max_iter": 300},
        "conformal": {"alphas": [0.2], "min_rows_per_kind": 10},
        "bootstrap": {"n_boot": 50},
        "output": {"output_root": str(tmp_path / "out")},
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
    assert len(rows_out) == report["n_rows"]["test"]
    assert {"conf_r1", "p_dial", "p_stack", "ablated_max_prob"} <= set(json.loads(rows_out[0]))
