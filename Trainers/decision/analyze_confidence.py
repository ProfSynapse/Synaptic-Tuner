#!/usr/bin/env python3
"""
Confidence analysis: is a decision model's confidence calibrated, and does an
internal probe know more than the readout says?

Location: Trainers/decision/analyze_confidence.py
Config:   Trainers/decision/configs/experiments/confidence_analysis_*.yaml
Recipe:   Trainers/recipes/decision_confidence_analysis_*.yaml

Read-only over a trained final_model. Writes to <output_root>/<timestamp>/:
    confidence_report.json   every arm + metric (see decision_core/confidence_analysis.py)
    test_rows.jsonl          per-row TEST records (split, confidence per arm, ablated twin; with
                             export.per_row also option probabilities, raw scores, direction scores)
    cal_rows.jsonl           export.cal_rows only: the same records for CAL rows
    fit_rows.jsonl           export.fit_rows only: the same records for FIT rows (in-sample probe scores)
    test_states.npz          export.states only: <answer> states of TEST rows (L{i} + row_index)
    analysis_config.json     the resolved config

Usage:
    python Trainers/decision/analyze_confidence.py --config Trainers/decision/configs/experiments/confidence_analysis_pointer.yaml --dry-run
    python Trainers/decision/analyze_confidence.py --config ... [--checkpoint <final_model>] [--run-timestamp T]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

TRAINER_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINER_DIR.parent.parent
sys.path.insert(0, str(TRAINER_DIR))
sys.path.insert(0, str(REPO_ROOT))

from decision_core.confidence_analysis import (  # noqa: E402
    load_analysis_config,
    load_direction,
    split_fit_cal_test,
    stratified_sample,
)
from decision_core.examples import load_examples  # noqa: E402


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def latest_final_model(pattern_root: Path) -> Path | None:
    """Newest <root>/<timestamp>/final_model under a run root, for 'latest:' checkpoints."""
    candidates = sorted(p for p in pattern_root.glob("*/final_model") if p.is_dir())
    return candidates[-1] if candidates else None


def resolve_checkpoint(spec: str) -> Path:
    """A final_model path, or 'latest:<run root>' for the newest run under that root."""
    if spec.startswith("latest:"):
        root = _resolve(spec[len("latest:"):])
        found = latest_final_model(root)
        if found is None:
            raise SystemExit(f"no <timestamp>/final_model under {root}")
        return found
    return _resolve(spec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Decision-model confidence analysis")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="Override the config's checkpoint.")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--run-timestamp", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Validate config + data splits; no model load.")
    args = ap.parse_args(argv)

    cfg = load_analysis_config(args.config)
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint
    if args.output_root:
        cfg.output.output_root = args.output_root

    rows = stratified_sample(load_examples([_resolve(f) for f in cfg.data.files]), cfg.data.max_rows,
                             cfg.data.seed)
    fit, cal, test = split_fit_cal_test(rows, cfg.data.fit_fraction, cfg.data.cal_fraction, cfg.data.seed)
    print(f"rows: fit {len(fit):,} | cal {len(cal):,} | test {len(test):,} "
          f"(ablated twins: {'on' if cfg.ablation.enabled else 'off'})")
    if args.dry_run:
        for d in cfg.directions:
            rec = load_direction(d, _resolve(d.path))
            print(f"direction {d.name}: {len(rec['vector'])}-d at hidden-state index {rec['resolved_layer']}")
        print(f"checkpoint spec: {cfg.checkpoint}")
        print("Dry run OK.")
        return 0

    from shared.env_bootstrap import init_trainer_env

    init_trainer_env(apply_windows_patches=False)
    import torch

    from decision_core.confidence_analysis import analyze
    from decision_core.modeling import DecisionModel

    ckpt = resolve_checkpoint(cfg.checkpoint)
    cfg.checkpoint = str(ckpt)
    print(f"checkpoint: {ckpt}")
    model = DecisionModel.load(ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
    report, records, arrays = analyze(model, cfg, REPO_ROOT)

    out = _resolve(cfg.output.output_root) / (args.run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "confidence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "analysis_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    for split, recs in records.items():
        with open(out / f"{split}_rows.jsonl", "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
    if arrays:
        import numpy as np

        np.savez_compressed(out / "test_states.npz", **arrays)

    arms = report["arms"]
    print(f"\nTEST accuracy {report['test_accuracy']:.4f} (abstention 0%)")
    for name, a in arms.items():
        ece = a.get("option_ece", a.get("ece_binary"))
        print(f"  {name:<14} AUROC {a['auroc']:.4f}  AURC {a['aurc']:.4f}  ECE {ece:.4f}  "
              f"confident-wrong {a['confident_wrong_rate']:.4f}  underconfident-right "
              f"{a['underconfident_right_rate']:.4f}")
    d = report["h2_dial_minus_r1"]
    print(f"  H2 AUROC(P_dial) - AUROC(R1) = {d['diff']:+.4f}  CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
          f"(dial best layer {report['probe_dial']['best_layer']})")
    if "humility_ablated" in report:
        h = report["humility_ablated"]
        print(f"  H3 ablated-twin gap {h['humility_gap']:.4f} (mean max-p {h['mean_max_prob']:.3f} vs chance "
              f"{h['mean_chance']:.3f}); confident on ablated {h['share_confident']:.3f}")
    if "knowledge" in report:
        k = report["knowledge"]
        for name, g in k["groups"].items():
            print(f"  {name:<9} n={g['n']:<5} acc {g['accuracy']:.3f}  conf {g['mean_confidence']:.3f} "
                  f"(chance {g['mean_chance']:.3f})  confident-wrong {g['confident_wrong_rate']:.3f}  "
                  f"underconfident-right {g['underconfident_right_rate']:.3f}")
        if "readout_auroc_known_vs_unknown" in k:
            line = f"  known-vs-unknown AUROC: readout {k['readout_auroc_known_vs_unknown']:.4f}"
            if "ku_probe" in k:
                d = k["ku_probe_minus_readout"]
                line += (f" | KU probe {k['ku_probe']['test_auroc_known_vs_unknown']:.4f} "
                         f"(layer {k['ku_probe']['best_layer']}; diff {d['diff']:+.4f} "
                         f"CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}])")
            print(line)
    for name, d in report.get("directions", {}).items():
        line = (f"  direction {name} (layer {d['layer']}): AUROC correct {d['auroc_correct']:.4f} "
                f"(minus R1 {d['auroc_correct_minus_r1']['diff']:+.4f})")
        if "auroc_known_vs_unknown" in d:
            line += f" | known-vs-unknown {d['auroc_known_vs_unknown']:.4f}"
        print(line)
    print(f"\nReport: {out / 'confidence_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
