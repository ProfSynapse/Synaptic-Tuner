"""
Confidence analysis: does a decision model's confidence carry what it knows?

Location: Trainers/decision/decision_core/confidence_analysis.py
Used by:  Trainers/decision/analyze_confidence.py

Read-only over a trained checkpoint. For held-out rows split FIT / CAL / TEST
(stratified by task) plus a state-ablated twin of every row, it computes:

  R0      raw max option probability
  R1      per-kind temperature fitted on CAL real rows (never the checkpoint's
          saved temperatures, which were fitted on overlapping held-out rows)
  P-dial  linear probe P(own answer correct) on the <answer> hidden state,
          layer swept by out-of-fold AUROC on FIT (MechInterp.probe.fit)
  P-gate  linear probe P(state is real, not ablated) -- whether the hidden
          state encodes that the evidence needed to answer is present
  S       logistic stack of logit(R1) and the P-dial score, fitted on CAL

and, on TEST: accuracy (100% coverage; the model never abstains), ECE / NLL /
Brier, AUROC(confidence -> correct) with a bootstrap floor, AURC, the
confident-wrong / underconfident-right pair, the confidence-over-chance gap on
ablated twins,
split-conformal LAC sets per kind, and ordinal intervals for score questions.

Optional extras (config-driven):
  directions  external frozen directions (``mechinterp-direction/v1`` JSON from
              MechInterp.probe.fit.freeze_direction) scored on every TEST row's
              <answer> state; AUROC vs correctness / prior knowledge per direction
  export      per-row option probabilities and raw scores in test_rows.jsonl,
              (opt-in) the same records for CAL / FIT rows (cal_rows.jsonl /
              fit_rows.jsonl, e.g. to fit abstention thresholds on CAL and apply
              them on TEST), and (opt-in) the TEST <answer> states as test_states.npz

Layer convention: a "layer" here is an index into the decoder's
``output_hidden_states`` tuple -- 0 is the embedding output, i >= 1 is the output
of decoder block i (``layers[i - 1]``); in HF Llama/Qwen-style decoders the last
index usually carries the final norm. MechInterp.extraction captures the same
tuple, so a direction frozen at index i there is scored at index i here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from shared.ml.calibration_metrics import (
    aurc,
    auroc,
    calibration_report,
    confident_wrong_rate,
    conformal_lac_qhat,
    expected_calibration_error,
    fit_temperature,
    lac_set,
    shortest_ordinal_interval,
    softmax,
    underconfident_right_rate,
)

from .examples import KINDS, DecisionExample, load_examples, split_by_task

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class DataSection:
    files: list[str] = field(default_factory=list)
    max_rows: int | None = 4000
    fit_fraction: float = 0.4
    cal_fraction: float = 0.2
    seed: int = 0


@dataclass
class AblationSection:
    enabled: bool = True
    state_text: str = "(no information about this case is available)"


@dataclass
class CaptureSection:
    layers: Any = "all"            # "all" or a list of hidden-state indices (0 = embeddings)
    batch_size: int = 8
    max_length: int | None = None  # null -> the checkpoint's max_length


@dataclass
class ProbeSection:
    n_components: int = 128
    n_splits: int = 5
    C: float = 1.0
    max_iter: int = 2000
    permuted_control: bool = True


@dataclass
class ConformalSection:
    alphas: list[float] = field(default_factory=lambda: [0.1, 0.2])
    min_rows_per_kind: int = 50


@dataclass
class ThresholdSection:
    confident: float = 0.8
    underconfident: float = 0.5
    ece_bins: int = 15


@dataclass
class BootstrapSection:
    n_boot: int = 1000
    seed: int = 0


@dataclass
class OutputSection:
    output_root: str = "decision_output/confidence"


@dataclass
class ExportSection:
    states: bool = False   # write test_states.npz: <answer> states of TEST rows (float16)
    layers: Any = None     # null -> every captured layer; else a subset of the captured indices
    per_row: bool = True   # add option probabilities and raw scores to the *_rows.jsonl records
    cal_rows: bool = False  # also write cal_rows.jsonl: the same per-row records for CAL rows
    fit_rows: bool = False  # also write fit_rows.jsonl (probe scores are in-sample there)


@dataclass
class DirectionSpec:
    """An external frozen direction (``mechinterp-direction/v1`` JSON) to score TEST rows along."""

    name: str = ""
    path: str = ""
    layer: int | None = None  # hidden-state index (capture.layers convention); null -> the JSON's layer


DIRECTION_SCHEMA = "mechinterp-direction/v1"
_DIRECTION_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")


@dataclass
class ConfidenceAnalysisConfig:
    checkpoint: str = ""
    data: DataSection = field(default_factory=DataSection)
    ablation: AblationSection = field(default_factory=AblationSection)
    capture: CaptureSection = field(default_factory=CaptureSection)
    probe: ProbeSection = field(default_factory=ProbeSection)
    conformal: ConformalSection = field(default_factory=ConformalSection)
    thresholds: ThresholdSection = field(default_factory=ThresholdSection)
    bootstrap: BootstrapSection = field(default_factory=BootstrapSection)
    output: OutputSection = field(default_factory=OutputSection)
    export: ExportSection = field(default_factory=ExportSection)
    directions: list[DirectionSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_directions(path: str | Path, value: Any) -> list[DirectionSpec]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path}: 'directions' must be a list of {{name, path, layer?}} mappings")
    allowed = {f.name for f in fields(DirectionSpec)}
    out: list[DirectionSpec] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: directions[{i}] must be a mapping")
        bad = sorted(set(item) - allowed)
        if bad:
            raise ValueError(f"{path}: unknown key(s) in directions[{i}]: {bad}")
        name, dpath, layer = str(item.get("name") or ""), str(item.get("path") or ""), item.get("layer")
        if not name or not set(name) <= _DIRECTION_NAME_CHARS:
            raise ValueError(f"{path}: directions[{i}].name must be non-empty [A-Za-z0-9_.-], got {name!r}")
        if not dpath:
            raise ValueError(f"{path}: directions[{i}] ({name}) needs a path")
        if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int) or layer < 0):
            raise ValueError(f"{path}: directions[{i}] ({name}).layer must be a hidden-state index >= 0")
        out.append(DirectionSpec(name=name, path=dpath, layer=layer))
    names = [d.name for d in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{path}: duplicate direction name(s) {dup}")
    return out


def load_analysis_config(path: str | Path) -> ConfidenceAnalysisConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sections = {f.name: f for f in fields(ConfidenceAnalysisConfig)}
    unknown = sorted(set(raw) - set(sections))
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {unknown}")
    cfg = ConfidenceAnalysisConfig()
    for name, value in raw.items():
        if name == "checkpoint":
            cfg.checkpoint = str(value or "")
            continue
        if name == "directions":
            cfg.directions = _parse_directions(path, value)
            continue
        section = getattr(cfg, name)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: section {name!r} must be a mapping")
        bad = sorted(set(value) - {f.name for f in fields(section)})
        if bad:
            raise ValueError(f"{path}: unknown key(s) in {name!r}: {bad}")
        for k, v in value.items():
            setattr(section, k, v)
    if not cfg.checkpoint:
        raise ValueError(f"{path}: checkpoint is required")
    if not cfg.data.files:
        raise ValueError(f"{path}: data.files is empty")
    if cfg.data.fit_fraction + cfg.data.cal_fraction >= 1.0:
        raise ValueError("data.fit_fraction + data.cal_fraction must leave rows for TEST")
    ex = cfg.export
    flags = {"states": ex.states, "per_row": ex.per_row, "cal_rows": ex.cal_rows, "fit_rows": ex.fit_rows}
    bad_flags = sorted(k for k, v in flags.items() if not isinstance(v, bool))
    if bad_flags:
        raise ValueError(f"{path}: export.{', export.'.join(bad_flags)} must be true/false")
    if ex.layers is not None:
        if (not isinstance(ex.layers, list) or not ex.layers
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in ex.layers)):
            raise ValueError(f"{path}: export.layers must be null or a non-empty list of hidden-state indices")
        if cfg.capture.layers != "all":
            missing = sorted(set(ex.layers) - {int(x) for x in cfg.capture.layers})
            if missing:
                raise ValueError(f"{path}: export.layers {missing} are not in capture.layers")
    if cfg.capture.layers != "all":
        captured = {int(x) for x in cfg.capture.layers}
        for d in cfg.directions:
            if d.layer is not None and d.layer not in captured:
                raise ValueError(f"{path}: directions {d.name!r} layer {d.layer} is not in capture.layers")
    return cfg


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def stratified_sample(rows: list[DecisionExample], limit: int | None, seed: int) -> list[DecisionExample]:
    """Seeded sample keeping each task's share (corpus files are grouped by task)."""
    if not limit or len(rows) <= limit:
        return list(rows)
    keep, _ = split_by_task(rows, val_fraction=1.0 - limit / len(rows), seed=seed)
    return keep


def split_fit_cal_test(rows: list[DecisionExample], fit: float, cal: float, seed: int
                       ) -> tuple[list[DecisionExample], list[DecisionExample], list[DecisionExample]]:
    rest, fit_rows = split_by_task(rows, val_fraction=fit, seed=seed)
    test_rows, cal_rows = split_by_task(rest, val_fraction=cal / (1.0 - fit), seed=seed + 1)
    return fit_rows, cal_rows, test_rows


def ablate(ex: DecisionExample, state_text: str) -> DecisionExample:
    """The unknowable twin: same question and options, evidence removed."""
    return DecisionExample(
        kind=ex.kind, state=state_text, instructions=ex.instructions, options=ex.options,
        label=ex.label, task=ex.task, weight=ex.weight, instruction_variants=list(ex.instruction_variants),
    )


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


@dataclass
class Capture:
    logits: list[np.ndarray]           # canonical order, uncalibrated
    states: dict[int, np.ndarray]      # hidden-state index -> (n_rows, d) at <answer>


def resolve_layers(spec: Any, n_hidden_states: int) -> list[int]:
    if spec == "all":
        return list(range(n_hidden_states))
    layers = [int(x) for x in spec]
    bad = [x for x in layers if not 0 <= x < n_hidden_states]
    if bad:
        raise ValueError(f"capture.layers {bad} outside 0..{n_hidden_states - 1}")
    return layers


def capture(model: Any, rows: list[DecisionExample], *, layers: Any, batch_size: int,
            max_length: int, dtype: Any = np.float16) -> Capture:
    """One canonical forward per batch: option logits + <answer> states per layer."""
    import torch

    from .collate import CollatorConfig, DecisionCollator
    from .readouts import gather_answer

    cfg = model.decision_config
    collator = DecisionCollator(
        model.tokenizer,
        CollatorConfig(max_length=max_length, marker_style=cfg.marker_style, headers=dict(cfg.headers)),
        train=False,
    )
    device = next(model.parameters()).device
    model.eval()
    logits_out: list[np.ndarray] = []
    states: dict[int, list[np.ndarray]] = {}
    chosen: list[int] | None = None
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            batch = collator(chunk)
            t = {k: batch[k].to(device) for k in
                 ("input_ids", "attention_mask", "option_index", "answer_index", "n_options")}
            out = model._decoder()(input_ids=t["input_ids"], attention_mask=t["attention_mask"],
                                   use_cache=False, output_hidden_states=True)
            hidden_all = out.hidden_states
            if chosen is None:
                chosen = resolve_layers(layers, len(hidden_all))
                states = {i: [] for i in chosen}
            logits = model.readout_logits(out.last_hidden_state, t["option_index"], t["answer_index"],
                                          t["n_options"]).float().cpu().numpy()
            for i in chosen:
                states[i].append(gather_answer(hidden_all[i], t["answer_index"]).float().cpu().numpy().astype(dtype))
            for i, ex in enumerate(chunk):
                logits_out.append(np.asarray(logits[i, : ex.n_options], dtype=np.float64))
    return Capture(logits=logits_out, states={i: np.concatenate(v) for i, v in states.items()})


# --------------------------------------------------------------------------
# Arms + metrics
# --------------------------------------------------------------------------


def fit_kind_temperatures(logits: list[np.ndarray], labels: list[int], kinds: list[str],
                          min_rows: int) -> dict[str, float]:
    temps = {}
    for kind in KINDS:
        idx = [i for i, k in enumerate(kinds) if k == kind]
        if len(idx) >= min_rows:
            temps[kind] = fit_temperature([logits[i] for i in idx], [labels[i] for i in idx])
    return temps


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def paired_auroc_diff(labels: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    """AUROC(a) - AUROC(b) on the same rows with a paired bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    point = auroc(labels, a) - auroc(labels, b)
    diffs = []
    n = labels.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if y.min() == y.max():
            continue
        diffs.append(auroc(y, a[idx]) - auroc(y, b[idx]))
    lo, hi = (np.quantile(diffs, [0.025, 0.975]) if diffs else (float("nan"), float("nan")))
    return {"diff": point, "ci95": [float(lo), float(hi)], "n_boot_used": len(diffs)}


# --------------------------------------------------------------------------
# External frozen directions
# --------------------------------------------------------------------------

DIRECTION_SCORE_RULE = "logistic_decision_value: raw_norm * (h @ vector) + intercept  (== h @ coef + intercept)"


def load_direction(spec: DirectionSpec, path: Path, hidden_size: int | None = None) -> dict:
    """Load and sanity-check a frozen direction written by ``MechInterp.probe.fit.freeze_direction``."""
    from MechInterp.probe.fit import load_frozen_direction

    if not path.is_file():
        raise FileNotFoundError(f"direction {spec.name!r}: no file at {path}")
    rec = load_frozen_direction(path)
    if rec.get("schema_version") != DIRECTION_SCHEMA:
        raise ValueError(f"direction {spec.name!r}: {path} has schema_version {rec.get('schema_version')!r}, "
                         f"expected {DIRECTION_SCHEMA!r}")
    missing = [k for k in ("vector", "intercept") if rec.get(k) is None]
    if rec.get("normalized") and rec.get("raw_norm") is None:
        missing.append("raw_norm")
    if spec.layer is None and rec.get("layer") is None:
        missing.append("layer")
    if missing:
        raise ValueError(f"direction {spec.name!r}: {path} lacks field(s) {missing}")
    dim = int(np.asarray(rec["vector"]).size)
    if rec.get("hidden_dim") is not None and int(rec["hidden_dim"]) != dim:
        raise ValueError(f"direction {spec.name!r}: vector has {dim} dims but hidden_dim says {rec['hidden_dim']}")
    if rec.get("mu") is not None and np.asarray(rec["mu"]).size != dim:
        raise ValueError(f"direction {spec.name!r}: mu and vector dimensions differ")
    if hidden_size is not None and dim != hidden_size:
        raise ValueError(f"direction {spec.name!r}: vector dimension {dim} != the model's hidden size {hidden_size}")
    rec["resolved_layer"] = int(spec.layer if spec.layer is not None else rec["layer"])
    return rec


def score_direction(record: dict, X: np.ndarray) -> np.ndarray:
    """Score rows with a frozen direction's own logistic decision function.

    ``freeze_direction`` fits ``score(h) = h @ coef + intercept`` and stores
    ``vector = coef / raw_norm`` when ``normalized``. Undoing that scale gives
    back exactly the score its ``sigma`` and ``calibration`` class statistics
    were computed on, so per-row values are comparable with the JSON. It equals
    ``raw_norm * ((h - mu) @ vector)`` plus a constant, so AUROCs match the
    mu-centred projection on the unit vector.
    """
    vector = np.asarray(record["vector"], dtype=np.float64)
    if X.shape[1] != vector.size:
        raise ValueError(f"direction vector has {vector.size} dims, captured states have {X.shape[1]}")
    scale = float(record["raw_norm"]) if record.get("normalized") else 1.0
    return scale * (X.astype(np.float64) @ vector) + float(record["intercept"])


def score_directions(directions: list[tuple[DirectionSpec, dict]], states: dict[int, np.ndarray]
                     ) -> dict[str, np.ndarray]:
    """Score captured <answer> states along each loaded direction; fail loudly on layer / size mismatch."""
    out: dict[str, np.ndarray] = {}
    for spec, rec in directions:
        layer = rec["resolved_layer"]
        if layer not in states:
            raise ValueError(f"direction {spec.name!r}: hidden-state index {layer} was not captured "
                             f"(captured: {sorted(states)}; set capture.layers or directions[].layer)")
        dim, size = states[layer].shape[1], int(np.asarray(rec["vector"]).size)
        if size != dim:
            raise ValueError(f"direction {spec.name!r}: vector dimension {size} != the model's hidden size {dim}")
        out[spec.name] = score_direction(rec, states[layer])
    return out


def _moments(x: np.ndarray) -> dict:
    return {"n": int(x.size), "mean": float(x.mean()) if x.size else float("nan"),
            "std": float(x.std()) if x.size else float("nan")}


def direction_block(score: np.ndarray, record: dict, spec: DirectionSpec, ok_t: np.ndarray, r1: np.ndarray,
                    knowledge: list[Any], boot: BootstrapSection) -> dict:
    """AUROCs and class-conditioned moments of one external direction's score on TEST rows."""
    block: dict[str, Any] = {
        "path": spec.path,
        "layer": record["resolved_layer"],
        "source_layer": record.get("layer"),
        "score_rule": DIRECTION_SCORE_RULE,
        "n": int(score.size),
        "auroc_correct": auroc(ok_t, score),
        "auroc_correct_minus_r1": paired_auroc_diff(ok_t, score, r1, boot.n_boot, boot.seed),
        "by_correctness": {"correct": _moments(score[ok_t == 1]), "wrong": _moments(score[ok_t == 0])},
    }
    ku = np.array([i for i, lab in enumerate(knowledge) if lab in ("known", "unknown")], dtype=int)
    if ku.size:
        y_ku = np.array([1 if knowledge[i] == "known" else 0 for i in ku])
        block["auroc_known_vs_unknown"] = auroc(y_ku, score[ku])
        block["auroc_known_vs_unknown_minus_r1"] = paired_auroc_diff(y_ku, score[ku], r1[ku], boot.n_boot,
                                                                     boot.seed)
        block["by_knowledge"] = {name: _moments(score[[i for i, lab in enumerate(knowledge) if lab == name]])
                                 for name in ("known", "unknown", "ambiguous")
                                 if any(lab == name for lab in knowledge)}
    return block


def export_arrays(states: dict[int, np.ndarray], layers: Any, n_rows: int) -> dict[str, np.ndarray]:
    """test_states.npz payload: ``L{i}`` (n_rows, d) float16 per hidden-state index, plus ``row_index``."""
    chosen = sorted(states) if layers is None else [int(x) for x in layers]
    missing = [i for i in chosen if i not in states]
    if missing:
        raise ValueError(f"export.layers {missing} were not captured (captured: {sorted(states)})")
    out = {f"L{i}": states[i].astype(np.float16) for i in chosen}
    out["row_index"] = np.arange(n_rows, dtype=np.int64)
    return out


def confidence_block(conf: np.ndarray, correct: np.ndarray, probs_for_ece: list[np.ndarray],
                     labels: list[int], th: ThresholdSection, boot: BootstrapSection) -> dict:
    from MechInterp.stats.gates import auroc_floor

    block = {
        "auroc": auroc(correct, conf),
        "aurc": aurc(conf, correct),
        "confident_wrong_rate": confident_wrong_rate(conf, correct, th.confident),
        "underconfident_right_rate": underconfident_right_rate(conf, correct, th.underconfident),
        "mean_confidence": float(conf.mean()),
    }
    if correct.min() != correct.max():
        floor = auroc_floor(correct.tolist(), conf.tolist(), seed=boot.seed, n_boot=boot.n_boot)
        block["auroc_ci_lb"] = floor["ci_lb"]
        block["auroc_se"] = floor["hanley_mcneil_se"]
    if probs_for_ece is not None:
        block.update({f"option_{k}": v for k, v in calibration_report(probs_for_ece, labels, th.ece_bins)
                      .as_dict().items() if k != "n"})
    else:
        # Binary confidence-of-correctness calibration (probe / stack arms).
        block["ece_binary"] = expected_calibration_error(
            [[1 - c, c] for c in conf], correct.astype(int).tolist(), th.ece_bins)
    return block


def run_probes(fit_states: dict[int, np.ndarray], fit_y: np.ndarray, cfg: ProbeSection, seed: int) -> dict:
    from MechInterp.probe.fit import cv_auroc, fit_full_probe, sweep_layers

    surface = sweep_layers({k: v.astype(np.float64) for k, v in fit_states.items()}, fit_y,
                           n_components=cfg.n_components, n_splits=cfg.n_splits, seed=seed,
                           C=cfg.C, max_iter=cfg.max_iter)
    best = int(surface["best_layer"])
    probe = fit_full_probe(fit_states[best].astype(np.float64), fit_y, n_components=cfg.n_components,
                           seed=seed, C=cfg.C, max_iter=cfg.max_iter)
    out = {"auroc_by_layer": {int(k): float(v) for k, v in surface["auroc_by_layer"].items()},
           "best_layer": best, "probe": probe}
    if cfg.permuted_control:
        rng = np.random.default_rng(seed)
        perm_auc, _, _ = cv_auroc(fit_states[best].astype(np.float64), rng.permutation(fit_y),
                                  n_components=cfg.n_components, n_splits=cfg.n_splits, seed=seed,
                                  C=cfg.C, max_iter=cfg.max_iter)
        out["permuted_label_cv_auroc"] = float(perm_auc)
    return out


def conformal_block(cal_probs: list[np.ndarray], cal_labels: list[int], cal_kinds: list[str],
                    test_sets: dict[str, tuple[list[np.ndarray], list[int], list[str]]],
                    cfg: ConformalSection) -> dict:
    out: dict[str, Any] = {}
    for alpha in cfg.alphas:
        qhat_all = conformal_lac_qhat([p[y] for p, y in zip(cal_probs, cal_labels)], alpha)
        qhat = {}
        for kind in KINDS:
            idx = [i for i, k in enumerate(cal_kinds) if k == kind]
            qhat[kind] = (conformal_lac_qhat([cal_probs[i][cal_labels[i]] for i in idx], alpha)
                          if len(idx) >= cfg.min_rows_per_kind else qhat_all)
        res = {"qhat_by_kind": qhat, "target_coverage": 1 - alpha}
        for name, (probs, labels, kinds) in test_sets.items():
            covered, sizes, rel = [], [], []
            for p, y, k in zip(probs, labels, kinds):
                s = lac_set(p, qhat[k])
                covered.append(y in s)
                sizes.append(len(s))
                rel.append(len(s) / len(p))
            res[name] = {"coverage": float(np.mean(covered)), "mean_set_size": float(np.mean(sizes)),
                         "mean_set_fraction_of_options": float(np.mean(rel))}
        out[f"alpha_{alpha}"] = res
    return out


def ordinal_interval_block(probs: list[np.ndarray], labels: list[int], kinds: list[str],
                           alphas: list[float]) -> dict:
    idx = [i for i, k in enumerate(kinds) if k == "score"]
    if not idx:
        return {}
    out = {}
    for alpha in alphas:
        cov, width = [], []
        for i in idx:
            lo, hi = shortest_ordinal_interval(probs[i], 1 - alpha)
            cov.append(lo <= labels[i] <= hi)
            width.append(hi - lo + 1)
        out[f"mass_{1 - alpha:.2f}"] = {"coverage": float(np.mean(cov)), "mean_width_levels": float(np.mean(width)),
                                        "n": len(idx)}
    return out


def analyze(model: Any, cfg: ConfidenceAnalysisConfig, repo_root: Path
            ) -> tuple[dict, dict[str, list[dict]], dict[str, np.ndarray]]:
    """Run every confidence arm.

    Returns (report, per-row records by split, test_states.npz arrays -- empty
    unless ``export.states``). Records always include ``"test"``; ``"cal"`` /
    ``"fit"`` are added by ``export.cal_rows`` / ``export.fit_rows``. Every
    record carries ``split``; ``row_index`` i is row i of that split (for TEST
    also the npz ``row_index`` entry that holds its states). All splits use the
    same CAL-fit temperatures, probes and stacker; on FIT the probe scores are
    in-sample (the probes were fit there).
    """
    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else repo_root / q

    # Load external directions before any forward pass so a bad file fails fast.
    hidden_size = _hidden_size(model)
    directions = [(d, load_direction(d, resolve(d.path), hidden_size)) for d in cfg.directions]

    rows = stratified_sample(load_examples([resolve(f) for f in cfg.data.files]), cfg.data.max_rows, cfg.data.seed)
    fit_rows, cal_rows, test_rows = split_fit_cal_test(rows, cfg.data.fit_fraction, cfg.data.cal_fraction,
                                                       cfg.data.seed)
    max_length = cfg.capture.max_length or model.decision_config.max_length
    cap = lambda rs: capture(model, rs, layers=cfg.capture.layers,  # noqa: E731
                             batch_size=cfg.capture.batch_size, max_length=max_length)
    splits = {"fit": cap(fit_rows), "cal": cap(cal_rows), "test": cap(test_rows)}
    if cfg.ablation.enabled:
        for name, rs in (("fit", fit_rows), ("cal", cal_rows), ("test", test_rows)):
            splits[f"{name}_ablated"] = cap([ablate(r, cfg.ablation.state_text) for r in rs])

    def facts(rs: list[DecisionExample], c: Capture) -> dict:
        labels = [r.label for r in rs]
        pred = [int(np.argmax(lg)) for lg in c.logits]
        return {"labels": labels, "kinds": [r.kind for r in rs], "pred": pred,
                "correct": np.array([int(p == y) for p, y in zip(pred, labels)])}

    F = {"fit": facts(fit_rows, splits["fit"]), "cal": facts(cal_rows, splits["cal"]),
         "test": facts(test_rows, splits["test"])}
    th, boot = cfg.thresholds, cfg.bootstrap

    # R1 temperatures from CAL real rows only.
    temps = fit_kind_temperatures(splits["cal"].logits, F["cal"]["labels"], F["cal"]["kinds"],
                                  cfg.conformal.min_rows_per_kind)
    probs = {name: [softmax(lg, temps.get(k, 1.0)) for lg, k in
                    zip(splits[name].logits, (F[name.replace("_ablated", "")]["kinds"]))]
             for name in splits}
    raw = {name: [softmax(lg) for lg in splits[name].logits] for name in splits}

    test = F["test"]
    y_t, k_t, ok_t = test["labels"], test["kinds"], test["correct"]
    r0 = np.array([p.max() for p in raw["test"]])
    r1 = np.array([p.max() for p in probs["test"]])

    # P-dial: probe of own correctness on FIT real rows.
    dial = run_probes(splits["fit"].states, F["fit"]["correct"], cfg.probe, cfg.data.seed)
    best = dial["best_layer"]
    dial_score = {n: _score(dial["probe"], splits[n].states[best]) for n in ("fit", "cal", "test")}
    p_dial_test = _sigmoid(dial_score["test"])

    # S: stack logit(R1) + dial score, fitted on CAL.
    from sklearn.linear_model import LogisticRegression

    r1_cal = np.array([p.max() for p in probs["cal"]])
    stack = LogisticRegression(C=1.0, max_iter=1000).fit(
        np.column_stack([_logit(r1_cal), dial_score["cal"]]), F["cal"]["correct"])
    stack_X_test = np.column_stack([_logit(r1), dial_score["test"]])
    p_stack = stack.predict_proba(stack_X_test)[:, 1]

    report: dict[str, Any] = {
        "schema": "decision-confidence-analysis/v1",
        "checkpoint": cfg.checkpoint,
        "n_rows": {"fit": len(fit_rows), "cal": len(cal_rows), "test": len(test_rows)},
        "abstention_rate": 0.0,
        "test_accuracy": float(ok_t.mean()),
        "test_accuracy_by_task": _by_group(ok_t, [r.task for r in test_rows]),
        "temperatures_cal": temps,
        "arms": {
            "R0_raw": confidence_block(r0, ok_t, raw["test"], y_t, th, boot),
            "R1_calibrated": confidence_block(r1, ok_t, probs["test"], y_t, th, boot),
            "P_dial": confidence_block(p_dial_test, ok_t, None, y_t, th, boot),
            "S_stacked": confidence_block(p_stack, ok_t, None, y_t, th, boot),
        },
        "probe_dial": {k: v for k, v in dial.items() if k != "probe"},
        "h2_dial_minus_r1": paired_auroc_diff(ok_t, dial_score["test"], r1, boot.n_boot, boot.seed),
        "h2_stack_minus_r1": paired_auroc_diff(ok_t, p_stack, r1, boot.n_boot, boot.seed),
        "stack_coef": {"logit_r1": float(stack.coef_[0][0]), "dial_score": float(stack.coef_[0][1]),
                       "intercept": float(stack.intercept_[0])},
        "conformal": conformal_block(
            probs["cal"], F["cal"]["labels"], F["cal"]["kinds"],
            {"test_real": (probs["test"], y_t, k_t),
             **({"test_ablated": (probs["test_ablated"], y_t, k_t)} if cfg.ablation.enabled else {})},
            cfg.conformal),
        "ordinal_intervals_test_real": ordinal_interval_block(probs["test"], y_t, k_t, cfg.conformal.alphas),
    }

    if cfg.ablation.enabled:
        abl = probs["test_ablated"]
        maxp = np.array([p.max() for p in abl])
        chance = np.array([1.0 / len(p) for p in abl])
        pred_abl = [int(np.argmax(p)) for p in abl]
        report["humility_ablated"] = {
            "humility_gap": float((maxp - chance).mean()),
            "mean_max_prob": float(maxp.mean()),
            "mean_chance": float(chance.mean()),
            "share_confident": float((maxp >= th.confident).mean()),
            "accuracy_on_ablated": float(np.mean([p == y for p, y in zip(pred_abl, y_t)])),
            "real_minus_ablated_mean_confidence": float(r1.mean() - maxp.mean()),
            "ordinal_intervals": ordinal_interval_block(abl, y_t, k_t, cfg.conformal.alphas),
        }
        gate_X = {i: np.concatenate([splits["fit"].states[i], splits["fit_ablated"].states[i]])
                  for i in splits["fit"].states}
        gate_y = np.concatenate([np.ones(len(fit_rows), int), np.zeros(len(fit_rows), int)])
        gate = run_probes(gate_X, gate_y, cfg.probe, cfg.data.seed)
        gs = np.concatenate([_score(gate["probe"], splits["test"].states[gate["best_layer"]]),
                             _score(gate["probe"], splits["test_ablated"].states[gate["best_layer"]])])
        report["probe_gate"] = {**{k: v for k, v in gate.items() if k != "probe"},
                                "test_auroc_real_vs_ablated": auroc(
                                    np.concatenate([np.ones(len(test_rows)), np.zeros(len(test_rows))]), gs)}
        # Does the readout itself separate real from ablated? (confidence as a gate)
        report["readout_as_gate_auroc"] = auroc(
            np.concatenate([np.ones(len(test_rows)), np.zeros(len(test_rows))]), np.concatenate([r1, maxp]))

    knowledge, ku_probe = knowledge_block(fit_rows, test_rows, splits, r1, dial_score["test"], ok_t, cfg)
    if knowledge:
        report["knowledge"] = knowledge

    test_states = splits["test"].states
    direction_scores = score_directions(directions, test_states)
    if directions:
        know_t = [r.meta.get("knowledge") if r.meta else None for r in test_rows]
        report["directions"] = {spec.name: direction_block(direction_scores[spec.name], rec, spec, ok_t, r1,
                                                           know_t, boot)
                                for spec, rec in directions}

    arrays = export_arrays(test_states, cfg.export.layers, len(test_rows)) if cfg.export.states else {}

    split_rows = {"fit": fit_rows, "cal": cal_rows, "test": test_rows}

    def split_records(name: str) -> list[dict]:
        """Per-row records for one split, scored exactly as TEST (CAL temperatures, FIT probes, CAL stacker)."""
        c_r0 = np.array([p.max() for p in raw[name]])
        c_r1 = np.array([p.max() for p in probs[name]])
        X = np.column_stack([_logit(c_r1), dial_score[name]])
        p_st, s_st = stack.predict_proba(X)[:, 1], stack.decision_function(X)
        p_di = _sigmoid(dial_score[name])
        ok = F[name]["correct"]
        dirs = direction_scores if name == "test" else score_directions(directions, splits[name].states)
        ku = (None if ku_probe is None
              else _score(ku_probe["probe"], splits[name].states[ku_probe["best_layer"]]))
        out = []
        for i, r in enumerate(split_rows[name]):
            rec = {"split": name, "row_index": i, "task": r.task, "kind": r.kind, "n_options": r.n_options,
                   "gold": r.label, "pred": F[name]["pred"][i], "correct": int(ok[i]), "conf_r0": float(c_r0[i]),
                   "conf_r1": float(c_r1[i]), "p_dial": float(p_di[i]), "p_stack": float(p_st[i])}
            if cfg.ablation.enabled:
                rec["ablated_max_prob"] = float(probs[f"{name}_ablated"][i].max())
                rec["ablated_pred"] = int(np.argmax(probs[f"{name}_ablated"][i]))
            if cfg.export.per_row:
                rec["probs_r0"] = [float(x) for x in raw[name][i]]      # canonical option order
                rec["probs_r1"] = [float(x) for x in probs[name][i]]
                rec["dial_score"] = float(dial_score[name][i])          # probe decision value (logit)
                rec["stack_score"] = float(s_st[i])                     # stacker decision value (logit)
                rec["ku_probe_score"] = None if ku is None else float(ku[i])
                rec["direction_scores"] = {n: float(v[i]) for n, v in dirs.items()}
            if r.meta:
                rec["meta"] = r.meta
            out.append(rec)
        return out

    exported = ["test"] + [n for n, on in (("cal", cfg.export.cal_rows), ("fit", cfg.export.fit_rows)) if on]
    records = {name: split_records(name) for name in exported}
    return report, records, arrays


def knowledge_block(fit_rows: list[DecisionExample], test_rows: list[DecisionExample],
                    splits: dict[str, Capture], r1: np.ndarray, dial_test: np.ndarray,
                    ok_t: np.ndarray, cfg: ConfidenceAnalysisConfig) -> tuple[dict, dict | None]:
    """Confidence vs a *prior* knowledge label on each row.

    Returns (block, the known-vs-unknown probe -- ``run_probes`` output with
    ``probe`` and ``best_layer`` -- or None when that probe was not fit).

    Present only when rows carry ``meta.knowledge`` (``known`` / ``unknown`` /
    ``ambiguous``), supplied by an external labeling protocol that records
    whether the base model already knew the answer before decision training.
    A well-calibrated decision model is confident and right on ``known`` rows,
    and still chooses, but with confidence near chance, on ``unknown`` rows.
    """
    labels = [r.meta.get("knowledge") for r in test_rows]
    if not any(lab in ("known", "unknown") for lab in labels):
        return {}, None
    th = cfg.thresholds
    chance = np.array([1.0 / r.n_options for r in test_rows])
    groups = {}
    for name in ("known", "unknown", "ambiguous"):
        idx = np.array([i for i, lab in enumerate(labels) if lab == name], dtype=int)
        if idx.size == 0:
            continue
        c, ok = r1[idx], ok_t[idx]
        groups[name] = {
            "n": int(idx.size),
            "accuracy": float(ok.mean()),
            "mean_confidence": float(c.mean()),
            "mean_chance": float(chance[idx].mean()),
            "humility_gap": float((c - chance[idx]).mean()),
            "confident_rate": float((c >= th.confident).mean()),
            "confident_wrong_rate": confident_wrong_rate(c, ok, th.confident),
            "underconfident_right_rate": underconfident_right_rate(c, ok, th.underconfident),
            "ece_binary": expected_calibration_error([[1 - x, x] for x in c], ok.astype(int).tolist(),
                                                     th.ece_bins),
        }
    out: dict[str, Any] = {"groups": groups}
    ku_fitted: dict | None = None

    ku = np.array([i for i, lab in enumerate(labels) if lab in ("known", "unknown")], dtype=int)
    y_ku = np.array([1 if labels[i] == "known" else 0 for i in ku])
    if y_ku.min() != y_ku.max():
        out["readout_auroc_known_vs_unknown"] = auroc(y_ku, r1[ku])
        out["dial_auroc_known_vs_unknown"] = auroc(y_ku, dial_test[ku])
        # Known-vs-unknown (KU) linear probe on the <answer> state, fitted on FIT rows only.
        fit_lab = [r.meta.get("knowledge") for r in fit_rows]
        fit_idx = np.array([i for i, lab in enumerate(fit_lab) if lab in ("known", "unknown")], dtype=int)
        fit_y = np.array([1 if fit_lab[i] == "known" else 0 for i in fit_idx])
        if fit_idx.size >= 20 and fit_y.min() != fit_y.max():
            fit_states = {k: v[fit_idx] for k, v in splits["fit"].states.items()}
            ku_probe = run_probes(fit_states, fit_y, cfg.probe, cfg.data.seed)
            ku_fitted = ku_probe
            ku_score = _score(ku_probe["probe"], splits["test"].states[ku_probe["best_layer"]])[ku]
            out["ku_probe"] = {k: v for k, v in ku_probe.items() if k != "probe"}
            out["ku_probe"]["test_auroc_known_vs_unknown"] = auroc(y_ku, ku_score)
            out["ku_probe_minus_readout"] = paired_auroc_diff(y_ku, ku_score, r1[ku], cfg.bootstrap.n_boot,
                                                              cfg.bootstrap.seed)

    pops = [r.meta.get("s_pop") for r in test_rows]
    if all(p is not None for p in pops):
        pops_arr = np.array(pops, dtype=float)
        edges = np.quantile(pops_arr, [0.25, 0.5, 0.75])
        bins = np.digitize(pops_arr, edges)
        out["by_popularity_quartile"] = {
            f"q{b + 1}": {"n": int((bins == b).sum()),
                          "s_pop_range": [float(pops_arr[bins == b].min()), float(pops_arr[bins == b].max())],
                          "accuracy": float(ok_t[bins == b].mean()),
                          "mean_confidence": float(r1[bins == b].mean()),
                          "known_share": float(np.mean([labels[i] == "known" for i in np.flatnonzero(bins == b)]))}
            for b in range(4) if (bins == b).any()
        }
    return out, ku_fitted


def _hidden_size(model: Any) -> int | None:
    try:
        return int(model._inner().config.hidden_size)
    except (AttributeError, TypeError, ValueError):
        return None


def _score(probe: dict, X: np.ndarray) -> np.ndarray:
    from MechInterp.probe.fit import score_full_probe

    return score_full_probe(probe, X.astype(np.float64))


def _by_group(correct: np.ndarray, groups: list[str]) -> dict[str, float]:
    out: dict[str, list[int]] = {}
    for c, g in zip(correct.tolist(), groups):
        out.setdefault(g, []).append(c)
    return {g: float(np.mean(v)) for g, v in sorted(out.items())}
