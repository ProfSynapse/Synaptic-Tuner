"""
Steer cell orchestrator: the six-block declarative intervention run.

This module holds the lane-agnostic logic that does not need a GPU or a model:
row loading, arm resolution (including the seeded permuted control and the
dose ladder), resume-from-output skipping, config-sha computation, and the smoke
state file that gates the full arms. The model-facing execution (loading the
model, registering the hook, generating) is left to the CLI layer so this module
stays importable and testable on CPU.

Arm resolution turns each ArmConfig into a per-row strength map: row_key -> value.
A row absent from the map is an untouched no-op. The baseline arm maps every row
to zero. A permuted control draws a count-matched set of rows uniformly at random
from a seeded generator, matching the count of the arm it controls, so it probes
the same dose on a different, randomly selected population.

A gain_field arm (continuous per-row coupling) maps every row carrying that field
to strength * row[gain_field], optionally clipped to +/- gain_clip. A permuted
control of a gain arm keeps the same row population but SHUFFLES the computed
gains across it (seeded), so the gain distribution is identical and only the
row-to-gain pairing is scrambled -- the placebo for a continuous coupling, as
opposed to the count-matched-random-subset placebo used for selection arms.

Whether a resolved value of exactly 0.0 counts as "active" (the law is applied,
writing a zero setpoint) or as a no-op depends on the caller: pending_rows takes
an explicit write_at_zero flag so a fixed-strength "ablate" arm (force_active on
its ArmConfig) can be distinguished from the "baseline" no-op, and a gain arm's
own rows are always active regardless of their computed value (a gain of exactly
0.0 for one row is a real coupling output, not an absence of coupling).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from MechInterp.config import ArmConfig, SteerCellConfig


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key_of(row: dict) -> str:
    for k in ("row_key", "id", "key"):
        if k in row:
            return str(row[k])
    raise KeyError("row has no row_key / id / key field")


def compute_config_sha(config: SteerCellConfig) -> str:
    """Deterministic sha256 over the canonicalized config (readouts by path).

    surface.expected_config_sha is excluded from the payload before hashing:
    that field holds the expected value of THIS hash, so including it would
    make the hash a function of itself -- filling it in would shift the
    computed sha out from under it, and the "expected == computed" guard in
    run_steer could never be satisfied. Every other field is content, and
    stays in.
    """
    payload = config.model_dump(mode="json")
    payload.get("surface", {}).pop("expected_config_sha", None)
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _active_keys_for_arm(arm: ArmConfig, rows: list[dict]) -> list[str]:
    """Return the row keys an arm activates, in row order."""
    keys = [row_key_of(r) for r in rows]
    if arm.flag_field is not None:
        return [row_key_of(r) for r in rows if bool(r.get(arm.flag_field))]
    if arm.score_field is not None:
        out = []
        for r in rows:
            val = r.get(arm.score_field)
            if val is not None and float(val) >= float(arm.threshold):
                out.append(row_key_of(r))
        return out
    # fixed-strength arm (baseline or uniform dose): every row is active
    return keys


def _gain_values_for_arm(arm: ArmConfig, rows: list[dict]) -> dict[str, float]:
    """Per-row continuous gain for a gain_field arm: strength * row[gain_field],
    optionally clipped to +/- gain_clip.

    A row missing the gain_field is not selected (absent from the returned
    map), mirroring score_field's "no value, no selection" convention. A row
    present but whose computed gain lands at exactly 0.0 IS selected -- the
    coupling law is applied at that row with a zero setpoint, per the couple
    mechanism (erase the projection, write gain*sigma even when gain is 0).
    """
    out: dict[str, float] = {}
    clip = abs(float(arm.gain_clip)) if arm.gain_clip is not None else None
    for r in rows:
        val = r.get(arm.gain_field)
        if val is None:
            continue
        g = float(arm.strength) * float(val)
        if clip is not None:
            g = max(-clip, min(clip, g))
        out[row_key_of(r)] = g
    return out


def resolve_arm_strengths(
    arm: ArmConfig,
    rows: list[dict],
    arm_by_name: dict[str, ArmConfig],
) -> dict[str, float]:
    """Return a row_key -> strength map for one arm.

    Fixed-strength arms map every row to arm.strength (baseline uses 0).
    Selection arms map only their active rows to arm.strength.
    A gain_field arm maps every row carrying the field to its own continuous
    value (see _gain_values_for_arm).
    A permuted control of a SELECTION arm draws a seeded, count-matched random
    subset of all rows, matching the count of the arm it controls, all at
    arm.strength. A permuted control of a GAIN arm instead shuffles the
    controlled arm's per-row gains across the same row population (seeded),
    preserving the gain distribution while scrambling the row pairing.
    """
    keys_in_order = [row_key_of(r) for r in rows]

    if arm.permuted_control_of is not None:
        controlled = arm_by_name[arm.permuted_control_of]

        if controlled.gain_field is not None:
            gains = _gain_values_for_arm(controlled, rows)
            ordered_keys = [k for k in keys_in_order if k in gains]
            values = [gains[k] for k in ordered_keys]
            rng = np.random.default_rng(arm.control_seed)
            perm = rng.permutation(len(values))
            shuffled = [values[i] for i in perm]
            return {k: float(v) for k, v in zip(ordered_keys, shuffled)}

        controlled_keys = _active_keys_for_arm(controlled, rows)
        n = len(controlled_keys)
        rng = np.random.default_rng(arm.control_seed)
        if n > len(keys_in_order):
            raise ValueError(
                f"arm {arm.name}: controlled count {n} exceeds row count"
            )
        chosen_idx = rng.choice(len(keys_in_order), size=n, replace=False)
        chosen = sorted(keys_in_order[i] for i in chosen_idx)
        strength = arm.strength if arm.strength != 0.0 else controlled.strength
        return {k: float(strength) for k in chosen}

    if arm.gain_field is not None:
        return _gain_values_for_arm(arm, rows)

    active = _active_keys_for_arm(arm, rows)
    return {k: float(arm.strength) for k in active}


def resolve_all_arms(config: SteerCellConfig, rows: list[dict]) -> dict[str, dict[str, float]]:
    """Return {arm_name: {row_key: strength}} for every arm in the config."""
    arm_by_name = {a.name: a for a in config.arms}
    return {
        a.name: resolve_arm_strengths(a, rows, arm_by_name) for a in config.arms
    }


def completed_keys(output_path: str | Path, arm: str) -> set[str]:
    """Return the set of (arm, row_key) already present in the output JSONL."""
    path = Path(output_path)
    if not path.exists():
        return set()
    done = set()
    for rec in load_jsonl(path):
        if rec.get("arm") == arm:
            done.add(str(rec.get("row_key")))
    return done


def pending_rows(
    rows: list[dict],
    strengths: dict[str, float],
    arm: str,
    output_path: str | Path,
    resume: bool,
    write_at_zero: bool = False,
) -> list[dict]:
    """Rows to run for an arm: those with a strength assignment, minus completed.

    A row with no strength assignment is a no-op and still runs (recorded with
    strength 0) so the baseline population is complete; selection arms only
    include their active rows plus all rows at strength 0 for the record. We keep
    every row in the pass so downstream gate arrays are aligned, tagging each with
    its resolved strength.

    _active marks whether the law should actually be applied to a row: a row
    outside the strengths map is never active. A row inside the map is active
    if its resolved value is nonzero, OR if write_at_zero is set -- the caller's
    way of saying "this arm applies at zero too" (the ablate / apply-zero case,
    as opposed to a baseline arm that happens to resolve every row to 0.0 and
    should stay a true no-op).
    """
    done = completed_keys(output_path, arm) if resume else set()
    pending = []
    for r in rows:
        k = row_key_of(r)
        if k in done:
            continue
        rr = dict(r)
        selected = k in strengths
        val = float(strengths.get(k, 0.0))
        rr["_strength"] = val
        rr["_active"] = selected and (val != 0.0 or write_at_zero)
        pending.append(rr)
    return pending


# --------------------------------------------------------------------------
# Smoke state file
# --------------------------------------------------------------------------


def smoke_state_path(output_path: str | Path) -> Path:
    p = Path(output_path)
    return p.with_suffix(p.suffix + ".smoke_ok.json")


def smoke_passed(output_path: str | Path, config_sha: str) -> bool:
    """True if a smoke pass for this exact config sha has been recorded."""
    sp = smoke_state_path(output_path)
    if not sp.exists():
        return False
    try:
        with open(sp) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return bool(state.get("passed")) and state.get("config_sha") == config_sha


def record_smoke(output_path: str | Path, config_sha: str, readback: dict) -> None:
    sp = smoke_state_path(output_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(
            {"passed": bool(readback.get("passed")), "config_sha": config_sha, "readback": readback},
            f,
            indent=2,
        )


def evaluate_smoke_readback(readback: dict, smoke_cfg) -> dict:
    """Judge a readback dict against smoke tolerances.

    readback carries commanded, measured, and offtarget_abs_max as produced by
    the intervention hook. For erase_write, the write check compares measured to
    commanded; for additive there is no commanded projection, so only the
    off-target parity check applies.
    """
    commanded = [c for c in readback.get("commanded", []) if c is not None]
    measured = readback.get("measured", [])
    offtarget = float(readback.get("offtarget_abs_max", 0.0))

    write_ok = True
    max_err = 0.0
    if commanded:
        errs = [abs(m - c) for m, c in zip(measured, commanded)]
        max_err = max(errs) if errs else 0.0
        mean_cmd = sum(abs(c) for c in commanded) / len(commanded)
        tol = smoke_cfg.write_rel_tol * mean_cmd if mean_cmd else smoke_cfg.write_abs_floor
        write_ok = max_err <= max(tol, smoke_cfg.write_abs_floor)

    parity_ok = offtarget <= smoke_cfg.offtarget_tol
    return {
        "passed": bool(write_ok and parity_ok),
        "write_ok": bool(write_ok),
        "parity_ok": bool(parity_ok),
        "max_write_error": max_err,
        "offtarget_abs_max": offtarget,
    }


def write_manifest(
    output_path: str | Path,
    config: SteerCellConfig,
    config_sha: str,
    arm_summaries: dict,
) -> Path:
    """Write a run manifest next to the output JSONL."""
    p = Path(output_path)
    manifest_path = p.with_suffix(p.suffix + ".manifest.json")
    manifest = {
        "config_sha": config_sha,
        "law": config.law.model_dump(mode="json"),
        "readouts": [r.model_dump(mode="json") for r in config.readouts],
        "arms": arm_summaries,
        "surface": config.surface.model_dump(mode="json"),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
