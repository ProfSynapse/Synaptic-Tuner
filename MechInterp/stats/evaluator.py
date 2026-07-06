"""
Declarative gate evaluator over per-row cell provenance.

A gates.yaml declares a list of named gates. Each gate names a primitive and its
inputs, expressed as filters over the per-row output the cell produced. The
evaluator loads the rows, computes each gate's inputs, calls the primitive, and
reports pass/fail against the declared threshold. This keeps the pass/fail policy
in configuration rather than code.

gates.yaml shape:

    seed: 20240101            # default seed for primitives that take one
    n_boot: 1000
    gates:
      - name: reach
        primitive: count_flips
        arm: primary          # which arm's rows to read
        before: baseline_positive   # per-row boolean field before intervention
        after: positive             # per-row boolean field after intervention
        from_state: true
        to_state: false
        pass_if: ">= 5"       # comparison against the primitive's scalar result

      - name: specificity
        primitive: kill_diff_vs_control
        primary_indicator: killed     # 0/1 per-row field in the primary arm
        control_indicator: killed     # 0/1 per-row field in the control arm
        pass_if_diff: ">= 5"
        pass_if_ci_excludes_zero: true

Rows are grouped by an "arm" field so a single per-row JSONL holds every arm.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from MechInterp.stats.gates import (
    count_flips,
    kill_diff_vs_control,
    permutation_p,
    auroc_floor,
)

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def _parse_comparison(expr: str):
    """Parse a "OP value" comparison string into (op_fn, threshold)."""
    parts = expr.strip().split()
    if len(parts) != 2 or parts[0] not in _OPS:
        raise ValueError(f"bad comparison expression {expr!r} (want 'OP value')")
    return _OPS[parts[0]], float(parts[1])


def _rows_for_arm(rows: list[dict], arm: Optional[str], arm_field: str) -> list[dict]:
    if arm is None:
        return rows
    return [r for r in rows if r.get(arm_field) == arm]


def _field(row: dict, name: str):
    if name not in row:
        raise KeyError(f"row is missing field {name!r}")
    return row[name]


def _eval_count_flips(gate: dict, rows: list[dict], arm_field: str) -> dict:
    arm_rows = _rows_for_arm(rows, gate.get("arm"), arm_field)
    before = [bool(_field(r, gate["before"])) for r in arm_rows]
    after = [bool(_field(r, gate["after"])) for r in arm_rows]
    result = count_flips(
        before,
        after,
        from_state=bool(gate.get("from_state", True)),
        to_state=bool(gate.get("to_state", False)),
    )
    op_fn, thr = _parse_comparison(gate["pass_if"])
    return {"value": result, "passed": bool(op_fn(result, thr)), "threshold": thr}


def _eval_kill_diff(
    gate: dict, rows: list[dict], arm_field: str, seed: int, n_boot: int
) -> dict:
    primary_rows = _rows_for_arm(rows, gate.get("primary_arm", "primary"), arm_field)
    control_rows = _rows_for_arm(rows, gate.get("control_arm", "control"), arm_field)
    p_ind = [int(_field(r, gate["primary_indicator"])) for r in primary_rows]
    c_ind = [int(_field(r, gate["control_indicator"])) for r in control_rows]
    # align lengths over a shared universe by padding the shorter with zeros
    n = max(len(p_ind), len(c_ind))
    p_ind += [0] * (n - len(p_ind))
    c_ind += [0] * (n - len(c_ind))
    stats = kill_diff_vs_control(
        p_ind, c_ind, seed=gate.get("seed", seed), n_boot=gate.get("n_boot", n_boot)
    )
    passed = True
    if "pass_if_diff" in gate:
        op_fn, thr = _parse_comparison(gate["pass_if_diff"])
        passed = passed and bool(op_fn(stats["diff"], thr))
    if gate.get("pass_if_ci_excludes_zero", False):
        passed = passed and (stats["ci_lo"] > 0)
    return {"value": stats, "passed": passed}


def _eval_permutation_p(
    gate: dict, rows: list[dict], arm_field: str, seed: int, n_perm: int
) -> dict:
    pool_rows = _rows_for_arm(rows, gate.get("pool_arm"), arm_field)
    primary_rows = _rows_for_arm(rows, gate.get("primary_arm", "primary"), arm_field)
    pool_ind = [bool(_field(r, gate["indicator"])) for r in pool_rows]
    primary_positive = sum(bool(_field(r, gate["indicator"])) for r in primary_rows)
    stats = permutation_p(
        primary_positive,
        pool_ind,
        n_primary=len(primary_rows),
        seed=gate.get("seed", seed),
        n_perm=gate.get("n_perm", n_perm),
    )
    op_fn, thr = _parse_comparison(gate.get("pass_if_p", "<= 0.05"))
    return {"value": stats, "passed": bool(op_fn(stats["p_value"], thr))}


def _eval_auroc_floor(
    gate: dict, rows: list[dict], arm_field: str, seed: int, n_boot: int
) -> dict:
    arm_rows = _rows_for_arm(rows, gate.get("arm"), arm_field)
    labels = [int(_field(r, gate["label"])) for r in arm_rows]
    scores = [float(_field(r, gate["score"])) for r in arm_rows]
    stats = auroc_floor(
        labels, scores, seed=gate.get("seed", seed), n_boot=gate.get("n_boot", n_boot)
    )
    passed = True
    if "pass_if_auroc" in gate:
        op_fn, thr = _parse_comparison(gate["pass_if_auroc"])
        passed = passed and bool(op_fn(stats["auroc"], thr))
    if "pass_if_floor" in gate:
        op_fn, thr = _parse_comparison(gate["pass_if_floor"])
        passed = passed and bool(op_fn(stats["ci_lb"], thr))
    return {"value": stats, "passed": passed}


_DISPATCH = {
    "count_flips": _eval_count_flips,
    "kill_diff_vs_control": _eval_kill_diff,
    "permutation_p": _eval_permutation_p,
    "auroc_floor": _eval_auroc_floor,
}


def evaluate_gates(
    gates_config: dict,
    rows: list[dict],
    arm_field: str = "arm",
) -> dict:
    """Evaluate every gate in gates_config against per-row output.

    Returns a report dict with one entry per gate plus an overall_pass flag that
    is True only if every gate passed.
    """
    seed = int(gates_config.get("seed", 0))
    n_boot = int(gates_config.get("n_boot", 1000))
    n_perm = int(gates_config.get("n_perm", 1000))
    results = {}
    all_pass = True
    for gate in gates_config.get("gates", []):
        name = gate["name"]
        primitive = gate["primitive"]
        if primitive not in _DISPATCH:
            raise ValueError(f"unknown gate primitive {primitive!r}")
        if primitive == "count_flips":
            res = _DISPATCH[primitive](gate, rows, arm_field)
        elif primitive == "permutation_p":
            res = _DISPATCH[primitive](gate, rows, arm_field, seed, n_perm)
        else:
            res = _DISPATCH[primitive](gate, rows, arm_field, seed, n_boot)
        res["primitive"] = primitive
        results[name] = res
        all_pass = all_pass and res["passed"]
    return {"gates": results, "overall_pass": all_pass}


def load_gates_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
