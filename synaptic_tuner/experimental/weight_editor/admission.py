"""Pure host-side admission of one model request.

Runs before any reservation or executor involvement. A model request can name
an admitted operation and fill its declared arguments; it cannot carry paths,
commands, imports, policy, budget or split overrides.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .canon import sha256_hex
from .config import AUTHORITY_NAMES, ID_RE, EditorConfig
from .ledger import Ledger

REQUEST_KEYS = frozenset({"operation", "arguments", "rationale", "label"})
_PATHLIKE = re.compile(r"(^[~$%])|(\.\.)|[/\\\x00]|(^[A-Za-z][A-Za-z0-9+.-]*:)")

PUBLIC = {
    "failed_turn": "provider turn did not complete; submit a new request",
    "interrupted_turn": "provider turn was interrupted; submit a new request",
    "capture_incomplete": "provider capture is incomplete; this turn cannot authorize an operation",
    "model_drift": "returned model identity did not match; this turn cannot authorize an operation",
    "model_identity_unverified": "returned model identity was not observed; this turn cannot authorize an operation",
    "phase_closed": "no further operations are accepted in the current phase",
    "malformed_request": "request is malformed",
    "forbidden_field": "request carries fields that grant no authority",
    "unadmitted_operation": "operation is not admitted",
    "unknown_argument": "request has an undeclared argument",
    "missing_argument": "request is missing a required argument",
    "invalid_argument": "argument value is invalid",
    "path_authority": "paths, URLs and commands are not artifact references",
    "unknown_object": "unknown or unadmitted object",
    "withheld_id": "unknown or unadmitted object",
    "source_or_base_write": "target is not a writable artifact name",
    "artifact_exists": "artifact name already used in this run",
    "budget_exceeded": "resource policy does not permit this request",
    "executor_validation": "operation rejected by executor validation",
}


@dataclass(frozen=True)
class Denied:
    reason: str
    detail: str

    @property
    def public(self) -> str:
        return PUBLIC.get(self.reason, "request denied")


@dataclass(frozen=True)
class Admitted:
    operation: str
    arguments: Mapping[str, Any]
    bound: Mapping[str, Any]
    cost: Mapping[str, float]


def _id_value(value: Any, where: str) -> Denied | str:
    if not isinstance(value, str):
        return Denied("invalid_argument", f"{where} must be an id string")
    if _PATHLIKE.search(value):
        return Denied("path_authority", f"{where} looks like a path/URL/command")
    if not ID_RE.match(value):
        return Denied("invalid_argument", f"{where} is not a well-formed id")
    return value


def admit(
    payload: Any,
    config: EditorConfig,
    ledger: Ledger,
    used_artifact_names: set[str],
    phase: str,
) -> Admitted | Denied:
    if phase != "discovery":
        return Denied("phase_closed", f"phase is {phase}")
    if not isinstance(payload, dict):
        return Denied("malformed_request", "request is not an object")
    extra = set(payload) - REQUEST_KEYS
    if extra:
        return Denied("forbidden_field", f"top-level fields {sorted(extra)}")
    op_name = payload.get("operation")
    if not isinstance(op_name, str):
        return Denied("malformed_request", "operation missing")
    spec = config.operations.get(op_name)
    if spec is None:
        return Denied("unadmitted_operation", f"operation {op_name!r}")
    args = payload.get("arguments", {})
    if not isinstance(args, dict):
        return Denied("malformed_request", "arguments is not an object")
    for key in args:
        if key not in spec.args:
            reason = "forbidden_field" if key in AUTHORITY_NAMES else "unknown_argument"
            return Denied(reason, f"argument {key!r}")
    for name, aspec in spec.args.items():
        if aspec.required and name not in args:
            return Denied("missing_argument", f"argument {name!r}")

    bound: dict[str, Any] = {}
    clean: dict[str, Any] = {}
    for name, value in args.items():
        aspec = spec.args[name]
        where = f"argument {name!r}"
        t, p = aspec.type, aspec.params
        if t in ("ref", "ref_list"):
            if t == "ref":
                values = [value]
            else:
                if not isinstance(value, list) or not value:
                    return Denied("invalid_argument", f"{where} must be a non-empty list")
                if len(value) > p["max_items"]:
                    return Denied("invalid_argument", f"{where} exceeds max_items")
                if len(set(map(str, value))) != len(value):
                    return Denied("invalid_argument", f"{where} has duplicates")
                values = value
            resolved = []
            for v in values:
                r = _id_value(v, where)
                if isinstance(r, Denied):
                    return r
                if sha256_hex(r.encode()) in config.withheld_digests:
                    return Denied("withheld_id", f"{where} id is registered as withheld")
                obj = config.capsule.objects.get(r)
                if obj is None or obj.kind != p["kind"]:
                    return Denied("unknown_object", f"{where} id {r!r} not an admitted {p['kind']}")
                resolved.append({"id": obj.id, "kind": obj.kind, "sha256": obj.sha256, "path": str(obj.abs_path)})
            bound[name] = resolved[0] if t == "ref" else resolved
            clean[name] = value
        elif t == "int":
            if isinstance(value, bool) or not isinstance(value, int) or not p["min"] <= value <= p["max"]:
                return Denied("invalid_argument", f"{where} not an int in [{p['min']},{p['max']}]")
            clean[name] = value
        elif t == "number":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not p["min"] <= value <= p["max"]
            ):
                return Denied("invalid_argument", f"{where} not a finite number in range")
            clean[name] = value
        elif t == "enum":
            if value not in p["values"]:
                return Denied("invalid_argument", f"{where} not in enum")
            clean[name] = value
        elif t == "text":
            if not isinstance(value, str) or len(value) > p["max_length"]:
                return Denied("invalid_argument", f"{where} must be text up to {p['max_length']} chars")
            clean[name] = value
        elif t == "artifact_name":
            r = _id_value(value, where)
            if isinstance(r, Denied):
                return r
            if r in config.capsule.objects:
                return Denied("source_or_base_write", f"{where} names a capsule source/base object")
            if r in used_artifact_names:
                return Denied("artifact_exists", f"{where} {r!r}")
            clean[name] = r

    cost = _cost(spec.cost, clean)
    cost["dispatches"] = cost.get("dispatches", 0) + 1
    over = ledger.exceeded_by(cost)
    if over:
        return Denied("budget_exceeded", f"exceeds limit: {over}")
    return Admitted(op_name, clean, bound, cost)


def _cost(spec: Mapping[str, Any], args: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for res, c in spec.items():
        if isinstance(c, dict) and "from_arg" in c:
            out[res] = float(args[c["from_arg"]]) if c["from_arg"] in args else 0.0
        elif isinstance(c, dict):
            out[res] = float(len(args.get(c["len_of"], []))) * c["per_item"]
        else:
            out[res] = c
    return out
