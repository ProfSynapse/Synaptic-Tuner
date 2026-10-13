"""Nonsecret controls actually consumed by ordinary evaluation.

SCORER_REVISION must change when verdict, assertion, projection or aggregation
semantics change. It is a semantic declaration, not code attestation. A basis
never proves the remote model or service implementation.
"""
from copy import deepcopy
import hashlib
import json
import math

SCORER_REVISION = "synaptic-evaluator-scoring/v1"
BASIS_SCHEMA = "synaptic-evaluation-comparison-basis/v1"
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_COMPARISON_BYTES = 4 * 1024 * 1024
MAX_CASES = 10000
GENERATION_KEYS = ("temperature", "top_p", "max_tokens", "seed", "chat_template_kwargs",
                   "presence_penalty", "top_k", "min_p", "repetition_penalty")


def bounded_json(value, maximum=MAX_REPORT_BYTES):
    remaining = 200000
    def check(item, depth=0, ancestors=()):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 32:
            raise ValueError("comparison value exceeds tree bound")
        if type(item) in (dict, list):
            if id(item) in ancestors:
                raise ValueError("comparison value cycle")
            ancestors += (id(item),)
            if type(item) is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("comparison keys must be strings")
                values = item.values()
            else:
                values = item
            for child in values:
                check(child, depth + 1, ancestors)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("comparison value is not JSON")
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("comparison value is not finite")
    check(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    if len(raw) > maximum:
        raise ValueError("comparison value exceeds byte bound")
    return raw


def digest(value):
    return hashlib.sha256(bounded_json(value)).hexdigest()


def backend_controls(client):
    """Optional explicit producer contract; never inspect arbitrary attributes."""
    operation = getattr(client, "comparison_configuration", None)
    if not callable(operation):
        return None
    value = operation()
    if (type(value) is not dict or set(value) != {"backend", "revision", "generation"}
            or type(value["backend"]) is not str or not value["backend"]
            or type(value["revision"]) is not str or not value["revision"]
            or type(value["generation"]) is not dict
            or set(value["generation"]) != set(GENERATION_KEYS)):
        raise ValueError("invalid nonsecret backend comparison configuration")
    bounded_json(value, 16384)
    generation = value["generation"]
    for key in ("temperature", "top_p", "presence_penalty", "min_p", "repetition_penalty"):
        if generation[key] is not None and type(generation[key]) not in (int, float):
            raise ValueError("invalid generation configuration")
    for key in ("max_tokens", "seed", "top_k"):
        if generation[key] is not None and type(generation[key]) is not int:
            raise ValueError("invalid generation configuration")
    return deepcopy(value)


def case_projection(groups):
    """Already rendered prompts and normalized declarations, not source rereads."""
    rows = []
    for ref, cases in groups:
        for case in cases:
            rows.append({"ref": ref, "case_id": case.case_id, "messages": case.chat_messages(),
                         "question": case.question, "tags": list(case.tags),
                         "metadata": {key: deepcopy(value) for key, value in case.metadata.items()
                                      if key != "config_dir"}})
    if len(rows) > MAX_CASES:
        raise ValueError("too many comparison cases")
    bounded_json(rows)
    return rows


def build_basis(groups, validator_config, controls, *, judge=False, selection=None):
    rows = case_projection(groups)
    reasons = []
    if controls is None:
        reasons.append("backend_controls_unavailable")
    if judge:
        reasons.append("judge_controls_unavailable")
    if any(any(row["metadata"].get(key) for key in
                   ("retrieval_config", "audio_config", "environment", "scoring")) for row in rows):
        reasons.append("external_scoring_inputs_unavailable")
    identities = [(row["ref"], row["case_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        reasons.append("ambiguous_case_identity")
    return {"schema_version": BASIS_SCHEMA, "status": "unverified" if reasons else "verified",
            "reasons": reasons, "scorer_revision": SCORER_REVISION,
            "inputs_digest": digest({"cases": rows, "selection": selection}),
            "validator_digest": digest({"wrappers": validator_config.wrapper_specs,
                                         "schemas": validator_config.tool_schemas}),
            "backend_digest": None if controls is None else digest({key: controls[key] for key in ("backend", "revision")}),
            "generation_digest": None if controls is None else digest(controls["generation"]),
            "judge_digest": None}


def unverified_basis(reason):
    return {"schema_version": BASIS_SCHEMA, "status": "unverified", "reasons": [reason],
            "scorer_revision": SCORER_REVISION, "inputs_digest": None,
            "validator_digest": None, "backend_digest": None, "generation_digest": None,
            "judge_digest": None}
