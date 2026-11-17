"""Forbidden access must have no executor effect; budgets are caller-owned and never reset."""

from __future__ import annotations

import json

from .conftest import QUERY12, final, of_type, patch, req, script, types


def denied_reasons(fx, run_id="r1"):
    return [(r["data"]["request_id"], r["data"]["reason"]) for r in of_type(fx.journal(run_id), "request_denied")]


def test_forbidden_requests_are_denied_with_no_executor_effects(make):
    turn1 = [
        {"type": "tool_request", "operation": "inspect_weights", "arguments": {"tensor": "w.layer0"},
         "command": "cat /etc/passwd"},                                                       # command authority
        req("inspect_weights", tensor="/etc/passwd"),                                          # absolute path
        req("inspect_weights", tensor="../../capsule/tensors/w.layer0.json"),                  # traversal
        req("inspect_weights", tensor="file:///etc/hostname"),                                 # URL scheme
        req("inspect_weights", tensor="dev-999"),                                              # unknown id
        req("query_development", items=["heldout-007"]),                                       # withheld id
        req("inspect_weights", tensor="dev-001"),                                              # wrong object kind
        req("run_shell", cmd="id"),                                                            # unadmitted op
        req("analyze", tensor="w.layer0"),                                                     # not configured
        req("inspect_weights", tensor="w.layer0", output_root="/"),                            # policy override
        req("inspect_weights", tensor="w.layer0", budget=10**9),                               # budget override
        req("inspect_weights"),                                                                # missing arg
        req("inspect_weights", tensor="w.layer0", extra=1),                                    # undeclared arg
        patch("w.layer0"),                                                                     # write onto base/source id
        patch("../../escape"),                                                                 # write traversal
        req("query_development", items=["dev-001", "dev-001"]),                                # duplicate ids
        req("query_development", items=["dev-001", "dev-002", "dev-003", "dev-004", "dev-001"]),  # over max_items
        {"type": "tool_request", "operation": "inspect_weights", "arguments": []},             # malformed
        {"type": "tool_request", "arguments": {"tensor": "w.layer0"}},                         # no op
        {"type": "tool_request", "operation": "inspect_weights", "arguments": {"tensor": "w.layer0"},
         "policy": {"admit": "all"}},                                                          # policy field
    ]
    fx = make(script=script(turn1, [final()]))
    code, out, _ = fx.run()
    assert code == 0, out
    assert fx.effect_lines() == []
    assert list((fx.effects / "dispatches").iterdir()) == []   # executor never touched at all
    assert out["dispatches"] == {}
    reasons = [r for _, r in denied_reasons(fx)]
    assert reasons == [
        "forbidden_field", "path_authority", "path_authority", "path_authority", "unknown_object",
        "withheld_id", "unknown_object", "unadmitted_operation", "unadmitted_operation", "forbidden_field",
        "forbidden_field", "missing_argument", "unknown_argument", "source_or_base_write", "path_authority",
        "invalid_argument", "invalid_argument", "malformed_request", "malformed_request", "forbidden_field",
    ]
    assert "dispatch_intent" not in types(fx.journal())
    assert not any((fx.paths["root"] / "escape").parent.glob("escape*"))
    # every attempt, denied or not, was charged to the requests meter
    assert out["budget"]["requests"]["committed"] == len(turn1)
    assert "dispatches" not in out["budget"]


def test_withheld_and_unknown_ids_look_identical_to_the_model(make):
    fx = make(script=script([req("query_development", items=["heldout-007"]),
                             req("query_development", items=["dev-999"])], [final()]))
    assert fx.run()[0] == 0
    msgs = [json.loads(r["data"]["message"]["content"]) for r in of_type(fx.journal(), "request_denied")]
    assert msgs[0]["detail"] == msgs[1]["detail"] == "unknown or unadmitted object"
    assert {k: v for k, v in msgs[0].items() if k != "request_id"} == {
        k: v for k, v in msgs[1].items() if k != "request_id"}
    audit = {r["data"]["reason"] for r in of_type(fx.journal(), "request_denied")}
    assert audit == {"withheld_id", "unknown_object"}   # distinguished only in the audit trail


def test_executor_validation_denial_has_no_effect_and_no_charge(make):
    fx = make(script=script([patch("p1", text="   ")], [final()]))
    # patch text is non-empty-checked by the executor port (side-effect-free validate)
    code, out, _ = fx.run()
    assert code == 0
    assert denied_reasons(fx) == [("req-0001", "executor_validation")]
    assert fx.effect_lines() == []
    assert out["budget"].get("cpu_units", {"charged": 0})["charged"] == 0


def test_budget_exhaustion_denies_without_effect_and_is_audited(make):
    fx = make(limits={"gpu_units": 1.0, "examples": 2},
              script=script([QUERY12, req("query_development", items=["dev-003"]), req("query_development", items=["dev-004"])],
                            [final()]))
    code, out, _ = fx.run()
    assert code == 0
    assert len(fx.effect_lines()) == 1   # only the first query fit under the physical budget
    assert denied_reasons(fx) == [("req-0002", "budget_exceeded"), ("req-0003", "budget_exceeded")]
    assert out["budget"]["gpu_units"]["committed"] == 1.0 and out["budget"]["gpu_units"]["reserved_outstanding"] == 0
    reserve = [e for r in of_type(fx.journal(), "request_admitted") for e in r["data"]["ledger"]]
    assert {e["op"] for e in reserve} == {"reserve"}   # reservation precedes the charge, both in the audit trail


def test_no_default_money_or_token_cap_is_invented(make):
    fx = make(script=script([req("inspect_weights", tensor="w.layer0")] * 25, [final()]))
    code, out, _ = fx.run()
    assert code == 0 and len(fx.effect_lines()) == 25
    assert all(row["limit"] is None for row in out["budget"].values())


def test_request_and_turn_limits_halt_and_do_not_reset_on_resume(make):
    fx = make(limits={"requests": 2}, script=script(
        [req("inspect_weights", tensor="w.layer0")] * 3, [final()]))
    code, out, _ = fx.run()
    assert code == 4 and out["status"] == "incomplete" and out["halt"]["code"] == "budget_exhausted"
    assert len(fx.effect_lines()) == 2 and denied_reasons(fx) == [("req-0003", "budget_exceeded")]
    turns_before = len(of_type(fx.journal(), "turn_start"))
    code2, out2, _ = fx.resume()
    assert code2 == 4 and out2["halt"]["code"] == "budget_exhausted"
    assert out2["budget"]["requests"]["committed"] == 3          # charges survive resume
    assert len(of_type(fx.journal(), "turn_start")) == turns_before   # model not called again

    fx2 = make(limits={"model_turns": 1}, script=script([req("inspect_weights", tensor="w.layer0")], [final()]))
    code, out, _ = fx2.run()
    assert code == 4 and out["halt"]["code"] == "budget_exhausted" and out["turns"] == 1
    assert len(fx2.effect_lines()) == 1


def test_nan_usage_from_executor_is_charged_at_reservation_and_flagged(make):
    fx = make(limits={"gpu_units": 10}, script=script([QUERY12], [final()]))
    code, out, _ = fx.run(fault="bad_usage")
    assert code == 0
    res = of_type(fx.journal(), "dispatch_result")[0]["data"]
    assert "usage_invalid_charged_reservation_only" in res["anomalies"]
    assert out["budget"]["gpu_units"]["committed"] == 1.0   # conservative: reservation, never zero, never NaN


def test_late_query_after_frozen_choice_has_no_effect(make):
    fx = make(script=script([patch("p1")], [final("patch", "req-0001"), QUERY12, req("inspect_weights", tensor="w.layer0")]))
    code, out, _ = fx.run()
    assert code == 0 and out["phase"] == "final_patch_frozen"
    assert len(fx.effect_lines()) == 1   # only the proposal; the post-freeze queries never reached the executor
    assert [r for _, r in denied_reasons(fx)] == ["phase_closed", "phase_closed"]
    assert types(fx.journal()).index("final_patch_frozen") < [
        i for i, r in enumerate(fx.journal()) if r["type"] == "request_denied"][0]
