from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

from Evaluator import compare
from Evaluator.compare import (
    EXIT_GATE_FAILED,
    EXIT_INPUT_ERROR,
    EXIT_OK,
    GateThresholds,
    build_comparison,
    derive_label,
    load_results,
    mcnemar_exact_p,
    parse_candidate_arg,
)




def _record(
    case_id: str,
    passed: bool,
    *,
    tags: Iterable[str] = ("general",),
    error: Optional[str] = None,
    latency: Optional[float] = 1.0,
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "question": f"question for {case_id}",
        "tags": list(tags),
        "latency_s": latency,
        "passed": passed,
        "correctness_passed": passed,
        "error": error,
    }


def _payload(records: List[Dict[str, Any]], **metadata: Any) -> Dict[str, Any]:
    base = {
        "backend": "llamacpp",
        "model": "model.gguf",
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 512,
        "seed": 7,
        "scenarios": ["tool_prompts.yaml"],
        "preset": None,
        "tags_filter": [],
        "limit": None,
        "dry_run": False,
    }
    base.update(metadata)
    return {
        "schema_version": "synaptic-evaluation-run-payload/v1",
        "metadata": base,
        "summary": {},
        "records": records,
    }


def _write(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reference_records() -> List[Dict[str, Any]]:
    # 10 cases, 8 pass. c8 and c9 fail.
    return [
        _record(f"c{i}", i < 8, tags=("alpha",) if i < 5 else ("beta",))
        for i in range(10)
    ]


def _candidate_records() -> List[Dict[str, Any]]:
    # Regress c0 (alpha) and c5 (beta); improve c9 (beta). 7/10 pass.
    records = _reference_records()
    flips = {"c0": False, "c5": False, "c9": True}
    for record in records:
        if record["case_id"] in flips:
            record["passed"] = flips[record["case_id"]]
            record["correctness_passed"] = flips[record["case_id"]]
        record["latency_s"] = 2.0
    return records


def _reference_payload(**metadata: Any) -> Dict[str, Any]:
    artifact = {
        "model": "merged",
        "backend": "vllm",
        "quantization": None,
        "quantization_source": None,
    }
    return _payload(_reference_records(), backend="vllm", model="merged", model_artifact=artifact, **metadata)


def _files(tmp_path: Path, candidate: Optional[Dict[str, Any]] = None):
    reference = load_results(_write(tmp_path / "ref.json", _reference_payload()))
    cand = load_results(
        _write(tmp_path / "cand.json", candidate or _payload(_candidate_records(), model="model-Q4_K_M.gguf"))
    )
    return reference, cand


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("regressions", "improvements", "expected"),
    [
        (0, 0, 1.0),
        (1, 0, 1.0),
        (3, 1, 0.625),
        (5, 0, 0.0625),
        (6, 0, 0.03125),
        (8, 2, 0.109375),
        (2, 8, 0.109375),
        (10, 10, 1.0),
    ],
)
def test_mcnemar_exact_p_known_values(regressions, improvements, expected):
    assert mcnemar_exact_p(regressions, improvements) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Deltas, flips, tags, latency, errors
# ---------------------------------------------------------------------------

def test_deltas_flips_and_tags(tmp_path):
    reference, candidate = _files(tmp_path)
    report = build_comparison(reference, [candidate], GateThresholds())
    item = report["candidates"][0]

    assert item["label"] == "Q4_K_M"
    assert item["comparable"] is True
    assert item["reference"]["pass_rate"] == pytest.approx(0.8)
    assert item["candidate"]["pass_rate"] == pytest.approx(0.7)
    assert item["deltas"]["pass_rate_pp"] == pytest.approx(-10.0)
    assert item["deltas"]["correctness_pass_rate_pp"] == pytest.approx(-10.0)
    assert item["flips"]["regressions"] == ["c0", "c5"]
    assert item["flips"]["improvements"] == ["c9"]
    assert item["flips"]["discordant"] == 3
    assert item["flips"]["mcnemar_p"] == pytest.approx(1.0)

    alpha, beta = item["by_tag"]["alpha"], item["by_tag"]["beta"]
    assert (alpha["reference_passed"], alpha["candidate_passed"]) == (5, 4)
    assert alpha["delta_pp"] == pytest.approx(-20.0)
    assert (alpha["regressions"], alpha["improvements"]) == (1, 0)
    assert (beta["reference_passed"], beta["candidate_passed"]) == (3, 3)
    assert beta["delta_pp"] == pytest.approx(0.0)
    assert (beta["regressions"], beta["improvements"]) == (1, 1)

    assert item["latency"]["ratio"] == pytest.approx(2.0)
    assert item["gate"] == {"active": False, "passed": True, "breaches": []}
    assert report["passed"] is True


def test_request_errors_and_missing_latency(tmp_path):
    records = _candidate_records()
    records[1].update(passed=False, correctness_passed=None, error="timeout", latency_s=None)
    reference, candidate = _files(tmp_path, _payload(records))
    item = build_comparison(reference, [candidate], GateThresholds())["candidates"][0]

    assert item["request_errors"] == {"reference": 0, "candidate": 1}
    assert item["candidate"]["correctness_tested"] == 9
    assert "c1" in item["flips"]["regressions"]
    assert item["latency"]["candidate_mean_s"] == pytest.approx(2.0)


def test_latency_ratio_absent_without_reference_latency(tmp_path):
    ref_records = [dict(record, latency_s=None) for record in _reference_records()]
    reference = load_results(_write(tmp_path / "ref.json", _payload(ref_records)))
    candidate = load_results(_write(tmp_path / "cand.json", _payload(_candidate_records())))
    item = build_comparison(reference, [candidate], GateThresholds())["candidates"][0]
    assert item["latency"]["ratio"] is None


def test_duplicate_case_ids_join_by_occurrence(tmp_path):
    records = [_record("dup", True), _record("dup", False)]
    results = load_results(_write(tmp_path / "dup.json", _payload(records)))
    assert list(results.cases) == ["dup", "dup#2"]
    assert results.duplicate_case_ids == ["dup"]


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------

def test_case_id_mismatch_is_non_comparable_and_fails_gate(tmp_path):
    records = _reference_records()[:-1]
    reference, candidate = _files(tmp_path, _payload(records))
    report = build_comparison(reference, [candidate], GateThresholds(max_pass_rate_drop=50))
    item = report["candidates"][0]

    assert item["comparable"] is False
    assert item["only_reference_case_ids"] == ["c9"]
    assert item["case_counts"] == {"shared": 9, "only_reference": 1, "only_candidate": 0}
    assert item["gate"]["passed"] is False
    assert "not comparable" in item["gate"]["breaches"][0]

    allowed = build_comparison(reference, [candidate], GateThresholds(max_pass_rate_drop=50, allow_mismatch=True))
    assert allowed["candidates"][0]["gate"]["passed"] is True


def test_settings_mismatch_detected(tmp_path):
    reference, candidate = _files(
        tmp_path,
        _payload(_candidate_records(), temperature=0.7, scenarios=["other.yaml"]),
    )
    item = build_comparison(reference, [candidate], GateThresholds())["candidates"][0]
    assert item["comparable"] is False
    joined = " ".join(item["mismatches"])
    assert "'temperature'" in joined
    assert "'scenarios'" in joined
    # Report-only mode never fails on a mismatch.
    assert item["gate"]["passed"] is True


def test_settings_missing_on_one_side_are_not_mismatches(tmp_path):
    payload = _payload(_candidate_records())
    del payload["metadata"]["scenarios"]
    reference, candidate = _files(tmp_path, payload)
    item = build_comparison(reference, [candidate], GateThresholds())["candidates"][0]
    assert item["comparable"] is True


def test_partial_results_are_non_comparable(tmp_path):
    reference, candidate = _files(tmp_path, _payload(_candidate_records(), artifact_status="partial"))
    item = build_comparison(reference, [candidate], GateThresholds())["candidates"][0]
    assert item["comparable"] is False
    assert any("partial" in mismatch for mismatch in item["mismatches"])


def test_reference_precision_warnings(tmp_path):
    loaded_4bit = _payload(
        _reference_records(),
        backend="unsloth",
        model_artifact={"quantization": None, "load_settings": {"load_in_4bit": True}},
    )
    reference = load_results(_write(tmp_path / "ref4.json", loaded_4bit))
    candidate = load_results(_write(tmp_path / "cand.json", _payload(_candidate_records())))
    warnings = build_comparison(reference, [candidate], GateThresholds())["warnings"]
    assert any("4-bit" in warning for warning in warnings)

    unrecorded = load_results(_write(tmp_path / "ref_old.json", _payload(_reference_records(), backend="unsloth")))
    warnings = build_comparison(unrecorded, [candidate], GateThresholds())["warnings"]
    assert any("not recorded" in warning for warning in warnings)

    quantized = load_results(
        _write(tmp_path / "ref_q8.json", _payload(_reference_records(), model="m-Q8_0.gguf"))
    )
    warnings = build_comparison(quantized, [candidate], GateThresholds())["warnings"]
    assert any("Q8_0" in warning for warning in warnings)

    full = load_results(
        _write(
            tmp_path / "ref_f16.json",
            _payload(_reference_records(), model="m-F16.gguf", model_artifact={"quantization": "F16"}),
        )
    )
    assert build_comparison(full, [candidate], GateThresholds())["warnings"] == []


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def test_gate_thresholds(tmp_path):
    reference, candidate = _files(tmp_path)

    def gate(**kwargs):
        return build_comparison(reference, [candidate], GateThresholds(**kwargs))["candidates"][0]["gate"]

    assert gate(max_pass_rate_drop=10)["passed"] is True
    breached = gate(max_pass_rate_drop=5)
    assert breached["passed"] is False
    assert "pass rate dropped 10.0pp" in breached["breaches"][0]
    assert gate(max_correctness_drop=5)["passed"] is False
    assert gate(max_regressions=2)["passed"] is True
    assert gate(max_regressions=1)["passed"] is False
    # alpha tag drops 20pp, beta 0pp.
    assert gate(max_tag_drop=25)["passed"] is True
    assert gate(max_tag_drop=15)["passed"] is False
    assert gate(max_tag_drop=15, min_tag_cases=6)["passed"] is True
    # With a significance requirement, a 3-discordant-case swing is noise.
    assert gate(max_pass_rate_drop=5, alpha=0.05)["passed"] is True


def test_gate_significance_catches_large_swing(tmp_path):
    ref_records = [_record(f"c{i}", True) for i in range(20)]
    cand_records = [_record(f"c{i}", i >= 8) for i in range(20)]
    reference = load_results(_write(tmp_path / "ref.json", _payload(ref_records)))
    candidate = load_results(_write(tmp_path / "cand.json", _payload(cand_records)))
    item = build_comparison(reference, [candidate], GateThresholds(max_pass_rate_drop=5, alpha=0.05))["candidates"][0]
    assert item["flips"]["mcnemar_p"] == pytest.approx(2 / 2 ** 8)
    assert item["gate"]["passed"] is False


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_parse_candidate_arg():
    assert parse_candidate_arg("Q4_K_M=results/q4.json") == ("Q4_K_M", "results/q4.json")
    assert parse_candidate_arg("results/q4.json") == (None, "results/q4.json")
    assert parse_candidate_arg("results/a=b.json") == (None, "results/a=b.json")


def test_derive_label_precedence():
    recorded = {"model_artifact": {"quantization": "Q5_K_M"}, "model": "x-Q8_0.gguf"}
    assert derive_label(Path("run.json"), recorded) == "Q5_K_M"
    assert derive_label(Path("run.json"), {"model": "/m/model.Q8_0.gguf"}) == "Q8_0"
    assert derive_label(Path("eval_iq2_xs.json"), {"model": "served"}) == "IQ2_XS"
    assert derive_label(Path("baseline.json"), {"model": "served"}) == "baseline"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_report_only_writes_outputs(tmp_path, capsys):
    ref = _write(tmp_path / "ref.json", _reference_payload())
    cand = _write(tmp_path / "q4.json", _payload(_candidate_records(), model="model-Q4_K_M.gguf"))
    out_json = tmp_path / "out" / "cmp.json"
    out_md = tmp_path / "out" / "cmp.md"

    code = compare.main(
        ["--reference", str(ref), "--candidate", str(cand), "--output", str(out_json), "--markdown", str(out_md)]
    )

    assert code == EXIT_OK
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == compare.COMPARISON_SCHEMA_VERSION
    assert report["candidates"][0]["label"] == "Q4_K_M"
    markdown = out_md.read_text(encoding="utf-8")
    assert "# Evaluation Comparison" in markdown
    assert "`c0`" in markdown
    console = capsys.readouterr().out
    assert "Q4_K_M" in console
    assert "-10.0pp" in console
    assert "report only" in console


def test_main_exit_codes(tmp_path):
    ref = _write(tmp_path / "ref.json", _reference_payload())
    cand = _write(tmp_path / "cand.json", _payload(_candidate_records()))
    def run(*args: str) -> int:
        return compare.main(["--reference", str(ref), *args])

    assert run("--candidate", f"Q4={cand}", "--max-pass-rate-drop", "20") == EXIT_OK
    assert run("--candidate", f"Q4={cand}", "--max-pass-rate-drop", "5") == EXIT_GATE_FAILED
    assert run("--candidate", f"Q4={cand}", "--candidate", f"Q8={cand}", "--max-regressions", "1") == EXIT_GATE_FAILED
    assert run("--candidate", str(tmp_path / "missing.json")) == EXIT_INPUT_ERROR
    assert run("--candidate", f"A={cand}", "--candidate", f"A={cand}") == EXIT_INPUT_ERROR
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert run("--candidate", str(bad)) == EXIT_INPUT_ERROR
    not_results = _write(tmp_path / "other.json", {"hello": "world"})
    assert run("--candidate", str(not_results)) == EXIT_INPUT_ERROR
    assert compare.main(["--reference", str(ref)]) == EXIT_INPUT_ERROR
