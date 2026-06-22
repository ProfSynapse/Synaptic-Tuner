"""Name-leak structural-gate regression test.

Location: tests/test_name_leak_gate_regression.py
Summary: A DURABLE, deterministic (zero-API) guard that the structural name-leak
         gate — a ``text_not_regex`` assertion over the serialized response
         ``$.content`` — catches a planted entity name in a GENERATED output and
         passes a properly-abstracted one. A privacy-abstraction contract rests on
         this gate hard-failing when an entity name leaks; this test locks that
         GREEN in permanently so a future refactor of the assertion verifier, the
         response view, or the regex semantics can't silently weaken it.

         The gate is PROMPT-INDEPENDENT: copywriting operators only mutate the
         PROMPT fed to the target, never the gate logic, so a deterministic regex
         test fully covers the gate's behavior with no live call. The mechanism is
         generic (text_not_regex over $.content); the planted name + pattern below
         use fictional entities and a neutral topic that exercise the same gate.
"""
from __future__ import annotations

import json

from shared.verifiers.builtins.assertion_verifier import evaluate_correctness
from Evaluator.response_view import build_response_view


def _name_leak_correct(pattern: str) -> dict:
    """The structural gate as a scenario declares it: a single text_not_regex over
    $.content (case-insensitive), under `all`."""
    return {
        "all": [
            {"type": "text_not_regex", "name": "no_name_leak", "path": "$.content", "pattern": pattern},
        ]
    }


def _content_view(content: str) -> dict:
    """Build the response view the runner feeds to evaluate_correctness, with a
    schema-nested result.{title,details} serialized into $.content."""
    return build_response_view({"content": content})


# A realistic abstracted (GOOD) output — no entity name, a ROLE named instead.
_ABSTRACTED = json.dumps({
    "result": {
        "title": "Battery Storage and the Future of Urban Transit Networks",
        "details": (
            "For a piece on grid-scale battery storage, I'm looking to connect with transit "
            "engineers and energy researchers who can discuss how cities balance peak demand "
            "and storage capacity. What is realistic over the next few years, and where are the "
            "current limits? Specialists and practitioners welcome."
        ),
    }
})

# The SAME output but with an entity name leaked into the details (BAD).
_LEAKED = json.dumps({
    "result": {
        "title": "Battery Storage and the Future of Urban Transit Networks",
        "details": (
            "Dr. Priya Nadeau, a transit engineer, is available to discuss grid-scale battery "
            "storage, including how cities balance peak demand and storage capacity. Reach out "
            "to arrange an interview with the specialist."
        ),
    }
})


# --- The gate CATCHES a planted name (the load-bearing regression guard) ------


def test_gate_fails_when_a_source_name_is_planted_in_the_output():
    """A planted source name in the generated output MUST hard-fail the gate."""
    result = evaluate_correctness(_name_leak_correct(r"(?i)nadeau"), _content_view(_LEAKED))
    assert result.passed is False, "name-leak gate did NOT catch the planted source name"


def test_gate_catches_multi_entity_leak_pattern():
    """A multi-token pattern (name OR company) catches either leak."""
    pattern = r"(?i)(okafor|helio grid|helio)"
    leaked_company = _content_view(json.dumps({
        "result": {
            "title": "Open-Banking Credit Models and Mid-Size Lenders",
            "details": (
                "A risk officer at Helio Grid Partners can speak to how mid-size lenders "
                "adapt credit models to open-banking data. Looking for risk analysts and "
                "lending specialists to discuss the pitfalls."
            ),
        }
    }))
    result = evaluate_correctness(_name_leak_correct(pattern), leaked_company)
    assert result.passed is False, "gate missed a leaked COMPANY token"


# --- The gate PASSES a properly-abstracted output (no false positive) ---------


def test_gate_passes_when_the_output_is_properly_abstracted():
    """A correctly-abstracted output (role named, no source identity) MUST pass."""
    result = evaluate_correctness(_name_leak_correct(r"(?i)nadeau"), _content_view(_ABSTRACTED))
    assert result.passed is True, "gate false-positived on a clean, abstracted output"


# --- Counter-test: the assertion is genuinely coupled to the pattern ----------


def test_gate_is_coupled_to_the_pattern_not_vacuously_passing():
    """Falsification guard: the SAME abstracted output that PASSES its own pattern
    must FAIL a pattern matching a token it actually contains ('transit'). This
    proves the gate's pass in the test above is the regex genuinely not matching —
    not the assertion vacuously passing regardless of content."""
    # 'transit' IS present in the abstracted output -> a not_regex on it must fail.
    coupled = evaluate_correctness(_name_leak_correct(r"(?i)transit"), _content_view(_ABSTRACTED))
    assert coupled.passed is False, (
        "text_not_regex did not react to a token actually present in $.content — "
        "the gate may be vacuously passing"
    )


def test_case_insensitivity_holds():
    """The (?i) gate must catch the name regardless of casing in the output."""
    upper = _content_view(json.dumps({"result": {"title": "T", "details": "Contact NADEAU today."}}))
    result = evaluate_correctness(_name_leak_correct(r"(?i)nadeau"), upper)
    assert result.passed is False, "case-insensitive gate missed an upper-cased leak"
