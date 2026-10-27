"""CPU host/retained-response controls; no actual model execution or admission."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest

from examples.training_qualification import native_evaluation as host
from examples.training_qualification import native_evaluation_cases as preparation
from tests.dataset_prep.test_conversation_messages import build
from tests.runtime.test_native_mlx_evaluation import selection
from tuner.runtime.native_mlx_material import canonical
from tuner.runtime.native_mlx_evaluation import read_cases, validate_record


@pytest.fixture
def private_root():
    previous = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-eval-tests-", dir=Path.home()) as name:
            root = Path(name)
            root.chmod(0o700)
            yield root
    finally:
        os.umask(previous)


@pytest.fixture
def prepared_case(private_root):
    prepared = build()
    identity = prepared.identity
    publication = private_root / identity.dataset_id
    publication.mkdir(mode=0o700)
    for name, raw in (("dataset.jsonl", prepared.dataset_raw), ("manifest.json", prepared.manifest_raw)):
        (publication / name).write_bytes(raw)
        (publication / name).chmod(0o400)
    config = {"schema": "synaptic-native-evaluation-preparation/v1", "output_root": str(private_root / "prepared"),
        "publications": [{"label": "configured", "publication": str(publication),
            "dataset_digest": identity.dataset_digest, "content_sha256": host.sha(prepared.dataset_raw),
            "expected_train_rows": 1, "expected_validation_rows": 1,
            "reference_equals": ["$.content"], "assertions": []}]}
    return private_root, config, publication


def test_real_publication_preparation_keeps_all_context_and_only_validation(prepared_case):
    root, config, publication = prepared_case
    result = preparation.prepare(config)
    assert result["case_count"] == 1 and result["sources"][0]["train_rows"] == 1
    cases = json.loads((root / "prepared/cases.json").read_bytes())["cases"]
    assert cases[0]["reference"] == "answer b"
    assert [m["role"] for m in cases[0]["messages"]] == ["system", "user", "assistant", "user"]
    assert [m["content"] for m in cases[0]["messages"]] == ["system b", "question b", "previous b", "follow up b"]
    assert "answer b" not in (root / "prepared/heldout-prompts.jsonl").read_text()
    train = (root / "prepared/training-for-contamination.jsonl").read_text()
    assert "answer a" in train and "answer b" not in train
    score = json.loads((root / "prepared/scoring-cases.json").read_bytes())[0]
    assert score["correct"]["any"][0]["assertions"] == [
        {"type": "jsonpath_equals", "path": "$.content", "value": "answer b"}]
    assert score["messages"] == cases[0]["messages"]
    assert score["provenance"]["split"] == "validation"
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in (root / "prepared").iterdir())
    assert all(p.stat().st_mode & 0o777 == 0o400 for p in publication.iterdir())
    before = {p.name: p.read_bytes() for p in (root / "prepared").iterdir()}
    with pytest.raises(FileExistsError): preparation.prepare(config)
    assert {p.name: p.read_bytes() for p in (root / "prepared").iterdir()} == before


@pytest.mark.parametrize("fault", ["digest", "content", "count", "reference_path", "reference_assertion", "empty_directory", "hardlink"])
def test_preparation_refuses_bad_provenance_and_missing_score_inputs(prepared_case, fault):
    root, config, publication = prepared_case
    source = config["publications"][0]
    if fault == "digest": source["dataset_digest"] = "0" * 64
    elif fault == "content": source["content_sha256"] = "0" * 64
    elif fault == "count": source["expected_validation_rows"] = 2
    elif fault == "reference_path": source["reference_equals"] = ["$.unknown"]
    elif fault == "reference_assertion": source["assertions"] = [{"type": "jsonpath_equals", "path": "$.content", "value": "wrong"}]
    elif fault == "empty_directory": (publication / "unexpected").mkdir(mode=0o700)
    else: os.link(publication / "dataset.jsonl", root / "aliased.jsonl")
    with pytest.raises(ValueError): preparation.prepare(config)
    assert not (root / "prepared").exists()


@pytest.fixture
def signed_selection(private_root):
    key, public, signature = (private_root / name for name in ("test-private.pem", "public.pem", "admission.sig"))
    subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(key)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    subprocess.run(["openssl", "pkey", "-in", str(key), "-pubout", "-out", str(public)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    config = {"attempt": "fixture-only", "root_public_key_sha256": host.sha(public.read_bytes()),
        "root_public_key_file": str(public), "admission_file": str(private_root / "admission.json"),
        "admission_signature_file": str(signature), "state_root": str(private_root / "state")}
    selected = {"config": config, "fixture": "synthetic CPU signature test; never model execution authority"}
    admission = {"schema": "synaptic-native-evaluation-admission/v1", "selection_digest": host.sha(canonical(selected)),
        "attempt": config["attempt"], "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    Path(config["admission_file"]).write_bytes(canonical(admission))
    subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(key), "-out", str(signature), config["admission_file"]],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    return private_root, config, selected, admission, key


def test_actual_detached_signature_and_external_pin_are_required(signed_selection):
    _, config, selected, admission, _ = signed_selection
    pin = config["root_public_key_sha256"]
    assert host.authenticate(config, selected, starting=True, trusted_root_key_sha256=pin) == admission
    for wrong in (None, "", "0" * 64):
        with pytest.raises(ValueError): host.authenticate(config, selected, starting=True, trusted_root_key_sha256=wrong)
    changed = deepcopy(selected); changed["fixture"] = "changed"
    with pytest.raises(ValueError): host.authenticate(config, changed, starting=True, trusted_root_key_sha256=pin)
    Path(config["admission_signature_file"]).write_bytes(b"changed")
    with pytest.raises(ValueError): host.authenticate(config, selected, starting=True, trusted_root_key_sha256=pin)


def test_expired_signature_is_read_authority_only(signed_selection):
    _, config, selected, admission, key = signed_selection
    admission["expires_at"] = "2020-01-01T00:00:00Z"
    Path(config["admission_file"]).write_bytes(canonical(admission))
    subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(key), "-out", config["admission_signature_file"], config["admission_file"]],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    pin = config["root_public_key_sha256"]
    assert host.authenticate(config, selected, starting=False, trusted_root_key_sha256=pin) == admission
    with pytest.raises(ValueError): host.authenticate(config, selected, starting=True, trusted_root_key_sha256=pin)


def test_unknown_action_cannot_bypass_expired_start_check():
    with pytest.raises(ValueError, match="action"):
        host.run("invented", Path("does-not-exist"))


def row_for(case):
    return {"id": case["id"], "messages_sha256": host.sha(canonical(case["messages"])), "prompt_tokens": [1, 2],
        "completion_tokens": [3, 0], "text": "answer", "finish_reason": "stop", "latency_seconds": .1,
        "reference": {"loss": 1.5, "supervised_tokens": 4}}


@pytest.mark.parametrize("fault", ["id", "messages", "empty_tokens", "bool_tokens", "finish", "early_stop", "length",
    "latency", "loss", "supervised", "text_bound", "extra"])
def test_retained_record_integrity_and_bounds_are_explicit(fault):
    w, raw = selection(); case = read_cases(raw, w)[0]; row = row_for(case)
    validate_record(row, case, w)
    if fault == "id": row["id"] = "different"
    elif fault == "messages": row["messages_sha256"] = "0" * 64
    elif fault == "empty_tokens": row["prompt_tokens"] = []
    elif fault == "bool_tokens": row["completion_tokens"] = [True, 0]
    elif fault == "finish": row["finish_reason"] = "unknown"
    elif fault == "early_stop": row["completion_tokens"] = [0, 0]
    elif fault == "length": row["finish_reason"] = "length"
    elif fault == "latency": row["latency_seconds"] = float("nan")
    elif fault == "loss": row["reference"]["loss"] = float("inf")
    elif fault == "supervised": row["reference"]["supervised_tokens"] = 0
    elif fault == "text_bound": w["profile"]["maximum_response_bytes"] = 1
    else: row["extra"] = True
    with pytest.raises(ValueError): validate_record(row, case, w)


def test_existing_evaluator_scores_retained_responses_without_global_fallback(monkeypatch):
    from Evaluator.comparison_basis import backend_controls, build_basis
    from Evaluator.native_mlx_retained import NativeMLXRetainedClient
    from Evaluator.prompt_sets import PromptCase
    from Evaluator.protocols import BackendError
    from Evaluator.runner import evaluate_cases
    from shared.validation import dataset_validator
    from shared.validation.dataset_validator import ValidatorConfig
    w, raw = selection(); case = read_cases(raw, w)[0]; row = row_for(case)
    monkeypatch.setattr(dataset_validator, "default_validator_config", lambda: pytest.fail("ambient validator fallback"))
    client = NativeMLXRetainedClient([row], generation=w["generation"], profile=w["profile"])
    declared = PromptCase(case["id"], "question", metadata={"messages": case["messages"],
        "correct": {"any": [{"assertions": [{"type": "jsonpath_equals", "path": "$.content", "value": "answer"}]}]}})
    validator = ValidatorConfig([], {})
    records = evaluate_cases([declared], client, validator_config=validator)
    assert len(records) == 1 and records[0].correctness.passed and records[0].status == "pass"
    assert client.records_consumed == 1
    basis = build_basis((("configured", [declared]),), validator, backend_controls(client))
    assert basis["status"] == "verified"  # comparison inputs, not model authentication
    with pytest.raises(BackendError): client.chat(case["messages"])
    other = NativeMLXRetainedClient([row], generation=w["generation"], profile=w["profile"])
    with pytest.raises(BackendError): other.chat([{"role": "user", "content": "different"}])
    g = deepcopy(w["generation"]);g["stop_token_ids"] = [9]
    changed = NativeMLXRetainedClient([row], generation=g, profile=w["profile"])
    assert changed.comparison_configuration()["revision"] != other.comparison_configuration()["revision"]


def sign_fixture(config, selected, admission, key):
    admission["selection_digest"] = host.sha(canonical(selected))
    Path(config["admission_file"]).write_bytes(canonical(admission))
    subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(key), "-out", config["admission_signature_file"],
        config["admission_file"]], check=True, capture_output=True, timeout=10)


@pytest.fixture
def complete_evidence(signed_selection, monkeypatch):
    """Synthetic CPU execution witness, real signature/export/portable scorer."""
    from types import SimpleNamespace
    root, config, selected, admission, key = signed_selection
    w, raw = selection()
    cases = read_cases(raw, w)
    selected["request"] = {"workload": w, "model_snapshot": {"synthetic": "model"},
        "tokenizer_snapshot": {"synthetic": "tokenizer"}, "adapter_snapshot": None}
    state = Path(config["state_root"])
    state.mkdir(mode=0o700); (state / "output").mkdir(mode=0o700)
    (state / "cases.json").write_bytes(raw)
    record = canonical(row_for(cases[0]))
    result = {"schema": "synaptic-native-evaluation-result/v1", "status": "COMPLETE",
        "workload_sha256": host.sha(canonical(w)),
        "model_snapshot_digest": host.sha(canonical(selected["request"]["model_snapshot"])),
        "tokenizer_snapshot_digest": host.sha(canonical(selected["request"]["tokenizer_snapshot"])),
        "adapter_snapshot_digest": None, "case_count": 1, "case_bytes": len(record),
        "case_files": [{"name": "case-0000.json", "size": len(record), "sha256": host.sha(record)}]}
    (state / "output/case-0000.json").write_bytes(record)
    (state / "output/result.json").write_bytes(canonical(result))
    (state / "selection.json").write_bytes(canonical(selected))
    sign_fixture(config, selected, admission, key)
    outcome = {"exit_code": 0, "termination_confirmed": True, "schema": "synthetic-cpu-test"}
    class Store:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    class Owner:
        def __init__(self, records): pass
        def observe(self, invocation): return {"state": "terminated", "outcome": outcome}
    monkeypatch.setattr(host, "SQLiteDurableStoreV1", Store)
    monkeypatch.setattr(host, "NativeOwnedProcess", Owner)
    return SimpleNamespace(root=root, config=config, selection=selected, admission=admission,
        key=key, workload=w, raw_cases=raw, result=result, outcome=outcome, state=state,
        invocation=object(), pin=config["root_public_key_sha256"])


@pytest.mark.parametrize("correctness_form", ["any", "all", "assertions"])
def test_complete_export_portable_scoring_and_collision_preservation(complete_evidence, correctness_form):
    p = complete_evidence
    receipt = host.readback(p.config, p.selection, p.invocation)
    assert receipt["case_count"] == 1 and not receipt["training_submitted"]
    result = host.export(p.config, p.selection, p.invocation, p.root / "bundle")
    selected, witness_receipt, rows, witness = preparation.evidence_bundle(p.root / "bundle", result["manifest_sha256"],
        trusted_root_key_sha256=p.pin)
    assert selected == p.selection and witness_receipt == receipt and rows[0]["text"] == "answer"
    assert "not portable provider authentication" in witness
    before = {f.name: f.read_bytes() for f in (p.root / "bundle").iterdir()}
    with pytest.raises(FileExistsError): host.export(p.config, p.selection, p.invocation, p.root / "bundle")
    assert {f.name: f.read_bytes() for f in (p.root / "bundle").iterdir()} == before
    case = read_cases(p.raw_cases, p.workload)[0]
    assertions = [{"type": "jsonpath_equals", "path": "$.content", "value": "answer"}]
    correct = ({"any": [{"assertions": assertions}]} if correctness_form == "any" else {correctness_form: assertions})
    scoring = canonical([{"id": case["id"], "question": "question", "messages": case["messages"],
        "tags": ["declared"], "correct": correct}])
    (p.root / "scoring.json").write_bytes(scoring)
    config = {"schema": "synaptic-native-evaluation-scoring/v1", "evidence_bundle": str(p.root / "bundle"),
        "evidence_manifest_sha256": result["manifest_sha256"], "scoring_cases_file": str(p.root / "scoring.json"),
        "scoring_cases_sha256": host.sha(scoring), "validator": {"wrapper_specs": [], "tool_schemas": {
            "configured": {"parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}}},
        "label": "synthetic CPU record", "output_root": str(p.root / "scored")}
    scored = preparation.score(config, trusted_root_key_sha256=p.pin)
    assert scored["case_count"] == 1 and scored["model_executed"] is False
    payload = json.loads((p.root / "scored/evaluation_results.json").read_bytes())
    assert payload["metadata"]["native_reference_metrics"]["declared"]["token_weighted_reference_loss"] == 1.5
    with pytest.raises(FileExistsError): preparation.score(config, trusted_root_key_sha256=p.pin)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("field", ["retrieval_config", "audio_config", "environment", "judge",
    "response_schema", "expected_context", "scoring", "future_verifier", "missing_correct",
    "wrapper_ref", "tool_dynamic_ref", "tool_recursive_ref", "correct_empty_any", "correct_typo_any",
    "correct_unknown_only", "correct_empty_all", "correct_empty_assertions", "correct_nested_empty",
    "correct_negated_unknown"])
def test_scoring_rejects_unadmitted_modes_before_any_case_dispatch(complete_evidence, monkeypatch, index, field):
    from Evaluator import runner
    from Evaluator.native_mlx_retained import NativeMLXRetainedClient
    p = complete_evidence
    # Extend this test-owned synthetic witness to two cases, so a bad later
    # case must refuse before even the earlier retained response is consumed.
    cases = read_cases(p.raw_cases, p.workload)
    second = deepcopy(cases[0]);second["id"] = "held-out-2"
    cases.append(second)
    p.raw_cases = canonical({"schema": "synaptic-native-evaluation-cases/v1", "cases": cases})
    p.workload.update(case_count=2, cases_sha256=host.sha(p.raw_cases))
    (p.state / "cases.json").write_bytes(p.raw_cases)
    raw_record = canonical(row_for(second))
    (p.state / "output/case-0001.json").write_bytes(raw_record)
    p.result.update(case_count=2, case_bytes=p.result["case_bytes"] + len(raw_record),
        workload_sha256=host.sha(canonical(p.workload)))
    p.result["case_files"].append({"name": "case-0001.json", "size": len(raw_record), "sha256": host.sha(raw_record)})
    (p.state / "output/result.json").write_bytes(canonical(p.result))
    (p.state / "selection.json").write_bytes(canonical(p.selection))
    sign_fixture(p.config, p.selection, p.admission, p.key)
    result = host.export(p.config, p.selection, p.invocation, p.root / "bundle")
    scoring = [{"id": case["id"], "question": "question", "messages": case["messages"],
        "correct": {"any": [{"assertions": [
            {"type": "jsonpath_equals", "path": "$.content", "value": "answer"}]}]}} for case in cases]
    if field in {"wrapper_ref", "tool_dynamic_ref", "tool_recursive_ref"}: pass
    elif field.startswith("correct_"):
        scoring[index]["correct"] = {
            "correct_empty_any": {"any": []}, "correct_typo_any": {"any": "typo"},
            "correct_unknown_only": {"future_verifier": {"mode": "new"}},
            "correct_empty_all": {"all": []}, "correct_empty_assertions": {"assertions": []},
            "correct_nested_empty": {"assertions": [{"type": "all", "assertions": []}]},
            "correct_negated_unknown": {"assertions": [{"type": "not", "assertion": {"type": "future_verifier"}}]},
        }[field]
    elif field == "missing_correct": del scoring[index]["correct"]
    elif field == "retrieval_config": scoring[index][field] = {"model": "unadmitted/model",
        "corpus": "unused", "queries": "unused", "qrels": "unused", "metrics": ["recall@1"]}
    elif field == "audio_config": scoring[index][field] = {"audio_paths": ["unused"]}
    else: scoring[index][field] = {"unadmitted": True}
    raw = canonical(scoring)
    (p.root / "scoring.json").write_bytes(raw)
    config = {"schema": "synaptic-native-evaluation-scoring/v1", "evidence_bundle": str(p.root / "bundle"),
        "evidence_manifest_sha256": result["manifest_sha256"], "scoring_cases_file": str(p.root / "scoring.json"),
        "scoring_cases_sha256": host.sha(raw), "validator": {"wrapper_specs": [], "tool_schemas": {}},
        "label": "synthetic CPU record", "output_root": str(p.root / "scored")}
    if field == "wrapper_ref": config["validator"]["wrapper_specs"] = [{"properties": {
        "value": {"$ref": "https://unadmitted.invalid/schema"}}}]
    elif field in {"tool_dynamic_ref", "tool_recursive_ref"}:
        keyword = "$dynamicRef" if field == "tool_dynamic_ref" else "$recursiveRef"
        config["validator"]["tool_schemas"] = {"configured": {"properties": {"value": {keyword: "#/local"}}}}
    calls = []
    def forbidden(*a, **kw):
        calls.append(True)
        pytest.fail("unadmitted evaluator/model/media/client dispatch")
    monkeypatch.setattr(runner, "evaluate_cases", forbidden)
    monkeypatch.setattr(runner.RetrievalVerifier, "evaluate_retrieval", forbidden)
    monkeypatch.setattr(runner.AudioVerifier, "evaluate_audio", forbidden)
    monkeypatch.setattr(NativeMLXRetainedClient, "chat", forbidden)
    refusal = ("validator must be inline" if field.endswith("ref") else
        "correctness declaration differs" if field.startswith("correct_") else "scoring metadata differs")
    with pytest.raises(ValueError, match=refusal):
        preparation.score(config, trusted_root_key_sha256=p.pin)
    assert not calls and not (p.root / "scored").exists()


@pytest.mark.parametrize("form", ["any", "all", "assertions", "nested"])
def test_admitted_correctness_forms_preserve_actual_assertions(form):
    from shared.verifiers.builtins.assertion_verifier import evaluate_correctness
    leaf = {"type": "jsonpath_equals", "path": "$.content", "value": "answer"}
    if form == "any": correct = {"any": [{"name": "declared", "assertions": [leaf]}]}
    elif form == "nested": correct = {"assertions": [{"type": "all", "assertions": [
        {"type": "any", "assertions": [leaf, {"type": "text_contains", "text": "other"}]},
        {"type": "not", "assertion": {"type": "text_regex", "pattern": "^wrong$"}}]}]}
    else: correct = {form: [leaf]}
    preparation.validate_correctness_declaration(correct)
    result = evaluate_correctness(correct, {"content": "answer"})
    assert result.passed and result.matched_path is not None and result.paths
    assert not evaluate_correctness(correct, {"content": "wrong"}).passed


@pytest.mark.parametrize("bad", [None, {}, {"any": [None]}, {"any": [{}]}, {"any": [{"assertions": [None]}]},
    {"any": [{"assertions": [{"type": "exists"}], "typo": True}]},
    {"all": [{"type": "exists"}], "assertions": [{"type": "exists"}]},
    {"assertions": [{"type": "equals"}]}, {"assertions": [{"type": "regex"}]},
    {"assertions": [{"type": "regex", "pattern": "["}]},
    {"assertions": [{"type": "length_min", "value": True}]},
    {"assertions": [{"type": "contains", "value": "a", "text": "b"}]},
    {"assertions": [{"type": "exists", "path": "$.a", "selector": "$.b"}]},
    {"assertions": [{"type": "exists", "typo": True}]},
    {"assertions": [{"type": "any", "assertions": [None]}]},
    {"assertions": [{"type": "not", "assertion": {}}]}])
def test_malformed_correctness_trees_refuse_without_evaluating(bad):
    with pytest.raises(ValueError, match="correctness declaration differs"):
        preparation.validate_correctness_declaration(bad)


@pytest.mark.parametrize("kind", ["exists", "absent", "equals", "not_equals", "contains", "not_contains",
    "regex", "not_regex", "length_equals", "length_min", "length_max", "json_subset"])
def test_existing_leaf_operations_and_aliases_are_admitted(kind):
    from shared.verifiers.builtins.assertion_verifier import evaluate_correctness
    fields = {"exists": {}, "absent": {}, "equals": {"value": "answer"}, "not_equals": {"value": "other"},
        "contains": {"text": "ans"}, "not_contains": {"value": "other"},
        "regex": {"pattern": "^answer$"}, "not_regex": {"pattern": "^other$"},
        "length_equals": {"length": 6}, "length_min": {"min": 6}, "length_max": {"max": 6},
        "json_subset": {"value": "answer"}}[kind]
    aliases = [kind] if kind == "json_subset" else [kind, "jsonpath_" + kind]
    if kind in {"contains", "not_contains", "regex", "not_regex"}: aliases.append("text_" + kind)
    for alias in aliases:
        correct = {"assertions": [{"type": alias, "selector": "$.missing" if kind == "absent" else "$.content", **fields}]}
        preparation.validate_correctness_declaration(correct)
        result = evaluate_correctness(correct, {"content": "answer"})
        assert result.passed and result.paths and result.paths[0].assertions


def test_correctness_list_depth_and_node_limits_have_exact_boundaries():
    leaf = {"type": "exists"}
    preparation.validate_correctness_declaration({"assertions": [leaf] * 128})
    with pytest.raises(ValueError): preparation.validate_correctness_declaration({"assertions": [leaf] * 129})
    nested = leaf
    for _ in range(32): nested = {"type": "not", "assertion": nested}
    preparation.validate_correctness_declaration({"assertions": [nested]})
    with pytest.raises(ValueError):
        preparation.validate_correctness_declaration({"assertions": [{"type": "not", "assertion": nested}]})
    groups = [{"type": "all", "assertions": [leaf] * 63} for _ in range(64)]
    preparation.validate_correctness_declaration({"assertions": groups})  # 64*(1+63) = 4096 nodes.
    with pytest.raises(ValueError): preparation.validate_correctness_declaration({"assertions": [*groups, leaf]})


@pytest.mark.parametrize("fault", ["manifest", "case", "directory", "receipt", "result", "extra_file", "key", "signature"])
def test_portable_evidence_refuses_tampering_even_with_new_manifest_hash(complete_evidence, fault):
    p = complete_evidence
    destination = p.root / "bundle"
    result = host.export(p.config, p.selection, p.invocation, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if fault == "manifest": manifest["schema"] = "different"
    elif fault == "directory": (destination / "unexpected").mkdir(mode=0o700)
    elif fault == "extra_file": (destination / "extra").write_bytes(b"extra")
    else:
        name = {"case": "case-0000.json", "receipt": "host-readback.json", "result": "result.json",
            "key": "root-public-key.pem", "signature": "admission.sig"}[fault]
        raw = (destination / name).read_bytes()
        if fault in {"key", "signature"}: raw += b"changed"
        else:
            value = json.loads(raw)
            if fault == "case": value["completion_tokens"] = [0, 0]
            elif fault == "receipt": value["training_submitted"] = True
            else: value["case_count"] = True
            raw = canonical(value)
        (destination / name).write_bytes(raw)
        member = next(m for m in manifest["members"] if m["name"] == name)
        member.update(size=len(raw), sha256=host.sha(raw))
    manifest_path.write_bytes(canonical(manifest))
    with pytest.raises(ValueError):
        preparation.evidence_bundle(destination, host.sha(canonical(manifest)), trusted_root_key_sha256=p.pin)
    assert result["model_executed"] is False


@pytest.mark.parametrize("fault", ["uncertain", "exit", "termination", "case", "directory", "selection"])
def test_live_readback_requires_owned_complete_unchanged_evidence(complete_evidence, fault):
    p = complete_evidence
    if fault == "uncertain": p.outcome.clear()
    elif fault == "exit": p.outcome["exit_code"] = 154
    elif fault == "termination": p.outcome["termination_confirmed"] = False
    elif fault == "case": (p.state / "output/case-0000.json").write_bytes(b"changed")
    elif fault == "directory": (p.state / "output/extra").mkdir(mode=0o700)
    else: (p.state / "selection.json").write_bytes(b"changed")
    with pytest.raises(ValueError): host.readback(p.config, p.selection, p.invocation)


def test_execute_claims_once_and_never_replays_uncertain_submission(signed_selection, monkeypatch):
    root, config, selected, admission, key = signed_selection
    selected["request"] = {"schema": "synthetic CPU request"}
    sign_fixture(config, selected, admission, key)
    config_path = root / "config.json"; config_path.write_bytes(canonical(config))
    monkeypatch.setattr(host, "plan", lambda *a, **kw: (selected, object(), b"capture", b"lock", b"cases", {}))
    calls = []
    class Store:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    class Owner:
        def __init__(self, records): pass
        def run_once(self, invocation):
            calls.append(invocation)
            raise RuntimeError("synthetic uncertain acknowledgement")
    monkeypatch.setattr(host, "SQLiteDurableStoreV1", Store)
    monkeypatch.setattr(host, "NativeOwnedProcess", Owner)
    with pytest.raises(RuntimeError): host.run("execute", config_path, trusted_root_key_sha256=config["root_public_key_sha256"])
    retained = {p.name: p.read_bytes() for p in Path(config["state_root"]).iterdir() if p.is_file()}
    with pytest.raises(FileExistsError): host.run("execute", config_path, trusted_root_key_sha256=config["root_public_key_sha256"])
    assert len(calls) == 1
    assert {p.name: p.read_bytes() for p in Path(config["state_root"]).iterdir() if p.is_file()} == retained


def test_existing_canonicalizer_destination_is_create_only(private_root, capsys):
    from examples.training_qualification import native_host
    source, destination = private_root / "edited.json", private_root / "canonical.json"
    source.write_text('{ "b": 2, "a": 1 }\n')
    assert native_host.main(["canonicalize", str(source), "--destination", str(destination)]) == 0
    assert destination.read_bytes() == b'{"a":1,"b":2}'
    assert destination.stat().st_mode & 0o777 == 0o600
    capsys.readouterr()
    assert native_host.main(["canonicalize", str(source), "--destination", str(destination)]) == 2
    assert destination.read_bytes() == b'{"a":1,"b":2}'


@pytest.fixture
def planned_host(private_root, monkeypatch):
    from dataclasses import asdict
    from tuner.runtime.native_files import NativeFile
    from tuner.runtime.native_mlx_input import NativeModelSnapshot
    w, cases = selection()
    capture = {"trees": [{"root": str(host.ROOT), "members": [
        {"path": "tuner/runtime/native_process_guardian.py", "sha256": "a" * 64},
        {"path": "tuner/runtime/native_mlx_child.py", "sha256": "b" * 64}]}],
        "executable": "/private/executable", "runtime": {"executable_sha256": "c" * 64}}
    (private_root / "capture.json").write_bytes(canonical(capture))
    (private_root / "lock").write_bytes(b"synthetic lock")
    (private_root / "cases.json").write_bytes(cases)
    snapshot = NativeModelSnapshot("selected/model", "c" * 40, str(private_root / "model"),
        (NativeFile("config.json", 2, host.sha(b"{}")),))
    monkeypatch.setattr(host, "source_record", lambda *a, **kw: ({"synthetic": True}, "b" * 64))
    monkeypatch.setattr(host, "hardware_record", lambda: ({"synthetic": True}, "d" * 64))
    monkeypatch.setattr(host, "_snapshot", lambda *a: snapshot)
    monkeypatch.setattr(host, "verify_native_material", lambda *a, **kw: None)
    config = {"schema": "synaptic-native-evaluation-host/v1", "engine_root": str(host.ROOT),
        "engine_commit": "d" * 40, "engine_ref": "synthetic", "engine_remote": "synthetic",
        "runtime_capture_file": str(private_root / "capture.json"), "dependency_lock_file": str(private_root / "lock"),
        "model_snapshot_file": str(private_root / "model.json"), "tokenizer_snapshot_file": str(private_root / "tokenizer.json"),
        "cases_file": str(private_root / "cases.json"), "adapter_file": None,
        "profile": {k: v for k, v in w["profile"].items() if k not in {"runtime_digest", "source_digest"}},
        "generation": w["generation"], "state_root": str(private_root / "state"), "attempt": "synthetic-plan",
        "limits": {"timeout_seconds": 30, "termination_grace_seconds": 5, "maximum_output_bytes": 4096,
            "maximum_file_bytes": 1 << 20}, "admission_file": str(private_root / "admission.json"),
        "admission_signature_file": str(private_root / "admission.sig"), "root_public_key_file": str(private_root / "public.pem"),
        "root_public_key_sha256": "e" * 64}
    return private_root, config, asdict(snapshot)


def test_host_plan_binds_exact_inputs_without_creating_state(planned_host):
    root, config, snapshot = planned_host
    selected, invocation, capture, lock, cases, files = host.plan(config)
    assert selected["request"]["model_snapshot"] == snapshot
    assert invocation.binding_digest == host.sha(canonical(selected))
    assert invocation.request_digest == host.sha(canonical(selected["request"]))
    assert invocation.limits.timeout_seconds == 30 and not files and not (root / "state").exists()
    config["generation"]["seed"] = 18
    assert host.plan(config)[1].binding_digest != invocation.binding_digest


@pytest.mark.parametrize("fault", ["occupied", "profile", "bounds", "cases", "attempt", "unknown"])
def test_host_plan_rejects_incompatible_or_claimed_selections_before_effects(planned_host, fault):
    root, config, _ = planned_host
    if fault == "occupied": (root / "state").mkdir(mode=0o700)
    elif fault == "profile": config["profile"]["model_revision"] = "f" * 40
    elif fault == "bounds": config["limits"]["maximum_file_bytes"] = 100
    elif fault == "cases": (root / "cases.json").write_bytes(b"{}")
    elif fault == "attempt": config["attempt"] = "../arbitrary"
    else: config["extra"] = True
    with pytest.raises((ValueError, KeyError)): host.plan(config)
    assert not (root / "state/workflow.sqlite3").exists()


def test_native_adapter_archive_checks_real_production_format():
    from types import SimpleNamespace
    from tests.runtime.test_native_mlx_publication import tensor_archive
    from tuner.runtime.native_mlx_publication import _archive, NativeArtifactBudget
    tensors = tensor_archive(.25)
    model = SimpleNamespace(repository="selected/model", revision="c" * 40, digest="a" * 64)
    metadata = {"schema": "synaptic-native-mlx-model/v1", "format": "mlx-lm-lora-safetensors",
        "repository": model.repository, "revision": model.revision, "snapshot_digest": model.digest,
        "runtime_digest": "d" * 64, "workload_digest": "e" * 64}
    config = {"fine_tune_type": "lora", "num_layers": 2, "lora_parameters": {
        "rank": 4, "scale": 8, "dropout": 0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}}
    values = {"adapters.safetensors": tensors, "0000050_adapters.safetensors": tensors,
        "native_model.json": canonical(metadata), "adapter_config.json": canonical(config)}
    budget = NativeArtifactBudget(1 << 20, 4 << 20)
    archive = _archive(values, budget)
    files, declared = host.adapter_files(archive, model=model)
    assert files == values and declared["adapter_sha256"] == host.sha(tensors)
    assert declared["lora"]["rank"] == 4 and declared["artifact_sha256"] == host.sha(archive)
    values["../outside"] = b"unsafe"
    with pytest.raises(ValueError): host.adapter_files(_archive(values, budget), model=model)
    del values["../outside"]
    metadata["snapshot_digest"] = "f" * 64;values["native_model.json"] = canonical(metadata)
    with pytest.raises(ValueError): host.adapter_files(_archive(values, budget), model=model)
