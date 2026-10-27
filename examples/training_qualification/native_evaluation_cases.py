"""Prepare held-out cases and score retained native responses via existing APIs.

Source selection and score paths are configuration, never model/runtime rules.
This module loads no model, executes no generated command and calls no judge.
"""
from copy import deepcopy
import argparse
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from native_evaluation import direct_names, document, mkdir, verify_admission, sha
from tuner.dataset_prep.conversation_messages import verify_bytes
from tuner.runtime.native_files import NativeTree, file_identity
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.runtime.native_mlx_evaluation import parse_workload, read_cases, validate_record, validate_result


def publication(path):
    with NativeTree(path, private=True) as tree:
        if direct_names(tree, 2) != ("dataset.jsonl", "manifest.json"):
            raise ValueError("native evaluation publication inventory differs")
        before = {name: file_identity(os.stat(name, dir_fd=tree.fd, follow_symlinks=False))
                  for name in ("dataset.jsonl", "manifest.json")}
        raw, manifest = tree.read("dataset.jsonl", maximum=8 << 20), tree.read("manifest.json", maximum=1 << 20)
        verified = verify_bytes(path, manifest, raw)
        if direct_names(tree, 2) != ("dataset.jsonl", "manifest.json") or before != {
                name: file_identity(os.stat(name, dir_fd=tree.fd, follow_symlinks=False)) for name in before}:
            raise ValueError("native evaluation publication changed")
        tree.check()
    return raw, verified


def prepare(config):
    if (type(config) is not dict or set(config) != {"schema", "publications", "output_root"}
            or config["schema"] != "synaptic-native-evaluation-preparation/v1"
            or type(config["publications"]) is not list or not 1 <= len(config["publications"]) <= 8):
        raise ValueError("native evaluation case preparation differs")
    from Evaluator.response_view import build_response_view
    from shared.verifiers.builtins.assertion_verifier import select_path, evaluate_correctness, _MISSING

    cases, scoring, train, evidence, labels = [], [], [], [], set()
    for source in config["publications"]:
        if (type(source) is not dict or set(source) != {"label", "publication", "dataset_digest",
                "content_sha256", "expected_train_rows", "expected_validation_rows", "reference_equals", "assertions"}
                or type(source["label"]) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", source["label"])
                or source["label"] in labels or type(source["reference_equals"]) is not list
                or len(source["reference_equals"]) > 16 or type(source["assertions"]) is not list
                or len(source["assertions"]) > 16):
            raise ValueError("native evaluation source declaration differs")
        labels.add(source["label"])
        raw, verified = publication(Path(source["publication"]))
        identity = verified.semantic_identity
        if (identity.dataset_digest != source["dataset_digest"] or sha(raw) != source["content_sha256"]
                or type(source["expected_train_rows"]) is not int or type(source["expected_validation_rows"]) is not int):
            raise ValueError("native evaluation source identity differs")
        rows = [strict_json(row, maximum=1 << 20) for row in raw.splitlines()]
        splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "validation")}
        if (len(splits["train"]) != source["expected_train_rows"]
                or len(splits["validation"]) != source["expected_validation_rows"]
                or len(splits["train"]) + len(splits["validation"]) != len(rows)
                or {r["group_id"] for r in splits["train"]} & {r["group_id"] for r in splits["validation"]}):
            raise ValueError("native evaluation held-out coverage differs")
        train.extend({"messages": row["messages"]} for row in splits["train"])
        for row in splits["validation"]:
            messages = deepcopy(row["messages"])
            if (row["target_message_index"] != len(messages) - 1 or messages[-1]["role"] != "assistant"
                    or len(messages) < 2 or messages[-2]["role"] != "user"):
                raise ValueError("native evaluation final target boundary differs")
            reference, prompt = messages[-1]["content"], messages[:-1]
            case_id = source["label"] + "-" + row["row_id"]
            case = {"id": case_id, "messages": prompt, "reference": reference}
            if len(case_id) > 128 or len(cases) >= 128:
                raise ValueError("native evaluation prepared case bound exceeded")
            view, assertions = build_response_view(reference), deepcopy(source["assertions"])
            for path in source["reference_equals"]:
                if type(path) is not str or len(path) > 256:
                    raise ValueError("native evaluation reference score path differs")
                value = select_path(view, path)
                if value is _MISSING:
                    raise ValueError("native evaluation reference score path is absent")
                assertions.append({"type": "jsonpath_equals", "path": path, "value": deepcopy(value)})
            if not assertions:
                raise ValueError("native evaluation requires configured assertions")
            correct = {"any": [{"name": "configured", "assertions": assertions}]}
            validate_correctness_declaration(correct)
            if not evaluate_correctness(correct, view).passed:
                raise ValueError("native evaluation reference does not satisfy declared assertions")
            cases.append(case)
            scoring.append({"id": case_id, "question": prompt[-1]["content"], "messages": prompt,
                "tags": [source["label"]], "correct": correct,
                "provenance": {"dataset_digest": identity.dataset_digest, "row_id": row["row_id"],
                    "group_id": row["group_id"], "split": "validation"}})
        evidence.append({"label": source["label"], "dataset_digest": identity.dataset_digest,
            "content_sha256": sha(raw), "train_rows": len(splits["train"]), "validation_rows": len(splits["validation"])})
    output = Path(config["output_root"])
    mkdir(output)
    values = {"configuration.json": canonical(config),
        "cases.json": canonical({"schema": "synaptic-native-evaluation-cases/v1", "cases": cases}),
        "scoring-cases.json": canonical(scoring),
        "training-for-contamination.jsonl": b"".join(canonical(row) + b"\n" for row in train),
        "heldout-prompts.jsonl": b"".join(canonical({"text": "\n".join(m["role"] + ":" + m["content"]
            for m in case["messages"])}) + b"\n" for case in cases)}
    if sum(len(v) for v in values.values()) > 16 << 20:
        raise ValueError("native evaluation preparation bytes exceeded")
    with NativeTree(output, private=True) as tree:
        for name, raw in values.items():
            tree.write_exclusive(name, raw, maximum=8 << 20)
        result = {"schema": "synaptic-native-evaluation-preparation-result/v1", "case_count": len(cases),
            "case_sha256": sha(values["cases.json"]), "scoring_sha256": sha(values["scoring-cases.json"]),
            "sources": evidence, "output_root": str(output), "training_submitted": False}
        tree.write_exclusive("result.json", canonical(result))
    return result


def evidence_bundle(path, expected_digest, *, trusted_root_key_sha256):
    """Verify integrity/admission without claiming portable model authentication."""
    path = Path(path)
    with NativeTree(path, private=True) as tree:
        names = direct_names(tree, 140)
        raw_manifest = tree.read("manifest.json", maximum=1 << 20)
        if sha(raw_manifest) != expected_digest:
            raise ValueError("native evaluation exported manifest digest differs")
        manifest = strict_json(raw_manifest)
        if (type(manifest) is not dict or set(manifest) != {"schema", "selection_digest", "members", "execution_witness"}
                or manifest["schema"] != "synaptic-native-evaluation-export/v1"
                or manifest["execution_witness"] != "operator-retained owned-process receipt; not portable provider authentication"
                or type(manifest["members"]) is not list or not 1 <= len(manifest["members"]) <= 135):
            raise ValueError("native evaluation exported manifest differs")
        files, total = {}, 0
        for item in manifest["members"]:
            if (type(item) is not dict or set(item) != {"name", "size", "sha256"}
                    or type(item["name"]) is not str or item["name"] not in names
                    or item["name"] in files or item["name"] == "manifest.json"
                    or type(item["size"]) is not int or not 0 < item["size"] <= 32 << 20):
                raise ValueError("native evaluation exported member differs")
            total += item["size"]
            if total > 64 << 20:
                raise ValueError("native evaluation exported aggregate exceeded")
            raw = tree.read(item["name"], maximum=item["size"])
            if len(raw) != item["size"] or sha(raw) != item["sha256"]:
                raise ValueError("native evaluation exported bytes differ")
            files[item["name"]] = raw
        if set(names) != {"manifest.json", *files}:
            raise ValueError("native evaluation exported inventory differs")
        selection = strict_json(files["selection.json"], maximum=4 << 20)
        if sha(canonical(selection)) != manifest["selection_digest"]:
            raise ValueError("native evaluation exported selection differs")
        verify_admission(selection, admission_file=path / "admission.json", signature_file=path / "admission.sig",
            key_file=path / "root-public-key.pem", trusted_root_key_sha256=trusted_root_key_sha256, starting=False)
        result, receipt = strict_json(files["result.json"]), strict_json(files["host-readback.json"])
        workload = parse_workload(canonical(selection["request"]["workload"]))
        validate_result(result, selection["request"])
        cases = read_cases(files["cases.json"], workload)
        if (type(receipt) is not dict or set(receipt) != {"schema", "status", "selection_digest", "result_sha256",
                "case_count", "outcome", "training_submitted"}
                or receipt["schema"] != "synaptic-native-evaluation-readback/v1" or receipt["training_submitted"] is not False
                or type(receipt["case_count"]) is not int or type(receipt["outcome"]) is not dict
                or type(receipt["outcome"].get("exit_code")) is not int
                or receipt["status"] != "COMPLETE" or receipt["selection_digest"] != manifest["selection_digest"]
                or receipt["result_sha256"] != sha(files["result.json"]) or receipt["case_count"] != len(cases)
                or receipt["outcome"].get("exit_code") != 0 or receipt["outcome"].get("termination_confirmed") is not True
                or result["status"] != "COMPLETE" or result["case_count"] != len(cases)
                or result["workload_sha256"] != sha(canonical(workload)) or len(result["case_files"]) != len(cases)):
            raise ValueError("native evaluation exported completion differs")
        for role in ("model", "tokenizer", "adapter"):
            snapshot = selection["request"][role + "_snapshot"]
            if result[role + "_snapshot_digest"] != (sha(canonical(snapshot)) if snapshot is not None else None):
                raise ValueError("native evaluation exported snapshot differs")
        rows, count = [], 0
        for index, (member, case) in enumerate(zip(result["case_files"], cases)):
            name = f"case-{index:04d}.json"
            if member["name"] != name or member["size"] != len(files[name]) or member["sha256"] != sha(files[name]):
                raise ValueError("native evaluation exported case manifest differs")
            row = strict_json(files[name])
            validate_record(row, case, workload)
            rows.append(row)
            count += len(files[name])
        expected = {"selection.json", "host-readback.json", "cases.json", "result.json", "admission.json",
            "admission.sig", "root-public-key.pem", *(f"case-{i:04d}.json" for i in range(len(cases)))}
        if (set(files) != expected or result["case_bytes"] != count or count > workload["profile"]["maximum_evidence_bytes"]
                or tree.read("manifest.json", maximum=1 << 20) != raw_manifest or direct_names(tree, 140) != names):
            raise ValueError("native evaluation exported final inventory differs")
    return selection, receipt, rows, manifest["execution_witness"]


def validate_correctness_declaration(config):
    """Admit nonempty assertion trees; keep shared permissive semantics unchanged."""
    def refuse():
        raise ValueError("native evaluation correctness declaration differs")

    def named(value):
        if "name" in value and (type(value["name"]) is not str or not 1 <= len(value["name"]) <= 512):
            refuse()

    if type(config) is not dict:
        refuse()
    modes = set(config) & {"any", "all", "assertions"}
    if len(modes) != 1 or set(config) - (modes | {"name"}):
        refuse()
    named(config)
    mode = next(iter(modes))
    paths = config[mode] if mode == "any" else [{"assertions": config[mode]}]
    if type(paths) is not list or not 1 <= len(paths) <= 128:
        refuse()
    pending = []
    for path in paths:
        if type(path) is not dict or set(path) - {"name", "assertions"} or "assertions" not in path:
            refuse()
        named(path)
        pending.append((path["assertions"], 0))
    leaves = {kind: kind for kind in ("exists", "absent", "equals", "not_equals", "contains",
        "not_contains", "regex", "not_regex", "length_equals", "length_min", "length_max")}
    leaves.update({"jsonpath_" + kind: kind for kind in tuple(leaves)})
    leaves.update({"text_" + kind: kind for kind in ("contains", "not_contains", "regex", "not_regex")})
    leaves["json_subset"] = "json_subset"
    count = 0
    while pending:
        assertions, depth = pending.pop()
        if type(assertions) is not list or not 1 <= len(assertions) <= 128 or depth > 32:
            refuse()
        for assertion in assertions:
            count += 1
            if count > 4096 or type(assertion) is not dict or type(assertion.get("type")) is not str:
                refuse()
            named(assertion)
            kind = assertion["type"]
            if kind in {"all", "any", "not"}:
                field = "assertion" if kind == "not" else "assertions"
                if set(assertion) - {"name", "type", field} or field not in assertion:
                    refuse()
                pending.append(([assertion[field]] if kind == "not" else assertion[field], depth + 1))
                continue
            if kind not in leaves:
                refuse()
            operation = leaves[kind]
            selectors = set(assertion) & {"path", "selector"}
            if len(selectors) > 1 or any(type(assertion[key]) is not str or not assertion[key].strip() for key in selectors):
                refuse()
            fields = set()
            if operation in {"equals", "not_equals", "json_subset"}:
                fields = {"value"}
            elif operation in {"contains", "not_contains"}:
                fields = {"value", "text"}
            elif operation in {"regex", "not_regex"}:
                fields = {"pattern"}
            elif operation.startswith("length_"):
                fields = {"value", {"length_equals": "length", "length_min": "min", "length_max": "max"}[operation]}
            selected = set(assertion) & fields
            if (fields and len(selected) != 1) or set(assertion) - ({"name", "type", "path", "selector"} | fields):
                refuse()
            if operation in {"regex", "not_regex"}:
                if type(assertion["pattern"]) is not str:
                    refuse()
                try:
                    re.compile(assertion["pattern"])
                except re.error:
                    refuse()
            if operation.startswith("length_"):
                number = assertion[next(iter(selected))]
                if type(number) is not int or number < 0:
                    refuse()


def score(config, *, trusted_root_key_sha256):
    if (type(config) is not dict or set(config) != {"schema", "evidence_bundle", "evidence_manifest_sha256", "scoring_cases_file",
            "scoring_cases_sha256", "validator", "label", "output_root"}
            or config["schema"] != "synaptic-native-evaluation-scoring/v1"):
        raise ValueError("native evaluation scoring declaration differs")
    selection, receipt, rows, witness = evidence_bundle(config["evidence_bundle"], config["evidence_manifest_sha256"],
        trusted_root_key_sha256=trusted_root_key_sha256)
    from Evaluator.comparison_basis import backend_controls, build_basis
    from Evaluator.native_mlx_retained import NativeMLXRetainedClient
    from Evaluator.prompt_sets import load_prompt_cases
    from Evaluator.reporting import build_run_payload
    from Evaluator.runner import evaluate_cases
    from shared.validation.dataset_validator import ValidatorConfig

    raw_scoring = Path(config["scoring_cases_file"])
    with NativeTree(raw_scoring.parent) as tree:
        raw = tree.read(raw_scoring.name, maximum=2 << 20)
        if sha(raw) != config["scoring_cases_sha256"]:
            raise ValueError("native evaluation scoring bytes differ")
        cases = load_prompt_cases(raw_scoring)
        if tree.read(raw_scoring.name, maximum=2 << 20) != raw:
            raise ValueError("native evaluation scoring changed")
    # Validate the entire collection before dispatch: the shared evaluator also
    # supports corpus/model/media modes that bypass the retained chat client.
    # This host admits only configured assertions on already generated text.
    for case in cases:
        if set(case.metadata) - {"messages", "correct", "provenance"} or "correct" not in case.metadata:
            raise ValueError("native evaluation scoring metadata differs")
        validate_correctness_declaration(case.metadata["correct"])
    workload = selection["request"]["workload"]
    if len(rows) != receipt["case_count"]:
        raise ValueError("native evaluation scoring coverage differs")
    if len(cases) != len(rows) or any(row["id"] != case.case_id
            or row["messages_sha256"] != sha(canonical(case.chat_messages())) for case, row in zip(cases, rows)):
        raise ValueError("native evaluation scoring case join differs")
    validator = config["validator"]
    if (type(validator) is not dict or set(validator) != {"wrapper_specs", "tool_schemas"}
            or type(validator["wrapper_specs"]) is not list or type(validator["tool_schemas"]) is not dict):
        raise ValueError("native evaluation validator differs")
    # JSON Schema references can retrieve resources even without a model or
    # environment validator. This offline host admits inline schemas only.
    pending = [validator]
    while pending:
        node = pending.pop()
        if type(node) is dict:
            if set(node) & {"$ref", "$dynamicRef", "$recursiveRef"}:
                raise ValueError("native evaluation validator must be inline")
            pending.extend(node.values())
        elif type(node) is list:
            pending.extend(node)
    validator_config = ValidatorConfig(validator["wrapper_specs"], validator["tool_schemas"])
    client = NativeMLXRetainedClient(rows, generation=workload["generation"], profile=workload["profile"])
    records = evaluate_cases(cases, client, parallel=False, validator_config=validator_config)
    if len(records) != receipt["case_count"] or client.records_consumed != len(records):
        raise ValueError("native evaluation scoring incomplete")
    controls = backend_controls(client)
    reference_metrics = {}
    for case, row in zip(cases, rows):
        for tag in case.tags:
            stats = reference_metrics.setdefault(tag, {"rows": 0, "stop": 0, "length": 0,
                "reference_rows": 0, "supervised_tokens": 0, "loss_sum": 0.0})
            stats["rows"] += 1
            stats[row["finish_reason"]] += 1
            if row["reference"] is not None:
                stats["reference_rows"] += 1
                stats["supervised_tokens"] += row["reference"]["supervised_tokens"]
                stats["loss_sum"] += row["reference"]["loss"] * row["reference"]["supervised_tokens"]
    for stats in reference_metrics.values():
        stats["token_weighted_reference_loss"] = stats.pop("loss_sum") / stats["supervised_tokens"] if stats["supervised_tokens"] else None
    metadata = {"model": config["label"], "backend": "native_mlx_retained",
        "temperature": 0, "max_tokens": workload["generation"]["max_completion_tokens"],
        "seed": workload["generation"]["seed"], "chat_template_kwargs": workload["generation"]["chat_template_kwargs"],
        "model_artifact": {"repository": workload["profile"]["model_ref"],
            "revision": workload["profile"]["model_revision"], "adapter": workload["adapter"]},
        "comparison_basis": build_basis((("configured-cases", cases),), validator_config, controls),
        "native_reference_metrics": reference_metrics,
        "native_receipt": receipt, "execution_witness": witness,
        "notes": "Configured assertions only; no judge or tool execution. Integrity/admission checks do not independently attest model execution."}
    payload = build_run_payload(records, metadata)
    output = Path(config["output_root"])
    mkdir(output)
    with NativeTree(output, private=True) as tree:
        tree.write_exclusive("configuration.json", canonical(config))
        tree.write_exclusive("evaluation_results.json", canonical(payload), maximum=8 << 20)
    return {"schema": "synaptic-native-evaluation-score-result/v1", "case_count": len(records),
        "report_sha256": sha(canonical(payload)), "output_root": str(output), "model_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--trusted-root-key-sha256")
    args = parser.parse_args()
    try:
        config = document(args.configuration)
        result = prepare(config) if args.action == "prepare" else score(config,
            trusted_root_key_sha256=args.trusted_root_key_sha256)
        encoded = canonical(result)
        if len(encoded) > 8192:
            raise ValueError("native evaluation case summary exceeded")
    except Exception:
        print('{"status":"REFUSED_OR_INCOMPLETE","automatic_retry":false}')
        return 2
    sys.stdout.buffer.write(encoded + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
