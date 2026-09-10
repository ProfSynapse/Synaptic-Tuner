"""Tiny mounted artifacts admitted through the real chat preparation chain."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

from tests.execution.providers import test_modal_coordinator_producer as producer_cases
from tests.execution.providers.test_modal_inference_launch_integration import (
    _launch_case,
)
from tests.inference.test_retrieved_model import _safe, _tar
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
)
from tuner.execution.providers.modal.inference_launch import prepare_modal_chat_launch
from tuner.inference.serving_target import capture_pinned_model_snapshot


def mounted_launch_case(tmp_path, monkeypatch, *, model_kind="full"):
    """Replace fixture producer bytes *before* native evidence is produced.

    The source binder, workload binder, preparation, Foundation STAGE and launch
    signer all run normally. No authenticated snapshot is rewritten afterward.
    """
    assert model_kind in {"full", "lora"}
    values = {}
    original_inventory = producer_cases._inventory

    def inventory(invocation, *, mutate=None):
        raw, files = original_inventory(invocation, mutate=mutate)
        document = parse_canonical_object(raw, name="test inventory")
        workload = parse_canonical_object(invocation.workload, name="test workload")
        model = workload["identities"]["model"]
        if model_kind == "lora":
            model_files = (
                (
                    "adapter_config.json",
                    json.dumps(
                        {"base_model_name_or_path": model["ref"], "peft_type": "LORA"},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode(),
                ),
                ("adapter_model.safetensors", _safe()),
            )
        else:
            model_files = (
                ("config.json", b'{"model_type":"fixture"}'),
                ("model.safetensors", _safe()),
            )
        values.update(
            final_model=_tar(model_files),
            tokenizer=_tar(
                (
                    ("tokenizer_config.json", b'{"tokenizer_class":"Fixture"}'),
                    (
                        "tokenizer.json",
                        b'{"model":{"type":"BPE","vocab":{"x":0}},"version":"1"}',
                    ),
                )
            ),
            training_lineage=b"{}",
            training_metrics=b"{}",
            workload_record=invocation.workload,
        )
        for item in document["artifacts"]:
            content = values[item["role"]]
            files[item["path"]] = content
            item["size"] = len(content)
            item["sha256"] = hashlib.sha256(content).hexdigest()
        return canonical_bytes(document), files

    # Keep the producer's test doubles scoped to construction; worker tests use
    # actual filesystem operations and the unmodified production byte reader.
    with monkeypatch.context() as construction:
        construction.setattr(producer_cases, "_inventory", inventory)
        case = _launch_case(construction)
    roots = {
        name: tmp_path / name
        for name in ("artifacts", "control", "cache", "destination")
    }
    for root in roots.values():
        root.mkdir()
    expectation = replace(
        case.expectation,
        artifact_root=str(roots["artifacts"]),
        control_root=str(roots["control"]),
        cache_root=str(roots["cache"]),
    )
    arguments = case.arguments | {"expectation": expectation}
    envelope = prepare_modal_chat_launch(**arguments)
    document = parse_canonical_object(
        envelope.preparation_snapshot, name="test preparation"
    )
    source = document["chat_input"]["source"]
    workload = document["chat_input"]["workload"]
    for member in source["members"]:
        content = values[member["role"]]
        assert member["sha256"] == hashlib.sha256(content).hexdigest()
        assert member["size"] == len(content)
        path = roots["artifacts"] / member["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    class Preparer:
        def __init__(self):
            self.calls = []

        def prepare(self, *, model_ref, revision):
            self.calls.append((model_ref, revision))
            base = roots["destination"] / "base"
            snapshot = base / "snapshot"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_bytes(b'{"model_type":"fixture"}')
            (snapshot / "weights.bin").write_bytes(b"fixture-weights")
            return capture_pinned_model_snapshot(
                model_ref=model_ref, revision=revision, root=base, snapshot="snapshot"
            )

    preparer = Preparer()
    kwargs = dict(
        expectation=expectation,
        verifier=case.authenticator,
        clock=case.clock,
        destination=roots["destination"],
        preparer=preparer,
    )
    return SimpleNamespace(**locals())
