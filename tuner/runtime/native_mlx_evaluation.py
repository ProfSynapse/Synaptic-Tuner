"""Generation-only MLX execution inside the existing captured native child.

No optimizer, training dispatch, network, paid judge or execution authority.
The consuming host supplies complete pinned inputs and owns the bounded process.
"""
from contextlib import ExitStack
import hashlib
import math
import os
from pathlib import Path
import re
import time

from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_artifacts import finite_adapter_tensors
from tuner.runtime.native_mlx_composition import NativeLoRA, attach_native_lora
from tuner.runtime.native_mlx_material import canonical, strict_json, verify_native_material


def _integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("native evaluation integer bound differs")
    return value


def _digest(value, length=64):
    if type(value) is not str or not re.fullmatch(r"[a-f0-9]{%d}" % length, value):
        raise ValueError("native evaluation digest differs")


def parse_workload(raw):
    value = strict_json(raw, maximum=2 << 20)
    if (type(value) is not dict or set(value) != {
            "schema_version", "method", "profile", "generation", "adapter", "cases_sha256", "case_count"}
            or value["schema_version"] != "synaptic-native-mlx-evaluation-workload/v1"
            or value["method"] != "evaluation" or canonical(value) != raw):
        raise ValueError("native evaluation workload differs")
    p, g = value["profile"], value["generation"]
    if type(p) is not dict or set(p) != {
            "profile_ref", "runtime_digest", "source_digest", "model_ref", "model_revision", "tokenizer_revision",
            "quantization_bits", "maximum_cases", "maximum_context_tokens", "maximum_response_bytes",
            "maximum_evidence_bytes"}:
        raise ValueError("native evaluation profile differs")
    if (type(p["profile_ref"]) is not str or not 1 <= len(p["profile_ref"]) <= 256
            or any(ord(c) < 32 for c in p["profile_ref"])):
        raise ValueError("native evaluation profile reference differs")
    for name in ("runtime_digest", "source_digest"):
        _digest(p[name])
    for name in ("model_revision", "tokenizer_revision"):
        _digest(p[name], 40)
    if type(p["model_ref"]) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", p["model_ref"]):
        raise ValueError("native evaluation model reference differs")
    _integer(p["quantization_bits"], 2, 16)
    _integer(p["maximum_cases"], 1, 128)
    _integer(p["maximum_context_tokens"], 2, 16384)
    _integer(p["maximum_response_bytes"], 1, 1 << 20)
    _integer(p["maximum_evidence_bytes"], 1024, 32 << 20)
    _integer(value["case_count"], 1, p["maximum_cases"])
    _digest(value["cases_sha256"])
    if type(g) is not dict or set(g) != {
            "mode", "max_prompt_tokens", "max_completion_tokens", "seed", "stop_token_ids",
            "use_fast", "chat_template_kwargs"} or g["mode"] != "greedy" or type(g["use_fast"]) is not bool:
        raise ValueError("native evaluation generation controls differ")
    _integer(g["max_prompt_tokens"], 1, p["maximum_context_tokens"])
    _integer(g["max_completion_tokens"], 1, 2048)
    _integer(g["seed"], 0, 2**32 - 1)
    if g["max_prompt_tokens"] + g["max_completion_tokens"] > p["maximum_context_tokens"]:
        raise ValueError("native evaluation context budget exceeded")
    stops = g["stop_token_ids"]
    if type(stops) is not list or not 1 <= len(stops) <= 32:
        raise ValueError("native evaluation stop tokens differ")
    for token in stops:
        _integer(token, 0, 2**31 - 1)
    if len(set(stops)) != len(stops):
        raise ValueError("native evaluation stop tokens repeat")
    from synaptic_tuner.api.v1.training_input import validate_chat_template_kwargs
    if g["chat_template_kwargs"] is not None:
        validate_chat_template_kwargs(g["chat_template_kwargs"])
    adapter = value["adapter"]
    if adapter is not None:
        if type(adapter) is not dict or set(adapter) != {"artifact_sha256", "adapter_sha256", "lora"}:
            raise ValueError("native evaluation adapter declaration differs")
        _digest(adapter["artifact_sha256"])
        _digest(adapter["adapter_sha256"])
        lora = _lora(adapter)
        if lora.dropout != 0:
            raise ValueError("native evaluation dropout must be zero")
    return value


def _lora(adapter):
    value = dict(adapter["lora"])
    if type(value.get("targets")) is not list:
        raise ValueError("native evaluation LoRA targets differ")
    value["targets"] = tuple(value["targets"])
    return NativeLoRA(**value)


def read_cases(raw, workload):
    if (type(raw) is not bytes or len(raw) > 2 << 20
            or hashlib.sha256(raw).hexdigest() != workload["cases_sha256"]):
        raise ValueError("native evaluation case bytes differ")
    value = strict_json(raw, maximum=2 << 20)
    if (type(value) is not dict or set(value) != {"schema", "cases"}
            or value["schema"] != "synaptic-native-evaluation-cases/v1" or canonical(value) != raw
            or type(value["cases"]) is not list or len(value["cases"]) != workload["case_count"]):
        raise ValueError("native evaluation cases differ")
    seen = set()
    for case in value["cases"]:
        if (type(case) is not dict or set(case) != {"id", "messages", "reference"}
                or type(case["id"]) is not str or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", case["id"])
                or case["id"] in seen):
            raise ValueError("native evaluation case identity differs")
        seen.add(case["id"])
        messages = case["messages"]
        if type(messages) is not list or not 1 <= len(messages) <= 64:
            raise ValueError("native evaluation messages differ")
        for message in messages:
            if (type(message) is not dict or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or type(message["content"]) is not str or not message["content"]
                    or len(message["content"].encode("utf-8")) > 65536):
                raise ValueError("native evaluation message differs")
        if messages[-1]["role"] != "user":
            raise ValueError("native evaluation prompt must end at user boundary")
        reference = case["reference"]
        if reference is not None and (type(reference) is not str or not reference
                or len(reference.encode("utf-8")) > 65536):
            raise ValueError("native evaluation reference differs")
    return value["cases"]


def prepare_cases(cases, workload, tokenizer):
    """Separate references from exact generation IDs; reject any lossy projection."""
    from shared.sft_preprocessing import materialize_sft_example
    g, prepared = workload["generation"], []
    eos = tokenizer.eos_token_id
    _integer(eos, 0, 2**31 - 1)
    if eos not in g["stop_token_ids"]:
        raise ValueError("native evaluation stops omit tokenizer EOS")
    for case in cases:
        ids = tokenizer.apply_chat_template(case["messages"], tokenize=True, return_dict=False,
            add_generation_prompt=True, **(g["chat_template_kwargs"] or {}))
        if type(ids) is not list or not 1 <= len(ids) <= g["max_prompt_tokens"]:
            raise ValueError("native evaluation prompt exceeds bound")
        for token in ids:
            _integer(token, 0, 2**31 - 1)
        example = None
        if case["reference"] is not None:
            example = materialize_sft_example(tokenizer=tokenizer,
                record={"messages": case["messages"] + [{"role": "assistant", "content": case["reference"]}]},
                max_seq_length=workload["profile"]["maximum_context_tokens"], assistant_only_loss=True,
                prompt_render="prompt_completion", chat_template_kwargs=g["chat_template_kwargs"])
            if (example.truncation_applied or example.drop_reason or example.mask_prefix_mismatch
                    or example.mask_fallback_reason or example.input_ids[:len(ids)] != ids
                    or example.labels[:len(ids)] != [-100] * len(ids)
                    or not any(label != -100 for label in example.labels[1:])
                    or example.attention_mask != [1] * len(example.input_ids)):
                raise ValueError("native evaluation reference projection differs")
        prepared.append((case, tuple(ids), example))
    return prepared


def greedy_completion(model, prompt, stops, maximum, *, mx, make_cache):
    cache, current, generated = make_cache(model), mx.array(prompt, dtype=mx.int32), []
    for _ in range(maximum):
        logits = model(current[None], cache=cache)[:, -1, :].astype(mx.float32)
        if not bool(mx.all(mx.isfinite(logits)).item()):
            raise ValueError("native evaluation logits are nonfinite")
        token = mx.argmax(logits, axis=-1)
        mx.eval(token)
        value = int(token.item())
        generated.append(value)
        if value in stops:
            return tuple(generated), "stop"
        current = token
    return tuple(generated), "length"


def reference_loss(model, example, *, mx, nn):
    tokens, labels = mx.array([example.input_ids], dtype=mx.int32), mx.array([example.labels], dtype=mx.int32)
    targets = labels[:, 1:]
    mask = targets != -100
    logits = model(tokens[:, :-1])
    ce = nn.losses.cross_entropy(logits, mx.where(mask, targets, 0))
    count = int(mask.sum().item())
    if count < 1:
        raise ValueError("native evaluation reference has no supervised tokens")
    loss = float((mx.where(mask, ce, 0).astype(mx.float32).sum() / count).item())
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("native evaluation reference loss differs")
    return {"loss": loss, "supervised_tokens": count}


def validate_record(record, case, workload):
    """Validate complete retained evidence without executing or loading a model."""
    if (type(record) is not dict or set(record) != {"id", "messages_sha256", "prompt_tokens",
            "completion_tokens", "text", "finish_reason", "latency_seconds", "reference"}
            or record["id"] != case["id"]
            or record["messages_sha256"] != hashlib.sha256(canonical(case["messages"])).hexdigest()):
        raise ValueError("native evaluation record binding differs")
    g = workload["generation"]
    for name, maximum in (("prompt_tokens", g["max_prompt_tokens"]), ("completion_tokens", g["max_completion_tokens"])):
        values = record[name]
        if type(values) is not list or not 1 <= len(values) <= maximum:
            raise ValueError("native evaluation retained tokens exceed bounds")
        for token in values:
            _integer(token, 0, 2**31 - 1)
    tokens, stops = record["completion_tokens"], set(g["stop_token_ids"])
    if record["finish_reason"] == "stop":
        valid_finish = tokens[-1] in stops and not any(token in stops for token in tokens[:-1])
    elif record["finish_reason"] == "length":
        valid_finish = len(tokens) == g["max_completion_tokens"] and not any(token in stops for token in tokens)
    else:
        valid_finish = False
    if (not valid_finish or type(record["text"]) is not str
            or len(record["text"].encode("utf-8")) > workload["profile"]["maximum_response_bytes"]
            or type(record["latency_seconds"]) not in (int, float)
            or not math.isfinite(record["latency_seconds"]) or record["latency_seconds"] < 0):
        raise ValueError("native evaluation retained response differs")
    reference = record["reference"]
    if case["reference"] is None:
        if reference is not None:
            raise ValueError("native evaluation unexpected reference evidence")
    elif (type(reference) is not dict or set(reference) != {"loss", "supervised_tokens"}
            or type(reference["loss"]) not in (int, float) or not math.isfinite(reference["loss"])
            or reference["loss"] < 0):
        raise ValueError("native evaluation reference evidence differs")
    else:
        _integer(reference["supervised_tokens"], 1, workload["profile"]["maximum_context_tokens"] - 1)


def validate_result(result, request):
    """Check complete manifest structure shared by host and portable readback."""
    workload = request["workload"]
    if (type(result) is not dict or set(result) != {"schema", "status", "workload_sha256",
            "model_snapshot_digest", "tokenizer_snapshot_digest", "adapter_snapshot_digest",
            "case_files", "case_count", "case_bytes"}
            or result["schema"] != "synaptic-native-evaluation-result/v1" or result["status"] != "COMPLETE"
            or result["workload_sha256"] != hashlib.sha256(canonical(workload)).hexdigest()
            or type(result["case_count"]) is not int or result["case_count"] != workload["case_count"]
            or type(result["case_files"]) is not list or len(result["case_files"]) != workload["case_count"]):
        raise ValueError("native evaluation result differs")
    _integer(result["case_bytes"], 1, workload["profile"]["maximum_evidence_bytes"])
    for role in ("model", "tokenizer", "adapter"):
        declared = request[role + "_snapshot"]
        if result[role + "_snapshot_digest"] != (hashlib.sha256(canonical(declared)).hexdigest() if declared is not None else None):
            raise ValueError("native evaluation result snapshot differs")
    total = 0
    for index, item in enumerate(result["case_files"]):
        if type(item) is not dict or set(item) != {"name", "size", "sha256"} or item["name"] != f"case-{index:04d}.json":
            raise ValueError("native evaluation case inventory differs")
        _integer(item["size"], 1, workload["profile"]["maximum_evidence_bytes"])
        _digest(item["sha256"])
        total += item["size"]
    if total != result["case_bytes"]:
        raise ValueError("native evaluation result aggregate differs")


def execute_request(request, *, request_root, _stage=None):
    """Called only after captured imports; host admission is independently required."""
    stage = _stage or (lambda _: None)
    stage("admission")
    fields = {"schema", "workload", "runtime_capture", "dependency_lock", "cases", "model_snapshot",
              "tokenizer_snapshot", "adapter_snapshot", "output"}
    if type(request) is not dict or set(request) != fields or request["schema"] != "synaptic-native-evaluation-child-request/v1":
        raise ValueError("native evaluation request differs")
    workload = parse_workload(canonical(request["workload"]))
    for name in ("runtime_capture", "dependency_lock", "cases", "output"):
        if type(request[name]) is not str or Path(request[name]).parent != request_root:
            raise ValueError("native evaluation transport differs")
    from tuner.runtime.native_mlx_child import _snapshot
    model, tokenizer = _snapshot(request["model_snapshot"]), _snapshot(request["tokenizer_snapshot"])
    p, g = workload["profile"], workload["generation"]
    if (model.repository, model.revision, tokenizer.repository, tokenizer.revision) != (
            p["model_ref"], p["model_revision"], p["model_ref"], p["tokenizer_revision"]):
        raise ValueError("native evaluation snapshot pins differ")
    adapter = _snapshot(request["adapter_snapshot"]) if request["adapter_snapshot"] is not None else None
    if (adapter is None) != (workload["adapter"] is None):
        raise ValueError("native evaluation adapter selection differs")
    with ExitStack() as stack:
        transport = stack.enter_context(NativeTree(request_root, private=True))
        runtime = transport.read(Path(request["runtime_capture"]).name, maximum=16 << 20)
        lock = transport.read(Path(request["dependency_lock"]).name, maximum=1 << 20)
        raw_cases = transport.read(Path(request["cases"]).name, maximum=2 << 20)
        verify_native_material(runtime, expected_digest=p["runtime_digest"], dependency_lock=lock)
        cases = read_cases(raw_cases, workload)
        model_tree = stack.enter_context(NativeTree(Path(model.root), private=True))
        token_tree = stack.enter_context(NativeTree(Path(tokenizer.root), private=True))
        output = stack.enter_context(NativeTree(Path(request["output"]), private=True))
        with os.scandir(output.fd) as entries:
            if next(entries, None) is not None:
                raise ValueError("native evaluation output must be unused")
        model.verify(model_tree)
        tokenizer.verify(token_tree)
        quantization = strict_json(model_tree.read("config.json", maximum=1 << 20))
        bits = quantization.get("quantization", quantization.get("quantization_config", {})).get("bits", 16)
        if bits != p["quantization_bits"]:
            raise ValueError("native evaluation quantization differs")
        adapter_tree, saved_bytes = None, None
        if adapter is not None:
            adapter_tree = stack.enter_context(NativeTree(Path(adapter.root), private=True))
            adapter.verify(adapter_tree)
            if (adapter.repository, adapter.revision) != (model.repository, model.revision):
                raise ValueError("native evaluation adapter base differs")
            saved_bytes = adapter_tree.read("adapters.safetensors", maximum=64 << 20)
            if hashlib.sha256(saved_bytes).hexdigest() != workload["adapter"]["adapter_sha256"]:
                raise ValueError("native evaluation adapter bytes differ")
            finite_adapter_tensors(saved_bytes)
        stage("tokenization")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer.root, local_files_only=True,
            trust_remote_code=False, use_fast=g["use_fast"])
        prepared = prepare_cases(cases, workload, tok)
        stage("model")
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.utils import load_model
        active, _ = load_model(Path(model.root), lazy=False, strict=True, trust_remote_code=False)
        mx.random.seed(g["seed"])
        if adapter is not None:
            stage("adapter")
            from mlx_lm.tuner.utils import linear_to_lora_layers
            _, shapes = attach_native_lora(active, _lora(workload["adapter"]),
                convert=linear_to_lora_layers, flatten=tree_flatten)
            if tuple((t.name, t.shape) for t in finite_adapter_tensors(saved_bytes)) != shapes:
                raise ValueError("native evaluation adapter tensor layout differs")
            saved = mx.load(str(Path(adapter.root) / "adapters.safetensors"))
            active.load_weights(list(saved.items()), strict=False)
            mx.eval(active.parameters())
            actual = dict(tree_flatten(active.trainable_parameters()))
            if set(actual) != set(saved) or any(actual[n].shape != saved[n].shape
                    or actual[n].dtype != saved[n].dtype or not bool(mx.all(actual[n] == saved[n]).item()) for n in saved):
                raise ValueError("native evaluation loaded adapter differs")
        active.eval()
        evidence, total = [], 0
        for index, (case, prompt, example) in enumerate(prepared):
            stage("generation")
            started = time.monotonic()
            tokens, finish = greedy_completion(active, prompt, set(g["stop_token_ids"]),
                g["max_completion_tokens"], mx=mx, make_cache=make_prompt_cache)
            text = tok.decode(list(tokens), skip_special_tokens=True)
            if len(text.encode("utf-8")) > p["maximum_response_bytes"]:
                raise ValueError("native evaluation response bytes exceeded")
            generation_seconds = time.monotonic() - started
            stage("reference")
            loss = reference_loss(active, example, mx=mx, nn=nn) if example is not None else None
            record = {"id": case["id"], "messages_sha256": hashlib.sha256(canonical(case["messages"])).hexdigest(),
                "prompt_tokens": list(prompt), "completion_tokens": list(tokens), "text": text,
                "finish_reason": finish, "latency_seconds": generation_seconds, "reference": loss}
            validate_record(record, case, workload)
            raw = canonical(record)
            total += len(raw)
            if total > p["maximum_evidence_bytes"]:
                raise ValueError("native evaluation evidence bytes exceeded")
            stage("publication")
            name = f"case-{index:04d}.json"
            output.write_exclusive(name, raw, maximum=p["maximum_evidence_bytes"])
            evidence.append({"name": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)})
        stage("custody")
        model.verify(model_tree)
        tokenizer.verify(token_tree)
        if adapter is not None:
            adapter.verify(adapter_tree)
        for name, expected, maximum in (("cases", raw_cases, 2 << 20),
                ("runtime_capture", runtime, 16 << 20), ("dependency_lock", lock, 1 << 20)):
            if transport.read(Path(request[name]).name, maximum=maximum) != expected:
                raise ValueError("native evaluation retained inputs changed")
        verify_native_material(runtime, expected_digest=p["runtime_digest"], dependency_lock=lock)
        output.write_exclusive("result.json", canonical({"schema": "synaptic-native-evaluation-result/v1",
            "status": "COMPLETE", "workload_sha256": hashlib.sha256(canonical(workload)).hexdigest(),
            "model_snapshot_digest": model.digest, "tokenizer_snapshot_digest": tokenizer.digest,
            "adapter_snapshot_digest": adapter.digest if adapter is not None else None,
            "case_files": evidence, "case_count": len(evidence), "case_bytes": total}))
        transport.check()
        output.check()
