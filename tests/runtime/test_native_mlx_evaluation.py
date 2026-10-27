"""Generation-only controls and independent CPU algebra; never import MLX here."""
from copy import deepcopy
import hashlib
import math
import platform
from types import SimpleNamespace

import numpy as np
import pytest

from tuner.runtime.native_mlx_evaluation import (
    greedy_completion, parse_workload, prepare_cases, read_cases, reference_loss,
)
from tuner.runtime.native_mlx_material import canonical


def selection():
    cases = {"schema": "synaptic-native-evaluation-cases/v1", "cases": [
        {"id": "held-out-1", "messages": [{"role": "user", "content": "question"}], "reference": "answer"}]}
    raw = canonical(cases)
    workload = {"schema_version": "synaptic-native-mlx-evaluation-workload/v1", "method": "evaluation",
        "profile": {"profile_ref": "selected-evaluation", "runtime_digest": "a" * 64, "source_digest": "b" * 64, "model_ref": "selected/model",
            "model_revision": "c" * 40, "tokenizer_revision": "c" * 40, "quantization_bits": 4,
            "maximum_cases": 16, "maximum_context_tokens": 1024,
            "maximum_response_bytes": 65536, "maximum_evidence_bytes": 1 << 20},
        "generation": {"mode": "greedy", "max_prompt_tokens": 512, "max_completion_tokens": 128,
            "seed": 17, "stop_token_ids": [0], "use_fast": True, "chat_template_kwargs": {"enable_thinking": False}},
        "adapter": None, "cases_sha256": hashlib.sha256(raw).hexdigest(), "case_count": 1}
    return workload, raw


class Tokenizer:
    eos_token_id = 0
    all_special_ids = [0]

    def __init__(self):
        self.generation_calls = []

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False,
                            return_dict=True, **kwargs):
        assert kwargs == {"enable_thinking": False}
        text = "".join(m["role"] + ":" + m["content"] + ";" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        if not tokenize:
            return text
        self.generation_calls.append(deepcopy(messages))
        ids = self.encode(text, add_special_tokens=False)
        return {"input_ids": ids} if return_dict else ids

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return list(map(ord, text))


def test_complete_workload_round_trip_and_reference_does_not_enter_generation():
    w, raw = selection()
    assert parse_workload(canonical(w)) == w
    cases = read_cases(raw, w)
    tokenizer = Tokenizer()
    prepared = prepare_cases(cases, w, tokenizer)
    assert tokenizer.generation_calls == [cases[0]["messages"]]
    _, prompt, example = prepared[0]
    assert "answer" not in "".join(map(chr, prompt))
    assert example.labels[:len(prompt)] == [-100] * len(prompt)
    assert example.input_ids[:len(prompt)] == list(prompt)
    assert example.labels[len(prompt):] == example.input_ids[len(prompt):]


@pytest.mark.parametrize("fault", ["method", "unknown", "context", "case_count", "duplicate_stop",
    "bool_limit", "renderer_collision", "mode", "adapter_digest", "dropout"])
def test_unsupported_or_unbounded_controls_refuse(fault):
    w, _ = selection()
    if fault == "method": w["method"] = "sft"
    elif fault == "unknown": w["unknown"] = True
    elif fault == "context": w["generation"]["max_prompt_tokens"] = 1024
    elif fault == "case_count": w["case_count"] = 0
    elif fault == "duplicate_stop": w["generation"]["stop_token_ids"] = [0, 0]
    elif fault == "bool_limit": w["profile"]["maximum_cases"] = True
    elif fault == "renderer_collision": w["generation"]["chat_template_kwargs"]["return_dict"] = True
    elif fault == "mode": w["generation"]["mode"] = "sample"
    else:
        w["adapter"] = {"artifact_sha256": "d" * 64, "adapter_sha256": "e" * 64,
            "lora": {"layers": 2, "rank": 4, "scale": 8, "dropout": 0, "targets": ["q_proj"]}}
        if fault == "dropout": w["adapter"]["lora"]["dropout"] = .1
        else: w["adapter"]["adapter_sha256"] = "invalid"
    with pytest.raises((ValueError, TypeError)):
        parse_workload(canonical(w))


@pytest.mark.parametrize("fault", ["changed_bytes", "count", "duplicate_id", "assistant_boundary",
    "reference_type", "extra_field", "tool_role"])
def test_case_commitments_and_generation_boundary_refuse(fault):
    w, raw = selection()
    value = __import__("json").loads(raw)
    c = value["cases"][0]
    if fault == "changed_bytes": raw += b" "
    elif fault == "count": w["case_count"] = 2
    elif fault == "duplicate_id": value["cases"].append(deepcopy(c));w["case_count"] = 2
    elif fault == "assistant_boundary": c["messages"][-1]["role"] = "assistant"
    elif fault == "reference_type": c["reference"] = 123
    elif fault == "extra_field": c["callback"] = "arbitrary"
    elif fault == "tool_role": c["messages"][-1]["role"] = "tool"
    if fault != "changed_bytes": raw = canonical(value);w["cases_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError): read_cases(raw, w)


def test_prompt_limit_and_missing_eos_refuse_before_model():
    w, raw = selection(); cases = read_cases(raw, w)
    w["generation"]["max_prompt_tokens"] = 1
    with pytest.raises(ValueError): prepare_cases(cases, w, Tokenizer())
    w["generation"]["max_prompt_tokens"] = 512
    w["generation"]["stop_token_ids"] = [123]
    with pytest.raises(ValueError): prepare_cases(cases, w, Tokenizer())


def test_reference_truncation_and_invalid_ids_refuse():
    w, raw = selection();cases = read_cases(raw, w)
    w["profile"]["maximum_context_tokens"] = 25
    with pytest.raises(ValueError): prepare_cases(cases, w, Tokenizer())
    t = Tokenizer(); t.apply_chat_template = lambda *a, **k: {"input_ids": [1]}
    with pytest.raises(ValueError): prepare_cases(cases, w, t)
    t.apply_chat_template = lambda *a, **k: [True]
    with pytest.raises(ValueError): prepare_cases(cases, w, t)


MX = SimpleNamespace(array=np.array, int32=np.int32, float32=np.float32, where=np.where,
    all=np.all, isfinite=np.isfinite, argmax=np.argmax, eval=lambda *a: None)


def test_greedy_ids_cache_and_natural_completion_or_length_are_exact():
    calls, cache = [], object()
    def model(tokens, *, cache):
        calls.append((tokens.copy(), cache))
        logits = np.full((1, tokens.shape[1], 4), -10., dtype=np.float32)
        logits[:, -1, 2 if len(calls) == 1 else 0] = 10.
        return logits
    tokens, reason = greedy_completion(model, [1, 3], {0}, 4, mx=MX, make_cache=lambda _: cache)
    assert tokens == (2, 0) and reason == "stop"
    assert calls[0][0].tolist() == [[1, 3]] and calls[1][0].tolist() == [[2]]
    assert all(c[1] is cache for c in calls)
    calls.clear()
    tokens, reason = greedy_completion(model, [1, 3], {0}, 1, mx=MX, make_cache=lambda _: cache)
    assert tokens == (2,) and reason == "length"


def test_nonfinite_generation_refuses():
    with pytest.raises(ValueError):
        greedy_completion(lambda tokens, **k: np.full((1, 1, 3), np.nan), [1], {0}, 2,
            mx=MX, make_cache=lambda _: None)


def test_reference_loss_matches_independent_shifted_masked_scalar_ce():
    logits = np.array([[[2., 1., 0.], [1., 3., 0.], [0., 1., 4.]]], dtype=np.float32)
    def ce(values, target):
        assert target.tolist() == [[0, 1, 2]]  # masked -100 never gathered
        return np.log(np.exp(values).sum(-1)) - np.take_along_axis(values, target[..., None], -1)[..., 0]
    def model(tokens):
        assert tokens.tolist() == [[1, 0, 1]]
        return logits
    example = SimpleNamespace(input_ids=[1, 0, 1, 2], labels=[-100, -100, 1, 2])
    result = reference_loss(model, example, mx=MX, nn=SimpleNamespace(losses=SimpleNamespace(cross_entropy=ce)))
    expected = sum(math.log(sum(math.exp(float(v)) for v in row)) - float(row[target])
        for row, target in zip(logits[0, 1:], [1, 2])) / 2
    assert result["supervised_tokens"] == 2 and math.isclose(result["loss"], expected, rel_tol=1e-6)


@pytest.fixture
def evaluation_pipeline(monkeypatch):
    """CPU doubles exercise actual worker dispatch/custody, never claim ML evidence."""
    from dataclasses import asdict
    import os
    from pathlib import Path
    import sys
    import tempfile
    from types import ModuleType
    from tuner.runtime import native_mlx_child as child, native_mlx_evaluation as worker
    from tuner.runtime.native_files import NativeTree
    from tuner.runtime.native_mlx_input import NativeModelSnapshot
    mask = os.umask(0o077)
    with tempfile.TemporaryDirectory(dir=Path.home(), prefix="native-eval-worker-") as directory:
        root = Path(directory)
        try:
            output, model = root / "output", root / "model"
            output.mkdir(mode=0o700);model.mkdir(mode=0o700)
            (model / "config.json").write_bytes(canonical({"quantization": {"bits": 4}}))
            with NativeTree(model, private=True) as tree:
                snapshot = asdict(NativeModelSnapshot("selected/model", "c" * 40, str(model), tree.inventory()))
            workload, cases = selection()
            capture = canonical({"schema": "synaptic-native-mlx-capture/v1", "config": {}})
            workload["profile"]["runtime_digest"] = hashlib.sha256(capture).hexdigest()
            request = {"schema": "synaptic-native-evaluation-child-request/v1", "workload": workload,
                "runtime_capture": str(root / "capture.json"), "dependency_lock": str(root / "lock"),
                "cases": str(root / "cases.json"), "model_snapshot": snapshot, "tokenizer_snapshot": snapshot,
                "adapter_snapshot": None, "output": str(output)}
            for name, raw in (("capture.json", capture), ("lock", b"lock"), ("cases.json", cases),
                    ("request.json", canonical(request))):
                (root / name).write_bytes(raw)
            monkeypatch.setattr(worker, "verify_native_material", lambda *a, **kw: None)
            monkeypatch.setattr(child, "install_capture_imports", lambda *a: None)
            monkeypatch.setattr(child, "sys", SimpleNamespace(flags=SimpleNamespace(isolated=True, no_site=True),
                dont_write_bytecode=True, argv=["worker", "--request", str(root / "request.json")]))
            modules = {name: ModuleType(name) for name in ("transformers", "mlx", "mlx.core", "mlx.nn", "mlx.utils",
                "mlx_lm", "mlx_lm.models", "mlx_lm.models.cache", "mlx_lm.utils")}
            tokenizer = Tokenizer()
            tokenizer.decode = lambda ids, **kw: "answer"
            modules["transformers"].AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
            mx = modules["mlx.core"]
            mx.__dict__.update(MX.__dict__)
            mx.random = SimpleNamespace(seed=lambda value: None)
            class Model:
                def __call__(self, tokens, **kwargs):
                    logits = np.zeros((1, tokens.shape[1], 128), dtype=np.float32)
                    logits[:, :, 0] = 2
                    return logits
                def eval(self): pass
            modules["mlx_lm.utils"].load_model = lambda *a, **kw: (Model(), {})
            modules["mlx_lm.models.cache"].make_prompt_cache = lambda model: object()
            modules["mlx.utils"].tree_flatten = lambda tree: list(tree.items())
            def ce(logits, targets):
                return np.log(np.exp(logits).sum(-1)) - np.take_along_axis(logits, targets[..., None], -1)[..., 0]
            modules["mlx.nn"].losses = SimpleNamespace(cross_entropy=ce)
            for name, value in modules.items(): monkeypatch.setitem(sys.modules, name, value)
            yield SimpleNamespace(root=root, output=output, model=model, child=child, worker=worker,
                request=request, modules=modules, tokenizer=tokenizer)
        finally:
            os.umask(mask)


def test_actual_generation_dispatch_publishes_complete_readable_evidence(evaluation_pipeline, capsys):
    import json
    from tuner.runtime.native_mlx_evaluation import validate_record, validate_result
    p = evaluation_pipeline
    assert p.child.main() == 0
    result = json.loads((p.output / "result.json").read_bytes())
    validate_result(result, p.request)
    row = json.loads((p.output / "case-0000.json").read_bytes())
    case = read_cases((p.root / "cases.json").read_bytes(), p.request["workload"])[0]
    validate_record(row, case, p.request["workload"])
    assert row["completion_tokens"] == [0] and row["finish_reason"] == "stop"
    assert set(f.name for f in p.output.iterdir()) == {"case-0000.json", "result.json"}
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("fault,expected", [("admission", 150), ("tokenization", 151), ("model", 152),
    ("generation", 154), ("reference", 155), ("publication", 156), ("custody", 157)])
def test_evaluation_operation_failures_emit_only_closed_exits(evaluation_pipeline, monkeypatch, capsys, fault, expected):
    p = evaluation_pipeline
    def fail(*a, **kw): raise RuntimeError("PRIVATE path prompt exception environment")
    if fault == "admission": monkeypatch.setattr(p.worker, "verify_native_material", fail)
    elif fault == "tokenization": monkeypatch.setattr(p.tokenizer, "apply_chat_template", fail)
    elif fault == "model": monkeypatch.setattr(p.modules["mlx_lm.utils"], "load_model", fail)
    elif fault == "generation": monkeypatch.setattr(p.worker, "greedy_completion", fail)
    elif fault == "reference": monkeypatch.setattr(p.worker, "reference_loss", fail)
    elif fault == "publication":
        from tuner.runtime.native_files import NativeTree
        monkeypatch.setattr(NativeTree, "write_exclusive", fail)
    else:
        reference = p.worker.reference_loss
        def changed(*a, **kw):
            result = reference(*a, **kw)
            (p.model / "config.json").write_bytes(b"changed")
            return result
        monkeypatch.setattr(p.worker, "reference_loss", changed)
    assert p.child.main() == expected
    assert not (p.output / "result.json").exists()
    assert capsys.readouterr() == ("", "")


def test_bad_tokenization_refuses_before_ml_imports(evaluation_pipeline, monkeypatch, capsys):
    import builtins
    p = evaluation_pipeline
    monkeypatch.setattr(p.tokenizer, "apply_chat_template", lambda *a, **kw: {"input_ids": [1]})
    original, imported = builtins.__import__, []
    def guard(name, *a, **kw):
        if name.split(".")[0] in {"mlx", "mlx_lm"}:
            imported.append(name)
            raise AssertionError("ML before tokenizer acceptance")
        return original(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", guard)
    assert p.child.main() == 151 and not imported
    assert not list(p.output.iterdir()) and capsys.readouterr() == ("", "")


@pytest.mark.parametrize("fault", ["callback", "extra_output_directory", "quantization"])
def test_evaluation_request_or_existing_output_refuses(evaluation_pipeline, capsys, fault):
    p = evaluation_pipeline
    if fault == "callback":
        p.request["_stage"] = "user.callback"
        (p.root / "request.json").write_bytes(canonical(p.request))
    elif fault == "extra_output_directory": (p.output / "extra").mkdir(mode=0o700)
    else:
        p.request["workload"]["profile"]["quantization_bits"] = 8
        (p.root / "request.json").write_bytes(canonical(p.request))
    assert p.child.main() == 150
    assert not (p.output / "result.json").exists() and capsys.readouterr() == ("", "")


@pytest.mark.parametrize("fault,expected", [(None, 0), ("attach", 153), ("shape", 153), ("loaded", 153)])
def test_saved_adapter_is_reloaded_and_compared_before_generation(evaluation_pipeline, monkeypatch, capsys, fault, expected):
    from dataclasses import asdict
    from types import ModuleType
    import sys
    from tests.runtime.test_native_mlx_publication import tensor_archive
    from tuner.runtime.native_files import NativeTree
    from tuner.runtime.native_mlx_input import NativeModelSnapshot
    p = evaluation_pipeline
    directory = p.root / "adapter";directory.mkdir(mode=0o700)
    raw = tensor_archive(.25);(directory / "adapters.safetensors").write_bytes(raw)
    with NativeTree(directory, private=True) as tree:
        snapshot = NativeModelSnapshot("selected/model", "c" * 40, str(directory), tree.inventory())
    p.request["adapter_snapshot"] = asdict(snapshot)
    p.request["workload"]["adapter"] = {"artifact_sha256": "e" * 64,
        "adapter_sha256": hashlib.sha256(raw).hexdigest(), "lora": {"layers": 2, "rank": 4,
            "scale": 8, "dropout": 0, "targets": ["self_attn.q_proj", "self_attn.v_proj"]}}
    (p.root / "request.json").write_bytes(canonical(p.request))
    shapes = tuple((t.name, t.shape) for t in p.worker.finite_adapter_tensors(raw))
    saved = {name: np.array([[.25, 0, 0, 0]], dtype=np.float32) for name, _ in shapes}
    original = p.modules["mlx_lm.utils"].load_model
    def load(*args, **kw):
        active, config = original(*args, **kw)
        active.weights = {n: np.zeros_like(v) for n, v in saved.items()}
        active.parameters = active.trainable_parameters = lambda: active.weights
        def weights(items, **kwargs):
            active.weights = {n: v + (1 if fault == "loaded" else 0) for n, v in items}
        active.load_weights = weights
        return active, config
    monkeypatch.setattr(p.modules["mlx_lm.utils"], "load_model", load)
    monkeypatch.setattr(p.modules["mlx.core"], "load", lambda path: saved, raising=False)
    module = ModuleType("mlx_lm.tuner.utils");module.linear_to_lora_layers = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner", ModuleType("mlx_lm.tuner"))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.utils", module)
    def attach(active, *a, **kw):
        if fault == "attach": raise ValueError("PRIVATE")
        return active.weights, (() if fault == "shape" else shapes)
    monkeypatch.setattr(p.worker, "attach_native_lora", attach)
    assert p.child.main() == expected
    assert (p.output / "result.json").exists() is (expected == 0)
    assert capsys.readouterr() == ("", "")


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="real MLX evaluation numerics require Apple silicon")
def test_real_mlx_evaluation_reference_loss_matches_scalar_ce():
    mx, nn = pytest.importorskip("mlx.core"), pytest.importorskip("mlx.nn")
    example = SimpleNamespace(input_ids=[1, 0, 1, 2], labels=[-100, -100, 1, 2])
    values = [[[2., 1., 0.], [1., 3., 0.], [0., 1., 4.]]]
    result = reference_loss(lambda tokens: mx.array(values, dtype=mx.float32), example, mx=mx, nn=nn)
    expected = sum(math.log(sum(math.exp(v) for v in row)) - row[target]
        for row, target in zip(values[0][1:], [1, 2])) / 2
    assert result["supervised_tokens"] == 2 and result["loss"] == pytest.approx(expected, abs=2e-6)
