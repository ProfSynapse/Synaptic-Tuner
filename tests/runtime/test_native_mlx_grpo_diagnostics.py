"""Closed worker exits with CPU doubles; never model/hardware qualification."""
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.runtime.test_native_mlx_grpo import fixture_grpo_publication
from tests.runtime import test_native_mlx_publication as fixtures
from tuner.runtime import native_mlx_child as child, native_mlx_grpo as grpo
from tuner.runtime.native_mlx_material import canonical, strict_json

publication = fixtures.publication


def fail(*args, **kwargs):
    raise RuntimeError("PRIVATE exception /path prompt argv environment traceback")


@pytest.fixture
def pipeline(publication, monkeypatch):
    """Run real admission, bounded loop, archive checks and publication on doubles.

    The NumPy backend has no autograd: fixed synthetic gradients only test phase
    transport and file/control flow. Existing independent math tests own loss
    correctness, and only separately admitted hardware can qualify training.
    """
    np = pytest.importorskip("numpy")
    state, projection, _, _ = fixture_grpo_publication(publication)
    for path in publication["output"].iterdir():
        path.unlink()  # this fixture owns its synthetic success files
    from tuner.runtime import native_mlx_material
    monkeypatch.setattr(native_mlx_material, "verify_native_material", lambda *a, **k: None)
    monkeypatch.setattr(grpo, "verify_native_material", lambda *a, **k: None)
    monkeypatch.setattr(child, "_load_published_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(child, "install_capture_imports", lambda *a: None)
    monkeypatch.setattr(child, "sys", SimpleNamespace(
        flags=SimpleNamespace(isolated=True, no_site=True), dont_write_bytecode=True,
        argv=["worker", "--request", str(state.root / "request.json")]))
    def apply_chat_template(messages, *, tokenize=True, return_dict=True,
                            add_generation_prompt=False, **kwargs):
        assert tokenize is True and add_generation_prompt is True
        assert kwargs == {"enable_thinking": False}
        # Actual pinned Transformers 5.20.2 defaults to a BatchEncoding mapping.
        return {"input_ids": [1, 2], "attention_mask": [1, 1]} if return_dict else [1, 2]
    tokenizer = SimpleNamespace(eos_token_id=0, apply_chat_template=apply_chat_template,
        decode=lambda tokens: "blue" if tokens[0] == 2 else "red")
    auto = SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    names = tuple(f"model.layers.{layer}.self_attn.{target}.lora_{suffix}"
                  for layer in range(2) for target in ("q_proj", "v_proj") for suffix in ("a", "b"))
    shapes = tuple((name, (1, 4)) for name in names)
    models, saved, samples = [], {}, []
    class LoRA:
        scale = 8.
    def load_model(*a, **k):
        model = SimpleNamespace(weights={name: np.zeros((1, 4), dtype=np.float32) for name in names},
            eval=lambda: None, named_modules=lambda: [("lora", LoRA())])
        # Real causal log-prob gathering runs against constant synthetic logits.
        model_type = type("Model", (), {"__call__": lambda self, tokens, **kw:
            np.zeros((1, tokens.shape[1], 3), dtype=np.float32)})
        wrapped = model_type()
        wrapped.__dict__.update(model.__dict__)
        wrapped.parameters = wrapped.trainable_parameters = lambda: wrapped.weights
        wrapped.load_weights = lambda weights, **kw: setattr(wrapped, "weights", dict(weights))
        models.append(wrapped)
        return wrapped, {}
    def save(path, weights):
        saved[path] = {name: value.copy() for name, value in weights.items()}
        header, payload = {}, bytearray()
        for name, value in sorted(weights.items()):
            start = len(payload)
            payload.extend(value.astype("<f4").tobytes())
            header[name] = {"dtype": "F32", "shape": list(value.shape), "data_offsets": [start, len(payload)]}
        raw = canonical(header)
        Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    def categorical(logits):
        # Eight tokens per completion; alternate completions for reward variation.
        token = 1 + ((len(samples) // 8) % 2)
        samples.append(token)
        return np.array([token])
    class AdamW:
        def __init__(self, **kwargs):
            self.state, self.step = {}, np.array(0)
        def update(self, model, gradients):
            model.weights = {name: value + gradients[name] for name, value in model.weights.items()}
            self.step += 1
    mx = ModuleType("mlx.core")
    for name in ("array", "float32", "int32", "mean", "square", "sqrt", "exp", "where", "sum",
                 "take_along_axis", "concatenate", "zeros", "stack", "all"):
        setattr(mx, name, getattr(np, name))
    mx.logsumexp = lambda value, axis: np.logaddexp.reduce(value, axis=axis)
    mx.stop_gradient = lambda value: value.copy()
    mx.eval = lambda *a: None
    mx.random = SimpleNamespace(seed=lambda seed: None, categorical=categorical)
    mx.save_safetensors, mx.load = save, lambda path: saved[path]
    def value_and_grad(model, loss_fn):
        return lambda current: (loss_fn(current), {name: np.full((1, 4), .01, dtype=np.float32) for name in names})
    modules = {name: ModuleType(name) for name in (
        "mlx", "mlx.nn", "mlx.optimizers", "mlx.utils", "mlx_lm", "mlx_lm.models",
        "mlx_lm.models.cache", "mlx_lm.tuner", "mlx_lm.tuner.lora", "mlx_lm.tuner.utils",
        "mlx_lm.utils", "transformers")}
    modules["mlx.core"] = mx
    modules["mlx.nn"].value_and_grad = value_and_grad
    modules["mlx.optimizers"].AdamW = AdamW
    modules["mlx.utils"].tree_flatten = lambda tree: list(tree.items())
    modules["mlx.utils"].tree_map = lambda fn, tree: {name: fn(value) for name, value in tree.items()}
    modules["mlx_lm.models.cache"].make_prompt_cache = lambda model: []
    modules["mlx_lm.tuner.lora"].LoRALinear = LoRA
    modules["mlx_lm.tuner.utils"].linear_to_lora_layers = lambda *a, **k: None
    modules["mlx_lm.utils"].load_model = load_model
    modules["transformers"].AutoTokenizer = auto
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(grpo, "attach_native_lora", lambda model, *a, **k: (model.weights, shapes))
    return SimpleNamespace(state=state, projection=projection, publication=publication,
        tokenizer=tokenizer, auto=auto, mx=mx, modules=modules, optimizer=AdamW)


# Literal expected codes are the external closed contract, independent of lookup.
@pytest.mark.parametrize("fault,expected", [
    ("request", 130), ("runtime", 131), ("portable", 132), ("input", 133),
    ("tokenizer", 134), ("tokenization", 135), ("imports", 136), ("model", 137),
    ("adapter", 138), ("initial", 139), ("generation", 140), ("reward", 141),
    ("reference", 142), ("objective", 143), ("update", 144), ("save", 145),
    ("reload", 146), ("publication", 147), ("custody", 148),
])
def test_actual_operation_failure_returns_only_closed_exit(pipeline, monkeypatch, capsys, fault, expected):
    p = pipeline
    from tuner.runtime import native_mlx_material
    if fault == "request":
        request = strict_json(p.state.request)
        request["workload"]["controls"]["beta"] = .9
        (p.state.root / "request.json").write_bytes(canonical(request))
    elif fault == "runtime":
        monkeypatch.setattr(native_mlx_material, "verify_native_material", fail)
    elif fault == "portable":
        monkeypatch.setattr(child, "_load_published_tokenizer", fail)
    elif fault == "input":
        (p.state.root / "dataset.jsonl").write_bytes(b"changed")
    elif fault == "tokenizer":
        monkeypatch.setattr(p.auto, "from_pretrained", fail)
    elif fault == "tokenization":
        monkeypatch.setattr(p.tokenizer, "apply_chat_template", fail)
    elif fault == "imports":
        monkeypatch.delattr(p.modules["mlx_lm.tuner.lora"], "LoRALinear")
    elif fault == "model":
        monkeypatch.setattr(p.modules["mlx_lm.utils"], "load_model", fail)
    elif fault == "adapter":
        monkeypatch.setattr(grpo, "attach_native_lora", fail)
    elif fault == "initial":
        monkeypatch.setattr(p.mx, "save_safetensors", fail)
    elif fault == "generation":
        monkeypatch.setattr(p.mx.random, "categorical", fail)
    elif fault == "reward":
        monkeypatch.setattr(p.tokenizer, "decode", fail)
    elif fault == "reference":
        monkeypatch.setattr(grpo, "completion_logprobs", fail)
    elif fault == "objective":
        monkeypatch.setattr(grpo, "grpo_objective", fail)
    elif fault == "update":
        monkeypatch.setattr(p.optimizer, "update", fail)
    elif fault == "save":
        save = p.mx.save_safetensors
        monkeypatch.setattr(p.mx, "save_safetensors", lambda path, weights:
            fail() if Path(path).name == "adapters.safetensors" else save(path, weights))
    elif fault == "reload":
        monkeypatch.setattr(p.mx, "load", fail)
    elif fault == "publication":
        monkeypatch.setattr(grpo, "_publish_grpo_documents", fail)
    elif fault == "custody":
        publish = grpo._publish_grpo_documents
        def publish_then_replace(destination, **kwargs):
            publish(destination, **kwargs)
            # Tamper an authenticated source only after successful publication.
            (Path(p.publication["model"].root) / "config.json").write_bytes(b"changed")
        monkeypatch.setattr(grpo, "_publish_grpo_documents", publish_then_replace)
    assert child.main() == expected
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert not (p.publication["output"] / "native_result.json").exists() or fault == "custody"
    if expected <= 139:
        assert not list(p.publication["output"].iterdir())


def test_success_inventory_and_public_verifier_unchanged(pipeline, capsys):
    from tuner.runtime.native_mlx_publication import verify_native_publication
    assert child.main() == 0
    output = pipeline.publication["output"]
    assert {p.name for p in output.iterdir()} == {
        "initial-adapters.safetensors", "adapters.safetensors", "adapter_config.json",
        "input_binding.json", "native_result.json", "rollout_evidence.json", "workload.json"}
    assert len(verify_native_publication(**pipeline.publication)) == 5
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("method,phase,expected", [
    (method, phase, expected) for method in ("sft", "grpo")
    for phase, expected in (("bootstrap", 120), ("capture_imports", 121))
] + [("sft", "run_request", 122)])
def test_legacy_bootstrap_and_sft_exits_preserved(pipeline, monkeypatch, capsys, method, phase, expected):
    request = strict_json(pipeline.state.request)
    request["workload"]["method"] = method
    (pipeline.state.root / "request.json").write_bytes(canonical(request))
    if phase == "bootstrap":
        monkeypatch.setattr(child, "_read", fail)
    elif phase == "capture_imports":
        monkeypatch.setattr(child, "install_capture_imports", fail)
    else:
        def legacy(request, *, request_root):
            # No new callback keyword may be passed to the legacy SFT path.
            fail()
        monkeypatch.setattr(child, "run_request", legacy)
    assert child.main() == expected
    assert capsys.readouterr() == ("", "")


def test_request_cannot_supply_callback(pipeline, monkeypatch, capsys):
    request = strict_json(pipeline.state.request)
    request["_stage"] = "arbitrary.module.callback"
    (pipeline.state.root / "request.json").write_bytes(canonical(request))
    assert child.main() == 130
    assert not list(pipeline.publication["output"].iterdir())
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("ids", [{"input_ids": [1, 2]}, "12", (1, 2), [], [True], [-1], [1.0], [[1, 2]], [1] * 65])
def test_malformed_token_ids_fail_in_phase135_before_any_ml_import(pipeline, monkeypatch, capsys, ids):
    import builtins
    imported = []
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "mlx_lm"}:
            imported.append(name)
            raise AssertionError("ML import reached before ID admission")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(pipeline.tokenizer, "apply_chat_template", lambda *a, **k: ids)
    monkeypatch.setattr(builtins, "__import__", guard)
    assert child.main() == 135
    assert not imported and not list(pipeline.publication["output"].iterdir())
    assert capsys.readouterr() == ("", "")
