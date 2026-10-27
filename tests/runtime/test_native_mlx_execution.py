from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import os
from types import SimpleNamespace, ModuleType

import pytest

from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.native_files import NativeFile
from tuner.runtime.native_mlx_artifacts import adapter_tensors, verify_adapter_delta
from tuner.runtime.native_mlx_composition import NativeAdamW, NativeLoRA, attach_native_lora
from tuner.runtime.native_mlx_input import (
    NativeModelSnapshot, NativeTokenization, read_retained_native_source, tokenize_native_source,
)
from tuner.runtime.native_mlx_sft import MLXSFTConfig
from tuner.training.contracts import RetainedTrainingInputStreamLease


class Source:
    def __init__(self, raw, *, lease_identity=None):
        self.raw = raw
        digest = hashlib.sha256(raw).hexdigest()
        self.identity = PreparedTrainingInputIdentity(
            "prepared://sha256/" + "a" * 64, "a" * 64, digest, len(raw), "syntunia-sft-row/v1",
        )
        self.lease_identity = lease_identity
        self.last_stream = None

    def open_lease(self):
        self.last_stream = io.BytesIO(self.raw)
        return RetainedTrainingInputStreamLease(self.lease_identity or self.identity, self.last_stream)


def source():
    return Source(b"".join(json.dumps({
        "schema_version": "syntunia-sft-row/v1", "format": "raw_text",
        "row_id": row_id, "text": text, "split": split,
    }).encode() + b"\n" for row_id, text, split in [
        ("train-1", "ab", "train"), ("val-1", "cd", "validation"),
        ("train-2", "ef", "train"), ("val-2", "g", "validation"),
    ]))


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def tokenize(prepared_source):
    return tokenize_native_source(
        source=prepared_source, identity=prepared_source.identity, tokenizer=Tokenizer(),
        tokenizer_snapshot=NativeModelSnapshot(
            "configured/model", "c" * 40, "/private/configured-model",
            (NativeFile("tokenizer.json", 3, "d" * 64),),
        ),
        tokenization=NativeTokenization(False, "full_conversation", None, True),
        config=MLXSFTConfig(2, 16, 32, 3, 1, 1, 1, 3),
        workload_digest="e" * 64, runtime_digest="f" * 64,
    )


def test_retained_bytes_are_shared_tokenized_without_resplit_or_second_eos():
    retained = source()
    bound = tokenize(retained)
    assert tuple(row.row_id for row in bound.prepared.train) == ("train-1", "train-2")
    assert tuple(row.row_id for row in bound.prepared.validation) == ("val-1", "val-2")
    assert bound.prepared.train[0].input_ids == (97, 98, 0)
    assert bound.prepared.train[0].labels == (97, 98, 0)
    assert retained.last_stream.closed
    assert len(bound.digest) == 64
    assert replace(bound, workload_digest="0" * 64).digest != bound.digest


def test_lease_identity_and_source_bytes_reject_before_tokenization():
    retained = source()
    retained.raw += b" "
    with pytest.raises(ValueError, match="bytes differ"):
        tokenize(retained)
    assert retained.last_stream.closed
    retained = source()
    retained.lease_identity = replace(retained.identity, content_digest="b" * 64)
    with pytest.raises(ValueError, match="lease identity"):
        read_retained_native_source(retained, retained.identity)
    assert retained.last_stream.closed


def test_unknown_split_and_duplicate_row_are_not_repaired():
    retained = source()
    with pytest.raises(ValueError, match="split"):
        tokenize(Source(retained.raw.replace(b'"validation"', b'"test"')))
    with pytest.raises(ValueError, match="more than once"):
        tokenize(Source(retained.raw.replace(b'"val-1"', b'"train-1"')))


def test_native_modules_keep_sdk_imports_lazy():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import tuner.runtime.native_mlx_composition; "
         "import tuner.training.native_mlx_family; import tuner.training.native_mlx_planning; "
         "assert not any(n.split('.')[0] in "
         "{'mlx','mlx_lm','transformers','torch','numpy','modal'} for n in sys.modules)"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def tensor_bytes(values, *, dtype="F32", name="layers.0.q_proj.lora_a", shape=None):
    shape = shape or [1, len(values)]
    if dtype == "BF16":
        body = b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16)
                        for v in values)
    else:
        body = struct.pack("<" + ("f" if dtype == "F32" else "e") * len(values), *values)
    header = json.dumps({name: {
        "dtype": dtype, "shape": shape, "data_offsets": [0, len(body)],
    }}, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header)) + header + body


def with_safetensors_metadata(raw, metadata):
    header_size = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8:8 + header_size])
    header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_size:]


@pytest.mark.parametrize("metadata", [None, {}, {"format": "mlx"}])
def test_native_adapter_accepts_pinned_mlx_null_or_string_metadata(metadata):
    # The pinned MLX save_safetensors emitted present JSON null metadata in
    # the retained r37 initial adapter. Full tensor and delta checks still run.
    initial = with_safetensors_metadata(tensor_bytes([0.0]), metadata)
    final = with_safetensors_metadata(tensor_bytes([0.5]), metadata)
    assert len(adapter_tensors(initial)) == 1
    assert verify_adapter_delta(
        initial, final, expected_shapes=(("layers.0.q_proj.lora_a", (1, 1)),),
    )["changed_parameters"] == 1


@pytest.mark.parametrize("metadata", [[], 1, "mlx", {"format": 1}, {"format": None}])
def test_native_adapter_rejects_other_metadata_shapes(metadata):
    with pytest.raises(ValueError, match="metadata"):
        adapter_tensors(with_safetensors_metadata(tensor_bytes([1.0]), metadata))


@pytest.mark.parametrize("dtype", ["F16", "BF16", "F32"])
def test_actual_saved_tensor_delta_and_finite_values(dtype):
    initial = tensor_bytes([0, 1], dtype=dtype)
    final = tensor_bytes([0.5, 1], dtype=dtype)
    expected = (("layers.0.q_proj.lora_a", (1, 2)),)
    result = verify_adapter_delta(initial, final, expected_shapes=expected)
    assert result["parameter_count"] == 2
    assert result["changed_parameters"] == 1
    assert result["delta_l2"] == .5
    assert result["final_sha256"] == hashlib.sha256(final).hexdigest()
    with pytest.raises(ValueError, match="no finite parameter update"):
        verify_adapter_delta(initial, initial, expected_shapes=expected)
    with pytest.raises(ValueError, match="nonfinite"):
        verify_adapter_delta(initial, tensor_bytes([float("nan"), 1], dtype=dtype), expected_shapes=expected)


@pytest.mark.parametrize("raw", [
    tensor_bytes([1], name="base.weight"),
    tensor_bytes([1]) + b"unclaimed",
    tensor_bytes([1], shape=[2, 2]),
    tensor_bytes([1])[:-1],
    struct.pack("<Q", 1024**3) + b"{}",
])
def test_native_adapter_rejects_unclaimed_or_malformed_bytes(raw):
    with pytest.raises(ValueError):
        adapter_tensors(raw)


def test_native_adapter_requires_exact_realized_inventory():
    before, after = tensor_bytes([0]), tensor_bytes([1])
    with pytest.raises(ValueError, match="inventory"):
        verify_adapter_delta(before, after, expected_shapes=(("other.lora_a", (1, 1)),))


def test_explicit_optimizer_controls_and_no_silent_schedule():
    config = NativeAdamW(.0001, .9, .999, 1e-8, .01, True, "constant")
    assert config.arguments() == dict(learning_rate=.0001, betas=[.9, .999], eps=1e-8,
                                      weight_decay=.01, bias_correction=True)
    with pytest.raises(ValueError):
        replace(config, schedule="cosine")
    with pytest.raises(ValueError):
        replace(config, learning_rate=float("nan"))
    with pytest.raises(ValueError):
        replace(config, bias_correction=1)


class Layer:
    def __init__(self):
        self.target = object()

    def named_modules(self):
        return [("q_proj", self.target)]


class Model:
    def __init__(self):
        self.layers = [Layer(), Layer()]
        self.parameters = {}
        self.frozen = False

    def named_modules(self):
        return [(f"layers.{index}.q_proj", layer.target) for index, layer in enumerate(self.layers)]

    def freeze(self):
        self.frozen = True

    def trainable_parameters(self):
        return self.parameters


def test_lora_targets_and_layer_extent_are_checked_against_realized_model():
    model = Model()
    config = NativeLoRA(1, 4, 8, 0, ("q_proj",))

    def convert(model, layers, parameters, use_dora):
        assert model.frozen and layers == 1 and use_dora is False
        assert parameters == dict(rank=4, scale=8, dropout=0, keys=["q_proj"])
        model.parameters = {
            "layers.1.q_proj.lora_a": SimpleNamespace(shape=(8, 4)),
            "layers.1.q_proj.lora_b": SimpleNamespace(shape=(4, 8)),
        }
    _, shapes = attach_native_lora(model, config, convert=convert, flatten=lambda v: v.items())
    assert shapes == (("layers.1.q_proj.lora_a", (8, 4)), ("layers.1.q_proj.lora_b", (4, 8)))
    with pytest.raises(ValueError, match="missing"):
        attach_native_lora(Model(), replace(config, targets=("missing",)), convert=convert, flatten=lambda v: v.items())
    with pytest.raises(ValueError, match="exceeds"):
        attach_native_lora(Model(), replace(config, layers=3), convert=convert, flatten=lambda v: v.items())


def test_unconfigured_trainable_base_weight_is_rejected():
    def convert(model, *args, **kwargs):
        model.parameters = {"base.weight": SimpleNamespace(shape=(4, 4))}
    with pytest.raises(ValueError, match="trainable parameters differ"):
        attach_native_lora(Model(), NativeLoRA(1, 4, 8, 0, ("q_proj",)),
                           convert=convert, flatten=lambda v: v.items())


@pytest.mark.parametrize("reload_mode", ["complete", "noop", "partial"])
@pytest.mark.parametrize("with_progress", [False, True])
def test_composition_evaluates_reloaded_saved_final_adapter_after_last_update(monkeypatch, reload_mode, with_progress):
    from tuner.runtime import native_mlx_composition as composition
    from tuner.runtime.native_files import NativeTree

    old_umask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-final-test-", dir=Path.home()) as name:
            root = Path(name)
            snapshot_root, output = root / "model", root / "output"
            snapshot_root.mkdir(mode=0o700)
            output.mkdir(mode=0o700)
            (snapshot_root / "config.json").write_text("{}")
            with NativeTree(snapshot_root) as tree:
                snapshot = NativeModelSnapshot("configured/model", "a" * 40, str(snapshot_root), tree.inventory())
            events = []
            progress_events = []
            names = ("layers.0.q_proj.lora_a", "layers.0.q_proj.lora_b")
            shapes = tuple((name, (1, 1)) for name in names)
            def adapter_bytes(value):
                headers = {name: {"dtype": "F32", "shape": [1, 1],
                                  "data_offsets": [4 * index, 4 * (index + 1)]}
                           for index, name in enumerate(names)}
                raw_header = json.dumps(headers, sort_keys=True, separators=(",", ":")).encode()
                return struct.pack("<Q", len(raw_header)) + raw_header + struct.pack("<ff", value, value)
            class Tensor:
                shape, dtype = (1, 1), "float32"

                def __init__(self, value):
                    self.value = value

                def __eq__(self, other):
                    return self.value == other.value

            tensors = {name: Tensor(.5) for name in names}
            models = []
            def load_model(*args, **kwargs):
                model = SimpleNamespace(weights={name: Tensor(0) for name in names}, parameters=lambda: {},
                                        trainable_parameters=lambda: model.weights)
                def load_weights(values, strict):
                    events.append(("reload", values, strict))
                    if reload_mode == "complete":
                        model.weights = dict(values)
                    elif reload_mode == "partial":
                        model.weights[names[0]] = dict(values)[names[0]]
                model.load_weights = load_weights
                models.append(model)
                return model, {}
            mx = ModuleType("mlx.core")
            mx.random = SimpleNamespace(seed=lambda seed: None)
            mx.eval = lambda *args: None
            mx.save_safetensors = lambda path, values: Path(path).write_bytes(adapter_bytes(0))
            mx.load = lambda path: tensors
            mx.all = lambda value: SimpleNamespace(item=lambda: value)
            modules = {
                "mlx": ModuleType("mlx"), "mlx.core": mx, "mlx.nn": ModuleType("mlx.nn"),
                "mlx.optimizers": ModuleType("mlx.optimizers"), "mlx.utils": ModuleType("mlx.utils"),
                "mlx_lm": ModuleType("mlx_lm"), "mlx_lm.utils": ModuleType("mlx_lm.utils"),
                "mlx_lm.tuner": ModuleType("mlx_lm.tuner"),
                "mlx_lm.tuner.utils": ModuleType("mlx_lm.tuner.utils"),
                "mlx_lm.tuner.trainer": ModuleType("mlx_lm.tuner.trainer"),
            }
            modules["mlx.optimizers"].AdamW = lambda **kwargs: SimpleNamespace(
                state={}, step=SimpleNamespace(item=lambda: 3),
            )
            modules["mlx.utils"].tree_flatten = lambda tree: tree.items()
            modules["mlx_lm.utils"].load_model = load_model
            modules["mlx_lm.tuner.utils"].linear_to_lora_layers = lambda *args, **kwargs: None

            def final_evaluate(**kwargs):
                assert events[-1][0] == "reload"
                assert events[-1][1] == list(tensors.items())
                assert kwargs["model"] is models[1] and models[1] is not models[0]
                assert kwargs["num_batches"] == -1
                events.append(("final-evaluation",))
                return 1.25
            modules["mlx_lm.tuner.trainer"].evaluate = final_evaluate
            for key, value in modules.items():
                monkeypatch.setitem(sys.modules, key, value)
            monkeypatch.setattr(composition, "verify_native_material", lambda *a, **k: None)
            monkeypatch.setattr(composition, "attach_native_lora", lambda model, *a, **k: (model.weights, shapes))
            monkeypatch.setattr(composition, "make_mlx_sft_hooks", lambda *a, **k: (object(), object()))

            def train(*args, **kwargs):
                if with_progress:
                    callback = kwargs["training_callback"]
                    callback.on_val_loss_report({"iteration": 0, "val_loss": 3.0})
                    callback.on_train_loss_report({"iteration": 3, "train_loss": 2.0,
                                                   "learning_rate": .001, "trained_tokens": 17})
                else:
                    assert "training_callback" not in kwargs
                events.append(("upstream-stale-validation",))
                Path(kwargs["adapter_file"]).write_bytes(adapter_bytes(.5))
                events.append(("final-update-and-save",))
            monkeypatch.setattr(composition, "run_prepared_mlx_sft", train)
            runtime = json.dumps({"runtime": {
                "python_version": "3.11.14", "executable_sha256": "b" * 64,
                "dependency_lock_sha256": "c" * 64,
                "distributions": {"mlx": "0.32.2", "mlx-lm": "0.32.2"},
            }}).encode()
            def execute():
                return json.loads(composition.execute_native_sft(
                    bound=tokenize(source()), model_snapshot=snapshot,
                    lora=NativeLoRA(1, 1, 1, 0, ("q_proj",)),
                    optimizer=NativeAdamW(.001, .9, .999, 1e-8, 0, True, "constant"),
                    output=output, runtime_material=runtime, runtime_digest="f" * 64, dependency_lock=b"lock",
                    **({"_progress": lambda *args: progress_events.append(args)} if with_progress else {}),
                ))
            if reload_mode in {"noop", "partial"}:
                with pytest.raises(ValueError, match="reloaded adapter tensor differs"):
                    execute()
                assert len(models) == 2
                return
            result = execute()
            assert [item[0] for item in events] == [
                "upstream-stale-validation", "final-update-and-save", "reload", "final-evaluation",
            ]
            assert result["parameter_delta"]["delta_l2"] == pytest.approx(2 ** -.5)
            assert result["final_validation"]["loss"] == 1.25
            assert result["final_validation"]["phase"] == "reloaded_saved_final_adapter"
            assert [value[0] for value in progress_events] == (["validation", "training", "reload", "publication"] if with_progress else [])
            if with_progress: assert progress_events[-2] == ("reload", 3, {"loss": 1.25})
    finally:
        os.umask(old_umask)
