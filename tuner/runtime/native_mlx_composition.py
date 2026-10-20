"""Explicit candidate MLX model/LoRA/optimizer composition in an admitted child.

This is not a launcher or public provider. The bounded supervisor and the host
grant must precede entry. No Hub download or implicit model/profile selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re

from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_artifacts import adapter_tensors, verify_adapter_delta
from tuner.runtime.native_mlx_input import BoundNativeInput, NativeModelSnapshot
from tuner.runtime.native_mlx_material import canonical, strict_json, verify_native_material
from tuner.runtime.native_mlx_runtime import MLXRuntimeExpectation
from tuner.runtime.native_mlx_sft import make_mlx_sft_hooks, run_prepared_mlx_sft


def _number(value, minimum, maximum, *, closed=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("native optimizer/LoRA number is invalid")
    if not (minimum <= value <= maximum if closed else minimum < value <= maximum):
        raise ValueError("native optimizer/LoRA number is outside bounds")


@dataclass(frozen=True)
class NativeLoRA:
    layers: int
    rank: int
    scale: float
    dropout: float
    targets: tuple[str, ...]

    def __post_init__(self):
        if (type(self.layers) is not int or not 1 <= self.layers <= 4096
                or type(self.rank) is not int or not 1 <= self.rank <= 4096):
            raise ValueError("native LoRA rank/layer count is invalid")
        _number(self.scale, 0, 65536)
        _number(self.dropout, 0, 1, closed=True)
        if self.dropout == 1:
            raise ValueError("native LoRA dropout must be below one")
        if (type(self.targets) is not tuple or not 1 <= len(self.targets) <= 256
                or self.targets != tuple(sorted(set(self.targets)))
                or any(type(key) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", key)
                       for key in self.targets)):
            raise ValueError("native LoRA targets must be unique explicit layer-relative keys")


@dataclass(frozen=True)
class NativeAdamW:
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    bias_correction: bool
    schedule: str

    def __post_init__(self):
        _number(self.learning_rate, 0, 1)
        _number(self.epsilon, 0, 1)
        for value in (self.beta1, self.beta2, self.weight_decay):
            _number(value, 0, 1, closed=True)
        if self.beta1 == 1 or self.beta2 == 1:
            raise ValueError("native AdamW beta must be below one")
        if type(self.bias_correction) is not bool or self.schedule != "constant":
            raise ValueError("native AdamW requires explicit constant schedule and bias policy")

    def arguments(self):
        return {
            "learning_rate": self.learning_rate, "betas": [self.beta1, self.beta2],
            "eps": self.epsilon, "weight_decay": self.weight_decay,
            "bias_correction": self.bias_correction,
        }


def attach_native_lora(model, config: NativeLoRA, *, convert, flatten):
    """Verify every selected target exists and only those adapters are trainable."""
    if type(config) is not NativeLoRA:
        raise TypeError("native LoRA controls must be exact")
    layers = model.layers
    if not config.layers <= len(layers):
        raise ValueError("native LoRA layer selection exceeds model")
    global_names = {}
    for name, module in model.named_modules():
        global_names.setdefault(id(module), []).append(name)
    expected = []
    for layer in layers[-config.layers:]:
        modules = dict(layer.named_modules())
        for key in config.targets:
            if key not in modules or len(global_names.get(id(modules[key]), [])) != 1:
                raise ValueError("native LoRA target is missing or aliased")
            full_name = global_names[id(modules[key])][0]
            expected.extend((full_name + ".lora_a", full_name + ".lora_b"))
    if len(set(expected)) != len(expected):
        raise ValueError("native LoRA target is shared between layers")
    model.freeze()
    convert(model, config.layers, {
        "rank": config.rank, "scale": config.scale, "dropout": config.dropout,
        "keys": list(config.targets),
    }, use_dora=False)
    trainable = dict(flatten(model.trainable_parameters()))
    if tuple(sorted(trainable)) != tuple(sorted(expected)):
        raise ValueError("native realized trainable parameters differ from configured targets")
    shapes = tuple((name, tuple(value.shape)) for name, value in sorted(trainable.items()))
    if any(len(shape) != 2 or config.rank not in shape for _, shape in shapes):
        raise ValueError("native realized LoRA dimensions differ")
    return trainable, shapes


def execute_native_sft(
    *, bound: BoundNativeInput, model_snapshot: NativeModelSnapshot,
    lora: NativeLoRA, optimizer: NativeAdamW, output: Path,
    runtime_material: bytes, runtime_digest: str, dependency_lock: bytes,
) -> bytes:
    """Produce final saved-adapter evidence only after exact offline readback.

    A successful return is still candidate worker evidence, not an authenticated
    public artifact manifest or hardware capability promotion.
    """
    if (type(bound) is not BoundNativeInput or type(model_snapshot) is not NativeModelSnapshot
            or type(lora) is not NativeLoRA or type(optimizer) is not NativeAdamW
            or bound.runtime_digest != runtime_digest):
        raise ValueError("native composition binding differs")
    verify_native_material(runtime_material, expected_digest=runtime_digest, dependency_lock=dependency_lock)
    runtime = strict_json(runtime_material)["runtime"]
    expectation = MLXRuntimeExpectation.from_dict({
        name: runtime[name] for name in MLXRuntimeExpectation.__dataclass_fields__
    })
    with NativeTree(Path(model_snapshot.root), private=True) as source, NativeTree(output, private=True) as destination:
        model_snapshot.verify(source)
        if destination.members():
            raise ValueError("native training output must be fresh and empty")
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optimizers
        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.tuner.trainer import evaluate

        mx.random.seed(bound.prepared.config.seed)
        # load_model accepts a Path and does not pass through Hub _download.
        model, _ = load_model(Path(model_snapshot.root), lazy=False, strict=True,
                              trust_remote_code=False)
        parameters, shapes = attach_native_lora(
            model, lora, convert=linear_to_lora_layers, flatten=tree_flatten,
        )
        mx.eval(parameters)
        initial_path = output / "initial-adapters.safetensors"
        final_path = output / "adapters.safetensors"
        mx.save_safetensors(str(initial_path), parameters)
        initial = destination.read(initial_path.name)
        adapter_tensors(initial)
        adam = optimizers.AdamW(**optimizer.arguments())
        run_prepared_mlx_sft(
            bound.prepared, model=model, optimizer=adam, adapter_file=str(final_path),
            runtime=expectation, dependency_lock=dependency_lock,
        )
        mx.eval(adam.state)
        completed_updates = adam.step.item()
        if type(completed_updates) is not int or completed_updates != bound.prepared.config.iterations:
            raise ValueError("native optimizer update count differs")
        final = destination.read(final_path.name)
        delta = verify_adapter_delta(initial, final, expected_shapes=shapes)
        saved = mx.load(str(final_path))
        if tuple(sorted(saved)) != tuple(name for name, _ in shapes):
            raise ValueError("native saved adapter inventory differs")
        model.load_weights(list(saved.items()), strict=False)
        mx.eval(model.parameters())
        # Upstream evaluates before its final update. Evaluate the saved adapter
        # separately after loading it, including the partial validation batch.
        iterate, loss = make_mlx_sft_hooks(bound.prepared, mx=mx, nn=nn)
        final_loss = evaluate(
            model=model, dataset=bound.prepared.validation,
            batch_size=bound.prepared.config.batch_size, num_batches=-1,
            max_seq_length=bound.prepared.config.max_seq_length,
            loss=loss, iterate_batches=iterate, clear_cache_threshold=0,
        )
        if type(final_loss) not in (int, float) or not math.isfinite(final_loss) or final_loss < 0:
            raise ValueError("native saved final adapter validation is nonfinite")
        if destination.read(final_path.name) != final:
            raise ValueError("native final adapter changed during evaluation")
        model_snapshot.verify(source)
        destination.check()
        adapter_config = canonical({
            "fine_tune_type": "lora", "num_layers": lora.layers,
            "lora_parameters": {
                "rank": lora.rank, "scale": lora.scale, "dropout": lora.dropout,
                "keys": list(lora.targets),
            },
        })
        destination.write_exclusive("adapter_config.json", adapter_config)
        result = canonical({
            "schema": "synaptic-native-mlx-sft-result/v1",
            "status": "CANDIDATE_WORKER_COMPLETE",
            "input_binding_digest": bound.digest, "workload_digest": bound.workload_digest,
            "runtime_digest": runtime_digest, "model_snapshot_digest": model_snapshot.digest,
            "lora": asdict(lora), "optimizer": asdict(optimizer),
            "completed_updates": completed_updates,
            "adapter_format": "mlx-lm-lora-safetensors",
            "parameter_delta": delta,
            "final_validation": {
                "adapter_sha256": hashlib.sha256(final).hexdigest(),
                "phase": "reloaded_saved_final_adapter",
                "loss": final_loss, "rows": len(bound.prepared.validation),
                "supervised_tokens": sum(sum(label != -100 for label in row.labels[1:])
                                         for row in bound.prepared.validation),
            },
        })
        destination.write_exclusive("native_result.json", result)
        return result
