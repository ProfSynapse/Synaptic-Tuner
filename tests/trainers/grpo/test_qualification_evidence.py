"""Tiny actual CPU tensors exercise measurements, not GPU GRPO qualification."""
from copy import deepcopy
import math
import os
from types import SimpleNamespace
import time

import pytest

if os.environ.get("SYNAPTIC_TEST_GRPO_CPU_EVIDENCE") != "1":
    pytest.skip("requires the explicit pinned CPU GRPO evidence lane", allow_module_level=True)

import torch
assert torch.__version__ == "2.9.0+cpu" and torch.version.cuda is None

from Trainers.grpo import runtime_v1 as runtime

POLICY = {"max_trainable_bytes": 1024, "max_trainable_tensors": 4, "max_reward_values": 8}
CORE = {"config": {"training": {"max_steps": 3}}, "identity": "tiny CPU control"}


def measure(policy=None):
    return runtime._QualificationMeasurements(policy or POLICY, time.monotonic() + 30)


def model(dtype=None):
    import torch
    module = torch.nn.Module()
    module.register_parameter("weight", torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=dtype or torch.float32)))
    return module


def args(mode="group"):
    return SimpleNamespace(world_size=1, scale_rewards=mode)


def reward_and_log(m, values=(0.0, 1.0), std=0.5, fraction=0.0):
    original = list(values)
    calls = []
    def actual(completions, prompts=None, **kwargs):
        calls.append((completions, prompts, kwargs))
        return original
    completions, prompts, state = ["a", "b"], ["same", "same"], object()
    result = m.reward_wrapper(actual)(completions, prompts=prompts, trainer_state=state)
    assert result is original and calls == [(completions, prompts, {"trainer_state": state})]
    log = {"step": 1}
    if std is not None:
        log["reward_std"] = std
    if fraction is not None:
        log["frac_reward_zero_std"] = fraction
    m.log(SimpleNamespace(global_step=1), log)
    return [log]


def validate(record, logs, policy=None):
    runtime.validate_qualification_evidence_record(record, policy=policy or POLICY, core=CORE, log_history=logs)


@pytest.mark.parametrize("update", [False, True])
def test_actual_optimizer_update_distinguished_from_steps_or_save(update):
    import torch
    value = model()
    m = measure()
    m.begin(value, args())
    logs = reward_and_log(m)
    optimizer = torch.optim.SGD(value.parameters(), lr=0.1)
    if update:
        loss = value.weight.square().sum()
        loss.backward()
        optimizer.step()
    m.end(value)
    record = m.record(CORE)
    validate(record, logs)
    assert record["parameters"]["changed"] is update
    assert record["measurement_result"] == ("changed_with_grouped_variation" if update else "inconclusive")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_signed_zero_is_not_parameter_change(dtype):
    import torch
    value = model(getattr(torch, dtype))
    with torch.no_grad():
        value.weight.zero_()
    m = measure()
    m.begin(value, args())
    with torch.no_grad():
        value.weight.fill_(-0.0)
    m.end(value)
    assert not m.record(CORE)["parameters"]["changed"]
    assert torch.signbit(value.weight).all()  # Measurement never mutates the model.


def test_initialization_before_training_boundary_not_an_update():
    import torch
    value = model()
    with torch.no_grad():
        value.weight.add_(5)
    m = measure()
    m.begin(value, args())
    m.end(value)
    assert m.record(CORE)["measurement_result"] == "inconclusive"


@pytest.mark.parametrize("kind", ["empty", "frozen", "sparse", "meta", "noncontiguous", "dtype", "sharded"])
def test_unsupported_population_refuses(kind):
    import torch
    value = model()
    if kind == "empty":
        value = torch.nn.Module()
    elif kind == "frozen":
        value.weight.requires_grad_(False)
    elif kind == "sparse":
        value.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to_sparse())
    elif kind == "meta":
        value.weight = torch.nn.Parameter(torch.empty(2, device="meta"))
    elif kind == "noncontiguous":
        value.weight = torch.nn.Parameter(torch.ones(2, 3).T)
    elif kind == "sharded":
        value.weight._is_flat_param = True
    else:
        value.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        measure().begin(value, args())


@pytest.mark.parametrize("change", ["shape", "dtype", "name", "model"])
def test_same_trainable_inventory_and_model_required(change):
    import torch
    value = model()
    m = measure()
    m.begin(value, args())
    if change == "shape":
        value.weight = torch.nn.Parameter(torch.ones(3))
    elif change == "dtype":
        value.weight = torch.nn.Parameter(value.weight.to(torch.float16))
    elif change == "name":
        value.register_parameter("renamed", value.weight)
        del value.weight
    else:
        value = model()
    with pytest.raises(ValueError):
        m.end(value)


def test_incremental_count_and_bytes_reject_before_any_transfer(monkeypatch):
    value = model()
    for policy in ({**POLICY, "max_trainable_bytes": 7}, {**POLICY, "max_trainable_tensors": 1}):
        import torch
        m = measure(policy)
        if policy["max_trainable_tensors"] == 1:
            value.register_parameter("extra", torch.nn.Parameter(torch.ones(1)))
        monkeypatch.setattr(m, "_copy_chunk", lambda chunk: pytest.fail("bound checked before transfer"))
        with pytest.raises(ValueError):
            m.begin(value, args())


def test_real_transfers_never_exceed_fixed_chunk(monkeypatch):
    import torch
    value = model()
    size = runtime._EVIDENCE_CHUNK_BYTES // 4 + 1
    value.weight = torch.nn.Parameter(torch.ones(size))
    m = measure({**POLICY, "max_trainable_bytes": size * 4})
    sizes = []
    copy = m._copy_chunk
    def observe(chunk):
        sizes.append(chunk.numel() * chunk.element_size())
        return copy(chunk)
    monkeypatch.setattr(m, "_copy_chunk", observe)
    m.begin(value, args())
    m.end(value)
    assert sizes == [runtime._EVIDENCE_CHUNK_BYTES, 4] * 2


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_parameters_refuse(bad):
    import torch
    value = model()
    with torch.no_grad():
        value.weight[0] = bad
    with pytest.raises(ValueError):
        measure().begin(value, args())


def test_original_deadline_checked_after_chunk_no_reset(monkeypatch):
    value = model()
    m = measure()
    original = m.deadline
    copy = m._copy_chunk
    def expire(chunk):
        raw = copy(chunk)
        monkeypatch.setattr(m, "_clock", lambda: original)
        return raw
    monkeypatch.setattr(m, "_copy_chunk", expire)
    with pytest.raises(TimeoutError):
        m.begin(value, args())
    assert m.deadline == original


@pytest.mark.parametrize("world", [True, 2, 0])
def test_distributed_or_invalid_world_size_refuses(world):
    with pytest.raises(ValueError):
        measure().begin(model(), SimpleNamespace(world_size=world, scale_rewards="group"))


@pytest.mark.parametrize("mode,std,fraction", [
    ("batch", 0.5, 0.0), (None, 0.5, 0.0), ("group", None, None),
    ("group", 1e-12, 1.0), ("group", 0.5, None), ("group", 0.0, 1.0),
])
def test_global_variation_or_missing_zero_group_evidence_is_inconclusive(mode, std, fraction):
    import torch
    value = model()
    m = measure()
    m.begin(value, args(mode))
    logs = reward_and_log(m, std=std, fraction=fraction)
    with torch.no_grad():
        value.weight.add_(1)
    m.end(value)
    record = m.record(CORE)
    validate(record, logs)
    assert record["rewards"]["nonconstant_callbacks"] == 1
    assert record["measurement_result"] == "inconclusive"


def test_reward_exhaustion_refuses_before_rescoring():
    value = model()
    m = measure({**POLICY, "max_reward_values": 2})
    m.begin(value, args())
    reward_and_log(m)
    with pytest.raises(ValueError):
        m.reward_wrapper(lambda *a, **kw: pytest.fail("must not score"))(["again"])


@pytest.mark.parametrize("bad", [True, math.nan, math.inf, None])
def test_invalid_actual_reward_values_refuse(bad):
    value = model()
    m = measure()
    m.begin(value, args())
    with pytest.raises(ValueError):
        m.reward_wrapper(lambda *a, **kw: [bad])(["actual"])


@pytest.mark.parametrize("change", ["changed", "count", "inventory", "result", "policy", "core", "group_step", "group_log"])
def test_native_evidence_tampering_refuses(change):
    value = model()
    m = measure()
    m.begin(value, args())
    logs = reward_and_log(m)
    m.end(value)
    record = deepcopy(m.record(CORE))
    if change == "changed": record["parameters"]["changed"] = True
    elif change == "count": record["rewards"]["value_count"] = True
    elif change == "inventory": record["parameters"]["after_inventory_sha256"] = "f" * 64
    elif change == "result": record["measurement_result"] = "changed_with_grouped_variation"
    elif change == "policy": record["policy"]["max_reward_values"] += 1
    elif change == "core": record["core_workload_sha256"] = "f" * 64
    elif change == "group_step": record["rewards"]["grouped_observations"][0]["step"] = -1
    else: record["rewards"]["grouped_observations"][0]["reward_std"] = 99.0
    with pytest.raises(ValueError):
        validate(record, logs)


def core_factory(events, *, update):
    """Synthetic training algorithm delegates real optimizer/reward/callback APIs."""
    import json
    import struct
    from pathlib import Path
    import torch
    from .test_runtime_v1 import Tokenizer, dependencies

    parameters = model()
    token = Tokenizer()
    class Trainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.args = kwargs["args"]
            self.args.world_size = 1
            self.args.scale_rewards = "group"
            self.reward = kwargs["reward_funcs"]
            self.callbacks = kwargs["callbacks"]
            self.state = SimpleNamespace(global_step=0, log_history=[])

        def train(self, resume_from_checkpoint=None):
            assert resume_from_checkpoint is None
            control = object()
            for callback in self.callbacks:
                if hasattr(callback, "on_train_begin"):
                    assert callback.on_train_begin(self.args, self.state, control, model=self.model) is control
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            for step in range(1, self.args.max_steps + 1):
                actual = self.reward(["OK", "bad"], prompts=["same", "same"], trainer_state=self.state)
                if update:
                    optimizer.zero_grad()
                    self.model.weight.square().sum().backward()
                    optimizer.step()
                self.state.global_step = step
                std = torch.tensor(actual).std(correction=1).item()
                logs = {"reward_std": std, "frac_reward_zero_std": float(std <= 1e-8)}
                self.state.log_history.append({"step": step, **logs})
                for callback in self.callbacks:
                    if hasattr(callback, "on_log"):
                        callback.on_log(self.args, self.state, control, logs=logs)
            for callback in self.callbacks:
                if hasattr(callback, "on_train_end"):
                    assert callback.on_train_end(self.args, self.state, control, model=self.model) is control
            events.append("train")
            return SimpleNamespace(metrics={"actual_control_steps": self.state.global_step})

        def save_model(self, destination):
            root = Path(destination)
            root.joinpath("adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
            data = self.model.weight.detach().numpy().tobytes()
            header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, len(data)]}}, separators=(",", ":")).encode()
            root.joinpath("adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
            events.append("save")
    def load(path, **kwargs):
        assert kwargs["offline"] is True
        events.append("local-model")
        return parameters, token, False
    base = dependencies([])()
    base.update(trainer_class=Trainer, callback_class=object,
        model_loader=SimpleNamespace(load_model_and_tokenizer=load,
                                     apply_lora_adapters=lambda **kwargs: kwargs["model"]))
    return lambda: base


@pytest.mark.parametrize("update", [False, True])
def test_real_retained_core_callbacks_rewards_and_saved_records(tmp_path, update):
    import json
    from .test_runtime_v1 import inputs
    values = inputs(tmp_path)
    values["model_identity"]["tokenizer_revision"] = values["model_identity"]["revision"]
    values["workload"]["model"] = deepcopy(values["model_identity"])
    values["workload"]["config"] = deepcopy(values["config"])
    events = []
    result = runtime.run_grpo(**values, _dependency_factory=core_factory(events, update=update),
        _packaged_artifacts=True, qualification_evidence=POLICY,
        _child_deadline=time.monotonic() + 30)
    lineage = json.loads(result["training_lineage"].read_bytes())
    metrics = json.loads(result["training_metrics"].read_bytes())
    assert lineage["qualification_evidence"] == metrics["qualification_evidence"]
    runtime.validate_qualification_evidence_record(lineage["qualification_evidence"],
        policy=POLICY, core=values["workload"], log_history=metrics["log_history"])
    evidence = lineage["qualification_evidence"]
    assert evidence["parameters"]["changed"] is update
    assert evidence["rewards"]["callback_count"] == 3
    assert evidence["rewards"]["value_count"] == 6
    assert evidence["rewards"]["minimum"] == 0 and evidence["rewards"]["maximum"] == 1
    assert len(evidence["rewards"]["grouped_observations"]) == 3
    assert events == ["local-model", "train", "save"]


@pytest.mark.parametrize("bad", [False, {}, {**POLICY, "max_trainable_bytes": True}])
def test_bad_optional_policy_refuses_before_dependency_factory(tmp_path, bad):
    from .test_runtime_v1 import inputs
    values = inputs(tmp_path)
    values["model_identity"]["tokenizer_revision"] = values["model_identity"]["revision"]
    values["workload"]["model"] = deepcopy(values["model_identity"])
    with pytest.raises(ValueError):
        runtime.run_grpo(**values, _dependency_factory=lambda: pytest.fail("no ML factory"),
            _packaged_artifacts=True, qualification_evidence=bad, _child_deadline=time.monotonic() + 30)


@pytest.mark.parametrize("change", ["varying_single", "single_moments"])
def test_impossible_reward_counts_and_single_value_moments_refuse(change):
    value = model()
    m = measure()
    m.begin(value, args())
    logs = reward_and_log(m)
    m.end(value)
    record = m.record(CORE)
    record["rewards"].update(callback_count=1, value_count=1)
    if change == "single_moments":
        record["rewards"]["nonconstant_callbacks"] = 0
    with pytest.raises(ValueError):
        validate(record, logs)
