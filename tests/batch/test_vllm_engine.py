"""Mocked contract tests for deterministic vLLM batch generation."""

from __future__ import annotations

import json
import sys
import types

import pytest

from tuner.batch.engines.base import GenerateItem, hash_token_ids
from tuner.batch.engines import vllm_engine
from tuner.batch.engines.vllm_engine import VLLMGenerateEngine


def _install_fake_vllm(monkeypatch, *, version: str = "0.23.0"):
    calls = {}

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            calls["structured"] = kwargs
            self.kwargs = kwargs

    class FakeStructuredOutputsConfig:
        def __init__(self, **kwargs):
            calls["structured_config"] = kwargs
            self.kwargs = kwargs

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            calls["sampling"] = kwargs
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["llm"] = kwargs

        def generate(self, prompts, params):
            calls["prompts"] = prompts
            calls["params"] = params
            return [
                types.SimpleNamespace(
                    prompt_token_ids=[10, index],
                    outputs=[
                        types.SimpleNamespace(
                            text=json.dumps({"answer": prompt}),
                            token_ids=[20, index],
                            finish_reason="stop",
                        )
                    ],
                )
                for index, prompt in enumerate(prompts)
            ]

    module = types.ModuleType("vllm")
    module.__version__ = version
    module.LLM = FakeLLM
    module.SamplingParams = FakeSamplingParams
    sampling_module = types.ModuleType("vllm.sampling_params")
    sampling_module.StructuredOutputsParams = FakeStructuredOutputsParams
    config_module = types.ModuleType("vllm.config")
    config_module.StructuredOutputsConfig = FakeStructuredOutputsConfig
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    return calls


def test_vllm_refuses_without_batch_invariant_before_import(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_BATCH_INVARIANT=1"):
        VLLMGenerateEngine(
            "model",
            expected_vllm_version="0.23.0",
            min_compute_capability="8.0",
        )


def test_vllm_refuses_unpinned_or_mismatched_version(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _install_fake_vllm(monkeypatch, version="0.23.0")
    with pytest.raises(ValueError, match="expected-vllm-version"):
        VLLMGenerateEngine("model")
    with pytest.raises(RuntimeError, match="version mismatch"):
        VLLMGenerateEngine(
            "model",
            expected_vllm_version="0.17.1",
            min_compute_capability="8.0",
        )
    with pytest.raises(ValueError, match="min-compute-capability"):
        VLLMGenerateEngine("model", expected_vllm_version="0.23.0")


def test_vllm_pins_engine_schema_sampling_and_prompt_evidence(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    calls = _install_fake_vllm(monkeypatch)
    devices = [
        {"index": 0, "name": "RTX 3090", "compute_capability": "8.6"},
        {"index": 1, "name": "RTX 3090", "compute_capability": "8.6"},
    ]
    hardware = {
        "devices": devices,
        "nvidia_driver_versions": ["591.86"],
        "cuda_runtime": "12.9",
        "torch_version": "2.9.0",
    }
    monkeypatch.setattr(
        vllm_engine,
        "_require_batch_invariant_hardware",
        lambda minimum, tensor_parallel_size: {
            **hardware,
            "devices": devices[:tensor_parallel_size],
        },
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    engine = VLLMGenerateEngine(
        "org/model",
        revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        dtype="bfloat16",
        seed=17,
        do_sample=True,
        temperature=0.25,
        top_p=0.8,
        max_new_tokens=96,
        min_new_tokens=1,
        json_schema=schema,
        structured_output_backend="xgrammar",
        structured_output_disable_any_whitespace=True,
        expected_vllm_version="0.23.0",
        min_compute_capability="8.0",
        tensor_parallel_size=2,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "audio": 0},
        gpu_memory_utilization=0.90,
    )

    assert calls["structured_config"] == {
        "backend": "xgrammar",
        "disable_any_whitespace": True,
    }
    expected_llm = {
        "model": "org/model",
        "trust_remote_code": False,
        "revision": "model-sha",
        "tokenizer_revision": "tokenizer-sha",
        "dtype": "bfloat16",
        "seed": 17,
        "tensor_parallel_size": 2,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
        "max_model_len": 2048,
        "limit_mm_per_prompt": {"image": 0, "audio": 0},
        "gpu_memory_utilization": 0.90,
    }
    structured_config = calls["llm"].pop("structured_outputs_config")
    assert structured_config.kwargs == {
        "backend": "xgrammar",
        "disable_any_whitespace": True,
    }
    assert calls["llm"] == expected_llm
    assert calls["structured"] == {"json": schema}

    results = engine.generate(
        [GenerateItem(id="a", prompt="alpha"), GenerateItem(id="b", prompt="beta")],
        batch_size=2,
    )
    assert calls["sampling"]["structured_outputs"].kwargs == {"json": schema}
    assert calls["sampling"]["seed"] == 17
    assert calls["sampling"]["temperature"] == 0.25
    assert calls["sampling"]["top_p"] == 0.8
    assert results[0].prompt_token_len == 2
    assert results[0].prompt_token_ids_sha256 == hash_token_ids([10, 0])
    assert engine.provenance() == {
        "vllm_version": "0.23.0",
        "vllm_batch_invariant": True,
        "structured_outputs": True,
        "structured_output_backend": "xgrammar",
        "structured_output_disable_any_whitespace": True,
        "hardware": hardware,
        "documented_compute_capability_floor": "8.0",
        "effective_compute_capability_floor": "8.0",
    }

    original_generate = engine.llm.generate
    oom_calls = []

    def oom_above_one(prompts, params):
        if len(prompts) > 1:
            raise RuntimeError("CUDA out of memory")
        return original_generate(prompts, params)

    engine.llm.generate = oom_above_one
    retried = engine.generate(
        [GenerateItem(id="c", prompt="gamma"), GenerateItem(id="d", prompt="delta")],
        batch_size=2,
        on_oom=lambda old, new: oom_calls.append((old, new)),
    )
    assert [result.id for result in retried] == ["c", "d"]
    assert oom_calls == [(2, 1)]


def test_documented_batch_invariance_floor_cannot_be_lowered():
    assert vllm_engine._documented_batch_invariance_floor("0.18.0") == (9, 0)
    assert vllm_engine._documented_batch_invariance_floor("0.22.1") == (9, 0)
    assert vllm_engine._documented_batch_invariance_floor("0.23.0") == (8, 0)
