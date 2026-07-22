"""Mocked contract tests for deterministic vLLM batch generation."""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from tuner.batch.engines.base import GenerateItem, hash_token_ids
from tuner.batch.engines import vllm_engine
from tuner.batch.engines.vllm_engine import VLLMGenerateEngine


def _install_fake_vllm(
    monkeypatch, *, version: str = "0.23.0", eos_token_id=1,
):
    calls = {}

    class FakeTokenizer:
        tokenizations = {
            "<turn|>": [106],
            # A normal bad_words string would also inspect this different
            # leading-space tokenization. The exact-ID path must not.
            " <turn|>": [400, 106],
            "<|tool_response>": [50],
            " <|tool_response>": [50],
            "<same-id>": [106],
            " <same-id>": [106],
            "two tokens": [7, 8],
            " two tokens": [9, 10],
            "<leading-alias>": [77],
            " <leading-alias>": [88],
            "<canonical-eos>": [1],
            " <canonical-eos>": [1],
        }

        def __init__(self):
            self.eos_token_id = eos_token_id

        def encode(self, text, add_special_tokens=False):
            calls.setdefault("tokenizer_encodes", []).append(
                (text, add_special_tokens)
            )
            return list(self.tokenizations.get(text, [999]))

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            calls["structured"] = kwargs
            self.kwargs = kwargs

    class FakeStructuredOutputsConfig:
        def __init__(self, **kwargs):
            calls["structured_config"] = kwargs
            self.kwargs = kwargs

    class FakeSamplingParams:
        def __init__(self, _bad_words_token_ids=None, **kwargs):
            if _bad_words_token_ids is not None:
                kwargs["_bad_words_token_ids"] = _bad_words_token_ids
            calls["sampling"] = kwargs
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["llm"] = kwargs

        def get_tokenizer(self):
            return FakeTokenizer()

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
            vllm_model_runner="v1",
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
        vllm_model_runner="v1",
        min_compute_capability="8.0",
        tensor_parallel_size=2,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "audio": 0},
        gpu_memory_utilization=0.90,
        suppress_tokens=["<turn|>", "<|tool_response>"],
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
    assert calls["sampling"]["_bad_words_token_ids"] == [[106], [50]]
    assert "bad_words" not in calls["sampling"]
    assert calls["tokenizer_encodes"] == [
        ("<turn|>", False),
        ("<|tool_response>", False),
    ]
    assert results[0].prompt_token_len == 2
    assert results[0].prompt_token_ids_sha256 == hash_token_ids([10, 0])
    assert engine.provenance() == {
        "vllm_version": "0.23.0",
        "vllm_batch_invariant": True,
        "vllm_model_runner": "v1",
        "structured_outputs": True,
        "structured_output_backend": "xgrammar",
        "structured_output_disable_any_whitespace": True,
        "hardware": hardware,
        "documented_compute_capability_floor": "8.0",
        "effective_compute_capability_floor": "8.0",
        "suppress_tokens": ["<turn|>", "<|tool_response>"],
        "suppressed_token_ids": [106, 50],
        "suppressed_bad_word_token_ids": [[106], [50]],
    }
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"

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


@pytest.mark.parametrize(
    ("suppress_tokens", "message"),
    [
        ([""], "non-empty strings"),
        (["<turn|>", "<turn|>"], "duplicate strings"),
        (["two tokens"], "exactly one"),
        (["<turn|>", "<same-id>"], "distinct token IDs"),
    ],
)
def test_vllm_rejects_invalid_suppress_tokens(
    monkeypatch, suppress_tokens, message,
):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _install_fake_vllm(monkeypatch)
    monkeypatch.setattr(
        vllm_engine,
        "_require_batch_invariant_hardware",
        lambda minimum, tensor_parallel_size: {
            "devices": [],
            "nvidia_driver_versions": [],
            "cuda_runtime": "test",
            "torch_version": "test",
        },
    )
    with pytest.raises(ValueError, match=message):
        VLLMGenerateEngine(
            "model",
            expected_vllm_version="0.23.0",
            vllm_model_runner="v1",
            min_compute_capability="8.0",
            suppress_tokens=suppress_tokens,
        )


def test_vllm_default_omits_suppression_sampling_and_provenance(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    calls = _install_fake_vllm(monkeypatch)
    monkeypatch.setattr(
        vllm_engine,
        "_require_batch_invariant_hardware",
        lambda minimum, tensor_parallel_size: {
            "devices": [],
            "nvidia_driver_versions": [],
            "cuda_runtime": "test",
            "torch_version": "test",
        },
    )
    engine = VLLMGenerateEngine(
        "model",
        expected_vllm_version="0.23.0",
        vllm_model_runner="v1",
        min_compute_capability="8.0",
    )
    engine._sampling_params()
    assert "bad_words" not in calls["sampling"]
    assert "_bad_words_token_ids" not in calls["sampling"]
    assert "suppress_tokens" not in engine.provenance()
    assert "suppressed_token_ids" not in engine.provenance()
    assert "suppressed_bad_word_token_ids" not in engine.provenance()


@pytest.mark.parametrize("eos_token_id", [1, [1, 106]])
def test_vllm_rejects_canonical_eos_suppression(monkeypatch, eos_token_id):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _install_fake_vllm(monkeypatch, eos_token_id=eos_token_id)
    monkeypatch.setattr(
        vllm_engine,
        "_require_batch_invariant_hardware",
        lambda minimum, tensor_parallel_size: {
            "devices": [],
            "nvidia_driver_versions": [],
            "cuda_runtime": "test",
            "torch_version": "test",
        },
    )
    suppressed = "<canonical-eos>" if eos_token_id == 1 else "<turn|>"
    with pytest.raises(ValueError, match="canonical EOS suppression is forbidden"):
        VLLMGenerateEngine(
            "model",
            expected_vllm_version="0.23.0",
            vllm_model_runner="v1",
            min_compute_capability="8.0",
            suppress_tokens=[suppressed],
        )


def test_vllm_refuses_suppression_without_exact_id_api(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _install_fake_vllm(monkeypatch)

    class SamplingParamsWithoutExactIds:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        sys.modules["vllm"], "SamplingParams", SamplingParamsWithoutExactIds
    )
    monkeypatch.setattr(
        vllm_engine,
        "_require_batch_invariant_hardware",
        lambda minimum, tensor_parallel_size: {
            "devices": [],
            "nvidia_driver_versions": [],
            "cuda_runtime": "test",
            "torch_version": "test",
        },
    )
    with pytest.raises(RuntimeError, match="exact-ID suppression API"):
        VLLMGenerateEngine(
            "model",
            expected_vllm_version="0.23.0",
            vllm_model_runner="v1",
            min_compute_capability="8.0",
            suppress_tokens=["<turn|>"],
        )
