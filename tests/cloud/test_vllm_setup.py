from types import SimpleNamespace
from unittest.mock import patch

from Evaluator import vllm_setup
from Evaluator.vllm_runtime import (
    ExplicitNetworkVLLMSource,
    VLLMStartupSpec,
    _projection,
)


def test_network_runtime_defaults_to_v1_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    environment = vllm_setup.network_runtime_environment()
    spec = VLLMStartupSpec(
        source=ExplicitNetworkVLLMSource("unsloth/qwen3-1.7b"),
        served_model_name="model",
    )
    projection = _projection(spec, cwd=tmp_path, environment=environment)
    assert projection.environment["TORCH_COMPILE_DISABLE"] == "1"
    assert projection.environment["VLLM_USE_V1"] == "1"
    assert "--enforce-eager" in projection.argv


def test_operator_tensor_parallel_auto_detects_devices():
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
        )
    )
    with patch.dict("sys.modules", {"torch": torch}):
        assert vllm_setup.resolve_tensor_parallel_size("Qwen/Qwen3-4B") == 4


def test_operator_tensor_parallel_disables_prequant_bnb_sharding():
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
        )
    )
    with patch.dict("sys.modules", {"torch": torch}):
        assert (
            vllm_setup.resolve_tensor_parallel_size("unsloth/qwen3-4b-unsloth-bnb-4bit")
            == 1
        )
