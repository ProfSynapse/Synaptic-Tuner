from unittest.mock import MagicMock, patch

from Evaluator.vllm_setup import start_vllm_server


def test_start_vllm_server_defaults_to_v1_engine():
    process = MagicMock()

    with patch("Evaluator.vllm_setup.subprocess.Popen", return_value=process) as mock_popen:
        started = start_vllm_server(
            model="unsloth/qwen3-1.7b",
            wait_for_ready=False,
            show_logs=False,
        )

    assert started is True
    env = mock_popen.call_args.kwargs["env"]
    assert env["TORCH_COMPILE_DISABLE"] == "1"
    assert env["VLLM_USE_V1"] == "1"
