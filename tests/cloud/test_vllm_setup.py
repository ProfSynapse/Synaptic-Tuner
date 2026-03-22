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
    cmd = mock_popen.call_args.args[0]
    assert env["TORCH_COMPILE_DISABLE"] == "1"
    assert env["VLLM_USE_V1"] == "1"
    assert "--enforce-eager" in cmd


def test_start_vllm_server_auto_detects_tensor_parallel_size():
    process = MagicMock()

    with patch("Evaluator.vllm_setup.subprocess.Popen", return_value=process) as mock_popen:
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=4):
                started = start_vllm_server(
                    model="Qwen/Qwen3-4B",
                    wait_for_ready=False,
                    show_logs=False,
                )

    assert started is True
    cmd = mock_popen.call_args.args[0]
    assert "--tensor-parallel-size" in cmd
    tp_index = cmd.index("--tensor-parallel-size")
    assert cmd[tp_index + 1] == "4"
    assert "--enforce-eager" in cmd


def test_start_vllm_server_disables_tensor_parallel_for_prequant_bnb_models():
    process = MagicMock()

    with patch("Evaluator.vllm_setup.subprocess.Popen", return_value=process) as mock_popen:
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=4):
                started = start_vllm_server(
                    model="unsloth/qwen3-4b-unsloth-bnb-4bit",
                    wait_for_ready=False,
                    show_logs=False,
                )

    assert started is True
    cmd = mock_popen.call_args.args[0]
    assert "--tensor-parallel-size" not in cmd
    assert "--enforce-eager" in cmd
