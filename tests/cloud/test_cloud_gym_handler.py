from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

from tuner.handlers.cloud_gym_handler import CloudGymHandler


def test_cloud_gym_defaults_to_vault_scenario_and_local_env(repo_root):
    args = Namespace(
        json=False,
        run="latest",
        method="sft",
        bucket="test-user/toolset-training-artifacts",
        scenario=None,
        tags=None,
        env_backend=None,
        env_template=None,
        env_tool_schema=None,
        env_exec_config=None,
        upload_to_hf=None,
        update_model_card=False,
        gpu=None,
        timeout_hours=2.0,
        auto_confirm=True,
    )
    handler = CloudGymHandler(args=args)
    handler._repo_root = repo_root

    with patch("tuner.handlers.cloud_gym_handler.CloudEvalHandler") as mock_eval_handler_cls:
        mock_eval_handler = MagicMock()
        mock_eval_handler.handle.return_value = 0
        mock_eval_handler_cls.return_value = mock_eval_handler

        exit_code = handler.handle()

    assert exit_code == 0
    eval_args = mock_eval_handler_cls.call_args.kwargs["args"]
    assert eval_args.run == "latest"
    assert eval_args.method == "sft"
    assert eval_args.bucket == "test-user/toolset-training-artifacts"
    assert eval_args.scenario == ["vault_gym.yaml"]
    assert eval_args.env_backend == "local"
