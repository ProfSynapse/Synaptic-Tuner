"""Convenience handler for running the vault gym against a trained cloud model."""

from __future__ import annotations

from argparse import Namespace
from typing import Optional

from tuner.handlers.base import BaseHandler
from tuner.handlers.cloud_eval_handler import CloudEvalHandler


class CloudGymHandler(BaseHandler):
    """Run the vault gym scenario pack against a bucketed training run on HF Jobs."""

    @property
    def name(self) -> str:
        return "cloud-gym"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _build_eval_args(self) -> Namespace:
        scenarios = getattr(self.args, "scenario", None) or ["vault_gym.yaml"]
        return Namespace(
            json=self.json_mode,
            run=getattr(self.args, "run", None),
            method=getattr(self.args, "method", None),
            bucket=getattr(self.args, "bucket", None),
            preset=None,
            scenario=scenarios,
            tags=getattr(self.args, "tags", None),
            env_backend=getattr(self.args, "env_backend", None) or "local",
            env_template=getattr(self.args, "env_template", None),
            env_tool_schema=getattr(self.args, "env_tool_schema", None),
            env_exec_config=getattr(self.args, "env_exec_config", None),
            upload_to_hf=getattr(self.args, "upload_to_hf", None),
            update_model_card=bool(getattr(self.args, "update_model_card", False)),
            gpu=getattr(self.args, "gpu", None),
            timeout_hours=getattr(self.args, "timeout_hours", None),
            auto_confirm=getattr(self.args, "auto_confirm", False),
        )

    def handle(self) -> int:
        handler = CloudEvalHandler(args=self._build_eval_args())
        handler._repo_root = self.repo_root
        return handler.handle()
