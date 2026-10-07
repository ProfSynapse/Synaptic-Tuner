"""Protected CLI boundary for the approval-bound HF A10G training smoke."""

from __future__ import annotations

from argparse import Namespace
from typing import Callable

from tuner.handlers.base import BaseHandler


_ACTIONS = frozenset({"preflight", "approve", "execute", "recover", "observe", "verify"})
_ERROR_MESSAGE = "Protected HF training smoke action failed."


class HFTrainingSmokeHandler(BaseHandler):
    """Route only the six frozen training-smoke transitions."""

    def __init__(
        self,
        args: Namespace | None = None,
        context=None,
        *,
        action_runner: Callable[..., dict[str, object]] | None = None,
    ) -> None:
        super().__init__(args=args, context=context)
        self._action_runner = action_runner

    @property
    def name(self) -> str:
        return "hf-training-smoke"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        try:
            action = str(getattr(self.args, "subcommand", "") or "").strip().lower()
            if action not in _ACTIONS:
                raise ValueError("invalid protected action")
            runner = self._action_runner
            if runner is None:
                from tuner.cloud.hf_training_smoke_operator import run_training_smoke_action

                runner = run_training_smoke_action
            result = runner(action, args=self.args, context=self.context)
            if not isinstance(result, dict) or any(not isinstance(key, str) for key in result):
                raise ValueError("invalid protected action result")
            self.output(result, f"Protected HF training smoke {action} completed.")
            return 0
        except Exception:
            # Provider/library exceptions may include request or credential details.
            # This boundary deliberately emits neither their text nor their type.
            self.output_error(_ERROR_MESSAGE, code="HF_TRAINING_SMOKE_ERROR")
            return 1


__all__ = ["HFTrainingSmokeHandler"]
