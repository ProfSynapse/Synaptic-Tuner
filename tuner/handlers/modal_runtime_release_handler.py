"""Closed CLI boundary for protected packaged Modal runtime releases."""

from __future__ import annotations

from argparse import Namespace
from typing import Callable

from tuner.handlers.base import BaseHandler


_ACTIONS = frozenset({"preflight", "approve", "execute", "recover", "observe", "verify"})
_ERROR_MESSAGE = "Protected Modal runtime release action failed."


class ModalRuntimeReleaseHandler(BaseHandler):
    def __init__(
        self, args: Namespace | None = None, context=None, *,
        action_runner: Callable[..., dict[str, object]] | None = None,
    ) -> None:
        super().__init__(args=args, context=context)
        self._action_runner = action_runner

    @property
    def name(self) -> str:
        return "modal-runtime-release"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        try:
            action = str(getattr(self.args, "subcommand", "") or "").strip().lower()
            if action not in _ACTIONS:
                raise ValueError("invalid protected action")
            runner = self._action_runner
            if runner is None:
                from tuner.cloud.modal_runtime_release_operator import (
                    run_modal_runtime_release_cli_action,
                )
                runner = run_modal_runtime_release_cli_action
            result = runner(action, args=self.args, context=self.context)
            if type(result) is not dict \
                    or any(type(key) is not str for key in result):
                raise ValueError("invalid protected action result")
            self.output(result, f"Protected Modal runtime release {action} completed.")
            return 0
        except Exception:
            # Provider errors may retain request, object, or credential detail.
            self.output_error(_ERROR_MESSAGE, code="MODAL_RUNTIME_RELEASE_ERROR")
            return 1


__all__ = ["ModalRuntimeReleaseHandler"]
