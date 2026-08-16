"""Side-effect-free capability list and describe handler."""

from __future__ import annotations

import sys
from argparse import Namespace
from uuid import uuid4

from synaptic_tuner.api.v1 import EventEnvelope, ResultEnvelope
from tuner.capabilities import builtin_registry, emit_diagnostic, write_event, write_result
from tuner.handlers.base import BaseHandler
from tuner.project import ProjectContext


class CapabilitiesHandler(BaseHandler):
    def __init__(self, args: Namespace, context: ProjectContext | None = None) -> None:
        super().__init__(args=args, context=context)

    @property
    def name(self) -> str:
        return "capabilities"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _machine_output(self, result: ResultEnvelope) -> None:
        if getattr(self.args, "events", None) == "jsonl":
            write_event(
                EventEnvelope(
                    event="capability.result",
                    capability=result.capability,
                    run_id=result.run_id,
                    sequence=0,
                    final=True,
                    result=result,
                )
            )
        else:
            write_result(result)

    def handle(self) -> int:
        subcommand = getattr(self.args, "subcommand", None) or "list"
        run_id = f"run_{uuid4().hex}"
        registry = builtin_registry()
        if subcommand == "list":
            descriptors = [item.to_dict() for item in registry.list()]
            result = ResultEnvelope(
                success=True,
                capability="capabilities.list",
                run_id=run_id,
                data={"capabilities": descriptors, "count": len(descriptors)},
            )
            human = "\n".join(f"{item['id']}: {item['summary']}" for item in descriptors)
        elif subcommand == "describe":
            capability_id = getattr(self.args, "capability_id", None)
            if not capability_id:
                return self._error(run_id, "Capability id is required.", "CAPABILITY_ID_REQUIRED")
            try:
                descriptor = registry.describe(capability_id)
            except KeyError:
                return self._error(run_id, f"Unknown capability: {capability_id}", "CAPABILITY_NOT_FOUND")
            result = ResultEnvelope(
                success=True,
                capability="capabilities.describe",
                run_id=run_id,
                data={"capability": descriptor.to_dict()},
            )
            human = f"{descriptor.id}: {descriptor.summary}"
        else:
            return self._error(run_id, f"Unknown capabilities subcommand: {subcommand}", "CAPABILITIES_SUBCOMMAND_INVALID")

        if self.json_mode or getattr(self.args, "events", None) == "jsonl":
            self._machine_output(result)
        else:
            print(human)
        return 0

    def _error(self, run_id: str, message: str, code: str) -> int:
        emit_diagnostic(message, details={"code": code})
        if self.json_mode or getattr(self.args, "events", None) == "jsonl":
            self._machine_output(
                ResultEnvelope(
                    success=False,
                    capability="capabilities.discovery",
                    run_id=run_id,
                    data={"error": {"code": code, "message": message}},
                )
            )
        return 2


__all__ = ["CapabilitiesHandler"]
