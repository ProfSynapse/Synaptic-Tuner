"""Run state folded purely from journal records.

Nothing here is trusted from memory: after a crash the broker rebuilds this
object by replaying the verified journal, so budgets and request dispositions
can never reset on resume.
"""

from __future__ import annotations

from typing import Mapping

from .canon import canonical_bytes, sha256_hex
from .ledger import Ledger

PHASES = ("registered", "discovery", "final_patch_frozen", "independent_evaluation", "closed")
TERMINAL_REQUEST = ("denied", "done", "rejected")
CONTEXT_GENESIS = "0" * 64


class RunState:
    def __init__(self, limits: Mapping[str, float]):
        self.phase = "registered"
        self.status = "running"  # running | halted | complete
        self.halt: dict | None = None
        self.session_id: str | None = None
        self.turn_count = 0
        self.open_turn: str | None = None
        self.last_turn_clean = True
        self.outbox: list[dict] = []
        self.context_digest = CONTEXT_GENESIS
        self.requests: dict[str, dict] = {}
        self.finals: dict[str, dict] = {}
        self.order: list[tuple[str, str]] = []  # arrival order of ("request"|"final", id)
        self.ledger = Ledger(limits)
        self.artifact_names: set[str] = set()
        self.artifacts: list[dict] = []
        self.results: list[dict] = []  # ordered dispatch results (evidence)
        self.capture_incomplete = False
        self.unqualified_turns: set[str] = set()
        self.unknown_events = 0
        self.returned_models: list[str] = []
        self.model_verified_turns: set[str] = set()
        self.completed_turns: set[str] = set()
        self.delivered_receipts: set[str] = set()
        self.needs_bootstrap = False
        self.turn_messages: list[dict] = []
        self.turn_receipts: set[str] = set()
        self.raw_referenced = 0
        self.frozen: dict | None = None
        self.frozen_record_digest: str | None = None
        self.terminal: dict | None = None
        self.resume_count = 0
        self.handoffs: list[dict] = []
        self.evaluation_reference: str | None = None
        self.registered: dict | None = None
        self.records_seen = 0

    # ------------------------------------------------------------ helpers
    def next_request_id(self) -> str:
        return f"req-{len(self.requests) + 1:04d}"

    def next_final_id(self) -> str:
        return f"fc-{len(self.finals) + 1:04d}"

    def pending(self) -> tuple[str, dict] | None:
        for kind, iid in self.order:
            if kind == "request":
                if self.requests[iid]["status"] not in TERMINAL_REQUEST:
                    return kind, self.requests[iid]
            elif self.finals[iid]["status"] == "received":
                return kind, self.finals[iid]
        return None

    def push_message(self, msg: Mapping[str, str]) -> None:
        self.outbox.append(dict(msg))

    # --------------------------------------------------------------- fold
    def apply(self, rec: dict) -> None:
        self.records_seen += 1
        d = rec["data"]
        for entry in d.get("ledger", ()):  # budget entries ride on the record that causes them
            self.ledger.apply(entry)
        fn = getattr(self, "_on_" + rec["type"], None)
        if fn:
            fn(rec, d)

    def _on_run_registered(self, rec, d):
        self.registered = d

    def _on_state_transition(self, rec, d):
        self.phase = d["to"]

    def _on_phase_advanced(self, rec, d):
        self.phase = d["to"]
        self.evaluation_reference = d.get("reference")

    def _on_turn_start(self, rec, d):
        self.turn_messages = list(d["messages"])
        self.turn_receipts = set()
        for rid, req in self.requests.items():
            if req.get("message") in d["messages"]:
                self.turn_receipts.add(rid)
        self.turn_count += 1
        self.open_turn = d["turn_id"]
        self.outbox = []
        self.context_digest = d["context_digest"]
        self.needs_bootstrap = False
        if d.get("session_id"):
            self.session_id = d["session_id"]

    def _on_turn_end(self, rec, d):
        self.open_turn = None
        self.last_turn_clean = d["status"] == "complete"
        if self.last_turn_clean:
            self.completed_turns.add(d["turn_id"])
            self.delivered_receipts.update(self.turn_receipts)
        else:
            self.capture_incomplete = True
            self.needs_bootstrap = True
            self.outbox = [m for m in self.turn_messages if m["role"] == "tool"] + self.outbox

    def _on_model_identity_verified(self, rec, d):
        self.model_verified_turns.add(d["turn_id"])
        self.returned_models.append(d["model"])

    def _on_provider_event(self, rec, d):
        self.raw_referenced = max(self.raw_referenced, d["raw"]["seq"])
        kind = d["kind"]
        if not d["capture"]["complete"]:
            self.capture_incomplete = True
            self.unqualified_turns.add(d["turn_id"])
        if kind == "unknown":
            self.unknown_events += 1
        if kind == "truncation":
            self.capture_incomplete = True
            self.unqualified_turns.add(d["turn_id"])
        if kind == "session":
            p = d["payload"]
            if isinstance(p.get("session_id"), str):
                self.session_id = p["session_id"]

    def _on_handoff(self, rec, d):
        self.handoffs.append(d)
        if d.get("to_session"):
            self.session_id = d["to_session"]
        self.push_message(d["message"])
        self.needs_bootstrap = True

    def _on_request_received(self, rec, d):
        self.requests[d["request_id"]] = {
            "id": d["request_id"], "status": "received", "payload": d["payload"],
            "turn_id": d["turn_id"], "attempts": 0, "seq": rec["seq"],
        }
        self.order.append(("request", d["request_id"]))

    def _on_request_denied(self, rec, d):
        self.requests[d["request_id"]]["status"] = "denied"
        self.requests[d["request_id"]]["denial"] = d["reason"]
        self.push_message(d["message"])

    def _on_request_admitted(self, rec, d):
        r = self.requests[d["request_id"]]
        r.update(status="admitted", operation=d["operation"], arguments=d["arguments"],
                 bound=d["bound"], dispatch_id=d["dispatch_id"], cost=d["cost"])
        self.artifact_names.update(d.get("artifact_names", ()))

    def _on_dispatch_intent(self, rec, d):
        r = self.requests[d["request_id"]]
        r["status"] = "intent"
        r["attempts"] = d["attempt"]

    def _on_dispatch_error(self, rec, d):
        r = self.requests[d["request_id"]]
        if d["outcome"] == "not_started":
            r["status"] = "rejected"
            self.push_message(d["message"])
        else:
            r["status"] = "unknown"

    def _on_dispatch_result(self, rec, d):
        r = self.requests[d["request_id"]]
        r["status"] = "done"
        r["receipt"] = d["receipt"]
        r["receipt_sha256"] = d["receipt_sha256"]
        r["artifacts"] = d["artifacts"]
        r["message"] = d["message"]
        self.results.append({"request_id": d["request_id"], "dispatch_id": d["dispatch_id"],
                             "receipt_sha256": d["receipt_sha256"]})
        for a in d["artifacts"]:
            self.artifacts.append({**a, "dispatch_id": d["dispatch_id"], "request_id": d["request_id"]})
        self.push_message(d["message"])

    def _on_final_choice_received(self, rec, d):
        self.finals[d["item_id"]] = {"id": d["item_id"], "status": "received", "payload": d["payload"],
                                     "turn_id": d["turn_id"]}
        self.order.append(("final", d["item_id"]))

    def _on_final_choice_denied(self, rec, d):
        self.finals[d["item_id"]]["status"] = "denied"
        self.push_message(d["message"])

    def _on_final_patch_frozen(self, rec, d):
        self.finals[d["item_id"]]["status"] = "accepted"
        self.phase = "final_patch_frozen"
        self.frozen = d
        self.frozen_record_digest = rec["digest"]

    def _on_run_halted(self, rec, d):
        self.status = "halted"
        self.halt = d

    def _on_resume_started(self, rec, d):
        self.resume_count += 1
        self.status = "running"
        self.halt = None

    def _on_editor_terminal(self, rec, d):
        self.terminal = d
        self.status = "complete"

    def _on_broker_notice(self, rec, d):
        self.push_message(d["message"])


def chain_context(prev: str, messages: list[Mapping[str, str]]) -> str:
    return sha256_hex(prev.encode() + canonical_bytes(messages))
