"""Deterministic fake ports and a synthetic fixture generator (CPU, stdlib).

These are the only ports shipped. ``FakeEffectExecutor`` keeps its own durable
effect ledger so tests can prove, from outside the broker, that an effect
happened once and only once. All content is synthetic.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from .canon import canonical_bytes, create_only, sha256_hex, strict_loads
from .errors import AdapterCancelled, AdapterTimeout, ExecutorRejection, PortError
from .ports import (
    BoundedWriter, Dispatch, Executor, ModelAdapter, PortContext, PortSet, Receipt,
    ReconcileResult, TransportEvent, TurnInput,
)

HELDOUT_ID = "heldout-007"  # synthetic withheld id; only its digest appears in config

# --------------------------------------------------------------- adapter


class ScriptedAdapter(ModelAdapter):
    """Emits scripted raw events per turn index. Never calls any model."""

    def __init__(self, script: Mapping[str, Any], model: Mapping[str, Any], script_sha256: str):
        self._script = script
        self._model = dict(model)
        self._script_sha = script_sha256

    def identity(self) -> dict:
        return {**self._model, "client_version": "scripted-0", "script_sha256": self._script_sha,
                "capabilities": {"raw_events": True, "reasoning_summaries": True,
                                 "captures_hidden_reasoning": False, "tool_isolation": "none"}}

    def run_turn(self, turn: TurnInput) -> Iterator[bytes]:
        turns = self._script.get("turns", [])
        if turn.turn_index > len(turns):
            return
        spec = turns[turn.turn_index - 1]
        for ev in spec.get("events", []):
            if set(ev) == {"transport_channel", "raw_b64"}:
                import base64
                yield TransportEvent(base64.b64decode(ev["raw_b64"]), ev["transport_channel"])
                continue
            if ev == {"self_sigkill": True}:  # real SIGKILL of this process mid-turn
                import signal
                os.kill(os.getpid(), signal.SIGKILL)
            if "raw" in ev and len(ev) == 1:
                yield ev["raw"].encode("utf-8")
            elif "raw_b64" in ev and len(ev) == 1:
                import base64
                yield base64.b64decode(ev["raw_b64"])
            else:
                yield json.dumps(ev).encode("utf-8")
        fail = spec.get("fail")
        if fail == "cancelled":
            raise AdapterCancelled("scripted cancellation")
        if fail == "timeout":
            raise AdapterTimeout("scripted timeout")
        if fail == "provider_error":
            raise RuntimeError("scripted provider failure")
        if spec.get("model_identity", True):
            # This is synthetic provider evidence, never used by the real adapter.
            yield json.dumps({"type": "session_started", "model": self._model["model"]}).encode("utf-8")

    def parse_event(self, raw: bytes) -> dict:
        ev = strict_loads(raw)
        if not isinstance(ev, dict):
            return {"kind": "unknown"}
        t = ev.get("type")
        rest = {k: v for k, v in ev.items() if k != "type"}
        table = {
            "assistant_message": "message", "reasoning_summary": "reasoning_summary",
            "tool_request": "request", "final_choice": "final_choice", "session_started": "session",
            "usage": "usage", "turn_finished": "finish", "handoff": "handoff", "truncation": "truncation",
        }
        kind = table.get(t)
        return {"kind": kind, **rest} if kind else {"kind": "unknown"}


# -------------------------------------------------------------- executor


class FakeEffectExecutor(Executor):
    """Synthetic executor with an external, fsynced effect ledger.

    Supported operations: inspect_weights, query_development, propose_patch.
    ``analyze`` is deliberately unsupported to exercise capability rejection.
    """

    OPS = frozenset({"inspect_weights", "query_development", "propose_patch"})

    def __init__(self, effects_dir: Path, pause_at: str | None = None, pause_op: str | None = None,
                 fault: str | None = None):
        self.dir = Path(effects_dir)
        (self.dir / "dispatches").mkdir(parents=True, exist_ok=True)
        self.pause_at, self.pause_op, self.fault = pause_at, pause_op, fault

    def identity(self) -> dict:
        return {"name": "fake-effect-executor", "version": 1, "operations": sorted(self.OPS)}

    def supported_operations(self) -> frozenset[str]:
        return self.OPS

    # ---- validation (side-effect free)
    def validate(self, operation: str, arguments: Mapping[str, Any], bound: Mapping[str, Any]) -> None:
        if operation == "propose_patch":
            for key in ("patch", "justification"):
                if not str(arguments.get(key, "")).strip():
                    raise ExecutorRejection(f"{key} must be non-empty")

    # ---- helpers
    def _log_path(self) -> Path:
        return self.dir / "effects.log"

    def _append_effect(self, dispatch: Dispatch) -> None:
        line = json.dumps({"dispatch_id": dispatch.dispatch_id, "op": dispatch.operation}) + "\n"
        fd = os.open(str(self._log_path()), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def _pause(self, stage: str, op: str) -> None:
        if self.pause_at == stage and self.pause_op in (None, op):
            create_only(self.dir / f"paused-{stage}", b"1")
            time.sleep(300)

    # ---- execution
    def execute(self, dispatch: Dispatch, writer: BoundedWriter) -> Receipt:
        if self.fault == "reject_execute":
            raise ExecutorRejection("fault: executor refuses before any effect")
        started = self.dir / "dispatches" / f"{dispatch.dispatch_id}.started"
        self._pause("before_started", dispatch.operation)
        create_only(started, b"1")  # fsynced before the effect so reconcile can tell "maybe started"
        self._pause("after_started", dispatch.operation)
        if self.fault == "escape_write":
            writer.write("../../escape.txt", b"x")
        self._append_effect(dispatch)
        receipt = self._compute(dispatch, writer)
        self._pause("before_done", dispatch.operation)
        if self.fault == "raise_after_effect":
            raise RuntimeError("fault: executor died after the effect")
        done = {"status": receipt.status, "output": receipt.output, "detail": dict(receipt.detail),
                "usage": dict(receipt.usage)}
        create_only(self.dir / "dispatches" / f"{dispatch.dispatch_id}.done.json", canonical_bytes(done))
        self._pause("after_effect", dispatch.operation)
        if self.fault == "bad_usage":
            return Receipt(receipt.status, receipt.output, receipt.detail, {"gpu_units": float("nan")})
        return receipt

    def _compute(self, dispatch: Dispatch, writer: BoundedWriter) -> Receipt:
        op, args, bound = dispatch.operation, dispatch.arguments, dispatch.bound
        if op == "inspect_weights":
            values = json.loads(Path(bound["tensor"]["path"]).read_text())["values"]
            out = {"tensor": bound["tensor"]["id"], "n": len(values), "sum": sum(values),
                   "min": min(values), "max": max(values)}
            return Receipt("ok", json.dumps(out, sort_keys=True), out, {"cpu_units": 1})
        if op == "query_development":
            rows = []
            for item in bound["items"]:
                score = int(sha256_hex(item["sha256"].encode())[:6], 16) % 100 / 100
                rows.append({"item": item["id"], "score": score})
            return Receipt("ok", json.dumps(rows, sort_keys=True), {"rows": len(rows)},
                           {"gpu_units": 0.5 * len(rows), "examples": len(rows)})
        if op == "propose_patch":
            text = args["patch"]
            reason = _patch_problem(text, bound["tensor"]["id"])
            writer.write(f"{args['name']}.patch.json", text.encode("utf-8"))
            status = "invalid" if reason else "ok"
            out = f"patch {args['name']} {status}" + (f": {reason}" if reason else "")
            return Receipt(status, out, {"reason": reason, "valid": reason is None}, {"cpu_units": 1})
        raise PortError(f"unsupported operation {op}")  # pragma: no cover

    # ---- reconciliation (never causes an effect)
    def reconcile(self, dispatch: Dispatch) -> ReconcileResult:
        done = self.dir / "dispatches" / f"{dispatch.dispatch_id}.done.json"
        started = self.dir / "dispatches" / f"{dispatch.dispatch_id}.started"
        if done.exists():
            d = json.loads(done.read_text())
            return ReconcileResult("complete", Receipt(d["status"], d["output"], d["detail"], d["usage"]))
        if started.exists():
            return ReconcileResult("unknown", note="effect may have started; executor has no completion record")
        return ReconcileResult(
            "unstarted",
            authority={"dispatch_id": dispatch.dispatch_id,
                       "basis": "executor ledger holds no started marker; marker is fsynced before any effect"},
        )


def _patch_problem(text: str, tensor_id: str) -> str | None:
    try:
        doc = strict_loads(text)
    except ValueError:
        return "patch is not strict JSON"
    if not isinstance(doc, dict) or doc.get("tensor") != tensor_id:
        return "patch must target the bound tensor"
    deltas = doc.get("deltas")
    if not isinstance(deltas, list) or not deltas or not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in deltas
    ):
        return "deltas must be a non-empty list of finite numbers"
    if max(abs(x) for x in deltas) > 0.5:
        return "realized delta exceeds the 0.5 limit"
    return None


# ----------------------------------------------------------------- build


def build_ports(ctx: PortContext) -> PortSet:
    """Factory for ``--ports synaptic_tuner.experimental.weight_editor.fakes:build_ports``.

    ports-args: script (path), effects_dir (path); optional model/provider override
    (simulates adapter drift), pause_at/pause_op (crash injection), fault.
    """
    args = ctx.args
    try:
        script_path, effects = Path(args["script"]), Path(args["effects_dir"])
    except KeyError as exc:
        raise PortError(f"fake ports need --ports-arg {exc.args[0]}=...") from exc
    blob = script_path.read_bytes()
    model = dict(ctx.model)
    if "model" in args:
        model["model"] = args["model"]
    if "provider" in args:
        model["provider"] = args["provider"]
    return PortSet(
        ScriptedAdapter(strict_loads(blob), model, sha256_hex(blob)),
        FakeEffectExecutor(effects, args.get("pause_at"), args.get("pause_op"), args.get("fault")),
    )


# --------------------------------------------------------------- fixture

TENSOR_FILES = {"w.layer0": [0.1, -0.2, 0.3], "w.layer1": [0.5, 0.4, -0.3]}
DEV_ITEMS = [f"dev-{i:03d}" for i in range(1, 5)]


def _j(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()


def write_fixture(dest: Path, *, script: Mapping[str, Any] | None = None, limits: Mapping[str, float] | None = None,
                  model: Mapping[str, Any] | None = None, extra_ops: Mapping[str, Any] | None = None) -> dict:
    """Create a synthetic capsule + config + script under ``dest`` (must not exist)."""
    dest = Path(dest)
    dest.mkdir(parents=True)
    cap = dest / "capsule"
    (cap / "tensors").mkdir(parents=True)
    (cap / "dev").mkdir()
    objects = []

    def add(oid: str, kind: str, rel: str, data: bytes) -> None:
        (cap / rel).write_bytes(data)
        objects.append({"id": oid, "kind": kind, "path": rel, "sha256": sha256_hex(data), "split": "development"})

    for tid, vals in TENSOR_FILES.items():
        add(tid, "tensor", f"tensors/{tid}.json", _j({"values": vals}))
    for did in DEV_ITEMS:
        add(did, "dev_item", f"dev/{did}.json", _j({"prompt": f"synthetic development prompt {did}"}))
    add("instr-main", "instructions", "instructions.txt", b"Synthetic editor task: propose a small safe edit.\n")
    (cap / "manifest.json").write_bytes(_j({"schema": "weight_editor_capsule/v1", "capsule_id": "synthetic-capsule",
                                            "objects": objects}))
    operations = {
        "inspect_weights": {"description": "bounded numeric read of an admitted tensor",
                            "arguments": {"tensor": {"type": "ref", "kind": "tensor"}}, "required": ["tensor"],
                            "cost": {"cpu_units": 1}},
        "query_development": {"description": "run admitted development items",
                              "arguments": {"items": {"type": "ref_list", "kind": "dev_item", "max_items": 4}},
                              "required": ["items"],
                              "cost": {"gpu_units": {"len_of": "items", "per_item": 0.5},
                                       "examples": {"len_of": "items", "per_item": 1}}},
        "propose_patch": {"description": "hash-bound patch plus visible justification", "selectable": True,
                          "arguments": {"name": {"type": "artifact_name"},
                                        "tensor": {"type": "ref", "kind": "tensor"},
                                        "patch": {"type": "text", "max_length": 2000},
                                        "justification": {"type": "text", "max_length": 2000}},
                          "required": ["name", "tensor", "patch", "justification"], "cost": {"cpu_units": 1}},
        **(extra_ops or {}),
    }
    mdl = dict(model or {"provider": "fake", "model": "fake-editor-1", "settings": {"effort": "fixture"}})
    config = {
        "schema": "weight_editor_config/v1", "model": mdl,
        "capsule": {"manifest": "capsule/manifest.json"},
        "instructions": [{"role": "system", "object": "instr-main"},
                         {"role": "user", "text": "Begin discovery."}],
        "operations": operations, "output_root": "runs", "read_only_roots": [],
        "final_choice": {"allow_no_edit": True},
        "withheld_id_digests": [sha256_hex(HELDOUT_ID.encode())],
    }
    if limits is not None:
        config["resources"] = {"limits": dict(limits)}
    (dest / "config.json").write_bytes(_j(config))
    (dest / "script.json").write_bytes(_j(script if script is not None else smoke_script()))
    (dest / "effects").mkdir()
    return {"root": dest, "config": dest / "config.json", "script": dest / "script.json",
            "effects": dest / "effects", "capsule_manifest": cap / "manifest.json", "capsule": cap,
            "runs": dest / "runs"}


def smoke_script() -> dict:
    ok_patch = json.dumps({"tensor": "w.layer0", "deltas": [0.1, -0.1, 0.05]})
    bad_patch = json.dumps({"tensor": "w.layer0", "deltas": [0.1, 9.0, 0.05]})
    return {"turns": [
        {"events": [
            {"type": "session_started", "session_id": "sess-1", "model": "fake-editor-1"},
            {"type": "assistant_message", "text": "I will inspect then query."},
            {"type": "reasoning_summary", "text": "synthetic visible summary"},
            {"type": "tool_request", "operation": "inspect_weights", "arguments": {"tensor": "w.layer0"}},
            {"type": "tool_request", "operation": "query_development",
             "arguments": {"items": ["dev-001", "dev-002"]}},
            {"type": "tool_request", "operation": "inspect_weights", "arguments": {"tensor": "../../etc/passwd"},
             "command": "cat /etc/passwd"},
            {"type": "turn_finished", "finish_reason": "tool_use"},
        ]},
        {"events": [
            {"type": "tool_request", "operation": "propose_patch",
             "arguments": {"name": "edit-a", "tensor": "w.layer0", "patch": ok_patch,
                           "justification": "small bounded change"}},
            {"type": "tool_request", "operation": "propose_patch",
             "arguments": {"name": "edit-b", "tensor": "w.layer0", "patch": bad_patch,
                           "justification": "oversized change, expected to be invalid"}},
            {"type": "handoff", "new_session_id": "sess-2",
             "capsule": {"summary": "synthetic compaction summary", "continuation_offset": 7}},
        ]},
        {"events": [
            {"type": "final_choice", "choice": "patch", "proposal": "req-0004",
             "rationale": "edit-a is the valid bounded proposal"},
        ]},
    ]}
