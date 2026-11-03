"""Host-owned experiment broker and recorder.

The broker owns admission, reservation, write-ahead dispatch, reconciliation,
the journals and the terminal boundary. The model adapter and executor are
trusted ports that never receive authority from a model request.
"""

from __future__ import annotations

import base64
import fcntl
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from . import __version__
from .admission import PUBLIC, Denied, admit
from .canon import (
    canonical_bytes, check_json, create_only, digest_of, load_strict_file, sha256_file, sha256_hex,
)
from .config import RUN_ID_RE, EditorConfig
from .errors import (
    AdapterCancelled, AdapterFailure, AdapterTimeout, EditorError, ExecutorRejection, IdentityDrift,
    JournalCorrupt, LockHeld, PartialTail, PhaseError, PortError, RunExists, RunNotFound, UnsupportedCapability,
    WriteBoundaryViolation,
)
from .journal import HashChainLog, genesis_prev, quarantine_partial_tail
from .ledger import clean_amount
from .ports import (
    EVENT_KINDS, BoundedWriter, Dispatch, PortSet, Receipt, ReconcileResult, TransportEvent,
    TurnInput, describe_port,
)
from .state import PHASES, RunState, chain_context

MANIFEST_SCHEMA = "weight_editor_run/v1"
PROTOCOL = (
    "You are the editor. Only the operations and objects in this catalog exist. A request is an object "
    "with 'operation' and 'arguments' only; paths, URLs, commands, imports, policy and budget fields "
    "carry no authority and are denied. Give a visible justification with every proposal. "
    "Finish with exactly one final_choice (a selectable proposal or no_edit). After a final choice "
    "no further operations are accepted."
)


class _Halt(Exception):
    def __init__(self, code: str, detail: str, **extra):
        super().__init__(detail)
        self.code, self.detail, self.extra = code, detail, extra


# ------------------------------------------------------------ identity


def validate_ports(config: EditorConfig, ports: PortSet) -> dict:
    """Explicit port validators. No fallback model, no unsupported capability."""
    ident = ports.adapter.identity()
    if not isinstance(ident, dict):
        raise PortError("adapter identity must be an object")
    req = {k: ident.get(k) for k in ("provider", "model", "settings")}
    if req != config.model.requested():
        raise PortError(
            "adapter requested model identity does not equal the configured model; no fallback is permitted",
            code="model_mismatch",
            configured=config.model.requested(),
            adapter=req,
        )
    caps = ident.get("capabilities", {})
    if not isinstance(caps, dict) or caps.get("raw_events") is not True:
        raise PortError("adapter must declare raw_events=true (complete emitted-traffic retention)", code="adapter_unqualified")
    if caps.get("captures_hidden_reasoning"):
        raise PortError("hidden reasoning capture is not a supported claim", code="adapter_unqualified")
    supported = ports.executor.supported_operations()
    missing = sorted(set(config.operations) - set(supported))
    if missing:
        raise UnsupportedCapability(
            f"executor does not support configured operations {missing}", unsupported=missing
        )
    return {
        "adapter": describe_port(ports.adapter),
        "executor": describe_port(ports.executor),
        "declared_isolation": caps.get("tool_isolation", "unspecified"),
        "isolation_note": "declared by the port; not verified or enforced by the broker",
    }


def compute_binding(config: EditorConfig, ports: PortSet, supplied: Mapping[str, Any]) -> dict:
    check_json(dict(supplied))
    return {
        "broker_sources": {str(p.relative_to(Path(__file__).parent)): sha256_file(p)
                           for p in sorted(Path(__file__).parent.rglob("*.py"))},
        "config_sha256": config.sha256,
        "capsule_manifest_sha256": config.capsule.manifest_sha256,
        "capsule_objects": {oid: o.sha256 for oid, o in sorted(config.capsule.objects.items())},
        "model": config.model.requested(),
        "adapter": describe_port(ports.adapter),
        "executor": describe_port(ports.executor),
        "supplied": dict(supplied),
    }


def _diff(old: Mapping, new: Mapping) -> list[str]:
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


# ---------------------------------------------------------------- run


class EditorRun:
    def __init__(self, config, ports, run_dir, manifest, lock_fd, journal, raw, clock):
        self.config: EditorConfig = config
        self.ports: PortSet = ports
        self.run_dir: Path = run_dir
        self.manifest: dict = manifest
        self.run_id: str = manifest["run_id"]
        self._lock_fd = lock_fd
        self.journal: HashChainLog = journal
        self.raw: HashChainLog = raw
        self.clock = clock
        self._drift: dict | None = None
        self.state = RunState(config.limits)
        for rec in journal.records:
            self.state.apply(rec)

    # ------------------------------------------------------- lifecycle
    @staticmethod
    def _run_dir_for(config: EditorConfig, run_id: str) -> Path:
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id) or ".." in run_id:
            raise EditorError("run id must match [a-z0-9][a-z0-9._-]{0,63}", code="bad_run_id")
        return config.output_root / run_id

    @staticmethod
    def _lock(path: Path) -> int:
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise LockHeld("another writer holds this run's lock") from exc
        return fd

    @classmethod
    def create(cls, config: EditorConfig, ports: PortSet, run_id: str,
               supplied: Mapping[str, Any] | None = None, clock: Callable[[], float] = time.time):
        validate_ports(config, ports)
        run_dir = cls._run_dir_for(config, run_id)
        config.output_root.mkdir(parents=True, exist_ok=True)
        if run_dir.parent.resolve() != config.output_root:
            raise EditorError("run directory escapes output root", code="output_root_escape")
        try:
            os.mkdir(run_dir, 0o700)
        except FileExistsError as exc:
            raise RunExists(f"run {run_id} already exists; runs are never overwritten") from exc
        lock_fd = cls._lock(run_dir / "lock")
        try:
            for sub in ("capsule", "provider", "journal", "requests", "artifacts"):
                (run_dir / sub).mkdir()
            binding = compute_binding(config, ports, supplied or {})
            bad = config.capsule.verify_objects()
            if bad:
                raise EditorError(f"capsule objects changed during registration: {bad}", code="capsule_drift")
            create_only(run_dir / "config.json", config.raw_bytes)
            create_only(run_dir / "capsule" / "manifest.json", config.capsule.manifest_bytes)
            manifest = {
                "schema": MANIFEST_SCHEMA, "run_id": run_id, "created_at": round(clock(), 6),
                "writer": f"weight_editor {__version__}", "binding": binding,
                "binding_digest": digest_of(binding),
                "state_machine": list(PHASES),
                "notes": ["hidden reasoning is not captured or claimed",
                          "adapter isolation is declared by the port, not enforced by the broker"],
            }
            create_only(run_dir / "manifest.json", canonical_bytes(manifest) + b"\n")
            journal = HashChainLog(run_dir / "journal" / "events.jsonl", run_id,
                                   genesis_prev(manifest["binding_digest"], "journal"), clock)
            raw = HashChainLog(run_dir / "provider" / "events.jsonl", run_id,
                               genesis_prev(manifest["binding_digest"], "provider"), clock)
        except BaseException:
            os.close(lock_fd)
            raise
        run = cls(config, ports, run_dir, manifest, lock_fd, journal, raw, clock)
        run._emit("run_registered", {
            "binding_digest": manifest["binding_digest"], "config_sha256": config.sha256,
            "capsule_manifest_sha256": config.capsule.manifest_sha256, "model_requested": config.model.requested(),
            "adapter": binding["adapter"], "executor": binding["executor"], "supplied": binding["supplied"],
        })
        run._emit("state_transition", {"from": "registered", "to": "discovery", "reason": "registered"})
        return run

    @classmethod
    def open(cls, config: EditorConfig, ports: PortSet, run_id: str,
             supplied: Mapping[str, Any] | None = None, recover_tail: bool = False,
             clock: Callable[[], float] = time.time):
        run_dir = cls._run_dir_for(config, run_id)
        if not (run_dir / "manifest.json").is_file():
            raise RunNotFound(f"no registered run {run_id} under output root")
        lock_fd = cls._lock(run_dir / "lock")
        try:
            manifest = read_manifest(run_dir)
            journal, raw, recovered = _open_logs(run_dir, manifest, recover_tail, clock)
        except BaseException:
            os.close(lock_fd)
            raise
        run = cls(config, ports, run_dir, manifest, lock_fd, journal, raw, clock)
        try:
            for r in recovered:
                run._emit("tail_recovery", r)
            cross_check(run_dir, journal.records, raw.records)
            try:
                validate_ports(config, ports)
                current = compute_binding(config, ports, supplied or {})
            except EditorError as exc:
                run._refuse("port_validation", {"code": exc.code, "message": str(exc)})
                raise
            drift = _diff(manifest["binding"], current)
            if drift:
                run._refuse("identity_drift", {"components": drift})
                raise IdentityDrift(
                    f"run binding changed ({', '.join(drift)}); the run cannot resume under changed bytes",
                    components=drift,
                )
            if run.state.phase in ("independent_evaluation", "closed"):
                run._refuse("phase_closed", {"phase": run.state.phase})
                raise PhaseError(f"run is in {run.state.phase}; the editor has no transition back to discovery")
            if run.state.phase == "registered":  # crashed between registration records
                run._emit("state_transition", {"from": "registered", "to": "discovery", "reason": "resume"})
            if run.state.status != "complete":
                run._emit("resume_started", {"prior_status": run.state.status, "prior_halt": run.state.halt,
                                             "open_turn": run.state.open_turn})
        except BaseException:
            run.close()
            raise
        return run

    def close(self) -> None:
        for log in (self.journal, self.raw):
            log.close()
        for port in (self.ports.adapter, self.ports.executor):
            try:
                port.close()
            except Exception:
                pass
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    # ---------------------------------------------------------- emit
    def _emit(self, rtype: str, data: dict, linked: dict | None = None, ledger: list | None = None) -> dict:
        if ledger:
            for e in ledger:
                clean_amount(e["amount"], "ledger amount")
            data = {**data, "ledger": ledger}
        rec = self.journal.append(rtype, data, linked)
        self.state.apply(rec)
        return rec

    def _refuse(self, reason: str, detail: dict) -> None:
        self._emit("resume_refused", {"reason": reason, **detail})

    def _halt(self, code: str, detail: str, **extra) -> None:
        self._emit("run_halted", {"code": code, "detail": detail, "incomplete": True, **extra})

    # ---------------------------------------------------------- drive
    def run(self) -> dict:
        st = self.state
        if st.phase not in ("discovery", "final_patch_frozen") or st.status == "complete":
            return self.summary()
        try:
            if st.open_turn is not None:
                self._recover_open_turn()
            if st.phase == "final_patch_frozen":
                self._finalize()
                return self.summary()
            while st.phase == "discovery" and st.status == "running":
                bad = self.config.capsule.verify_objects()
                if bad:
                    raise _Halt("capsule_drift", f"capsule objects changed: {bad}", objects=bad)
                self._process_pending()
                if st.phase != "discovery" or st.status != "running":
                    break
                self._run_turn()
                self._process_pending()
            if st.phase == "final_patch_frozen" and st.status == "running":
                self._finalize()
        except _Halt as h:
            self._halt(h.code, h.detail, **h.extra)
        return self.summary()

    # ----------------------------------------------------------- turn
    def _initial_messages(self) -> list[dict]:
        msgs = []
        for ins in self.config.instructions:
            if "text" in ins:
                text = ins["text"]
            else:
                text = self.config.capsule.objects[ins["object"]].abs_path.read_text(encoding="utf-8")
            msgs.append({"role": ins["role"], "content": text})
        catalog = {
            "protocol": PROTOCOL,
            "capsule": self.config.capsule.listing(),
            "operations": {
                n: {"description": o.description, "selectable": o.selectable,
                    "arguments": {a.name: {"type": a.type, "required": a.required, **a.params}
                                  for a in o.args.values()}}
                for n, o in sorted(self.config.operations.items())
            },
        }
        msgs.append({"role": "system", "content": "BROKER CATALOG\n" + canonical_bytes(catalog).decode()})
        return msgs

    def _run_turn(self) -> None:
        st = self.state
        req_limit = st.ledger.limits.get("requests")
        if st.ledger.exhausted("model_turns") or (req_limit is not None and st.ledger.charged("requests") >= req_limit):
            raise _Halt("budget_exhausted", "model_turns/requests limit reached; configured limits are not raised on resume")
        idx = st.turn_count + 1
        turn_id = f"turn-{idx:04d}"
        bootstrap = idx == 1 or st.session_id is None or st.needs_bootstrap
        messages = (self._initial_messages() if bootstrap else []) + list(st.outbox)
        ctx = chain_context(st.context_digest, messages)
        resumed = not st.last_turn_clean
        self._emit("turn_start", {
            "turn_id": turn_id, "turn_index": idx, "session_id": st.session_id, "messages": messages,
            "context_digest": ctx, "resumed": resumed,
        }, {"turn_id": turn_id}, ledger=[{"op": "commit", "ref": turn_id, "resource": "model_turns", "amount": 1}])
        ti = TurnInput(self.run_id, turn_id, idx, st.session_id,
                       tuple(dict(m) for m in messages), ctx, resumed)
        try:
            for raw in self.ports.adapter.run_turn(ti):
                self._ingest_raw(turn_id, raw)
        except Exception as exc:  # provider failure, cancellation, timeout, adapter bug
            cls = ("cancelled" if isinstance(exc, AdapterCancelled) else
                   "timeout" if isinstance(exc, (AdapterTimeout, TimeoutError)) else "provider_error")
            self._emit("adapter_error", {"turn_id": turn_id, "classification": cls,
                                         "category": type(exc).__name__,
                                         "message_sha256": sha256_hex(str(exc).encode())}, {"turn_id": turn_id})
            self._discard_turn_items(turn_id, "failed_turn")
            self._emit("turn_end", {"turn_id": turn_id, "status": "failed", "reason": cls}, {"turn_id": turn_id})
            raise _Halt("adapter_failure", f"adapter {cls}: {type(exc).__name__}; the turn is not retried implicitly")
        if self._drift is not None:
            drift, self._drift = self._drift, None
            self._discard_turn_items(turn_id, "model_drift")
            self._emit("turn_end", {"turn_id": turn_id, "status": "failed", "reason": "model_drift"}, {"turn_id": turn_id})
            raise _Halt("model_drift", f"provider returned model {drift['returned']!r}, not the configured model "
                        "or an allowed alias", **drift)
        if turn_id in st.unqualified_turns:
            self._discard_turn_items(turn_id, "capture_incomplete")
            self._emit("turn_end", {"turn_id": turn_id, "status": "failed", "reason": "capture_incomplete"},
                       {"turn_id": turn_id})
            raise _Halt("capture_incomplete", "unknown or truncated stream: turn items discarded, raw bytes retained")
        if turn_id not in st.model_verified_turns:
            self._discard_turn_items(turn_id, "model_identity_unverified")
            self._emit("turn_end", {"turn_id": turn_id, "status": "failed",
                                    "reason": "model_identity_unverified"}, {"turn_id": turn_id})
            raise _Halt("model_identity_unverified", "no returned model identity was observed in this turn")
        self._emit("turn_end", {"turn_id": turn_id, "status": "complete"}, {"turn_id": turn_id})
        if st.pending() is None and st.phase == "discovery":
            raise _Halt("model_ended_without_choice",
                        "the turn produced no request or final choice; final text is not success")

    def _ingest_raw(self, turn_id: str, raw: Any) -> None:
        channel = "provider"
        if isinstance(raw, TransportEvent):
            channel, raw = raw.channel, raw.data
            if channel not in ("provider", "client_input", "client_stderr"):
                raise AdapterFailure("adapter yielded an unsupported transport channel")
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if not isinstance(raw, (bytes, bytearray)):
            raise AdapterFailure("adapter yielded a non-bytes event")
        raw = bytes(raw)
        rec = self.raw.append("provider_raw", {
            "turn_id": turn_id, "raw_b64": base64.b64encode(raw).decode(), "raw_sha256": sha256_hex(raw),
            "size": len(raw),
            "channel": channel,
        })
        self._normalize(turn_id, rec, recovered=False)

    def _normalize(self, turn_id: str, rawrec: dict, recovered: bool) -> None:
        raw = base64.b64decode(rawrec["data"]["raw_b64"])
        ref = {"seq": rawrec["seq"], "sha256": rawrec["data"]["raw_sha256"], "digest": rawrec["digest"]}
        error = None
        try:
            channel = rawrec["data"].get("channel", "provider")
            ev = (self.ports.adapter.parse_event(raw) if channel == "provider" else
                  {"kind": "message", "channel": channel, "content": raw.decode("utf-8", errors="replace")})
            check_json(ev)
            if not isinstance(ev, dict) or ev.get("kind") not in EVENT_KINDS:
                raise ValueError("unrecognized event kind")
        except Exception as exc:
            error = type(exc).__name__
            ev = {"kind": "unknown"}
        kind = ev["kind"]
        payload = {k: v for k, v in ev.items() if k != "kind"}
        complete = kind != "unknown"
        self._emit("provider_event", {
            "kind": kind, "payload": payload, "raw": ref, "turn_id": turn_id,
            "capture": {"complete": complete, **({"note": "unsupported/unparsed event retained raw",
                                                  "parse_error": error} if not complete else {})},
            "recovered_from_raw": recovered,
        }, {"turn_id": turn_id, "raw_seq": ref["seq"]})
        st = self.state
        self._check_returned_model(turn_id, payload)
        if kind in ("handoff", "truncation"):
            cap = payload.get("capsule", payload)
            to_session = payload.get("new_session_id") or payload.get("to_session")
            self._emit("handoff", {
                "kind": kind, "from_session": st.session_id, "to_session": to_session, "capsule": cap,
                "capsule_sha256": digest_of(cap), "raw": ref, "turn_id": turn_id,
                "message": {"role": "system", "content": f"{kind.upper()} CAPSULE\n" + canonical_bytes(cap).decode()},
            }, {"turn_id": turn_id, "raw_seq": ref["seq"]})
        elif kind == "request":
            rid = st.next_request_id()
            self._emit("request_received", {
                "request_id": rid, "turn_id": turn_id, "payload": payload, "raw": ref,
            }, {"request_id": rid, "turn_id": turn_id, "raw_seq": ref["seq"]},
                ledger=[{"op": "commit", "ref": rid, "resource": "requests", "amount": 1}])
        elif kind == "final_choice":
            fid = st.next_final_id()
            self._emit("final_choice_received", {
                "item_id": fid, "turn_id": turn_id, "payload": payload, "raw": ref,
            }, {"item_id": fid, "turn_id": turn_id, "raw_seq": ref["seq"]})

    def _check_returned_model(self, turn_id: str, payload: dict) -> None:
        returned = payload.get("model")
        if returned is None:
            return
        ok = {self.config.model.model, *self.config.model.allowed_returned}
        if returned not in ok and self._drift is None:
            # Keep consuming and retaining the turn's traffic; discard its requests afterwards.
            self._drift = {"requested": self.config.model.model, "returned": returned}
        elif returned in ok:
            self._emit("model_identity_verified", {"turn_id": turn_id, "model": returned}, {"turn_id": turn_id})

    def _discard_turn_items(self, turn_id: str, reason: str) -> None:
        for kind, iid in list(self.state.order):
            if kind == "request" and self.state.requests[iid]["turn_id"] == turn_id and \
                    self.state.requests[iid]["status"] == "received":
                self._deny(iid, Denied("phase_closed", reason), override_reason=reason)
            elif kind == "final" and self.state.finals[iid]["turn_id"] == turn_id and \
                    self.state.finals[iid]["status"] == "received":
                self._deny_final(iid, reason, "turn discarded")

    def _recover_open_turn(self) -> None:
        """A crash left a turn open: normalize raw events never referenced, then close it."""
        st = self.state
        turn_id = st.open_turn
        try:
            for rawrec in list(self.raw.records):
                if rawrec["seq"] > st.raw_referenced:
                    self._normalize(rawrec["data"]["turn_id"], rawrec, recovered=True)
        finally:
            self._discard_turn_items(turn_id, "interrupted_turn")
            self._emit("turn_end", {"turn_id": turn_id, "status": "interrupted", "reason": "process_crash_or_kill"},
                       {"turn_id": turn_id})

    # ------------------------------------------------------- requests
    def _process_pending(self) -> None:
        st = self.state
        while True:
            item = st.pending()
            if item is None or st.status != "running":
                return
            kind, obj = item
            if kind == "request":
                self._process_request(obj["id"])
            else:
                self._process_final(obj["id"])

    def _tool_message(self, request_id: str, status: str, **body) -> dict:
        return {"role": "tool", "content": canonical_bytes({"request_id": request_id, "status": status, **body}).decode()}

    def _deny(self, rid: str, denial: Denied, override_reason: str | None = None) -> None:
        reason = override_reason or denial.reason
        public = PUBLIC.get(denial.reason, "request denied")
        public_reason = "unknown_object" if reason in ("withheld_id", "unknown_object") else reason
        self._emit("request_denied", {
            "request_id": rid, "reason": reason, "detail": denial.detail,
            "message": self._tool_message(rid, "denied", reason=public_reason,
                                          detail=public),
        }, {"request_id": rid})

    def _process_request(self, rid: str) -> None:
        st = self.state
        req = st.requests[rid]
        if req["status"] == "received":
            req_limit = st.ledger.limits.get("requests")
            if req_limit is not None and int(rid.split("-")[1]) > req_limit:
                return self._deny(rid, Denied("budget_exceeded", f"requests limit {req_limit} exceeded"))
            adm = admit(req["payload"], self.config, st.ledger, st.artifact_names, st.phase)
            if isinstance(adm, Denied):
                return self._deny(rid, adm)
            bad = self.config.capsule.verify_objects(
                {b["id"] for v in adm.bound.values() for b in (v if isinstance(v, list) else [v])})
            if bad:
                raise _Halt("capsule_drift", f"capsule objects changed before dispatch: {bad}", objects=bad)
            try:
                self.ports.executor.validate(adm.operation, adm.arguments, adm.bound)
            except ExecutorRejection as exc:
                return self._deny(rid, Denied("executor_validation", str(exc)))
            dispatch_id = "dsp-" + digest_of({"run": self.run_id, "req": rid, "op": adm.operation,
                                              "args": adm.arguments})[:24]
            cost = dict(adm.cost)
            self._emit("request_admitted", {
                "request_id": rid, "operation": adm.operation, "arguments": dict(adm.arguments),
                "bound": _public_bound(adm.bound), "dispatch_id": dispatch_id, "cost": cost,
                "artifact_names": [v for n, v in adm.arguments.items()
                                   if self.config.operations[adm.operation].args[n].type == "artifact_name"],
            }, {"request_id": rid, "dispatch_id": dispatch_id},
                ledger=[{"op": "reserve", "ref": dispatch_id, "resource": r, "amount": a}
                        for r, a in cost.items() if a > 0])
            req = st.requests[rid]
        if req["status"] == "admitted":
            self._dispatch(req, attempt=1, basis="initial")
        elif req["status"] in ("intent", "unknown"):
            self._reconcile(req)

    def _make_dispatch(self, req: dict, attempt: int) -> Dispatch:
        bound = self._resolve_bound(req["operation"], req["arguments"])
        return Dispatch(req["dispatch_id"], self.run_id, req["id"], req["operation"],
                        dict(req["arguments"]), bound, attempt)

    def _resolve_bound(self, op: str, args: Mapping[str, Any]) -> dict:
        spec = self.config.operations[op]
        out: dict[str, Any] = {}
        for name, value in args.items():
            if spec.args[name].type in ("ref", "ref_list"):
                ids = [value] if spec.args[name].type == "ref" else value
                objs = [self.config.capsule.objects[i] for i in ids]
                rs = [{"id": o.id, "kind": o.kind, "sha256": o.sha256, "path": str(o.abs_path)} for o in objs]
                out[name] = rs[0] if spec.args[name].type == "ref" else rs
        return out

    def _artifact_dir(self, dispatch_id: str) -> Path:
        return self.run_dir / "artifacts" / dispatch_id

    def _dispatch(self, req: dict, attempt: int, basis: str) -> None:
        if self.state.ledger.exhausted("dispatch_attempts"):
            raise _Halt("budget_exhausted", "dispatch_attempts limit reached before executor effect")
        bad = self.config.capsule.verify_objects()
        if bad:
            raise _Halt("capsule_drift", f"capsule objects changed before dispatch: {bad}", objects=bad)
        dispatch = self._make_dispatch(req, attempt)
        # Write-ahead: the intent is durable before the executor can cause any effect.
        self._emit("dispatch_intent", {
            "dispatch_id": dispatch.dispatch_id, "request_id": req["id"], "attempt": attempt, "basis": basis,
            "operation": req["operation"],
        }, {"request_id": req["id"], "dispatch_id": dispatch.dispatch_id},
            ledger=[{"op": "commit", "ref": dispatch.dispatch_id, "resource": "dispatch_attempts", "amount": 1}])
        writer = BoundedWriter(self._artifact_dir(dispatch.dispatch_id), self.config.read_only_roots)
        try:
            receipt = self.ports.executor.execute(dispatch, writer)
        except ExecutorRejection as exc:
            self._reject_dispatch(req, exc)
            return
        except BaseException as exc:  # ambiguous: the effect may or may not have happened
            self._emit("dispatch_error", {
                "dispatch_id": dispatch.dispatch_id, "request_id": req["id"], "outcome": "unknown",
                "category": type(exc).__name__, "message_sha256": sha256_hex(str(exc).encode()),
                "boundary_violation": isinstance(exc, WriteBoundaryViolation),
            }, {"request_id": req["id"], "dispatch_id": dispatch.dispatch_id})
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise _Halt("reconciliation_needed",
                        f"dispatch {dispatch.dispatch_id} outcome unknown ({type(exc).__name__}); "
                        "no implicit retry; reconcile by dispatch id", dispatch_id=dispatch.dispatch_id)
        self._complete(req, receipt, via="execute")

    def _reject_dispatch(self, req: dict, exc: Exception) -> None:
        did = req["dispatch_id"]
        released = [{"op": "release", "ref": did, "resource": r, "amount": a}
                    for r, a in self.state.ledger.reserved.get(did, {}).items() if a > 0]
        self._emit("dispatch_error", {
            "dispatch_id": did, "request_id": req["id"], "outcome": "not_started",
            "category": type(exc).__name__, "message_sha256": sha256_hex(str(exc).encode()),
            "message": self._tool_message(req["id"], "rejected", detail="operation rejected by executor"),
        }, {"request_id": req["id"], "dispatch_id": did}, ledger=released)

    def _complete(self, req: dict, receipt: Any, via: str) -> None:
        did, rid = req["dispatch_id"], req["id"]
        anomalies: list[str] = []
        if not isinstance(receipt, Receipt) or not isinstance(receipt.status, str) or not isinstance(receipt.output, str):
            anomalies.append("receipt_malformed")
            receipt = Receipt("malformed_receipt", "", {}, {})
        try:
            check_json(dict(receipt.detail))
        except ValueError:
            anomalies.append("detail_not_json")
            receipt = Receipt(receipt.status, receipt.output, {}, receipt.usage)
        usage: dict[str, float] = {}
        try:
            for res, amt in dict(receipt.usage).items():
                usage[str(res)] = clean_amount(amt, f"usage {res}")
        except (ValueError, TypeError):
            anomalies.append("usage_invalid_charged_reservation_only")
            usage = {}
        reserved = dict(self.state.ledger.reserved.get(did, {}))
        charge = {r: max(reserved.get(r, 0), usage.get(r, 0)) for r in set(reserved) | set(usage)}
        artifacts = self._scan_artifacts(did)
        body = {"status": receipt.status, "output": receipt.output, "detail": dict(receipt.detail), "usage": usage}
        receipt_doc = {"dispatch_id": did, "request_id": rid, "receipt": body, "artifacts": artifacts}
        rdir = self.run_dir / "requests" / rid
        rdir.mkdir(exist_ok=True)
        blob = canonical_bytes(receipt_doc) + b"\n"
        rfile = rdir / "receipt.json"
        if rfile.exists():
            if rfile.read_bytes() != blob:
                raise _Halt("receipt_conflict", f"receipt file for {rid} exists with different content")
        else:
            create_only(rfile, blob)
        ledger = [{"op": "release", "ref": did, "resource": r, "amount": a} for r, a in reserved.items() if a > 0]
        ledger += [{"op": "commit", "ref": did, "resource": r, "amount": a} for r, a in charge.items() if a > 0]
        msg = self._tool_message(rid, receipt.status, dispatch_id=did, output=receipt.output,
                                 artifacts=[{"name": a["name"], "sha256": a["sha256"]} for a in artifacts])
        self._emit("dispatch_result", {
            "dispatch_id": did, "request_id": rid, "via": via, "receipt": body,
            "receipt_sha256": sha256_hex(canonical_bytes(receipt_doc)), "artifacts": artifacts,
            "charge": charge, "anomalies": anomalies, "message": msg,
        }, {"request_id": rid, "dispatch_id": did}, ledger=ledger)

    def _scan_artifacts(self, dispatch_id: str) -> list[dict]:
        d = self._artifact_dir(dispatch_id)
        if not d.is_dir():
            return []
        out = []
        for f in sorted(d.iterdir()):
            if f.is_file() and not f.is_symlink():
                out.append({"name": f.name, "relpath": f"artifacts/{dispatch_id}/{f.name}",
                            "sha256": sha256_file(f), "size": f.stat().st_size})
        return out

    def _reconcile(self, req: dict) -> None:
        did = req["dispatch_id"]
        dispatch = self._make_dispatch(req, req["attempts"])
        self._emit("reconcile_attempt", {"dispatch_id": did, "request_id": req["id"], "attempt": req["attempts"]},
                   {"request_id": req["id"], "dispatch_id": did})
        try:
            res = self.ports.executor.reconcile(dispatch)
            if not isinstance(res, ReconcileResult) or res.status not in ("complete", "unstarted", "unknown"):
                raise ValueError("malformed reconcile result")
        except Exception as exc:
            res = ReconcileResult("unknown", note=f"reconcile failed: {type(exc).__name__}")
        status = res.status
        authority = dict(res.authority) if res.authority else None
        if status == "unstarted":
            if not (authority and authority.get("dispatch_id") == did
                    and isinstance(authority.get("basis"), str) and authority["basis"].strip()):
                status = "unknown"
        if status == "complete" and not isinstance(res.receipt, Receipt):
            status = "unknown"
        self._emit("reconcile_result", {
            "dispatch_id": did, "request_id": req["id"], "status": status, "reported_status": res.status,
            "authority": authority if status == "unstarted" else None, "note": res.note[:500],
        }, {"request_id": req["id"], "dispatch_id": did})
        if status == "complete":
            self._complete(req, res.receipt, via="reconciliation")
        elif status == "unstarted":
            self._dispatch(req, attempt=req["attempts"] + 1, basis="proven_unstarted_reconciliation")
        else:
            raise _Halt("reconciliation_needed",
                        f"dispatch {did} outcome is unknown; refusing to re-execute", dispatch_id=did)

    # ------------------------------------------------------- final choice
    def _deny_final(self, fid: str, reason: str, detail: str) -> None:
        msg = {"role": "tool", "content": canonical_bytes(
            {"item_id": fid, "status": "denied", "reason": reason, "detail": detail}).decode()}
        self._emit("final_choice_denied", {"item_id": fid, "reason": reason, "detail": detail, "message": msg},
                   {"item_id": fid})

    def _process_final(self, fid: str) -> None:
        st = self.state
        item = st.finals[fid]
        payload = item["payload"]
        if st.phase != "discovery":
            return self._deny_final(fid, "phase_closed", f"phase is {st.phase}")
        if not isinstance(payload, dict) or set(payload) - {"choice", "proposal", "rationale"}:
            return self._deny_final(fid, "malformed", "final choice has unknown fields")
        rationale = payload.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > self.config.max_text_chars:
            return self._deny_final(fid, "rationale_required", "a visible justification is required")
        choice = payload.get("choice")
        proposal = payload.get("proposal")
        prop_info: dict = {}
        if choice == "no_edit":
            if proposal is not None or not self.config.allow_no_edit:
                return self._deny_final(fid, "not_allowed", "no_edit with a proposal, or no_edit not enabled")
        elif choice == "patch":
            req = st.requests.get(proposal) if isinstance(proposal, str) else None
            if req is None or req["status"] != "done" or \
                    not self.config.operations[req["operation"]].selectable or req["receipt"]["status"] != "ok":
                return self._deny_final(fid, "not_selectable",
                                        "proposal must be a completed selectable request with an ok receipt")
            if req["id"] not in st.delivered_receipts:
                return self._deny_final(fid, "receipt_not_seen", "select a proposal after receiving its validated receipt")
            prop_info = {
                "proposal_request_id": req["id"], "proposal_dispatch_id": req["dispatch_id"],
                "proposal_receipt_sha256": req["receipt_sha256"], "proposal_artifacts": req["artifacts"],
                "proposal_arguments_sha256": digest_of(req["arguments"]),
            }
            prop_info["proposal_sha256"] = digest_of(prop_info)
        else:
            return self._deny_final(fid, "malformed", "choice must be 'patch' or 'no_edit'")
        sel = {
            "schema": "weight_editor_final_selection/v1", "run_id": self.run_id,
            "binding_digest": self.manifest["binding_digest"], "item_id": fid, "choice": choice,
            "rationale": rationale, **prop_info,
            "evidence_digest": digest_of(st.results),
            "artifacts_digest": digest_of(st.artifacts),
            "budget_digest": digest_of(st.ledger.snapshot()),
            "journal_head": self.journal.head,
            "raw_head": self.raw.head,
        }
        path = self.run_dir / "final_selection.json"
        if path.exists():
            old, _ = load_strict_file(path, what="final_selection.json")
            stable = lambda d: {k: v for k, v in d.items() if k not in ("journal_head",)}
            if stable(old) != stable(sel):
                raise _Halt("final_selection_conflict", "existing final_selection.json disagrees with the journal")
            sel = old
            blob = path.read_bytes()
        else:
            blob = canonical_bytes(sel) + b"\n"
            create_only(path, blob)
        self._emit("final_patch_frozen", {
            "item_id": fid, "choice": choice, "proposal_request_id": prop_info.get("proposal_request_id"),
            "final_selection_sha256": sha256_hex(blob), "journal_head": sel["journal_head"],
            "raw_head": sel["raw_head"], "evidence_digest": sel["evidence_digest"],
            "artifacts_digest": sel["artifacts_digest"], "budget_digest": sel["budget_digest"],
        }, {"item_id": fid})

    def _finalize(self) -> None:
        st = self.state
        if st.terminal is not None:
            return
        arts = self.run_dir / "artifacts" / "manifest.json"
        a_blob = canonical_bytes({"run_id": self.run_id, "artifacts": st.artifacts}) + b"\n"
        if not arts.exists():
            create_only(arts, a_blob)
        elif arts.read_bytes() != a_blob:
            raise _Halt("artifacts_manifest_conflict", "artifacts/manifest.json disagrees with the journal")
        terminal = {
            "schema": "weight_editor_terminal/v1", "run_id": self.run_id, "status": "complete",
            "binding_digest": self.manifest["binding_digest"],
            "frozen_record_digest": st.frozen_record_digest,
            "final_selection_sha256": st.frozen["final_selection_sha256"],
            "raw_head_at_freeze": st.frozen["raw_head"],
            "artifacts_manifest_sha256": sha256_hex(a_blob),
            "budget_digest": st.frozen["budget_digest"],
            "capture_incomplete": st.capture_incomplete, "unknown_events": st.unknown_events,
            "capture_qualification": self._capture_qualification(),
            "hidden_reasoning_captured": False,
        }
        t_blob = canonical_bytes(terminal) + b"\n"
        tpath = self.run_dir / "terminal_manifest.json"
        if tpath.exists():
            if tpath.read_bytes() != t_blob:
                raise _Halt("terminal_conflict", "terminal manifest disagrees with the journal")
        else:
            create_only(tpath, t_blob)
        self._emit("editor_terminal", {"terminal_manifest_sha256": sha256_hex(t_blob), "status": "complete"})
        from .replay import render_markdown  # derived view, written once

        tfile = self.run_dir / "transcript.md"
        if not tfile.exists():
            create_only(tfile, render_markdown(self.run_dir, self.journal.records, self.raw.records).encode())

    # ------------------------------------------------------- summaries
    def _capture_qualification(self) -> dict:
        st = self.state
        missing = sorted(st.completed_turns - st.model_verified_turns)
        return {"qualified": not st.capture_incomplete and not missing,
                "capture_incomplete": st.capture_incomplete, "missing_model_turns": missing,
                "basis": "retained emitted streams and per-turn returned model identity"}

    def summary(self) -> dict:
        st = self.state
        return {
            "run_id": self.run_id, "run_dir": str(self.run_dir),
            "status": "complete" if st.status == "complete" else "incomplete",
            "phase": st.phase, "halt": st.halt, "turns": st.turn_count, "resumes": st.resume_count,
            "journal_records": len(self.journal.records), "raw_records": len(self.raw.records),
            "journal_head": self.journal.head, "raw_head": self.raw.head,
            "capture_incomplete": st.capture_incomplete, "unknown_events": st.unknown_events,
            "capture_qualification": self._capture_qualification(),
            "budget": st.ledger.snapshot(),
            "dispatches": {r["dispatch_id"]: r["status"] for r in st.requests.values() if "dispatch_id" in r},
            "final_selection_sha256": st.frozen["final_selection_sha256"] if st.frozen else None,
            "hidden_reasoning_captured": False,
        }


def _public_bound(bound: Mapping[str, Any]) -> dict:
    strip = lambda b: {k: v for k, v in b.items() if k != "path"}
    return {k: ([strip(x) for x in v] if isinstance(v, list) else strip(v)) for k, v in bound.items()}


# ---------------------------------------------------- integrity helpers


def read_manifest(run_dir: Path) -> dict:
    manifest, _ = load_strict_file(run_dir / "manifest.json", what="run manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA or digest_of(manifest["binding"]) != manifest.get("binding_digest"):
        raise JournalCorrupt("run manifest binding digest does not match its binding", code="manifest_tampered")
    return manifest


def _open_logs(run_dir: Path, manifest: dict, recover_tail: bool, clock):
    bd, rid = manifest["binding_digest"], manifest["run_id"]
    recovered: list[dict] = []
    logs = []
    for name, sub in (("journal", "journal"), ("provider", "provider")):
        path = run_dir / sub / "events.jsonl"
        genesis = genesis_prev(bd, name)
        try:
            logs.append(HashChainLog(path, rid, genesis, clock))
        except PartialTail:
            if not recover_tail:
                raise
            info = quarantine_partial_tail(path, run_dir / "quarantine", rid, genesis)
            recovered.append(info)
            logs.append(HashChainLog(path, rid, genesis, clock))
    return logs[0], logs[1], recovered


def cross_check(run_dir: Path, journal: list[dict], raw: list[dict]) -> None:
    by_seq = {r["seq"]: r for r in raw}
    for rec in journal:
        ref = rec["data"].get("raw") if isinstance(rec["data"], dict) else None
        if ref:
            target = by_seq.get(ref["seq"])
            if target is None or target["digest"] != ref["digest"] or target["data"]["raw_sha256"] != ref["sha256"]:
                raise JournalCorrupt(f"journal seq {rec['seq']} references missing/altered raw event {ref['seq']}",
                                     code="raw_reference_broken")


def _writer_session(run_dir: Path, clock=time.time):
    """Lock + verify + fold a run without any ports (host-side maintenance actions)."""
    if not (run_dir / "manifest.json").is_file():
        raise RunNotFound(f"no registered run at {run_dir}")
    lock_fd = EditorRun._lock(run_dir / "lock")
    try:
        manifest = read_manifest(run_dir)
        journal, raw, _ = _open_logs(run_dir, manifest, False, clock)
        cross_check(run_dir, journal.records, raw.records)
    except BaseException:
        os.close(lock_fd)
        raise
    state = RunState({})
    for rec in journal.records:
        state.apply(rec)
    return lock_fd, journal, raw, state


def freeze_anchor(run_dir: Path) -> dict:
    """Hash-only observation to keep under independent custody before evaluation."""
    from .replay import verify_run

    report = verify_run(run_dir)
    if not report["ok"] or report["status"] != "complete" or not report["capture_qualification"]["qualified"]:
        raise EditorError("anchor requires a verified, capture-qualified completed run", code="run_unqualified")
    manifest = read_manifest(run_dir)
    return {"schema": "weight_editor_freeze_anchor/v1", "run_id": manifest["run_id"],
            "binding_digest": manifest["binding_digest"],
            "final_selection_sha256": sha256_file(run_dir / "final_selection.json"),
            "terminal_manifest_sha256": sha256_file(run_dir / "terminal_manifest.json"),
            "journal_head": report["journal_head"], "raw_head": report["raw_head"]}


def write_anchor(run_dir: Path, output: Path) -> dict:
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.is_relative_to(run_dir.parent):
        raise EditorError("anchor must be outside the run output tree", code="anchor_location")
    lock_fd, journal, raw, _ = _writer_session(run_dir)
    try:
        doc = freeze_anchor(run_dir)
        create_only(output, canonical_bytes(doc) + b"\n")
        return {"ok": True, "anchor_sha256": sha256_file(output), "anchor": doc}
    finally:
        journal.close()
        raw.close()
        os.close(lock_fd)


def advance_phase(run_dir: Path, to: str, reference: str, clock=time.time, *, anchor_file: Path | None = None) -> dict:
    """Host action: freeze -> independent_evaluation -> closed. Never reopens discovery."""
    nxt = {"independent_evaluation": "final_patch_frozen", "closed": "independent_evaluation"}
    lock_fd, journal, raw, state = _writer_session(Path(run_dir), clock)
    try:
        if to not in nxt or state.phase != nxt[to] or state.status != "complete":
            raise EditorError(f"cannot advance from {state.phase}/{state.status} to {to}", code="bad_transition")
        from .replay import verify_run

        if anchor_file is None or Path(anchor_file).resolve().is_relative_to(Path(run_dir).resolve().parent):
            raise EditorError("evaluation requires an external freeze anchor", code="anchor_required")
        report = verify_run(Path(run_dir), anchor_file=Path(anchor_file))
        if not report["ok"] or not report["capture_qualification"]["qualified"]:
            raise EditorError("run evidence or external anchor failed verification", code="run_unqualified")
        rec = journal.append("phase_advanced", {"from": state.phase, "to": to, "reference": reference,
                                               "anchor_sha256": sha256_file(Path(anchor_file))})
        return {"run_id": journal.run_id, "phase": to, "journal_head": rec["digest"]}
    finally:
        journal.close()
        raw.close()
        os.close(lock_fd)


def record_refusal(run_dir: Path, reason: str, detail: dict, clock=time.time) -> bool:
    """Best-effort audit of a resume refused before ports were opened. Never raises."""
    try:
        lock_fd, journal, raw, _ = _writer_session(Path(run_dir), clock)
    except Exception:
        return False
    try:
        journal.append("resume_refused", {"reason": reason, **detail})
        return True
    finally:
        journal.close()
        raw.close()
        os.close(lock_fd)
