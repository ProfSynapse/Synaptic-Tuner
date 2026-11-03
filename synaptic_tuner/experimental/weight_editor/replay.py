"""Independent verification and readable replay of a retained run.

Reads only the run directory (no ports, no original config, no model). Replay
reconstructs what was observed; it never re-asks a nondeterministic model.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from .canon import canonical_bytes, load_strict_file, sha256_file, sha256_hex
from .errors import EditorError
from .journal import genesis_prev, verify_file
from .state import RunState


def load_verified(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    from .broker import cross_check, read_manifest

    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").is_file():
        raise EditorError(f"{run_dir} is not a retained editor run", code="run_not_found")
    manifest = read_manifest(run_dir)
    bd, rid = manifest["binding_digest"], manifest["run_id"]
    journal = verify_file(run_dir / "journal" / "events.jsonl", rid, genesis_prev(bd, "journal"))
    raw = verify_file(run_dir / "provider" / "events.jsonl", rid, genesis_prev(bd, "provider"))
    cross_check(run_dir, journal, raw)
    return manifest, journal, raw


def _limits(run_dir: Path) -> dict:
    cfg, _ = load_strict_file(run_dir / "config.json", what="retained config")
    return cfg.get("resources", {}).get("limits", {})


def verify_run(run_dir: Path, *, anchor_file: Path | None = None) -> dict:
    """Fail-closed verification. Returns ``{"ok": bool, "problems": [...], ...}``."""
    run_dir = Path(run_dir)
    problems: list[dict] = []
    try:
        manifest, journal, raw = load_verified(run_dir)
    except EditorError as exc:
        return {"ok": False, "problems": [{"code": exc.code, "message": str(exc)}]}
    state = RunState(_limits(run_dir))
    for rec in journal:
        state.apply(rec)
    anchor_checked = False
    if anchor_file is not None:
        try:
            anchor, _ = load_strict_file(Path(anchor_file), what="external freeze anchor")
            expected = {"schema": "weight_editor_freeze_anchor/v1", "run_id": manifest["run_id"],
                        "binding_digest": manifest["binding_digest"],
                        "final_selection_sha256": sha256_file(run_dir / "final_selection.json"),
                        "terminal_manifest_sha256": sha256_file(run_dir / "terminal_manifest.json"),
                        "raw_head": raw[-1]["digest"] if raw else None}
            if (set(anchor) != set(expected) | {"journal_head"} or
                    any(anchor.get(k) != v for k, v in expected.items()) or
                    anchor.get("journal_head") not in {r["digest"] for r in journal}):
                problems.append({"code": "external_anchor_mismatch"})
            else:
                anchor_checked = True
        except (EditorError, OSError, ValueError):
            problems.append({"code": "external_anchor_unreadable"})
    for rec in journal:
        if rec["type"] != "dispatch_result":
            continue
        d = rec["data"]
        for a in d["artifacts"]:
            f = run_dir / a["relpath"]
            if not f.is_file() or sha256_file(f) != a["sha256"]:
                problems.append({"code": "artifact_mismatch", "artifact": a["relpath"]})
        rf = run_dir / "requests" / d["request_id"] / "receipt.json"
        try:
            doc, blob = load_strict_file(rf, what="receipt")
            if blob != canonical_bytes(doc) + b"\n" or sha256_hex(canonical_bytes(doc)) != d["receipt_sha256"]:
                problems.append({"code": "receipt_mismatch", "request": d["request_id"]})
        except EditorError:
            problems.append({"code": "receipt_missing", "request": d["request_id"]})
    if (run_dir / "terminal_manifest.json").is_file() and not state.terminal:
        problems.append({"code": "journal_truncated_before_terminal",
                         "message": "terminal manifest exists but the journal has no terminal record"})
    if state.frozen:
        fs = run_dir / "final_selection.json"
        if not fs.is_file() or sha256_file(fs) != state.frozen["final_selection_sha256"]:
            problems.append({"code": "final_selection_mismatch"})
    if state.terminal:
        tm = run_dir / "terminal_manifest.json"
        if not tm.is_file() or sha256_file(tm) != state.terminal["terminal_manifest_sha256"]:
            problems.append({"code": "terminal_manifest_mismatch"})
        else:
            term, _ = load_strict_file(tm, what="terminal manifest")
            if term["frozen_record_digest"] != state.frozen_record_digest:
                problems.append({"code": "terminal_freeze_binding_mismatch"})
            am = run_dir / "artifacts" / "manifest.json"
            if not am.is_file() or sha256_file(am) != term["artifacts_manifest_sha256"]:
                problems.append({"code": "artifacts_manifest_mismatch"})
    recorded = {a["relpath"] for a in state.artifacts}
    unrecorded = []
    adir = run_dir / "artifacts"
    for f in sorted(adir.glob("dsp-*/*")) if adir.is_dir() else []:
        rel = str(f.relative_to(run_dir))
        if rel not in recorded:
            unrecorded.append(rel)
    return {
        "ok": not problems,
        "problems": problems,
        "run_id": manifest["run_id"],
        "status": "complete" if state.status == "complete" else "incomplete",
        "phase": state.phase,
        "halt": state.halt,
        "journal_records": len(journal),
        "raw_records": len(raw),
        "journal_head": journal[-1]["digest"] if journal else None,
        "raw_head": raw[-1]["digest"] if raw else None,
        "unknown_events": state.unknown_events,
        "capture_incomplete": state.capture_incomplete,
        "capture_qualification": {"qualified": not state.capture_incomplete and
                                  not (state.completed_turns - state.model_verified_turns),
                                  "missing_model_turns": sorted(state.completed_turns - state.model_verified_turns)},
        "external_anchor_checked": anchor_checked,
        "integrity_scope": "external freeze anchor" if anchor_checked else "internal consistency; no independent authenticity anchor",
        "unrecorded_artifact_files": unrecorded,
        "unresolved_dispatches": sorted(
            r["dispatch_id"] for r in state.requests.values() if r["status"] in ("intent", "unknown")
        ),
        "budget": state.ledger.snapshot(),
    }


# ------------------------------------------------------------- markdown


def fence(text: str, lang: str = "") -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    bar = "`" * max(3, longest + 1)
    return f"{bar}{lang}\n{text}\n{bar}"


def _raw_text(rec: dict) -> str:
    blob = base64.b64decode(rec["data"]["raw_b64"])
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        return "[non-UTF-8 bytes, base64] " + rec["data"]["raw_b64"]


def render_markdown(run_dir: Path, journal: list[dict], raw: list[dict]) -> str:
    run_dir = Path(run_dir)
    manifest, _ = load_strict_file(run_dir / "manifest.json", what="run manifest")
    raw_by_seq = {r["seq"]: r for r in raw}
    state = RunState(_limits(run_dir))
    out: list[str] = []
    w = out.append
    b = manifest["binding"]
    w(f"# Weight editor transcript `{manifest['run_id']}`")
    w("")
    w("Derived, read-only view of `journal/events.jsonl` and `provider/events.jsonl`. "
      "Raw provider events and named client streams are shown as retained; non-UTF-8 bytes use base64. "
      "Independent pipe ordering is observation order. Hidden reasoning is not captured or claimed; "
      "only provider-exposed events and summaries appear. Replay reconstructs observation, it does not re-ask the model.")
    w("")
    w(f"- Requested model: `{canonical_bytes(b['model']).decode()}`")
    w(f"- Binding digest: `{manifest['binding_digest']}`")
    w(f"- Config sha256: `{b['config_sha256']}`; capsule manifest sha256: `{b['capsule_manifest_sha256']}`")
    w(f"- Adapter: `{b['adapter']['class']}`; executor: `{b['executor']['class']}`")
    w("")
    for rec in journal:
        state.apply(rec)
        d, t = rec["data"], rec["type"]
        tag = f"journal #{rec['seq']}"
        if t == "turn_start":
            w(f"## Turn {d['turn_index']} `{d['turn_id']}` (session `{d['session_id']}`, {tag})")
            w(f"Context digest `{d['context_digest']}`; resumed after interrupted/failed turn: {d['resumed']}")
            w("")
            w("### Model-visible input")
            for m in d["messages"]:
                w(f"**{m['role']}**")
                w(fence(m["content"]))
        elif t == "provider_event":
            r = raw_by_seq[d["raw"]["seq"]]
            mark = "" if d["capture"]["complete"] else " **UNKNOWN/UNPARSED EVENT, capture incomplete**"
            rec_note = " (normalized after crash recovery)" if d.get("recovered_from_raw") else ""
            channel = r["data"].get("channel", "provider")
            w(f"- Stream `{channel}`, event `{d['kind']}` — raw #{d['raw']['seq']} sha256 `{d['raw']['sha256']}`{mark}{rec_note}")
            w(fence(_raw_text(r)))
        elif t == "request_received":
            w(f"### Request `{d['request_id']}` ({tag}, raw #{d['raw']['seq']})")
            p = d["payload"]
            w(f"Operation: `{p.get('operation')}`" if isinstance(p, dict) else "Malformed request payload")
            args = p.get("arguments", {}) if isinstance(p, dict) else {}
            for k, v in (args.items() if isinstance(args, dict) else []):
                if isinstance(v, str):
                    w(f"Argument `{k}` (exact text):")
                    w(fence(v))
                else:
                    w(f"Argument `{k}`: `{canonical_bytes(v).decode()}`")
            extras = {k: v for k, v in p.items() if k not in ("operation", "arguments")} if isinstance(p, dict) else {}
            if extras:
                w(f"Other request fields (carry no authority): `{canonical_bytes(extras).decode()}`")
        elif t == "request_denied":
            w(f"**Denied** `{d['request_id']}` reason `{d['reason']}` — {d['detail']}")
            w("Model saw:")
            w(fence(d["message"]["content"]))
        elif t == "request_admitted":
            w(f"Admitted `{d['request_id']}` → dispatch `{d['dispatch_id']}`; reserved cost `{canonical_bytes(d['cost']).decode()}`; "
              f"bound objects `{canonical_bytes(d['bound']).decode()}`")
        elif t == "dispatch_intent":
            w(f"Write-ahead dispatch intent `{d['dispatch_id']}` attempt {d['attempt']} ({d['basis']}, {tag})")
        elif t == "dispatch_error":
            w(f"**Dispatch error** `{d['dispatch_id']}` outcome `{d['outcome']}` category `{d['category']}`")
        elif t == "reconcile_attempt":
            w(f"Reconciling `{d['dispatch_id']}` by dispatch id (no re-execution)")
        elif t == "reconcile_result":
            w(f"Reconcile result for `{d['dispatch_id']}`: **{d['status']}**"
              + (f", authority `{canonical_bytes(d['authority']).decode()}`" if d.get("authority") else "")
              + (f" — {d['note']}" if d.get("note") else ""))
        elif t == "dispatch_result":
            rid = d["request_id"]
            w(f"Result `{rid}` dispatch `{d['dispatch_id']}` via {d['via']}; receipt status **{d['receipt']['status']}**; "
              f"charge `{canonical_bytes(d['charge']).decode()}`"
              + (f"; anomalies {d['anomalies']}" if d["anomalies"] else ""))
            w(f"Receipt file: [requests/{rid}/receipt.json](requests/{rid}/receipt.json) sha256 `{d['receipt_sha256']}`")
            for a in d["artifacts"]:
                w(f"- Artifact [{a['relpath']}]({a['relpath']}) sha256 `{a['sha256']}` ({a['size']} bytes)")
            w("Tool output exactly as shown to the model:")
            w(fence(d["message"]["content"]))
        elif t == "handoff":
            w(f"### {d['kind'].title()} `{d['from_session']}` → `{d['to_session']}` capsule sha256 `{d['capsule_sha256']}`")
            w(fence(canonical_bytes(d["capsule"]).decode()))
        elif t == "adapter_error":
            w(f"**Adapter error** {d['classification']} (`{d['category']}`) in `{d['turn_id']}`")
        elif t == "turn_end":
            w(f"Turn `{d['turn_id']}` ended: {d['status']}" + (f" ({d['reason']})" if d.get("reason") else ""))
        elif t == "final_choice_received":
            w(f"### Final choice received `{d['item_id']}` (raw #{d['raw']['seq']})")
            w(fence(canonical_bytes(d["payload"]).decode()))
        elif t == "final_choice_denied":
            w(f"**Final choice denied** `{d['item_id']}`: {d['reason']} — {d['detail']}")
        elif t == "final_patch_frozen":
            w(f"## Final choice frozen `{d['choice']}` proposal `{d['proposal_request_id']}`")
            w(f"[final_selection.json](final_selection.json) sha256 `{d['final_selection_sha256']}`; journal head before freeze `{d['journal_head']}`; raw head `{d['raw_head']}`")
        elif t == "run_halted":
            w(f"**Run halted (incomplete): `{d['code']}`** — {d['detail']}")
        elif t == "resume_started":
            w(f"Resume started (prior status `{d['prior_status']}`)")
        elif t == "resume_refused":
            w(f"**Resume refused**: `{d['reason']}` {canonical_bytes({k: v for k, v in d.items() if k != 'reason'}).decode()}")
        elif t == "tail_recovery":
            w(f"**Audited tail recovery** `{d['log']}`: {d['quarantined_bytes']} bytes quarantined at `{d['quarantined_file']}` sha256 `{d['quarantined_sha256']}`")
        elif t == "editor_terminal":
            w(f"## Terminal manifest bound [terminal_manifest.json](terminal_manifest.json) sha256 `{d['terminal_manifest_sha256']}`")
        elif t == "phase_advanced":
            w(f"Phase advanced {d['from']} → {d['to']} (reference `{d['reference']}`)")
        elif t in ("run_registered", "state_transition"):
            w(f"- {t.replace('_', ' ')} ({tag})")
    w("")
    w("## Budget ledger")
    for res, row in state.ledger.snapshot().items():
        w(f"- `{res}`: limit {row['limit']}, committed {row['committed']}, reserved outstanding {row['reserved_outstanding']}")
    w("")
    w(f"Journal head `{journal[-1]['digest'] if journal else None}`; raw head `{raw[-1]['digest'] if raw else None}`; "
      f"unknown events {state.unknown_events}; capture incomplete: {state.capture_incomplete}")
    return "\n".join(out) + "\n"
