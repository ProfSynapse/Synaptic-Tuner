"""Codex account adapter against a fake executable. Synthetic only: no model, no auth, no network."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from synaptic_tuner.experimental.weight_editor.adapters import codex_account as ca
from synaptic_tuner.experimental.weight_editor.adapters.codex_account import (
    PROVIDER, CodexAccountAdapter, build_codex_account_ports, normalize,
)
from synaptic_tuner.experimental.weight_editor.broker import EditorRun
from synaptic_tuner.experimental.weight_editor.config import load_config
from synaptic_tuner.experimental.weight_editor.errors import (
    AdapterCancelled, AdapterFailure, AdapterTimeout, PortError, UnsupportedCapability,
)
from synaptic_tuner.experimental.weight_editor.fakes import FakeEffectExecutor, write_fixture
from synaptic_tuner.experimental.weight_editor.ports import PortContext, TransportEvent, TurnInput

REPO = Path(__file__).resolve().parents[3]
FAKE_SRC = Path(__file__).with_name("codex_fake") / "fake_codex.py"
MODEL = {"provider": PROVIDER, "model": "synthetic-model-1", "settings": {"reasoning_effort": "medium"}}
THREAD = "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0001"


# ------------------------------------------------------------ synthetic stream builders

def line(obj) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def envelope(**kw) -> str:
    base = {"action": None, "operation": None, "arguments_json": None, "choice": None, "proposal": None,
            "rationale": ""}
    base.update(kw)
    return json.dumps(base)


def request_env(op, args, why="synthetic rationale"):
    return envelope(action="request", operation=op, arguments_json=json.dumps(args), rationale=why)


def final_env(choice, proposal=None, why="synthetic final rationale"):
    return envelope(action="final_choice", choice=choice, proposal=proposal, rationale=why)


def stream(text, thread=THREAD, model="synthetic-model-1", extra=(), complete=True, reasoning=True):
    started = {"type": "thread.started", "thread_id": thread}
    if model:
        started["model"] = model  # synthetic: models a client that reports returned identity
    out = line(started) + line({"type": "turn.started"})
    if reasoning:
        out += line({"type": "item.completed", "item": {"id": "item_0", "type": "reasoning",
                                                         "text": "synthetic exposed summary"}})
    for e in extra:
        out += e if isinstance(e, bytes) else line(e)
    if text is not None:
        out += line({"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": text}})
    if complete:
        out += line({"type": "turn.completed", "usage": {"input_tokens": 11, "cached_input_tokens": 0,
                                                         "output_tokens": 7, "reasoning_output_tokens": 3}})
    return out


# ------------------------------------------------------------ harness

class Rig:
    def __init__(self, root: Path, turns, **scen):
        self.root = root
        self.bin = root / "bin"
        self.work = root / "work"
        self.control = root / "control"
        self.home = root / "home"
        for d in (self.bin, self.work, self.control, self.home):
            d.mkdir(parents=True)
        self.exe = self.bin / "codex"
        self.exe.write_text(f"#!{sys.executable}\n" + FAKE_SRC.read_text())
        self.exe.chmod(0o755)
        self.set_turns(turns, **scen)
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin"), "HOME": str(self.home), "LANG": "C.UTF-8",
                    "PYTHONDONTWRITEBYTECODE": "1", "UNRELATED_SECRETISH": "should-not-pass"}

    def set_turns(self, turns, **scen):
        norm = []
        for t in turns:
            t = dict(t)
            if "stdout" in t:
                t["stdout_b64"] = base64.b64encode(t.pop("stdout")).decode()
            if "stderr" in t:
                t["stderr_b64"] = base64.b64encode(t.pop("stderr")).decode()
            norm.append(t)
        (self.bin / "scenario.json").write_text(json.dumps({"turns": norm, **scen}))

    @property
    def sha(self):
        return hashlib.sha256(self.exe.read_bytes()).hexdigest()

    def markers(self):
        out = []
        for d in (self.work, *self.work.parents):
            for n in ca.ANCESTOR_MARKERS:
                if (d / n).exists():
                    out.append(str(d / n))
        return tuple(out)

    def adapter(self, **kw):
        args = dict(model=MODEL, codex_path=self.exe, expected_sha256=self.sha,
                    expected_version="codex-cli 0.166.0", work_dir=self.work, control_dir=self.control,
                    mode="probe", base_env=self.env, acknowledged_markers=self.markers(), kill_grace_s=1.0)
        args.update(kw)
        return CodexAccountAdapter(**args)

    def calls(self):
        p = self.bin / "calls.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


@pytest.fixture
def root(request):
    r = REPO / "scratch" / "test-codex-account-adapter" / uuid.uuid4().hex[:8]
    r.mkdir(parents=True)
    yield r
    rep = getattr(request.node, "rep_call", None)
    if rep is None or not rep.failed:
        shutil.rmtree(r, ignore_errors=True)


def turn_input(idx=1, session=None, messages=({"role": "user", "content": "Begin discovery."},), resumed=False):
    return TurnInput("run-x", f"turn-{idx:04d}", idx, session, tuple(messages), "ctx", resumed)


def drain(adapter, ti):
    events, err = [], None
    try:
        for ev in adapter.run_turn(ti):
            events.append(ev)
    except AdapterFailure as exc:
        err = exc
    return events, err


def chan(events, name):
    return b"".join(e.data for e in events if e.channel == name)


# ------------------------------------------------------------ construction / identity / flags

def test_flags_disable_every_tool_and_bind_identity(root):
    rig = Rig(root, [])
    a = rig.adapter()
    argv = a.argv_for(None)
    assert argv[:2] == [str(rig.exe), "exec"] and argv[-1] == "-"
    for flag in ("--json", "--ignore-user-config", "--ignore-rules", "--strict-config", "--skip-git-repo-check",
                 "--output-schema"):
        assert flag in argv
    assert "--ephemeral" not in argv  # resume needs the persisted provider thread
    cs = [argv[i + 1] for i, x in enumerate(argv) if x == "-c"]
    for feat in ca.DISABLED_FEATURES:
        assert f"features.{feat}=false" in cs
    for needed in ('forced_login_method="chatgpt"', 'sandbox_mode="read-only"', 'web_search="disabled"',
                   "tools.update_plan.enabled=false", "skills.include_instructions=false", "mcp_servers={}",
                   "project_doc_max_bytes=0", 'model_reasoning_effort="medium"', "show_raw_agent_reasoning=false"):
        assert needed in cs
    assert argv[argv.index("-m") + 1] == "synthetic-model-1"
    resume = a.argv_for(THREAD)
    assert resume[1:3] == ["exec", "resume"] and resume[-2:] == [THREAD, "-"]
    assert resume[3:-2] == argv[2:-1]  # identical effective flags on resume

    ident = a.identity()
    assert {k: ident[k] for k in ("provider", "model", "settings")} == MODEL
    assert ident["client_sha256"] == rig.sha and ident["client_version"] == "codex-cli 0.166.0"
    assert ident["capabilities"]["raw_events"] is True
    assert ident["capabilities"]["captures_hidden_reasoning"] is False
    assert ident["qualification"]["qualified"] is False and ident["qualification"]["blockers"]
    assert ident["returned_identity"]["synthesized"] is False
    assert "UNRELATED_SECRETISH" not in ident["effective_config"]["env_names"]
    assert json.loads((rig.control / "response-schema.json").read_text())["additionalProperties"] is False
    # stable across construction (broker drift check relies on it)
    assert rig.adapter().identity() == ident


def test_refusals_fail_closed(root):
    rig = Rig(root, [])
    with pytest.raises(PortError, match="API-key"):
        rig.adapter(base_env={**rig.env, "OPENAI_API_KEY": "synthetic"})
    with pytest.raises(PortError, match="sha256"):
        rig.adapter(expected_sha256="0" * 64)
    with pytest.raises(PortError, match="version"):
        rig.adapter(expected_version="codex-cli 9.9.9")
    with pytest.raises(UnsupportedCapability):
        rig.adapter(model={**MODEL, "settings": {"temperature": "0"}})
    with pytest.raises(UnsupportedCapability):
        rig.adapter(model={**MODEL, "provider": "openai-api"})
    with pytest.raises(PortError, match="markers"):
        rig.adapter(acknowledged_markers=())
    with pytest.raises(PortError, match="mode"):
        rig.adapter(mode="auto")
    with pytest.raises(PortError, match="qualification_record"):
        rig.adapter(mode="qualified")
    (rig.control / "instructions.md").write_text("tampered")
    with pytest.raises(PortError, match="differs"):
        rig.adapter()
    (rig.control / "instructions.md").unlink()
    (rig.work / "stray").write_text("x")
    with pytest.raises(PortError, match="empty"):
        rig.adapter()


def test_api_key_login_mode_refused(root):
    rig = Rig(root, [], login="Logged in using an API key")
    with pytest.raises(PortError, match="ChatGPT"):
        rig.adapter()


def test_qualified_mode_requires_matching_record(root):
    rig = Rig(root, [])
    probe = rig.adapter()
    rec = {"schema": ca.QUALIFICATION_SCHEMA, "client_sha256": rig.sha,
           "effective_config_sha256": probe.identity()["effective_config_sha256"], "model": MODEL,
           "verified": {"native_tool_catalogue_empty": True, "model_visible_instructions": True,
                        "returned_model_reported": True, "no_client_internal_retry": True},
           "evidence_sha256": "e" * 64}
    p = root / "qual.json"
    p.write_text(json.dumps(rec))
    q = rig.adapter(mode="qualified", qualification_record=p)
    assert q.identity()["qualification"]["qualified"] is True and q.identity()["qualification"]["blockers"] == []
    for bad in ({"effective_config_sha256": "x"}, {"verified": {**rec["verified"], "returned_model_reported": False}}):
        p.write_text(json.dumps({**rec, **bad}))
        with pytest.raises(PortError):
            rig.adapter(mode="qualified", qualification_record=p)


def test_composition_helper_needs_real_executor(root):
    rig = Rig(root, [])
    ctx = PortContext(args={}, model=MODEL)
    kw = dict(codex_path=rig.exe, expected_sha256=rig.sha, expected_version="codex-cli 0.166.0",
              work_dir=rig.work, control_dir=rig.control, mode="probe", base_env=rig.env,
              acknowledged_markers=rig.markers())
    with pytest.raises(PortError):
        build_codex_account_ports(ctx, object(), **kw)
    with pytest.raises(PortError):
        build_codex_account_ports(ctx, FakeEffectExecutor(root / "fx"), model=MODEL, **kw)
    ports = build_codex_account_ports(ctx, FakeEffectExecutor(root / "fx"), **kw)
    assert isinstance(ports.adapter, CodexAccountAdapter)


# ------------------------------------------------------------ exact capture and mapping

def test_exact_raw_stdout_stderr_and_input_retention(root):
    out = stream(request_env("inspect_weights", {"tensor": "w.layer0"}))
    err = b"synthetic diagnostic \xff\x00 bytes\npartial-no-newline"
    rig = Rig(root, [{"stdout": out, "stderr": err, "chunk": 7}])
    a = rig.adapter()
    ti = turn_input()
    events, exc = drain(a, ti)
    assert exc is None
    assert all(isinstance(e, TransportEvent) for e in events)
    assert chan(events, "provider") == out  # byte-exact, newlines included, despite 7-byte chunking
    assert [e.data for e in events if e.channel == "provider"] == out.splitlines(keepends=True)
    assert chan(events, "client_stderr") == err
    launch, stdin = [e.data for e in events if e.channel == "client_input"]
    assert events[0].channel == events[1].channel == "client_input"  # recorded before the process ran
    call = rig.calls()[0]
    assert base64.b64decode(call["stdin_b64"]) == stdin
    assert json.loads(launch)["argv"] == [str(rig.exe)] + call["argv"]
    assert json.loads(stdin)["messages"] == [dict(m) for m in ti.messages]
    assert "UNRELATED_SECRETISH" not in call["env_names"] and "OPENAI_API_KEY" not in call["env_names"]
    assert call["cwd"] == str(rig.work)
    kinds = [a.parse_event(e.data)["kind"] for e in events if e.channel == "provider"]
    assert kinds == ["session", "session", "reasoning_summary", "request", "finish"]
    req = a.parse_event(out.splitlines(keepends=True)[3])
    assert req == {"kind": "request", "operation": "inspect_weights", "arguments": {"tensor": "w.layer0"},
                   "rationale": "synthetic rationale"}
    sess = a.parse_event(out.splitlines(keepends=True)[0])
    assert sess["session_id"] == THREAD and sess["model"] == "synthetic-model-1"


def test_mapping_rules():
    fin = normalize(line({"type": "item.completed", "item": {"type": "agent_message", "id": "i",
                                                              "text": final_env("no_edit")}}))
    assert fin == {"kind": "final_choice", "choice": "no_edit", "rationale": "synthetic final rationale"}
    fin = normalize(line({"type": "item.completed", "item": {"type": "agent_message", "id": "i",
                                                              "text": final_env("patch", "req-0002")}}))
    assert fin["proposal"] == "req-0002"
    plain = normalize(line({"type": "item.completed", "item": {"type": "agent_message", "id": "i",
                                                                "text": "please run inspect_weights"}}))
    assert plain["kind"] == "message"  # ordinary text never becomes a request
    for bad in (envelope(action="request", operation="x", arguments_json="[1]"),
                envelope(action="request", operation="x", arguments_json='{"a":1,"a":2}'),
                envelope(action="final_choice", choice="maybe"),
                json.dumps({"action": "request", "operation": "x", "arguments_json": "{}", "rationale": "",
                            "choice": None, "proposal": None, "extra": 1})):
        with pytest.raises(ValueError):
            normalize(line({"type": "item.completed", "item": {"type": "agent_message", "id": "i", "text": bad}}))
    assert normalize(line({"type": "item.started", "item": {"type": "command_execution", "id": "c"}}))["kind"] == "unknown"
    assert normalize(line({"type": "future.event"}))["kind"] == "unknown"
    with pytest.raises(ValueError):
        normalize(b"not json\n")


# ------------------------------------------------------------ failure detection (raise after retaining)

FAILS = {
    "native_tool_event": dict(extra=[{"type": "item.started", "item": {"id": "c1", "type": "command_execution",
                                                                       "command": "ls"}}]),
    "unknown_provider_event": dict(extra=[{"type": "future.packet", "x": 1}]),
    "malformed_provider_event": dict(extra=[b"{not json\n"]),
    "provider_error": dict(extra=[{"type": "error", "message": "Reconnecting... 1/5"}]),
    "missing_turn_completion": dict(complete=False),
    "returned_model_unobserved": dict(model=None),
}


@pytest.mark.parametrize("code", sorted(FAILS))
def test_failures_retained_then_raised(root, code):
    out = stream(request_env("inspect_weights", {"tensor": "w.layer0"}), **FAILS[code])
    rig = Rig(root, [{"stdout": out}])
    events, exc = drain(rig.adapter(), turn_input())
    assert exc is not None and exc.code == code
    got = chan(events, "provider")
    assert out.startswith(got) and got  # everything observed before the kill is retained exactly


def test_multiple_actions_and_missing_action(root):
    two = stream(request_env("a", {}), extra=[{"type": "item.completed", "item": {
        "id": "x", "type": "agent_message", "text": request_env("b", {})}}])
    rig = Rig(root, [{"stdout": two}, {"stdout": stream("just prose, no action")}])
    a = rig.adapter()
    assert drain(a, turn_input())[1].code == "multiple_actions"
    assert drain(a, turn_input())[1].code == "no_structured_action"


def test_truncated_stream_nonzero_exit_and_trailing_output(root):
    full = stream(final_env("no_edit"))
    rig = Rig(root, [{"stdout": full[:-5]}, {"stdout": full, "exit": 1},
                     {"stdout": full + line({"type": "turn.started"})}])
    a = rig.adapter()
    ev, exc = drain(a, turn_input())
    assert exc.code == "truncated_stream" and chan(ev, "provider") == full[:-5]
    ev, exc = drain(a, turn_input())
    assert exc.code == "nonzero_exit" and chan(ev, "provider") == full
    assert drain(a, turn_input())[1].code == "output_after_completion"


def test_native_write_into_work_dir_detected(root):
    rig = Rig(root, [{"stdout": stream(final_env("no_edit")), "write_into_cwd": True}])
    assert drain(rig.adapter(), turn_input())[1].code == "work_dir_dirty"


# ------------------------------------------------------------ session / resume / drift

def test_resume_uses_journaled_thread_and_detects_drift(root):
    rig = Rig(root, [{"stdout": stream(request_env("a", {}))}, {"stdout": stream(final_env("no_edit"))},
                     {"stdout": stream(final_env("no_edit"), thread="0199-other-thread")}])
    a = rig.adapter()
    assert drain(a, turn_input(1))[1] is None
    tool_reply = ({"role": "tool", "content": '{"request_id":"req-0001","status":"ok"}'},)
    assert drain(a, turn_input(2, THREAD, tool_reply))[1] is None
    calls = rig.calls()
    assert calls[1]["argv"][:2] == ["exec", "resume"] and calls[1]["argv"][-2:] == [THREAD, "-"]
    assert json.loads(base64.b64decode(calls[1]["stdin_b64"]))["messages"] == [dict(tool_reply[0])]
    assert drain(a, turn_input(3, THREAD, tool_reply))[1].code == "session_drift"


def test_resume_refusals_before_spawn(root):
    rig = Rig(root, [{"stdout": stream(final_env("no_edit"))}])
    a = rig.adapter()
    for ti, code in ((turn_input(2, THREAD, resumed=True), "resume_after_interruption"),
                     (turn_input(2, "bad id with spaces"), "session_invalid"),
                     (turn_input(2, None, messages=()), "missing_bootstrap")):
        ev, exc = drain(a, ti)
        assert exc.code == code and ev == []
    # a fresh bootstrap without a session is allowed (core resends instructions/catalog)
    assert drain(a, turn_input(5, None))[1] is None
    assert rig.calls()[0]["argv"][1] != "resume"
    # client bytes swapped mid-run
    rig.exe.write_text(rig.exe.read_text() + "\n# changed\n")
    assert drain(a, turn_input(6, None))[1].code == "client_drift"


# ------------------------------------------------------------ cancellation / timeout / cleanup

def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # zombie counts as dead
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def test_timeout_kills_owned_process_group(root):
    rig = Rig(root, [{"stdout": line({"type": "thread.started", "thread_id": THREAD, "model": "m"}),
                      "spawn_child": True, "sleep": 300}])
    a = rig.adapter(turn_timeout_s=1.5)
    t0 = time.monotonic()
    with pytest.raises(AdapterTimeout):
        list(a.run_turn(turn_input()))
    assert time.monotonic() - t0 < 20
    pids = json.loads((rig.bin / "pids.json").read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and (_alive(pids["parent"]) or _alive(pids["child"])):
        time.sleep(0.1)
    assert not _alive(pids["parent"]) and not _alive(pids["child"])


def test_cancel_from_another_thread(root):
    rig = Rig(root, [{"stdout": line({"type": "thread.started", "thread_id": THREAD}), "spawn_child": True,
                      "sleep": 300}])
    a = rig.adapter()
    threading.Timer(1.0, a.cancel).start()
    with pytest.raises(AdapterCancelled):
        list(a.run_turn(turn_input()))
    pids = json.loads((rig.bin / "pids.json").read_text())
    time.sleep(0.5)
    assert not _alive(pids["child"])


def test_generator_close_cleans_up(root):
    rig = Rig(root, [{"stdout": line({"type": "thread.started", "thread_id": THREAD}), "spawn_child": True,
                      "sleep": 300}])
    a = rig.adapter()
    gen = a.run_turn(turn_input())
    for ev in gen:
        if ev.channel == "provider":
            break
    gen.close()  # e.g. broker stopped consuming
    pids = json.loads((rig.bin / "pids.json").read_text())
    time.sleep(0.5)
    assert not _alive(pids["parent"]) and not _alive(pids["child"])


# ------------------------------------------------------------ through the real broker

def _broker(root, rig, model=MODEL):
    fx = write_fixture(root / "fx", model=model)
    cfg = load_config(fx["config"])
    from synaptic_tuner.experimental.weight_editor.ports import PortSet
    ports = PortSet(rig.adapter(), FakeEffectExecutor(fx["effects"]))
    return fx, EditorRun.create(cfg, ports, "run-codex-1")


def test_broker_end_to_end_synthetic(root):
    patch = json.dumps({"tensor": "w.layer0", "deltas": [0.1, -0.1, 0.05]})
    t1 = stream(request_env("inspect_weights", {"tensor": "w.layer0"}))
    t2 = stream(request_env("propose_patch", {"name": "edit-a", "tensor": "w.layer0", "patch": patch,
                                              "justification": "small"}))
    t3 = stream(final_env("patch", "req-0002"))
    rig = Rig(root, [{"stdout": t1, "stderr": b"diag-1\n"}, {"stdout": t2}, {"stdout": t3}])
    fx, run = _broker(root, rig)
    try:
        summary = run.run()
    finally:
        run.close()
    assert summary["status"] == "complete", summary
    assert summary["capture_incomplete"] is False
    raw = [json.loads(x) for x in (Path(summary["run_dir"]) / "provider" / "events.jsonl").read_text().splitlines()]
    prov = b"".join(base64.b64decode(r["data"]["raw_b64"]) for r in raw if r["data"]["channel"] == "provider")
    assert prov == t1 + t2 + t3
    errs = b"".join(base64.b64decode(r["data"]["raw_b64"]) for r in raw if r["data"]["channel"] == "client_stderr")
    assert errs == b"diag-1\n"
    effects = (fx["effects"] / "effects.log").read_text().splitlines()
    assert [json.loads(x)["op"] for x in effects] == ["inspect_weights", "propose_patch"]
    calls = rig.calls()
    assert calls[0]["argv"][1] != "resume" and all(c["argv"][1] == "resume" for c in calls[1:])
    first = json.loads(base64.b64decode(calls[0]["stdin_b64"]))["messages"]
    assert any("BROKER CATALOG" in m["content"] for m in first)


def test_broker_halts_without_effects_on_native_tool(root):
    t1 = stream(request_env("inspect_weights", {"tensor": "w.layer0"}),
                extra=[{"type": "item.completed", "item": {"id": "w", "type": "web_search", "query": "q"}}])
    rig = Rig(root, [{"stdout": t1}])
    fx, run = _broker(root, rig)
    try:
        summary = run.run()
    finally:
        run.close()
    assert summary["status"] == "incomplete" and summary["halt"]["code"] == "adapter_failure"
    assert summary["capture_incomplete"] is True
    assert not (fx["effects"] / "effects.log").exists()


def test_broker_halts_without_returned_model(root):
    rig = Rig(root, [{"stdout": stream(request_env("inspect_weights", {"tensor": "w.layer0"}), model=None)}])
    fx, run = _broker(root, rig)
    try:
        summary = run.run()
    finally:
        run.close()
    assert summary["status"] == "incomplete"
    assert not (fx["effects"] / "effects.log").exists()
