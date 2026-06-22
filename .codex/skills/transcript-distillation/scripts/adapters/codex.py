"""Adapters for Codex CLI rollouts.

Two ways to find the same rollout transcripts:

- ``codex`` (CodexAdapter): filesystem glob over ``~/.codex/sessions/**/*.jsonl``.
  Sessions are filed by date, not project, so scope is the ``cwd`` recorded in
  the session_meta event.
- ``codex_sqlite`` (CodexSqliteAdapter): read Codex's thread catalog at
  ``~/.codex/sqlite/state_*.sqlite`` and follow each thread's ``rollout_path``.
  Portable across mac/linux/windows — the DB lives at a fixed location per
  platform and records the canonical absolute path of every rollout, so it
  finds them even when the files are scattered (e.g. a Windows-mounted home
  under WSL). Scope is the ``cwd`` column (cheap; no per-file pre-parse).

Both parse the same rollout format: response_item payloads with
function_call / function_call_output pairs; exit status is plaintext
"Process exited with code N".
"""
from __future__ import annotations
import glob
import json
import os
import sqlite3
from pathlib import Path

from .base import Adapter, blocks_text


def _command_of(arguments) -> str:
    """Best-effort shell command from a function_call's arguments."""
    if isinstance(arguments, str):
        try:
            d = json.loads(arguments)
            if isinstance(d, dict):
                return str(d.get("cmd") or d.get("command") or arguments)
        except json.JSONDecodeError:
            return arguments
        return arguments
    if isinstance(arguments, dict):
        return str(arguments.get("cmd") or arguments.get("command") or "")
    return ""


def _reasoning_text(payload) -> str:
    """Readable reasoning from a `reasoning` payload. The verbatim CoT lives in
    `encrypted_content` (opaque); the human-readable part is the `summary` (and
    occasionally `content`) text blocks."""
    parts = []
    for key in ("summary", "content"):
        items = payload.get(key) or []
        if isinstance(items, list):
            for b in items:
                if isinstance(b, dict) and b.get("text"):
                    parts.append(str(b["text"]))
    return "\n".join(parts)


def cwd_of_rollout(path: str) -> str:
    """The cwd recorded in a rollout's session_meta event (or "")."""
    for line in open(path, errors="ignore"):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") == "session_meta":
            return (o.get("payload") or {}).get("cwd", "") or ""
    return ""


def parse_codex_rollout(path: str):
    """Parse one rollout JSONL -> (events, signals). Shared by both adapters."""
    events: list[dict] = []
    signals = {"pr_created": False}
    pending_cmd = ""
    pending_reasoning = ""   # reasoning precedes the assistant turn it explains

    for line in open(path, errors="ignore"):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") != "response_item":
            continue
        p = o.get("payload") or {}
        pt = p.get("type")

        if pt == "reasoning":
            r = _reasoning_text(p)
            if r:
                pending_reasoning = (pending_reasoning + "\n" + r) if pending_reasoning else r

        elif pt == "message":
            role = p.get("role")
            text = blocks_text(p.get("content"))
            if role == "assistant":
                events.append({"role": "assistant", "text": text, "tool_calls": [],
                               "reasoning": pending_reasoning})
                pending_reasoning = ""
            elif role == "user":
                events.append({"role": "human", "text": text})

        elif pt in ("function_call", "custom_tool_call"):
            args = p.get("arguments") or p.get("input") or ""
            cmd = _command_of(args)
            if "gh pr create" in cmd:
                signals["pr_created"] = True
            events.append({"role": "assistant", "text": "",
                           "tool_calls": [{"name": p.get("name"), "input": args}],
                           "reasoning": pending_reasoning})
            pending_reasoning = ""
            pending_cmd = cmd

        elif pt in ("function_call_output", "custom_tool_call_output"):
            out = p.get("output")
            s = out if isinstance(out, str) else json.dumps(out)
            low = s.lower()
            err = ("exited with code" in low and "exited with code 0" not in low) \
                or "traceback (most recent call last)" in low
            events.append({"role": "tool", "tool_error": err,
                           "command": pending_cmd, "output": s})
            pending_cmd = ""
    return events, signals


class CodexAdapter(Adapter):
    name = "codex"

    def scope_key(self, path: str, root: str) -> str:
        return cwd_of_rollout(path)

    def project(self, path: str, root: str) -> str:
        return Path(cwd_of_rollout(path)).name

    def parse(self, path: str):
        return parse_codex_rollout(path)


class CodexSqliteAdapter(Adapter):
    """Index-backed discovery via Codex's thread catalog. `root` points at the
    sqlite dir (default ~/.codex/sqlite); `glob` is ignored."""
    name = "codex_sqlite"

    def __init__(self):
        self._cwd_by_path: dict[str, str] = {}

    def _find_db(self, root: str):
        """The newest state_*.sqlite under `root` that has a `threads` table."""
        cands = sorted(glob.glob(os.path.join(root, "state_*.sqlite")))
        for db in reversed(cands):
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                has = con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='threads'"
                ).fetchone()
                con.close()
                if has:
                    return db
            except sqlite3.Error:
                continue
        return None

    def discover(self, root: str, glob_pat: str):
        db = self._find_db(root)
        if not db:
            return
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT rollout_path, cwd FROM threads").fetchall()
        finally:
            con.close()
        for rollout_path, cwd in rows:
            if rollout_path and os.path.exists(rollout_path):
                self._cwd_by_path[rollout_path] = cwd or ""
                yield rollout_path

    def scope_key(self, path: str, root: str) -> str:
        # cwd cached during discover(); fall back to parsing session_meta.
        return self._cwd_by_path.get(path) or cwd_of_rollout(path)

    def project(self, path: str, root: str) -> str:
        return Path(self.scope_key(path, root)).name

    def parse(self, path: str):
        return parse_codex_rollout(path)
