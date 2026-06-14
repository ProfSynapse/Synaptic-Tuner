"""Adapter for Codex CLI rollouts (~/.codex/sessions/**/*.jsonl).

Sessions are filed by date, not project, so scope is filtered by the `cwd`
recorded in the session_meta event. Tool actions are function_call /
function_call_output pairs; exit status is plaintext "Process exited with code N".
"""
from __future__ import annotations
import json

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


class CodexAdapter(Adapter):
    name = "codex"

    def _cwd(self, path: str) -> str:
        for line in open(path, errors="ignore"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") == "session_meta":
                return (o.get("payload") or {}).get("cwd", "") or ""
        return ""

    def scope_key(self, path: str, root: str) -> str:
        return self._cwd(path)

    def project(self, path: str, root: str) -> str:
        from pathlib import Path
        return Path(self._cwd(path)).name

    def parse(self, path: str):
        events: list[dict] = []
        signals = {"pr_created": False}
        pending_cmd = ""

        for line in open(path, errors="ignore"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") != "response_item":
                continue
            p = o.get("payload") or {}
            pt = p.get("type")

            if pt == "message":
                role = p.get("role")
                text = blocks_text(p.get("content"))
                if role == "assistant":
                    events.append({"role": "assistant", "text": text, "tool_calls": []})
                elif role == "user":
                    events.append({"role": "human", "text": text})

            elif pt in ("function_call", "custom_tool_call"):
                args = p.get("arguments") or p.get("input") or ""
                cmd = _command_of(args)
                if "gh pr create" in cmd:
                    signals["pr_created"] = True
                events.append({"role": "assistant", "text": "",
                               "tool_calls": [{"name": p.get("name"), "input": args}]})
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
