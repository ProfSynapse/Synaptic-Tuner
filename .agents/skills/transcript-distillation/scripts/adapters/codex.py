"""Adapter for Codex CLI rollouts (~/.codex/sessions/**/*.jsonl).

Sessions are filed by date, not project, so scope is filtered by the `cwd`
recorded in the session_meta event. Tool actions are function_call /
function_call_output pairs; exit status is plaintext "Process exited with code N".
"""
from __future__ import annotations
import json
from pathlib import Path

from .base import Adapter, blocks_text


def _fallback_tool_call_id(path: str, seq: int) -> str:
    return f"codex_{Path(path).stem}_{seq:04d}"


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
        pending_calls: list[tuple[str, str]] = []
        call_seq = 0
        orphan_result_seq = 0

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
                call_seq += 1
                tool_call_id = str(p.get("call_id") or p.get("id") or _fallback_tool_call_id(path, call_seq))
                if "gh pr create" in cmd:
                    signals["pr_created"] = True
                events.append({"role": "assistant", "text": "",
                               "tool_calls": [{"id": tool_call_id, "name": p.get("name"), "input": args}]})
                pending_calls.append((tool_call_id, cmd))

            elif pt in ("function_call_output", "custom_tool_call_output"):
                out = p.get("output")
                s = out if isinstance(out, str) else json.dumps(out)
                low = s.lower()
                err = ("exited with code" in low and "exited with code 0" not in low) \
                    or "traceback (most recent call last)" in low
                source_id = p.get("call_id")
                if source_id:
                    tool_call_id = str(source_id)
                    cmd = ""
                    for idx, (pending_id, pending_cmd) in enumerate(pending_calls):
                        if pending_id == tool_call_id:
                            cmd = pending_cmd
                            pending_calls.pop(idx)
                            break
                elif pending_calls:
                    tool_call_id, cmd = pending_calls.pop(0)
                else:
                    orphan_result_seq += 1
                    tool_call_id, cmd = _fallback_tool_call_id(path, call_seq + orphan_result_seq), ""
                events.append({"role": "tool", "tool_error": err,
                               "tool_call_id": tool_call_id,
                               "command": cmd, "output": s})
        return events, signals
