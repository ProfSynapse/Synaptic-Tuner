"""Adapter for Claude Code transcripts (~/.claude/projects/**.jsonl).

Main sessions and subagent runs share one schema. Subagents are nested at
<project>/<session>/subagents/agent-*.jsonl and inherit their parent session's
outcome (the commit/PR/test happens in the parent, never in the subagent).
"""
from __future__ import annotations
import json
from pathlib import Path

from .base import Adapter, blocks_text


def _command_of(tool_use: dict) -> str:
    inp = tool_use.get("input")
    if isinstance(inp, dict):
        return str(inp.get("command", "") or "")
    return ""


class ClaudeCodeAdapter(Adapter):
    name = "claude_code"

    def scope_key(self, path: str, root: str) -> str:
        # project slug is the first path component under the projects root
        return Path(path).resolve().relative_to(Path(root).resolve()).parts[0]

    def project(self, path: str, root: str) -> str:
        return self.scope_key(path, root)

    def parent_path(self, path: str):
        # .../project/<session>/subagents/agent.jsonl -> .../project/<session>.jsonl
        if "/subagents/" not in path.replace("\\", "/"):
            return None
        session_dir = Path(path).parents[1]
        return str(session_dir.with_name(session_dir.name + ".jsonl"))

    def parse(self, path: str):
        events: list[dict] = []
        signals = {"pr_created": False}
        id2cmd: dict[str, str] = {}

        for line in open(path, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = o.get("type")

            if t == "pr-link":
                signals["pr_created"] = True
                continue

            msg = o.get("message") or {}
            content = msg.get("content")

            if t == "assistant":
                text, tool_calls = "", []
                if isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        bt = b.get("type")
                        if bt == "text":
                            text += str(b.get("text", ""))
                        elif bt == "tool_use":
                            tool_calls.append({"name": b.get("name"), "input": b.get("input") or {}})
                            if b.get("id"):
                                id2cmd[b["id"]] = _command_of(b)
                elif isinstance(content, str):
                    text = content
                events.append({"role": "assistant", "text": text, "tool_calls": tool_calls})

            elif t == "user":
                results = []
                if isinstance(content, list):
                    results = [b for b in content
                               if isinstance(b, dict) and b.get("type") == "tool_result"]
                if results:
                    for b in results:
                        out = str(b.get("content", ""))
                        err = bool(b.get("is_error"))
                        if not err:  # some errors only show in the text, not the flag
                            low = out[:200].lower()
                            err = "error" in low or "traceback" in low
                        events.append({"role": "tool", "tool_error": err,
                                       "command": id2cmd.get(b.get("tool_use_id"), ""),
                                       "output": out})
                else:
                    events.append({"role": "human", "text": blocks_text(content)})
            # other event types (attachment, system, last-prompt, ...) are skipped
        return events, signals
