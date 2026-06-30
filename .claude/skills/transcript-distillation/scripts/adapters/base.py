"""Transcript-format adapter contract.

The engine (distill.py) is 100% format-agnostic. Everything format-specific
lives in an adapter. To support a new transcript format, implement this
interface and register it in adapters/__init__.py.

NORMALIZED EVENT MODEL (the contract every adapter emits)
---------------------------------------------------------
parse(path) returns (events, signals).

events: ordered list of dicts, one per logical turn:
  human turn     -> {"role": "human",     "text": str}
  assistant turn -> {"role": "assistant", "text": str,
                     "tool_calls": [{"id": str, "name": str, "input": any}]}
  tool result    -> {"role": "tool", "tool_error": bool,
                     "tool_call_id": str, # matches assistant tool_calls[].id
                     "command": str,   # the command this result is answering
                     "output": str}    # the result text (for outcome detection)

  Why tool events carry command+output: it lets the engine detect "tests
  passed / build clean / committed" generically from regexes in config —
  no format-specific outcome code.

signals: dict of session-level booleans the engine folds into outcomes, e.g.
  {"pr_created": True}  (Claude logs an explicit pr-link event)

Adapters also answer cheap path questions WITHOUT a full parse where possible:
  scope_key(path, root) -> the string matched against scope include/exclude
                           (a project slug, a recorded cwd, ...)
  project(path, root)   -> a human label for the project
  parent_path(path)     -> for nested/subagent transcripts, the parent session
                           file whose outcome should be inherited; else None
"""
from __future__ import annotations


class Adapter:
    name = "base"

    def scope_key(self, path: str, root: str) -> str:
        raise NotImplementedError

    def project(self, path: str, root: str) -> str:
        raise NotImplementedError

    def parse(self, path: str):
        """-> (events: list[dict], signals: dict)"""
        raise NotImplementedError

    def parent_path(self, path: str):
        """-> str | None. Override for nested/subagent transcripts."""
        return None


def blocks_text(content) -> str:
    """Join text blocks from a content list (or pass a str through). Shared
    helper — most chat formats use either a string or a list of typed blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict) and b.get("type") in ("text", "input_text", "output_text"):
                out.append(str(b.get("text", "")))
        return "".join(out)
    return ""
