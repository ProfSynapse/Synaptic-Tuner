"""Adapter for the claude.ai Data Export (``conversations.json``).

This is the bulk export you get from claude.ai -> Settings -> Privacy ->
Export data (NOT a local CLI transcript). It is a single JSON file holding a
*list* of conversations, so it breaks the usual one-file-one-session
assumption: one file contains hundreds or thousands of independent chats.

We resolve that with the ``discover()`` hook: instead of yielding the file
once, we yield one **virtual path per conversation** of the form
``<realpath>::<index>``. Each virtual path then flows through the engine as its
own session (its own scope/outcome/context boundary). ``parse()`` splits the
index back off and parses just that conversation. The parsed file is cached on
the adapter instance so the (potentially large) JSON is read from disk only
once, not once per conversation.

Unlike the local Claude Code CLI transcripts (where ``thinking`` blocks carry
only a server-side ``signature`` and no plaintext), this export DOES include
the verbatim ``thinking`` text — so ``reasoning`` capture is populated here.

Export shape (the bits we use):
  [ { "uuid", "name", "created_at",
      "chat_messages": [
        { "sender": "human"|"assistant",
          "text": str,
          "content": [ {"type": "text", "text": ...},
                       {"type": "thinking", "thinking": ...},
                       {"type": "tool_use", "name": ..., "input": ...},
                       {"type": "tool_result", "content": ..., "is_error": ...},
                       ... ] } ] }, ... ]

Generic across anyone's export — no personal paths or names are baked in.
"""
from __future__ import annotations
import json
import os

from .base import Adapter, blocks_text

_DELIM = "::"


def _command_of(inp) -> str:
    """Best-effort command/intent string from a tool_use ``input``. Most
    claude.ai tools (artifacts, web_search, repl) carry no shell command; a
    bash-style tool will have ``command``/``cmd``. Used only for outcome
    detection, so a miss just means 'no outcome signal' (honest bronze)."""
    if isinstance(inp, str):
        return inp
    if isinstance(inp, dict):
        return str(inp.get("command") or inp.get("cmd") or "")
    return ""


def _result_text(block) -> str:
    """Text out of a tool_result block. Its ``content`` is usually a list of
    text blocks, sometimes a bare string."""
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return blocks_text(c)
    return ""


def _thinking_text(block) -> str:
    """Plaintext chain-of-thought from a thinking block. The export stores it
    verbatim under ``thinking`` (``signature`` is the server token we ignore)."""
    return str(block.get("thinking") or "")


def parse_conversation(conv: dict):
    """One claude.ai conversation -> (events, signals).

    A single assistant message can interleave text / thinking / tool_use /
    tool_result blocks. We stream them in order: text+thinking+tool_use
    accumulate into an assistant event, and each tool_result flushes the
    pending assistant event then emits a ``tool`` event (command paired from
    the preceding tool_use) so the engine's generic outcome detection works.
    """
    events: list[dict] = []
    signals = {"pr_created": False}

    for m in conv.get("chat_messages", []) or []:
        sender = m.get("sender")
        blocks = m.get("content") or []

        if sender == "human":
            text = blocks_text(blocks) or (m.get("text") or "")
            events.append({"role": "human", "text": text})
            continue

        # assistant: accumulate, flushing an event each time a tool_result lands
        acc_text: list[str] = []
        acc_reasoning: list[str] = []
        acc_calls: list[dict] = []
        pending_cmd = ""
        emitted = False

        def flush():
            nonlocal acc_text, acc_reasoning, acc_calls, emitted
            if acc_text or acc_reasoning or acc_calls:
                events.append({
                    "role": "assistant",
                    "text": "".join(acc_text),
                    "tool_calls": acc_calls,
                    "reasoning": "\n".join(t for t in acc_reasoning if t),
                })
                acc_text, acc_reasoning, acc_calls = [], [], []
                emitted = True

        for b in blocks:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                acc_text.append(str(b.get("text", "")))
            elif bt == "thinking":
                acc_reasoning.append(_thinking_text(b))
            elif bt == "tool_use":
                inp = b.get("input")
                cmd = _command_of(inp)
                if "gh pr create" in cmd:
                    signals["pr_created"] = True
                acc_calls.append({"name": b.get("name"), "input": inp})
                pending_cmd = cmd
            elif bt == "tool_result":
                flush()
                out = _result_text(b)
                events.append({
                    "role": "tool",
                    "tool_error": bool(b.get("is_error")),
                    "command": pending_cmd,
                    "output": out,
                })
                pending_cmd = ""
            # token_budget / flag / unknown -> ignored

        flush()
        # fall back to the flat `text` field if no content blocks produced one
        if not emitted and (m.get("text") or "").strip():
            events.append({"role": "assistant", "text": m["text"],
                           "tool_calls": [], "reasoning": ""})

    return events, signals


class ClaudeAiExportAdapter(Adapter):
    """``claude_ai_export`` — one conversations.json -> many sessions.

    ``root`` may point at the export file itself, or at a directory containing
    one or more ``*.json`` exports (``glob`` selects them; default
    ``conversations.json``). Scope is the conversation *title*, so you can
    include/exclude chats by topic via ``scope.*_substrings``.
    """
    name = "claude_ai_export"

    def __init__(self):
        self._cache: dict[str, list] = {}

    def _load(self, real: str) -> list:
        if real not in self._cache:
            with open(real, errors="ignore") as f:
                data = json.load(f)
            self._cache[real] = data if isinstance(data, list) else []
        return self._cache[real]

    @staticmethod
    def _split(path: str):
        """`<realpath>::<idx>` -> (realpath, idx). rsplit so Windows drive
        colons (``C:\\...``) don't confuse the split."""
        if _DELIM in path:
            head, tail = path.rsplit(_DELIM, 1)
            if tail.isdigit():
                return head, int(tail)
        return path, None

    def _export_files(self, root: str):
        if os.path.isfile(root):
            return [root]
        import glob as _glob
        pat = "conversations.json"
        return sorted(_glob.glob(os.path.join(root, "**", pat), recursive=True)) \
            or sorted(_glob.glob(os.path.join(root, "*.json")))

    def discover(self, root: str, glob_pat: str):
        for f in self._export_files(root):
            try:
                convs = self._load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for i in range(len(convs)):
                yield f"{f}{_DELIM}{i}"

    def _conv(self, path: str):
        real, idx = self._split(path)
        convs = self._load(real)
        if idx is None or idx >= len(convs):
            return {}
        return convs[idx]

    def scope_key(self, path: str, root: str) -> str:
        return self._conv(path).get("name") or ""

    def project(self, path: str, root: str) -> str:
        return self._conv(path).get("name") or "(untitled)"

    def parse(self, path: str):
        return parse_conversation(self._conv(path))
