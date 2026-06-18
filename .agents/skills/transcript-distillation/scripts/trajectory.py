#!/usr/bin/env python3
"""Native tool-trajectory emit mode for the transcript-distillation engine.

WHERE THIS FITS
---------------
distill.py emits ONE chat row per assistant turn, and DELIBERATELY stubs tool
I/O ("[tool result]") and folds tool calls to prose — it is a *chat* distiller.
That output is structurally useless for tool-use SFT.

This module is the *trajectory* distiller: ONE row per conversation (or per
windowed segment of a long conversation), carrying the real agentic structure
the model must learn to imitate:

  - assistant turns with STRUCTURED ``tool_calls`` (OpenAI/Qwen shape:
    ``[{"type":"function","function":{"name","arguments"}}]``) so the Qwen3.5
    chat template renders native ``<tool_call>`` markup. Source ``input`` →
    ``arguments``; Codex emits a JSON STRING, Claude an object — both are
    normalized to an OBJECT (the template calls ``.items()`` and crashes on a
    string).
  - ``tool`` turns with the REAL tool output (role="tool"), scrubbed by the
    same Redactor and length-truncated (head+tail) so a giant file dump doesn't
    blow the budget while kesping the head/tail a model needs.
  - ``reasoning_content`` ONLY on turns that actually have it (no synthesis;
    traceless turns render content-only so ``assistant_only`` masking never
    trains an empty ``<think></think>``).
  - per-trajectory tool SCHEMAS in the system prompt, INFERRED from observed
    tool ``input`` keys when the transcript carries no tool definitions (the
    personal corpus has none — schema-source = INFER).

The emitted messages are consumed downstream by
``shared.sft_preprocessing.materialize_sft_example(loss_mask_mode="assistant_only",
tool_call_mode="native")`` — this module produces the message list; the trainer
does the tokenize+mask against the STOCK chat template. Empty ``<think></think>``
blocks the template injects on traceless turns are masked at the LABEL level by
assistant_only (never trained), so this module emits reasoning ONLY where it
genuinely exists. So this module is the DATA-shape authority and the trainer is
the LOSS-mask authority; they meet at the OpenAI-style ``messages`` contract.

SECURITY: tool ``output`` is the top leak surface (.env reads, config dumps).
The caller (distill.py) runs the Redactor over every emitted message's content,
including these now-REAL tool turns. Client-work exclusion is TWO-LAYER: the
caller's slug-level scope.exclude_substrings (conversation title / project slug)
PLUS this module's CONTENT-level scan (``client_exclude`` regexes over every
turn's text/output/tool-input) which drops a whole conversation when a
third-party client term appears in the BODY — slug-level alone misses clients
mentioned only in conversation content. This module never writes to disk and
never logs row text — only aggregate stats.

NB: this file has NO third-party deps and does NOT import the trainer (it runs
inside the skill's standalone scripts/ dir). The ``messages`` it emits are the
handoff; tokenization happens in the trainer process.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# tool-call argument normalization (mirror of the trainer-side normalize, kept
# local so the skill stays dependency-free; both must agree on the shape)
# ---------------------------------------------------------------------------
def normalize_arguments(raw: Any) -> dict:
    """Coerce a source tool ``input`` into an OBJECT for ``arguments``.

    Claude logs ``input`` as a dict; Codex logs ``arguments`` as a JSON STRING.
    The Qwen3.5 template calls ``.items()`` on ``arguments`` and HARD-CRASHES on
    a string, so every call must carry an object. A non-JSON or non-object
    string is wrapped as ``{"_raw": <string>}`` so it still renders (and the
    degenerate-loop corpus hazard — a model that spat a giant
    ``"to=functions.shell ..."`` blob as its only "argument" — does not crash
    the template; it lands in a single inspectable key)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    if raw is None:
        return {}
    return {"_raw": str(raw)}


# ---------------------------------------------------------------------------
# tool-output truncation (head + tail; keep the parts a model actually uses)
# ---------------------------------------------------------------------------
def truncate_output(text: str, *, keep: int, head: int, tail: int) -> tuple[str, bool]:
    """Return (possibly-truncated text, was_truncated).

    If ``len <= keep`` keep verbatim. Otherwise keep ``head`` chars + a marker +
    ``tail`` chars. The marker reports how many chars were dropped so the model
    sees an honest elision, not a silent cut."""
    if text is None:
        return "", False
    n = len(text)
    if n <= keep:
        return text, False
    dropped = n - head - tail
    if dropped <= 0:  # head+tail >= keep but < n only if misconfigured; be safe
        return text, False
    marker = f"\n…[truncated {dropped} chars]…\n"
    return text[:head] + marker + text[-tail:], True


# ---------------------------------------------------------------------------
# tool-scope policy (BLACKLIST agent-teams ceremony; KEEP everything else)
# ---------------------------------------------------------------------------
# User steer: BLACKLIST, not whitelist. We keep coding/file/shell tools AND the
# user's own custom MCP tools — and crucially, an UNRECOGNIZED tool name is kept
# (it is far more likely one of the user's MCPs, which they explicitly want in
# training, than ceremony). A whitelist would have silently dropped those MCPs;
# the blacklist protects them. We only ever exclude the PACT / agent-teams
# orchestration plumbing ("we shouldnt have the pact stuff in there really" —
# including from our own sessions). Config overrides `tool_blacklist` wholesale.
DEFAULT_TOOL_BLACKLIST = [
    # PACT / agent-teams orchestration ceremony
    "spawn_agent", "wait_agent", "close_agent", "send_input",
    "TaskCreate", "TaskUpdate", "TaskGet", "TaskList", "TaskStop", "TaskOutput",
    "TeamCreate", "TeamDelete",
    "update_plan", "SendMessage", "Monitor", "TodoWrite",
    "EnterPlanMode", "ExitPlanMode",
    "PushNotification", "CronCreate", "CronList", "CronDelete", "RemoteTrigger",
    "ToolSearch", "Agent", "Skill",
    # interaction/worktree plumbing (user ruled "keep real tools, drop plumbing":
    # KEEP str_replace/create_file/view/web_search as genuine file/search built-ins;
    # DROP the prompt-for-input, message-composer, file-presenter, in-client tool
    # search, worktree-enter/exit, and the claude.ai artifacts pane — these are
    # client/agent ceremony, not generalizable tool use).
    "AskUserQuestion", "ask_user_input_v0", "message_compose_v1", "present_files",
    "tool_search", "tool_search_output", "EnterWorktree", "ExitWorktree", "artifacts",
]

# KEEP-PATTERN INSURANCE: a tool name matching any of these substrings is FORCE-
# KEPT even if it lands in the blacklist. This protects MCP / namespaced custom
# tools (``mcp__srv__tool``, ``local__srv__tool``, ``srv:tool``) unconditionally
# — cheap future-proofing so a renamed ceremony entry can never accidentally
# swallow an MCP, and so MCP-bearing transcripts that arrive later are safe by
# default. Inventory shows these forms ARE present as real calls today
# (mcp__workspace__bash, nexus-synaptic-labs:toolManager_useTools, bare
# noteEditor/vaultLibrarian, hubspot:hubspotContact). Config-overridable.
DEFAULT_TOOL_KEEP_PATTERNS = ["mcp__", "local__", "::", ":"]

# HARD-DROP PATTERNS: a tool whose name CONTAINS any of these (case-insensitive)
# is dropped UNCONDITIONALLY — it overrides keep-patterns. This is for
# third-party PII surfaces the user explicitly wants out of training, NOT for
# ceremony (ceremony lives in the blacklist). ``hubspot`` is here because the
# user ruled that ALL hubspot:* tool uses must be removed — every form
# (``hubspot:hubspotContact``, ``local__hubspot__*``, bare ``hubspotContact``,
# ``hubspot_create_deal``, ``mcp__claude_ai_Hubspot__*``) carries third-party
# CRM contact data (= client PII). A substring match is required because the
# bare/prefixed forms have no namespace marker, so a keep-pattern (``:``) would
# otherwise FORCE-KEEP them. Inventory: 55 distinct hubspot names / 1,807 calls,
# 23 of those names land in the "unknown" bucket (bare/prefixed) and would
# leak through a marker-only filter. Config-overridable via ``tool_drop_patterns``.
DEFAULT_TOOL_DROP_PATTERNS = ["hubspot"]


# CONTENT-LEVEL CLIENT EXCLUSION (third-party data — HARD drop, security-critical)
# ---------------------------------------------------------------------------
# Client mentions do NOT only live in the conversation TITLE / project slug (the
# only thing scope.exclude_substrings + adapter.scope_key can see). They appear
# in the BODY of conversations tagged under benign titles (meeting notes, "ai
# nurse pricing", a github URL). The trajectory schema (messages[]+metadata) is
# NOT the chat rows.jsonl schema the downstream `client_exclude.py` scanner
# consumes, so that content-level scan never reached this path — a real leak.
# We therefore scan the WHOLE conversation text here and DROP the conversation
# if any client term appears.
#
# SINGLE SOURCE OF TRUTH: when the project's canonical client-exclusion module
# (``client_exclude._CLIENT_TERMS``) is importable, we USE IT directly so the
# trajectory path can never drift from the chat-assemble step — the documented
# failure mode is two copies of the pattern diverging. The literal list below is
# only a FALLBACK for using this skill in isolation (the skill must stay usable
# without the private personal_finetune package). bare "baby" is intentionally
# excluded (an unrelated "baby's development" project exists). Override via
# config `client_exclude`.
import re as _re_ce  # noqa: E402

_FALLBACK_CLIENT_EXCLUDE_PATTERNS = [
    r"plan[\s\-]?your[\s\-]?baby",
    r"client-a",
    r"ai[\s\-]?nurse",
]


def _canonical_client_patterns():
    """Return the project's canonical client-term patterns if importable, else
    the in-skill fallback. Importing the canonical list makes it the SINGLE
    SOURCE OF TRUTH so this path can never drift from the chat-assemble step."""
    try:
        from client_exclude import _CLIENT_TERMS  # noqa: E402
        return list(_CLIENT_TERMS)
    except Exception:
        return list(_FALLBACK_CLIENT_EXCLUDE_PATTERNS)


# Public default used by TrajectoryConfig — resolved at import time, preferring
# the canonical source. Drift is additionally asserted by a test.
DEFAULT_CLIENT_EXCLUDE_PATTERNS = _canonical_client_patterns()


def _compile_client_re(patterns):
    pats = patterns if patterns is not None else DEFAULT_CLIENT_EXCLUDE_PATTERNS
    if not pats:
        return None
    return _re_ce.compile("|".join(pats), _re_ce.IGNORECASE)


def _events_have_client_term(events: list[dict], client_re) -> bool:
    """True if ANY turn's text / reasoning / tool output / tool-call input
    mentions a third-party client term. Scans the whole conversation (client
    data hides in any role), short-circuiting on the first hit."""
    if client_re is None:
        return False
    for ev in events:
        for key in ("text", "reasoning", "output", "command"):
            v = ev.get(key)
            if isinstance(v, str) and client_re.search(v):
                return True
        for c in ev.get("tool_calls") or []:
            inp = c.get("input")
            if isinstance(inp, str):
                if client_re.search(inp):
                    return True
            elif isinstance(inp, dict):
                if client_re.search(json.dumps(inp, ensure_ascii=False)):
                    return True
    return False


def _norm_tool_name(name: str) -> str:
    return (name or "").strip()


def _matches_keep_pattern(name: str, keep_patterns) -> bool:
    """True if the name looks like an MCP / namespaced custom tool. The bare
    ``:`` / ``::`` markers catch ``server:tool`` MCP names; ``mcp__``/``local__``
    catch the harness-prefixed forms. (A ``:`` is not a legal char in a plain
    coding-tool name, so this won't force-keep ceremony.)"""
    nm = _norm_tool_name(name)
    return any(p in nm for p in (keep_patterns or ()))


def _matches_drop_pattern(name: str, drop_patterns) -> bool:
    """True if the name contains any hard-drop substring (case-insensitive).
    Hard-drop OVERRIDES keep-patterns — for third-party PII surfaces (hubspot)
    the user wants fully out of training regardless of namespace form."""
    nm = _norm_tool_name(name).lower()
    return any(p.lower() in nm for p in (drop_patterns or ()))


def _is_kept_tool(name: str, blacklist: set[str], keep_patterns=None,
                  drop_patterns=None) -> bool:
    """A tool is KEPT unless excluded. Order of precedence (highest first):
    (1) HARD-DROP — a name matching a drop-pattern (e.g. ``hubspot``) is always
        dropped, overriding keep-patterns (third-party PII the user excluded);
    (2) KEEP-PATTERN — an MCP/namespaced name is force-kept even if blacklisted;
    (3) BLACKLIST — a ceremony name is dropped;
    (4) otherwise KEPT (unknown names are likely user MCPs).
    An empty/blank name is dropped (malformed)."""
    nm = _norm_tool_name(name)
    if not nm:
        return False
    if _matches_drop_pattern(nm, drop_patterns):
        return False
    if _matches_keep_pattern(nm, keep_patterns):
        return True
    return nm not in blacklist


# ---------------------------------------------------------------------------
# schema inference (transcripts carry NO tool definitions -> INFER from inputs)
# ---------------------------------------------------------------------------
# Known descriptions for Claude Code / Codex built-ins so an inferred schema is
# still informative. Anything not listed gets a generic description. This is the
# ONLY place tool prose lives; everything else is derived from observed inputs.
_KNOWN_TOOL_DESCRIPTIONS = {
    "Read": "Read a file from the filesystem.",
    "Edit": "Perform an exact string replacement in a file.",
    "MultiEdit": "Apply multiple exact string replacements to a single file.",
    "Write": "Write (create or overwrite) a file.",
    "NotebookEdit": "Edit a cell of a Jupyter notebook.",
    "Bash": "Execute a bash command and return its output.",
    "Grep": "Search file contents with a regular expression.",
    "Glob": "Find files matching a glob pattern.",
    "WebFetch": "Fetch a URL and process its content.",
    "WebSearch": "Search the web and return results.",
    "shell_command": "Execute a shell command and return its output.",
    "shell": "Execute a shell command and return its output.",
    "apply_patch": "Apply a unified-diff style patch to the workspace.",
}

# apply_patch's argument is a freeform patch blob, not a key/value object, so a
# per-input-key schema would be misleading. Pin a fixed schema for it.
_FIXED_TOOL_SCHEMAS = {
    "apply_patch": {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "A unified-diff patch to apply."}
        },
        "required": ["input"],
    },
}


def _py_type_to_json(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "string"


def infer_tool_schemas(events: list[dict],
                       blacklist: set[str] | None = None,
                       keep_patterns=None,
                       drop_patterns=None) -> tuple[list[dict], dict]:
    """Infer a JSON-schema-ish definition per KEPT tool from observed call inputs.

    Returns (schemas, stats). ``schemas`` is a list of
    ``{"name","description","parameters":{type:object, properties, required}}``
    objects (one per distinct KEPT tool name seen — blacklisted ceremony tools
    are excluded so they never appear in the system prompt). A parameter is
    ``required`` only if it appeared in EVERY observed call of that tool. Type is
    the JSON type of the first non-null value seen for the key. Tools in
    ``_FIXED_TOOL_SCHEMAS`` use their pinned schema regardless of observed keys
    (e.g. apply_patch)."""
    blacklist = blacklist or set()
    # name -> list of input dicts seen
    seen_inputs: dict[str, list[dict]] = {}
    for ev in events:
        if ev.get("role") != "assistant":
            continue
        for call in ev.get("tool_calls") or []:
            name = _norm_tool_name(call.get("name"))
            if not _is_kept_tool(name, blacklist, keep_patterns, drop_patterns):
                continue
            args = normalize_arguments(call.get("input"))
            seen_inputs.setdefault(name, []).append(args)

    schemas = []
    for name in sorted(seen_inputs):
        if name in _FIXED_TOOL_SCHEMAS:
            schemas.append({
                "name": name,
                "description": _KNOWN_TOOL_DESCRIPTIONS.get(name, f"The {name} tool."),
                "parameters": _FIXED_TOOL_SCHEMAS[name],
            })
            continue
        calls = seen_inputs[name]
        all_keys: list[str] = []
        key_types: dict[str, str] = {}
        key_present: Counter = Counter()
        for args in calls:
            for k, v in args.items():
                if k not in key_types and v is not None:
                    key_types[k] = _py_type_to_json(v)
                if k not in all_keys:
                    all_keys.append(k)
            for k in set(args):
                key_present[k] += 1
        props = {k: {"type": key_types.get(k, "string")} for k in all_keys}
        required = [k for k in all_keys if key_present[k] == len(calls)]
        schemas.append({
            "name": name,
            "description": _KNOWN_TOOL_DESCRIPTIONS.get(name, f"The {name} tool."),
            "parameters": {"type": "object", "properties": props,
                           "required": required},
        })
    stats = {"tools": len(schemas), "schema_source": "infer"}
    return schemas, stats


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryConfig:
    """Policy for the trajectory emit, read from config ``trajectory:`` block."""
    enabled: bool = False
    system_prompt: str = "You are a helpful coding agent with access to tools."
    # BLACKLIST of agent-teams ceremony tools to EXCLUDE; every other tool —
    # including unrecognized names (likely user MCPs) — is KEPT.
    tool_blacklist: list[str] = field(default_factory=lambda: list(DEFAULT_TOOL_BLACKLIST))
    # KEEP-PATTERN insurance: names matching any of these substrings are FORCE-
    # KEPT even if blacklisted (protects MCP / namespaced custom tools).
    tool_keep_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_TOOL_KEEP_PATTERNS))
    # HARD-DROP patterns: names CONTAINING any of these (case-insensitive) are
    # dropped unconditionally, overriding keep-patterns (third-party PII the user
    # excluded, e.g. ``hubspot``). The matching tool_result is dropped too.
    tool_drop_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_TOOL_DROP_PATTERNS))
    # CONTENT-LEVEL client-exclusion regexes (third-party client work). A
    # conversation is DROPPED ENTIRELY if any turn's text/output/tool-input
    # matches. Default = the canonical 3-client list; set [] to disable.
    client_exclude: list[str] = field(default_factory=lambda: list(DEFAULT_CLIENT_EXCLUDE_PATTERNS))
    # tool-output truncation (chars)
    output_keep: int = 2000
    output_head: int = 1200
    output_tail: int = 600
    # windowing
    max_seq_tokens: int = 16384
    # chars-per-token used to ESTIMATE a window's token count for budgeting.
    # Empirically (Qwen3.5-4B tokenizer, 7 real agentic windows) a 4.0 estimate
    # UNDER-counts real tokens by ~13% mean / 23% worst, so a window estimated at
    # budget can tokenize OVER budget and get hard-clipped by the trainer (losing
    # the tail — incl. a final reasoning+answer turn). 3.3 ≈ 4.0 / 1.2 gives the
    # windower headroom so real windows land at/under max_seq_tokens. Override in
    # config if you measure a different ratio for your tokenizer.
    chars_per_token: float = 3.3
    # require at least one KEPT (non-ceremony) tool call to emit a trajectory.
    # CONVERSATION-LEVEL gate: a conversation with zero kept tool calls is not
    # emitted as a trajectory at all (pure-chat convs flow through the chat path).
    require_tool_call: bool = True
    # WINDOW-LEVEL gate (independent of the conversation-level one above). When
    # True (default), each emitted window must itself contain >=1 kept tool call;
    # a no-tool window of a tool conversation (a clean native prose/reasoning
    # segment — the native emit NEVER folds tools into prose, so no anti-pattern)
    # is dropped. Set False to PRESERVE those chat/reasoning windows of a
    # qualifying tool conversation (the conversation still had to pass the
    # conversation-level gate to be emitted at all). Only meaningful when
    # require_tool_call is True; pure-chat convs are still excluded entirely.
    require_tool_call_per_window: bool = True
    # where reasoning_content is allowed to survive (NEVER synthesized — this
    # only ever REMOVES recorded reasoning, never adds it):
    #   "latest" — keep it ONLY on the last assistant turn that carries it.
    #              This is the Qwen-family multi-turn convention: stale <think>
    #              blocks from earlier turns are dropped so history stays clean.
    #   "all"    — keep it on every turn that has it.
    reasoning_scope: str = "latest"

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrajectoryConfig":
        d = d or {}
        scope = d.get("reasoning_scope", "latest")
        if scope not in ("latest", "all"):
            raise ValueError(f"Unsupported reasoning_scope: {scope!r} (expected 'latest' or 'all')")
        out = {
            "enabled": d.get("enabled", False),
            "system_prompt": d.get("system_prompt", cls.system_prompt),
            "output_keep": int(d.get("output_keep", 2000)),
            "output_head": int(d.get("output_head", 1200)),
            "output_tail": int(d.get("output_tail", 600)),
            "max_seq_tokens": int(d.get("max_seq_tokens", 16384)),
            "chars_per_token": float(d.get("chars_per_token", 3.3)),
            "require_tool_call": d.get("require_tool_call", True),
            "require_tool_call_per_window": d.get("require_tool_call_per_window", True),
            "reasoning_scope": scope,
        }
        bl = d.get("tool_blacklist")
        if bl is not None:
            out["tool_blacklist"] = list(bl)
        kp = d.get("tool_keep_patterns")
        if kp is not None:
            out["tool_keep_patterns"] = list(kp)
        dp = d.get("tool_drop_patterns")
        if dp is not None:
            out["tool_drop_patterns"] = list(dp)
        ce = d.get("client_exclude")
        if ce is not None:
            out["client_exclude"] = list(ce)
        return cls(**out)


# ---------------------------------------------------------------------------
# message building
# ---------------------------------------------------------------------------
def _build_system_message(base_prompt: str, schemas: list[dict]) -> dict:
    """The system turn = base prompt + the per-trajectory tool schema block.

    The schemas go in the system prompt (not a separate ``tools=`` kwarg) so the
    rendered training text is self-contained — the model learns the
    tool→schema association in-context exactly as it sees it at inference."""
    lines = [base_prompt.strip(), "", "# Available tools", "",
             "You can call the following tools. Each is described with its JSON "
             "parameter schema:", ""]
    for s in schemas:
        lines.append(f"## {s['name']}")
        if s.get("description"):
            lines.append(s["description"])
        lines.append("```json")
        lines.append(json.dumps(s["parameters"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return {"role": "system", "content": "\n".join(lines).rstrip()}


def _structured_tool_calls(tool_calls: list[dict], blacklist: set[str],
                           keep_patterns=None, drop_patterns=None) -> list[dict]:
    """Map normalized adapter tool_calls -> OpenAI/Qwen structured shape, KEEPING
    every tool not in the ceremony blacklist (unknown names kept — likely user
    MCPs; MCP/namespaced names force-kept via keep_patterns; hard-drop names like
    hubspot excluded). ``input`` -> ``arguments`` (object-normalized)."""
    out = []
    for c in tool_calls or []:
        name = _norm_tool_name(c.get("name"))
        if not _is_kept_tool(name, blacklist, keep_patterns, drop_patterns):
            continue
        out.append({
            "type": "function",
            "function": {"name": name,
                         "arguments": normalize_arguments(c.get("input"))},
        })
    return out


def build_messages(events: list[dict], cfg: TrajectoryConfig,
                   schemas: list[dict]) -> tuple[list[dict], dict]:
    """Turn normalized adapter events into a native OpenAI-style ``messages``
    list (system + alternating user/assistant/tool turns).

    - assistant turns keep STRUCTURED tool_calls (every non-ceremony tool) and
      carry ``reasoning_content`` ONLY when the source had reasoning text.
    - tool turns become role="tool" with the REAL (truncated) output.
    - an assistant turn whose tool_calls are ALL dropped by the blacklist, and
      which has no text, is omitted (it would render as an empty turn); its
      following tool result(s) are also dropped to keep the pairing coherent.

    Returns (messages, stats)."""
    blacklist = set(cfg.tool_blacklist or [])
    keep_patterns = list(cfg.tool_keep_patterns or [])
    drop_patterns = list(cfg.tool_drop_patterns or [])
    messages: list[dict] = [_build_system_message(cfg.system_prompt, schemas)]
    stats = {"tool_calls_kept": 0, "tool_calls_dropped": 0,
             "tool_turns": 0, "outputs_truncated": 0,
             "assistant_with_reasoning": 0, "assistant_turns": 0}

    # Decide, per assistant turn, whether it survives the blacklist so we can
    # also drop the orphaned tool results that answer a dropped (ceremony) call.
    drop_next_tool = 0  # how many upcoming tool results to drop
    for ev in events:
        role = ev.get("role")
        if role == "human":
            text = (ev.get("text") or "").strip()
            if text:
                messages.append({"role": "user", "content": text})
            drop_next_tool = 0
        elif role == "assistant":
            text = (ev.get("text") or "").strip()
            raw_calls = ev.get("tool_calls") or []
            kept = _structured_tool_calls(raw_calls, blacklist, keep_patterns, drop_patterns)
            dropped_n = len(raw_calls) - len(kept)
            stats["tool_calls_kept"] += len(kept)
            stats["tool_calls_dropped"] += dropped_n
            # how many tool results to drop next = number of calls we dropped
            drop_next_tool = dropped_n
            if not text and not kept:
                # fully-dropped ceremony turn — omit it entirely.
                continue
            msg: dict = {"role": "assistant", "content": text}
            if kept:
                msg["tool_calls"] = kept
            reasoning = (ev.get("reasoning") or "").strip()
            if reasoning:
                msg["reasoning_content"] = reasoning
                stats["assistant_with_reasoning"] += 1
            messages.append(msg)
            stats["assistant_turns"] += 1
        elif role == "tool":
            if drop_next_tool > 0:
                drop_next_tool -= 1
                continue
            out, was_trunc = truncate_output(
                ev.get("output") or "", keep=cfg.output_keep,
                head=cfg.output_head, tail=cfg.output_tail)
            if was_trunc:
                stats["outputs_truncated"] += 1
            messages.append({"role": "tool", "content": out})
            stats["tool_turns"] += 1

    # reasoning_scope="latest": strip reasoning_content from all but the LAST
    # assistant turn that carries it (Qwen multi-turn convention; never adds
    # reasoning, only removes stale earlier <think> blocks). "all" keeps every.
    if cfg.reasoning_scope == "latest":
        last_idx = max(
            (i for i, m in enumerate(messages)
             if m.get("role") == "assistant" and m.get("reasoning_content")),
            default=None)
        for i, m in enumerate(messages):
            if m.get("role") == "assistant" and m.get("reasoning_content") and i != last_idx:
                del m["reasoning_content"]
    # recount AFTER the scope pass so the stat reflects what is actually emitted.
    stats["assistant_with_reasoning"] = sum(
        1 for m in messages
        if m.get("role") == "assistant" and m.get("reasoning_content"))
    return messages, stats


# ---------------------------------------------------------------------------
# windowing (split over-length trajectories at natural answer boundaries)
# ---------------------------------------------------------------------------
def _est_tokens(messages: list[dict], cpt: float) -> int:
    chars = 0
    for m in messages:
        chars += len(m.get("content") or "")
        for c in m.get("tool_calls") or []:
            chars += len(json.dumps(c.get("function", {}).get("arguments", {}),
                                    ensure_ascii=False))
        chars += len(m.get("reasoning_content") or "")
    return int(chars / cpt) + 1


def _is_answer_boundary(msg: dict) -> bool:
    """A natural cut point: an assistant turn that ENDS the agent's action —
    i.e. has content but NO pending tool_calls (a final/intermediate answer).
    Cutting here keeps each window a self-contained system→...→answer unit."""
    return msg.get("role") == "assistant" and not msg.get("tool_calls")


def window_messages(messages: list[dict], cfg: TrajectoryConfig) -> tuple[list[list[dict]], dict]:
    """Split a too-long ``messages`` list into <= max_seq_tokens windows at
    assistant-answer boundaries. The system message is replayed at the head of
    every window so each is independently trainable. Never hard-truncates or
    silently drops content — an oversize SINGLE turn is kept in its own window
    and reported (the trainer's max_seq_length will clip it, logged here).

    TEMPLATE CONTRACT: the stock Qwen3.5 chat template RAISES "No user query
    found in messages" if a window has no ``user`` turn. A window that starts
    mid-conversation (every window after the first) would otherwise begin with an
    assistant/tool turn and crash the trainer. So we track the most recent
    preceding ``user`` turn and CARRY it into the head of any window that lacks
    one (right after the replayed system message) — it is real prior context, not
    synthesized. The carried turn is masked downstream by ``assistant_only``
    anyway (only assistant spans train), so this adds context without adding
    spurious supervision.

    Returns (windows, stats)."""
    cpt = cfg.chars_per_token
    budget = cfg.max_seq_tokens
    if not messages:
        return [], {"windows": 0, "split": False, "oversize_singletons": 0,
                    "user_carried_windows": 0}
    system = messages[0] if messages[0].get("role") == "system" else None
    body = messages[1:] if system else messages

    total = _est_tokens(messages, cpt)
    if total <= budget:
        return [messages], {"windows": 1, "split": False,
                            "oversize_singletons": 0, "user_carried_windows": 0}

    windows: list[list[dict]] = []
    sys_cost = _est_tokens([system], cpt) if system else 0
    cur: list[dict] = []
    oversize_singletons = 0
    user_carried_windows = 0
    # the most recent user turn seen BEFORE the current buffer started — carried
    # into a window that would otherwise have no user query.
    carry_user: dict | None = None
    last_user_before_cur: dict | None = None

    def flush():
        nonlocal user_carried_windows
        if not cur:
            return
        has_user = any(m.get("role") == "user" for m in cur)
        head = [system] if system else []
        if not has_user and carry_user is not None:
            head = head + [carry_user]
            user_carried_windows += 1
        windows.append(head + list(cur))

    for msg in body:
        mcost = _est_tokens([msg], cpt)
        cur_cost = _est_tokens(cur, cpt) if cur else 0
        if sys_cost + cur_cost + mcost > budget and cur:
            # would overflow; cut here. The next window inherits the last user
            # turn seen so far as its carried context.
            carry_user = last_user_before_cur
            flush()
            cur = []
            cur_cost = 0
        if sys_cost + mcost > budget:
            # a single message exceeds the whole budget on its own.
            carry_user = last_user_before_cur
            flush()
            cur = []
            single_head = [system] if system else []
            if msg.get("role") != "user" and carry_user is not None:
                single_head = single_head + [carry_user]
                user_carried_windows += 1
            windows.append(single_head + [msg])
            oversize_singletons += 1
            continue
        if msg.get("role") == "user":
            last_user_before_cur = msg
        cur.append(msg)
        # cut eagerly at answer boundaries to keep windows self-contained.
        if _is_answer_boundary(msg) and (sys_cost + _est_tokens(cur, cpt)) >= budget * 0.6:
            carry_user = last_user_before_cur
            flush()
            cur = []
    flush()
    return windows, {"windows": len(windows), "split": True,
                     "user_carried_windows": user_carried_windows,
                     "oversize_singletons": oversize_singletons}


# ---------------------------------------------------------------------------
# top-level: one conversation's events -> 0+ trajectory rows
# ---------------------------------------------------------------------------
def _window_tool_names(win: list[dict]) -> list[str]:
    """The distinct KEPT tool names actually called inside a window."""
    return sorted({
        c["function"]["name"]
        for m in win if m.get("role") == "assistant"
        for c in (m.get("tool_calls") or [])
    })


def _assert_schema_invariant(win: list[dict], schema_names: set[str]) -> None:
    """HARD GENERALIZABILITY INVARIANT: every tool a window CALLS must have its
    (inferred) schema present in that window's own system prompt. The model must
    never see a tool call whose schema it wasn't given — that is what forces
    schema-conditioned tool use instead of name memorization. Schemas are built
    over the WHOLE conversation's KEPT tools and the system message is replayed
    into every window, so this should always hold; we assert it so a future
    refactor that breaks the coupling fails loud instead of silently shipping a
    non-generalizable row."""
    for name in _window_tool_names(win):
        if name not in schema_names:
            raise AssertionError(
                f"schema invariant violated: window calls tool {name!r} with no "
                f"schema in its system prompt (have: {sorted(schema_names)})")


def emit_trajectory_rows(events: list[dict], *, source_kind: str, project: str,
                         rel_id: str, cfg: TrajectoryConfig) -> tuple[list[dict], dict]:
    """Build native trajectory row(s) for ONE conversation.

    A row is ``{"messages": [...], "metadata": {...}}``. Long conversations
    yield multiple rows (windows). Returns (rows, stats).

    TWO INDEPENDENT GATES:
      - CONVERSATION-level (``require_tool_call``): emits nothing (with a reason
        in stats) when the conversation has no KEPT (non-ceremony) tool call
        anywhere — pure-chat convs are excluded so they flow through the chat
        path instead, never double-represented.
      - WINDOW-level (``require_tool_call_per_window``): after windowing, a window
        that contains NO kept tool call is dropped when this is True (default —
        keeps the trajectory set focused on tool use). When False, those no-tool
        windows of a QUALIFYING tool conversation are PRESERVED — they are clean
        native prose/reasoning segments (native emit never folds tools into prose,
        so keeping them re-introduces no anti-pattern). The conversation-level
        gate still applies, so a pure-chat conv emits nothing either way.

    A window that mixes kept tool calls with leftover ceremony tokens is RARE
    (build_messages already strips ceremony turns) and is KEPT but counted
    (``mixed_windows``) so we can report how often it happens."""
    # CONTENT-LEVEL CLIENT EXCLUSION (third-party data) — checked FIRST so a
    # client conversation is dropped before any tool/schema work. This closes the
    # gap where slug-level scope.exclude_substrings (conversation title only)
    # misses clients mentioned in the conversation BODY.
    client_re = _compile_client_re(cfg.client_exclude)
    if _events_have_client_term(events, client_re):
        return [], {"emitted": 0, "skipped": "client_excluded",
                    "tools": 0, "schema_source": "infer"}

    blacklist = set(cfg.tool_blacklist or [])
    keep_patterns = list(cfg.tool_keep_patterns or [])
    drop_patterns = list(cfg.tool_drop_patterns or [])
    schemas, schema_stats = infer_tool_schemas(events, blacklist, keep_patterns, drop_patterns)
    schema_names = {s["name"] for s in schemas}
    # gate: at least one KEPT (non-ceremony, non-hard-dropped) tool call present?
    has_kept_call = any(
        _is_kept_tool(c.get("name"), blacklist, keep_patterns, drop_patterns)
        for ev in events if ev.get("role") == "assistant"
        for c in (ev.get("tool_calls") or [])
    )
    if cfg.require_tool_call and not has_kept_call:
        return [], {"emitted": 0, "skipped": "no_kept_tool_call",
                    **schema_stats}

    messages, build_stats = build_messages(events, cfg, schemas)
    # a trajectory needs at least a user turn + an assistant turn to be useful.
    non_system = [m for m in messages if m.get("role") != "system"]
    if len([m for m in non_system if m["role"] == "assistant"]) == 0:
        return [], {"emitted": 0, "skipped": "no_assistant_turn", **schema_stats}

    windows, win_stats = window_messages(messages, cfg)
    rows = []
    windows_dropped_no_tool = 0
    chat_windows_kept = 0  # no-tool windows preserved (per-window gate relaxed)
    mixed_windows = 0  # windows where ceremony tokens survived alongside kept calls
    kept_window_index = 0
    # WINDOW-level gate is independent of the conversation-level one: the conv
    # already passed require_tool_call above, so here we only decide whether to
    # KEEP its no-tool windows (clean native prose/reasoning) or drop them.
    for win in windows:
        tool_names = _window_tool_names(win)
        # SEGMENT-LEVEL gate: a no-tool window is dropped only when the per-window
        # requirement is on. When relaxed, it is KEPT (recovered chat/reasoning
        # window of a qualifying tool conversation) and counted.
        if not tool_names:
            if cfg.require_tool_call and cfg.require_tool_call_per_window:
                windows_dropped_no_tool += 1
                continue
            chat_windows_kept += 1
        # HARD invariant: every called tool's schema is in the window's system.
        _assert_schema_invariant(win, schema_names)
        # mixed-window check: did any ceremony tool name survive into the text?
        # build_messages drops ceremony turns, so this is the rare case where a
        # blacklisted name appears inline in surviving content. We detect it by
        # scanning surviving tool_call names (always clean) — kept here for the
        # reporting hook; ceremony never enters tool_calls so this is ~always 0,
        # but we keep the counter wired for the HOLD report.
        if any(nm in blacklist for nm in tool_names):
            mixed_windows += 1
        rows.append({
            "messages": win,
            "metadata": {
                "source_kind": source_kind,
                "project": project,
                "rel_id": rel_id,
                "window_index": kept_window_index,
                "tool_names": tool_names,
                "schema_source": schema_stats["schema_source"],
                "n_messages": len(win),
                "est_tokens": _est_tokens(win, cfg.chars_per_token),
            },
        })
        kept_window_index += 1
    # backfill window_count now that we know how many SURVIVED segment filtering.
    for r in rows:
        r["metadata"]["window_count"] = len(rows)
    if not rows:
        return [], {"emitted": 0, "skipped": "all_windows_no_tool",
                    "windows_dropped_no_tool": windows_dropped_no_tool,
                    "chat_windows_kept": chat_windows_kept,
                    **schema_stats, **build_stats, **win_stats}
    stats = {"emitted": len(rows), "skipped": None,
             "windows_dropped_no_tool": windows_dropped_no_tool,
             "chat_windows_kept": chat_windows_kept,
             "mixed_windows": mixed_windows,
             **schema_stats, **build_stats, **win_stats}
    return rows, stats


# ---------------------------------------------------------------------------
# REAL-TOKENIZER LENGTH GUARD (rebuild-time hard guarantee)
# ---------------------------------------------------------------------------
# The char-based windower (``window_messages``) budgets with ``chars_per_token``
# (3.3, deliberately overcounting), so its output is an ESTIMATE — a small tail
# of rows can still tokenize over ``max_seq_tokens`` on the real tokenizer
# (measured: 28/3,382 = 0.8%). This guard runs at full-rebuild time, when the
# real tokenizer is loaded, and converts the estimate into a hard guarantee:
# every emitted row that ACTUALLY tokenizes over budget is re-windowed at the
# next answer boundary (with a progressively tighter char budget) and, if it is
# an unsplittable singleton (a single oversize turn), DROPPED. Returns the
# guarded rows + a stats dict so the rebuild can report the count. This is a
# no-op unless a tokenizer is supplied, so the no-tokenizer path (probes,
# offline tests) is unchanged.

def _real_token_len(messages: list[dict], tokenizer) -> int:
    """Real token count of a row, via the SAME path the trainer uses
    (``materialize_sft_example(loss_mask_mode='assistant_only',
    tool_call_mode='native')``) so the guard measures EXACTLY what the trainer
    sees. When the shared helper is importable we use it and let any error
    propagate (a crash there is a real defect the rebuild must surface, not
    something to paper over with a divergent renderer). The plain-template
    fallback is ONLY for isolation/tests where the shared package is absent
    (e.g. a hermetic fake tokenizer) — it is never silently used to mask a
    materialize failure on a real tokenizer."""
    try:
        from shared.sft_preprocessing import materialize_sft_example  # type: ignore
    except Exception:
        materialize_sft_example = None  # shared pkg absent -> isolation fallback
    if materialize_sft_example is not None:
        # IMPORTANT: materialize truncates input_ids to max_seq_length, so we pass
        # an effectively-infinite max_seq_length to get the UNTRUNCATED real length
        # (otherwise len(input_ids) would be capped and the guard could never see
        # an over-budget row). The render itself is identical regardless.
        prepared = materialize_sft_example(
            tokenizer=tokenizer, record={"messages": messages},
            max_seq_length=10 ** 9,
            loss_mask_mode="assistant_only", tool_call_mode="native")
        return len(prepared.input_ids)
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def guard_row_real_length(row: dict, *, cfg: TrajectoryConfig, schema_names: set[str],
                          tokenizer, _depth: int = 0) -> tuple[list[dict], dict]:
    """Ensure a single emitted ``row`` tokenizes at/under ``cfg.max_seq_tokens``
    on the REAL tokenizer. Returns (rows, stats):

      - within budget                     -> [row]   (no-op)
      - over budget, re-windowable        -> the re-windowed rows (each re-checked
                                             recursively; schema invariant re-asserted;
                                             no-kept-tool sub-windows dropped)
      - over budget, unsplittable single  -> []      (dropped; counted)

    Re-windowing reuses ``window_messages`` with a tightened ``chars_per_token``
    so the char-estimator carves smaller pieces; we then re-measure each piece on
    the real tokenizer. Depth-bounded so a pathological row can't loop forever —
    at max depth an still-oversize, multi-message row is dropped rather than
    shipped over budget (fail-closed: we never emit an over-budget row)."""
    msgs = row["messages"]
    # A row with no user turn cannot be rendered by the chat template (the stock
    # Qwen3.5 template raises "No user query found in messages", and the trainer
    # would reject it identically). window_messages carries the prior user turn
    # into user-less windows, but a first/oversize-singleton window can still land
    # without one. Drop it fail-closed BEFORE measuring rather than crash the
    # whole rebuild on apply_chat_template — same policy as the re-windowed
    # no-user fragment below, applied at the entry point.
    if not any(m.get("role") == "user" for m in msgs):
        return [], {"retokenized_over": 0, "rewindowed": 0,
                    "dropped_oversize_singleton": 0,
                    "dropped_no_user_fragment": 1, "checked": 1}
    real = _real_token_len(msgs, tokenizer)
    if real <= cfg.max_seq_tokens:
        return [row], {"retokenized_over": 0, "rewindowed": 0,
                       "dropped_oversize_singleton": 0, "checked": 1}
    body = [m for m in msgs if m.get("role") != "system"]
    if len(body) <= 1 or _depth >= 4:
        # unsplittable (one oversize turn) OR we've tightened as far as is sane:
        # drop rather than ship an over-budget row.
        return [], {"retokenized_over": 1, "rewindowed": 0,
                    "dropped_oversize_singleton": 1, "checked": 1}
    # re-window with a tighter char budget so the estimator cuts smaller pieces.
    tight = TrajectoryConfig(**{**cfg.__dict__,
                                "chars_per_token": cfg.chars_per_token * 0.8})
    subwins, _ = window_messages(msgs, tight)
    if len(subwins) <= 1:
        # estimator still won't split it (e.g. the overflow lives inside one
        # answer segment with no internal boundary) -> tighten harder via recursion
        tight2 = TrajectoryConfig(**{**cfg.__dict__,
                                     "chars_per_token": cfg.chars_per_token * 0.6})
        subwins, _ = window_messages(msgs, tight2)
        if len(subwins) <= 1:
            return [], {"retokenized_over": 1, "rewindowed": 0,
                        "dropped_oversize_singleton": 1, "checked": 1}
    out_rows: list[dict] = []
    agg = {"retokenized_over": 1, "rewindowed": 0,
           "dropped_oversize_singleton": 0, "dropped_no_user_fragment": 0,
           "checked": 1}
    for win in subwins:
        names = _window_tool_names(win)
        # honor the per-window gate: drop a no-tool sub-window only when the
        # window-level requirement is on; when relaxed, keep it (a recovered
        # chat/reasoning fragment of a qualifying tool conversation).
        if cfg.require_tool_call and cfg.require_tool_call_per_window and not names:
            continue  # a no-tool sub-window teaches no tool use -> drop
        # a sub-window with no user turn cannot be rendered by the chat template
        # (and the trainer would reject it the same way). window_messages carries
        # the prior user turn into user-less windows, but a re-windowed piece can
        # still land without one -> drop it fail-closed rather than emit an
        # unrenderable row. Counted as a dropped oversize fragment.
        if not any(m.get("role") == "user" for m in win):
            agg["dropped_no_user_fragment"] += 1
            continue
        _assert_schema_invariant(win, schema_names)  # schemas still present
        sub = {"messages": win,
               "metadata": {**row["metadata"], "n_messages": len(win),
                            "tool_names": names,
                            "est_tokens": _est_tokens(win, cfg.chars_per_token)}}
        deep, dstats = guard_row_real_length(
            sub, cfg=cfg, schema_names=schema_names, tokenizer=tokenizer,
            _depth=_depth + 1)
        out_rows.extend(deep)
        agg["rewindowed"] += dstats["rewindowed"]
        agg["dropped_oversize_singleton"] += dstats["dropped_oversize_singleton"]
        agg["dropped_no_user_fragment"] += dstats.get("dropped_no_user_fragment", 0)
    agg["rewindowed"] += 1
    # reindex window_index/count over the surviving sub-rows of THIS row
    for i, r in enumerate(out_rows):
        r["metadata"]["window_index"] = i
        r["metadata"]["window_count"] = len(out_rows)
    return out_rows, agg
