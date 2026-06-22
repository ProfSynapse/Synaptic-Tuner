#!/usr/bin/env python3
"""Distill training rows from agent transcripts (Claude Code, Codex, or any
format with an adapter). Format-agnostic engine — all policy is in config.

Pipeline: parse (adapter) -> per-assistant-turn rows -> secret scrub ->
token-budget context -> deterministic accept/borderline labels ->
quality funnel (drop ceremony/filler/dups) -> session-outcome tiering
(gold/silver/bronze) -> JSONL.

Usage:
    python distill.py --config config.yaml                  # full run
    python distill.py --config config.yaml --smoke --limit 20 --show 3
    python distill.py --config config.yaml --sources codex --max-context-tokens 4096

Paths in --config (and output.dir) are resolved relative to the current
working directory; source roots support ~ expansion. Only dependency: pyyaml.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sanitize import Redactor          # noqa: E402
from adapters import get_adapter       # noqa: E402
from trajectory import (               # noqa: E402
    TrajectoryConfig,
    emit_trajectory_rows,
    guard_row_real_length,
)


def _schema_names_from_system(sys_msg: dict | None) -> set[str]:
    """Recover the set of tool names whose schema is present in a row's system
    message (each appears as a ``## <name>`` header in the tool-schema block).
    Used to re-assert the schema invariant when the length guard re-windows a
    row. Returns an empty set if there is no system message."""
    if not sys_msg:
        return set()
    names = set()
    for line in (sys_msg.get("content") or "").splitlines():
        if line.startswith("## "):
            names.add(line[3:].strip())
    return names


# ----------------------------------------------------------------------------
# scope
# ----------------------------------------------------------------------------
def scope_ok(key: str, scope: dict) -> bool:
    low = (key or "").lower()
    inc = scope.get("include_substrings") or []
    if inc and not any(s in low for s in inc):
        return False
    if any(s in low for s in scope.get("exclude_substrings") or []):
        return False
    return True


# ----------------------------------------------------------------------------
# row emission + deterministic labeling  (operates on normalized events)
# ----------------------------------------------------------------------------
def render_completion(ev: dict) -> str:
    parts = []
    if ev.get("text", "").strip():
        parts.append(ev["text"].strip())
    if ev.get("tool_calls"):
        parts.append("Tool calls: " + json.dumps(ev["tool_calls"], ensure_ascii=False))
    return "\n".join(parts)


def _is_correction(text: str, markers: list[str], max_len: int) -> bool:
    low = text.strip().lower()
    if not low or len(low) > max_len:
        return False
    return any(low.startswith(m) for m in markers)


def classify_feedback(text: str, fb: dict):
    """Per-turn user-feedback weight, read off the NEXT human turn. Applies to
    EVERY source (a praised/accepted coding turn is extra-good; a corrected one
    is suspect). Returns (label, score):
        praised  +2   explicit approval ("perfect", "that works", "lgtm")
        accepted +1   short acceptance/proceed ("yes", "continue", "apply")
        corrected -1  dissatisfaction ("doesn't work", "that's not what i meant")
        neutral   0   no signal
    Order is conservative: corrected > praised > accepted. Real feedback is
    front-loaded, so praise/reject are matched only within the first
    `window_chars` of the turn — this avoids false positives from a marker
    buried in pasted code/JSON/a long prompt ("...you misunderstood..." deep in
    a quote is not feedback). Accept must be a SHORT turn that starts with a
    marker, so "yes but you forgot X" is not mistaken for clean acceptance."""
    if not fb or not fb.get("enabled"):
        return "neutral", 0
    low = (text or "").strip().lower()
    # a turn that opens with a markdown quote is pasted/quoted content, not a
    # reaction — skip it so prose words ("perfect", "love") aren't read as praise.
    if not low or low.startswith(">"):
        return "neutral", 0
    if any(m in low[:fb.get("window_chars", 100)] for m in fb.get("reject_markers") or []):
        return "corrected", -1
    if any(m in low[:fb.get("praise_window_chars", 50)] for m in fb.get("praise_markers") or []):
        return "praised", 2
    if len(low) <= fb.get("max_accept_chars", 45) and \
            any(low.startswith(m) for m in fb.get("accept_markers") or []):
        return "accepted", 1
    return "neutral", 0


# ----------------------------------------------------------------------------
# domain classifier  (deterministic, regex-only — NO LLM)
# ----------------------------------------------------------------------------
# CRITICAL: "coding" means AUTHORING CODE, not merely invoking a tool. A chat
# turn that calls web_search / analysis / artifacts / any MCP tool is STILL
# "chat" — domain is decided purely on the row's TEXT content (prompt +
# completion + reasoning), NEVER on tool_names presence. That's the whole point.
_DOM_LANG_TAGS = ("python", "py", "js", "javascript", "ts", "typescript", "tsx",
                  "jsx", "bash", "sh", "shell", "rust", "rs", "go", "golang",
                  "java", "cpp", "c++", "ruby", "rb", "php", "sql", "yaml", "yml",
                  "json", "toml", "html", "css", "dockerfile", "make")
_DOM_EXT = re.compile(r"\.(py|ts|tsx|js|jsx|rs|go|java|cpp|cc|hpp|h|rb|php|sql|sh|ya?ml|toml|ipynb)\b")
_DOM_CMD = re.compile(r"\b(pytest|npm |yarn |pnpm |cargo |go build|go test|docker |kubectl |"
                      r"git (?:commit|push|rebase|merge|diff)|pip install|tsc\b|webpack|vite |make\b|gradle |mvn )")
_DOM_TRACE = re.compile(r"(traceback \(most recent call last\)|^\s*at .+\(.+:\d+\)|panicked at|segmentation fault)", re.I | re.M)
_DOM_DIFF = re.compile(r"(^diff --git |^@@ .+ @@|^\+\+\+ |^--- )", re.M)
_DOM_SYNTAX = re.compile(r"(def |import |from \w+ import|function |class |const |let |=> |#include|fn |async def )")
_DOM_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)


def _domain_fenced_is_code(text, lang_tags):
    """A fenced block is code if its tag is a known language, OR (untagged) its
    body has >=4 programming symbols or matches a syntax marker — so prose/quote
    fences don't count."""
    for tag, body in _DOM_FENCE.findall(text):
        t = tag.strip().lower()
        if t in lang_tags:
            return True
        sym = sum(body.count(c) for c in "{};=()")
        if sym >= 4 or _DOM_SYNTAX.search(body):
            return True
    return False


def classify_domain(text, dom_cfg):
    """Per-turn content domain: "coding" or "chat". Deterministic/regex only.

    "coding" = the text AUTHORS or operates on code: a code fence with real
    code, a unified diff, a traceback, a build/test/git command invocation, or
    a code-file extension alongside programming syntax. Everything else is
    "chat" — including tool-calling turns whose text is conversational. This is
    decided on the row's concatenated text, INDEPENDENT of tool_names / source.

    Returns "chat" when disabled (default-enabled)."""
    if dom_cfg is not None and not dom_cfg.get("enabled", True):
        return "chat"
    cfg = dom_cfg or {}
    lang_tags = tuple(t.lower() for t in cfg.get("lang_tags", _DOM_LANG_TAGS))
    t = text or ""
    if _domain_fenced_is_code(t, lang_tags):
        return "coding"
    if _DOM_DIFF.search(t):
        return "coding"
    if _DOM_TRACE.search(t):
        return "coding"
    if _DOM_CMD.search(t):
        return "coding"
    if _DOM_EXT.search(t) and _DOM_SYNTAX.search(t):
        return "coding"
    return "chat"


def emit_rows(events, *, source_kind, project, rel_id, lab, ctx_budget,
              capture_reasoning=False, fb=None, dom=None):
    rows = []
    corr_markers = lab["correction_markers"]
    intr_markers = lab["interrupt_markers"]
    max_tokens = ctx_budget.get("max_context_tokens", 8192)
    cpt = ctx_budget.get("chars_per_token", 4.0)
    max_msgs = ctx_budget.get("max_context_messages", 60)
    est = lambda s: int(len(s) / cpt) + 1   # noqa: E731

    def _content(e):
        if e["role"] == "tool":
            return "[tool result]"
        return render_completion(e) if e["role"] == "assistant" else e.get("text", "")

    for i, ev in enumerate(events):
        if ev["role"] != "assistant":
            continue
        had_tool = bool(ev.get("tool_calls"))

        tool_error = False
        if had_tool:
            for j in range(i + 1, min(i + 3, len(events))):
                if events[j]["role"] == "tool":
                    tool_error = bool(events[j].get("tool_error"))
                    break

        next_human = None
        for j in range(i + 1, len(events)):
            if events[j]["role"] == "human":
                next_human = events[j].get("text", "")
                break

        interrupted = any(m in ev.get("text", "").lower() for m in intr_markers) or (
            next_human is not None and any(m in next_human.lower() for m in intr_markers))
        corrected = next_human is not None and _is_correction(
            next_human, corr_markers, lab["max_correction_chars"])
        feedback, feedback_score = classify_feedback(next_human or "", fb)

        failure_labels = []
        if corrected:
            label, tier = False, "bad"
            failure_labels.append("user_corrected")
            if tool_error:
                failure_labels.append("tool_error")
            if interrupted:
                failure_labels.append("interrupted")
        elif tool_error or interrupted:
            label, tier = None, "borderline"
            if tool_error:
                failure_labels.append("tool_error")
            if interrupted:
                failure_labels.append("interrupted")
        else:
            label, tier = True, "good"

        completion_str = render_completion(ev)
        reasoning_str = ev.get("reasoning") or ""
        budget = max_tokens - est(completion_str)
        oversize = budget < 0
        conv_rev = [{"role": "assistant", "content": completion_str}]
        used = 0
        for e in reversed(events[:i]):
            if len(conv_rev) >= max_msgs:
                break
            c = _content(e)
            t = est(c)
            if used + t > budget and len(conv_rev) > 1:
                break
            used += t
            role = {"human": "user", "assistant": "assistant", "tool": "tool"}[e["role"]]
            conv_rev.append({"role": role, "content": c})
        conv = list(reversed(conv_rev))

        prompt = ""
        for e in reversed(events[:i]):
            if e["role"] == "human":
                prompt = e.get("text", "")
                break

        # domain is decided on the row's CONTENT (prompt + completion +
        # reasoning), independent of tool_names / source. Coding == authoring
        # code, not invoking a tool.
        domain = classify_domain(
            prompt + " " + completion_str + " " + reasoning_str, dom)

        row = {
            "conversations": conv,
            "prompt": prompt,
            "completion": completion_str,
            "label": label,
            "score_tier": tier,
            "failure_labels": failure_labels,
            "source_example_id": f"{source_kind}:{rel_id}:{i}",
            "metadata": {
                "source_kind": source_kind,
                "project": project,
                "turn_index": i,
                "tool_names": [c.get("name") for c in ev.get("tool_calls", [])],
                "had_tool_error": tool_error,
                "interrupted": interrupted,
                "next_correction": corrected,
                "user_feedback": feedback,        # praised|accepted|neutral|corrected
                "feedback_score": feedback_score,  # +2 / +1 / 0 / -1
                "domain": domain,                 # coding|chat (content, not tools)
                "est_tokens": used + est(completion_str),
                "oversize": oversize,
            },
        }
        if capture_reasoning:
            row["reasoning"] = ev.get("reasoning") or ""
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# session outcome  (generic: reads command/output off normalized tool events)
# ----------------------------------------------------------------------------
def classify_session(events, signals, oc) -> dict:
    test_re = re.compile(oc["test_command_re"], re.I)
    build_re = re.compile(oc["build_command_re"], re.I)
    commit_re = re.compile(oc["commit_command_re"], re.I)
    pass_re = re.compile(oc["pass_re"], re.I)
    fail_re = re.compile(oc["fail_re"], re.I)
    approve_re = re.compile(oc["approve_re"], re.I)
    amax = oc.get("approve_max_chars", 80)

    res = {"tests_passed": None, "clean_build": None, "committed": False,
           "pr_created": bool((signals or {}).get("pr_created")), "user_approved": False}
    last_human = ""

    def category(cmd):
        if commit_re.search(cmd):
            return "commit"
        if test_re.search(cmd):
            return "tests"
        if build_re.search(cmd):
            return "build"
        return None

    for e in events:
        if e["role"] == "human" and e.get("text", "").strip():
            last_human = e["text"].strip()
        elif e["role"] == "tool":
            cmd = e.get("command") or ""
            if "gh pr create" in cmd:
                res["pr_created"] = True
            cat = category(cmd)
            if not cat:
                continue
            out = e.get("output") or ""
            ok = (not e.get("tool_error")) and not fail_re.search(out)
            if cat == "tests":
                res["tests_passed"] = bool(ok and (pass_re.search(out) or not out.strip()))
            elif cat == "build":
                res["clean_build"] = bool(ok)
            elif cat == "commit" and ok:
                res["committed"] = True

    if last_human and len(last_human) <= amax and approve_re.search(last_human):
        res["user_approved"] = True

    return {"outcomes": [k for k, v in res.items() if v is True], "evidence": res}


def derive_tier(outcomes, tiers_cfg) -> str:
    s = set(outcomes)
    if s & set(tiers_cfg.get("gold", [])):
        return "gold"
    if s & set(tiers_cfg.get("silver", [])):
        return "silver"
    return "bronze"


# ----------------------------------------------------------------------------
# quality funnel
# ----------------------------------------------------------------------------
def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def should_keep(row, q, seen, cpt):
    comp = row["completion"]
    tools = [t for t in row["metadata"]["tool_names"] if t]
    drop_tools = set(q.get("drop_tools", []))
    drop_skills = q.get("drop_skills", [])

    if tools:
        kept = [t for t in tools if t not in drop_tools]
        ceremony_skill = "Skill" in tools and any(sk in comp for sk in drop_skills)
        if not kept or (len(kept) == 1 and kept[0] == "Skill" and ceremony_skill):
            return False, "ceremony_tool"
    else:
        if int(len(comp) / cpt) < q.get("min_completion_tokens", 12):
            return False, "trivial_text"

    markers = q.get("drop_prompt_markers") or []
    if markers and any(m in row["prompt"][:300] for m in markers):
        return False, "ritual_prompt"

    content_subs = q.get("drop_content_substrings") or []
    if content_subs:
        # include reasoning: a client/topic mention in captured thinking must
        # also drop the row, not just one in the prompt/completion.
        blob = (row["prompt"] + " " + comp + " " + (row.get("reasoning") or "")).lower()
        if any(s.lower() in blob for s in content_subs):
            return False, "content_excluded"

    if q.get("dedup", {}).get("enabled"):
        key = hash(_norm(comp))
        if key in seen:
            return False, "duplicate"
        seen.add(key)
    return True, "keep"


# ----------------------------------------------------------------------------
def gather_files(sources_cfg, scope, only):
    """Yield (source_kind, adapter, path, root) for every in-scope transcript."""
    for kind, sc in sources_cfg.items():
        if only and kind not in only:
            continue
        if not sc.get("enabled", True):
            continue
        adapter = get_adapter(sc["format"])
        root = os.path.expanduser(sc["root"])
        for path in adapter.discover(root, sc.get("glob", "")):
            try:
                if not scope_ok(adapter.scope_key(path, root), scope):
                    continue
            except Exception:
                continue
            yield kind, adapter, path, root


# ----------------------------------------------------------------------------
# trajectory emit mode  (native tool-use rows — one per conversation/window)
# ----------------------------------------------------------------------------
def _redact_nested(value, redactor, counts):
    """Recursively redact every STRING reachable from a tool_call ``arguments``
    value, returning a redacted copy. Tool arguments are arbitrarily nested
    (a value can be a list of strings, a dict of dicts, etc.), so a flat
    top-level-only scan would let a secret nested inside a structured argument
    (e.g. an env dump passed as a list, or a config object) bypass the security
    gate. This walks dict values and list items so EVERY string is scrubbed."""
    if isinstance(value, str):
        return redactor.redact(value, counts)
    if isinstance(value, dict):
        return {k: _redact_nested(v, redactor, counts) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_nested(v, redactor, counts) for v in value]
    return value


def _redact_messages(messages, redactor, counts):
    """Redact EVERY message's content in place — incl. the now-REAL tool-turn
    output (the top leak surface) and the rendered tool_call arguments (at EVERY
    nesting depth). This is the SECURITY gate the trajectory mode depends on."""
    for m in messages:
        if m.get("content"):
            m["content"] = redactor.redact(m["content"], counts)
        if m.get("reasoning_content"):
            m["reasoning_content"] = redactor.redact(m["reasoning_content"], counts)
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, (dict, list)):
                # recurse so secrets nested inside structured arguments are also
                # scrubbed (a flat top-level-only scan misses nested values).
                fn["arguments"] = _redact_nested(args, redactor, counts)


def run_trajectory(cfg, args, redactor, redaction_counts):
    """Native tool-trajectory emit: one row per conversation (windowed if long),
    carrying structured tool_calls + real scrubbed/truncated tool outputs +
    reasoning-where-present + per-trajectory inferred tool schemas. Client-work
    exclusion stays conversation-level (scope_ok). The Redactor runs over every
    emitted message (incl. tool turns) before write."""
    scope = cfg.get("scope", {})
    tcfg = TrajectoryConfig.from_dict(cfg.get("trajectory"))
    only = set(s.strip() for s in args.sources.split(",") if s.strip())

    # SINGLE-SOURCE client exclusion: when config sets
    # `client_exclude.canonical_module_dir`, import the project's canonical
    # _CLIENT_TERMS from that dir and use it for the content-level client scan, so
    # the trajectory path can NEVER drift from the chat-assemble step. Fail LOUD
    # if the import fails (a silent fallback to a possibly-drifted copy is exactly
    # the leak this guards against). Unset -> the skill's built-in default is used.
    ce_cfg = (cfg.get("client_exclude") or {})
    canon_dir = ce_cfg.get("canonical_module_dir")
    if canon_dir:
        canon_dir = os.path.expanduser(canon_dir)
        if canon_dir not in sys.path:
            sys.path.insert(0, canon_dir)
        from client_exclude import _CLIENT_TERMS  # fail loud if unavailable
        tcfg.client_exclude = list(_CLIENT_TERMS)
        print(f"  [client-exclude] using canonical _CLIENT_TERMS "
              f"({len(_CLIENT_TERMS)} patterns) from {canon_dir}", file=sys.stderr)

    # REAL-TOKENIZER LENGTH GUARD (optional, rebuild-time). When
    # `length_guard.tokenizer` is set in config (or --length-guard-tokenizer is
    # passed), every emitted row is re-measured on the real tokenizer and any row
    # over max_seq_tokens is re-windowed at the next answer boundary (or dropped
    # if an unsplittable singleton). No-op when unset, so probes/offline tests are
    # unaffected. This turns the char-based estimate into a hard guarantee.
    guard_tok = None
    lg = (cfg.get("length_guard") or {})
    guard_tok_name = getattr(args, "length_guard_tokenizer", "") or lg.get("tokenizer")
    if guard_tok_name:
        # the guard measures via shared.sft_preprocessing (the trainer's path);
        # ensure the repo root is importable when distill runs standalone so the
        # measurement matches what the trainer will actually see.
        repo_root = Path(__file__).resolve().parents[4]
        for cand in (repo_root, *repo_root.parents):
            if (cand / "shared" / "sft_preprocessing.py").exists():
                sys.path.insert(0, str(cand))
                break
        from transformers import AutoTokenizer
        print(f"  [length-guard] loading real tokenizer {guard_tok_name} ...",
              file=sys.stderr)
        guard_tok = AutoTokenizer.from_pretrained(guard_tok_name)
    guard_stats = Counter()

    out_dir = Path(os.path.expanduser(cfg["output"]["dir"]))
    if args.smoke:
        out_dir = out_dir / cfg["output"].get("smoke_subdir", "smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["output"].get("trajectory_file", "trajectories.jsonl")

    per_source_files: Counter = Counter()
    skip_counts: Counter = Counter()
    tool_name_counts: Counter = Counter()
    agg = Counter()              # summed build/window stats
    schema_source_counts: Counter = Counter()
    n_rows = 0
    n_convs = 0
    samples = []

    with open(out_path, "w") as fout:
        for kind, adapter, path, root in gather_files(cfg["sources"], scope, only):
            if args.limit and per_source_files[kind] >= args.limit:
                continue
            per_source_files[kind] += 1
            try:
                events, _signals = adapter.parse(path)
            except Exception as e:
                print(f"  ! parse failed {path}: {e}", file=sys.stderr)
                continue
            n_convs += 1
            project = adapter.project(path, root)
            rel_id = Path(path).name
            rows, stats = emit_trajectory_rows(
                events, source_kind=kind, project=project, rel_id=rel_id, cfg=tcfg)
            if stats.get("skipped"):
                skip_counts[stats["skipped"]] += 1
                continue
            for k in ("tool_calls_kept", "tool_calls_dropped", "tool_turns",
                      "outputs_truncated", "assistant_with_reasoning",
                      "assistant_turns", "oversize_singletons"):
                if k in stats:
                    agg[k] += stats[k]
            agg["windowed_convs"] += 1 if stats.get("split") else 0
            schema_source_counts[stats.get("schema_source", "?")] += 1
            for r in rows:
                _redact_messages(r["messages"], redactor, redaction_counts)
                # optional real-tokenizer guard: re-window/drop over-budget rows.
                guarded = [r]
                if guard_tok is not None:
                    sys_msg = next((m for m in r["messages"]
                                    if m.get("role") == "system"), None)
                    schema_names = _schema_names_from_system(sys_msg)
                    guarded, gs = guard_row_real_length(
                        r, cfg=tcfg, schema_names=schema_names, tokenizer=guard_tok)
                    for k, v in gs.items():
                        guard_stats[k] += v
                for gr in guarded:
                    for t in gr["metadata"]["tool_names"]:
                        tool_name_counts[t] += 1
                    fout.write(json.dumps(gr, ensure_ascii=False) + "\n")
                    n_rows += 1
                    if len(samples) < args.show:
                        samples.append(gr)

    print("=" * 64)
    print(f"TRAJECTORY MODE — wrote {n_rows:,} rows -> {out_path}")
    print(f"  conversations scanned: {n_convs:,}   files/source: {dict(per_source_files)}")
    print(f"  skipped conversations: {dict(skip_counts)}")
    print(f"  windowed (split) conversations: {agg['windowed_convs']:,}"
          f"   oversize singletons: {agg['oversize_singletons']:,}")
    if guard_tok is not None:
        print(f"  [length-guard] real-tokenizer pass on {guard_stats['checked']:,} "
              f"rows: {guard_stats['retokenized_over']:,} over budget -> "
              f"{guard_stats['rewindowed']:,} re-windowed, "
              f"{guard_stats['dropped_oversize_singleton']:,} dropped "
              f"(unsplittable), {guard_stats['dropped_no_user_fragment']:,} dropped "
              f"(no-user fragment). All written rows now tokenize <= "
              f"{tcfg.max_seq_tokens} on {guard_tok_name}.")
    print(f"  schema source: {dict(schema_source_counts)}")
    print(f"  tool_calls kept/dropped: {agg['tool_calls_kept']:,} / "
          f"{agg['tool_calls_dropped']:,}")
    print(f"  tool turns: {agg['tool_turns']:,}   outputs truncated: "
          f"{agg['outputs_truncated']:,}")
    print(f"  assistant turns w/ reasoning: {agg['assistant_with_reasoning']:,} / "
          f"{agg['assistant_turns']:,}")
    print(f"  tool name histogram: {dict(tool_name_counts.most_common(25))}")
    total_redacted = sum(redaction_counts.values())
    print(f"  secrets redacted: {total_redacted:,}", dict(redaction_counts) or "")
    print("=" * 64)
    for k, s in enumerate(samples):
        roles = [m["role"] for m in s["messages"]]
        print(f"\n--- trajectory sample {k+1} [{s['metadata']['source_kind']}] "
              f"win {s['metadata']['window_index']+1}/{s['metadata']['window_count']} "
              f"tools={s['metadata']['tool_names']} roles={roles} ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to distill config YAML")
    ap.add_argument("--smoke", action="store_true", help="write to smoke/ subdir")
    ap.add_argument("--limit", type=int, default=0, help="cap files per source (0=all)")
    ap.add_argument("--sources", default="", help="comma list to restrict sources")
    ap.add_argument("--show", type=int, default=0, help="print N sample rows")
    ap.add_argument("--max-context-tokens", type=int, default=0,
                    help="override context_budget.max_context_tokens")
    ap.add_argument("--trajectory", action="store_true",
                    help="native tool-trajectory emit (one row per conversation, "
                         "structured tool_calls + real tool outputs) instead of "
                         "per-turn chat rows")
    ap.add_argument("--length-guard-tokenizer", default="",
                    help="HF tokenizer id; when set, re-measure every emitted "
                         "trajectory row on the REAL tokenizer and re-window/drop "
                         "any row over max_seq_tokens (overrides length_guard."
                         "tokenizer in config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    redactor = Redactor(cfg.get("sanitize") or {"enabled": False})
    redaction_counts: Counter = Counter()

    # Native tool-trajectory emit is a wholly separate pipeline (one row per
    # conversation, structured tool_calls + real tool outputs) — branch early.
    if args.trajectory or (cfg.get("trajectory") or {}).get("enabled"):
        run_trajectory(cfg, args, redactor, redaction_counts)
        return

    scope = cfg.get("scope", {})
    lab = cfg["labeling"]
    if args.max_context_tokens:
        cfg.setdefault("context_budget", {})["max_context_tokens"] = args.max_context_tokens
    qcfg = cfg.get("quality_filter") or {"enabled": False}
    q_enabled = qcfg.get("enabled", False)
    oc_cfg = cfg.get("session_outcome") or {"enabled": False}
    oc_enabled = oc_cfg.get("enabled", False)
    tiers_cfg = cfg.get("quality_tiers") or {}
    inherit_parent = tiers_cfg.get("inherit_parent_for_subagents", True)
    require_outcomes = set(qcfg.get("require_outcomes") or [])
    cpt = cfg.get("context_budget", {}).get("chars_per_token", 4.0)
    capture_reasoning = (cfg.get("reasoning") or {}).get("capture", False)
    fb_cfg = cfg.get("feedback") or {"enabled": False}
    dom_cfg = cfg.get("domain") or {"enabled": True}

    seen: set = set()
    drop_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    tier_counts: Counter = Counter()
    feedback_counts: Counter = Counter()
    domain_counts: Counter = Counter()
    parent_cache: dict = {}
    only = set(s.strip() for s in args.sources.split(",") if s.strip())

    out_dir = Path(os.path.expanduser(cfg["output"]["dir"]))
    if args.smoke:
        out_dir = out_dir / cfg["output"].get("smoke_subdir", "smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["output"].get("rows_file", "rows.jsonl")

    per_source_files: Counter = Counter()
    stats = {"rows": 0, "accept": 0, "reject": 0, "borderline": 0, "oversize": 0}
    by_source: dict = {}
    samples = []

    with open(out_path, "w") as fout:
        for kind, adapter, path, root in gather_files(cfg["sources"], scope, only):
            if args.limit and per_source_files[kind] >= args.limit:
                continue
            per_source_files[kind] += 1
            try:
                events, signals = adapter.parse(path)
            except Exception as e:
                print(f"  ! parse failed {path}: {e}", file=sys.stderr)
                continue
            project = adapter.project(path, root)
            rel_id = Path(path).name

            outcomes = []
            if oc_enabled:
                clf = path
                parent = adapter.parent_path(path) if inherit_parent else None
                if parent and os.path.exists(parent):
                    clf = parent
                if clf in parent_cache:
                    outcomes = parent_cache[clf]
                else:
                    try:
                        ev2, sig2 = (events, signals) if clf == path else adapter.parse(clf)
                        outcomes = classify_session(ev2, sig2, oc_cfg)["outcomes"]
                    except Exception as e:
                        print(f"  ! outcome scan failed {clf}: {e}", file=sys.stderr)
                        outcomes = []
                    parent_cache[clf] = outcomes
            tier = derive_tier(outcomes, tiers_cfg)

            rows = emit_rows(events, source_kind=kind, project=project, rel_id=rel_id,
                             lab=lab, ctx_budget=cfg.get("context_budget", {}),
                             capture_reasoning=capture_reasoning, fb=fb_cfg,
                             dom=dom_cfg)
            bs = by_source.setdefault(kind, {"accept": 0, "reject": 0, "borderline": 0})
            for r in rows:
                r["metadata"]["session_outcome"] = outcomes
                r["metadata"]["good_path"] = bool(outcomes)
                r["metadata"]["quality_tier"] = tier
                r["prompt"] = redactor.redact(r["prompt"], redaction_counts)
                r["completion"] = redactor.redact(r["completion"], redaction_counts)
                if "reasoning" in r:
                    r["reasoning"] = redactor.redact(r["reasoning"], redaction_counts)
                if q_enabled:
                    keep, reason = should_keep(r, qcfg, seen, cpt)
                    if not keep:
                        drop_counts[reason] += 1
                        continue
                if require_outcomes and not (set(outcomes) & require_outcomes):
                    drop_counts["no_good_outcome"] += 1
                    continue
                for o2 in (outcomes or ["(none)"]):
                    outcome_counts[o2] += 1
                tier_counts[tier] += 1
                feedback_counts[r["metadata"]["user_feedback"]] += 1
                domain_counts[r["metadata"]["domain"]] += 1
                for m in r["conversations"]:
                    m["content"] = redactor.redact(m["content"], redaction_counts)
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                stats["rows"] += 1
                bucket = ("accept" if r["label"] is True else
                          "reject" if r["label"] is False else "borderline")
                stats[bucket] += 1
                bs[bucket] += 1
                if r["metadata"]["oversize"]:
                    stats["oversize"] += 1
                if len(samples) < args.show and r["completion"].strip():
                    samples.append(r)

    print("=" * 64)
    dropped = sum(drop_counts.values())
    if dropped:
        print(f"quality funnel dropped {dropped:,} rows: {dict(drop_counts)}")
    print(f"wrote {stats['rows']:,} rows -> {out_path}")
    print(f"  files per source: {dict(per_source_files)}")
    print(f"  ACCEPT {stats['accept']:,} | REJECT {stats['reject']:,} | "
          f"BORDERLINE {stats['borderline']:,} (-> judge)")
    for k, b in by_source.items():
        print(f"    {k:18s}: {b['accept']:,} / {b['reject']:,} / {b['borderline']:,}")
    print(f"  oversize rows: {stats['oversize']:,}")
    print(f"  session outcome (turn counts): {dict(outcome_counts)}")
    print(f"  QUALITY TIER: {dict(tier_counts)}")
    print(f"  USER FEEDBACK: {dict(feedback_counts)}")
    print(f"  DOMAIN: {dict(domain_counts)}")
    total_redacted = sum(redaction_counts.values())
    print(f"  secrets redacted: {total_redacted:,}", dict(redaction_counts) or "")
    print("=" * 64)
    for k, s in enumerate(samples):
        print(f"\n--- sample {k+1} [{s['metadata']['source_kind']} / {s['score_tier']} / "
              f"{s['metadata']['quality_tier']}] tools={s['metadata']['tool_names']} ---")
        print("PROMPT   :", (s["prompt"] or "")[:160].replace("\n", " "))
        print("COMPLETE :", s["completion"][:240].replace("\n", " "))


if __name__ == "__main__":
    main()
