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


def emit_rows(events, *, source_kind, project, rel_id, lab, ctx_budget):
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

        rows.append({
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
                "est_tokens": used + est(completion_str),
                "oversize": oversize,
            },
        })
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
        for path in glob.glob(os.path.join(root, sc["glob"]), recursive=True):
            try:
                if not scope_ok(adapter.scope_key(path, root), scope):
                    continue
            except Exception:
                continue
            yield kind, adapter, path, root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to distill config YAML")
    ap.add_argument("--smoke", action="store_true", help="write to smoke/ subdir")
    ap.add_argument("--limit", type=int, default=0, help="cap files per source (0=all)")
    ap.add_argument("--sources", default="", help="comma list to restrict sources")
    ap.add_argument("--show", type=int, default=0, help="print N sample rows")
    ap.add_argument("--max-context-tokens", type=int, default=0,
                    help="override context_budget.max_context_tokens")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scope = cfg.get("scope", {})
    lab = cfg["labeling"]
    if args.max_context_tokens:
        cfg.setdefault("context_budget", {})["max_context_tokens"] = args.max_context_tokens
    redactor = Redactor(cfg.get("sanitize") or {"enabled": False})
    redaction_counts: Counter = Counter()
    qcfg = cfg.get("quality_filter") or {"enabled": False}
    q_enabled = qcfg.get("enabled", False)
    oc_cfg = cfg.get("session_outcome") or {"enabled": False}
    oc_enabled = oc_cfg.get("enabled", False)
    tiers_cfg = cfg.get("quality_tiers") or {}
    inherit_parent = tiers_cfg.get("inherit_parent_for_subagents", True)
    require_outcomes = set(qcfg.get("require_outcomes") or [])
    cpt = cfg.get("context_budget", {}).get("chars_per_token", 4.0)

    seen: set = set()
    drop_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    tier_counts: Counter = Counter()
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
                             lab=lab, ctx_budget=cfg.get("context_budget", {}))
            bs = by_source.setdefault(kind, {"accept": 0, "reject": 0, "borderline": 0})
            for r in rows:
                r["metadata"]["session_outcome"] = outcomes
                r["metadata"]["good_path"] = bool(outcomes)
                r["metadata"]["quality_tier"] = tier
                r["prompt"] = redactor.redact(r["prompt"], redaction_counts)
                r["completion"] = redactor.redact(r["completion"], redaction_counts)
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
