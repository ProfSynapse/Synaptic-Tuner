#!/usr/bin/env python3
"""
Build a decision-training corpus from public classification datasets.

Location: Trainers/decision/build_corpus.py
Purpose:  Config-driven wrapper over the pinned ``strands-decider`` corpus
          builder (``strands-decider data build``), which converts ~28 public
          Hugging Face classification datasets into (state, question, answer)
          rows. We reuse it rather than reimplement its dataset recipes; its
          JSONL row shape is our ``decision-row/v1`` (see decision_core/examples.py).

          After the build it:
            - derives filtered files (e.g. the held-out set without RuleTaker),
            - hashes every output and compares with ``expected_sha256`` from the
              config (the published strands reproduction hashes),
            - writes ``corpus_manifest.json`` (counts per task/kind, hashes,
              builder version, exact argv) next to the data.

          The builder imports torch, so run it where the training stack lives
          (the decision local-run recipe image), not on a bare host:
            python tuner.py local-run --job-config Trainers/recipes/decision_strands_corpus_build.yaml --yes

Usage:
    python Trainers/decision/build_corpus.py --config Trainers/decision/configs/corpus_strands_v5.yaml --dry-run
    python Trainers/decision/build_corpus.py --config Trainers/decision/configs/corpus_strands_v5.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = ("builder", "output_dir", "train_file", "recipes")
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")
    if cfg["builder"] != "strands-decider":
        raise ValueError(f"{path}: unsupported builder {cfg['builder']!r} (only 'strands-decider')")
    return cfg


def build_argv(cfg: dict[str, Any], out_dir: Path) -> list[str]:
    argv = [
        sys.executable, "-c", "from strands_decider.cli import app; app()",
        "data", "build",
        "--out", str(out_dir / cfg["train_file"]),
        "--max-options", str(int(cfg.get("max_options", 24))),
        "--max-chars", str(int(cfg.get("max_chars", 2000))),
        "--seed", str(int(cfg.get("seed", 0))),
    ]
    if cfg.get("max_examples_per_recipe"):
        argv += ["--max-examples", str(int(cfg["max_examples_per_recipe"]))]
    for name in cfg["recipes"]:
        argv += ["-r", str(name)]
    for name in cfg.get("holdout") or []:
        argv += ["--holdout", str(name)]
    return argv


def holdout_path(out_dir: Path, train_file: str) -> Path:
    """strands-decider writes held-out tasks to <stem>.holdout.jsonl beside the output."""
    stem = train_file[: -len(".jsonl")] if train_file.endswith(".jsonl") else train_file
    return out_dir / f"{stem}.holdout.jsonl"


def derive_filtered(src: Path, dst: Path, exclude_tasks: list[str]) -> int:
    """Copy src to dst keeping each kept line's original bytes, so hashes are reproducible."""
    exclude = set(exclude_tasks)
    kept = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for line in fin:
            if not line.strip():
                continue
            if json.loads(line).get("task") in exclude:
                continue
            fout.write(line)
            kept += 1
    return kept


def describe(path: Path) -> dict[str, Any]:
    tasks: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    rows = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks[row.get("task", "unknown")] += 1
            kinds[row.get("kind", "?")] += 1
            rows += 1
    return {"rows": rows, "kinds": dict(kinds), "tasks": dict(sorted(tasks.items())),
            "sha256": sha256_file(path)}


def builder_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("strands-decider")
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="Print the builder command and exit.")
    ap.add_argument("--skip-build", action="store_true",
                    help="Reuse existing builder output; only derive files, hash and write the manifest.")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    out_dir = Path(cfg["output_dir"])
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    cmd = build_argv(cfg, out_dir)

    print("Builder command:")
    print("  " + " ".join(cmd))
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        installed = builder_version()
        pinned = cfg.get("builder_version")
        if installed is None:
            raise SystemExit("strands-decider is not installed; the recipe's setup.pip installs it")
        if pinned and installed != str(pinned):
            raise SystemExit(f"strands-decider {installed} installed but the config pins {pinned}")
        result = subprocess.run(cmd, cwd=str(out_dir))
        if result.returncode != 0:
            raise SystemExit(f"strands-decider data build failed with exit code {result.returncode}")

    outputs: dict[str, Path] = {cfg["train_file"]: out_dir / cfg["train_file"]}
    hold = holdout_path(out_dir, cfg["train_file"])
    if hold.exists():
        outputs[hold.name] = hold
    for derived in cfg.get("derived") or []:
        src = out_dir / derived["from"]
        dst = out_dir / derived["file"]
        n = derive_filtered(src, dst, list(derived.get("exclude_tasks") or []))
        print(f"Derived {dst.name}: {n:,} rows")
        outputs[dst.name] = dst

    expected = cfg.get("expected_sha256") or {}
    strict = str(cfg.get("verify_hashes", "warn")).lower() == "strict"
    files: dict[str, Any] = {}
    mismatches: list[str] = []
    for name, path in outputs.items():
        info = describe(path)
        want = expected.get(name)
        info["expected_sha256"] = want
        info["hash_match"] = None if want is None else info["sha256"] == want
        if want is not None and not info["hash_match"]:
            mismatches.append(name)
        files[name] = info
        status = "n/a" if want is None else ("match" if info["hash_match"] else "MISMATCH")
        print(f"  {name:<28} {info['rows']:>8,} rows  sha256 {info['sha256'][:12]}…  [{status}]")

    manifest = {
        "schema": "decision-corpus-manifest/v1",
        "config": str(cfg_path),
        "builder": cfg["builder"],
        "builder_version": builder_version(),
        "argv": cmd[3:],
        "built_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (out_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {out_dir / 'corpus_manifest.json'}")

    if mismatches:
        msg = (f"hash mismatch for {mismatches}: the upstream Hugging Face datasets load without a "
               "pinned revision, so a changed source changes the corpus")
        if strict:
            raise SystemExit(msg)
        print(f"WARNING: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
