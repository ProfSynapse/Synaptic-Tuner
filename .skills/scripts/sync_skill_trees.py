#!/usr/bin/env python3
"""Sync canonical .skills content into .agents/skills and .claude/skills."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _skill_dirs(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "SKILL.md").exists()
    }


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _compare_dirs(src: Path, dst: Path) -> list[str]:
    issues: list[str] = []
    if not dst.exists():
        issues.append(f"missing target directory: {dst}")
        return issues

    comparison = filecmp.dircmp(src, dst)
    for name in sorted(comparison.left_only):
        issues.append(f"missing in target {dst}: {name}")
    for name in sorted(comparison.right_only):
        issues.append(f"extra in target {dst}: {name}")
    for name in sorted(comparison.diff_files):
        issues.append(f"file differs in target {dst}: {name}")
    for name in sorted(comparison.funny_files):
        issues.append(f"uncomparable file in target {dst}: {name}")
    for child in sorted(comparison.common_dirs):
        issues.extend(_compare_dirs(src / child, dst / child))
    return issues


def sync_skill_tree(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    source_skills = _skill_dirs(source_root)
    target_skills = _skill_dirs(target_root)

    for stale in sorted(target_skills - source_skills):
        shutil.rmtree(target_root / stale)

    for skill_name in sorted(source_skills):
        _copy_tree(source_root / skill_name, target_root / skill_name)


def check_skill_tree(source_root: Path, target_root: Path) -> list[str]:
    issues: list[str] = []
    source_skills = _skill_dirs(source_root)
    target_skills = _skill_dirs(target_root)

    for stale in sorted(target_skills - source_skills):
        issues.append(f"extra skill in {target_root}: {stale}")
    for missing in sorted(source_skills - target_skills):
        issues.append(f"missing skill in {target_root}: {missing}")
    for skill_name in sorted(source_skills & target_skills):
        issues.extend(_compare_dirs(source_root / skill_name, target_root / skill_name))
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync canonical .skills into .agents/skills and .claude/skills.")
    parser.add_argument("--check", action="store_true", help="Check for drift instead of copying files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _repo_root()
    source_root = repo_root / ".skills"
    targets = [
        repo_root / ".agents" / "skills",
        repo_root / ".claude" / "skills",
    ]

    if args.check:
        issues: list[str] = []
        for target in targets:
            issues.extend(check_skill_tree(source_root, target))
        if issues:
            for issue in issues:
                print(issue)
            return 1
        print("Skill trees are in sync.")
        return 0

    for target in targets:
        sync_skill_tree(source_root, target)
        print(f"Synced {source_root} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
