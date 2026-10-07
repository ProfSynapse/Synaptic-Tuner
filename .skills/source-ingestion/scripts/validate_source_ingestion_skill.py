#!/usr/bin/env python3
"""Validate the exact project-local source-ingestion skill tree.

The check enforces the approved inventory, slim router, protocol chaining,
reference links, one-recipe boundary, and explicit no-package delivery. It does
not synchronize mirrors or package the skill.

Usage:
  python validate_source_ingestion_skill.py SKILL_ROOT

Exit codes:
  0  valid
  1  violations found
  2  usage error
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


EXPECTED = {
    "SKILL.md",
    "skill-spec.md",
    "protocols/ingest-local-markdown.md",
    "protocols/configure-markdown-recipe.md",
    "protocols/execute-ingestion.md",
    "protocols/diagnose-ingestion.md",
    "protocols/verify-and-handoff.md",
    "protocols/self-refine.md",
    "protocols/validate-and-sync.md",
    "protocols/deliver-project-local.md",
    "references/supported-surfaces.md",
    "references/ingestion-cli-config.md",
    "references/markdown-yaml-frontmatter.md",
    "references/lifecycle-and-diagnostics.md",
    "references/privacy-determinism-and-bundles.md",
    "references/dataset-handoff.md",
    "references/refinement-log.md",
    "templates/markdown-frontmatter-ingestion.json",
    "templates/ingestion-checkpoint.md",
    "scripts/validate_ingestion_config.py",
    "scripts/validate_ingestion_result.py",
    "scripts/validate_source_ingestion_skill.py",
}
SUBDIRS = {"protocols", "references", "templates", "scripts"}
LINK = re.compile(r"`((?:\.\.?/)?(?:protocols|references|templates|scripts)/[^`]+)`")


def _inventory(root: Path) -> set[str]:
    found = {item.name for item in root.iterdir() if item.is_file()}
    for directory in SUBDIRS:
        child = root / directory
        if not child.is_dir():
            continue
        found.update(
            f"{directory}/{item.name}" for item in child.iterdir() if item.is_file()
        )
        found.update(
            f"{directory}/{item.name}/" for item in child.iterdir() if item.is_dir()
        )
    found.update(f"{item.name}/" for item in root.iterdir() if item.is_dir() and item.name not in SUBDIRS)
    return found


def check(root: Path) -> list[str]:
    violations: list[str] = []
    found = _inventory(root)
    for missing in sorted(EXPECTED - found):
        violations.append(f"{root / missing}: missing required file")
    for extra in sorted(found - EXPECTED):
        violations.append(f"{root / extra}: unexpected skill member")
    if violations:
        return violations

    texts: dict[str, str] = {}
    for relative in sorted(EXPECTED):
        path = root / relative
        try:
            texts[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            violations.append(f"{path}: must be readable UTF-8")
    if violations:
        return violations

    router = texts["SKILL.md"]
    if len(router.splitlines()) > 80:
        violations.append(f"{root / 'SKILL.md'}: router exceeds 80 lines")
    if "name: source-ingestion" not in router or "## Workflow" not in router:
        violations.append(f"{root / 'SKILL.md'}: invalid frontmatter or workflow")

    for relative, content in texts.items():
        if relative.startswith("protocols/"):
            if "## Steps" not in content or "## Next" not in content:
                violations.append(f"{root / relative}: protocol lacks Steps or Next")
        if relative.endswith(".md"):
            for match in LINK.finditer(content):
                candidate = (root / relative).parent / match.group(1)
                if not candidate.resolve().is_file():
                    violations.append(
                        f"{root / relative}: unresolved local link {match.group(1)}"
                    )

    combined = "\n".join(texts.values()).lower()
    if "markdown with optional yaml frontmatter" not in combined:
        violations.append(f"{root}: sole recipe is not stated explicitly")
    delivery = texts["protocols/deliver-project-local.md"]
    if "never a standalone `.skill` artifact" not in delivery:
        violations.append(f"{root / 'protocols/deliver-project-local.md'}: no-package exception missing")
    if "python .skills/scripts/sync_skill_trees.py --check" not in texts["protocols/validate-and-sync.md"]:
        violations.append(f"{root / 'protocols/validate-and-sync.md'}: mirror sync check missing")
    if "tuner.ingestion" in texts["scripts/validate_ingestion_config.py"]:
        violations.append(f"{root / 'scripts/validate_ingestion_config.py'}: private runtime import forbidden")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill_root", type=Path)
    args = parser.parse_args()
    if not args.skill_root.is_dir():
        print("error: skill_root must be an existing directory", file=sys.stderr)
        return 2
    violations = check(args.skill_root)
    for violation in violations:
        print(violation)
    if violations:
        print(f"INVALID: {len(violations)} violation(s)")
        return 1
    print("VALID: canonical source-ingestion skill tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
