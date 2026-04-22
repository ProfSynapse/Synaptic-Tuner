#!/usr/bin/env python3
"""Merge regenerated seed outputs into new latest per-agent dataset versions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from cli_schema_utils import next_version_path
from utils import read_jsonl


DEFAULT_BASE_DATASETS = {
    "contentManager": "Datasets/tools_datasets/non_thinking/contentManager/tools_v2.5.jsonl",
    "memoryManager": "Datasets/tools_datasets/non_thinking/memoryManager/tools_v2.5.jsonl",
    "promptManager": "Datasets/tools_datasets/non_thinking/promptManager/tools_v2.7.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regen-dir",
        default="Datasets/tools_datasets/reports/cli_schema/regen_outputs",
        help="Directory containing regenerated per-tool output JSONL files",
    )
    parser.add_argument(
        "--base",
        action="append",
        default=[],
        help="Override base dataset path as agent=path",
    )
    return parser.parse_args()


def parse_overrides(values: List[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --base override: {value}")
        agent, raw_path = value.split("=", 1)
        overrides[agent] = Path(raw_path)
    return overrides


def load_jsonl(path: Path) -> List[Dict]:
    return read_jsonl(path)


def load_passed_rows(path: Path) -> List[Dict]:
    rows = load_jsonl(path)
    report_path = Path(f"{path}.improve_report.json")
    if not report_path.exists():
        return rows

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    passed_line_numbers = {
        int(item["line_number"])
        for item in report.get("results", [])
        if item.get("passed") is True
    }
    if not passed_line_numbers:
        return []
    return [row for idx, row in enumerate(rows, start=1) if idx in passed_line_numbers]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    regen_dir = Path(args.regen_dir)
    overrides = parse_overrides(args.base)

    base_paths = {agent: Path(path) for agent, path in DEFAULT_BASE_DATASETS.items()}
    base_paths.update(overrides)

    regen_mapping = {
        "contentManager": regen_dir / "contentManager_replace_regenerated.jsonl",
        "memoryManager": regen_dir / "memoryManager_updateWorkspace_regenerated.jsonl",
        "promptManager": regen_dir / "promptManager_executePrompts_regenerated.jsonl",
    }

    manifest = {}
    for agent, base_path in base_paths.items():
        regen_path = regen_mapping[agent]
        if not regen_path.exists():
            raise FileNotFoundError(f"Missing regenerated file for {agent}: {regen_path}")

        base_rows = load_jsonl(base_path)
        regen_rows = load_passed_rows(regen_path)
        output_path = next_version_path(base_path)
        merged_rows = list(base_rows) + list(regen_rows)
        write_jsonl(output_path, merged_rows)
        manifest[agent] = {
            "base_path": str(base_path),
            "regen_path": str(regen_path),
            "output_path": str(output_path),
            "base_count": len(base_rows),
            "regen_count": len(regen_rows),
            "output_count": len(merged_rows),
        }
        print(
            f"{agent}: base={len(base_rows)} regen={len(regen_rows)} "
            f"output={len(merged_rows)} -> {output_path}"
        )

    manifest_path = regen_dir / "merge_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"Wrote merge manifest: {manifest_path}")


if __name__ == "__main__":
    main()
