#!/usr/bin/env python3
"""Fetch and summarize Hugging Face Jobs hardware and pricing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


DEFAULT_URL = "https://huggingface.co/api/jobs/hardware"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List Hugging Face Jobs hardware and pricing.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Hardware API URL.")
    parser.add_argument("--json", action="store_true", help="Print raw API payload.")
    parser.add_argument("--job-config", help="Optional cloud job YAML to highlight the current flavor.")
    parser.add_argument(
        "--sort-by",
        choices=("price", "vram", "flavor"),
        default="price",
        help="Sort output rows.",
    )
    parser.add_argument("--min-vram", type=float, default=0.0, help="Filter rows below this VRAM (GB).")
    parser.add_argument("--max-price", type=float, default=0.0, help="Filter rows above this hourly price.")
    return parser.parse_args(argv)


def fetch_hardware_payload(url: str) -> Any:
    headers = {"Accept": "application/json"}
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_rows(payload: Any) -> List[Dict[str, Any]]:
    candidates: Iterable[Any]
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        for key in ("hardware", "flavors", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break
        else:
            candidates = []
    else:
        candidates = []

    rows: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        row = {
            "flavor": _first_str(item, "flavor", "id", "name", "sku"),
            "gpu": _first_str(item, "gpu", "gpu_name", "accelerator", "label", "display_name"),
            "gpus": _first_num(item, "gpus", "gpu_count", "count", "quantity", default=1),
            "vram_gb": _first_num(
                item,
                "vram_gb",
                "gpu_memory_gb",
                "gpuMemoryGb",
                "memory_gb",
                "memoryGb",
                "vram",
                default=0.0,
            ),
            "price_hr": _first_num(
                item,
                "price_hr",
                "hourly_price",
                "hourlyPrice",
                "hourlyPriceUsd",
                "price",
                "priceUsd",
                default=0.0,
            ),
            "cpu": _first_num(item, "cpu", "cpus", "vcpu", "vcpus", default=0),
            "ram_gb": _first_num(item, "ram_gb", "ramGb", "memory", default=0.0),
            "raw": item,
        }
        if row["flavor"]:
            rows.append(row)
    return rows


def load_job_flavor(job_config_path: Optional[str]) -> Optional[str]:
    if not job_config_path:
        return None
    path = Path(job_config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        return None
    job_cfg = config.get("job") or {}
    if not isinstance(job_cfg, dict):
        return None
    flavor = job_cfg.get("flavor")
    return str(flavor).strip() if flavor else None


def filter_and_sort_rows(
    rows: List[Dict[str, Any]],
    *,
    min_vram: float,
    max_price: float,
    sort_by: str,
) -> List[Dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if float(row.get("vram_gb") or 0.0) >= min_vram
        and (max_price <= 0.0 or float(row.get("price_hr") or 0.0) <= max_price)
    ]

    if sort_by == "vram":
        filtered.sort(key=lambda row: (float(row.get("vram_gb") or 0.0), str(row.get("flavor") or "")))
    elif sort_by == "flavor":
        filtered.sort(key=lambda row: str(row.get("flavor") or ""))
    else:
        filtered.sort(key=lambda row: (float(row.get("price_hr") or 0.0), str(row.get("flavor") or "")))
    return filtered


def print_table(rows: List[Dict[str, Any]], *, current_flavor: Optional[str]) -> None:
    headers = ["Use", "Flavor", "GPU", "GPUs", "VRAM(GB)", "RAM(GB)", "Price/hr"]
    body: List[List[str]] = []
    for row in rows:
        marker = "*" if current_flavor and row["flavor"] == current_flavor else ""
        body.append(
            [
                marker,
                str(row.get("flavor") or ""),
                str(row.get("gpu") or ""),
                _fmt_num(row.get("gpus")),
                _fmt_num(row.get("vram_gb")),
                _fmt_num(row.get("ram_gb")),
                _fmt_price(row.get("price_hr")),
            ]
        )

    widths = [len(header) for header in headers]
    for line in body:
        for index, value in enumerate(line):
            widths[index] = max(widths[index], len(value))

    def _render(columns: List[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(columns))

    print(_render(headers))
    print(_render(["-" * width for width in widths]))
    for line in body:
        print(_render(line))

    if current_flavor:
        print(f"\n* current job flavor from config: {current_flavor}")


def _first_str(item: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_num(item: Dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip().replace("$", "")
            try:
                return float(stripped)
            except ValueError:
                continue
    return float(default)


def _fmt_num(value: Any) -> str:
    if value is None:
        return ""
    num = float(value)
    if num.is_integer():
        return str(int(num))
    return f"{num:.1f}"


def _fmt_price(value: Any) -> str:
    num = float(value or 0.0)
    return f"${num:.2f}" if num else ""


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        payload = fetch_hardware_payload(args.url)
    except urllib.error.HTTPError as exc:
        print(f"HTTP error fetching hardware list: {exc}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Network error fetching hardware list: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    rows = normalize_rows(payload)
    current_flavor = load_job_flavor(args.job_config)
    rows = filter_and_sort_rows(
        rows,
        min_vram=args.min_vram,
        max_price=args.max_price,
        sort_by=args.sort_by,
    )

    if not rows:
        print("No hardware rows matched the current filters.", file=sys.stderr)
        return 1

    print_table(rows, current_flavor=current_flavor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
