#!/usr/bin/env python3
"""Prune dataset rows using finished per-example loss artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "shared").exists() and (parent / "Datasets").exists() and (parent / ".skills").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.utilities.env import get_hf_token, load_env_file  # noqa: E402


STRATEGIES = {
    "loss_threshold": {
        "description": "Drop any example with loss >= min_loss.",
        "default_min_loss": 2.0,
    },
    "top_percent": {
        "description": "Drop the worst top_percent examples by loss.",
        "default_top_percent": 0.02,
    },
    "result_echo_conservative": {
        "description": "Repo-specific preset: drop text_only result-echo examples with loss >= 2.5.",
        "default_min_loss": 2.5,
    },
    "result_echo_linesdelta_recommendations": {
        "description": "Repo-specific preset: drop text_only result-echo examples with linesDelta/recommendations and loss >= 2.0.",
        "default_min_loss": 2.0,
    },
    "result_echo_plus_roleplay": {
        "description": "Repo-specific preset: drop the recommended result-echo family plus roleplay-style recap rows with loss >= 2.0.",
        "default_min_loss": 2.0,
    },
    "result_echo_doneish": {
        "description": "Repo-specific preset: drop the broader result-echo family including Done./All./Open. recap rows with loss >= 2.0.",
        "default_min_loss": 2.0,
    },
    "result_echo_long_recap": {
        "description": "Repo-specific preset: drop result-echo rows with long assistant recap/narration and loss >= 2.0.",
        "default_min_loss": 2.0,
    },
    "result_echo_broad": {
        "description": "Repo-specific preset: drop any text_only result-echo example with loss >= 2.0.",
        "default_min_loss": 2.0,
    },
}


@dataclass(frozen=True)
class PruneDecision:
    should_prune: bool
    reasons: list[str]


@dataclass(frozen=True)
class PruneResult:
    dataset_path: Path
    output_path: Path
    removed_path: Path
    metadata_path: Path
    total_rows: int
    kept_rows: int
    removed_rows: int
    strategy: str
    min_loss: float
    loss_source: str
    publish_repo: Optional[str] = None


@dataclass(frozen=True)
class AnalysisResult:
    dataset_path: Path
    loss_source: str
    total_rows: int
    loss_summary: dict[str, Any]
    top_slice_count: int
    feature_report: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]


def _normalize_token_value(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    token = token.strip()
    return token or None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _open_text(path_or_uri: str):
    if path_or_uri.startswith("hf://"):
        load_env_file()
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=_normalize_token_value(get_hf_token()))
        return fs.open(path_or_uri, "r")
    return Path(path_or_uri).open("r", encoding="utf-8")


def load_loss_rows(path_or_uri: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _open_text(path_or_uri) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_loss_path_from_experiment(experiment_id: str, *, repo_root: Path = REPO_ROOT) -> str:
    experiment_path = repo_root / ".tracking" / "experiments" / experiment_id / "experiment.json"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_path}")
    payload = json.loads(experiment_path.read_text(encoding="utf-8"))
    artifact_root = (
        payload.get("artifact_roots", {}).get("loss")
        or payload.get("stage_details", {}).get("loss", {}).get("artifact_root")
    )
    if not artifact_root:
        raise ValueError(f"Experiment {experiment_id} does not have a loss artifact root.")
    return f"{artifact_root.rstrip('/')}/per_example_losses.jsonl"


def build_default_output_paths(dataset_path: Path, strategy: str) -> tuple[Path, Path, Path]:
    date_suffix = datetime.now().strftime("%m.%d.%y")
    stem = dataset_path.name[:-6] if dataset_path.name.endswith(".jsonl") else dataset_path.stem
    output_path = dataset_path.with_name(f"{stem}_pruned_{strategy}_{date_suffix}.jsonl")
    removed_path = dataset_path.with_name(f"{stem}_removed_{strategy}_{date_suffix}.jsonl")
    metadata_path = dataset_path.with_name(f"{stem}_pruned_{strategy}_{date_suffix}.metadata.json")
    return output_path, removed_path, metadata_path


def _first_message_content(row: dict[str, Any], *, role: str) -> str:
    for message in row.get("conversations", []):
        if message.get("role") == role:
            content = message.get("content", "")
            return "" if content is None else str(content)
    return ""


def classify_example(row: dict[str, Any]) -> dict[str, bool]:
    user_text = _first_message_content(row, role="user")
    assistant_text = _first_message_content(row, role="assistant")
    pattern = row.get("pattern")
    return {
        "pattern_text_only": pattern == "text_only",
        "pattern_tool_text": pattern == "tool_text",
        "pattern_tool_only": pattern == "tool_only",
        "text_only_result": pattern == "text_only" and user_text.startswith("Result:"),
        "linesdelta": "linesDelta" in user_text,
        "recommendations": "recommendations" in user_text,
        "batchresults": '"results"' in user_text,
        "opened": '"opened": true' in user_text,
        "long_recap": len(assistant_text.split()) > 80,
        "doneish": assistant_text.startswith("Done.")
        or assistant_text.startswith("Done!")
        or assistant_text.startswith("All ")
        or assistant_text.startswith("Open."),
        "roleplay": assistant_text.startswith("🧙"),
    }


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.floor((len(ordered) - 1) * fraction)))
    return ordered[index]


def analyze_dataset_against_loss(
    *,
    dataset_path: str | Path,
    loss_source: str,
    top_percent: float = 0.05,
) -> AnalysisResult:
    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.exists() or not dataset.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset}")
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be within (0, 1].")

    dataset_rows = _read_jsonl(dataset)
    loss_rows = load_loss_rows(loss_source)
    if len(dataset_rows) != len(loss_rows):
        raise ValueError(
            f"Dataset row count ({len(dataset_rows)}) does not match loss row count ({len(loss_rows)})."
        )

    indexed = []
    losses = []
    feature_counts = Counter()
    feature_loss_buckets: dict[str, list[float]] = {}
    for loss_row in loss_rows:
        index = int(loss_row["index"])
        loss = float(loss_row["loss"])
        row = dataset_rows[index]
        flags = classify_example(row)
        indexed.append((index, loss, flags))
        losses.append(loss)
        for feature, enabled in flags.items():
            if enabled:
                feature_counts[feature] += 1
                feature_loss_buckets.setdefault(feature, []).append(loss)

    indexed.sort(key=lambda item: item[1], reverse=True)
    top_slice_count = max(1, math.ceil(len(indexed) * top_percent))
    top_slice = indexed[:top_slice_count]
    top_counts = Counter()
    for _, _, flags in top_slice:
        for feature, enabled in flags.items():
            if enabled:
                top_counts[feature] += 1

    feature_report = []
    for feature, total_count in sorted(feature_counts.items()):
        if total_count == 0:
            continue
        top_count = top_counts.get(feature, 0)
        overall_rate = total_count / len(indexed)
        top_rate = top_count / top_slice_count
        feature_losses = sorted(feature_loss_buckets[feature])
        feature_report.append(
            {
                "feature": feature,
                "count": total_count,
                "top_slice_count": top_count,
                "overall_rate": round(overall_rate, 4),
                "top_slice_rate": round(top_rate, 4),
                "enrichment": round((top_rate / overall_rate), 4) if overall_rate else None,
                "mean_loss": round(sum(feature_losses) / len(feature_losses), 4),
                "p90_loss": round(_quantile(feature_losses, 0.9), 4),
                "p95_loss": round(_quantile(feature_losses, 0.95), 4),
                "max_loss": round(feature_losses[-1], 4),
            }
        )

    feature_report.sort(
        key=lambda item: (
            item["enrichment"] or 0.0,
            item["mean_loss"],
            item["count"],
        ),
        reverse=True,
    )

    suggestions = []
    if any(item["feature"] == "text_only_result" and item["enrichment"] and item["enrichment"] >= 2.0 for item in feature_report):
        suggestions.append(
            {
                "strategy": "result_echo_linesdelta_recommendations",
                "reason": "High-loss slice is strongly enriched for text_only result-echo recap rows with linesDelta/recommendations.",
            }
        )
    if any(item["feature"] == "long_recap" and item["enrichment"] and item["enrichment"] >= 2.0 for item in feature_report):
        suggestions.append(
            {
                "strategy": "result_echo_long_recap",
                "reason": "High-loss slice is strongly enriched for result-echo rows followed by long assistant recap/narration.",
            }
        )
    suggestions.append(
        {
            "strategy": "loss_threshold",
            "reason": "Generic fallback when no single family is strongly enriched; use with a conservative min_loss threshold.",
        }
    )
    suggestions.append(
        {
            "strategy": "top_percent",
            "reason": "Generic fallback when you want a simple capped prune budget without hardcoding a family.",
        }
    )

    return AnalysisResult(
        dataset_path=dataset,
        loss_source=loss_source,
        total_rows=len(indexed),
        loss_summary={
            "mean_loss": round(sum(losses) / len(losses), 4) if losses else 0.0,
            "p90_loss": round(_quantile(losses, 0.9), 4),
            "p95_loss": round(_quantile(losses, 0.95), 4),
            "max_loss": round(max(losses), 4) if losses else 0.0,
        },
        top_slice_count=top_slice_count,
        feature_report=feature_report,
        suggestions=suggestions,
    )


def evaluate_prune_decision(
    row: dict[str, Any],
    *,
    loss: float,
    loss_rank: int,
    total_rows: int,
    strategy: str,
    min_loss: Optional[float],
    top_percent: Optional[float],
) -> PruneDecision:
    flags = classify_example(row)
    reasons: list[str] = []
    resolved_min_loss = min_loss if min_loss is not None else STRATEGIES.get(strategy, {}).get("default_min_loss")

    if strategy == "loss_threshold":
        match = resolved_min_loss is not None and loss >= resolved_min_loss
        return PruneDecision(should_prune=bool(match), reasons=[f"loss>={resolved_min_loss:g}"] if match else [])

    if strategy == "top_percent":
        resolved_top_percent = top_percent if top_percent is not None else STRATEGIES[strategy]["default_top_percent"]
        cutoff = max(1, math.ceil(total_rows * resolved_top_percent))
        match = loss_rank < cutoff
        return PruneDecision(should_prune=match, reasons=[f"top_percent={resolved_top_percent:g}", f"loss_rank<{cutoff}"] if match else [])

    if not flags["text_only_result"] or resolved_min_loss is None or loss < resolved_min_loss:
        return PruneDecision(should_prune=False, reasons=[])

    reasons = ["text_only_result", f"loss>={resolved_min_loss:g}"]
    if strategy == "result_echo_conservative":
        return PruneDecision(should_prune=True, reasons=reasons)
    if strategy == "result_echo_linesdelta_recommendations":
        match = flags["linesdelta"] or flags["recommendations"]
        return PruneDecision(
            should_prune=match,
            reasons=reasons + [name for name in ("linesdelta", "recommendations") if flags[name]],
        )
    if strategy == "result_echo_plus_roleplay":
        match = flags["linesdelta"] or flags["recommendations"] or flags["roleplay"]
        return PruneDecision(
            should_prune=match,
            reasons=reasons + [name for name in ("linesdelta", "recommendations", "roleplay") if flags[name]],
        )
    if strategy == "result_echo_doneish":
        match = flags["linesdelta"] or flags["recommendations"] or flags["roleplay"] or flags["doneish"]
        return PruneDecision(
            should_prune=match,
            reasons=reasons + [name for name in ("linesdelta", "recommendations", "roleplay", "doneish") if flags[name]],
        )
    if strategy == "result_echo_long_recap":
        match = flags["long_recap"]
        return PruneDecision(
            should_prune=match,
            reasons=reasons + [name for name in ("long_recap",) if flags[name]],
        )
    if strategy == "result_echo_broad":
        return PruneDecision(should_prune=True, reasons=reasons)
    raise ValueError(f"Unknown strategy: {strategy}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_publish_module():
    skill_path = REPO_ROOT / ".skills" / "dataset-publishing" / "scripts" / "publish_dataset_to_hf.py"
    spec = importlib.util.spec_from_file_location("dataset_publish_skill_script", skill_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load dataset publish script at {skill_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def prune_dataset(
    *,
    dataset_path: str | Path,
    loss_source: str,
    strategy: str,
    min_loss: Optional[float] = None,
    top_percent: Optional[float] = None,
    output_path: str | Path | None = None,
    removed_output_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    publish_repo: Optional[str] = None,
    publish_path_in_repo: Optional[str] = None,
    publish_commit_message: Optional[str] = None,
    publish_private: bool = False,
    publish_dry_run: bool = False,
) -> PruneResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.exists() or not dataset.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset}")

    resolved_min_loss = min_loss if min_loss is not None else STRATEGIES.get(strategy, {}).get("default_min_loss")
    resolved_output, resolved_removed, resolved_metadata = build_default_output_paths(dataset, strategy)
    output = Path(output_path).expanduser().resolve() if output_path else resolved_output
    removed_output = Path(removed_output_path).expanduser().resolve() if removed_output_path else resolved_removed
    metadata_output = Path(metadata_path).expanduser().resolve() if metadata_path else resolved_metadata

    dataset_rows = _read_jsonl(dataset)
    loss_rows = load_loss_rows(loss_source)
    loss_by_index = {int(item["index"]): item for item in loss_rows}
    missing_indices = [index for index in range(len(dataset_rows)) if index not in loss_by_index]
    if missing_indices:
        raise ValueError(
            f"Loss source is missing {len(missing_indices)} dataset row(s); first missing index: {missing_indices[0]}"
        )

    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    removed_summary: list[dict[str, Any]] = []

    ranked = sorted(loss_rows, key=lambda item: float(item["loss"]), reverse=True)
    rank_by_index = {int(item["index"]): rank for rank, item in enumerate(ranked)}

    for index, row in enumerate(dataset_rows):
        loss_row = loss_by_index[index]
        decision = evaluate_prune_decision(
            row,
            loss=float(loss_row["loss"]),
            loss_rank=rank_by_index[index],
            total_rows=len(dataset_rows),
            strategy=strategy,
            min_loss=resolved_min_loss,
            top_percent=top_percent,
        )
        if decision.should_prune:
            removed_rows.append(row)
            removed_summary.append(
                {
                    "index": index,
                    "loss": float(loss_row["loss"]),
                    "jsonl_hash": loss_row.get("jsonl_hash"),
                    "num_completion_tokens": loss_row.get("num_completion_tokens"),
                    "num_total_tokens": loss_row.get("num_total_tokens"),
                    "pattern": row.get("pattern"),
                    "label": row.get("label"),
                    "reasons": decision.reasons,
                    "user_preview": _first_message_content(row, role="user")[:240],
                    "assistant_preview": _first_message_content(row, role="assistant")[:240],
                }
            )
        else:
            kept_rows.append(row)

    _write_jsonl(output, kept_rows)
    _write_jsonl(removed_output, removed_rows)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        "source_dataset_path": str(dataset),
        "loss_source": loss_source,
        "strategy": strategy,
        "strategy_description": STRATEGIES[strategy]["description"],
        "min_loss": resolved_min_loss,
        "top_percent": top_percent,
        "total_rows": len(dataset_rows),
        "kept_rows": len(kept_rows),
        "removed_rows": len(removed_rows),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "output_dataset_path": str(output),
        "removed_dataset_path": str(removed_output),
        "removed_examples_preview": removed_summary[:50],
    }
    metadata_output.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if publish_repo:
        publish_module = _load_publish_module()
        publish_module.publish_dataset(
            output,
            publish_repo,
            path_in_repo=publish_path_in_repo,
            metadata_path=metadata_output,
            include_metadata=True,
            private=publish_private,
            commit_message=publish_commit_message or f"Upload pruned dataset {output.name}",
            dry_run=publish_dry_run,
        )

    return PruneResult(
        dataset_path=dataset,
        output_path=output,
        removed_path=removed_output,
        metadata_path=metadata_output,
        total_rows=len(dataset_rows),
        kept_rows=len(kept_rows),
        removed_rows=len(removed_rows),
        strategy=strategy,
        min_loss=resolved_min_loss,
        loss_source=loss_source,
        publish_repo=publish_repo,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune dataset rows using finished per-example loss artifacts.")
    parser.add_argument("--dataset-path", required=True, help="Path to the local dataset JSONL file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--loss-source", help="Local path or hf:// bucket URI to per_example_losses.jsonl.")
    source.add_argument("--experiment-id", help="Experiment id to resolve the loss artifact automatically.")
    parser.add_argument("--analyze-only", action="store_true", help="Report enriched high-loss families and exit without writing a dataset.")
    parser.add_argument("--analysis-report-path", help="Optional JSON path for the analysis report.")
    parser.add_argument(
        "--analysis-top-percent",
        type=float,
        default=0.05,
        help="Top loss slice to analyze for enrichment. Default: 0.05",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES.keys()),
        help="Pruning strategy. Omit when using --analyze-only.",
    )
    parser.add_argument("--min-loss", type=float, help="Override the strategy default loss threshold.")
    parser.add_argument("--top-percent", type=float, help="Top fraction to prune when using strategy=top_percent.")
    parser.add_argument("--output-path", help="Output path for the filtered dataset JSONL.")
    parser.add_argument("--removed-output-path", help="Optional output path for the removed rows JSONL.")
    parser.add_argument("--metadata-path", help="Optional metadata/report sidecar path.")
    parser.add_argument("--publish-repo", help="Optional HF dataset repo to publish the filtered dataset to.")
    parser.add_argument("--publish-path-in-repo", help="Optional target filename in the HF dataset repo.")
    parser.add_argument("--publish-commit-message", help="Optional HF commit message when publishing.")
    parser.add_argument("--publish-private", action="store_true", help="Create the HF dataset repo as private if needed.")
    parser.add_argument("--publish-dry-run", action="store_true", help="Build outputs locally but do not upload to HF.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    loss_source = args.loss_source or resolve_loss_path_from_experiment(args.experiment_id)
    if args.analyze_only:
        analysis = analyze_dataset_against_loss(
            dataset_path=args.dataset_path,
            loss_source=loss_source,
            top_percent=args.analysis_top_percent,
        )
        report = {
            "dataset_path": str(analysis.dataset_path),
            "loss_source": analysis.loss_source,
            "total_rows": analysis.total_rows,
            "loss_summary": analysis.loss_summary,
            "top_slice_count": analysis.top_slice_count,
            "feature_report": analysis.feature_report,
            "suggestions": analysis.suggestions,
        }
        if args.analysis_report_path:
            path = Path(args.analysis_report_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Dataset: {analysis.dataset_path}")
        print(f"Loss source: {analysis.loss_source}")
        print(f"Rows: {analysis.total_rows}")
        print(f"Loss summary: mean={analysis.loss_summary['mean_loss']:.4f}, p95={analysis.loss_summary['p95_loss']:.4f}, max={analysis.loss_summary['max_loss']:.4f}")
        print(f"Top slice analyzed: {analysis.top_slice_count} row(s) ({args.analysis_top_percent:.1%})")
        print("Top enriched families:")
        for item in analysis.feature_report[:10]:
            print(
                f"  - {item['feature']}: enrichment={item['enrichment']}, mean_loss={item['mean_loss']}, "
                f"top_slice={item['top_slice_count']}/{analysis.top_slice_count}, overall={item['count']}/{analysis.total_rows}"
            )
        print("Suggested strategies:")
        for suggestion in analysis.suggestions:
            print(f"  - {suggestion['strategy']}: {suggestion['reason']}")
        return 0

    if not args.strategy:
        parser.error("--strategy is required unless --analyze-only is set.")

    result = prune_dataset(
        dataset_path=args.dataset_path,
        loss_source=loss_source,
        strategy=args.strategy,
        min_loss=args.min_loss,
        top_percent=args.top_percent,
        output_path=args.output_path,
        removed_output_path=args.removed_output_path,
        metadata_path=args.metadata_path,
        publish_repo=args.publish_repo,
        publish_path_in_repo=args.publish_path_in_repo,
        publish_commit_message=args.publish_commit_message,
        publish_private=args.publish_private,
        publish_dry_run=args.publish_dry_run,
    )

    print(f"Dataset: {result.dataset_path}")
    print(f"Loss source: {result.loss_source}")
    print(f"Strategy: {result.strategy} (min_loss={result.min_loss:g})")
    print(f"Kept rows: {result.kept_rows}")
    print(f"Removed rows: {result.removed_rows}")
    print(f"Filtered dataset: {result.output_path}")
    print(f"Removed rows JSONL: {result.removed_path}")
    print(f"Metadata: {result.metadata_path}")
    if result.publish_repo:
        suffix = " (dry run)" if args.publish_dry_run else ""
        print(f"Published to: {result.publish_repo}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
