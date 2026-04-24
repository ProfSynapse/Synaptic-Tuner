"""SynthChat Sanitize Mode - Apply privacy preprocessing to docs or datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..config.privacy import load_privacy_profiles, resolve_privacy_settings
from ..services.privacy_preprocess import PrivacyPreprocessor


_DOC_EXTENSIONS = {".md", ".txt", ".html", ".htm"}


def sanitize_mode(args, *, load_settings):
    """Sanitize input docs or datasets using a named privacy preprocess profile."""
    print("=== SynthChat: Sanitize Mode ===\n")

    config_dir = Path(args.config_dir or "SynthChat/config")
    settings = load_settings(config_dir)
    privacy_overrides: Dict[str, Any] = {}
    if getattr(args, "privacy_profile", None):
        privacy_overrides = {"enabled": True, "profile": args.privacy_profile}
    settings["privacy_preprocess"] = resolve_privacy_settings(settings, privacy_overrides)

    profile_name = str(settings.get("privacy_preprocess", {}).get("profile") or "").strip()
    if not profile_name:
        print("Error: Privacy sanitize mode requires a profile via settings.yaml or --privacy-profile")
        sys.exit(1)

    profiles_registry = load_privacy_profiles(config_dir / "privacy_profiles.yaml")
    preprocessor = PrivacyPreprocessor.from_registry(
        profile_name=profile_name,
        profiles_registry=profiles_registry,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    if input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_sanitized"
        summary = _sanitize_directory(input_path, output_dir, preprocessor)
        report_path = output_dir / "privacy_sanitize_report.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Sanitized {summary['files_written']} file(s)")
        print(f"Output: {output_dir}")
        print(f"Report: {report_path}")
        return

    if input_path.suffix.lower() == ".jsonl":
        output_file = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_sanitized.jsonl")
        summary = _sanitize_jsonl(input_path, output_file, preprocessor)
        report_path = output_file.with_name(f"{output_file.name}.privacy_report.json")
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Sanitized {summary['records_processed']} JSONL record(s)")
        print(f"Output: {output_file}")
        print(f"Report: {report_path}")
        return

    output_file = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_sanitized{input_path.suffix}")
    summary = _sanitize_text_file(input_path, output_file, preprocessor)
    report_path = output_file.with_name(f"{output_file.name}.privacy_report.json")
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Sanitized 1 file")
    print(f"Output: {output_file}")
    print(f"Report: {report_path}")


def _sanitize_directory(input_dir: Path, output_dir: Path, preprocessor: PrivacyPreprocessor) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_reports: List[Dict[str, Any]] = []
    files_written = 0

    for source in sorted(path for path in input_dir.rglob("*") if path.is_file()):
        if source.suffix.lower() not in _DOC_EXTENSIONS and source.suffix.lower() != ".jsonl":
            continue
        relative = source.relative_to(input_dir)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".jsonl":
            summary = _sanitize_jsonl(source, target, preprocessor)
        else:
            summary = _sanitize_text_file(source, target, preprocessor)
        file_reports.append(summary)
        files_written += 1

    return {
        "mode": "directory",
        "profile": preprocessor.profile_name,
        "input": str(input_dir),
        "output": str(output_dir),
        "files_written": files_written,
        "files": file_reports,
    }


def _sanitize_text_file(input_file: Path, output_file: Path, preprocessor: PrivacyPreprocessor) -> Dict[str, Any]:
    text = input_file.read_text(encoding="utf-8")
    result = preprocessor.process_text(text, scope_key=str(input_file))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(result.sanitized_text, encoding="utf-8")
    return {
        "mode": "text",
        "profile": preprocessor.profile_name,
        "input": str(input_file),
        "output": str(output_file),
        "changed": result.changed,
        "metadata": result.to_metadata(),
    }


def _sanitize_jsonl(input_file: Path, output_file: Path, preprocessor: PrivacyPreprocessor) -> Dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    records_processed = 0
    changed_records = 0
    record_reports: List[Dict[str, Any]] = []

    with input_file.open("r", encoding="utf-8") as src, output_file.open("w", encoding="utf-8") as dst:
        for line_number, raw_line in enumerate(src, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and "_meta" in payload:
                payload.setdefault("_meta", {})
                payload["_meta"]["privacy_preprocess"] = {"profile": preprocessor.profile_name}
                dst.write(json.dumps(payload) + "\n")
                continue

            sanitized_payload, reports = preprocessor.sanitize_payload(payload, scope_key=f"{input_file}:{line_number}")
            if isinstance(sanitized_payload, dict):
                metadata = sanitized_payload.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["privacy_preprocess"] = {
                        "profile": preprocessor.profile_name,
                        "changed": bool(reports),
                        "report_count": len(reports),
                        "reports": reports,
                    }
            dst.write(json.dumps(sanitized_payload) + "\n")
            records_processed += 1
            if reports:
                changed_records += 1
            record_reports.append(
                {
                    "line_number": line_number,
                    "changed": bool(reports),
                    "report_count": len(reports),
                }
            )

    return {
        "mode": "jsonl",
        "profile": preprocessor.profile_name,
        "input": str(input_file),
        "output": str(output_file),
        "records_processed": records_processed,
        "changed_records": changed_records,
        "records": record_reports,
    }

