#!/usr/bin/env python3
"""Create a deterministic, read-only private snapshot of explicitly configured Nexus notes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class SnapshotError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class SafeArgumentParser(argparse.ArgumentParser):
    """Convert parser diagnostics to the private-safe JSON error contract."""

    def error(self, message: str) -> None:
        raise SnapshotError("INVALID_ARGUMENTS")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SnapshotError(code)
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:
        # YAML errors deliberately collapse to a non-prose machine code.
        raise SnapshotError("CONFIG_READ_FAILED") from error
    return _mapping(value, "INVALID_CONFIG")


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SnapshotError("UNSAFE_METADATA")
        return value
    if isinstance(value, str) or value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, list):
        return [_safe_metadata(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or re.search(r"(?:secret|token|password|api[_-]?key)", key, re.IGNORECASE):
                raise SnapshotError("UNSAFE_METADATA")
            result[key] = _safe_metadata(item)
        return result
    raise SnapshotError("UNSAFE_METADATA")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    nexus = _mapping(config.get("nexus"), "INVALID_NEXUS_CONFIG")
    for key in ("vault", "workspace", "session"):
        if not isinstance(nexus.get(key), str) or not nexus[key]:
            raise SnapshotError("INVALID_NEXUS_CONFIG")
    notes = config.get("notes")
    if not isinstance(notes, list) or not notes:
        raise SnapshotError("INVALID_NOTES_CONFIG")
    ids: set[str] = set()
    normalized_notes: list[dict[str, Any]] = []
    for note in notes:
        item = _mapping(note, "INVALID_NOTE_CONFIG")
        note_id, note_path = item.get("id"), item.get("path")
        if not isinstance(note_id, str) or not note_id or note_id in ids or not isinstance(note_path, str) or not note_path:
            raise SnapshotError("INVALID_NOTE_CONFIG")
        ids.add(note_id)
        metadata = _safe_metadata(item.get("metadata", {}))
        if not isinstance(metadata, dict):
            raise SnapshotError("UNSAFE_METADATA")
        normalized_notes.append({"id": note_id, "path": note_path, "metadata": metadata})
    output_path = config.get("output_path")
    if not isinstance(output_path, str) or not output_path:
        raise SnapshotError("INVALID_OUTPUT_CONFIG")
    options = _mapping(config.get("options", {}), "INVALID_OPTIONS_CONFIG")
    if not isinstance(options.get("strip_frontmatter", False), bool):
        raise SnapshotError("INVALID_OPTIONS_CONFIG")
    return {"nexus": nexus, "notes": normalized_notes, "output_path": output_path, "options": options}


def nexus_argv(nexus: dict[str, str], note_path: str) -> list[str]:
    return [
        "nexus", "use", "--json", "--vault", nexus["vault"], "--workspace", nexus["workspace"], "--session", nexus["session"],
        "--memory", "snapshotting configured notes", "--goal", "read configured note without mutation",
        "--", "content", "read", note_path, "1",
    ]


def resolve_nexus_command(logical_argv: list[str]) -> list[str]:
    """Resolve an executable command without interpreting any shim shell text."""
    if not logical_argv or logical_argv[0] != "nexus":
        raise SnapshotError("NEXUS_RUNTIME_UNAVAILABLE")
    executable = shutil.which("nexus")
    if not executable:
        raise SnapshotError("NEXUS_RUNTIME_UNAVAILABLE")
    shim = Path(executable)
    if os.name != "nt" or shim.suffix.lower() not in {".cmd", ".bat"}:
        return [executable, *logical_argv[1:]]
    node = shutil.which("node")
    cli = shim.with_name("nexus-cli.js")
    if not node:
        raise SnapshotError("NEXUS_RUNTIME_UNAVAILABLE")
    if shim.name.lower() not in {"nexus.cmd", "nexus.bat"} or not cli.is_file():
        raise SnapshotError("NEXUS_SHIM_UNSUPPORTED")
    return [node, str(cli), *logical_argv[1:]]


def parse_nexus_response(raw: str, expected_path: str) -> str:
    try:
        outer = _mapping(json.loads(raw), "NEXUS_ENVELOPE_INVALID")
        content = outer.get("content")
        if not isinstance(content, list) or len(content) != 1:
            raise SnapshotError("NEXUS_ENVELOPE_INVALID")
        block = _mapping(content[0], "NEXUS_ENVELOPE_INVALID")
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise SnapshotError("NEXUS_ENVELOPE_INVALID")
        inner = _mapping(json.loads(block["text"]), "NEXUS_ENVELOPE_INVALID")
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        raise SnapshotError("NEXUS_ENVELOPE_INVALID") from error
    if inner.get("success") is not True or inner.get("path") != expected_path or not isinstance(inner.get("content"), str):
        raise SnapshotError("NEXUS_RESPONSE_REJECTED")
    return reconstruct_numbered_lines(inner["content"])


def reconstruct_numbered_lines(value: str) -> str:
    if value == "":
        return ""
    reconstructed: list[str] = []
    for expected, line in enumerate(value.splitlines(), start=1):
        match = re.fullmatch(r"(\d+): (.*)", line)
        if match is None or int(match.group(1)) != expected:
            raise SnapshotError("NUMBERED_CONTENT_INVALID")
        reconstructed.append(match.group(2))
    return "\n".join(reconstructed)


def strip_frontmatter(value: str, enabled: bool) -> str:
    if not enabled or not value.startswith("---\n"):
        return value
    lines = value.split("\n")
    for index in range(1, len(lines)):
        if lines[index] == "---":
            return "\n".join(lines[index + 1:])
    raise SnapshotError("FRONTMATTER_MALFORMED")


def read_note(nexus: dict[str, str], note_path: str) -> str:
    command = resolve_nexus_command(nexus_argv(nexus, note_path))
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, encoding="utf-8")
    except OSError as error:
        raise SnapshotError("NEXUS_EXECUTION_FAILED") from error
    if completed.returncode != 0:
        raise SnapshotError("NEXUS_READ_FAILED")
    return parse_nexus_response(completed.stdout, note_path)


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for note in config["notes"]:
        full_text = read_note(config["nexus"], note["path"])
        selected_text = strip_frontmatter(full_text, config["options"].get("strip_frontmatter", False))
        rows.append({"id": note["id"], "source_path": note["path"], "metadata": note["metadata"], "text": selected_text, "full_content_sha256": _hash(full_text), "selected_text_sha256": _hash(selected_text)})
    return rows


def write_snapshot(rows: list[dict[str, Any]], output_path: Path) -> str:
    payload = "".join(_json(row) + "\n" for row in rows).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    temporary: Path | None = None
    try:
        if output_path.exists():
            if output_path.read_bytes() == payload:
                return digest
            raise SnapshotError("OUTPUT_COLLISION")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=".nexus-snapshot-", dir=output_path.parent)
        temporary = Path(temporary_name)
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError:
            if output_path.read_bytes() == payload:
                return digest
            raise SnapshotError("OUTPUT_COLLISION")
        temporary.unlink()
        temporary = None
        return digest
    except SnapshotError:
        raise
    except (OSError, ValueError) as error:
        raise SnapshotError("OUTPUT_WRITE_FAILED") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    try:
        parser = SafeArgumentParser(description="Snapshot explicitly configured Nexus notes without mutation.")
        parser.add_argument("--config", required=True, type=Path)
        args = parser.parse_args(argv)
        config = validate_config(load_config(args.config))
        rows = build_rows(config)
        digest = write_snapshot(rows, Path(config["output_path"]))
        print(_json({"status": "ok", "row_count": len(rows), "output_sha256": digest}))
        return 0
    except SnapshotError as error:
        print(_json({"status": "error", "error": {"code": error.code}}))
        return 2
    except Exception:
        print(_json({"status": "error", "error": {"code": "SNAPSHOT_FAILED"}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
