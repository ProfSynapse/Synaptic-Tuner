"""Capture file names, integrity checks, and optional per-row resume journal."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import re

from shared.utilities.run_log import RunLog


def capture_run_identity(model: str, revision: str | None, adapter: str | None,
                         render_fn, content_end_fn) -> dict:
    """Bind CLI resumes to pinned Hub sources or complete local file bytes."""
    def source_identity(source, *, allow_hub):
        if source is None:
            return None
        path = Path(source)
        if path.is_dir():
            # Include tokenizer/config and all checkpoint shards. Local adapters
            # include their complete file set; no revision string authenticates them.
            files = sorted(p for p in path.rglob("*") if p.is_file())
            if not files:
                raise ValueError("empty local source cannot identify a capture resume")
            return {"path": str(path.resolve()), "files": {
                str(p.relative_to(path)): file_info(p)["sha256"] for p in files}}
        if not allow_hub or not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("resumable capture requires a pinned Hub model or a local source directory")
        return {"model": source, "revision": revision}

    def callable_identity(fn):
        if fn is None:
            return None
        source = inspect.getsourcefile(fn)
        if not source or not Path(source).is_file():
            raise ValueError("capture resume requires file-backed render/content-end callables")
        return {"callable": f"{fn.__module__}:{fn.__qualname__}",
                "source_sha256": file_info(Path(source))["sha256"]}

    return {"model": source_identity(model, allow_hub=True),
            "revision": revision, "adapter": source_identity(adapter, allow_hub=False),
            "render": callable_identity(render_fn), "content_end": callable_identity(content_end_fn)}


def row_key(row: dict) -> str:
    value = row.get("row_key", row.get("id"))
    if value is None or not str(value):
        raise ValueError("extraction requires a non-empty row_key or id")
    return str(value)


def safe_key(key: str) -> str:
    """Keep simple historical names; encode unsafe names without lossy rewrites.

    The caller also rejects duplicate resulting names before writing anything.
    This covers a deliberately chosen key matching an encoded key's name.
    """
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
    }
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key) and key.upper() not in reserved:
        return key
    return "row-" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def file_info(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        # Persist tensor bytes before the journal can mark this row complete.
        os.fsync(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "sha256": digest.hexdigest()}


def verified_file(root: Path, info: dict) -> Path:
    name = info["file"]
    if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("capture manifest contains an unsafe file name")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("committed capture file is missing or symlinked")
    if file_info(path)["sha256"] != info["sha256"]:
        raise ValueError("committed capture file hash mismatch")
    return path


def verify_record(root: Path, record: dict) -> None:
    for info in record.get("capture_files", {}).values():
        verified_file(root, info)
    for info in record.get("site_captures", {}).values():
        verified_file(root, info)


def write_manifest(root: Path, manifest: dict) -> None:
    temporary = root / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / "manifest.json")


class CaptureJournal:
    """Single-writer journal. A resume verifies identity and all committed files."""

    def __init__(self, root: Path, identity: dict, *, resume: bool):
        self.root = root
        self.identity = identity
        self.resume = resume
        self.log = None
        self.records: dict[str, dict] = {}

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "capture.runlog.jsonl"
        if any(self.root.iterdir()) and (not self.resume or not path.with_name(path.name + ".meta.json").is_file()):
            raise ValueError("extraction output is non-empty; use a new directory or resume its matching journal")
        if self.resume:
            self.log = RunLog(path, self.identity, key_field="row_key")
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    key = record["row_key"]
                    if key in self.records:
                        raise ValueError("duplicate row key in capture journal")
                    verify_record(self.root, record)
                    self.records[key] = record
            except BaseException:
                self.log.close()
                raise
        return self

    def record(self, record: dict) -> None:
        if self.log:
            self.log.record(record["row_key"], record)

    def finalize(self, manifest: dict) -> None:
        write_manifest(self.root, manifest)
        if self.log:
            self.log.finalize({"n_rows": manifest["n_rows"], "manifest": "manifest.json"})

    def __exit__(self, *exc):
        if self.log:
            self.log.close()
