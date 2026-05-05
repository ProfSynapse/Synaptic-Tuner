"""Local temp-directory environment runtime."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

from .base import EnvironmentRuntime
from .fixture_parser import EnvironmentFixture


class LocalEnvironmentRuntime(EnvironmentRuntime):
    """Filesystem-backed runtime using a temporary local directory."""

    def __init__(self):
        self._temp_base: Path | None = None
        self._root: Path | None = None

    def setup(self, fixture: EnvironmentFixture) -> None:
        self._temp_base = _runtime_temp_base()
        self._temp_base.mkdir(parents=True, exist_ok=True)
        self._root = self._temp_base / f"synaptic_env_{uuid.uuid4().hex}"
        self._root.mkdir(parents=False, exist_ok=False)

        for directory in fixture.directories:
            self.mkdir(directory)
        for path, content in fixture.files.items():
            self.write_text(path, content)

    def teardown(self) -> None:
        if self._root is not None:
            root = self._root.resolve()
            base = self._temp_base.resolve() if self._temp_base is not None else None
            if base is not None and root.parent == base and root.name.startswith("synaptic_env_"):
                shutil.rmtree(root, ignore_errors=True)
        self._temp_base = None
        self._root = None

    def mkdir(self, path: str) -> None:
        resolved = self._resolve(path)
        resolved.mkdir(parents=True, exist_ok=True)

    def write_text(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content or "", encoding="utf-8")

    def read_text(self, path: str) -> str:
        resolved = self._resolve(path)
        return resolved.read_text(encoding="utf-8")

    def list_dir(self, path: str = ".") -> List[str]:
        resolved = self._resolve(path)
        if not resolved.exists() or not resolved.is_dir():
            return []
        return sorted(item.name for item in resolved.iterdir())

    def move(self, path: str, new_path: str, overwrite: bool = False) -> None:
        src = self._resolve(path)
        dst = self._resolve(new_path)
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {path}")
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {new_path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and overwrite:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))

    def copy(self, path: str, new_path: str, overwrite: bool = False) -> None:
        src = self._resolve(path)
        dst = self._resolve(new_path)
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {path}")
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {new_path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and overwrite:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def delete(self, path: str, recursive: bool = False) -> None:
        target = self._resolve(path)
        if not target.exists():
            return
        if target.is_dir():
            if recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def search(self, query: str, path: str = ".") -> List[str]:
        base = self._resolve(path)
        if not base.exists():
            return []

        root = self._require_root().resolve()
        needle = (query or "").strip().lower()
        query_terms = _search_terms(query)
        matches: List[str] = []
        for file_path in base.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.resolve().relative_to(root).as_posix()
            rel_lower = rel.lower()
            if needle and needle in rel_lower:
                matches.append(rel)
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            content_lower = content.lower()
            if needle and needle in content_lower:
                matches.append(rel)
                continue
            if query_terms and _all_terms_match(query_terms, f"{rel_lower}\n{content_lower}"):
                matches.append(rel)

        return sorted(set(matches))

    def snapshot(self, limit: int = 200) -> Dict[str, Any]:
        root = self._require_root()
        files = []
        directories = []

        for path in root.rglob("*"):
            rel = str(path.relative_to(root))
            if path.is_dir():
                directories.append(rel)
            else:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                files.append({"path": rel, "size": size})

            if len(files) + len(directories) >= limit:
                break

        return {
            "runtime": "local",
            "root": str(root),
            "directories": sorted(directories),
            "files": sorted(files, key=lambda x: x["path"]),
            "truncated": (len(files) + len(directories)) >= limit,
        }

    def _resolve(self, path: str) -> Path:
        root = self._require_root().resolve()
        normalized = str(path or ".").strip().replace("\\", "/").lstrip("/")
        if not normalized:
            normalized = "."
        candidate = (root / normalized).resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"Path escapes runtime root: {path}")
        return candidate

    def _require_root(self) -> Path:
        if self._root is None:
            raise RuntimeError("LocalEnvironmentRuntime is not initialized")
        return self._root


def _runtime_temp_base() -> Path:
    configured = os.getenv("SYNTHCHAT_ENV_TMPDIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / ".runtime_tmp" / "environments").resolve()


_SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _search_terms(query: str) -> List[str]:
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", query or "").lower()
    return [
        token
        for token in normalized.split()
        if len(token) > 1 and token not in _SEARCH_STOPWORDS
    ]


def _all_terms_match(terms: List[str], haystack: str) -> bool:
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", haystack or "").lower()
    padded = f" {normalized} "
    return all(f" {term} " in padded for term in terms)
