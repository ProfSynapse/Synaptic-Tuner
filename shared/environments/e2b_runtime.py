"""E2B-backed environment runtime.

The implementation is intentionally conservative and command-based so this
package can degrade gracefully when running without E2B credentials.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
from typing import Any, Dict, List, Optional

from .base import EnvironmentRuntime
from .fixture_parser import EnvironmentFixture

try:  # E2B Python SDK (newer)
    from e2b import Sandbox  # type: ignore
except ImportError:  # Legacy package name fallback
    try:
        from e2b_code_interpreter import Sandbox  # type: ignore
    except ImportError:
        Sandbox = None


class E2BEnvironmentRuntime(EnvironmentRuntime):
    """Environment runtime powered by E2B sandboxes."""

    def __init__(
        self,
        template: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
    ):
        self.template = template
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.root = "/workspace"
        self._sandbox = None

    def setup(self, fixture: EnvironmentFixture) -> None:
        self._sandbox = self._create_sandbox()
        self._run(f"mkdir -p {shlex.quote(self.root)}")

        for directory in fixture.directories:
            self.mkdir(directory)
        for path, content in fixture.files.items():
            self.write_text(path, content)

    def teardown(self) -> None:
        sandbox = self._sandbox
        self._sandbox = None
        if sandbox is None:
            return
        if hasattr(sandbox, "close"):
            sandbox.close()
        elif hasattr(sandbox, "kill"):
            sandbox.kill()

    def mkdir(self, path: str) -> None:
        full_path = self._to_full_path(path)
        self._run(f"mkdir -p {shlex.quote(full_path)}")

    def write_text(self, path: str, content: str) -> None:
        full_path = self._to_full_path(path)
        payload = base64.b64encode((content or "").encode("utf-8")).decode("ascii")
        py = (
            "import base64, pathlib; "
            f"p=pathlib.Path({full_path!r}); "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            f"p.write_text(base64.b64decode({payload!r}).decode('utf-8'), encoding='utf-8')"
        )
        self._run(f"python -c {shlex.quote(py)}")

    def read_text(self, path: str) -> str:
        full_path = self._to_full_path(path)
        py = (
            "import pathlib, json; "
            f"print(json.dumps(pathlib.Path({full_path!r}).read_text(encoding='utf-8')))"
        )
        result = self._run(f"python -c {shlex.quote(py)}")
        stdout = _result_stdout(result).strip()
        return json.loads(stdout) if stdout else ""

    def list_dir(self, path: str = ".") -> List[str]:
        full_path = self._to_full_path(path)
        py = (
            "import pathlib, json; "
            f"p=pathlib.Path({full_path!r}); "
            "items=sorted([x.name for x in p.iterdir()]) if p.exists() and p.is_dir() else []; "
            "print(json.dumps(items))"
        )
        result = self._run(f"python -c {shlex.quote(py)}")
        stdout = _result_stdout(result).strip()
        return json.loads(stdout) if stdout else []

    def move(self, path: str, new_path: str, overwrite: bool = False) -> None:
        src = self._to_full_path(path)
        dst = self._to_full_path(new_path)
        py = (
            "import pathlib, shutil; "
            f"src=pathlib.Path({src!r}); dst=pathlib.Path({dst!r}); "
            "if not src.exists(): raise FileNotFoundError(src); "
            "dst.parent.mkdir(parents=True, exist_ok=True); "
            f"overwrite={overwrite!r}; "
            "if dst.exists() and not overwrite: raise FileExistsError(dst); "
            "if dst.exists() and overwrite: "
            "    shutil.rmtree(dst) if dst.is_dir() else dst.unlink(); "
            "shutil.move(str(src), str(dst))"
        )
        self._run(f"python -c {shlex.quote(py)}")

    def copy(self, path: str, new_path: str, overwrite: bool = False) -> None:
        src = self._to_full_path(path)
        dst = self._to_full_path(new_path)
        py = (
            "import pathlib, shutil; "
            f"src=pathlib.Path({src!r}); dst=pathlib.Path({dst!r}); "
            "if not src.exists(): raise FileNotFoundError(src); "
            "dst.parent.mkdir(parents=True, exist_ok=True); "
            f"overwrite={overwrite!r}; "
            "if dst.exists() and not overwrite: raise FileExistsError(dst); "
            "if dst.exists() and overwrite: "
            "    shutil.rmtree(dst) if dst.is_dir() else dst.unlink(); "
            "shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)"
        )
        self._run(f"python -c {shlex.quote(py)}")

    def delete(self, path: str, recursive: bool = False) -> None:
        target = self._to_full_path(path)
        py = (
            "import pathlib, shutil; "
            f"target=pathlib.Path({target!r}); "
            "recursive="
            f"{recursive!r}; "
            "if not target.exists(): pass\n"
            "elif target.is_dir(): "
            "    shutil.rmtree(target) if recursive else target.rmdir()\n"
            "else: target.unlink()"
        )
        self._run(f"python -c {shlex.quote(py)}")

    def exists(self, path: str) -> bool:
        full_path = self._to_full_path(path)
        result = self._run(f"test -e {shlex.quote(full_path)}; echo $?")
        return _result_stdout(result).strip().endswith("0")

    def search(self, query: str, path: str = ".") -> List[str]:
        full_path = self._to_full_path(path)
        query_lower = (query or "").lower()
        py = (
            "import pathlib, json; "
            f"base=pathlib.Path({full_path!r}); q={query_lower!r}; out=[]; "
            "if base.exists():\n"
            "  for file in base.rglob('*'):\n"
            "    if not file.is_file():\n"
            "      continue\n"
            "    rel=str(file.relative_to(base))\n"
            "    if q in rel.lower():\n"
            "      out.append(rel)\n"
            "      continue\n"
            "    try:\n"
            "      content=file.read_text(encoding='utf-8')\n"
            "    except Exception:\n"
            "      continue\n"
            "    if q in content.lower():\n"
            "      out.append(rel)\n"
            "print(json.dumps(sorted(set(out))))"
        )
        result = self._run(f"python -c {shlex.quote(py)}")
        stdout = _result_stdout(result).strip()
        return json.loads(stdout) if stdout else []

    def snapshot(self, limit: int = 200) -> Dict[str, Any]:
        py = (
            "import pathlib, json; "
            f"root=pathlib.Path({self.root!r}); limit={int(limit)!r}; "
            "files=[]; dirs=[]; count=0; "
            "if root.exists():\n"
            "  for p in root.rglob('*'):\n"
            "    rel=str(p.relative_to(root))\n"
            "    if p.is_dir():\n"
            "      dirs.append(rel)\n"
            "    else:\n"
            "      size=p.stat().st_size if p.exists() else 0\n"
            "      files.append({'path': rel, 'size': size})\n"
            "    count += 1\n"
            "    if count >= limit:\n"
            "      break\n"
            "print(json.dumps({'runtime': 'e2b', 'root': str(root), 'directories': sorted(dirs), 'files': sorted(files, key=lambda x: x['path']), 'truncated': count >= limit}))"
        )
        result = self._run(f"python -c {shlex.quote(py)}")
        stdout = _result_stdout(result).strip()
        return json.loads(stdout) if stdout else {"runtime": "e2b", "root": self.root}

    def _create_sandbox(self):
        if Sandbox is None:
            raise RuntimeError(
                "E2B SDK is not installed. Install `e2b` (or `e2b_code_interpreter`) to use --env-backend e2b."
            )
        kwargs: Dict[str, Any] = {}
        if self.template:
            kwargs["template"] = self.template
        if self.api_key:
            kwargs["api_key"] = self.api_key
        try:
            return Sandbox(**kwargs)
        except TypeError:
            # Fallback for SDK variants with different constructor signatures.
            if kwargs:
                return Sandbox(self.template) if self.template else Sandbox()
            return Sandbox()

    def _run(self, command: str):
        sandbox = self._require_sandbox()
        if hasattr(sandbox, "commands") and hasattr(sandbox.commands, "run"):
            result = sandbox.commands.run(command, timeout=self.timeout_seconds)
        elif hasattr(sandbox, "run_command"):
            result = sandbox.run_command(command, timeout=self.timeout_seconds)
        else:
            raise RuntimeError("Unsupported E2B SDK: no command execution method found")

        code = _result_exit_code(result)
        if code != 0:
            stderr = _result_stderr(result)
            raise RuntimeError(stderr or f"E2B command failed (exit code {code}): {command}")
        return result

    def _require_sandbox(self):
        if self._sandbox is None:
            raise RuntimeError("E2BEnvironmentRuntime is not initialized")
        return self._sandbox

    def _to_full_path(self, path: str) -> str:
        normalized = str(path or ".").strip().replace("\\", "/").lstrip("/")
        if normalized in {"", "."}:
            return self.root
        return f"{self.root}/{normalized}"


def _result_stdout(result: Any) -> str:
    value = getattr(result, "stdout", "")
    return value if isinstance(value, str) else str(value)


def _result_stderr(result: Any) -> str:
    value = getattr(result, "stderr", "")
    return value if isinstance(value, str) else str(value)


def _result_exit_code(result: Any) -> int:
    if hasattr(result, "exit_code"):
        return int(getattr(result, "exit_code") or 0)
    if hasattr(result, "code"):
        return int(getattr(result, "code") or 0)
    return 0

