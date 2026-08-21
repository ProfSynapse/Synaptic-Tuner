"""Create the exact isolated CPython 3.12.7 HF training-smoke launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

EXPECTED_DIRECT = (
    "huggingface_hub==1.27.0",
    "jsonschema==4.23.0",
    "packaging==24.1",
    "python-dotenv==1.0.1",
    "PyYAML==6.0.2",
)
EXPECTED_PYTHON = {"implementation": "CPython", "version": [3, 12, 7]}
ML_DISTRIBUTIONS = frozenset({"torch", "transformers", "unsloth", "unsloth-zoo", "trl", "peft", "accelerate", "datasets"})
INJECTION_ENV = frozenset(
    {
        "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT",
        "PIP_CONFIG_FILE", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST",
        "HF_TOKEN", "HF_API_KEY", "WANDB_API_KEY",
    }
)
_LOCK_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;@]+) --hash=sha256:([0-9a-f]{64})$")
_PUBLIC_VERSION = re.compile(
    r"^[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?$"
)


class LauncherContractError(RuntimeError):
    pass


def canonicalize_name(value: str) -> str:
    """PEP 503 name normalization without importing pre-install dependencies."""

    return re.sub(r"[-_.]+", "-", value).lower()


def authenticated_repo_root() -> Path:
    script = Path(__file__).resolve(strict=True)
    if script.is_symlink() or not script.is_file() or script.name != "setup_hf_training_smoke_launcher.py":
        raise LauncherContractError("Launcher setup script identity is invalid")
    root = script.parents[1]
    if script.parent.name != "scripts" or (root / "scripts" / script.name).resolve(strict=True) != script:
        raise LauncherContractError("Launcher setup repository root is invalid")
    return root


def _run_checked(argv: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, check=True, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LauncherContractError("Launcher subprocess failed") from exc


def _regular_text(path: Path, maximum: int) -> str:
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_size > maximum:
        raise LauncherContractError("Launcher contract input must be a bounded regular file")
    return path.read_text(encoding="utf-8")


def validate_direct_requirements(path: Path) -> None:
    lines = _regular_text(path, 16 * 1024).splitlines()
    if tuple(lines) != EXPECTED_DIRECT or len({canonicalize_name(line.split("==", 1)[0]) for line in lines}) != len(lines):
        raise LauncherContractError("Direct requirements must be the exact ordered five pins")


def validate_hashed_lock(path: Path) -> dict[str, str]:
    text = _regular_text(path, 256 * 1024)
    lines = text.splitlines()
    if not lines or any(not line or line != line.strip() for line in lines):
        raise LauncherContractError("Hashed lock must contain only canonical nonblank lines")
    installed: dict[str, str] = {}
    names_in_order: list[str] = []
    for line in lines:
        match = _LOCK_LINE.fullmatch(line)
        if not match:
            raise LauncherContractError("Hashed lock contains an unpinned, unhashed, ranged, editable, VCS, URL, local, or marked requirement")
        raw_name, version, _digest = match.groups()
        name = canonicalize_name(raw_name)
        if raw_name != name or not _PUBLIC_VERSION.fullmatch(version):
            raise LauncherContractError("Hashed lock names and versions must be canonical exact public identities")
        if name in ML_DISTRIBUTIONS:
            raise LauncherContractError("Hashed launcher lock cannot contain ML distributions")
        if name in installed:
            raise LauncherContractError("Hashed lock contains a duplicate or case-aliased distribution")
        installed[name] = version
        names_in_order.append(name)
    expected_direct = {canonicalize_name(line.split("==", 1)[0]): line.split("==", 1)[1] for line in EXPECTED_DIRECT}
    if any(installed.get(name) != version for name, version in expected_direct.items()):
        raise LauncherContractError("Hashed lock does not contain the exact direct pins")
    if names_in_order != sorted(names_in_order):
        raise LauncherContractError("Hashed lock distribution order is not canonical")
    return installed


def validate_installed_allowlist(path: Path, *, lock_path: Path) -> dict[str, str]:
    try:
        value = json.loads(_regular_text(path, 256 * 1024))
    except json.JSONDecodeError as exc:
        raise LauncherContractError("Installed allowlist is not JSON") from exc
    required = {"schema_version", "python", "lock_sha256", "distributions"}
    if not isinstance(value, dict) or set(value) != required:
        raise LauncherContractError("Installed allowlist has an unexpected shape")
    lock_bytes = lock_path.read_bytes()
    if value["schema_version"] != "synaptic-hf-training-launcher-installed/v1" or value["python"] != EXPECTED_PYTHON or value["lock_sha256"] != hashlib.sha256(lock_bytes).hexdigest():
        raise LauncherContractError("Installed allowlist does not bind this interpreter and lock")
    distributions = value["distributions"]
    if not isinstance(distributions, dict) or not distributions:
        raise LauncherContractError("Installed allowlist distributions are missing")
    normalized: dict[str, str] = {}
    for raw_name, version in distributions.items():
        if not isinstance(raw_name, str) or not isinstance(version, str) or not version:
            raise LauncherContractError("Installed allowlist distribution entry is invalid")
        name = canonicalize_name(raw_name)
        if name != raw_name or name in normalized:
            raise LauncherContractError("Installed allowlist names must be canonical and unique")
        normalized[name] = version
    locked = validate_hashed_lock(lock_path)
    expected_installed = {**locked, "pip": normalized.get("pip", "")}
    if not normalized.get("pip") or normalized != dict(sorted(expected_installed.items())) or set(normalized) & ML_DISTRIBUTIONS:
        raise LauncherContractError("Installed allowlist differs from the lock or contains ML packages")
    if list(distributions) != sorted(distributions):
        raise LauncherContractError("Installed allowlist order is not canonical")
    return normalized


def require_exact_python(python: Path) -> None:
    code = "import json,platform,sys;print(json.dumps({'implementation':platform.python_implementation(),'version':list(sys.version_info[:3])},sort_keys=True))"
    completed = _run_checked([str(python), "-I", "-c", code], capture_output=True, text=True)
    try:
        observed = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LauncherContractError("Launcher interpreter identity is invalid") from exc
    if observed != EXPECTED_PYTHON:
        raise LauncherContractError("HF training-smoke launcher requires CPython 3.12.7 exactly")


def sanitized_environment() -> dict[str, str]:
    present = sorted(key for key in INJECTION_ENV if os.environ.get(key))
    if present:
        raise LauncherContractError("Launcher setup rejects ambient injection or credential variables")
    environment = {
        "PATH": os.defpath,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    }
    for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP"):
        if os.environ.get(key):
            environment[key] = os.environ[key]
    return environment


def verify_installed(*, launcher_python: Path, repo_root: Path, expected: dict[str, str], environment: dict[str, str]) -> None:
    payload = json.dumps(expected, sort_keys=True)
    root = repo_root.resolve(strict=True)
    script = (
        "import importlib.metadata as m,json,os,site,sys,pathlib;"
        f"expected=json.loads({payload!r});"
        "assert sys.prefix!=sys.base_prefix and site.ENABLE_USER_SITE is False;"
        "actual={};bad=[];"
        "[(actual.setdefault(__import__('packaging.utils').utils.canonicalize_name(d.metadata['Name']),d.version),"
        "bad.append(d.metadata['Name']) if (d.read_text('direct_url.json') or '').strip() else None) for d in m.distributions()];"
        "assert actual==expected and not bad;"
        "roots=[pathlib.Path(p) for p in site.getsitepackages()];"
        "assert roots and all(str(p.resolve()).startswith(str(pathlib.Path(sys.prefix).resolve())) for p in roots);"
        "pth=[line for r in roots for f in r.glob('*.pth') for line in f.read_text(encoding='utf-8',errors='strict').splitlines() if line.lstrip().startswith(('import ','import\\t'))];"
        "assert not pth;"
        f"sys.path.insert(0,{str(root)!r});"
        "import huggingface_hub,jsonschema,dotenv,yaml,packaging;"
        "import tuner.cloud.hf_training_image_lock;"
        "assert not ({'torch','transformers','unsloth','trl','peft','accelerate','datasets'} & set(sys.modules))"
    )
    _run_checked(
        [str(launcher_python), "-I", "-m", "pip", "check"],
        cwd=root, env=environment, capture_output=True, text=True,
    )
    _run_checked([str(launcher_python), "-I", "-c", script], cwd=root, env=environment, capture_output=True, text=True)


def setup_launcher(
    *, python: Path, venv: Path, direct: Path, lock: Path, allowlist: Path,
    wheelhouse: Path, repo_root: Path,
) -> Path:
    validate_direct_requirements(direct)
    expected = validate_installed_allowlist(allowlist, lock_path=lock)
    require_exact_python(python)
    environment = sanitized_environment()
    if venv.exists():
        raise FileExistsError("HF training-smoke launcher target must be fresh")
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise LauncherContractError("Launcher wheelhouse must be a regular external directory")
    _run_checked([str(python), "-I", "-m", "venv", str(venv)], env=environment)
    launcher = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run_checked(
        [
            str(launcher), "-I", "-m", "pip", "install", "--no-index",
            "--find-links", str(wheelhouse), "--require-hashes", "--no-deps",
            "--requirement", str(lock),
        ],
        env=environment,
    )
    verify_installed(launcher_python=launcher, repo_root=repo_root, expected=expected, environment=environment)
    return launcher


def build_parser() -> argparse.ArgumentParser:
    root = authenticated_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--direct", type=Path, default=root / "requirements-hf-training-smoke.direct.txt")
    parser.add_argument("--lock", type=Path, default=root / "requirements-hf-training-smoke.lock")
    parser.add_argument("--allowlist", type=Path, default=root / "requirements-hf-training-smoke-installed.json")
    parser.add_argument("--wheelhouse", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        launcher = setup_launcher(
            python=args.python.resolve(), venv=args.venv.resolve(), direct=args.direct.resolve(),
            lock=args.lock.resolve(), allowlist=args.allowlist.resolve(), repo_root=args.repo_root.resolve(),
            wheelhouse=args.wheelhouse.resolve(),
        )
    except LauncherContractError:
        print("HF training-smoke launcher setup failed: CONTRACT_INVALID", file=sys.stderr)
        return 1
    except (OSError, ValueError, subprocess.SubprocessError):
        print("HF training-smoke launcher setup failed: EXECUTION_FAILED", file=sys.stderr)
        return 1
    print(f"HF training-smoke launcher ready: {launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
