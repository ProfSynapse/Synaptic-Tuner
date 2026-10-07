"""Create the isolated Python 3.12 Hugging Face JP launcher environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_REQUIREMENTS = (
    "huggingface_hub==1.27.0",
    "jsonschema==4.23.0",
    "packaging==24.1",
    "python-dotenv==1.0.1",
    "PyYAML==6.0.2",
)
EXPECTED_DISTRIBUTIONS = {
    "huggingface_hub": "1.27.0",
    "jsonschema": "4.23.0",
    "packaging": "24.1",
    "python-dotenv": "1.0.1",
    "PyYAML": "6.0.2",
}


def validate_requirements(path: Path) -> None:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if tuple(lines) != EXPECTED_REQUIREMENTS:
        raise ValueError(
            "HF JP requirements must contain exactly the approved pinned launcher dependencies."
        )


def require_python_312(python: Path) -> None:
    completed = subprocess.run(
        [str(python), "-c", "import json,sys;print(json.dumps(list(sys.version_info[:2])))"],
        check=True,
        capture_output=True,
        text=True,
    )
    if json.loads(completed.stdout) != [3, 12]:
        raise ValueError("HF JP launcher requires Python 3.12 exactly.")


def verify_launcher(*, launcher_python: Path, repo_root: Path) -> None:
    """Prove protected imports and help routes without credentials or effects."""

    root = repo_root.resolve(strict=True)
    expected_root = Path(__file__).resolve().parents[1]
    if root != expected_root or not (root / "tuner" / "__main__.py").is_file():
        raise ValueError("HF JP launcher smoke requires the exact repository worktree.")
    environment = os.environ.copy()
    environment.pop("HF_TOKEN", None)
    environment.pop("HF_API_KEY", None)
    environment.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(root),
    )
    version_payload = json.dumps(EXPECTED_DISTRIBUTIONS, sort_keys=True)
    root_payload = json.dumps(str(root))
    import_check = (
        "import importlib.metadata as m,json,sys;"
        f"sys.path.insert(0,json.loads({root_payload!r}));"
        f"expected=json.loads({version_payload!r});"
        "actual={name:m.version(name) for name in expected};"
        "assert sys.version_info[:2]==(3,12);"
        "assert actual==expected;"
        "import tuner.handlers.hf_source_handler;"
        "import tuner.handlers.hf_smoke_handler;"
        "assert 'torch' not in sys.modules;"
        "assert 'transformers' not in sys.modules;"
        "assert 'unsloth' not in sys.modules"
    )
    subprocess.run(
        [str(launcher_python), "-I", "-c", import_check],
        check=True,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    for command in ("hf-source", "hf-smoke"):
        completed = subprocess.run(
            [str(launcher_python), "-m", "tuner", command, "--help"],
            check=True,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
        )
        if command not in completed.stdout:
            raise ValueError(f"HF JP protected route help is incomplete for {command}.")


def setup_launcher(
    *, python: Path, venv: Path, requirements: Path, repo_root: Path
) -> Path:
    validate_requirements(requirements)
    require_python_312(python)
    if venv.exists():
        raise FileExistsError("HF JP launcher target already exists; refusing to repair or overwrite it.")
    subprocess.run([str(python), "-m", "venv", str(venv)], check=True)
    launcher_python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [str(launcher_python), "-m", "pip", "install", "--requirement", str(requirements)],
        check=True,
    )
    verify_launcher(launcher_python=launcher_python, repo_root=repo_root)
    return launcher_python


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="Explicit Python 3.12 interpreter")
    parser.add_argument("--venv", type=Path, required=True, help="New isolated launcher directory")
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "requirements-hf-jp.txt",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Exact repository worktree used for protected-route smoke checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        launcher = setup_launcher(
            python=args.python.resolve(),
            venv=args.venv.resolve(),
            requirements=args.requirements.resolve(),
            repo_root=args.repo_root.resolve(),
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"HF JP launcher setup failed: {exc}", file=sys.stderr)
        return 1
    print(f"HF JP launcher ready: {launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
