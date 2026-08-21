"""Capture candidate-only HF training-smoke launcher lock evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


def _authenticated_repo_root() -> Path:
    script = Path(__file__).resolve(strict=True)
    if script.is_symlink() or not script.is_file() or script.name != "capture_hf_training_smoke_launcher_lock.py":
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    root = script.parents[1]
    if script.parent.name != "scripts" or (root / "scripts" / script.name).resolve(strict=True) != script:
        raise RuntimeError("SCRIPT_IDENTITY_INVALID")
    return root


REPO_ROOT = _authenticated_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import setup_hf_training_smoke_launcher as setup  # noqa: E402


CANDIDATE_SCHEMA = "synaptic-hf-training-launcher-lock-candidate/v1"
TARGET = {
    "implementation": "CPython",
    "version": [3, 12, 7],
    "pip": "24.2",
    "platform": "win_amd64",
    "abi": "cp312",
}
MAX_OUTPUT_BYTES = 256 * 1024
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_WHEELS = 128
MAX_WHEEL_BYTES = 512 * 1024 * 1024
REASON_CODES = frozenset(
    {
        "INPUT_INVALID",
        "TARGET_INVALID",
        "COMMAND_FAILED",
        "REPORT_INVALID",
        "WHEELHOUSE_INVALID",
        "INSPECTION_INVALID",
        "OUTPUT_INVALID",
    }
)


class CandidateCaptureError(RuntimeError):
    def __init__(self, reason_code: str):
        if reason_code not in REASON_CODES:
            reason_code = "COMMAND_FAILED"
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    timeout_seconds: int = 300
    maximum_output_bytes: int = MAX_OUTPUT_BYTES


@dataclass(frozen=True)
class CommandResult:
    stdout: bytes = b""


Runner = Callable[[CommandSpec], CommandResult]


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_object(raw: bytes, *, maximum: int, reason: str) -> dict[str, object]:
    if not raw or len(raw) > maximum:
        raise CandidateCaptureError(reason)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise CandidateCaptureError(reason) from exc
    if not isinstance(value, dict):
        raise CandidateCaptureError(reason)
    return value


def subprocess_runner(spec: CommandSpec) -> CommandResult:
    try:
        completed = subprocess.run(
            list(spec.argv), cwd=spec.cwd, env=dict(spec.env), check=False,
            capture_output=True, timeout=spec.timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CandidateCaptureError("COMMAND_FAILED") from exc
    if completed.returncode != 0 or len(completed.stdout) > spec.maximum_output_bytes or len(completed.stderr) > spec.maximum_output_bytes:
        raise CandidateCaptureError("COMMAND_FAILED")
    return CommandResult(stdout=completed.stdout)


def _run(runner: Runner, argv: list[str], *, cwd: Path, env: Mapping[str, str]) -> bytes:
    try:
        result = runner(CommandSpec(tuple(argv), cwd, env))
    except CandidateCaptureError:
        raise
    except Exception as exc:
        raise CandidateCaptureError("COMMAND_FAILED") from exc
    if not isinstance(result, CommandResult) or len(result.stdout) > MAX_OUTPUT_BYTES:
        raise CandidateCaptureError("COMMAND_FAILED")
    return result.stdout


def _require_external_fresh(path: Path, *, repo_root: Path, reason: str) -> Path:
    try:
        resolved_parent = path.parent.resolve(strict=True)
        resolved = resolved_parent / path.name
        root = repo_root.resolve(strict=True)
    except OSError as exc:
        raise CandidateCaptureError(reason) from exc
    if path.exists() or path.is_symlink() or resolved == root or root in resolved.parents:
        raise CandidateCaptureError(reason)
    return resolved


def _launcher(venv: Path) -> Path:
    return venv / "Scripts" / "python.exe"


def _target_probe(python: Path, runner: Runner, cwd: Path, env: Mapping[str, str]) -> None:
    code = (
        "import json,platform,sys,pip;"
        "print(json.dumps({'implementation':platform.python_implementation(),"
        "'version':list(sys.version_info[:3]),'pip':pip.__version__,"
        "'platform':'win_amd64' if sys.platform=='win32' and platform.machine().lower() in ('amd64','x86_64') else '',"
        "'abi':'cp312' if sys.implementation.cache_tag.startswith('cpython-312') else ''},sort_keys=True))"
    )
    observed = _strict_object(
        _run(runner, [str(python), "-I", "-c", code], cwd=cwd, env=env),
        maximum=4096,
        reason="TARGET_INVALID",
    )
    if observed != TARGET:
        raise CandidateCaptureError("TARGET_INVALID")


def _report_to_lock(report_path: Path) -> tuple[bytes, dict[str, str], dict[str, str]]:
    try:
        raw = report_path.read_bytes()
    except OSError as exc:
        raise CandidateCaptureError("REPORT_INVALID") from exc
    report = _strict_object(raw, maximum=MAX_REPORT_BYTES, reason="REPORT_INVALID")
    if set(report) != {"version", "pip_version", "install", "environment"} or report["pip_version"] != "24.2":
        raise CandidateCaptureError("REPORT_INVALID")
    install = report["install"]
    if not isinstance(install, list) or not 1 <= len(install) <= MAX_WHEELS:
        raise CandidateCaptureError("REPORT_INVALID")
    locked: dict[str, tuple[str, str]] = {}
    filenames: dict[str, str] = {}
    requested_names: set[str] = set()
    for item in install:
        if not isinstance(item, dict) or set(item) != {"download_info", "is_direct", "is_yanked", "metadata", "requested"}:
            raise CandidateCaptureError("REPORT_INVALID")
        metadata = item["metadata"]
        download = item["download_info"]
        if not isinstance(metadata, dict) or not isinstance(download, dict):
            raise CandidateCaptureError("REPORT_INVALID")
        raw_name, version = metadata.get("name"), metadata.get("version")
        archive = download.get("archive_info")
        url = download.get("url")
        if not isinstance(raw_name, str) or not isinstance(version, str) or not isinstance(archive, dict) or not isinstance(url, str):
            raise CandidateCaptureError("REPORT_INVALID")
        name = setup.canonicalize_name(raw_name)
        hashes = archive.get("hashes")
        digest = hashes.get("sha256") if isinstance(hashes, dict) else None
        filename = url.rsplit("/", 1)[-1]
        if (
            not setup._PUBLIC_VERSION.fullmatch(version)
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not filename.endswith(".whl") or name in locked or name in setup.ML_DISTRIBUTIONS
            or item["is_direct"] is not False or item["is_yanked"] is not False
        ):
            raise CandidateCaptureError("REPORT_INVALID")
        locked[name] = (version, digest)
        filenames[name] = filename
        if item["requested"] is True:
            requested_names.add(name)
        elif item["requested"] is not False:
            raise CandidateCaptureError("REPORT_INVALID")
    direct = {setup.canonicalize_name(pin.split("==", 1)[0]): pin.split("==", 1)[1] for pin in setup.EXPECTED_DIRECT}
    if requested_names != set(direct) or any(locked.get(name, (None,))[0] != version for name, version in direct.items()):
        raise CandidateCaptureError("REPORT_INVALID")
    lock = "".join(
        f"{name}=={locked[name][0]} --hash=sha256:{locked[name][1]}\n"
        for name in sorted(locked)
    ).encode("ascii")
    return lock, {name: locked[name][0] for name in sorted(locked)}, filenames


def _validate_wheelhouse(
    wheelhouse: Path, *, filenames: Mapping[str, str], lock_bytes: bytes
) -> list[dict[str, object]]:
    hashes_by_name = {
        name_version.split("==", 1)[0]: line.rsplit("sha256:", 1)[1]
        for line in lock_bytes.decode("ascii").splitlines()
        for name_version in [line.split(" --hash=", 1)[0]]
    }
    expected_hashes = {filenames[name]: digest for name, digest in hashes_by_name.items()}
    try:
        entries = sorted(wheelhouse.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise CandidateCaptureError("WHEELHOUSE_INVALID") from exc
    if len(entries) != len(filenames) or {entry.name for entry in entries} != set(filenames.values()):
        raise CandidateCaptureError("WHEELHOUSE_INVALID")
    inventory: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        try:
            info = entry.lstat()
            raw = entry.read_bytes()
        except OSError as exc:
            raise CandidateCaptureError("WHEELHOUSE_INVALID") from exc
        digest = _sha256(raw)
        if entry.is_symlink() or not entry.is_file() or not 1 <= info.st_size <= MAX_WHEEL_BYTES or expected_hashes.get(entry.name) != digest or digest in seen:
            raise CandidateCaptureError("WHEELHOUSE_INVALID")
        seen.add(digest)
        inventory.append({"filename": entry.name, "sha256": digest, "size": info.st_size})
    return inventory


def _inspect_install(
    python: Path, *, runner: Runner, cwd: Path, env: Mapping[str, str], locked: Mapping[str, str]
) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json,platform,re,sys,pip;"
        "pairs=[];bad=[];"
        "[(pairs.append((re.sub(r'[-_.]+','-',x.metadata['Name']).lower(),x.version)),"
        "bad.append(x.metadata['Name']) if (x.read_text('direct_url.json') or '').strip() else None) for x in m.distributions()];"
        "assert not bad and len({n for n,v in pairs})==len(pairs);d=dict(pairs);"
        "print(json.dumps({'target':{'implementation':platform.python_implementation(),'version':list(sys.version_info[:3]),"
        "'pip':pip.__version__,'platform':'win_amd64' if sys.platform=='win32' and platform.machine().lower() in ('amd64','x86_64') else '',"
        "'abi':'cp312' if sys.implementation.cache_tag.startswith('cpython-312') else ''},'distributions':d},sort_keys=True))"
    )
    value = _strict_object(
        _run(runner, [str(python), "-I", "-c", code], cwd=cwd, env=env),
        maximum=MAX_OUTPUT_BYTES,
        reason="INSPECTION_INVALID",
    )
    distributions = value.get("distributions")
    expected = dict(sorted({**locked, "pip": "24.2"}.items()))
    if value.get("target") != TARGET or distributions != expected or set(expected) & setup.ML_DISTRIBUTIONS:
        raise CandidateCaptureError("INSPECTION_INVALID")
    return expected


def capture_candidate(
    *, python: Path, direct: Path, workspace: Path, output: Path,
    repo_root: Path = REPO_ROOT, runner: Runner = subprocess_runner,
) -> Path:
    output = _require_external_fresh(output, repo_root=repo_root, reason="OUTPUT_INVALID")
    workspace = _require_external_fresh(workspace, repo_root=repo_root, reason="INPUT_INVALID")
    try:
        setup.validate_direct_requirements(direct)
        env = setup.sanitized_environment()
    except (OSError, setup.LauncherContractError) as exc:
        raise CandidateCaptureError("INPUT_INVALID") from exc
    workspace.mkdir()
    resolver = workspace / "resolver"
    verifier = workspace / "verifier"
    wheelhouse = workspace / "wheelhouse"
    wheelhouse.mkdir()
    report = workspace / "report.json"
    _target_probe(python, runner, workspace, env)
    _run(runner, [str(python), "-I", "-m", "venv", str(resolver)], cwd=workspace, env=env)
    _target_probe(_launcher(resolver), runner, workspace, env)
    target = ["--only-binary=:all:", "--platform", "win_amd64", "--python-version", "3.12.7", "--implementation", "cp", "--abi", "cp312"]
    _run(
        runner,
        [str(_launcher(resolver)), "-I", "-m", "pip", "install", "--dry-run", "--ignore-installed", "--report", str(report), *target, "--requirement", str(direct)],
        cwd=workspace, env=env,
    )
    lock_bytes, locked, filenames = _report_to_lock(report)
    _run(
        runner,
        [str(_launcher(resolver)), "-I", "-m", "pip", "download", "--dest", str(wheelhouse), *target, "--requirement", str(direct)],
        cwd=workspace, env=env,
    )
    inventory = _validate_wheelhouse(wheelhouse, filenames=filenames, lock_bytes=lock_bytes)
    lock_path = workspace / "requirements-hf-training-smoke.lock"
    lock_path.write_bytes(lock_bytes)
    _run(runner, [str(python), "-I", "-m", "venv", str(verifier)], cwd=workspace, env=env)
    _run(
        runner,
        [str(_launcher(verifier)), "-I", "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), "--require-hashes", "--no-deps", "--requirement", str(lock_path)],
        cwd=workspace, env=env,
    )
    _run(runner, [str(_launcher(verifier)), "-I", "-m", "pip", "check"], cwd=workspace, env=env)
    installed = _inspect_install(_launcher(verifier), runner=runner, cwd=workspace, env=env, locked=locked)
    allowlist = {
        "schema_version": "synaptic-hf-training-launcher-installed/v1",
        "python": setup.EXPECTED_PYTHON,
        "lock_sha256": _sha256(lock_bytes),
        "distributions": installed,
    }
    allowlist_bytes = _canonical_json(allowlist)
    evidence = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "CANDIDATE",
        "target": TARGET,
        "direct_requirements": list(setup.EXPECTED_DIRECT),
        "lock_sha256": _sha256(lock_bytes),
        "allowlist_sha256": _sha256(allowlist_bytes),
        "report_sha256": _sha256(report.read_bytes()),
        "wheelhouse": inventory,
        "verification": {"offline_install": True, "pip_check": True, "inspection": True},
    }
    staging = output.with_name(output.name + ".partial")
    if staging.exists() or staging.is_symlink():
        raise CandidateCaptureError("OUTPUT_INVALID")
    try:
        staging.mkdir()
        (staging / "requirements-hf-training-smoke.lock.candidate").write_bytes(lock_bytes)
        (staging / "requirements-hf-training-smoke-installed.candidate.json").write_bytes(allowlist_bytes)
        (staging / "hf-training-smoke-launcher-lock.candidate.json").write_bytes(_canonical_json(evidence))
        shutil.copytree(wheelhouse, staging / "wheelhouse")
        if _validate_wheelhouse(
            staging / "wheelhouse", filenames=filenames, lock_bytes=lock_bytes
        ) != inventory:
            raise CandidateCaptureError("OUTPUT_INVALID")
        staging.rename(output)
    except OSError as exc:
        raise CandidateCaptureError("OUTPUT_INVALID") from exc
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct", type=Path, default=REPO_ROOT / "requirements-hf-training-smoke.direct.txt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = capture_candidate(
            python=args.python.resolve(), direct=args.direct.resolve(),
            workspace=args.workspace, output=args.output,
        )
    except CandidateCaptureError as exc:
        print(f"HF training-smoke launcher capture failed: {exc.reason_code}", file=sys.stderr)
        return 1
    print(f"HF training-smoke launcher candidate: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
