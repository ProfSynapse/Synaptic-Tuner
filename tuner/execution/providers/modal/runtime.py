"""Concrete remote-only dual-clone and process ports for Modal v1."""
from __future__ import annotations

import hashlib
import base64
import binascii
import hmac
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tuner.project.execution_source import ExecutionSourceV1

from .worker_ports import ModalProcessResult, ModalRemotePhaseError
from .resolution import ModalDeploymentSelectionV1
from .config import ModalRuntimeLockV1


class EnvironmentHmacAuthenticator:
    """Authenticate remote control records from one explicitly allowed secret."""

    __slots__ = ("_environment_key", "_key_ref")

    def __init__(self, *, environment_key: str, key_ref: str) -> None:
        if not isinstance(environment_key, str) or not environment_key or not environment_key.isupper():
            raise ValueError("evidence environment key must be explicit")
        from ...contracts import safe_ref
        self._environment_key = environment_key
        self._key_ref = safe_ref(key_ref, "key_ref")

    def _key(self, key_ref: str) -> bytes:
        if key_ref != self._key_ref:
            raise ValueError("evidence key reference mismatch")
        value = os.environ.get(self._environment_key)
        if not isinstance(value, str) or not value or not value.isascii():
            raise ValueError("evidence key is unavailable")
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("evidence key is invalid") from None
        if len(decoded) != 32 or base64.b64encode(decoded).decode("ascii") != value:
            raise ValueError("evidence key is invalid")
        return decoded

    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes:
        if not isinstance(purpose, str) or not purpose or not isinstance(payload, bytes):
            raise TypeError("evidence purpose and payload are required")
        return hmac.new(self._key(key_ref), purpose.encode("ascii") + b"\0" + payload, hashlib.sha256).digest()

    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool:
        if not isinstance(tag, bytes):
            return False
        return hmac.compare_digest(self.sign(purpose, payload, key_ref), tag)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_executable(left: str, right: str) -> bool:
    """Return whether two provider-visible paths name the same binary."""
    try:
        return Path(left).samefile(Path(right))
    except OSError:
        return False


class GitDualCloneMaterializer:
    """Clone and independently reverify the one accepted dual-clone topology."""

    __slots__ = ("_run",)

    def __init__(
        self,
        *,
        command_runner: Callable[[Sequence[str]], bytes] | None = None,
    ) -> None:
        self._run = command_runner or self._subprocess

    @staticmethod
    def _subprocess(argv: Sequence[str]) -> bytes:
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": "/tmp/synaptic-modal-git-home",
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
        }
        completed = subprocess.run(
            tuple(argv),
            shell=False,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=600,
            env=environment,
        )
        if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024:
            raise ValueError("remote source command failed")
        return completed.stdout

    def _git(self, *arguments: str) -> str:
        try:
            value = self._run(("git", *arguments))
        except Exception:
            raise ValueError("remote source verification failed") from None
        if not isinstance(value, bytes) or len(value) > 1024 * 1024:
            raise ValueError("remote source verification failed")
        try:
            return value.decode("utf-8").strip()
        except UnicodeError:
            raise ValueError("remote source verification failed") from None

    def _clone(self, url: str, commit: str, destination: Path) -> None:
        self._git("clone", "--no-checkout", "--filter=blob:none", "--", url, str(destination))
        self._git("-C", str(destination), "checkout", "--detach", commit)
        if self._git("-C", str(destination), "rev-parse", "HEAD").lower() != commit.lower():
            raise ValueError("remote source commit mismatch")
        if self._git("-C", str(destination), "remote", "get-url", "origin") != url:
            raise ValueError("remote source origin mismatch")
        if self._git("-C", str(destination), "status", "--porcelain=v1", "--untracked-files=all"):
            raise ValueError("remote source checkout is not clean")

    def prepare_and_verify(
        self,
        source: ExecutionSourceV1,
        deployment: ModalDeploymentSelectionV1,
    ) -> None:
        if type(source) is not ExecutionSourceV1 or type(deployment) is not ModalDeploymentSelectionV1:
            raise TypeError("canonical source and deployment are required")
        project = Path(source.roots["project"])
        engine = Path(source.roots["engine"])
        expected_run_root = Path("/workspace/run") / source.run_id
        writable = {name: Path(source.roots[name]) for name in ("artifacts", "state", "tracking", "cache", "tmp")}
        if (
            project != Path("/workspace/project")
            or engine != Path("/workspace/engine")
            or project == engine
            or any(path.parent != expected_run_root for path in writable.values())
            or set(path.name for path in writable.values()) != {"artifacts", "state", "tracking", "cache", "tmp"}
        ):
            raise ModalRemotePhaseError(124, "source_topology_invalid")
        if project.exists() or engine.exists() or expected_run_root.exists():
            raise ModalRemotePhaseError(122, "artifact_layout_collision")
        actual_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if (
            sys.implementation.name != "cpython"
            or actual_version != source.python_version
            or not _same_executable(sys.executable, source.python_executable)
            or _file_digest(Path(source.python_executable)) != source.python_executable_digest
        ):
            raise ModalRemotePhaseError(121, "runtime_identity_mismatch")
        try:
            self._clone(
                source.project_source.location.canonical_url,
                source.project_source.commit,
                project,
            )
        except Exception:
            raise ModalRemotePhaseError(124, "project_clone_failed") from None
        try:
            self._clone(
                source.engine_source.location.canonical_url,
                source.engine_source.commit,
                engine,
            )
        except Exception:
            raise ModalRemotePhaseError(124, "engine_clone_failed") from None
        gitlink = self._git(
            "-C", str(project), "ls-tree", "HEAD", "--", source.engine_submodule_path
        ).split()
        if len(gitlink) < 3 or gitlink[0] != "160000" or gitlink[1] != "commit" or gitlink[2].lower() != source.engine_source.commit.lower():
            raise ModalRemotePhaseError(124, "engine_gitlink_mismatch")
        try:
            runtime_lock = ModalRuntimeLockV1.packaged()
            runtime_lock.validate_selection(deployment)
        except Exception:
            raise ModalRemotePhaseError(121, "runtime_lock_mismatch") from None
        checks = {
            engine / member["path"]: member["sha256"]
            for member in runtime_lock.document["locked_files"].values()
        }
        if any(not path.is_file() or _file_digest(path) != expected for path, expected in checks.items()):
            raise ModalRemotePhaseError(124, "locked_source_mismatch")
        try:
            expected_run_root.mkdir(parents=True, exist_ok=False)
            for path in writable.values():
                path.mkdir(exist_ok=False)
        except Exception:
            raise ModalRemotePhaseError(122, "artifact_layout_failed") from None


class SubprocessSftRunner:
    """Invoke the fixed runtime without a shell and without returning secret-bearing output."""

    __slots__ = ("_secret_keys", "_model_token_key", "_timeout")

    def __init__(self, *, secret_keys: tuple[str, ...], model_token_key: str, timeout_seconds: int) -> None:
        if not secret_keys or len(secret_keys) != len(set(secret_keys)):
            raise ValueError("exact runtime secret keys are required")
        self._secret_keys = tuple(secret_keys)
        if model_token_key not in self._secret_keys:
            raise ValueError("model credential must be a named runtime secret")
        self._model_token_key = model_token_key
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400:
            raise ValueError("runtime timeout must be bounded")
        self._timeout = timeout_seconds

    def _prepare_model(self, environment: dict[str, str], stdin: bytes) -> None:
        from .model_snapshot import prepare_model_snapshot

        try:
            model = json.loads(stdin)["configuration"]["document"]["model"]
            lexical_root = Path("/workspace/run")
            lexical_cache = Path(environment["SYNAPTIC_CACHE_ROOT"])
            relative = lexical_cache.relative_to(lexical_root)
            if len(relative.parts) != 2 or relative.parts[1] != "cache" or not relative.parts[0].startswith("run-"):
                raise ValueError("model cache binding differs")
            expected = lexical_cache / "model" / ("models--" + model["ref"].replace("/", "--")) / "snapshots" / model["revision"]
            if environment["SYNAPTIC_MODEL_SNAPSHOT"] != str(expected):
                raise ValueError("model snapshot binding differs")
            # Only the authenticated mount boundary may resolve elsewhere.
            physical_root = lexical_root.resolve(strict=True)
            cache = physical_root / relative
            persistent = physical_root / "model-cache"
            persistent.mkdir(exist_ok=True)
            scratch = Path("/workspace/model-preparation")
            scratch.mkdir(exist_ok=True)
            prepare_model_snapshot(
                model_ref=model["ref"], revision=model["revision"],
                token=os.environ[self._model_token_key],
                persistent_root=persistent, destination_root=cache,
                scratch_root=scratch,
            )
        except Exception:
            raise ModalRemotePhaseError(124, "model_preparation_failed") from None

    def run(
        self,
        argv: tuple[str, str, str],
        *,
        cwd: str,
        environment: dict[str, str],
        stdin: bytes,
        commit_prepared: Callable[[], None],
    ) -> ModalProcessResult:
        if len(argv) != 3 or argv[2] != "--canonical-workload-stdin":
            raise ValueError("runtime command is not fixed")
        process_environment = dict(environment)
        for key in self._secret_keys:
            value = os.environ.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ModalRemotePhaseError(120, "credential_unavailable")
            process_environment.pop(key, None)
        self._prepare_model(process_environment, stdin)
        try:
            commit_prepared()
        except Exception:
            raise ModalRemotePhaseError(122, "model_cache_commit_failed") from None
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=process_environment,
                input=stdin,
                shell=False,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._timeout,
            )
        except Exception:
            raise ModalRemotePhaseError(123, "trainer_invocation_failed") from None
        if completed.returncode == 0:
            return ModalProcessResult(0)
        returncode, diagnostic_code = {
            2: (121, "runtime_unclassified_rejection"),
            20: (124, "runtime_workload_rejected"),
            21: (122, "runtime_artifact_precondition"),
            22: (124, "runtime_invocation_rejected"),
            23: (123, "runtime_trainer_failed"),
            24: (123, "runtime_evidence_rejected"),
            25: (122, "runtime_artifact_rejected"),
            30: (124, "runtime_workload_document_rejected"),
            31: (124, "runtime_workload_engine_rejected"),
            32: (124, "runtime_workload_schema_rejected"),
            33: (124, "runtime_workload_reconstruction_rejected"),
            34: (124, "runtime_workload_fingerprint_rejected"),
            35: (124, "runtime_workload_roots_rejected"),
        }.get(completed.returncode, (123, "trainer_nonzero"))
        return ModalProcessResult(returncode, diagnostic_code=diagnostic_code)


__all__ = ["EnvironmentHmacAuthenticator", "GitDualCloneMaterializer", "SubprocessSftRunner"]
