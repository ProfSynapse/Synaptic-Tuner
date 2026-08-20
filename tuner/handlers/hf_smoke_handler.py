"""Protected CLI handler for one fixed Hugging Face bootstrap smoke."""

from __future__ import annotations

import os
import re
import stat
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from shared.experiment_tracking.experiment import load_experiment
from shared.experiment_tracking.service import TrackingService
from tuner.cloud.hf_bootstrap_smoke import WORKLOAD_SHA256
from tuner.cloud.hf_jobs import (
    HFBootstrapSmokeSubmission,
    observe_submitted_bootstrap_smoke,
    submit_approved_bootstrap_smoke,
)
from tuner.cloud.hf_provisioning import consume_hf_source_transport, load_canonical_json
from tuner.cloud.hf_run_approval import build_hf_run_approval, validate_hf_run_approval
from tuner.cloud.runtime_layout import build_runtime_layout
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler
from tuner.handlers.hf_source_handler import _require_explicit_env_file
from tuner.handlers.stages._util import hf_source_preparation_from_consumable


_MAX_SECRET_FILE_BYTES = 64 * 1024
_DECLARATION_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class _HFSecretFileClaim:
    root: Path
    path: Path
    identity: tuple[int, int, int, int, int]


class HFSmokeHandler(BaseHandler):
    """Approve, submit, or observe only the immutable bootstrap workload."""

    def __init__(
        self,
        args: Namespace | None = None,
        context=None,
        *,
        provider_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(args=args, context=context)
        self._provider_factory = provider_factory

    @property
    def name(self) -> str:
        return "hf-smoke"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        try:
            action = str(getattr(self.args, "subcommand", "") or "").strip().lower()
            if action == "approve":
                result = self._approve()
            elif action == "execute":
                result = self._execute()
            elif action == "observe":
                result = self._observe()
            else:
                raise CloudProviderError("hf-smoke requires approve, execute, or observe.")
            self.output(result, f"HF bootstrap smoke {action} completed.")
            return 0
        except Exception as exc:
            self.output_error(str(exc), code="HF_BOOTSTRAP_SMOKE_ERROR")
            return 1

    def _state(self):
        experiment_id = str(getattr(self.args, "experiment_id", "") or "").strip()
        if not experiment_id:
            raise CloudProviderError("hf-smoke requires --experiment-id.")
        tracking = TrackingService(project_context=self.context)
        experiment = load_experiment(experiment_id, tracking.base_dir)
        tracking.require_consumable_hf_transport(experiment)
        required = (
            experiment.source_lock_uri,
            experiment.source_lock_sha256,
            experiment.source_transport_uri,
            experiment.source_transport_sha256,
            experiment.provisioning_evidence_uri,
            experiment.provisioning_evidence_sha256,
        )
        if any(not isinstance(value, str) or not value for value in required):
            raise CloudProviderError("HF bootstrap smoke tracking bindings are incomplete.")
        descriptor_path = tracking.resolve_uri(experiment.source_transport_uri)
        evidence_path = tracking.resolve_uri(experiment.provisioning_evidence_uri)
        evidence = load_canonical_json(evidence_path, maximum_bytes=64 * 1024)
        consumed = consume_hf_source_transport(
            self.context,
            transport_root=descriptor_path.parent,
            descriptor_uri=experiment.source_transport_uri,
            source_lock_uri=experiment.source_lock_uri,
            evidence=evidence,
        )
        if consumed.prepared.descriptor_sha256 != experiment.source_transport_sha256:
            raise CloudProviderError("HF bootstrap smoke descriptor tracking digest changed.")
        if consumed.evidence_sha256 != experiment.provisioning_evidence_sha256:
            raise CloudProviderError("HF bootstrap smoke evidence tracking digest changed.")
        preparation = hf_source_preparation_from_consumable(
            consumed,
            context=self.context,
            runtime_layout=build_runtime_layout(self.context),
            provisioning_evidence_uri=experiment.provisioning_evidence_uri,
        )
        return tracking, experiment, preparation

    def _approve(self) -> dict[str, object]:
        tracking, experiment, preparation = self._state()
        descriptor = preparation.consumable_transport.prepared.descriptor
        approval = build_hf_run_approval(
            experiment_id=experiment.experiment_id,
            run_id=preparation.source_lock.run_id,
            descriptor_uri=preparation.descriptor_uri,
            descriptor_sha256=preparation.descriptor_sha256,
            provisioning_evidence_uri=preparation.provisioning_evidence_uri,
            provisioning_evidence_sha256=preparation.provisioning_evidence_sha256,
            source_lock_uri=preparation.source_lock_uri,
            source_lock_sha256=preparation.source_lock_sha256,
            bundle_sha256=descriptor["bundle"]["content_sha256"],
            capsule_manifest_sha256=descriptor["capsule"]["manifest"]["sha256"],
            checkout_policy_sha256=descriptor["checkout_policy"]["sha256"],
            hardware_flavor="cpu-basic",
            user_authorization_reference=_required_arg(self.args, "authorization_reference"),
            issued_at=_required_arg(self.args, "issued_at"),
            expires_at=_required_arg(self.args, "expires_at"),
            hourly_price_usd=_required_arg(self.args, "hourly_price_usd"),
            projected_cost_usd=_required_arg(self.args, "projected_cost_usd"),
            quoted_at=_required_arg(self.args, "quoted_at"),
        )
        tracking.record_hf_run_approval(experiment, approval)
        return {
            "status": "APPROVED",
            "authorization_id": approval.authorization_id,
            "approval_id": approval.approval_id,
            "workload_sha256": WORKLOAD_SHA256,
            "submitted": False,
        }

    def _execute(self) -> dict[str, object]:
        tracking, experiment, preparation = self._state()
        if experiment.hf_submission_state != "APPROVED":
            raise CloudProviderError("HF bootstrap smoke execute requires exact APPROVED state.")
        approval_path = tracking.resolve_uri(str(experiment.hf_run_approval_uri))
        approval = validate_hf_run_approval(load_canonical_json(approval_path, maximum_bytes=64 * 1024))
        secret_file = _preflight_secret_file(
            getattr(self.args, "env_file", None), context=self.context
        )

        def token_factory() -> str:
            return _read_claimed_hf_token(secret_file)

        submission = submit_approved_bootstrap_smoke(
            tracking_service=tracking,
            experiment=experiment,
            approval=approval,
            preparation=preparation,
            token_factory=token_factory,
            provider_factory=self._provider_factory,
        )
        return {
            "status": "SUBMITTED",
            "namespace": submission.namespace,
            "job_id": submission.job_id,
            "authorization_id": submission.authorization_id,
            "retry_allowed": False,
        }

    def _observe(self) -> dict[str, object]:
        tracking, experiment, _ = self._state()
        if experiment.hf_submission_state != "SUBMITTED":
            raise CloudProviderError("HF bootstrap smoke observe requires exact SUBMITTED state.")
        event_path = tracking.resolve_uri(str(experiment.hf_submission_event_uri))
        event = load_canonical_json(event_path, maximum_bytes=64 * 1024)
        provider_job = event.get("provider_job")
        if not isinstance(provider_job, dict) or set(provider_job) != {"namespace", "job_id"}:
            raise CloudProviderError("HF bootstrap smoke submitted job identity is unavailable.")
        secret_file = _preflight_secret_file(
            getattr(self.args, "env_file", None), context=self.context
        )
        approval_path = tracking.resolve_uri(str(experiment.hf_run_approval_uri))
        approval = validate_hf_run_approval(
            load_canonical_json(approval_path, maximum_bytes=64 * 1024)
        )

        def token_factory() -> str:
            return _read_claimed_hf_token(secret_file)

        observation = observe_submitted_bootstrap_smoke(
            HFBootstrapSmokeSubmission(
                namespace=str(provider_job["namespace"]),
                job_id=str(provider_job["job_id"]),
                authorization_id=str(experiment.hf_authorization_id),
            ),
            tracking_service=tracking,
            experiment=experiment,
            approval=approval,
            token_factory=token_factory,
            provider_factory=self._provider_factory,
        )
        return {
            "status": observation.stage,
            "namespace": observation.namespace,
            "job_id": observation.job_id,
            "elapsed_seconds": observation.elapsed_seconds,
            "cancel_attempted": observation.cancel_attempted,
            "result": observation.result,
        }


def _required_arg(args: Namespace | None, name: str) -> str:
    value = str(getattr(args, name, "") or "").strip()
    if not value:
        raise CloudProviderError(f"hf-smoke approve requires --{name.replace('_', '-')}.")
    return value


def _preflight_secret_file(value: object, *, context) -> _HFSecretFileClaim:
    """Bind a trusted explicit file without reading its secret bytes."""

    if "HF_TOKEN" in os.environ or "HF_API_KEY" in os.environ:
        raise CloudProviderError(
            "HF smoke rejects ambient Hugging Face credentials; authority must come only from the explicit file."
        )
    path = _require_explicit_env_file(value, context=context)
    roots = tuple(
        dict.fromkeys(
            Path(root).resolve(strict=True)
            for root in (context.project_root, context.config_root)
        )
    )
    root = max(
        (candidate for candidate in roots if _is_relative_to(path, candidate)),
        key=lambda candidate: len(candidate.parts),
    )
    _assert_link_free_chain(root, path)
    try:
        info = path.lstat()
    except OSError:
        raise CloudProviderError("HF smoke secret file is unavailable.") from None
    if info.st_size <= 0 or info.st_size > _MAX_SECRET_FILE_BYTES:
        raise CloudProviderError("HF smoke secret file must be non-empty and bounded.")
    return _HFSecretFileClaim(root=root, path=path, identity=_file_identity(info))


def _read_claimed_hf_token(claim: _HFSecretFileClaim) -> str:
    """Read the exact preflighted file once, without environment mutation."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        _assert_link_free_chain(claim.root, claim.path)
        descriptor = os.open(claim.path, flags)
        try:
            opened_before = os.fstat(descriptor)
            if _file_identity(opened_before) != claim.identity:
                raise CloudProviderError("HF smoke secret file changed after preflight.")
            chunks: list[bytes] = []
            remaining = _MAX_SECRET_FILE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = claim.path.lstat()
        _assert_link_free_chain(claim.root, claim.path)
    except CloudProviderError:
        raise
    except OSError:
        raise CloudProviderError("HF smoke secret file could not be read safely.") from None
    if len(raw) > _MAX_SECRET_FILE_BYTES:
        raise CloudProviderError("HF smoke secret file exceeds its bound.")
    if (
        _file_identity(opened_after) != claim.identity
        or _file_identity(current) != claim.identity
        or not stat.S_ISREG(opened_after.st_mode)
    ):
        raise CloudProviderError("HF smoke secret file changed during authorization.")
    try:
        document = raw.decode("utf-8")
    except UnicodeError:
        raise CloudProviderError("HF smoke secret file must be valid UTF-8.") from None
    return _parse_strict_hf_token(document)


def _parse_strict_hf_token(document: str) -> str:
    """Accept comments/blanks and exactly one deterministic HF_TOKEN assignment."""

    if "\x00" in document:
        raise CloudProviderError("HF smoke secret file contains invalid dotenv syntax.")
    token: str | None = None
    for line in document.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _DECLARATION_RE.fullmatch(line)
        if match is None:
            raise CloudProviderError("HF smoke secret file contains invalid dotenv syntax.")
        key, encoded = match.groups()
        if key != "HF_TOKEN":
            raise CloudProviderError("HF smoke secret file may declare only HF_TOKEN.")
        if token is not None:
            raise CloudProviderError("HF smoke secret file declares HF_TOKEN more than once.")
        token = _decode_token_value(encoded)
    if token is None or not token:
        raise CloudProviderError("HF smoke secret file must declare exactly one non-empty HF_TOKEN.")
    return token


def _decode_token_value(encoded: str) -> str:
    value = encoded.strip()
    if not value:
        raise CloudProviderError("HF smoke secret file must declare a non-empty HF_TOKEN.")
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote or quote in value[1:-1] or "\\" in value:
            raise CloudProviderError("HF smoke secret file contains invalid quoted syntax.")
        value = value[1:-1]
    elif any(character.isspace() or character in "#'\"\\" for character in value):
        raise CloudProviderError("HF smoke secret file contains invalid unquoted syntax.")
    if not value or not _TOKEN_RE.fullmatch(value):
        raise CloudProviderError("HF smoke secret file must declare a non-empty HF_TOKEN.")
    return value


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)


def _assert_link_free_chain(root: Path, path: Path) -> None:
    current = root
    items = [root]
    for part in path.relative_to(root).parts:
        current = current / part
        items.append(current)
    for index, item in enumerate(items):
        try:
            info = item.lstat()
        except OSError:
            raise CloudProviderError("HF smoke secret path is unavailable.") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CloudProviderError("HF smoke secret path cannot traverse links or reparse points.")
        final = index == len(items) - 1
        if final and not stat.S_ISREG(info.st_mode):
            raise CloudProviderError("HF smoke secret path must identify a regular file.")
        if not final and not stat.S_ISDIR(info.st_mode):
            raise CloudProviderError("HF smoke secret path has an invalid parent chain.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = ["HFSmokeHandler"]
