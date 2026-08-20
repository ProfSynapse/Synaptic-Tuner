"""Protected handler for one exact HF source provisioning operation."""

from __future__ import annotations

import os
import stat
from argparse import Namespace
from pathlib import Path
from typing import Callable

from shared.experiment_tracking.experiment import load_experiment
from shared.experiment_tracking.service import TrackingService
from tuner.cloud.hf_provider_adapter import HFProviderAdapter, load_hf_jp_provider
from tuner.cloud.hf_provisioning import canonical_json_bytes, consume_hf_source_transport
from tuner.cloud.hf_provisioning_operator import (
    HFProvisioningOutcome,
    provision_hf_source_transport,
)
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler


class HFSourceHandler(BaseHandler):
    """Provision, acknowledge, and consume one PREPARED descriptor; never submit."""

    def __init__(
        self,
        args: Namespace | None = None,
        context=None,
        *,
        provider_factory: Callable[..., HFProviderAdapter] = load_hf_jp_provider,
    ) -> None:
        super().__init__(args=args, context=context)
        self._provider_factory = provider_factory

    @property
    def name(self) -> str:
        return "hf-source"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        try:
            result = self.provision()
            if not result.succeeded:
                assert result.failure is not None
                self.output_error(result.failure.message, code=result.failure.code)
                return 1
            self.output(
                {
                    "status": "CONSUMABLE",
                    "evidence_sha256": result.evidence_sha256,
                    "provider_mutated": result.mutated,
                    "submitted": False,
                },
                "HF source transport is verified as CONSUMABLE; no job was submitted.",
            )
            return 0
        except Exception as exc:
            self.output_error(str(exc), code="HF_SOURCE_PROVISIONING_ERROR")
            return 1

    def provision(self) -> HFProvisioningOutcome:
        args = self.args
        if args is None:
            raise CloudProviderError("HF source provisioning requires explicit arguments.")
        experiment_id = str(getattr(args, "experiment_id", "") or "").strip()
        actor = str(getattr(args, "actor", "") or "").strip()
        authority = str(getattr(args, "authority", "operator") or "operator").strip()
        env_value = getattr(args, "env_file", None)
        if not experiment_id or not actor:
            raise CloudProviderError("HF source provisioning requires experiment_id and actor.")
        env_file = _require_explicit_env_file(env_value, context=self.context)
        token = _resolve_explicit_hf_token(env_file)

        tracking = TrackingService(project_context=self.context)
        experiment = load_experiment(experiment_id, tracking.base_dir)
        if experiment.source_transport_state != "PREPARED":
            raise CloudProviderError("HF source provisioning requires exact PREPARED tracking state.")
        if not experiment.source_transport_uri or not experiment.source_transport_sha256:
            raise CloudProviderError("HF PREPARED tracking state is missing its descriptor binding.")
        if not experiment.source_lock_uri:
            raise CloudProviderError("HF PREPARED tracking state is missing its SourceLock binding.")
        descriptor_path = tracking.resolve_uri(experiment.source_transport_uri)
        transport_root = descriptor_path.parent
        if descriptor_path.name != "descriptor.json":
            raise CloudProviderError("HF tracked source transport must identify descriptor.json.")

        provider = self._provider_factory(token=token)
        outcome = provision_hf_source_transport(
            self.context,
            transport_root=transport_root,
            descriptor_uri=experiment.source_transport_uri,
            source_lock_uri=experiment.source_lock_uri,
            provider=provider,
            actor=actor,
            authority=authority,
        )
        if not outcome.succeeded:
            return outcome
        assert outcome.evidence is not None and outcome.evidence_sha256 is not None

        evidence_path = transport_root / "provisioning-evidence.json"
        evidence_uri = tracking.tracking_uri(evidence_path)
        _persist_immutable(evidence_path, canonical_json_bytes(outcome.evidence))
        tracking.record_provisioning_acknowledged(
            experiment,
            uri=evidence_uri,
            sha256=outcome.evidence_sha256,
        )
        consumed = consume_hf_source_transport(
            self.context,
            transport_root=transport_root,
            descriptor_uri=experiment.source_transport_uri,
            source_lock_uri=experiment.source_lock_uri,
            evidence=outcome.evidence,
        )
        if consumed.evidence_sha256 != outcome.evidence_sha256:
            raise CloudProviderError("HF JP evidence digest changed before local consumption.")
        tracking.mark_source_transport_consumable(experiment)
        return outcome


def _require_explicit_env_file(value: object, *, context) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CloudProviderError("HF JP requires an explicit --env-file selection.")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = context.invocation_cwd / raw
    path = Path(os.path.abspath(raw))
    allowed_roots = tuple(
        dict.fromkeys(
            Path(root).resolve(strict=True)
            for root in (context.project_root, context.config_root)
        )
    )
    containing = [root for root in allowed_roots if _is_relative_to(path, root)]
    if not containing:
        raise CloudProviderError(
            "HF JP environment selection must remain within the project/config boundary."
        )
    root = max(containing, key=lambda item: len(item.parts))
    _require_link_free_regular_path(root, path)
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise CloudProviderError("HF JP environment selection cannot traverse links.")
    return resolved


def _resolve_explicit_hf_token(path: Path) -> str:
    """Read exactly HF_TOKEN without mutating or consulting process authority."""

    try:
        from dotenv.parser import parse_stream
    except ImportError:
        raise CloudProviderError("HF JP cannot safely parse the explicit environment file.") from None

    selected: dict[str, str | None] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for binding in parse_stream(stream):
                if binding.error:
                    raise CloudProviderError("HF JP environment file contains invalid dotenv syntax.")
                key = binding.key
                if key not in {"HF_TOKEN", "HF_API_KEY"}:
                    continue
                if key in selected:
                    raise CloudProviderError(
                        f"HF JP environment file declares {key} more than once."
                    )
                selected[key] = binding.value
    except CloudProviderError:
        raise
    except (OSError, UnicodeError):
        raise CloudProviderError("HF JP environment file could not be read safely.") from None

    if "HF_API_KEY" in selected:
        raise CloudProviderError("HF JP rejects HF_API_KEY; declare exactly HF_TOKEN.")
    token = selected.get("HF_TOKEN")
    if not isinstance(token, str) or not token.strip():
        raise CloudProviderError(
            "The explicitly selected environment file must declare a non-empty HF_TOKEN."
        )
    if "HF_TOKEN" in os.environ or "HF_API_KEY" in os.environ:
        raise CloudProviderError(
            "HF JP rejects ambient Hugging Face credentials; authority must come only from the explicit file."
        )
    return token.strip()


def _require_link_free_regular_path(root: Path, path: Path) -> None:
    items = [root]
    cursor = root
    for part in path.relative_to(root).parts:
        cursor = cursor / part
        items.append(cursor)
    for index, item in enumerate(items):
        try:
            info = item.lstat()
        except OSError:
            raise CloudProviderError("HF JP environment selection is unavailable.") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CloudProviderError("HF JP environment selection cannot traverse links.")
        is_final = index == len(items) - 1
        if is_final and not stat.S_ISREG(info.st_mode):
            raise CloudProviderError("HF JP environment selection must be a regular file.")
        if not is_final and not stat.S_ISDIR(info.st_mode):
            raise CloudProviderError("HF JP environment selection has an invalid directory chain.")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _persist_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError:
            raise CloudProviderError("HF JP evidence artifact is unavailable.") from None
        if existing != payload:
            raise CloudProviderError("HF JP evidence artifact already exists with different bytes.")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise CloudProviderError("HF JP evidence artifact raced with different bytes.") from None
    except OSError:
        raise CloudProviderError("HF JP evidence artifact could not be persisted.") from None


__all__ = ["HFSourceHandler"]
