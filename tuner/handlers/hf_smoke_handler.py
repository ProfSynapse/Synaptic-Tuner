"""Protected CLI handler for one fixed Hugging Face bootstrap smoke."""

from __future__ import annotations

from argparse import Namespace
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
from tuner.core.exceptions import CloudProviderError
from tuner.handlers._hf_secret_file import (
    HFSecretFileClaim,
    preflight_hf_secret_file,
    read_claimed_hf_token,
)
from tuner.handlers.base import BaseHandler
from tuner.handlers.hf_source_handler import _external_runtime_layout, _require_external_base_dir
from tuner.handlers.stages._util import hf_source_preparation_from_consumable


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
            crossover = (
                ("--source-config", getattr(self.args, "source_config", None)),
                ("--actor", getattr(self.args, "actor", None)),
                ("--authority", getattr(self.args, "authority", None)),
            )
            selected = [flag for flag, value in crossover if value is not None]
            if selected:
                raise CloudProviderError(f"hf-smoke does not accept {', '.join(selected)}.")
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
        base_dir = _require_external_base_dir(
            getattr(self.args, "base_dir", None), context=self.context
        )
        tracking = TrackingService(base_dir=base_dir)
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
            runtime_layout=_external_runtime_layout(self.context, base_dir),
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
        secret_file = preflight_hf_secret_file(
            getattr(self.args, "env_file", None), context=self.context
        )

        def token_factory() -> str:
            return read_claimed_hf_token(secret_file)

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
        secret_file = preflight_hf_secret_file(
            getattr(self.args, "env_file", None), context=self.context
        )
        approval_path = tracking.resolve_uri(str(experiment.hf_run_approval_uri))
        approval = validate_hf_run_approval(
            load_canonical_json(approval_path, maximum_bytes=64 * 1024)
        )

        def token_factory() -> str:
            return read_claimed_hf_token(secret_file)

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


# Compatibility aliases keep the audited test seam while sharing one implementation.
_HFSecretFileClaim = HFSecretFileClaim
_preflight_secret_file = preflight_hf_secret_file
_read_claimed_hf_token = read_claimed_hf_token

__all__ = ["HFSmokeHandler"]
