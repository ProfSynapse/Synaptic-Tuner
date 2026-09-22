"""Resolve live source/quote facts and compose a host without submitting training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_CEILING
import hashlib
from pathlib import Path
from typing import Mapping

from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.providers.modal.composition import (
    ModalSourceVerificationPorts,
    ModalVerificationPolicyV1,
    compose_modal_source_finalizer,
)
from tuner.execution.providers.modal.coordinator_adapter import ModalPreparationAdapter
from tuner.execution.providers.modal.coordinator_preflight import (
    AuthenticatedModalQuote,
    ModalOperationalPreflightAdapter,
    ModalQuoteBody,
    QUOTE_PURPOSE,
    TrustedEvidenceIdentity,
)
from tuner.execution.providers.modal.coordinator_retention import (
    ModalRetainedPreparation,
)
from tuner.execution.providers.modal.config import ModalProviderProfileV1
from tuner.project.context import ProjectContext
from tuner.runtime.offline_sft_worker import load_packaged_offline_sft_worker_manifest
from tuner.training import TrainingService, default_recipe_registry
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
from tuner.training.recipes import RecipeRegistry

from .authority import HMACAuthenticator
from .deployment import ModalChatOwnedDeployment, ModalChatScope
from .host import ModalChatHost, compose_modal_chat_host
from .remote import ModalChatGitRemote
from .requests import ModalChatTrainingRequests
from .resolution import ModalChatRichTrainingResolver
from .storage import ModalChatStorage


class ModalChatTrainingCompositionError(RuntimeError):
    """Closed failure from training-host composition."""


@dataclass(frozen=True, slots=True)
class ModalChatTrainingHost:
    host: ModalChatHost
    requests: ModalChatTrainingRequests
    preparation: ModalPreparationAdapter
    operational_preflight: ModalOperationalPreflightAdapter
    material: CoordinatorResolvedMaterial
    retained: ModalRetainedPreparation
    recipes: RecipeRegistry
    selection: object
    launch_source: object
    rate_calculation: bytes


@dataclass(frozen=True, slots=True)
class ModalChatRunIdentity:
    request_id: str
    run: TrainingRunRef
    request_digest: str

    def __post_init__(self):
        safe_ref(self.request_id, "request_id")
        if (
            type(self.request_digest) is not str
            or len(self.request_digest) != 64
            or any(char not in "0123456789abcdef" for char in self.request_digest)
        ):
            raise ValueError("exact request digest required")
        if type(self.run) is not TrainingRunRef:
            raise TypeError("exact preallocated run required")

    def allocate(self, *, project_ref, request_digest):
        if project_ref != self.run.project_ref or request_digest != self.request_digest:
            raise ModalChatTrainingCompositionError("modal_chat_identity_changed")
        return self.request_id, TrainingRunRef.from_dict(self.run.to_dict())


class _CaptureFinalizer:
    def __init__(self, finalizer):
        self._finalizer, self.result = finalizer, None

    def finalize(self, *args, **kwargs):
        if self.result is not None:
            raise ModalChatTrainingCompositionError(
                "modal_chat_source_already_finalized"
            )
        result = self._finalizer.finalize(*args, **kwargs)
        self.result = result
        return result


class _AllocatedIdentity:
    def __init__(self, delegate, project_ref, request_digest):
        self._project, self._digest = project_ref, request_digest
        self._value = delegate.allocate(
            project_ref=project_ref, request_digest=request_digest
        )

    def allocate(self, *, project_ref, request_digest):
        if (project_ref, request_digest) != (self._project, self._digest):
            raise ModalChatTrainingCompositionError("modal_chat_identity_changed")
        return self._value


def _training_rate_calculation(
    *,
    scope,
    binding,
    maximum_cost_minor_units,
):
    if (
        type(binding) is not dict
        or set(binding)
        != {
            "provider_id",
            "profile_ref",
            "account_ref",
            "namespace_ref",
            "resource_digest",
            "timeout_seconds",
        }
        or type(maximum_cost_minor_units) is not int
        or maximum_cost_minor_units < 0
    ):
        raise ValueError
    scope.observe(scope.client)
    workspace = scope.sdk.Workspace.from_context(client=scope.client)
    workspace.hydrate(scope.client)
    if (
        workspace.is_hydrated is not True
        or workspace.name != scope.binding.workspace_ref
    ):
        raise ValueError
    rates = workspace.billing.rates()
    if not isinstance(rates, Mapping):
        raise ValueError
    selected = {}
    for name in ("gpu_hour_cost_a10g", "cpu_hour_cost", "mem_gib_hour_cost"):
        value = rates.get(name)
        if type(value) is not Decimal or not value.is_finite() or value <= 0:
            raise ValueError
        selected[name] = value
    # Coordinator binding declares one A10 and a timeout, but no CPU/memory
    # quantity. The operator quote retains the observed rates and authorizes the
    # GPU nominal; it is explicitly not a provider billing cap.
    timeout = binding["timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise ValueError
    estimate = selected["gpu_hour_cost_a10g"] * Decimal(timeout) / Decimal(3600)
    estimate_minor = int((estimate * 100).to_integral_value(rounding=ROUND_CEILING))
    calculation = canonical_bytes(
        {
            "schema_version": "synaptic-modal-chat-training-rate-calculation/v1",
            "execution_binding": {
                "provider_id": binding["provider_id"],
                "profile_ref": binding["profile_ref"],
                "account_ref": binding["account_ref"],
                "namespace_ref": binding["namespace_ref"],
                "resource_digest": binding["resource_digest"],
                "timeout_seconds": timeout,
            },
            "rates_usd_per_hour": {key: str(value) for key, value in selected.items()},
            "gpu_only_timeout_estimate_usd": str(estimate),
            "gpu_only_timeout_estimate_minor_units": estimate_minor,
            "included_billing_dimensions": ["one-a10g-for-timeout"],
            "excluded_billing_dimensions": [
                "build",
                "cpu",
                "memory",
                "storage",
                "usage-beyond-timeout",
            ],
            "operator_authorization_minor_units": maximum_cost_minor_units,
            "authorization_semantics": "operator-maximum-not-provider-billing-cap",
        }
    )
    scope.observe(scope.client)
    return calculation, estimate_minor


def _training_quote(
    *,
    scope,
    binding,
    maximum_cost_minor_units,
    issued_at,
    expires_at,
    issuer_ref,
    audience_ref,
    challenge_nonce,
    key_ref,
    authenticator,
    clock,
):
    if clock.now_iso() != issued_at or type(maximum_cost_minor_units) is not int:
        raise ValueError
    calculation, estimate_minor = _training_rate_calculation(
        scope=scope,
        binding=binding,
        maximum_cost_minor_units=maximum_cost_minor_units,
    )
    if maximum_cost_minor_units < estimate_minor:
        raise ValueError
    evidence_ref = "rate-" + hashlib.sha256(calculation).hexdigest()
    raw = canonical_bytes(
        {
            "schema_version": "synaptic-modal-quote-body/v1",
            "provider_id": binding["provider_id"],
            "profile_ref": binding["profile_ref"],
            "account_ref": binding["account_ref"],
            "namespace_ref": binding["namespace_ref"],
            "resource_digest": binding["resource_digest"],
            "maximum_cost_minor_units": maximum_cost_minor_units,
            "currency": "USD",
            "issued_at": issued_at,
            "expires_at": expires_at,
            "issuer_ref": safe_ref(issuer_ref, "issuer_ref"),
            "evidence_ref": evidence_ref,
            "audience_ref": safe_ref(audience_ref, "audience_ref"),
            "challenge_nonce": safe_ref(challenge_nonce, "challenge_nonce"),
            "key_ref": safe_ref(key_ref, "key_ref"),
        }
    )
    body = ModalQuoteBody.parse(raw)
    quote = AuthenticatedModalQuote(
        body.canonical_bytes,
        authenticator.sign(QUOTE_PURPOSE, body.canonical_bytes, key_ref),
    )
    return (
        quote,
        TrustedEvidenceIdentity(issuer_ref, key_ref, audience_ref),
        calculation,
    )


def compose_modal_training_host(
    *,
    scope: ModalChatScope,
    deployment: ModalChatOwnedDeployment,
    profile: ModalProviderProfileV1,
    storage: ModalChatStorage,
    context: ProjectContext,
    request_json: str,
    project_ref: str,
    request_identity,
    dataset_project_path: Path,
    allowed_git_refs: frozenset[tuple[str, str]],
    load_in_4bit: bool,
    runtime_environment: dict[str, str],
    timeout_seconds: int,
    evidence_authenticator: HMACAuthenticator,
    evidence_key_ref: str,
    source_policy: ModalVerificationPolicyV1,
    replay,
    clock,
    created_at: str,
    quote_issuer_ref: str,
    quote_audience_ref: str,
    quote_challenge_nonce: str,
    quote_key_ref: str,
    maximum_cost_minor_units: int,
    log_terminal_policy: bytes,
    control_volume_id: str,
    artifact_volume_id: str,
    stage_key_ref: str,
    artifact_key: bytes,
    grant_key: bytes,
    receipt_key: bytes,
    invalid_evidence_key: bytes,
    assessment_key: bytes,
    cursor_key: bytes,
    observed_at: str,
) -> ModalChatTrainingHost:
    """Resolve and compose without deploying, submitting, or retrying training."""
    try:
        if (
            type(scope) is not ModalChatScope
            or type(deployment) is not ModalChatOwnedDeployment
            or type(profile) is not ModalProviderProfileV1
            or type(storage) is not ModalChatStorage
            or type(context) is not ProjectContext
        ):
            raise TypeError
        facade = deployment.facade()  # proves an acknowledged owned deployment
        selection = deployment.selection
        policy = parse_canonical_object(log_terminal_policy, name="log terminal policy")
        if (
            set(policy)
            != {
                "schema_version",
                "generation",
                "max_log_chunks",
                "max_chunk_bytes",
                "max_terminal_bytes",
            }
            or policy["schema_version"] != "synaptic-modal-log-terminal-policy/v2"
        ):
            raise ValueError
        finalizer = _CaptureFinalizer(
            compose_modal_source_finalizer(
                ModalSourceVerificationPorts(
                    ModalChatGitRemote(allowed_git_refs),
                    facade,
                    evidence_authenticator,
                    replay,
                    clock.now_iso,
                ),
                source_policy,
            )
        )
        request_digest = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        identities = _AllocatedIdentity(request_identity, project_ref, request_digest)
        run = identities._value[1]
        if type(run) is not TrainingRunRef:
            raise TypeError
        recipes = default_recipe_registry()
        resolver = ModalChatRichTrainingResolver(
            context=context,
            run=run,
            created_at=created_at,
            dataset_project_path=dataset_project_path,
            load_in_4bit=load_in_4bit,
            deployment=selection,
            source_finalizer=finalizer,
            audience_ref=source_policy.audience_ref,
        )
        service = TrainingService(context=context, resolver=resolver, recipes=recipes)
        requests = ModalChatTrainingRequests(
            service=service,
            recipes=recipes,
            project_ref=project_ref,
            identities=identities,
            request_catalog=storage.catalog(
                "training-requests", encode=bytes, decode=bytes
            ),
            material_catalog=storage.catalog(
                "training-materials", encode=bytes, decode=bytes
            ),
        )
        public_request = requests.load(request_json)
        resolved = requests.resolve(public_request)
        material_bytes = storage.catalog(
            "training-materials", encode=bytes, decode=bytes
        ).resolve(public_request.request_id)
        if finalizer.result is None or type(material_bytes) is not bytes:
            raise ValueError
        provisional = ModalPreparationAdapter(
            profile=profile,
            binding=scope.binding,
            resolved=resolved,
            runtime_environment=runtime_environment,
            quote_digest="0" * 64,
            timeout_seconds=timeout_seconds,
            clock=clock,
        )
        _, execution, _ = provisional._snapshot()
        quote_issued_at = clock.now_iso()
        quote_expires_at = (
            (
                datetime.fromisoformat(quote_issued_at.replace("Z", "+00:00"))
                + timedelta(seconds=300)
            )
            .isoformat()
            .replace("+00:00", "Z")
        )
        quote, quote_trust, calculation = _training_quote(
            scope=scope,
            binding={
                "provider_id": execution.provider.provider_id,
                "profile_ref": execution.provider.profile_ref,
                "account_ref": execution.scope.account_ref,
                "namespace_ref": execution.scope.namespace_ref,
                "resource_digest": execution.resource_digest,
                "timeout_seconds": selection.timeout_seconds,
            },
            maximum_cost_minor_units=maximum_cost_minor_units,
            issued_at=quote_issued_at,
            expires_at=quote_expires_at,
            issuer_ref=quote_issuer_ref,
            audience_ref=quote_audience_ref,
            challenge_nonce=quote_challenge_nonce,
            key_ref=quote_key_ref,
            authenticator=evidence_authenticator,
            clock=clock,
        )
        for name, payload in (
            ("training-rate-calculations", calculation),
            (
                "training-authenticated-quotes",
                canonical_bytes(
                    {
                        "body": parse_canonical_object(quote.body_bytes, name="quote"),
                        "tag_hex": quote.tag.hex(),
                    }
                ),
            ),
        ):
            catalog = storage.catalog(name, encode=bytes, decode=bytes)
            catalog.publish_if_absent(quote.body.quote_digest, payload)
            if catalog.resolve(quote.body.quote_digest) != payload:
                raise ValueError
        preparation = ModalPreparationAdapter(
            profile=profile,
            binding=scope.binding,
            resolved=resolved,
            runtime_environment=runtime_environment,
            quote_digest=quote.body.quote_digest,
            timeout_seconds=timeout_seconds,
            clock=clock,
        )
        verified = finalizer.result.deployment
        operational = ModalOperationalPreflightAdapter(
            preparation,
            execution_source_bytes=finalizer.result.execution_source.canonical_bytes,
            deployment_bytes=canonical_bytes(verified.to_dict()),
            quote=quote,
            control_volume_id=control_volume_id,
            artifact_volume_id=artifact_volume_id,
            facade=facade,
            authenticator=evidence_authenticator,
            clock=clock,
            source_trust=TrustedEvidenceIdentity(
                source_policy.source_issuer_ref,
                source_policy.source_key_ref,
                source_policy.audience_ref,
            ),
            deployment_trust=TrustedEvidenceIdentity(
                source_policy.deployment_issuer_ref,
                source_policy.deployment_key_ref,
                source_policy.audience_ref,
            ),
            quote_trust=quote_trust,
        )
        closure = load_packaged_offline_sft_worker_manifest().canonical_bytes
        retained = ModalRetainedPreparation(
            preparation.snapshot(),
            canonical_bytes(verified.to_dict()),
            material_bytes,
            recipes,
            log_terminal_policy,
            closure,
            control_volume_id,
            artifact_volume_id,
            stage_key_ref,
            private_dataset_bytes=resolver.private_dataset_bytes,
            prepared_input_source=resolver.prepared_input_source,
        )
        host = compose_modal_chat_host(
            preparation=preparation,
            operational_preflight=operational,
            facade=facade,
            deployment=verified,
            recipes=recipes,
            requests=requests,
            retained=retained,
            evidence_authenticator=evidence_authenticator,
            evidence_key_ref=evidence_key_ref,
            artifact_key=artifact_key,
            clock=clock,
            observed_at=observed_at,
            grant_key=grant_key,
            receipt_key=receipt_key,
            invalid_evidence_key=invalid_evidence_key,
            assessment_key=assessment_key,
            cursor_key=cursor_key,
            approved_cost_minor_units=maximum_cost_minor_units,
        )
        return ModalChatTrainingHost(
            host,
            requests,
            preparation,
            operational,
            retained.material,
            retained,
            recipes,
            selection,
            host.foundation_ports.launch_catalog,
            calculation,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ModalChatTrainingCompositionError(
            "modal_chat_training_composition_invalid"
        ) from None


__all__ = [
    "ModalChatTrainingCompositionError",
    "ModalChatTrainingHost",
    "ModalChatRunIdentity",
    "compose_modal_training_host",
]
