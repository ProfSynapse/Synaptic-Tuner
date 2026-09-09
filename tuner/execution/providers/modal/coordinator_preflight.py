"""Operational preflight facts for the Foundation-native Modal coordinator."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from synaptic_tuner.api.v1.planning import TrainingPlan
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement, TrainingPreflight
from tuner.execution.coordinator_v1.ports import CoordinatorClockPortV1
from tuner.execution.evidence import (
    DEPLOYMENT_EVIDENCE_POLICY, SOURCE_EVIDENCE_POLICY, SOURCE_EVIDENCE_PURPOSE,
    EvidenceAuthenticator, EvidenceFreshnessPolicyV1, canonical_utc, parse_utc,
    validate_evidence_window,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, domain_digest, exact_fields, parse_canonical_object, safe_ref,
)
from tuner.project.execution_source import ExecutionSourceV1

from .binding import ModalClientBinding, Readiness, readiness_report
from .config import ModalProviderProfileV1, ModalRuntimeLockV1
from .coordinator_adapter import ModalPreparationAdapter
from .facade import ExplicitModal154ReadFacade
from .resolution import VerifiedModalDeploymentIdentityV1
from .resolution import ModalDeploymentSelectionV1


QUOTE_PURPOSE = "modal-quote-evidence/v1"
QUOTE_EVIDENCE_POLICY = EvidenceFreshnessPolicyV1(300, 300, 0)
_QUOTE_FIELDS = frozenset({
    "schema_version", "provider_id", "profile_ref", "account_ref", "namespace_ref",
    "resource_digest", "maximum_cost_minor_units", "currency", "issued_at", "expires_at",
    "issuer_ref", "evidence_ref", "audience_ref", "challenge_nonce", "key_ref",
})


class ModalQuoteBody:
    __slots__ = ("_raw",)

    def __init_subclass__(cls, **kwargs):
        raise TypeError("ModalQuoteBody is final")

    def __init__(self, *args, **kwargs):
        raise TypeError("Modal quote bodies are parser-minted")

    @classmethod
    def parse(cls, raw: bytes) -> "ModalQuoteBody":
        if type(raw) is not bytes or not raw or len(raw) > 16 * 1024:
            raise ValueError("Modal quote body exceeds its closed bound")
        document = parse_canonical_object(raw, name="Modal quote body")
        exact_fields(document, _QUOTE_FIELDS, "Modal quote body")
        if document["schema_version"] != "synaptic-modal-quote-body/v1":
            raise ValueError("unsupported Modal quote body")
        for name in (
            "provider_id", "profile_ref", "account_ref", "namespace_ref", "issuer_ref",
            "evidence_ref", "audience_ref", "challenge_nonce", "key_ref",
        ):
            safe_ref(document[name], name)
        from tuner.execution.foundation_v2.canonical import digest_text
        digest_text(document["resource_digest"], "resource_digest")
        amount = document["maximum_cost_minor_units"]
        if type(amount) is not int or amount < 0:
            raise ValueError("Modal quote maximum cost is invalid")
        if document["currency"] != "USD":
            raise ValueError("Modal quote currency must be USD")
        issued = parse_utc(canonical_utc(document["issued_at"], "issued_at"))
        expiry = parse_utc(canonical_utc(document["expires_at"], "expires_at"))
        if issued >= expiry:
            raise ValueError("Modal quote expiry must follow issuance")
        value = object.__new__(cls)
        object.__setattr__(value, "_raw", bytes(raw))
        return value

    @property
    def canonical_bytes(self) -> bytes:
        return bytes(self._raw)

    @property
    def quote_digest(self) -> str:
        return domain_digest("synaptic-modal-quote-body/v1", self._raw)

    def __getattr__(self, name: str):
        if name in _QUOTE_FIELDS:
            return parse_canonical_object(self._raw, name="Modal quote body")[name]
        raise AttributeError(name)


@dataclass(frozen=True, slots=True)
class AuthenticatedModalQuote:
    body_bytes: bytes
    tag: bytes

    def __post_init__(self) -> None:
        if (type(self.body_bytes) is not bytes or type(self.tag) is not bytes
                or not self.tag or len(self.tag) > 128):
            raise TypeError("exact authenticated Modal quote bytes are required")
        ModalQuoteBody.parse(self.body_bytes)
        object.__setattr__(self, "body_bytes", bytes(self.body_bytes))
        object.__setattr__(self, "tag", bytes(self.tag))

    @property
    def body(self) -> ModalQuoteBody:
        return ModalQuoteBody.parse(self.body_bytes)


@dataclass(frozen=True, slots=True)
class TrustedEvidenceIdentity:
    issuer_ref: str
    key_ref: str
    audience_ref: str

    def __post_init__(self) -> None:
        for name in ("issuer_ref", "key_ref", "audience_ref"):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))


class ModalOperationalPreflightAdapter:
    """Candidate operational planning adapter; not a public registration."""

    def __init__(
        self, preparation: ModalPreparationAdapter, *, execution_source_bytes: bytes,
        deployment_bytes: bytes, quote: AuthenticatedModalQuote,
        control_volume_id: str, artifact_volume_id: str,
        facade: ExplicitModal154ReadFacade, authenticator: EvidenceAuthenticator,
        clock: CoordinatorClockPortV1, source_trust: TrustedEvidenceIdentity,
        deployment_trust: TrustedEvidenceIdentity, quote_trust: TrustedEvidenceIdentity,
    ) -> None:
        if type(preparation) is not ModalPreparationAdapter or type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact Modal preparation and read facade are required")
        if not isinstance(authenticator, EvidenceAuthenticator) or not callable(getattr(clock, "now_iso", None)):
            raise TypeError("evidence authenticator and coordinator clock are required")
        for value in (source_trust, deployment_trust, quote_trust):
            if type(value) is not TrustedEvidenceIdentity:
                raise TypeError("exact trusted evidence identities are required")
        self._preparation = preparation
        if type(execution_source_bytes) is not bytes or type(deployment_bytes) is not bytes:
            raise TypeError("exact retained operational bytes are required")
        if type(quote) is not AuthenticatedModalQuote:
            raise TypeError("exact authenticated Modal quote is required")
        self._source_bytes = bytes(execution_source_bytes)
        self._deployment_bytes = bytes(deployment_bytes)
        self._quote = AuthenticatedModalQuote(quote.body_bytes, quote.tag)
        self._control_id = safe_ref(control_volume_id, "control_volume_id")
        self._artifact_id = safe_ref(artifact_volume_id, "artifact_volume_id")
        if self._control_id == self._artifact_id:
            raise ValueError("Modal volumes must differ")
        self._facade, self._auth, self._clock = facade, authenticator, clock
        self._source_trust, self._deployment_trust, self._quote_trust = source_trust, deployment_trust, quote_trust

    def describe(self, provider):
        return self._preparation.describe(provider)

    def context(self, resolved, provider):
        return self._preparation.context(resolved, provider)

    def _authenticate(self, purpose, payload, tag, evidence, trust, now, policy) -> None:
        if (evidence.issuer_ref, evidence.key_ref, evidence.audience_ref) != (
            trust.issuer_ref, trust.key_ref, trust.audience_ref,
        ):
            raise ValueError("evidence trust mismatch")
        validate_evidence_window(
            verified_at=evidence.verified_at, expires_at=evidence.expires_at,
            now=now, policy=policy,
        )
        if hashlib.sha256(payload).hexdigest() != evidence.attestation_digest:
            raise ValueError("evidence attestation mismatch")
        if self._auth.verify(purpose, payload, tag, evidence.key_ref) is not True:
            raise ValueError("evidence authentication failed")

    def preflight(self, plan: TrainingPlan) -> TrainingPreflight:
        _, binding, expected_plan = self._preparation._snapshot()
        if type(plan) is not TrainingPlan or plan != expected_plan:
            return self._preparation.preflight(plan)
        checked_at = self._clock.now_iso()
        fallback_expiry = canonical_utc(checked_at, "checked_at")
        try:
            source = ExecutionSourceV1.from_dict(parse_canonical_object(self._source_bytes, name="execution source"))
            deployment = VerifiedModalDeploymentIdentityV1.from_dict(parse_canonical_object(self._deployment_bytes, name="deployment"))
            quote = self._quote.body
            source_evidence = source.source_evidence
            self._authenticate(SOURCE_EVIDENCE_PURPOSE, source_evidence.authenticated_payload, source_evidence.tag, source_evidence, self._source_trust, checked_at, SOURCE_EVIDENCE_POLICY)
            self._authenticate("modal-deployment-evidence/v1", deployment.authenticated_payload, deployment.tag, deployment, self._deployment_trust, checked_at, DEPLOYMENT_EVIDENCE_POLICY)
            if (quote.issuer_ref, quote.key_ref, quote.audience_ref) != (self._quote_trust.issuer_ref, self._quote_trust.key_ref, self._quote_trust.audience_ref):
                raise ValueError("quote trust mismatch")
            now = parse_utc(canonical_utc(checked_at, "checked_at"))
            validate_evidence_window(
                verified_at=quote.issued_at, expires_at=quote.expires_at,
                now=checked_at, policy=QUOTE_EVIDENCE_POLICY,
            )
            if not parse_utc(quote.issued_at) <= now < parse_utc(quote.expires_at):
                raise ValueError("quote is outside its validity window")
            if self._auth.verify(QUOTE_PURPOSE, quote.canonical_bytes, self._quote.tag, quote.key_ref) is not True:
                raise ValueError("quote authentication failed")
            snapshot = parse_canonical_object(self._preparation.snapshot(), name="preparation snapshot")
            configuration = snapshot["configuration"]
            profile = ModalProviderProfileV1.from_mapping(configuration["profile"])
            selection = deployment.selection
            retained_selection = ModalDeploymentSelectionV1.from_dict(configuration["selection"])
            namespace = domain_digest("synaptic-modal-namespace/v1", canonical_bytes({"workspace_ref": selection.workspace_ref, "environment_ref": selection.environment_ref}))
            if (quote.provider_id, quote.profile_ref, quote.account_ref, quote.namespace_ref, quote.resource_digest, quote.quote_digest) != (binding.provider.provider_id, binding.provider.profile_ref, binding.scope.account_ref, binding.scope.namespace_ref, binding.resource_digest, binding.quote_digest):
                raise ValueError("quote does not bind execution")
            if (selection != retained_selection or source.fingerprint != plan.basis.source_digest
                    or namespace != binding.scope.namespace_ref
                    or configuration["quote_digest"] != quote.quote_digest):
                raise ValueError("retained preparation differs from quote")
            ModalRuntimeLockV1.packaged().validate_selection(selection)
            if (source.deployment_member_sha256 != hashlib.sha256(self._deployment_bytes).hexdigest()
                    or source.python_version != selection.python_version
                    or source.python_executable != selection.python_executable
                    or source.python_executable_digest != selection.python_executable_digest
                    or source.secret_requirements_digest != profile.secret_requirements_digest
                    or source.provider_runtime_requirements_digest
                    != selection.provider_runtime_requirements_digest):
                raise ValueError("execution source differs from deployment")
            expected_client = ModalClientBinding(selection.account_ref, selection.workspace_ref, selection.environment_ref, selection.client_ref, selection.sdk_version)
            if self._facade.binding != expected_client or readiness_report(expected_client, self._facade).status is not Readiness.READY or self._facade.inspect_deployment(app_name=selection.app_name, function_name=selection.function_name) != selection:
                raise ValueError("Modal deployment is not currently ready")
            for volume_id, name in ((self._control_id, profile.control_volume_ref), (self._artifact_id, profile.artifact_volume_ref)):
                volume = self._facade._volume(volume_id)
                if volume.object_id != volume_id or self._facade.volume_name(volume_id) != name:
                    raise ValueError("Modal volume identity changed")
            for requirement in profile.secrets:
                secret = self._facade.sdk.Secret.from_name(requirement.name, environment_name=expected_client.environment_ref, required_keys=list(requirement.required_keys), client=self._facade.client)
                secret.hydrate(self._facade.client)
                if not secret.is_hydrated or not str(secret.object_id).startswith("st-") or secret.name != requirement.name:
                    raise ValueError("Modal secret requirement unavailable")
            expiry = min(parse_utc(source_evidence.expires_at), parse_utc(deployment.expires_at), parse_utc(quote.expires_at)).strftime("%Y-%m-%dT%H:%M:%SZ")
            return TrainingPreflight(plan.plan_fingerprint, True, checked_at, expiry, (AuthorizationRequirement("training.start", True, quote.maximum_cost_minor_units, "USD"),))
        except Exception:
            from datetime import timedelta
            expiry = (parse_utc(fallback_expiry) + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            return TrainingPreflight(plan.plan_fingerprint, False, checked_at, expiry, diagnostic_codes=("modal_operational_preflight_failed",))


__all__: list[str] = []
