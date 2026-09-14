"""Issue an authenticated Modal chat configuration from reviewed runtime evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from decimal import Decimal, ROUND_CEILING

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    safe_ref,
)
from tuner.execution.providers.modal import inference_runtime as runtime
from tuner.execution.providers.modal.inference_preparation import (
    AuthenticatedModalInferencePreparationConfig,
    CONFIG_EVIDENCE_PURPOSE,
    ModalInferencePreparationConfig,
)
from tuner.execution.providers.modal.coordinator_preflight import (
    TrustedEvidenceIdentity,
)
from tuner.execution.providers.modal.coordinator_preflight import (
    AuthenticatedModalQuote,
    ModalQuoteBody,
    QUOTE_EVIDENCE_POLICY,
    QUOTE_PURPOSE,
)
from tuner.execution.evidence import canonical_utc, validate_evidence_window
from tuner.execution.providers.modal.mounted_io import hash_regular, read_regular

from scripts.capture_modal_inference_runtime import _parse_candidate

from .authority import HMACAuthenticator
from .deployment import ModalChatScope

_MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
RATE_CALCULATION_PURPOSE = "modal-chat-rate-calculation/v1"


class ModalChatConfigurationError(RuntimeError):
    """Closed failure at the reviewed configuration-issuance boundary."""


def _reviewed(path: Path, expected_sha256: str) -> tuple[bytes, dict[str, object]]:
    digest_text(expected_sha256, "reviewed evidence digest")
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("reviewed evidence path must be absolute")
    payload = read_regular(path.parent, path, _MAX_EVIDENCE_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("reviewed evidence digest differs")
    value = json.loads(payload.decode("utf-8"))
    if type(value) is not dict or canonical_bytes(value) + b"\n" != payload:
        raise ValueError("reviewed evidence is not canonical")
    return payload, value


def _qualification(
    capture_path: Path,
    capture_sha256: str,
    readback_path: Path,
    readback_sha256: str,
) -> dict[str, object]:
    _, capture = _reviewed(capture_path, capture_sha256)
    _, readback = _reviewed(readback_path, readback_sha256)
    capture_fields = {
        "schema_version",
        "candidate",
        "engine_wheel_name",
        "engine_wheel_sha256",
        "inspection_script_sha256",
        "modal_additions_sha256",
        "operator_selection_only",
        "provider_image_id",
        "python_preparation",
        "sandbox_id",
    }
    if type(capture) is not dict or set(capture) != capture_fields:
        raise ValueError("reviewed capture is malformed")
    if set(readback) != {
        "schema_version",
        "status",
        "returncode",
        "sandbox_id",
        "candidate",
    }:
        raise ValueError("reviewed readback is malformed")
    candidate = capture["candidate"]
    if (
        type(candidate) is not dict
        or type(candidate.get("operator_selection")) is not dict
    ):
        raise ValueError("reviewed candidate is malformed")
    selection = candidate["operator_selection"]
    parsed = _parse_candidate(
        canonical_bytes(candidate) + b"\n",
        image=selection.get("image"),
        source_commit=selection.get("source_commit"),
        check_private_directories=True,
        verify_runtime_lock_digest=candidate.get("packaged_runtime", {}).get(
            "runtime_lock_digest"
        ),
    )
    if (
        capture["schema_version"] != "synaptic-modal-inference-runtime-capture/v1"
        or capture["operator_selection_only"] is not True
        or readback["schema_version"] != "synaptic-modal-inference-runtime-read/v1"
        or readback["status"] != "CANDIDATE_ONLY"
        or readback["returncode"] != 0
        or readback["sandbox_id"] != capture["sandbox_id"]
        or readback["candidate"] != parsed
    ):
        raise ValueError("reviewed qualification and readback differ")
    return {**capture, "candidate": parsed}


def _packaged() -> tuple[dict[str, object], dict[str, object]]:
    payload, manifest = runtime._manifest()
    root = runtime._runtime_root()
    by_path = {item["path"]: item for item in manifest["source_inventory"]}
    for item in manifest["source_inventory"]:
        size, digest = hash_regular(root, root / item["path"], item["size_bytes"])
        if (size, digest) != (item["size_bytes"], item["sha256"]):
            raise ValueError("packaged runtime source differs")
    dependency = by_path[manifest["dependency_lock_path"]]
    closure_member = by_path[manifest["worker_closure_manifest_path"]]
    closure = read_regular(
        root, root / closure_member["path"], closure_member["size_bytes"]
    )
    closure_digest = runtime._worker_closure(
        closure,
        manifest["source_inventory"],
        closure_member["path"],
        dependency["path"],
    )
    return manifest, {
        "dependency_lock_digest": dependency["sha256"],
        "runtime_lock_digest": hashlib.sha256(payload).hexdigest(),
        "source_lock_digest": runtime._source_digest(manifest["source_inventory"]),
        "worker_closure_digest": closure_digest,
        "python_version": manifest["python"]["version"],
        "python_executable": manifest["python"]["executable"],
        "python_executable_digest": manifest["python"]["executable_sha256"],
    }


def build_authenticated_inference_configuration(
    *,
    scope: ModalChatScope,
    reviewed_capture_path: Path,
    reviewed_capture_sha256: str,
    reviewed_readback_path: Path,
    reviewed_readback_sha256: str,
    provider: dict[str, object],
    application: dict[str, object],
    volumes: dict[str, object],
    resources: dict[str, object],
    serving: dict[str, object],
    policy: dict[str, object],
    secrets: list[dict[str, object]],
    evidence: dict[str, object],
    authenticator: HMACAuthenticator,
) -> tuple[AuthenticatedModalInferencePreparationConfig, TrustedEvidenceIdentity]:
    """Build and sign one bounded config; this does not issue a provider quote."""
    try:
        if (
            type(scope) is not ModalChatScope
            or type(authenticator) is not HMACAuthenticator
        ):
            raise TypeError("exact scope and HMAC authority required")
        scope.observe(scope.client)
        capture = _qualification(
            reviewed_capture_path,
            reviewed_capture_sha256,
            reviewed_readback_path,
            reviewed_readback_sha256,
        )
        manifest, packaged = _packaged()
        candidate = capture["candidate"]
        if (
            candidate["distributions"] != manifest["distributions"]
            or candidate["packaged_runtime"]
            != {
                **{
                    key: packaged[key]
                    for key in (
                        "dependency_lock_digest",
                        "runtime_lock_digest",
                        "source_lock_digest",
                        "worker_closure_digest",
                    )
                },
                "status": "PACKAGED_RUNTIME_VERIFIED",
            }
            or candidate["python"]
            != {
                "implementation": manifest["python"]["implementation"],
                "version": packaged["python_version"],
                "executable": packaged["python_executable"],
                "executable_sha256": packaged["python_executable_digest"],
            }
            or candidate["requirements"]["modal"]
            != {"present": True, "version": manifest["sdk_version"]}
            or candidate["operator_selection"]["image"]
            != manifest["base_registry_reference"]
        ):
            raise ValueError("reviewed qualification differs from packaged runtime")
        binding = scope.binding
        document = {
            "schema_version": "synaptic-modal-inference-preparation-config/v1",
            "provider": provider,
            "client": {
                "account_ref": binding.account_ref,
                "workspace_ref": binding.workspace_ref,
                "environment_ref": binding.environment_ref,
                "client_ref": binding.client_ref,
                "sdk_version": binding.sdk_version,
            },
            "application": application,
            "image": {
                "registry_reference": manifest["base_registry_reference"],
                "image_digest": manifest["base_registry_reference"].rsplit(
                    "@sha256:", 1
                )[1],
                "provider_image_id": capture["provider_image_id"],
            },
            "runtime": packaged,
            "volumes": volumes,
            "resources": resources,
            "serving": serving,
            "policy": policy,
            "secrets": secrets,
            "evidence": evidence,
        }
        config = ModalInferencePreparationConfig.build(document)
        trust = TrustedEvidenceIdentity(
            evidence["issuer_ref"], evidence["key_ref"], evidence["audience_ref"]
        )
        tag = authenticator.sign(
            CONFIG_EVIDENCE_PURPOSE, config.canonical_bytes, trust.key_ref
        )
        return (
            AuthenticatedModalInferencePreparationConfig(config.canonical_bytes, tag),
            trust,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ModalChatConfigurationError("modal_chat_configuration_invalid") from None


def issue_authenticated_modal_quote(
    *,
    scope: ModalChatScope,
    configuration: AuthenticatedModalInferencePreparationConfig,
    namespace_ref: str,
    maximum_cost_minor_units: int,
    execution_kind: str,
    issued_at: str,
    expires_at: str,
    issuer_ref: str,
    audience_ref: str,
    challenge_nonce: str,
    key_ref: str,
    authenticator: HMACAuthenticator,
    clock: object,
) -> tuple[AuthenticatedModalQuote, TrustedEvidenceIdentity, bytes, bytes]:
    """Issue an inference-only quote using its exact preparation resource digest."""
    try:
        if (
            type(scope) is not ModalChatScope
            or type(authenticator) is not HMACAuthenticator
        ):
            raise TypeError("exact scope and HMAC authority required")
        if type(configuration) is not AuthenticatedModalInferencePreparationConfig:
            raise TypeError("exact authenticated configuration required")
        if type(maximum_cost_minor_units) is not int or maximum_cost_minor_units < 0:
            raise ValueError("maximum cost must be nonnegative minor units")
        if execution_kind not in {"function", "sandbox"}:
            raise ValueError("execution kind is unsupported")
        issued_at = canonical_utc(issued_at, "issued_at")
        expires_at = canonical_utc(expires_at, "expires_at")
        if (
            not callable(getattr(clock, "now_iso", None))
            or clock.now_iso() != issued_at
        ):
            raise ValueError("quote issuance time is not fresh")
        validate_evidence_window(
            verified_at=issued_at,
            expires_at=expires_at,
            now=clock.now_iso(),
            policy=QUOTE_EVIDENCE_POLICY,
        )
        body = ModalInferencePreparationConfig.parse(configuration.body_bytes).document
        config_key_ref = body["evidence"]["key_ref"]
        if (
            authenticator.verify(
                CONFIG_EVIDENCE_PURPOSE,
                configuration.body_bytes,
                configuration.tag,
                config_key_ref,
            )
            is not True
        ):
            raise ValueError("configuration authentication failed")
        binding = scope.binding
        if body["client"] != {
            "account_ref": binding.account_ref,
            "workspace_ref": binding.workspace_ref,
            "environment_ref": binding.environment_ref,
            "client_ref": binding.client_ref,
            "sdk_version": binding.sdk_version,
        }:
            raise ValueError("configuration scope differs")
        scope.observe(scope.client)
        workspace = scope.sdk.Workspace.from_context(client=scope.client)
        workspace.hydrate(scope.client)
        if (
            workspace.is_hydrated is not True
            or workspace.name != scope.binding.workspace_ref
        ):
            raise ValueError("rate workspace differs from selected scope")
        raw_rates = workspace.billing.rates()
        if not isinstance(raw_rates, Mapping):
            raise TypeError("Modal rates must be a mapping")
        suffix = "_sandbox" if execution_kind == "sandbox" else ""
        keys = (
            "gpu_hour_cost_a10g",
            "cpu_hour_cost" + suffix,
            "mem_gib_hour_cost" + suffix,
        )
        rates: dict[str, Decimal] = {}
        for name in keys:
            value = raw_rates.get(name)
            if type(value) is not Decimal:
                raise ValueError("Modal rate is malformed")
            if not value.is_finite() or value <= 0:
                raise ValueError("Modal rate is malformed")
            rates[name] = value
        resources = body["resources"]
        if resources["accelerator"].upper() not in {"A10", "A10G"}:
            raise ValueError("only the selected A10 rate is supported")
        hourly = (
            rates["gpu_hour_cost_a10g"] * resources["accelerator_count"]
            + rates["cpu_hour_cost" + suffix]
            * Decimal(resources["cpu_millicores"])
            / Decimal(1000)
            + rates["mem_gib_hour_cost" + suffix]
            * Decimal(resources["memory_mb"])
            / Decimal(1024)
        )
        estimate = (
            hourly * Decimal(resources["provider_timeout_seconds"]) / Decimal(3600)
        )
        estimate_minor = int((estimate * 100).to_integral_value(rounding=ROUND_CEILING))
        if maximum_cost_minor_units < estimate_minor:
            raise ValueError("operator authorization is below the nominal estimate")
        resource_digest = domain_digest(
            "synaptic-modal-inference-resource/v1",
            canonical_bytes(
                {
                    "resources": resources,
                    "volumes": body["volumes"],
                    "application": body["application"],
                }
            ),
        )
        scope.observe(scope.client)
        calculation = canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-rate-calculation/v1",
                "provider_id": body["provider"]["provider_id"],
                "profile_ref": body["provider"]["profile_ref"],
                "account_ref": body["client"]["account_ref"],
                "namespace_ref": safe_ref(namespace_ref, "namespace_ref"),
                "resource_digest": resource_digest,
                "execution_kind": execution_kind,
                "rates_usd_per_hour": {name: str(rates[name]) for name in keys},
                "nominal_hourly_usd": str(hourly),
                "nominal_timeout_estimate_usd": str(estimate),
                "nominal_timeout_estimate_minor_units": estimate_minor,
                "operator_authorization_minor_units": maximum_cost_minor_units,
                "currency": "USD",
                "issued_at": issued_at,
                "expires_at": expires_at,
                "issuer_ref": safe_ref(issuer_ref, "issuer_ref"),
                "audience_ref": safe_ref(audience_ref, "audience_ref"),
                "challenge_nonce": safe_ref(challenge_nonce, "challenge_nonce"),
                "key_ref": safe_ref(key_ref, "key_ref"),
                "authorization_semantics": "operator-maximum-not-provider-billing-cap",
                "excluded_charges": [
                    "build",
                    "storage",
                    "burst",
                    "other-provider-charges",
                ],
            }
        )
        calculation_tag = authenticator.sign(
            RATE_CALCULATION_PURPOSE, calculation, key_ref
        )
        evidence_ref = "rate-" + hashlib.sha256(calculation).hexdigest()
        quote_bytes = canonical_bytes(
            {
                "schema_version": "synaptic-modal-quote-body/v1",
                "provider_id": body["provider"]["provider_id"],
                "profile_ref": body["provider"]["profile_ref"],
                "account_ref": body["client"]["account_ref"],
                "namespace_ref": namespace_ref,
                "resource_digest": resource_digest,
                "maximum_cost_minor_units": maximum_cost_minor_units,
                "currency": "USD",
                "issued_at": issued_at,
                "expires_at": expires_at,
                "issuer_ref": issuer_ref,
                "evidence_ref": evidence_ref,
                "audience_ref": audience_ref,
                "challenge_nonce": challenge_nonce,
                "key_ref": key_ref,
            }
        )
        quote_body = ModalQuoteBody.parse(quote_bytes)
        quote_tag = authenticator.sign(
            QUOTE_PURPOSE, quote_body.canonical_bytes, key_ref
        )
        trust = TrustedEvidenceIdentity(issuer_ref, key_ref, audience_ref)
        return (
            AuthenticatedModalQuote(quote_body.canonical_bytes, quote_tag),
            trust,
            calculation,
            calculation_tag,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ModalChatConfigurationError("modal_chat_quote_invalid") from None


__all__ = [
    "ModalChatConfigurationError",
    "RATE_CALCULATION_PURPOSE",
    "build_authenticated_inference_configuration",
    "issue_authenticated_modal_quote",
]
