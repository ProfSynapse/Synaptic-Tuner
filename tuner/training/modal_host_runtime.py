"""Scoped Modal runtime quotation and one-shot host preparation.

The exact A100-80GB billing key is intentionally unset until a reviewed,
scoped read-only observation establishes Modal's provider-returned field.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import base64
import hashlib
import secrets
from pathlib import Path
import re
import time
from typing import Callable, Mapping, TypeVar

from synaptic_tuner.api.v1.ports import SecretResolverPort
from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.facade import MODAL_VOLUME_V1
from tuner.execution.providers.modal.packaged_binding import ModalPackagedRuntimeFactsV1
from tuner.execution.providers.modal.runtime_build import (
    ModalBoundedOperationFailure, ModalBuildStageFailure, SourceArchiveInvalid,
    _bounded, build_modal_runtime_release_v2, capture_modal_build_candidate,
    plan_modal_build_material,
)
from tuner.execution.providers.modal.runtime_release_deployment import (
    EXACT_PACKAGED_TRAINING_MODULE, EXACT_PACKAGED_TRAINING_QUALNAME,
    EXACT_QUALIFICATION_SECRET_REQUIRED_KEYS,
    EXACT_SELF_CHECK_MODULE, EXACT_SELF_CHECK_QUALNAME,
    ExplicitModal154ReleaseDeploymentReader, ModalRuntimeReleaseDeployer,
    require_bounded_modal_deployment_host,
    ModalRuntimeReleaseDeploymentPlanV1, ModalRuntimeReleaseFunctionSpecV1,
    ModalRuntimeReleaseDeploymentFactsV1,
    ModalRuntimeReleaseSecretSpecV1, ModalRuntimeReleaseVolumeSpecV1,
    MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA,
)
from tuner.runtime.runtime_release_modal_training import run_modal_packaged_training
from tuner.runtime.runtime_release_modal_self_check import run_runtime_release_self_check
from tuner.training.contracts import ResourceSpec


# Observed read-only in the selected Modal 1.5.4 workspace/environment on
# 2026-09-22; the value itself is always fetched afresh from scoped billing.
_A100_80GB_RATE_KEY = "gpu_hour_cost_a100_80gb"
_RATE_KEY = re.compile(r"^gpu_hour_cost_[a-z0-9_]+$")
_T = TypeVar("_T")
_CLOSED_BOOTSTRAP_DIAGNOSTICS = {
    "SOURCE_ARCHIVE_INVALID": ("SOURCE_WHEEL", "SOURCE_ARCHIVE_INVALID", "runtime_build.prepare_current_source_wheel"),
    "APP_START_TIMEOUT": ("APP_START", "TIMEOUT", "modal_host_runtime.build_app"),
    "APP_START_OPERATION_FAILED": ("APP_START", "OPERATION_FAILED", "modal_host_runtime.build_app"),
    "APP_CLEANUP_TIMEOUT": ("APP_CLEANUP", "TIMEOUT", "modal_host_runtime.close_build_app"),
    "APP_CLEANUP_OPERATION_FAILED": ("APP_CLEANUP", "OPERATION_FAILED", "modal_host_runtime.close_build_app"),
    "SOURCE_WHEEL_LOCAL_BUILD_FAILED": ("SOURCE_WHEEL", "LOCAL_BUILD_FAILED", "runtime_build.prepare_current_source_wheel"),
    "SOURCE_WHEEL_SOURCE_STATE_INVALID": ("SOURCE_WHEEL", "SOURCE_STATE_INVALID", "runtime_build.prepare_current_source_wheel"),
    "SOURCE_WHEEL_BUILDER_SETUP_FAILED": ("SOURCE_WHEEL", "BUILDER_SETUP_FAILED", "runtime_build.prepare_current_source_wheel"),
    "SOURCE_WHEEL_OFFLINE_WHEEL_TIMEOUT": ("SOURCE_WHEEL", "OFFLINE_WHEEL_TIMEOUT", "runtime_build.prepare_current_source_wheel"),
    "SOURCE_WHEEL_OFFLINE_WHEEL_FAILED": ("SOURCE_WHEEL", "OFFLINE_WHEEL_FAILED", "runtime_build.prepare_current_source_wheel"),
    "SOURCE_WHEEL_WHEEL_INVENTORY_INVALID": ("SOURCE_WHEEL", "WHEEL_INVENTORY_INVALID", "runtime_build.prepare_current_source_wheel"),
    "BUILD_INPUTS_INVALID": ("BUILD_INPUTS", "INVALID", "runtime_build.prepare_build_inputs"),
    "IMAGE_BUILD_TIMEOUT": ("IMAGE_BUILD", "TIMEOUT", "runtime_build.build_image"),
    "IMAGE_BUILD_OPERATION_FAILED": ("IMAGE_BUILD", "OPERATION_FAILED", "runtime_build.build_image"),
    "IMAGE_BUILD_IDENTITY_MISSING": ("IMAGE_BUILD", "IDENTITY_MISSING", "runtime_build.build_image"),
    "CAPTURE_CREATE_TIMEOUT": ("CAPTURE_CREATE", "TIMEOUT", "runtime_build.create_capture_sandbox"),
    "CAPTURE_CREATE_OPERATION_FAILED": ("CAPTURE_CREATE", "OPERATION_FAILED", "runtime_build.create_capture_sandbox"),
    "CAPTURE_OUTPUT_TIMEOUT": ("CAPTURE_OUTPUT", "TIMEOUT", "runtime_build.capture_output"),
    "CAPTURE_OUTPUT_OPERATION_FAILED": ("CAPTURE_OUTPUT", "OPERATION_FAILED", "runtime_build.capture_output"),
    "CAPTURE_OUTPUT_INSPECTOR_REJECTED": ("CAPTURE_OUTPUT", "INSPECTOR_REJECTED", "runtime_build.capture_output"),
    "CAPTURE_OUTPUT_OUTPUT_INVALID": ("CAPTURE_OUTPUT", "OUTPUT_INVALID", "runtime_build.capture_output"),
    "CAPTURE_CLEANUP_TIMEOUT": ("CAPTURE_CLEANUP", "TIMEOUT", "runtime_build.cleanup_capture_sandbox"),
    "CAPTURE_CLEANUP_OPERATION_FAILED": ("CAPTURE_CLEANUP", "OPERATION_FAILED", "runtime_build.cleanup_capture_sandbox"),
    "CAPTURE_VALIDATE_INVALID": ("CAPTURE_VALIDATE", "INVALID", "runtime_build.validate_capture"),
    "RELEASE_VALIDATE_INVALID": ("RELEASE_VALIDATE", "INVALID", "modal_host_runtime.build_release"),
}


class ModalHostBootstrapUnavailable(RuntimeError):
    """Closed, non-retryable diagnosis for a known host bootstrap boundary."""

    __slots__ = ("_phase", "_failure_class", "_location")

    def __init__(self, diagnosis: str = "SOURCE_ARCHIVE_INVALID") -> None:
        if type(diagnosis) is not str or diagnosis not in _CLOSED_BOOTSTRAP_DIAGNOSTICS:
            raise ValueError("Modal bootstrap diagnosis is invalid")
        super().__init__("modal_host_bootstrap_unavailable")
        phase, failure_class, location = _CLOSED_BOOTSTRAP_DIAGNOSTICS[diagnosis]
        object.__setattr__(self, "_phase", phase)
        object.__setattr__(self, "_failure_class", failure_class)
        object.__setattr__(self, "_location", location)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_phase", "_failure_class", "_location", "phase",
                    "failure_class", "location", "retry_authorized"}:
            raise AttributeError("Modal bootstrap diagnosis is immutable")
        super().__setattr__(name, value)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def failure_class(self) -> str:
        return self._failure_class

    @property
    def location(self) -> str:
        return self._location

    @property
    def retry_authorized(self) -> bool:
        return False


def _closed_provider_call(operation: Callable[[], _T], code: str) -> _T:
    try:
        return _bounded(operation, deadline=time.monotonic() + 60, code=code)
    except Exception:
        raise RuntimeError(code) from None


@dataclass(frozen=True, slots=True)
class ModalRuntimeQuoteV1:
    rate_key: str
    gpu_hourly_usd: str
    gpu_only_timeout_estimate_minor_units: int
    maximum_cost_minor_units: int
    resource: ResourceSpec
    quote_digest: str
    excluded_billing_dimensions: tuple[str, ...] = (
        "build", "cpu", "memory", "storage", "usage-beyond-timeout",
    )
    authorization_semantics: str = "operator-maximum-not-provider-billing-cap"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "synaptic-modal-runtime-quote/v1",
            "rate_key": self.rate_key,
            "gpu_hourly_usd": self.gpu_hourly_usd,
            "gpu_only_timeout_estimate_minor_units": self.gpu_only_timeout_estimate_minor_units,
            "maximum_cost_minor_units": self.maximum_cost_minor_units,
            "resource": {
                "accelerator": self.resource.accelerator,
                "accelerator_count": self.resource.accelerator_count,
                "timeout_seconds": self.resource.timeout_seconds,
            },
            "excluded_billing_dimensions": list(self.excluded_billing_dimensions),
            "authorization_semantics": self.authorization_semantics,
            "quote_digest": self.quote_digest,
        }


def _scoped_rates(*, sdk: object, client: object, client_binding: ModalClientBinding) -> Mapping[str, Decimal]:
    if (getattr(sdk, "__version__", None) != "1.5.4" or client is None
            or type(client_binding) is not ModalClientBinding
            or client_binding.sdk_version != "1.5.4"):
        raise ValueError("exact scoped Modal client required")
    workspace = _closed_provider_call(
        lambda: sdk.Workspace.from_context(client=client), "modal_rate_scope_unavailable",
    )
    _closed_provider_call(lambda: workspace.hydrate(client), "modal_rate_scope_unavailable")
    environment = _closed_provider_call(
        lambda: sdk.Environment.from_name(
            client_binding.environment_ref, create_if_missing=False, client=client,
        ), "modal_rate_scope_unavailable",
    )
    _closed_provider_call(lambda: environment.hydrate(client), "modal_rate_scope_unavailable")
    if (getattr(workspace, "is_hydrated", False) is not True
            or getattr(workspace, "name", None) != client_binding.workspace_ref
            or client_binding.account_ref != client_binding.workspace_ref
            or getattr(environment, "is_hydrated", False) is not True
            or getattr(environment, "name", None) != client_binding.environment_ref):
        raise ValueError("Modal rate scope differs")
    rates = _closed_provider_call(lambda: workspace.billing.rates(), "modal_rate_read_failed")
    if not isinstance(rates, Mapping):
        raise ValueError("Modal rates are unavailable")
    return rates


def observe_scoped_modal_gpu_rates(*, sdk: object, client: object,
                                   client_binding: ModalClientBinding) -> dict[str, str]:
    """Read only validated GPU rate keys for human review; no cost authority."""
    raw = _scoped_rates(sdk=sdk, client=client, client_binding=client_binding)
    selected: dict[str, str] = {}
    for key in raw:
        if type(key) is str and _RATE_KEY.fullmatch(key):
            value = raw[key]
            if type(value) is not Decimal or not value.is_finite() or value <= 0:
                raise ValueError("Modal GPU rate is invalid")
            selected[key] = str(value)
    if not selected or len(selected) > 64:
        raise ValueError("Modal GPU rate inventory is invalid")
    return dict(sorted(selected.items()))


def quote_modal_runtime_for_host(
    *, sdk: object, client: object, client_binding: ModalClientBinding,
    recipe_resource: ResourceSpec, maximum_cost_minor_units: int,
) -> ModalRuntimeQuoteV1:
    """Bounded GPU-only nominal quote, never a provider billing cap."""
    if (type(recipe_resource) is not ResourceSpec
            or recipe_resource.accelerator != "A100-80GB"
            or recipe_resource.accelerator_count != 1
            or not 1 <= recipe_resource.timeout_seconds <= 86400
            or type(maximum_cost_minor_units) is not int
            or maximum_cost_minor_units < 1):
        raise ValueError("Modal runtime resource or operator ceiling is unsupported")
    key = _A100_80GB_RATE_KEY
    if type(key) is not str or _RATE_KEY.fullmatch(key) is None:
        raise ValueError("A100-80GB scoped rate key is not independently verified")
    rates = _scoped_rates(sdk=sdk, client=client, client_binding=client_binding)
    value = rates.get(key)
    if (type(value) is not Decimal or not value.is_finite() or value <= 0
            or not -6 <= value.adjusted() <= 5
            or len(value.as_tuple().digits) > 12):
        raise ValueError("A100-80GB scoped rate is invalid")
    estimate = value * Decimal(recipe_resource.timeout_seconds) / Decimal(3600)
    minor = int((estimate * 100).to_integral_value(rounding=ROUND_CEILING))
    if minor < 1 or minor > maximum_cost_minor_units:
        raise ValueError("operator ceiling is below nominal GPU estimate")
    statement = {
        "schema_version": "synaptic-modal-runtime-quote/v1",
        "rate_key": key,
        "gpu_hourly_usd": str(value),
        "gpu_only_timeout_estimate_minor_units": minor,
        "maximum_cost_minor_units": maximum_cost_minor_units,
        "resource": {
            "accelerator": recipe_resource.accelerator,
            "accelerator_count": recipe_resource.accelerator_count,
            "timeout_seconds": recipe_resource.timeout_seconds,
        },
        "excluded_billing_dimensions": ["build", "cpu", "memory", "storage", "usage-beyond-timeout"],
        "authorization_semantics": "operator-maximum-not-provider-billing-cap",
    }
    return ModalRuntimeQuoteV1(
        key, str(value), minor, maximum_cost_minor_units, recipe_resource,
        hashlib.sha256(canonical_bytes(statement)).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ModalHostRuntimeV1:
    release: object
    deployment_facts: ModalRuntimeReleaseDeploymentFactsV1
    facts: ModalPackagedRuntimeFactsV1
    quote: ModalRuntimeQuoteV1
    volume_names_by_id: tuple[tuple[str, str], ...]
    qualification_key: bytes
    deployment_reader: object

    @property
    def quote_digest(self) -> str:
        return self.quote.quote_digest


class _CurrentPackagedDeploymentReader:
    """Rebind current scoped layout, three Volumes and both Secret IDs."""

    def __init__(self, *, sdk: object, client: object, binding: ModalClientBinding,
                 facts: ModalPackagedRuntimeFactsV1, names: Mapping[str, str],
                 secret_ids: tuple[tuple[str, str, tuple[str, ...]], ...]) -> None:
        self._sdk, self._client, self._binding = sdk, client, binding
        self._facts, self._names = facts, dict(names)
        if (type(secret_ids) is not tuple or len(secret_ids) != 2
                or len({item[0] for item in secret_ids}) != 2):
            raise ValueError("packaged deployment Secret inventory is invalid")
        self._secret_ids = secret_ids
        self._reader = ExplicitModal154ReleaseDeploymentReader(sdk=sdk)

    def observe(self, *, client: object, app_name: str, function_name: str,
                environment_name: str) -> ModalPackagedRuntimeFactsV1:
        facts = self._facts
        if (client is not self._client or app_name != facts.app_name
                or function_name != facts.function_name
                or environment_name != self._binding.environment_ref):
            raise ValueError("packaged deployment observation scope differs")
        current = self._reader.observe(
            client=client, app_name=app_name, environment_name=environment_name,
        )
        expected_layout = tuple(sorted(((facts.function_name, facts.function_id),
                                        (facts.self_check_function_name, facts.self_check_function_id))))
        if (current is None or current.deployed is not True
                or current.app_id != facts.app_id
                or current.generation != facts.deployment_generation
                or current.function_ids != expected_layout or current.class_ids):
            raise ValueError("packaged deployment current layout differs")
        for role, identity in (("control", facts.control_volume_id),
                               ("artifacts", facts.artifact_volume_id),
                               ("model_cache", facts.model_cache_volume_id)):
            volume = _closed_provider_call(
                lambda: self._sdk.Volume.from_name(
                    self._names[role], environment_name=environment_name,
                    create_if_missing=False, version=MODAL_VOLUME_V1, client=client,
                ), "modal_packaged_volume_read_failed",
            )
            _closed_provider_call(
                lambda: volume.hydrate(client), "modal_packaged_volume_read_failed",
            )
            if getattr(volume, "is_hydrated", False) is not True or volume.object_id != identity:
                raise ValueError("packaged deployment Volume identity differs")
        for name, identity, required_keys in self._secret_ids:
            secret = _closed_provider_call(
                lambda: self._sdk.Secret.from_name(
                    name, environment_name=environment_name,
                    required_keys=list(required_keys), client=client,
                ), "modal_packaged_secret_read_failed",
            )
            _closed_provider_call(
                lambda: secret.hydrate(client), "modal_packaged_secret_read_failed",
            )
            if getattr(secret, "is_hydrated", False) is not True or secret.object_id != identity:
                raise ValueError("packaged deployment Secret identity differs")
        return facts


def prepare_modal_runtime_for_host(
    *, sdk: object, client: object, client_binding: ModalClientBinding,
    profile_path: Path, runtime_material_intent_digest: str,
    recipe_resource: ResourceSpec, maximum_cost_minor_units: int,
    private_storage: object, secret_resolver: SecretResolverPort,
    hf_token_ref: SecretRef, qualification_secret_name: str,
    hf_token_secret_name: str, app_name: str, environment_name: str,
    builder_cache_root: Path | None = None,
) -> ModalHostRuntimeV1:
    """Prepare one named runtime; all mutations follow a durable attempt claim."""
    require_bounded_modal_deployment_host()
    if (type(hf_token_ref) is not SecretRef or not callable(getattr(secret_resolver, "resolve", None))
            or not callable(getattr(getattr(private_storage, "attempts", None), "claim", None))
            or type(runtime_material_intent_digest) is not str
            or plan_modal_build_material(profile_path)["intent_digest"] != runtime_material_intent_digest
            or environment_name != client_binding.environment_ref):
        raise ValueError("Modal host runtime authority differs")
    quote = quote_modal_runtime_for_host(
        sdk=sdk, client=client, client_binding=client_binding,
        recipe_resource=recipe_resource,
        maximum_cost_minor_units=maximum_cost_minor_units,
    )
    for name in (qualification_secret_name, hf_token_secret_name, app_name):
        if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9-]{0,27}", name) is None:
            raise ValueError("Modal host resource name is invalid")
    if qualification_secret_name == hf_token_secret_name:
        raise ValueError("Modal runtime Secret names must differ")
    nonce = secrets.token_hex(12)
    names = {
        "control": f"{app_name}-control-{nonce}",
        "artifacts": f"{app_name}-artifacts-{nonce}",
        "model_cache": f"{app_name}-cache-{nonce}",
    }
    qualification_name = f"{qualification_secret_name}-{nonce}"
    token_name = f"{hf_token_secret_name}-{nonce}"
    deployment_name = f"{app_name}-{nonce}"
    if len({*names.values(), qualification_name, token_name, deployment_name}) != 6:
        raise ValueError("Modal runtime resource names collide")
    claim = canonical_bytes({
        "schema_version": "synaptic-modal-host-bootstrap-claim/v1",
        "intent_digest": runtime_material_intent_digest,
        "quote_digest": quote.quote_digest,
        "deployment_name": deployment_name,
        "volumes": names,
        "secrets": [qualification_name, token_name],
    })
    private_storage.attempts.claim("build-" + runtime_material_intent_digest, claim)

    # No credential value enters the recipe, durable claim, report, or logs.
    qualification_key = secrets.token_bytes(32)
    token = _closed_provider_call(
        lambda: secret_resolver.resolve(hf_token_ref), "modal_host_credential_unavailable",
    )
    if type(token) is not str or not token.strip():
        raise ValueError("named HF credential is unavailable")
    ids: dict[str, str] = {}
    for role, name in names.items():
        _closed_provider_call(
            lambda: sdk.Volume.objects.create(
                name, version=MODAL_VOLUME_V1, allow_existing=False,
                environment_name=environment_name, client=client,
            ), "modal_host_volume_create_indeterminate",
        )
        volume = _closed_provider_call(
            lambda: sdk.Volume.from_name(
                name, environment_name=environment_name,
                create_if_missing=False, version=MODAL_VOLUME_V1, client=client,
            ), "modal_host_volume_read_failed",
        )
        _closed_provider_call(lambda: volume.hydrate(client), "modal_host_volume_read_failed")
        identity = getattr(volume, "object_id", None)
        if (getattr(volume, "is_hydrated", False) is not True or type(identity) is not str
                or re.fullmatch(r"vo-[A-Za-z0-9]{1,64}", identity) is None
                or identity in ids.values()):
            raise ValueError("created Modal Volume identity is invalid")
        ids[role] = identity
    _closed_provider_call(
        lambda: sdk.Secret.objects.create(
            qualification_name,
            {"SYNAPTIC_MODAL_QUALIFICATION_HMAC_KEY": base64.b64encode(qualification_key).decode("ascii")},
            allow_existing=False, environment_name=environment_name, client=client,
        ), "modal_host_qualification_secret_create_indeterminate",
    )
    _closed_provider_call(
        lambda: sdk.Secret.objects.create(
            token_name, {"HF_TOKEN": token}, allow_existing=False,
            environment_name=environment_name, client=client,
        ), "modal_host_hf_secret_create_indeterminate",
    )
    token = ""

    deadline = time.monotonic() + 3600
    try:
        build_app = _bounded(
            lambda: sdk.App(deployment_name, include_source=False),
            deadline=time.monotonic() + 60, code="modal_host_build_app_construction_failed",
        )
        context = _bounded(
            lambda: build_app.run(name=deployment_name + "-build", client=client,
                                  environment_name=environment_name),
            deadline=time.monotonic() + 60, code="modal_host_build_app_start_indeterminate",
        )
        entered = _bounded(context.__enter__, deadline=deadline,
                           code="modal_build_app_start_ambiguous",
                           late_cleanup=lambda _value: context.__exit__(None, None, None))
    except ModalBoundedOperationFailure as error:
        raise ModalHostBootstrapUnavailable("APP_START_" + error.reason) from None
    capture_failure: BaseException | None = None
    try:
        try:
            candidate = capture_modal_build_candidate(
                sdk=sdk, client=client, profile_path=profile_path,
                build_claim=lambda: None, app_name=deployment_name,
                environment_name=environment_name,
                expected_intent_digest=runtime_material_intent_digest,
                build_app=entered,
                builder_cache_root=builder_cache_root,
            )
        except SourceArchiveInvalid:
            raise ModalHostBootstrapUnavailable() from None
        except ModalBuildStageFailure as error:
            raise ModalHostBootstrapUnavailable(error.stage + "_" + error.reason) from None
    except BaseException as error:
        capture_failure = error
        raise
    finally:
        try:
            _bounded(lambda: context.__exit__(None, None, None),
                     deadline=time.monotonic() + 60,
                     code="modal_build_app_cleanup_unresolved")
        except ModalBoundedOperationFailure as error:
            if capture_failure is None:
                raise ModalHostBootstrapUnavailable("APP_CLEANUP_" + error.reason) from None
    try:
        release = build_modal_runtime_release_v2(
            candidate, release_ref="runtime:modal-" + nonce,
        )
    except Exception:
        raise ModalHostBootstrapUnavailable("RELEASE_VALIDATE_INVALID") from None
    private_storage.attempts.claim(
        "deploy-" + release.manifest_digest,
        canonical_bytes({
            "schema_version": "synaptic-modal-host-deploy-claim/v1",
            "release_digest": release.manifest_digest,
            "capture_digest": candidate.capture_digest,
            "quote_digest": quote.quote_digest,
            "deployment_name": deployment_name,
        }),
    )
    plan = ModalRuntimeReleaseDeploymentPlanV1(
        release=release, app_name=deployment_name,
        environment_name=environment_name,
        functions=(
            ModalRuntimeReleaseFunctionSpecV1(
                "training", "packaged-training", EXACT_PACKAGED_TRAINING_MODULE,
                EXACT_PACKAGED_TRAINING_QUALNAME,
                ("control", "artifacts", "model_cache"),
                (qualification_name, token_name),
                1000, 4096, recipe_resource.timeout_seconds,
                recipe_resource.accelerator, False,
            ),
            ModalRuntimeReleaseFunctionSpecV1(
                "self_check", "packaged-self-check", EXACT_SELF_CHECK_MODULE,
                EXACT_SELF_CHECK_QUALNAME,
                ("control", "artifacts"), (qualification_name,),
                1000, 512, 120, None, True,
                restrict_modal_access=False,
            ),
        ),
        volumes=(
            ModalRuntimeReleaseVolumeSpecV1("control", names["control"], "/mnt/control"),
            ModalRuntimeReleaseVolumeSpecV1("artifacts", names["artifacts"], "/mnt/artifacts"),
            ModalRuntimeReleaseVolumeSpecV1("model_cache", names["model_cache"], "/mnt/model-cache"),
        ),
        secrets=(
            ModalRuntimeReleaseSecretSpecV1(
                qualification_name, EXACT_QUALIFICATION_SECRET_REQUIRED_KEYS,
            ),
            ModalRuntimeReleaseSecretSpecV1(token_name, ("HF_TOKEN",)),
        ),
        schema_version=MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA,
    )
    reader = ExplicitModal154ReleaseDeploymentReader(sdk=sdk)
    deployer = ModalRuntimeReleaseDeployer(
        sdk=sdk, client=client, client_binding=client_binding, reader=reader,
    )
    # The fresh attempt name must be absent before the one authorized deploy.
    if deployer.observe(plan) is not None:
        raise ValueError("fresh Modal deployment name already exists")
    acknowledged = deployer.deploy_once(
        plan, candidate=candidate,
        entrypoints={
            "training": run_modal_packaged_training,
            "self_check": run_runtime_release_self_check,
        },
    )
    facts = ModalPackagedRuntimeFactsV1.from_release_deployment(acknowledged)
    current = _CurrentPackagedDeploymentReader(
        sdk=sdk, client=client, binding=client_binding, facts=facts, names=names,
        secret_ids=tuple(sorted(
            (item.spec.name, item.secret_id, item.spec.required_keys)
            for item in acknowledged.secrets
        )),
    )
    return ModalHostRuntimeV1(
        release, acknowledged, facts, quote,
        tuple(sorted((identity, names[role]) for role, identity in ids.items())),
        qualification_key, current,
    )
