"""One-shot CPU qualification of an acknowledged packaged Modal deployment.

This is a host adapter over the existing release qualification operator and
reader. It does not prepare training input, invoke the GPU function, or infer a
Function-to-Image link from floating current state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

from tuner.cloud.modal_runtime_qualification_operator import ModalRuntimeQualificationOperator
from tuner.cloud.modal_runtime_qualification_operator import ModalRuntimeQualificationOutcome
from tuner.execution.foundation_v2.canonical import canonical_bytes, safe_ref
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.runtime_build import _bounded
from tuner.execution.providers.modal.runtime_release_deployment import (
    ExplicitModal154ReleaseDeploymentReader, ModalRuntimeReleaseDeploymentFactsV1,
)
from tuner.execution.providers.modal.runtime_release_qualification import (
    QUALIFICATION_HMAC_KEY_REF, QUALIFICATION_RESULT_SCHEMA,
    ModalRuntimeQualificationHmacAuthenticator,
    ModalRuntimeReleaseFixtureReceiptV1,
    ModalRuntimeReleaseQualificationDispatchV1,
    ModalRuntimeReleaseQualificationPolicyV1,
    ModalRuntimeReleaseQualificationReceiptV1,
    build_modal_runtime_release_qualification_dispatch,
)
from tuner.execution.providers.modal.runtime_release_qualification_reader import (
    ModalRuntimeReleaseQualificationObservation, ModalRuntimeReleaseQualificationReader,
)
from tuner.runtime.releases import ProviderRuntimeBindingV1
from tuner.training.modal_host_runtime import ModalHostRuntimeV1
from tuner.training.modal_host_scope import observe_modal_host_scope


class ModalHostQualificationUnavailable(RuntimeError):
    """Fixed closed failure after a consumed qualification claim."""

    _STAGES = {
        "FIXTURE_STAGE": ("UNAVAILABLE", "modal_host_qualification.stage_fixture"),
        "DISPATCH_SUBMIT": ("INDETERMINATE", "modal_host_qualification.submit_once"),
        "DISPATCH_FUNCTION_IDENTITY": (
            "UNAVAILABLE", "modal_runtime_qualification_operator.function_identity"),
        "DISPATCH_SPAWN_INDETERMINATE": (
            "INDETERMINATE", "modal_runtime_qualification_operator.spawn"),
        "DISPATCH_CATALOG_INDETERMINATE": (
            "INDETERMINATE", "modal_runtime_qualification_operator.call_catalog"),
        "CALL_OBSERVE": ("UNAVAILABLE", "modal_host_qualification.observe_call"),
        "RECEIPT_VERIFY": ("UNAVAILABLE", "modal_host_qualification.verify_receipt"),
    }

    def __init__(self, phase: str) -> None:
        if phase not in self._STAGES:
            raise ValueError("qualification diagnostic is invalid")
        super().__init__("modal_host_cpu_qualification_unavailable")
        self._phase = phase

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def failure_class(self) -> str:
        return self._STAGES[self._phase][0]

    @property
    def location(self) -> str:
        return self._STAGES[self._phase][1]

    @property
    def retry_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class ModalHostCPUQualificationV1:
    receipt: ModalRuntimeReleaseQualificationReceiptV1
    output_sha256: str
    provider_call_id: str
    runtime_release_digest: str
    deployment_facts_digest: str
    training_executed: bool = False
    gpu_qualified: bool = False


class _CurrentQualificationObserver:
    def __init__(self, *, sdk: object, client: object, binding: object,
                 runtime: ModalHostRuntimeV1):
        self._client, self._binding, self._runtime = client, binding, runtime
        self._reader = ExplicitModal154ReleaseDeploymentReader(sdk=sdk)

    def observe(self, facts: ModalRuntimeReleaseDeploymentFactsV1):
        if facts != self._runtime.deployment_facts:
            raise ValueError("qualification deployment facts differ")
        packaged = self._runtime.facts
        self._runtime.deployment_reader.observe(
            client=self._client, app_name=packaged.app_name,
            function_name=packaged.function_name,
            environment_name=self._binding.environment_ref,
        )
        return self._reader.observe(
            client=self._client, app_name=facts.app_name,
            environment_name=self._binding.environment_ref,
        )


def _call_id_bytes(value: object) -> bytes:
    return safe_ref(value, "qualification_call_id").encode("ascii")


def _call_id_from_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise ValueError("qualification call catalog is invalid")
    return safe_ref(value.decode("ascii", "strict"), "qualification_call_id")


def qualify_modal_runtime_for_host(
    *, sdk: object, client: object, client_binding: object,
    runtime: ModalHostRuntimeV1, private_storage: object,
    effect_id: str, environment_name: str,
) -> ModalHostCPUQualificationV1:
    """Stage one fixed fixture, invoke CPU self-check once, verify its receipt."""
    if (type(runtime) is not ModalHostRuntimeV1
            or type(runtime.deployment_facts) is not ModalRuntimeReleaseDeploymentFactsV1
            or client is None or client_binding != runtime.facts.client_binding
            or environment_name != runtime.facts.environment_ref
            or getattr(sdk, "__version__", None) != "1.5.4"
            or not callable(getattr(getattr(private_storage, "attempts", None), "claim", None))
            or not callable(getattr(private_storage, "catalog", None))):
        raise ValueError("exact CPU qualification host authority required")
    effect_id = safe_ref(effect_id, "qualification_effect_id")
    deployment = runtime.deployment_facts
    if (deployment.runtime_release_digest != runtime.release.manifest_digest
            or deployment.client_binding != client_binding):
        raise ValueError("qualification deployment release differs")
    artifact_ids = tuple(item.volume_id for item in deployment.volumes
                         if item.spec.role == "artifacts")
    if len(artifact_ids) != 1:
        raise ValueError("qualification artifact Volume differs")
    auth = ModalRuntimeQualificationHmacAuthenticator(runtime.qualification_key)
    fixture = ModalRuntimeReleaseFixtureReceiptV1.create(
        effect_id=effect_id, artifact_volume_id=artifact_ids[0],
    )
    binding = ProviderRuntimeBindingV1.build(
        provider_ref="modal", runtime_release=runtime.release,
        provider_facts_schema=deployment.schema_version,
        provider_facts_digest=deployment.facts_digest,
    )
    dispatch = ModalRuntimeReleaseQualificationDispatchV1(
        effect_id, runtime.release, binding, deployment, fixture,
        ModalRuntimeReleaseQualificationPolicyV1(), QUALIFICATION_HMAC_KEY_REF,
    )
    payload = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    observer = _CurrentQualificationObserver(
        sdk=sdk, client=client, binding=client_binding, runtime=runtime,
    )
    facade = ExplicitModal154ReadFacade(
        client_binding, sdk=sdk, client=client,
        scope_observer=lambda supplied: observe_modal_host_scope(
            sdk=sdk, client=supplied, binding=client_binding,
        ),
        deployment_observer=runtime.deployment_reader.observe,
        volume_names=dict(runtime.volume_names_by_id),
    )
    calls = private_storage.catalog(
        "modal-runtime-qualification-calls",
        encode=_call_id_bytes, decode=_call_id_from_bytes,
    )
    operator = ModalRuntimeQualificationOperator(
        facade=facade, deployment_observer=observer,
        verifier=auth, call_catalog=calls,
    )
    reader = ModalRuntimeReleaseQualificationReader(
        facade=facade, deployment_observer=observer, verifier=auth,
    )
    private_storage.attempts.claim(
        "qualify-" + deployment.facts_digest,
        canonical_bytes({
            "schema_version": "synaptic-modal-host-cpu-qualification-claim/v1",
            "effect_id": effect_id,
            "runtime_release_digest": runtime.release.manifest_digest,
            "deployment_facts_digest": deployment.facts_digest,
            "dispatch_digest": dispatch.dispatch_digest,
        }),
    )
    phase = "FIXTURE_STAGE"
    try:
        staged = _bounded(
            lambda: operator.stage_fixture_once(
                effect_id=effect_id, deployment_facts=deployment,
            ), deadline=time.monotonic() + 60,
            code="modal_host_cpu_fixture_indeterminate",
        )
        if staged != fixture:
            raise ValueError
        phase = "DISPATCH_SUBMIT"
        outcome = _bounded(
            lambda: operator.submit_once(payload, expected_facts=deployment),
            deadline=time.monotonic() + 60,
            code="modal_host_cpu_dispatch_indeterminate",
        )
        failure_phases = {
            "FUNCTION_IDENTITY": "DISPATCH_FUNCTION_IDENTITY",
            "SPAWN_INDETERMINATE": "DISPATCH_SPAWN_INDETERMINATE",
            "CATALOG_INDETERMINATE": "DISPATCH_CATALOG_INDETERMINATE",
        }
        failure_phase = failure_phases.get(
            outcome.failure_stage if type(outcome) is ModalRuntimeQualificationOutcome else None
        )
        if failure_phase is not None and outcome.disposition == "indeterminate":
            raise ModalHostQualificationUnavailable(failure_phase) from None
        if outcome.disposition != "found" or type(outcome.provider_call_id) is not str:
            raise ValueError
        call_id = outcome.provider_call_id
        phase = "CALL_OBSERVE"
        deadline = time.monotonic() + 180
        while True:
            def poll():
                call = sdk.FunctionCall.from_id(call_id, client=client)
                hydrated = call.hydrate(client)
                if (hydrated is not call
                        or getattr(call, "is_hydrated", False) is not True
                        or getattr(call, "object_id", None) != call_id):
                    raise ValueError
                try:
                    return call.get(timeout=0)
                except TimeoutError:
                    return None

            result = _bounded(
                poll, deadline=min(deadline, time.monotonic() + 30),
                code="modal_host_cpu_call_observation_unavailable",
            )
            if result is None:
                if time.monotonic() >= deadline:
                    raise ValueError
                time.sleep(1)
                continue
            if (type(result) is not dict or result != {
                    "schema_version": QUALIFICATION_RESULT_SCHEMA,
                    "status_code": "completed",
            }):
                raise ValueError
            phase = "RECEIPT_VERIFY"
            observed = _bounded(
                lambda: reader.observe(dispatch, provider_call_id=call_id),
                deadline=min(deadline, time.monotonic() + 30),
                code="modal_host_cpu_receipt_unavailable",
            )
            if (type(observed) is not ModalRuntimeReleaseQualificationObservation
                    or type(observed.receipt) is not ModalRuntimeReleaseQualificationReceiptV1
                    or observed.receipt.provider_call_id != call_id
                    or observed.receipt.dispatch_digest != dispatch.dispatch_digest
                    or hashlib.sha256(observed.output).hexdigest()
                    != observed.receipt.output_sha256):
                raise ValueError
            return ModalHostCPUQualificationV1(
                observed.receipt, observed.receipt.output_sha256,
                call_id, runtime.release.manifest_digest,
                deployment.facts_digest,
            )
    except ModalHostQualificationUnavailable:
        raise
    except Exception:
        raise ModalHostQualificationUnavailable(phase) from None


__all__ = [
    "ModalHostCPUQualificationV1", "ModalHostQualificationUnavailable",
    "qualify_modal_runtime_for_host",
]
