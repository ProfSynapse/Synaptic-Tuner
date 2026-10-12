"""One-shot CPU qualification of an acknowledged packaged Modal deployment.

This is a host adapter over the existing release qualification operator and
reader. It does not prepare training input, invoke the GPU function, or infer a
Function-to-Image link from floating current state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
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
    PARENT_RUNTIME_FAILURE_STAGES,
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
        "TRANSPORT_MARKER_STAGE": ("UNAVAILABLE", "modal_host_qualification.stage_transport_marker"),
        "DISPATCH_SUBMIT": ("INDETERMINATE", "modal_host_qualification.submit_once"),
        "DISPATCH_FUNCTION_IDENTITY": (
            "UNAVAILABLE", "modal_runtime_qualification_operator.function_identity"),
        "DISPATCH_SPAWN_INDETERMINATE": (
            "INDETERMINATE", "modal_runtime_qualification_operator.spawn"),
        "DISPATCH_CATALOG_INDETERMINATE": (
            "INDETERMINATE", "modal_runtime_qualification_operator.call_catalog"),
        "CALL_OBSERVE": ("UNAVAILABLE", "modal_host_qualification.observe_call"),
        "CALL_PARENT_SETUP": (
            "UNAVAILABLE", "runtime_release_modal_self_check.parent_setup"),
        "CALL_INSTALLED_CHILD": (
            "UNAVAILABLE", "modal_runtime_release_qualification.installed_child"),
        "CALL_PARENT_RELEASE": (
            "UNAVAILABLE", "packaged_training_worker.parent_release"),
        "CALL_CHILD_RESULT": (
            "UNAVAILABLE", "packaged_training_worker.child_result"),
        "RECEIPT_VERIFY": ("UNAVAILABLE", "modal_host_qualification.verify_receipt"),
        **{"CALL_" + stage: ("UNAVAILABLE", "modal_runtime_release_qualification.parent_runtime")
           for stage in PARENT_RUNTIME_FAILURE_STAGES},
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
    deployment_facts: object = None
    transport_probe: object = None
    transport_verification: object = None
    transport_capture_path: str | None = None
    transport_capture_sha256: str | None = None
    transport_status: str = "not_requested"


def _retain_transport_capture(root: Path, effect_id: str, call_id: str, capture):
    """Private create-only/fsynced capture; bytes cannot rehydrate authority."""
    from tuner.training.modal_local_preparation import private_directory
    from tuner.execution.providers.modal.runtime_release_qualification_reader import ModalCPUTransportCaptureV1
    if type(capture) is not ModalCPUTransportCaptureV1:
        raise ValueError("exact transport capture required")
    raw = json.dumps(capture.to_dict(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not 0 < len(raw) <= 1048576:
        raise ValueError("transport capture exceeds bound")
    directory = private_directory(root / "cpu-transport-evidence")
    name = hashlib.sha256(canonical_bytes({"effect_id": effect_id, "call_id": call_id})).hexdigest() + ".json"
    parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = None
    try:
        parent_info = os.fstat(parent)
        if parent_info.st_uid != os.geteuid() or stat.S_IMODE(parent_info.st_mode) != 0o700:
            raise ValueError
        descriptor = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        original = os.fstat(descriptor)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise ValueError
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.fsync(parent)
        os.lseek(descriptor, 0, os.SEEK_SET)
        retained = bytearray()
        while len(retained) <= 1048576:
            block = os.read(descriptor, min(65536, 1048577 - len(retained)))
            if not block:
                break
            retained.extend(block)
        final, named = os.fstat(descriptor), os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (bytes(retained) != raw or (final.st_dev, final.st_ino) != (original.st_dev, original.st_ino)
                or (named.st_dev, named.st_ino) != (original.st_dev, original.st_ino)
                or final.st_uid != os.geteuid() or stat.S_IMODE(final.st_mode) != 0o600
                or final.st_nlink != 1 or final.st_size != len(raw)
                or os.fstat(parent).st_uid != os.geteuid() or stat.S_IMODE(os.fstat(parent).st_mode) != 0o700
                or (directory.lstat().st_dev, directory.lstat().st_ino) != (parent_info.st_dev, parent_info.st_ino)):
            raise ValueError
        from tuner.runtime.packaged_worker_closure import stable_read
        private_directory(root, create=False)
        private_directory(directory, create=False)
        if stable_read(directory / name, 1048576) != raw:
            raise ValueError
        return str(directory / name), hashlib.sha256(raw).hexdigest(), raw
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


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
    capture_progress: bool = False, transport_root: Path | None = None,
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
    if type(capture_progress) is not bool:
        raise ValueError("exact transport selection required")
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
    probe, marker = None, None
    if (capture_progress and runtime.deployment_facts.schema_version == "synaptic-modal-runtime-release-deployment-facts/v2"
            and runtime.facts.schema_version == "synaptic-modal-packaged-runtime-facts/v2"):
        from tuner.execution.providers.modal.runtime_release_qualification import ModalCPUTransportProbeV1
        from tuner.training.modal_local_preparation import private_directory
        if not isinstance(transport_root, Path) or not transport_root.is_absolute():
            raise ValueError("private transport evidence root required")
        private_directory(transport_root, create=False)
        marker = secrets.token_bytes(32)
        probe = ModalCPUTransportProbeV1("probe-" + secrets.token_hex(16) + ".bin", hashlib.sha256(marker).hexdigest())
    dispatch = ModalRuntimeReleaseQualificationDispatchV1(
        effect_id, runtime.release, binding, deployment, fixture,
        ModalRuntimeReleaseQualificationPolicyV1(), QUALIFICATION_HMAC_KEY_REF,
        **({"transport_probe": probe} if probe is not None else {}),
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
            **({"transport_probe": probe.to_dict()} if probe is not None else {}),
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
        if probe is not None:
            phase = "TRANSPORT_MARKER_STAGE"
            _bounded(lambda: operator.stage_transport_marker(probe=probe, marker_bytes=marker,
                deployment_facts=deployment), deadline=time.monotonic() + 30,
                code="modal_host_cpu_transport_marker_unavailable")
        phase = "DISPATCH_SUBMIT"
        outcome = operator.submit_once(
            payload, expected_facts=deployment,
            provider_invoker=lambda operation: _bounded(
                operation, deadline=time.monotonic() + 60,
                code="modal_host_cpu_dispatch_indeterminate",
            ),
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
        capture, capture_path, capture_sha256, retained_capture = None, None, None, None
        transport_status = "not_requested" if probe is None else "capture_unavailable"
        if probe is not None:
            try:
                # Optional visibility must leave a finite budget for primary CPU proof.
                capture = reader.capture_transport_probe(dispatch, provider_call_id=call_id,
                    host_deadline=deadline - 60)
                transport_status = "retention_unavailable"
                capture_path, capture_sha256, retained_capture = _retain_transport_capture(
                    transport_root, effect_id, call_id, capture)
                transport_status = "verification_pending"
            except Exception:
                capture = None
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
            failed = {
                "schema_version": QUALIFICATION_RESULT_SCHEMA,
                "status_code": "failed",
            }
            if type(result) is dict and result == {**failed, "failure_stage": "PARENT_SETUP"}:
                raise ModalHostQualificationUnavailable("CALL_PARENT_SETUP") from None
            if type(result) is dict and result == {**failed, "failure_stage": "INSTALLED_CHILD"}:
                raise ModalHostQualificationUnavailable("CALL_INSTALLED_CHILD") from None
            if type(result) is dict and result == {**failed, "failure_stage": "PARENT_RELEASE"}:
                raise ModalHostQualificationUnavailable("CALL_PARENT_RELEASE") from None
            if (type(result) is dict and result.get("failure_stage") in PARENT_RUNTIME_FAILURE_STAGES
                    and result == {**failed, "failure_stage": result["failure_stage"]}):
                raise ModalHostQualificationUnavailable("CALL_" + result["failure_stage"]) from None
            if type(result) is dict and result == {**failed, "failure_stage": "CHILD_RESULT"}:
                raise ModalHostQualificationUnavailable("CALL_CHILD_RESULT") from None
            completed = {
                "schema_version": QUALIFICATION_RESULT_SCHEMA,
                "status_code": "completed",
            }
            optional_unavailable = {**completed, "transport_probe_status": "unavailable"}
            if (type(result) is not dict or (result != completed
                    and not (probe is not None and result == optional_unavailable))):
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
            transport_verification = None
            if probe is not None and capture is not None:
                try:
                    transport_verification = _bounded(lambda: reader.verify_transport_probe(
                        dispatch, provider_call_id=call_id, capture=capture,
                        retained_capture_bytes=retained_capture),
                        deadline=min(deadline, time.monotonic() + 30),
                        code="modal_host_cpu_transport_verify_unavailable")
                    transport_status = transport_verification.to_dict()["status"]
                except Exception:
                    transport_status = "verification_unavailable"
                    transport_verification = None
            return ModalHostCPUQualificationV1(
                observed.receipt, observed.receipt.output_sha256,
                call_id, runtime.release.manifest_digest,
                deployment.facts_digest,
                deployment_facts=deployment, transport_probe=probe,
                transport_verification=transport_verification,
                transport_capture_path=capture_path, transport_capture_sha256=capture_sha256,
                transport_status=transport_status,
            )
    except ModalHostQualificationUnavailable:
        raise
    except Exception:
        raise ModalHostQualificationUnavailable(phase) from None


__all__ = [
    "ModalHostCPUQualificationV1", "ModalHostQualificationUnavailable",
    "qualify_modal_runtime_for_host",
]
