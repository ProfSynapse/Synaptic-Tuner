"""Unactivated deployment builder for the Foundation-native Modal worker."""
from __future__ import annotations

import re

from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object, safe_ref

from .coordinator_producer import MountedModalCoordinatorProducer
from .coordinator_worker import ModalWorkerStaticExpectation, MountedModalCoordinatorWorker
from .binding import ModalClientBinding
from .config import ModalProviderProfileV1, ModalRuntimeLockV1
from .deployment_v1 import (
    APP_NAME, ARTIFACT_MOUNT, BOOTSTRAP_SOURCE_MODULES, CONTROL_MOUNT, GPU,
    ModalDeploymentObjectsV1, ModalDeploymentSpecV1,
)
from .facade import EXACT_MODAL_SDK_VERSION, MODAL_VOLUME_V1, ModalFacadeError
from .resolution import ModalDeploymentSelectionV1
from .runtime import EnvironmentHmacAuthenticator, GitDualCloneMaterializer, SubprocessSftRunner


def build_modal_coordinator_deployment(
    *, sdk: object, client: object, environment_name: str,
    spec: ModalDeploymentSpecV1, profile: ModalProviderProfileV1,
    deployment_selection_bytes: bytes,
    provider_id: str, profile_ref: str, executor_id: str,
    executor_implementation_version: str, control_volume_id: str,
    artifact_volume_id: str, evidence_environment_key: str,
    evidence_key_ref: str, model_token_key: str,
) -> ModalDeploymentObjectsV1:
    """Build, but never deploy or invoke, the candidate coordinator function."""
    if getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION:
        raise ModalFacadeError("modal_sdk_version_mismatch")
    if client is None or type(spec) is not ModalDeploymentSpecV1 or type(profile) is not ModalProviderProfileV1:
        raise TypeError("explicit client, profile, and exact Modal deployment spec are required")
    if type(deployment_selection_bytes) is not bytes:
        raise TypeError("exact deployment selection bytes are required")
    environment_name = safe_ref(environment_name, "environment_name")
    selection = ModalDeploymentSelectionV1.from_dict(
        parse_canonical_object(deployment_selection_bytes, name="deployment selection")
    )
    if canonical_bytes(selection.to_dict()) != deployment_selection_bytes:
        raise ValueError("deployment selection is not canonical")
    for name, value in (
        ("provider_id", provider_id), ("profile_ref", profile_ref),
        ("executor_id", executor_id),
        ("executor_implementation_version", executor_implementation_version),
        ("control_volume_id", control_volume_id),
        ("artifact_volume_id", artifact_volume_id),
        ("evidence_key_ref", evidence_key_ref),
    ):
        safe_ref(value, name)
    if environment_name != selection.environment_ref:
        raise ValueError("candidate deployment environment differs")
    if (
        provider_id != "modal"
        or executor_id != "modal-coordinator-executor"
        or executor_implementation_version != "0.1.0"
        or selection.sdk_version != EXACT_MODAL_SDK_VERSION
    ):
        raise ValueError("candidate deployment provider, executor, or SDK differs")
    runtime_lock = ModalRuntimeLockV1.packaged()
    if (
        selection.app_name != APP_NAME
        or selection.function_name != spec.function_name
        or selection.deployment_ref != spec.deployment_ref
        or spec.registry_reference != runtime_lock.registry_reference
        or selection.timeout_seconds != spec.timeout_seconds
        or selection.accelerator != GPU
        or selection.max_retries != 0
        or dict(selection.runtime_environment) != dict(spec.environment)
    ):
        raise ValueError("candidate deployment static selection differs")
    runtime_lock.validate_selection(selection)
    if (
        profile.profile != profile_ref
        or profile.app_name != APP_NAME
        or profile.function_name != spec.function_name
        or profile.deployment_ref != spec.deployment_ref
        or profile.control_volume_ref != spec.control_volume_name
        or profile.artifact_volume_ref != spec.artifact_volume_name
        or len(profile.secrets) != 1
        or profile.secrets[0].name != spec.runtime_secret_name
        or profile.secrets[0].required_keys != spec.runtime_secret_keys
    ):
        raise ValueError("candidate deployment profile and spec differ")
    expected_selection = ModalDeploymentSelectionV1.from_profile(
        profile, binding=ModalClientBinding(
            selection.account_ref, selection.workspace_ref, selection.environment_ref,
            selection.client_ref, selection.sdk_version,
        ),
        runtime_environment=spec.environment, timeout_seconds=spec.timeout_seconds,
    )
    if selection != expected_selection:
        raise ValueError("candidate deployment selection differs from canonical profile")
    if control_volume_id == artifact_volume_id:
        raise ValueError("candidate deployment volumes must differ")
    environment_symbol = re.compile(r"[A-Z][A-Z0-9_]{0,127}").fullmatch
    if (
        type(evidence_environment_key) is not str
        or type(model_token_key) is not str
        or environment_symbol(evidence_environment_key) is None
        or environment_symbol(model_token_key) is None
        or evidence_environment_key == model_token_key
        or set(spec.runtime_secret_keys) != {evidence_environment_key, model_token_key}
        or {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}.intersection(spec.runtime_secret_keys)
    ):
        raise ValueError("candidate worker Secret must declare only model and evidence keys")

    static = ModalWorkerStaticExpectation(
        deployment_selection_bytes, provider_id, profile_ref, executor_id,
        executor_implementation_version, control_volume_id, artifact_volume_id,
        spec.control_volume_name, spec.artifact_volume_name, evidence_key_ref,
        CONTROL_MOUNT, ARTIFACT_MOUNT, "/workspace/worker-control",
    )

    control = sdk.Volume.from_name(
        spec.control_volume_name, environment_name=environment_name,
        create_if_missing=False, version=MODAL_VOLUME_V1, client=client,
    )
    artifact = sdk.Volume.from_name(
        spec.artifact_volume_name, environment_name=environment_name,
        create_if_missing=False, version=MODAL_VOLUME_V1, client=client,
    )
    secret = sdk.Secret.from_name(
        spec.runtime_secret_name, environment_name=environment_name,
        required_keys=list(spec.runtime_secret_keys), client=client,
    )
    image = (
        sdk.Image.from_registry(spec.registry_reference).entrypoint([])
        .env(dict(spec.environment))
        .add_local_python_source(*BOOTSTRAP_SOURCE_MODULES, copy=False, ignore=[])
    )
    app = sdk.App(APP_NAME, image=image, include_source=False)
    @app.function(
        name=spec.function_name, serialized=True, image=image, gpu=GPU,
        volumes={CONTROL_MOUNT: control, ARTIFACT_MOUNT: artifact},
        secrets=[secret], retries=0, timeout=spec.timeout_seconds,
        include_source=False, restrict_modal_access=True, single_use_containers=True,
    )
    def coordinator_worker_entry(dispatch_bytes: bytes):
        if type(dispatch_bytes) is not bytes:
            raise ValueError("canonical Modal worker dispatch bytes are required")
        job_ref = sdk.current_function_call_id()
        if type(job_ref) is not str or not job_ref:
            raise ValueError("Modal function call identity is unavailable")
        authenticator = EnvironmentHmacAuthenticator(
            environment_key=evidence_environment_key, key_ref=evidence_key_ref,
        )
        worker = MountedModalCoordinatorWorker(
            verifier=authenticator, sources=GitDualCloneMaterializer(),
            processes=SubprocessSftRunner(
                secret_keys=spec.runtime_secret_keys, model_token_key=model_token_key,
                timeout_seconds=spec.timeout_seconds,
            ), completion=MountedModalCoordinatorProducer(authenticator), static=static,
        )
        result = worker(dispatch_bytes, job_ref, artifact.commit)
        artifact.commit()
        control.commit()
        return result

    return ModalDeploymentObjectsV1(app, coordinator_worker_entry, image, control, artifact, secret)


__all__: list[str] = []
