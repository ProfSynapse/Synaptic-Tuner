"""Factory for the one fixed Modal v1 deployment.

The optional SDK and explicit authenticated client are injected by the host.
Nothing in this module reads ambient Modal configuration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from ...contracts import safe_ref
from .deployment_identity import validate_modal_function_identity
from .facade import EXACT_MODAL_SDK_VERSION, MODAL_VOLUME_V1, ModalFacadeError

APP_NAME = "synaptic-training-v1"
FUNCTION_FAMILY = "run_sft_v1"
GPU = "A10"
CONTROL_MOUNT = "/workspace/control"
ARTIFACT_MOUNT = "/workspace/run"
_PINNED_IMAGE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
_SECRET_NAME = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|CREDENTIALS?)(?:_|$)"
)


@dataclass(frozen=True, slots=True)
class ModalDeploymentSpecV1:
    deployment_ref: str
    function_name: str
    registry_reference: str
    control_volume_name: str
    artifact_volume_name: str
    runtime_secret_name: str
    runtime_secret_keys: tuple[str, ...]
    environment: Mapping[str, str]
    timeout_seconds: int = 3600

    def __post_init__(self) -> None:
        deployment_ref, function_name = validate_modal_function_identity(
            self.deployment_ref, self.function_name
        )
        object.__setattr__(self, "deployment_ref", deployment_ref)
        object.__setattr__(self, "function_name", function_name)
        if not isinstance(self.registry_reference, str) or _PINNED_IMAGE.fullmatch(self.registry_reference) is None:
            raise ValueError("Modal runtime image must be digest pinned")
        for name in ("control_volume_name", "artifact_volume_name", "runtime_secret_name"):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))
        if self.control_volume_name == self.artifact_volume_name:
            raise ValueError("Modal control and artifact volumes must differ")
        keys = tuple(safe_ref(key, "runtime_secret_key") for key in self.runtime_secret_keys)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("runtime secret keys must be nonempty and unique")
        object.__setattr__(self, "runtime_secret_keys", keys)
        environment = dict(self.environment)
        if any(not isinstance(key, str) or not key or not isinstance(value, str) for key, value in environment.items()):
            raise ValueError("runtime environment must be a closed string map")
        forbidden_symbols = set(keys) | {
            "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SYNAPTIC_EVIDENCE_MAC_KEY",
            "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
        }
        if any(
            key in forbidden_symbols
            or _SECRET_NAME.search(key.upper()) is not None
            or value in forbidden_symbols
            for key, value in environment.items()
        ):
            raise ValueError("runtime secrets must be supplied only by named Modal Secrets")
        object.__setattr__(self, "environment", MappingProxyType(environment))
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 86400:
            raise ValueError("Modal timeout must be a bounded exact integer")


@dataclass(frozen=True, slots=True)
class ModalDeploymentObjectsV1:
    app: object
    function: object
    image: object
    control_volume: object
    artifact_volume: object
    runtime_secret: object


def build_modal_deployment(
    *,
    sdk: object,
    client: object,
    environment_name: str,
    spec: ModalDeploymentSpecV1,
    worker: Callable[[bytes, str], object],
) -> ModalDeploymentObjectsV1:
    """Build, but do not deploy or invoke, the immutable Modal application."""
    if getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION:
        raise ModalFacadeError("modal_sdk_version_mismatch")
    if client is None or not callable(worker):
        raise TypeError("explicit Modal client and worker are required")
    environment_name = safe_ref(environment_name, "environment_name")
    if type(spec) is not ModalDeploymentSpecV1:
        raise TypeError("ModalDeploymentSpecV1 is required")
    control = sdk.Volume.from_name(
        spec.control_volume_name,
        environment_name=environment_name,
        create_if_missing=False,
        version=MODAL_VOLUME_V1,
        client=client,
    )
    artifact = sdk.Volume.from_name(
        spec.artifact_volume_name,
        environment_name=environment_name,
        create_if_missing=False,
        version=MODAL_VOLUME_V1,
        client=client,
    )
    secret = sdk.Secret.from_name(
        spec.runtime_secret_name,
        environment_name=environment_name,
        required_keys=list(spec.runtime_secret_keys),
        client=client,
    )
    image = sdk.Image.from_registry(spec.registry_reference).entrypoint([]).env(dict(spec.environment))
    app = sdk.App(APP_NAME, image=image, include_source=True)

    @app.function(
        name=spec.function_name,
        serialized=True,
        image=image,
        gpu=GPU,
        volumes={CONTROL_MOUNT: control, ARTIFACT_MOUNT: artifact},
        secrets=[secret],
        retries=0,
        timeout=spec.timeout_seconds,
        include_source=True,
        restrict_modal_access=True,
        single_use_containers=True,
    )
    def run_sft_v1(canonical_command: bytes):
        if not isinstance(canonical_command, bytes):
            raise ValueError("canonical mutation command bytes are required")
        job_ref = sdk.current_function_call_id()
        if not isinstance(job_ref, str) or not job_ref:
            raise ValueError("Modal function call identity is unavailable")
        result = worker(canonical_command, job_ref)
        artifact.commit()
        control.commit()
        return result

    return ModalDeploymentObjectsV1(app, run_sft_v1, image, control, artifact, secret)


__all__ = [
    "APP_NAME", "ARTIFACT_MOUNT", "CONTROL_MOUNT", "FUNCTION_FAMILY", "GPU",
    "ModalDeploymentObjectsV1", "ModalDeploymentSpecV1", "build_modal_deployment",
]
