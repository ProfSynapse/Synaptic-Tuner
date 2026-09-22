"""Read-only admission of an already deployed packaged Modal worker.

Deployment and image creation are release operations outside this adapter.  A
host injects an authenticated current-state reader using its explicit SDK
client; this module compares the complete observation with committed facts and
never deploys, mutates, or adopts a floating function.
"""

from __future__ import annotations

from typing import Protocol

from tuner.runtime.releases import (
    PackagedTrainingRuntimeReleaseV1,
    ProviderRuntimeBindingV1,
)

from .binding import ModalClientBinding
from .facade import EXACT_MODAL_SDK_VERSION
from .packaged_binding import ModalPackagedRuntimeFactsV1


class ModalPackagedDeploymentReadPort(Protocol):
    def observe(
        self,
        *,
        client: object,
        app_name: str,
        function_name: str,
        environment_name: str,
    ) -> ModalPackagedRuntimeFactsV1 | bytes: ...


class ModalPackagedDeploymentObserver:
    """Authenticate one exact private deployment without provider mutation."""

    __slots__ = ("_sdk", "_client", "_binding", "_reader")

    def __init__(
        self,
        *,
        sdk: object,
        client: object,
        client_binding: ModalClientBinding,
        reader: ModalPackagedDeploymentReadPort,
    ) -> None:
        if getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION:
            raise ValueError("Modal SDK version differs from packaged deployment")
        if client is None or type(client_binding) is not ModalClientBinding:
            raise TypeError("explicit Modal client and exact client binding required")
        if not hasattr(reader, "observe"):
            raise TypeError("packaged deployment read port required")
        if client_binding.sdk_version != EXACT_MODAL_SDK_VERSION:
            raise ValueError("client binding has unsupported Modal SDK version")
        self._sdk, self._client = sdk, client
        self._binding, self._reader = client_binding, reader

    @property
    def client(self) -> object:
        return self._client

    @property
    def client_binding(self) -> ModalClientBinding:
        return self._binding

    def observe(
        self,
        expected: ModalPackagedRuntimeFactsV1,
        *,
        runtime_release: PackagedTrainingRuntimeReleaseV1,
        provider_binding: ProviderRuntimeBindingV1,
    ) -> ModalPackagedRuntimeFactsV1:
        if type(expected) is not ModalPackagedRuntimeFactsV1:
            raise TypeError("exact committed Modal packaged facts required")
        if type(runtime_release) is not PackagedTrainingRuntimeReleaseV1:
            raise TypeError("exact packaged runtime release required")
        if type(provider_binding) is not ProviderRuntimeBindingV1:
            raise TypeError("exact provider runtime binding required")
        if expected.client_binding != self._binding:
            raise ValueError("committed deployment scope differs from explicit client")
        failure = False
        try:
            value = self._reader.observe(
                client=self._client,
                app_name=expected.app_name,
                function_name=expected.function_name,
                environment_name=expected.environment_ref,
            )
        except Exception:
            # Raise after leaving the handler so neither __cause__ nor
            # __context__ retains provider response text or credential data.
            failure = True
            value = None
        if failure:
            raise ValueError("Modal packaged deployment observation unavailable")
        if type(value) is bytes:
            observed = ModalPackagedRuntimeFactsV1.parse(value)
        elif type(value) is ModalPackagedRuntimeFactsV1:
            observed = ModalPackagedRuntimeFactsV1.from_dict(value.to_dict())
        else:
            raise ValueError("Modal packaged deployment observation is invalid")
        expected.validate_release(runtime_release)
        observed.validate_release(runtime_release)
        if (
            observed != expected
            or provider_binding != expected.build_provider_binding(runtime_release)
            or observed.build_provider_binding(runtime_release) != provider_binding
        ):
            raise ValueError("Modal packaged deployment changed")
        return observed


def observe_modal_packaged_deployment(
    observer: ModalPackagedDeploymentObserver,
    expected: ModalPackagedRuntimeFactsV1,
    *,
    runtime_release: PackagedTrainingRuntimeReleaseV1,
    provider_binding: ProviderRuntimeBindingV1,
) -> ModalPackagedRuntimeFactsV1:
    if type(observer) is not ModalPackagedDeploymentObserver:
        raise TypeError("exact packaged deployment observer required")
    return observer.observe(
        expected,
        runtime_release=runtime_release,
        provider_binding=provider_binding,
    )


__all__ = [
    "ModalPackagedDeploymentObserver",
    "ModalPackagedDeploymentReadPort",
    "observe_modal_packaged_deployment",
]
