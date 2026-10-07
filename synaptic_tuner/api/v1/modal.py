"""Explicit Modal v1 contracts for host composition and persistence.

Importing this provider-specific module loads no Modal SDK.  The consuming host
still supplies an authenticated explicit SDK client and its own persistence.
"""

from tuner.execution.providers.modal.config import (
    ModalProviderProfileV1,
    ModalRuntimeLockV1,
    ModalSecretProfileV1,
)
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.composition import (
    ModalSourceVerificationPorts,
    ModalVerificationPolicyV1,
    compose_modal_source_finalizer,
)
from tuner.execution.providers.modal.deployment_v1 import (
    ModalDeploymentObjectsV1,
    ModalDeploymentSpecV1,
)
from tuner.execution.providers.modal.coordinator_deployment import build_modal_coordinator_deployment
from tuner.execution.providers.modal.coordinator_composition import (
    ModalCoordinatorComposition, ModalCoordinatorStorePorts,
    ModalFoundationCompositionPorts, compose_modal_coordinator,
)
from tuner.execution.providers.modal.coordinator_adapter import ModalPreparationAdapter
from tuner.execution.providers.modal.coordinator_effects import (
    ModalFoundationEffectExecutor, ModalFoundationReconciliationAdapter,
)
from tuner.execution.providers.modal.coordinator_factories import modal_coordinator_registration
from tuner.execution.providers.modal.coordinator_preflight import (
    AuthenticatedModalQuote, ModalOperationalPreflightAdapter, ModalQuoteBody,
    TrustedEvidenceIdentity,
)
from tuner.execution.providers.modal.coordinator_reader import ModalCoordinatorRunReader
from tuner.execution.providers.modal.coordinator_read_transport import ModalFoundationReadTransport
from tuner.execution.providers.modal.coordinator_retention import (
    ModalFoundationRetentionDelegate, ModalRetainedPreparation,
)
from tuner.execution.providers.modal.coordinator_transport import ModalFoundationHostTransport
from tuner.execution.providers.modal.deployment_identity import modal_function_name
from tuner.execution.providers.modal.facade import (
    EXACT_MODAL_SDK_VERSION,
    ExplicitModal154ReadFacade,
    ModalFacadeError,
    ModalFunctionCallState,
)
from tuner.execution.providers.modal.resolution import (
    ModalDeploymentSelectionV1,
    ModalDualCloneSourceFinalizer,
    ModalExecutionSourceResolutionV1,
    VerifiedModalDeploymentIdentityV1,
)
from tuner.execution.providers.modal.runtime import (
    EnvironmentHmacAuthenticator,
    GitDualCloneMaterializer,
    SubprocessSftRunner,
)

__all__ = [
    "AuthenticatedModalQuote",
    "ModalCoordinatorComposition",
    "ModalCoordinatorStorePorts",
    "ModalFoundationCompositionPorts",
    "compose_modal_coordinator",
    "ModalCoordinatorRunReader",
    "ModalFoundationEffectExecutor",
    "ModalFoundationHostTransport",
    "ModalFoundationReadTransport",
    "ModalFoundationReconciliationAdapter",
    "ModalFoundationRetentionDelegate",
    "ModalOperationalPreflightAdapter",
    "ModalPreparationAdapter",
    "ModalQuoteBody",
    "ModalRetainedPreparation",
    "ModalSourceVerificationPorts",
    "TrustedEvidenceIdentity",
    "ModalClientBinding",
    "ModalDeploymentObjectsV1",
    "ModalDeploymentSelectionV1",
    "ModalDeploymentSpecV1",
    "ModalDualCloneSourceFinalizer",
    "ModalExecutionSourceResolutionV1",
    "ModalFacadeError",
    "ModalFunctionCallState",
    "ModalProviderProfileV1",
    "ModalRuntimeLockV1",
    "ModalSecretProfileV1",
    "ModalVerificationPolicyV1",
    "EnvironmentHmacAuthenticator",
    "EXACT_MODAL_SDK_VERSION",
    "ExplicitModal154ReadFacade",
    "GitDualCloneMaterializer",
    "SubprocessSftRunner",
    "VerifiedModalDeploymentIdentityV1",
    "build_modal_coordinator_deployment",
    "modal_coordinator_registration",
    "modal_function_name",
    "compose_modal_source_finalizer",
]
