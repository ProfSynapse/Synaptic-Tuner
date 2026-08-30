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
    ModalVerificationPolicyV1,
    compose_modal_source_finalizer,
)
from tuner.execution.providers.modal.deployment_v1 import (
    ModalDeploymentObjectsV1,
    ModalDeploymentSpecV1,
    build_modal_deployment,
)
from tuner.execution.providers.modal.deployment_identity import modal_function_name
from tuner.execution.providers.modal.facade import (
    EXACT_MODAL_SDK_VERSION,
    ExplicitModal154ReadFacade,
    ModalFacadeError,
    ModalFunctionCallState,
)
from tuner.execution.providers.modal.producer import MountedCompletionProducerV1
from tuner.execution.providers.modal.remote import MountedModalWorkerV1
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
from tuner.execution.providers.modal.training import (
    ModalDurablePreparationV1,
    ModalPreparedRunV1,
    ModalPlanContextV1,
    ModalTrainingOperations,
    ModalTrainingRepository,
    compose_modal_training_operations,
)

__all__ = [
    "ModalDurablePreparationV1",
    "ModalPreparedRunV1",
    "ModalClientBinding",
    "ModalDeploymentObjectsV1",
    "ModalDeploymentSelectionV1",
    "ModalDeploymentSpecV1",
    "ModalDualCloneSourceFinalizer",
    "ModalExecutionSourceResolutionV1",
    "ModalFacadeError",
    "ModalFunctionCallState",
    "ModalPlanContextV1",
    "ModalProviderProfileV1",
    "ModalRuntimeLockV1",
    "ModalSecretProfileV1",
    "ModalTrainingOperations",
    "ModalTrainingRepository",
    "ModalVerificationPolicyV1",
    "MountedCompletionProducerV1",
    "MountedModalWorkerV1",
    "EnvironmentHmacAuthenticator",
    "EXACT_MODAL_SDK_VERSION",
    "ExplicitModal154ReadFacade",
    "GitDualCloneMaterializer",
    "SubprocessSftRunner",
    "VerifiedModalDeploymentIdentityV1",
    "build_modal_deployment",
    "modal_function_name",
    "compose_modal_source_finalizer",
    "compose_modal_training_operations",
]
