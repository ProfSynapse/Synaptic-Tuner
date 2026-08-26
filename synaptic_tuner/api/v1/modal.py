"""Explicit Modal v1 contracts for host composition and persistence.

Importing this provider-specific module loads no Modal SDK.  The consuming host
still supplies an authenticated explicit SDK client and its own persistence.
"""

from tuner.execution.providers.modal.config import (
    ModalProviderProfileV1,
    ModalRuntimeLockV1,
    ModalSecretProfileV1,
)
from tuner.execution.providers.modal.training import (
    ModalDurablePreparationV1,
    ModalPlanContextV1,
    ModalTrainingOperations,
    ModalTrainingRepository,
    compose_modal_training_operations,
)

__all__ = [
    "ModalDurablePreparationV1",
    "ModalPlanContextV1",
    "ModalProviderProfileV1",
    "ModalRuntimeLockV1",
    "ModalSecretProfileV1",
    "ModalTrainingOperations",
    "ModalTrainingRepository",
    "compose_modal_training_operations",
]
