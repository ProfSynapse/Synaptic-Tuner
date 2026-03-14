"""
Cloud training backend implementations.

Location: tuner/backends/training/cloud/__init__.py
Purpose: Export cloud backend classes and register them in TrainingBackendRegistry
Used by: tuner/backends/registry.py, tuner/handlers/cloud_train_handler.py

Each cloud provider backend is imported conditionally (try/except) so that
the CLI works even when provider SDKs are not installed. Only backends whose
SDKs are available will be exported and registered.

Providers:
- HFJobsBackend: HuggingFace Jobs (requires huggingface_hub >= 0.27.0)
- ModalBackend: Modal (requires modal SDK)
- RunPodBackend: RunPod (requires runpod SDK)
"""

import logging

logger = logging.getLogger(__name__)

# Track which backends are available
AVAILABLE_BACKENDS = {}

# HuggingFace Jobs backend
try:
    from .hf_jobs_backend import HFJobsBackend
    AVAILABLE_BACKENDS["hf_jobs"] = HFJobsBackend
except ImportError as e:
    logger.debug("HF Jobs backend not available: %s", e)
    HFJobsBackend = None

# Modal backend
try:
    from .modal_backend import ModalBackend
    AVAILABLE_BACKENDS["modal"] = ModalBackend
except ImportError as e:
    logger.debug("Modal backend not available: %s", e)
    ModalBackend = None

# RunPod backend
try:
    from .runpod_backend import RunPodBackend
    AVAILABLE_BACKENDS["runpod"] = RunPodBackend
except ImportError as e:
    logger.debug("RunPod backend not available: %s", e)
    RunPodBackend = None

__all__ = [
    "AVAILABLE_BACKENDS",
    "HFJobsBackend",
    "ModalBackend",
    "RunPodBackend",
]
