"""
Location: /mnt/f/Code/Toolset-Training/tuner/backends/training/__init__.py

Purpose:
    Export training backend implementations for easy importing.

Usage:
    from tuner.backends.training import RTXBackend, MacBackend, ITrainingBackend

Dependencies:
    - tuner.backends.training.base (re-exports ITrainingBackend)
    - tuner.backends.training.rtx_backend
    - tuner.backends.training.mac_backend
    - tuner.backends.training.cloud (optional cloud backends)
"""

from .base import ITrainingBackend
from .rtx_backend import RTXBackend
from .mac_backend import MacBackend

__all__ = [
    'ITrainingBackend',
    'RTXBackend',
    'MacBackend',
]

# Conditionally export cloud backends if available
try:
    from .cloud import HFJobsBackend, ModalBackend, RunPodBackend, AVAILABLE_BACKENDS
    if HFJobsBackend is not None:
        __all__.append('HFJobsBackend')
    if ModalBackend is not None:
        __all__.append('ModalBackend')
    if RunPodBackend is not None:
        __all__.append('RunPodBackend')
    __all__.append('AVAILABLE_BACKENDS')
except ImportError:
    AVAILABLE_BACKENDS = {}
    HFJobsBackend = None
    ModalBackend = None
    RunPodBackend = None
