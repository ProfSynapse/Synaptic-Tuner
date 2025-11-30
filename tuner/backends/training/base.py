"""
Location: /mnt/f/Code/Toolset-Training/tuner/backends/training/base.py

Purpose:
    Re-export ITrainingBackend interface for convenience.
    This allows backend implementations to import from .base instead of tuner.core.interfaces.

Usage:
    from .base import ITrainingBackend

    class RTXBackend(ITrainingBackend):
        ...
"""

from tuner.core.interfaces import ITrainingBackend

__all__ = ['ITrainingBackend']
