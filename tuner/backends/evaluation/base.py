"""
Base evaluation backend interface.

Location: /mnt/f/Code/Toolset-Training/tuner/backends/evaluation/base.py
Purpose: Re-export IEvaluationBackend from core.interfaces
Used by: All evaluation backend implementations

This module provides a convenient import location for the evaluation backend
interface, avoiding circular imports and providing a clear API boundary.
"""

from tuner.core.interfaces import IEvaluationBackend

__all__ = ["IEvaluationBackend"]
