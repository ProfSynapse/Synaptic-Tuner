"""
Backend registry for training and evaluation backends.

Location: /mnt/f/Code/Toolset-Training/tuner/backends/registry.py
Purpose: Central registry for backend discovery and instantiation
Used by: Handlers to get backend instances without tight coupling

This module implements the Registry pattern to decouple handlers from concrete
backend implementations. Handlers request backends by name (e.g., 'rtx', 'ollama')
and the registry returns the appropriate implementation.

Design decisions:
- Pre-register all known backends at module import time
- Use class methods for registry operations (singleton pattern)
- Return instances from get() method (not classes)
- Support dynamic registration for extensibility
- Separate registries for training vs evaluation backends
"""

from typing import Dict, Type, List
from pathlib import Path

from tuner.core.interfaces import ITrainingBackend, IEvaluationBackend
from tuner.backends.training import RTXBackend, MacBackend
from tuner.backends.evaluation import OllamaBackend, LMStudioBackend


class TrainingBackendRegistry:
    """
    Registry for training backend implementations.

    This registry manages the mapping between platform names ('rtx', 'mac')
    and their corresponding backend implementations. It provides a central
    point for backend discovery and instantiation.

    The registry pre-registers known backends at module import time, but
    supports dynamic registration for extensibility.

    Example:
        # Get a backend instance
        backend = TrainingBackendRegistry.get('rtx', repo_root=Path('/path/to/repo'))

        # List available backends
        backends = TrainingBackendRegistry.list()  # ['rtx', 'mac']

        # Register a custom backend
        TrainingBackendRegistry.register('custom', CustomBackend)
    """

    _backends: Dict[str, Type[ITrainingBackend]] = {}

    @classmethod
    def register(cls, name: str, backend: Type[ITrainingBackend]) -> None:
        """
        Register a new training backend.

        Args:
            name: Backend identifier (e.g., 'rtx', 'mac', 'custom')
            backend: Backend class (not instance) implementing ITrainingBackend

        Example:
            class CustomBackend(ITrainingBackend):
                # Implementation...
                pass

            TrainingBackendRegistry.register('custom', CustomBackend)

        Note:
            If a backend with the same name already exists, it will be overwritten.
        """
        cls._backends[name] = backend

    @classmethod
    def get(cls, name: str, **kwargs) -> ITrainingBackend:
        """
        Get a training backend instance by name.

        Args:
            name: Backend identifier (e.g., 'rtx', 'mac')
            **kwargs: Arguments to pass to backend constructor

        Returns:
            Backend instance ready for use

        Raises:
            ValueError: If backend name is not registered

        Example:
            backend = TrainingBackendRegistry.get(
                'rtx',
                repo_root=Path('/path/to/repo')
            )
            methods = backend.get_available_methods()  # ['sft', 'kto']
        """
        if name not in cls._backends:
            available = ", ".join(cls.list())
            raise ValueError(
                f"Unknown training backend: '{name}'. "
                f"Available backends: {available}"
            )
        return cls._backends[name](**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        """
        List all registered training backend names.

        Returns:
            List of backend identifiers (e.g., ['rtx', 'mac'])

        Example:
            for backend_name in TrainingBackendRegistry.list():
                print(f"Available: {backend_name}")
        """
        return list(cls._backends.keys())


class EvaluationBackendRegistry:
    """
    Registry for evaluation backend implementations.

    This registry manages the mapping between backend names ('ollama', 'lmstudio')
    and their corresponding backend implementations. It provides a central
    point for backend discovery and instantiation.

    The registry pre-registers known backends at module import time, but
    supports dynamic registration for extensibility.

    Example:
        # Get a backend instance
        backend = EvaluationBackendRegistry.get('ollama')

        # List available backends
        backends = EvaluationBackendRegistry.list()  # ['ollama', 'lmstudio']

        # Register a custom backend
        EvaluationBackendRegistry.register('custom', CustomEvalBackend)
    """

    _backends: Dict[str, Type[IEvaluationBackend]] = {}

    @classmethod
    def register(cls, name: str, backend: Type[IEvaluationBackend]) -> None:
        """
        Register a new evaluation backend.

        Args:
            name: Backend identifier (e.g., 'ollama', 'lmstudio', 'custom')
            backend: Backend class (not instance) implementing IEvaluationBackend

        Example:
            class CustomEvalBackend(IEvaluationBackend):
                # Implementation...
                pass

            EvaluationBackendRegistry.register('custom', CustomEvalBackend)

        Note:
            If a backend with the same name already exists, it will be overwritten.
        """
        cls._backends[name] = backend

    @classmethod
    def get(cls, name: str, **kwargs) -> IEvaluationBackend:
        """
        Get an evaluation backend instance by name.

        Args:
            name: Backend identifier (e.g., 'ollama', 'lmstudio')
            **kwargs: Arguments to pass to backend constructor (usually none needed)

        Returns:
            Backend instance ready for use

        Raises:
            ValueError: If backend name is not registered

        Example:
            backend = EvaluationBackendRegistry.get('ollama')
            models = backend.list_models()  # ['llama2:7b', 'mistral:latest']
        """
        if name not in cls._backends:
            available = ", ".join(cls.list())
            raise ValueError(
                f"Unknown evaluation backend: '{name}'. "
                f"Available backends: {available}"
            )
        return cls._backends[name](**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        """
        List all registered evaluation backend names.

        Returns:
            List of backend identifiers (e.g., ['ollama', 'lmstudio'])

        Example:
            for backend_name in EvaluationBackendRegistry.list():
                print(f"Available: {backend_name}")
        """
        return list(cls._backends.keys())


# Pre-register known backends at module import time
# This ensures they're always available without manual registration

TrainingBackendRegistry.register("rtx", RTXBackend)
TrainingBackendRegistry.register("mac", MacBackend)

EvaluationBackendRegistry.register("ollama", OllamaBackend)
EvaluationBackendRegistry.register("lmstudio", LMStudioBackend)
