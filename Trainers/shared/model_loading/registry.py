"""
Model loader registry.
"""

from typing import Dict, List, Optional, Type

from .base import BaseModelLoader


class ModelLoaderRegistry:
    """
    Registry for model loaders.

    Allows registration of new loaders and lookup by name.
    """

    _loaders: Dict[str, Type[BaseModelLoader]] = {}

    @classmethod
    def register(cls, name: str, loader_class: Type[BaseModelLoader]) -> None:
        """
        Register a new model loader.

        Args:
            name: Unique identifier for the loader
            loader_class: Loader class to register
        """
        cls._loaders[name] = loader_class

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseModelLoader:
        """
        Get a loader instance by name.

        Args:
            name: Name of the loader
            **kwargs: Arguments to pass to loader constructor

        Returns:
            Loader instance

        Raises:
            ValueError: If loader name is not found
        """
        loader_class = cls._loaders.get(name)
        if not loader_class:
            available = ", ".join(cls.list_loaders())
            raise ValueError(
                f"Unknown model loader: '{name}'. "
                f"Available loaders: {available}"
            )

        return loader_class(**kwargs)

    @classmethod
    def list_loaders(cls) -> List[str]:
        """
        List all registered loader names.

        Returns:
            List of loader names
        """
        return list(cls._loaders.keys())
