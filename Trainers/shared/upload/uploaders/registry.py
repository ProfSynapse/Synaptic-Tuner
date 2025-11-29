"""
Uploader registry.
"""

from typing import Dict, List, Type

from .base import BaseUploader


class UploaderRegistry:
    """
    Registry for uploaders.

    Allows registration of new uploaders and lookup by name.
    """

    _uploaders: Dict[str, Type[BaseUploader]] = {}

    @classmethod
    def register(cls, name: str, uploader_class: Type[BaseUploader]) -> None:
        """
        Register a new uploader.

        Args:
            name: Unique identifier for the uploader
            uploader_class: Uploader class to register
        """
        cls._uploaders[name] = uploader_class

    @classmethod
    def get(cls, name: str) -> BaseUploader:
        """
        Get an uploader instance by name.

        Args:
            name: Name of the uploader

        Returns:
            Uploader instance

        Raises:
            ValueError: If uploader name is not found
        """
        uploader_class = cls._uploaders.get(name)
        if not uploader_class:
            available = ", ".join(cls.list_uploaders())
            raise ValueError(
                f"Unknown uploader: '{name}'. "
                f"Available uploaders: {available}"
            )

        return uploader_class()

    @classmethod
    def list_uploaders(cls) -> List[str]:
        """
        List all registered uploader names.

        Returns:
            List of uploader names
        """
        return list(cls._uploaders.keys())
