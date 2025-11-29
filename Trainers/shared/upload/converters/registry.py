"""
Converter registry.
"""

from typing import Dict, List, Optional, Type

from .base import BaseConverter
from ..core.interfaces import IModelLoader


class ConverterRegistry:
    """
    Registry for format converters.

    Allows registration of new converters and lookup by name.
    """

    _converters: Dict[str, Type[BaseConverter]] = {}

    @classmethod
    def register(cls, name: str, converter_class: Type[BaseConverter]) -> None:
        """
        Register a new converter.

        Args:
            name: Unique identifier for the converter
            converter_class: Converter class to register
        """
        cls._converters[name] = converter_class

    @classmethod
    def get(
        cls,
        name: str,
        model_loader: Optional[IModelLoader] = None
    ) -> BaseConverter:
        """
        Get a converter instance by name.

        Args:
            name: Name of the converter
            model_loader: Model loader to inject

        Returns:
            Converter instance

        Raises:
            ValueError: If converter name is not found
        """
        converter_class = cls._converters.get(name)
        if not converter_class:
            available = ", ".join(cls.list_converters())
            raise ValueError(
                f"Unknown converter: '{name}'. "
                f"Available converters: {available}"
            )

        return converter_class(model_loader=model_loader)

    @classmethod
    def list_converters(cls) -> List[str]:
        """
        List all registered converter names.

        Returns:
            List of converter names
        """
        return list(cls._converters.keys())
