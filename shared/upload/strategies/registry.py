"""
Strategy registry for save strategies.

Provides a central registry for looking up save strategies by name,
enabling dynamic strategy selection and extensibility.
"""

from typing import Dict, List, Optional, Type

from .base import BaseSaveStrategy
from ..core.interfaces import IModelLoader


class SaveStrategyRegistry:
    """
    Registry for save strategies.

    Allows registration of new strategies and lookup by name.
    Follows the Registry pattern for plugin-like extensibility.
    """

    _strategies: Dict[str, Type[BaseSaveStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseSaveStrategy]) -> None:
        """
        Register a new save strategy.

        Args:
            name: Unique identifier for the strategy
            strategy_class: Strategy class to register
        """
        cls._strategies[name] = strategy_class

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a save strategy.

        Args:
            name: Name of strategy to unregister

        Returns:
            True if strategy was removed, False if not found
        """
        if name in cls._strategies:
            del cls._strategies[name]
            return True
        return False

    @classmethod
    def get(
        cls,
        name: str,
        model_loader: Optional[IModelLoader] = None
    ) -> BaseSaveStrategy:
        """
        Get a strategy instance by name.

        Args:
            name: Name of the strategy
            model_loader: Model loader to inject into the strategy

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy name is not found
        """
        strategy_class = cls._strategies.get(name)
        if not strategy_class:
            available = ", ".join(cls.list_strategies())
            raise ValueError(
                f"Unknown save strategy: '{name}'. "
                f"Available strategies: {available}"
            )

        return strategy_class(model_loader=model_loader)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all registered strategy names.

        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls, name: str) -> Dict[str, any]:
        """
        Get information about a strategy.

        Args:
            name: Name of the strategy

        Returns:
            Dictionary with strategy information
        """
        strategy_class = cls._strategies.get(name)
        if not strategy_class:
            return None

        # Create temporary instance for info
        temp = strategy_class(model_loader=None)
        return {
            "name": temp.name,
            "requires_gpu": temp.requires_gpu(),
            "class": strategy_class.__name__,
        }

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered strategies.

        Primarily for testing purposes.
        """
        cls._strategies.clear()
