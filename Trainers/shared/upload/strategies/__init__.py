"""
Save strategies for different model formats.

Each strategy encapsulates the logic for saving a model in a specific format,
following the Strategy pattern for extensibility.
"""

from .base import BaseSaveStrategy
from .lora import LoRASaveStrategy
from .merged_16bit import Merged16BitStrategy
from .merged_4bit import Merged4BitStrategy
from .registry import SaveStrategyRegistry

# Register built-in strategies
SaveStrategyRegistry.register("lora", LoRASaveStrategy)
SaveStrategyRegistry.register("merged_16bit", Merged16BitStrategy)
SaveStrategyRegistry.register("merged_4bit", Merged4BitStrategy)

__all__ = [
    "BaseSaveStrategy",
    "LoRASaveStrategy",
    "Merged16BitStrategy",
    "Merged4BitStrategy",
    "SaveStrategyRegistry",
]
