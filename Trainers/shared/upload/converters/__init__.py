"""
Format converters for model deployment.

Each converter handles converting models to a specific deployment format.
"""

from .base import BaseConverter
from .gguf import GGUFConverter  # Legacy converter (kept for reference)
from .gguf_reliable import ReliableGGUFConverter  # New reliable converter
from .webgpu import WebGPUConverter  # WebGPU/MLC-LLM converter for browsers
from .registry import ConverterRegistry

# Register built-in converters
# Use ReliableGGUFConverter as default - faster, handles VL models better
ConverterRegistry.register("gguf", ReliableGGUFConverter)
ConverterRegistry.register("gguf_legacy", GGUFConverter)  # Keep old one available
ConverterRegistry.register("webgpu", WebGPUConverter)  # Browser deployment via WebLLM

__all__ = [
    "BaseConverter",
    "GGUFConverter",
    "ReliableGGUFConverter",
    "WebGPUConverter",
    "ConverterRegistry",
]
