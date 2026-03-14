"""
Adaptive Memory Management for KTO Training

Automatically adjusts batch sizes and memory settings based on available VRAM
to prevent OOM errors and optimize GPU utilization.
"""

import torch
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class AdaptiveMemoryManager:
    """Manages memory adaptively during training."""

    def __init__(self, target_utilization: float = 0.85):
        """
        Initialize adaptive memory manager.

        Args:
            target_utilization: Target VRAM utilization (0.0-1.0). Default 0.85 (85%)
        """
        self.target_utilization = target_utilization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information in GB."""
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "reserved": 0, "free": 0}

        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = total - reserved

        return {
            "total": total,
            "allocated": allocated,
            "reserved": reserved,
            "free": free
        }

    def calculate_optimal_batch_size(
        self,
        base_batch_size: int,
        model_size: str = "7b"
    ) -> Tuple[int, int]:
        """
        Calculate optimal batch size based on available VRAM.

        Args:
            base_batch_size: Base batch size from config
            model_size: Model size ("3b", "7b", "13b", "20b")

        Returns:
            Tuple of (batch_size, gradient_accumulation_steps)
        """
        mem_info = self.get_gpu_memory_info()
        total_vram = mem_info["total"]

        # Estimated VRAM usage per batch (GB) - based on empirical testing
        vram_per_batch = {
            "3b": 0.8,   # ~0.8GB per batch
            "7b": 2.0,   # ~2.0GB per batch
            "13b": 3.5,  # ~3.5GB per batch
            "20b": 5.0   # ~5.0GB per batch
        }

        # Base memory overhead (model + optimizer state)
        base_overhead = {
            "3b": 6.0,   # ~6GB base
            "7b": 8.0,   # ~8GB base
            "13b": 12.0, # ~12GB base
            "20b": 16.0  # ~16GB base
        }

        size_key = model_size.lower()
        if size_key not in vram_per_batch:
            logger.warning(f"Unknown model size {model_size}, using 7b defaults")
            size_key = "7b"

        # Calculate how much VRAM we can use for batches
        target_vram = total_vram * self.target_utilization
        available_for_batches = target_vram - base_overhead[size_key]

        # Calculate optimal batch size
        optimal_batch_size = max(1, int(available_for_batches / vram_per_batch[size_key]))

        # Clamp to reasonable bounds
        optimal_batch_size = min(optimal_batch_size, 16)  # Max batch size
        optimal_batch_size = max(optimal_batch_size, 1)   # Min batch size

        # Calculate gradient accumulation to maintain effective batch size of 32
        target_effective_batch = 32
        gradient_accumulation = max(1, target_effective_batch // optimal_batch_size)

        logger.info(f"Adaptive Memory Manager:")
        logger.info(f"  Total VRAM: {total_vram:.1f}GB")
        logger.info(f"  Target utilization: {self.target_utilization*100:.0f}%")
        logger.info(f"  Recommended batch size: {optimal_batch_size}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation}")
        logger.info(f"  Effective batch size: {optimal_batch_size * gradient_accumulation}")

        return optimal_batch_size, gradient_accumulation

    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage and return statistics."""
        mem_info = self.get_gpu_memory_info()

        utilization = (mem_info["reserved"] / mem_info["total"]) * 100 if mem_info["total"] > 0 else 0

        return {
            "total_gb": mem_info["total"],
            "reserved_gb": mem_info["reserved"],
            "allocated_gb": mem_info["allocated"],
            "free_gb": mem_info["free"],
            "utilization_percent": utilization
        }

    def suggest_batch_size_adjustment(self, current_batch_size: int) -> Tuple[int, str]:
        """
        Suggest batch size adjustment based on current memory usage.

        Returns:
            Tuple of (suggested_batch_size, reason)
        """
        stats = self.check_memory_usage()
        utilization = stats["utilization_percent"]

        if utilization > 95:
            return max(1, current_batch_size // 2), "Very high memory usage (>95%)"
        elif utilization > 90:
            return max(1, int(current_batch_size * 0.75)), "High memory usage (>90%)"
        elif utilization < 60:
            return current_batch_size + 1, "Low memory usage (<60%)"
        else:
            return current_batch_size, "Memory usage optimal"

    def optimize_memory_settings(
        self,
        model_size: str = "7b",
        current_batch_size: int = 4,
        enable_gradient_checkpointing: bool = False
    ) -> Dict[str, any]:
        """
        Provide complete memory optimization recommendations.

        Args:
            model_size: Model size ("3b", "7b", "13b", "20b")
            current_batch_size: Current batch size
            enable_gradient_checkpointing: Whether gradient checkpointing is enabled

        Returns:
            Dictionary with optimization recommendations
        """
        mem_info = self.get_gpu_memory_info()
        optimal_batch, optimal_accum = self.calculate_optimal_batch_size(
            current_batch_size, model_size
        )

        recommendations = {
            "batch_size": optimal_batch,
            "gradient_accumulation": optimal_accum,
            "gradient_checkpointing": enable_gradient_checkpointing,
            "memory_info": mem_info
        }

        # If VRAM is very limited, suggest gradient checkpointing
        if mem_info["total"] < 16:
            recommendations["gradient_checkpointing"] = True
            recommendations["reason"] = "Limited VRAM detected (<16GB)"
        elif model_size in ["13b", "20b"] and not enable_gradient_checkpointing:
            recommendations["gradient_checkpointing"] = True
            recommendations["reason"] = "Large model detected, gradient checkpointing recommended"

        return recommendations


def get_adaptive_settings(
    model_size: str = "7b",
    target_utilization: float = 0.85
) -> Dict[str, any]:
    """
    Convenience function to get adaptive memory settings.

    Args:
        model_size: Model size ("3b", "7b", "13b", "20b")
        target_utilization: Target VRAM utilization (0.0-1.0)

    Returns:
        Dictionary with recommended settings
    """
    manager = AdaptiveMemoryManager(target_utilization=target_utilization)

    # Start with conservative base batch size
    base_batch_size = 2 if model_size in ["13b", "20b"] else 4

    return manager.optimize_memory_settings(
        model_size=model_size,
        current_batch_size=base_batch_size
    )


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    print("\nAdaptive Memory Management Demo")
    print("=" * 50)

    manager = AdaptiveMemoryManager(target_utilization=0.85)

    # Check current memory
    mem_stats = manager.check_memory_usage()
    print(f"\nCurrent GPU Memory:")
    print(f"  Total: {mem_stats['total_gb']:.1f} GB")
    print(f"  Reserved: {mem_stats['reserved_gb']:.1f} GB")
    print(f"  Free: {mem_stats['free_gb']:.1f} GB")
    print(f"  Utilization: {mem_stats['utilization_percent']:.1f}%")

    # Get recommendations for different model sizes
    for model_size in ["3b", "7b", "13b"]:
        print(f"\nRecommendations for {model_size} model:")
        print("-" * 50)
        settings = get_adaptive_settings(model_size=model_size)
        print(f"  Batch size: {settings['batch_size']}")
        print(f"  Gradient accumulation: {settings['gradient_accumulation']}")
        print(f"  Effective batch size: {settings['batch_size'] * settings['gradient_accumulation']}")
        print(f"  Gradient checkpointing: {settings['gradient_checkpointing']}")
