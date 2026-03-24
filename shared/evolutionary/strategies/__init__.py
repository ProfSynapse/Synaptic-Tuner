"""
Gradient modification strategies for evolutionary training.

Each strategy provides a way to generate candidate gradient updates
from the base computed gradient.
"""

from .base import BaseStrategy, GradientCandidate
from .gradient_noise import GradientNoiseStrategy
from .antithetic_noise import AntitheticNoiseStrategy
from .scale_variation import ScaleVariationStrategy
from .combined import CombinedStrategy

__all__ = [
    "BaseStrategy",
    "GradientCandidate",
    "GradientNoiseStrategy",
    "AntitheticNoiseStrategy",
    "ScaleVariationStrategy",
    "CombinedStrategy",
]


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    Factory function to get strategy by name.

    Args:
        name: Strategy name ('gradient_noise', 'antithetic_noise', 'scale_variation', 'combined')
        **kwargs: Strategy-specific parameters

    Returns:
        Configured strategy instance
    """
    strategies = {
        "gradient_noise": GradientNoiseStrategy,
        "antithetic_noise": AntitheticNoiseStrategy,
        "scale_variation": ScaleVariationStrategy,
        "combined": CombinedStrategy,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    # Filter kwargs to only those accepted by the strategy
    strategy_cls = strategies[name]
    import inspect
    sig = inspect.signature(strategy_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return strategy_cls(**filtered_kwargs)
