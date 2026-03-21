"""
Evolutionary fine-tuning module.

Implements Gradient + ES Hybrid approach for training:
1. Compute gradient once per step
2. Generate multiple candidate updates
3. Evaluate each candidate's fitness
4. Apply only the best candidate

Usage:
    from shared.evolutionary import (
        EvolutionaryConfig,
        EvolutionaryTrainerWrapper,
        CandidateGenerator,
    )

    # Wrap your trainer
    config = EvolutionaryConfig(
        enabled=True,
        num_candidates=4,
        validation_config_path="configs/fitness/tool_calling.yaml",
    )
    evo_trainer = EvolutionaryTrainerWrapper(trainer, config, tokenizer)
"""

from .config import EvolutionaryConfig
from .candidate_generator import CandidateGenerator
from .trainer_wrapper import EvolutionaryTrainerWrapper

__all__ = [
    "EvolutionaryConfig",
    "CandidateGenerator",
    "EvolutionaryTrainerWrapper",
    "LoRASurgeon",
    "SurgeryConfig",
    "SurgeryResult",
    "OperationResult",
]

# Lazy imports for surgery (requires torch + safetensors)
def __getattr__(name):
    if name in ("LoRASurgeon", "SurgeryConfig", "SurgeryResult", "OperationResult"):
        from .lora_surgery import LoRASurgeon, SurgeryConfig, SurgeryResult, OperationResult
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
