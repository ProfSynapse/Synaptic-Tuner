"""
Training Configuration for RTX 3090 KTO Fine-tuning
Based on rtx3090-kto-finetuning.md specification
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    # Model selection (choose one)
    # Tier 1 (3B models): Fast iteration
    # model_name: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    # model_name: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

    # Tier 2 (7B models): Production quality (RECOMMENDED)
    model_name: str = "unsloth/mistral-7b-v0.3-bnb-4bit"
    # model_name: str = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
    # model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

    # Tier 3 (13B+ models): Advanced
    # model_name: str = "unsloth/llama-2-13b-bnb-4bit"

    # Model parameters
    max_seq_length: int = 2048  # 1024-4096 depending on VRAM
    dtype: Optional[str] = None  # Auto-detection
    load_in_4bit: bool = True  # Essential for memory efficiency


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    # LoRA parameters - adjust based on model size
    # For Mistral-7B: r=64, alpha=128 (recommended)
    # For GPT-OSS-20B: r=128, alpha=256
    # For 3B models: r=32, alpha=64

    r: int = 64  # LoRA rank (Mistral-7B default)
    lora_alpha: int = 128  # LoRA scaling factor (Mistral-7B default)
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_gradient_checkpointing: str = "unsloth"  # Unsloth's optimized version
    random_state: int = 3407


@dataclass
class KTOTrainingConfig:
    """KTO training configuration optimized for RTX 3090 24GB."""

    # Output directory
    output_dir: str = "./kto_output_rtx3090"

    # Batch size configuration - OPTIMIZED FOR 24GB VRAM
    # For 3B models: batch_size=16, accumulation=2 (effective=32)
    # For 7B models: batch_size=8, accumulation=4 (effective=32) - USES ~20GB VRAM
    # For 13B models: batch_size=4, accumulation=8 (effective=32)
    per_device_train_batch_size: int = 8  # 2x increase for 7B models (uses ~20GB)
    gradient_accumulation_steps: int = 4  # Reduced since batch is larger

    # KTO-specific parameters
    beta: float = 0.1  # KTO beta parameter (0.01-0.5, default 0.1)
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0

    # Learning rate
    learning_rate: float = 5e-7  # Conservative for KTO (5e-7 to 1e-6)
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    # Sequence lengths
    max_length: int = 2048  # Must match ModelConfig.max_seq_length
    max_prompt_length: int = 1024  # Should be ≤ max_length / 2

    # Memory optimizations
    gradient_checkpointing: bool = False  # Not needed with 24GB for 7B
    optim: str = "adamw_8bit"  # 8-bit optimizer saves ~2GB VRAM
    fp16: bool = False  # Set dynamically based on GPU
    bf16: bool = True  # RTX 3090 supports BF16 (Ampere)

    # Training schedule
    num_train_epochs: int = 1  # Adjust based on dataset size
    warmup_ratio: float = 0.1  # 10% warmup

    # Logging and saving
    logging_steps: int = 5  # Log every 5 steps for table display
    save_steps: int = 50  # Save checkpoint every 50 steps
    save_total_limit: int = 3  # Keep last 3 checkpoints

    # Performance
    dataloader_num_workers: int = 4  # 2-4 workers (set to 0 on Windows)
    dataloader_pin_memory: bool = True
    group_by_length: bool = False

    # Evaluation (optional)
    eval_strategy: str = "no"  # "steps" or "no"
    eval_steps: int = 100


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    # Dataset source
    dataset_name: str = "professorsynapse/claudesidian-synthetic-dataset"
    dataset_file: str = "syngen_tools_11.14.25.jsonl"

    # Or use local file
    # local_file: Optional[str] = "./data/train.jsonl"

    # Dataset processing
    num_proc: int = 1  # Set to 1 on Windows to avoid multiprocessing issues
    test_size: float = 0.1  # For train/validation split (optional)

    # Format settings
    chat_template: str = "chatml"  # or "mistral", "llama", etc.


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: KTOTrainingConfig = field(default_factory=KTOTrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Experiment tracking (optional)
    use_wandb: bool = False
    wandb_project: str = "kto-finetuning"
    wandb_run_name: Optional[str] = None

    # Random seed
    seed: int = 42


# Preset configurations for different model sizes
def get_3b_config() -> Config:
    """Configuration optimized for 3B models (fast iteration) - USES ~16GB VRAM."""
    config = Config()
    config.model.model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    config.model.max_seq_length = 2048
    config.lora.r = 32
    config.lora.lora_alpha = 64
    config.training.per_device_train_batch_size = 16  # Increased for 24GB VRAM
    config.training.gradient_accumulation_steps = 2   # Reduced (still effective=32)
    config.training.max_length = 2048
    config.training.max_prompt_length = 1024
    config.training.gradient_checkpointing = False
    return config


def get_7b_config() -> Config:
    """Configuration optimized for 7B models (production quality) - USES ~20GB VRAM."""
    config = Config()
    config.model.model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
    config.model.max_seq_length = 2048
    config.lora.r = 64
    config.lora.lora_alpha = 128
    config.training.per_device_train_batch_size = 8  # 2x increase for 24GB VRAM
    config.training.gradient_accumulation_steps = 4  # Reduced (still effective=32)
    config.training.max_length = 2048
    config.training.max_prompt_length = 1024
    config.training.gradient_checkpointing = False
    return config


def get_13b_config() -> Config:
    """Configuration optimized for 13B models (advanced) - USES ~22GB VRAM."""
    config = Config()
    config.model.model_name = "unsloth/llama-2-13b-bnb-4bit"
    config.model.max_seq_length = 2048
    config.lora.r = 64
    config.lora.lora_alpha = 128
    config.training.per_device_train_batch_size = 4  # Increased for 24GB VRAM
    config.training.gradient_accumulation_steps = 8  # Reduced (still effective=32)
    config.training.max_length = 2048
    config.training.max_prompt_length = 1024
    config.training.gradient_checkpointing = True  # Still needed for 13B
    return config


def get_20b_config() -> Config:
    """Configuration optimized for GPT-OSS 20B models."""
    config = Config()
    config.model.model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    config.model.max_seq_length = 4096
    config.lora.r = 128
    config.lora.lora_alpha = 256
    config.training.per_device_train_batch_size = 4
    config.training.gradient_accumulation_steps = 8
    config.training.beta = 0.05  # Lower beta for larger models
    config.training.learning_rate = 5e-7
    config.training.max_length = 4096
    config.training.max_prompt_length = 2048
    config.training.gradient_checkpointing = True
    config.training.warmup_ratio = 0.10
    return config
