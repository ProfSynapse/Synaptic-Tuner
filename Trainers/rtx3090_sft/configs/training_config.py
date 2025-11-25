"""
Training Configuration for RTX 3090 SFT Fine-tuning
Supervised Fine-Tuning for tool-calling instruction learning
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
    max_seq_length: int = 2048  # Dataset 99th percentile: 1506 tokens
    dtype: Optional[str] = None  # Auto-detection
    load_in_4bit: bool = True  # Essential for memory efficiency


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_gradient_checkpointing: str = "unsloth"  # Unsloth's optimized version
    random_state: int = 3407


@dataclass
class SFTTrainingConfig:
    """SFT training configuration optimized for RTX 3090 24GB."""

    # Output directory
    output_dir: str = "./sft_output_rtx3090"

    # Batch configuration (SFT uses less VRAM, so we can increase batch size)
    per_device_train_batch_size: int = 6  # vs KTO's 4
    gradient_accumulation_steps: int = 4  # vs KTO's 6 (effective batch = 24)

    # Learning rate - MUCH HIGHER than KTO (100x+)
    # SFT uses higher LR because it's direct supervision without KL penalty
    learning_rate: float = 2e-4  # vs KTO's 2e-7
    max_grad_norm: float = 1.0  # vs KTO's 0.5
    lr_scheduler_type: str = "cosine"

    # SFT-specific parameters
    max_seq_length: int = 2048  # Replaces KTO's max_length
    packing: bool = False  # Pack multiple examples per sequence
    completion_only_loss: bool = True  # Train only on assistant completions
    assistant_only_loss: bool = False  # For multi-turn (not needed for single-turn)

    # Memory optimizations
    gradient_checkpointing: bool = True  # Reduces memory usage
    optim: str = "adamw_8bit"  # 8-bit optimizer saves ~2GB VRAM
    fp16: bool = False  # Set dynamically based on GPU
    bf16: bool = True  # RTX 3090 supports BF16 (Ampere)

    # Training schedule - Multi-epoch for better learning
    num_train_epochs: int = 3  # vs KTO's 1 (more epochs to internalize patterns)
    warmup_ratio: float = 0.1  # 10% warmup (vs KTO's 15%)

    # Logging and saving
    logging_steps: int = 5  # Log every 5 steps
    save_steps: int = 50  # Save checkpoint every 50 steps
    save_total_limit: int = 3  # Keep last 3 checkpoints

    # Performance
    dataloader_num_workers: int = 0  # MUST be 0 on WSL2 (multiprocessing hangs)
    dataloader_pin_memory: bool = True
    group_by_length: bool = False  # Can cause hangs with multiprocessing

    # Evaluation (optional)
    eval_strategy: str = "no"  # "steps" or "no"
    eval_steps: int = 50


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    # Dataset source
    dataset_name: str = "professorsynapse/claudesidian-synthetic-dataset"
    dataset_file: str = "syngen_tools_sft_11.25.25.jsonl"  # Tool calling examples only (5,286 examples)

    # Use local file (relative to project root)
    local_file: Optional[str] = "../../Datasets/syngen_tools_sft_11.25.25.jsonl"

    # Dataset processing
    num_proc: int = 1  # Set to 1 on Windows to avoid multiprocessing issues
    test_size: float = 0.1  # For train/validation split (optional)
    split_dataset: bool = False  # Enable to create train/val split

    # SFT-specific: filter for desirable examples only
    filter_desirable: bool = False  # False because SFT dataset already filtered


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: SFTTrainingConfig = field(default_factory=SFTTrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Experiment tracking (optional)
    use_wandb: bool = False
    wandb_project: str = "sft-finetuning"
    wandb_run_name: Optional[str] = None

    # Random seed
    seed: int = 42


# Preset configurations for different model sizes
def get_3b_config() -> Config:
    """Configuration optimized for 3B models (fast iteration) - USES ~12GB VRAM."""
    config = Config()
    config.model.model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    config.model.max_seq_length = 2048
    config.lora.r = 32
    config.lora.lora_alpha = 64
    config.training.per_device_train_batch_size = 12  # Increased (SFT uses less VRAM)
    config.training.gradient_accumulation_steps = 2   # Effective batch = 24
    config.training.max_seq_length = 2048
    config.training.gradient_checkpointing = False  # Not needed for 3B
    return config


def get_7b_config() -> Config:
    """Configuration optimized for 7B models (production quality) - USES ~9GB VRAM."""
    config = Config()
    config.model.model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
    config.model.max_seq_length = 2048
    config.lora.r = 64
    config.lora.lora_alpha = 128
    config.training.per_device_train_batch_size = 6  # Increased from KTO's 4
    config.training.gradient_accumulation_steps = 4  # Effective batch = 24
    config.training.max_seq_length = 2048
    config.training.gradient_checkpointing = True
    return config


def get_13b_config() -> Config:
    """Configuration optimized for 13B models (advanced) - USES ~14GB VRAM."""
    config = Config()
    config.model.model_name = "unsloth/llama-2-13b-bnb-4bit"
    config.model.max_seq_length = 2048
    config.lora.r = 64
    config.lora.lora_alpha = 128
    config.training.per_device_train_batch_size = 4  # Same as KTO
    config.training.gradient_accumulation_steps = 6  # Effective batch = 24
    config.training.max_seq_length = 2048
    config.training.gradient_checkpointing = True  # Required for 13B
    return config


def get_20b_config() -> Config:
    """Configuration optimized for GPT-OSS 20B models."""
    config = Config()
    config.model.model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    config.model.max_seq_length = 2048
    config.lora.r = 128
    config.lora.lora_alpha = 256
    config.training.per_device_train_batch_size = 4
    config.training.gradient_accumulation_steps = 6
    config.training.learning_rate = 1e-4  # Slightly lower for larger model
    config.training.max_seq_length = 2048
    config.training.gradient_checkpointing = True
    config.training.warmup_ratio = 0.10
    return config
