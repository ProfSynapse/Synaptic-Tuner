"""
Windows Training Configuration
All settings in one place - just edit and run
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration"""

    # YOUR MODEL (edit this)
    model_name: str = "unsloth/mistral-7b-v0.3-bnb-4bit"

    # Other options:
    # model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    # model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    # model_name: str = "unsloth/llama-3.1-13b-bnb-4bit"

    max_seq_length: int = 2048
    dtype: Optional[str] = None  # Auto-detect
    load_in_4bit: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration"""

    r: int = 64  # LoRA rank
    lora_alpha: int = 128  # Scaling factor
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Output
    output_dir: str = "./kto_output_rtx3090"

    # Dataset
    dataset_file: str = "syngen_tools_11.14.25.jsonl"

    # Training
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = train for full epochs
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 10
    max_grad_norm: float = 1.0

    # Optimizer
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01

    # Precision
    fp16: bool = False
    bf16: bool = True  # RTX 30/40 series

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3

    # CRITICAL FOR WINDOWS
    dataset_num_proc: int = 1  # Must be 1
    dataloader_num_workers: int = 0  # Must be 0

    # HuggingFace
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


# Create config instances
model_config = ModelConfig()
lora_config = LoRAConfig()
training_config = TrainingConfig()
