"""
Windows Training Configuration
Simplified config for Unsloth on Windows
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class WindowsTrainingConfig:
    """Training configuration for Windows"""

    # Model settings
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list = None

    # Training settings
    output_dir: str = "./unsloth_output"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 10
    max_steps: int = 100

    # Windows-specific settings (CRITICAL)
    dataset_num_proc: int = 1  # Must be 1 on Windows
    dataloader_num_workers: int = 0  # Must be 0 on Windows

    # Optimization
    fp16: bool = False
    bf16: bool = True  # RTX 30/40 series support bf16
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Logging
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: Optional[int] = None

    # Dataset
    dataset_file: str = "syngen_tools_11.14.25.jsonl"

    # HuggingFace settings
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]


# Pre-configured sizes
def get_small_config():
    """Fast test configuration"""
    return WindowsTrainingConfig(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=512,
        num_train_epochs=1,
        max_steps=50,
        per_device_train_batch_size=1,
    )


def get_7b_config():
    """7B model configuration"""
    return WindowsTrainingConfig(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
    )


def get_13b_config():
    """13B model configuration (requires more VRAM)"""
    return WindowsTrainingConfig(
        model_name="unsloth/llama-3.1-13b-bnb-4bit",
        max_seq_length=1024,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
    )
