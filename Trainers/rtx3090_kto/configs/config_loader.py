"""
YAML Configuration Loader for KTO Training
Loads config.yaml and converts to Python dataclass objects
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


def load_yaml_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to config.yaml (defaults to configs/config.yaml)

    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        # Default to config.yaml in same directory as this file
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_name: str
    max_seq_length: int
    dtype: Optional[str]
    load_in_4bit: bool


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: List[str]
    use_gradient_checkpointing: str
    random_state: int
    use_rslora: bool = False
    use_dora: bool = False


@dataclass
class KTOTrainingConfig:
    """KTO training configuration."""
    output_dir: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    beta: float
    desirable_weight: float
    undesirable_weight: float
    learning_rate: float
    max_grad_norm: float
    lr_scheduler_type: str
    use_kto_s: bool
    use_two_stage_lr: bool
    lr_reduction_step: int
    lr_reduction_factor: float
    max_length: int
    max_prompt_length: int
    gradient_checkpointing: bool
    optim: str
    fp16: bool
    bf16: bool
    num_train_epochs: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    save_total_limit: int
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    group_by_length: bool
    eval_strategy: str
    eval_steps: int


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    dataset_name: str
    dataset_file: str
    local_file: Optional[str]
    num_proc: int
    test_size: float
    chat_template: str


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool
    project: str
    run_name: Optional[str]
    entity: Optional[str]


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    model: ModelConfig
    lora: LoRAConfig
    training: KTOTrainingConfig
    dataset: DatasetConfig
    wandb: WandbConfig
    seed: int = 42

    @property
    def use_wandb(self) -> bool:
        """Backwards compatibility: access wandb.enabled as use_wandb."""
        return self.wandb.enabled

    @use_wandb.setter
    def use_wandb(self, value: bool):
        """Backwards compatibility: set wandb.enabled via use_wandb."""
        self.wandb.enabled = value

    @property
    def wandb_project(self) -> Optional[str]:
        """Backwards compatibility: access wandb.project as wandb_project."""
        return self.wandb.project

    @wandb_project.setter
    def wandb_project(self, value: Optional[str]):
        """Backwards compatibility: set wandb.project via wandb_project."""
        self.wandb.project = value

    @property
    def wandb_run_name(self) -> Optional[str]:
        """Backwards compatibility: access wandb.run_name as wandb_run_name."""
        return self.wandb.run_name

    @wandb_run_name.setter
    def wandb_run_name(self, value: Optional[str]):
        """Backwards compatibility: set wandb.run_name via wandb_run_name."""
        self.wandb.run_name = value


def dict_to_dataclass(cls, data: Dict[str, Any]):
    """
    Convert dictionary to dataclass instance.
    Handles type conversion for numeric fields that might be strings.
    """
    import typing

    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    converted_data = {}

    for k, v in data.items():
        if k not in fieldtypes:
            continue

        field_type = fieldtypes[k]

        # Handle Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is typing.Union:
            # Get the non-None type from Optional
            types = [t for t in field_type.__args__ if t is not type(None)]
            if types:
                field_type = types[0]

        # Convert strings to appropriate numeric types
        if field_type == float and isinstance(v, str):
            converted_data[k] = float(v)
        elif field_type == int and isinstance(v, str):
            converted_data[k] = int(v)
        else:
            converted_data[k] = v

    return cls(**converted_data)


def load_config(config_path: str = None) -> Config:
    """
    Load YAML config and convert to Config dataclass.

    Args:
        config_path: Path to config.yaml

    Returns:
        Config object with all settings
    """
    yaml_config = load_yaml_config(config_path)

    # Convert each section to dataclass
    model_config = dict_to_dataclass(ModelConfig, yaml_config['model'])
    lora_config = dict_to_dataclass(LoRAConfig, yaml_config['lora'])
    training_config = dict_to_dataclass(KTOTrainingConfig, yaml_config['training'])
    dataset_config = dict_to_dataclass(DatasetConfig, yaml_config['dataset'])
    wandb_config = dict_to_dataclass(WandbConfig, yaml_config.get('wandb', {}))

    return Config(
        model=model_config,
        lora=lora_config,
        training=training_config,
        dataset=dataset_config,
        wandb=wandb_config,
        seed=yaml_config.get('seed', 42)
    )


# Backwards compatibility: Provide same function names as old Python config
def get_7b_config(config_path: str = None) -> Config:
    """Load default 7B config from YAML."""
    return load_config(config_path)


def get_3b_config(config_path: str = None) -> Config:
    """Load config and override for 3B model."""
    config = load_config(config_path)
    config.training.per_device_train_batch_size = 8
    return config


def get_13b_config(config_path: str = None) -> Config:
    """Load config and override for 13B model."""
    config = load_config(config_path)
    config.training.per_device_train_batch_size = 2
    return config


def get_20b_config(config_path: str = None) -> Config:
    """Load config and override for 20B model."""
    config = load_config(config_path)
    config.training.per_device_train_batch_size = 4
    return config
