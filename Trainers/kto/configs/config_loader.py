"""
YAML Configuration Loader for KTO Training
Loads config.yaml and converts to Python dataclass objects
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from shared.training_utils import dict_to_dataclass, reject_unknown_config_keys


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
    # Optional dot-path into each raw row (e.g. "metadata.scenario"). When set
    # and a validation split is created, rows sharing a group value stay on the
    # same side and test_size applies over groups. None ⇒ random row split.
    validation_group_key: Optional[str] = None


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


def load_config(config_path: str = None) -> Config:
    """
    Load YAML config and convert to Config dataclass.

    Every key, at every nesting level, must be declared by the Config dataclass
    tree; anything else raises UnknownConfigKeysError listing each offending
    dotted path (with a "did you mean" suggestion).

    Args:
        config_path: Path to config.yaml

    Returns:
        Config object with all settings
    """
    yaml_config = load_yaml_config(config_path)
    reject_unknown_config_keys(
        Config,
        yaml_config,
        source=str(config_path or Path(__file__).parent / "config.yaml"),
    )

    # Convert each section to dataclass
    model_config = dict_to_dataclass(ModelConfig, yaml_config['model'], section='model')
    lora_config = dict_to_dataclass(LoRAConfig, yaml_config['lora'], section='lora')
    training_config = dict_to_dataclass(KTOTrainingConfig, yaml_config['training'], section='training')
    dataset_config = dict_to_dataclass(DatasetConfig, yaml_config['dataset'], section='dataset')
    wandb_config = dict_to_dataclass(WandbConfig, yaml_config.get('wandb', {}), section='wandb')

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
