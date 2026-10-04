"""
DecisionModelConfig: the resolved, self-contained description of a decision model.

Location: Trainers/decision/decision_core/model_config.py
Used by:  modeling.py (build/save/load), train_decision.py (resolution from the
          run config + model registry; torch-free so --dry-run can resolve it).

Saved as ``final_model/decision_config.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import READOUTS
from .prompting import HEADERS, PROMPT_FORMAT

CONFIG_FILE = "decision_config.json"


@dataclass
class DecisionModelConfig:
    """Everything needed to rebuild a trained model; saved as decision_config.json.

    Built by train_decision.py from the run config plus the resolved model
    registry entry, so a checkpoint loads without the registry or run config.
    """

    hf_id: str
    lora_targets: list[str]
    registry_name: str | None = None
    revision: str | None = None
    loader: str = "causal_lm"            # causal_lm | text_tower (see model_registry.yaml)
    causal_lm_class: str | None = None   # required for text_tower
    readout: str = "pointer"
    marker_style: str = "numbers"
    headers: dict[str, str] = field(default_factory=lambda: dict(HEADERS))
    pointer_dim: int = 256
    head_dropout: float = 0.05
    max_length: int = 4096
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Filled by the calibrate stage. 1.0 is a no-op.
    temperature_by_kind: dict[str, float] = field(default_factory=dict)
    ordinal_smoothing: float = 0.0
    prompt_format: str = PROMPT_FORMAT

    def __post_init__(self) -> None:
        if self.readout not in READOUTS:
            raise ValueError(f"unknown readout {self.readout!r}; expected one of {READOUTS}")
        if self.loader not in ("causal_lm", "text_tower"):
            raise ValueError(f"unknown loader {self.loader!r}")
        if self.loader == "text_tower" and not self.causal_lm_class:
            raise ValueError("loader text_tower needs causal_lm_class")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
