"""Unknown config keys and unsupported trainer arguments fail loudly.

Covers the shared helpers in shared/training_utils.py and the SFT/KTO/DPO
dataclass config loaders that use them. No GPU, network, model, or TRL import:
the loaders are plain yaml/dataclasses and TRL configs are faked.
"""

from __future__ import annotations

import copy
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
import yaml

from shared.training_utils import (
    MAX_REPORTED_UNKNOWN_KEYS,
    UnknownConfigKeysError,
    UnsupportedTrainerArgumentError,
    accepted_config_parameters,
    apply_wandb_destination,
    dict_to_dataclass,
    find_unknown_config_keys,
    partition_trainer_kwargs,
    reject_unknown_config_keys,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_loader(method: str):
    """Import Trainers/<method>/configs/config_loader.py under a unique name."""
    path = ROOT / "Trainers" / method / "configs" / "config_loader.py"
    spec = importlib.util.spec_from_file_location(f"strict_config_loader_{method}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOADERS = {method: _load_loader(method) for method in ("sft", "kto", "dpo")}


def _write(tmp_path: Path, data: Dict[str, Any]) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Shared key walker
# ---------------------------------------------------------------------------


@dataclass
class _Inner:
    alpha: int = 1
    beta: Optional[int] = None


@dataclass
class _Outer:
    inner: _Inner = field(default_factory=_Inner)
    maybe_inner: Optional[_Inner] = None
    free: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0


class TestFindUnknownConfigKeys:
    def test_dataclass_tree_reports_dotted_paths_with_suggestions(self):
        data = {
            "inner": {"alpah": 2, "beta": 3},
            "maybe_inner": {"gamma": 1},
            "free": {"anything": {"goes": True}},
            "sead": 1,
        }
        found = find_unknown_config_keys(_Outer, data)
        assert found == [
            ("inner.alpah", "alpha"),
            ("maybe_inner.gamma", None),
            ("sead", "seed"),
        ]

    def test_literal_schema_walks_lists_of_sections(self):
        schema = {"items": [{"name": None, "weight": None}], "extra": None}
        data = {"items": [{"name": "a"}, {"name": "b", "wieght": 1}], "extra": {"x": 1}}
        assert find_unknown_config_keys(schema, data) == [("items[1].wieght", "weight")]

    def test_shape_mismatch_is_not_walked(self):
        assert find_unknown_config_keys({"section": {"a": None}}, {"section": 5}) == []
        assert find_unknown_config_keys({"items": [{"a": None}]}, {"items": "x"}) == []

    def test_reject_lists_every_offender_and_caps_the_message(self):
        data = {f"bogus_{index}": index for index in range(MAX_REPORTED_UNKNOWN_KEYS + 5)}
        with pytest.raises(UnknownConfigKeysError) as excinfo:
            reject_unknown_config_keys({"real": None}, data, source="cfg.yaml")
        error = excinfo.value
        assert len(error.paths) == MAX_REPORTED_UNKNOWN_KEYS + 5
        message = str(error)
        assert message.startswith("cfg.yaml:")
        assert "... and 5 more" in message
        assert f"bogus_{MAX_REPORTED_UNKNOWN_KEYS + 4}" not in message

    def test_extra_top_level_keys_are_admitted_only_at_top_level(self):
        schema = {"section": {"a": None}}
        reject_unknown_config_keys(
            schema, {"section": {"a": 1}, "owned": {"x": 1}},
            source="s", extra_top_level_keys=("owned",),
        )
        with pytest.raises(UnknownConfigKeysError) as excinfo:
            reject_unknown_config_keys(
                schema, {"section": {"owned": 1}},
                source="s", extra_top_level_keys=("owned",),
            )
        assert excinfo.value.paths == ["section.owned"]

    def test_dict_to_dataclass_refuses_unknown_keys_and_coerces_numbers(self):
        assert dict_to_dataclass(_Inner, {"alpha": "7"}).alpha == 7
        with pytest.raises(UnknownConfigKeysError) as excinfo:
            dict_to_dataclass(_Inner, {"alpha": 1, "alhpa": 2}, section="inner")
        assert excinfo.value.paths == ["inner.alhpa"]


# ---------------------------------------------------------------------------
# SFT / KTO / DPO loaders
# ---------------------------------------------------------------------------

CHECKED_IN_CONFIGS = [
    ("sft", "Trainers/sft/configs/config.yaml"),
    ("sft", "Trainers/sft/configs/aux_head_example.yaml"),
    ("sft", "Trainers/sft/configs/aux_head_phase_b_example.yaml"),
    ("kto", "Trainers/kto/configs/config.yaml"),
    ("dpo", "Trainers/dpo/configs/config.yaml"),
]
PROTECTED_RECIPE = ROOT / "Trainers/recipes/protected/hf_smollm2_135m_training_smoke.yaml"


@pytest.mark.parametrize("method, relative", CHECKED_IN_CONFIGS)
def test_checked_in_trainer_configs_load_strictly(method, relative):
    LOADERS[method].load_config(str(ROOT / relative))


def test_protected_recipe_loads_with_its_envelope_and_only_with_it():
    sft = LOADERS["sft"]
    config = sft.load_config(
        str(PROTECTED_RECIPE), envelope_keys=sft.PROTECTED_RECIPE_ENVELOPE_KEYS
    )
    assert config.model.model_revision
    with pytest.raises(UnknownConfigKeysError) as excinfo:
        sft.load_config(str(PROTECTED_RECIPE))
    assert sorted(excinfo.value.paths) == sorted(sft.PROTECTED_RECIPE_ENVELOPE_KEYS)


@pytest.mark.parametrize("method", ["sft", "kto", "dpo"])
def test_typos_at_every_level_are_refused_together(method, tmp_path):
    loader = LOADERS[method]
    data = copy.deepcopy(loader.load_yaml_config())
    data["training"]["lerning_rate"] = data["training"].pop("learning_rate")
    data["lora"]["rank"] = 8
    data["wandb"]["entitty"] = None
    data["sede"] = 1
    with pytest.raises(UnknownConfigKeysError) as excinfo:
        loader.load_config(_write(tmp_path, data))
    error = excinfo.value
    assert sorted(error.paths) == sorted(
        ["training.lerning_rate", "lora.rank", "wandb.entitty", "sede"]
    )
    message = str(error)
    assert "training.lerning_rate (did you mean 'learning_rate'?)" in message
    assert "sede (did you mean 'seed'?)" in message


@pytest.mark.parametrize("method", ["kto", "dpo"])
def test_dead_dataset_chat_template_is_refused(method, tmp_path):
    # Nothing in the KTO/DPO trainers read dataset.chat_template; it is gone.
    loader = LOADERS[method]
    data = copy.deepcopy(loader.load_yaml_config())
    data["dataset"]["chat_template"] = "chatml"
    with pytest.raises(UnknownConfigKeysError) as excinfo:
        loader.load_config(_write(tmp_path, data))
    assert excinfo.value.paths == ["dataset.chat_template"]


def test_sft_nested_evolutionary_and_aux_head_typos_are_refused(tmp_path):
    loader = LOADERS["sft"]
    data = copy.deepcopy(loader.load_yaml_config())
    data["evolutionary"]["selection"] = {"method": "best", "min_improvment": 0.1}
    data["evolutionary"]["strategy"] = {"type": "gradient_noise", "params": {"free": 1}}
    data["aux_head"] = {"enabled": False, "lyer": 3}
    with pytest.raises(UnknownConfigKeysError) as excinfo:
        loader.load_config(_write(tmp_path, data))
    assert sorted(excinfo.value.paths) == [
        "aux_head.lyer",
        "evolutionary.selection.min_improvment",
    ]


# ---------------------------------------------------------------------------
# W&B destination wiring (wandb.project / wandb.entity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sft", "kto", "dpo"])
def test_trainers_route_wandb_project_and_entity(method):
    source = (ROOT / "Trainers" / method / f"train_{method}.py").read_text(encoding="utf-8")
    assert "apply_wandb_destination(config.wandb.project, config.wandb.entity)" in source


def test_sft_forwards_group_by_length_to_sft_config():
    source = (ROOT / "Trainers" / "sft" / "train_sft.py").read_text(encoding="utf-8")
    assert '"group_by_length": config.training.group_by_length,' in source


def test_apply_wandb_destination_sets_only_configured_values():
    with patch.dict(os.environ, {"WANDB_ENTITY": "from-env"}, clear=False):
        os.environ.pop("WANDB_PROJECT", None)
        apply_wandb_destination("proj", None)
        assert os.environ["WANDB_PROJECT"] == "proj"
        assert os.environ["WANDB_ENTITY"] == "from-env"
        apply_wandb_destination(None, "team")
        assert os.environ["WANDB_ENTITY"] == "team"


# ---------------------------------------------------------------------------
# Trainer config arguments (fake TRL config classes)
# ---------------------------------------------------------------------------


@dataclass
class FakeGRPOConfig:
    output_dir: str = "out"
    learning_rate: float = 1e-6
    beta: float = 0.04
    temperature: float = 1.0


class FakeUnslothGRPOConfig(FakeGRPOConfig):
    """Mimics Unsloth's patched configs: explicit extras plus **kwargs."""

    def __init__(self, output_dir="out", unsloth_num_chunks=-1, **kwargs):
        self.unsloth_num_chunks = unsloth_num_chunks
        super().__init__(output_dir=output_dir, **kwargs)


class TestTrainerConfigArguments:
    def test_accepted_parameters_cover_forwarded_dataclass_fields(self):
        assert accepted_config_parameters(FakeGRPOConfig) == {
            "output_dir", "learning_rate", "beta", "temperature",
        }
        assert accepted_config_parameters(FakeUnslothGRPOConfig) == {
            "output_dir", "learning_rate", "beta", "temperature", "unsloth_num_chunks",
        }

    def test_user_set_unsupported_argument_raises_with_origin_and_version(self):
        kwargs = {"output_dir": "o", "learning_rate": 1e-5, "importance_sampling_level": "sequence", "top_p": 0.9}
        origins = {
            "learning_rate": "training.learning_rate",
            "importance_sampling_level": "training.use_gspo",
            "top_p": "training.extra_args.top_p",
        }
        with pytest.raises(UnsupportedTrainerArgumentError) as excinfo:
            partition_trainer_kwargs(
                FakeGRPOConfig, kwargs, origins=origins, installed_version="9.9.9",
            )
        error = excinfo.value
        assert error.arguments == ["importance_sampling_level", "top_p"]
        message = str(error)
        assert "installed trl 9.9.9" in message
        assert "importance_sampling_level (set by training.use_gspo)" in message
        assert "top_p (set by training.extra_args.top_p)" in message

    def test_user_set_argument_is_refused_even_if_listed_version_dependent(self):
        with pytest.raises(UnsupportedTrainerArgumentError) as excinfo:
            partition_trainer_kwargs(
                FakeGRPOConfig,
                {"max_prompt_length": 512},
                origins={"max_prompt_length": "training.max_prompt_length"},
                version_dependent_defaults={"max_prompt_length"},
                installed_version="0.28.0",
            )
        assert excinfo.value.arguments == ["max_prompt_length"]

    def test_only_named_version_dependent_internal_defaults_are_omitted(self):
        accepted, omitted = partition_trainer_kwargs(
            FakeGRPOConfig,
            {"output_dir": "o", "beta": 0.1, "max_prompt_length": 4096},
            origins={"beta": "training.beta"},
            version_dependent_defaults={"max_prompt_length"},
            installed_version="0.28.0",
        )
        assert accepted == {"output_dir": "o", "beta": 0.1}
        assert omitted == ["max_prompt_length"]

    def test_unlisted_internal_default_is_refused(self):
        with pytest.raises(UnsupportedTrainerArgumentError, match=r"vllm_mode \(internal default\)"):
            partition_trainer_kwargs(
                FakeGRPOConfig, {"vllm_mode": "colocate"}, origins={},
                installed_version="0.1.0",
            )
