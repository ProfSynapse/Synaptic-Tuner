"""GRPO config key schemas: strict loading, drift detection, and GRPOConfig args.

The GRPO entrypoints import torch/unsloth/trl at module load, so the schema
literals and ``_build_grpo_config`` are read from source with ``ast`` instead of
importing the trainers. No GPU, network, model, or TRL is needed.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import pytest
import yaml

from shared.training_utils import (
    UnknownConfigKeysError,
    UnsupportedTrainerArgumentError,
    build_trainer_config,
    reject_unknown_config_keys,
)

ROOT = Path(__file__).resolve().parents[3]
GRPO = ROOT / "Trainers" / "grpo"
TRAIN_GRPO = GRPO / "train_grpo.py"
TRAIN_ENV_GRPO = GRPO / "train_env_grpo.py"

# Keys the loaders inject themselves after validation (never user config).
INTERNAL_KEYS = {"_config_path"}


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal(path: Path, name: str) -> Any:
    for node in _module_tree(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
            ):
                return frozenset(ast.literal_eval(value.args[0]))
            return ast.literal_eval(value)
    raise AssertionError(f"{name} not found in {path}")


GRPO_SCHEMA = _literal(TRAIN_GRPO, "GRPO_CONFIG_SCHEMA")
ENV_GRPO_SCHEMA = _literal(TRAIN_ENV_GRPO, "ENV_GRPO_CONFIG_SCHEMA")


def _load_yaml(name: str) -> Dict[str, Any]:
    return yaml.safe_load((GRPO / "configs" / name).read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# Checked-in configs and typo refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema_name, config_name",
    [
        ("grpo", "config.yaml"),
        ("grpo", "pivot_config.yaml"),
        ("env", "env_config.yaml"),
    ],
)
def test_checked_in_grpo_configs_satisfy_their_schema(schema_name, config_name):
    schema = GRPO_SCHEMA if schema_name == "grpo" else ENV_GRPO_SCHEMA
    reject_unknown_config_keys(schema, _load_yaml(config_name), source=config_name)


def test_grpo_typos_are_refused_with_paths_and_suggestions():
    config = copy.deepcopy(_load_yaml("pivot_config.yaml"))
    config["training"]["num_generatons"] = 8
    config["pivot"]["filtering"]["min_candidate"] = 10
    config["rewards"]["items"][0]["wieght"] = 2.0
    config["schema"] = {"tool_schema_path": "x"}
    with pytest.raises(UnknownConfigKeysError) as excinfo:
        reject_unknown_config_keys(GRPO_SCHEMA, config, source="pivot_config.yaml")
    error = excinfo.value
    assert sorted(error.paths) == sorted(
        [
            "training.num_generatons",
            "pivot.filtering.min_candidate",
            "rewards.items[0].wieght",
            "schema",
        ]
    )
    message = str(error)
    assert "training.num_generatons (did you mean 'num_generations'?)" in message
    assert "rewards.items[0].wieght (did you mean 'weight'?)" in message


def test_env_grpo_typos_are_refused():
    config = copy.deepcopy(_load_yaml("env_config.yaml"))
    config["env_training"]["max_turn"] = 3
    config["env_training"]["runtime"]["python_package"] = []
    config["rewards"]["sucess_reward"] = 1.0
    with pytest.raises(UnknownConfigKeysError) as excinfo:
        reject_unknown_config_keys(ENV_GRPO_SCHEMA, config, source="env_config.yaml")
    assert sorted(excinfo.value.paths) == sorted(
        [
            "env_training.max_turn",
            "env_training.runtime.python_package",
            "rewards.sucess_reward",
        ]
    )


def test_loaders_validate_before_returning():
    for path, schema_name in ((TRAIN_GRPO, "GRPO_CONFIG_SCHEMA"), (TRAIN_ENV_GRPO, "ENV_GRPO_CONFIG_SCHEMA")):
        load_config = next(
            node for node in _module_tree(path).body
            if isinstance(node, ast.FunctionDef) and node.name == "load_config"
        )
        source = ast.unparse(load_config)
        assert f"reject_unknown_config_keys({schema_name}" in source, path


# ---------------------------------------------------------------------------
# Drift: declared keys == keys the code reads
# ---------------------------------------------------------------------------

# (file, function or None for the whole module) -> {variable: schema path}.
# A schema path is dotted; ``[]`` steps into a list-of-sections item.
GRPO_READERS = {
    (TRAIN_GRPO, None): {
        "config": "",
        "training": "training",
        "training_cfg": "training",
        "model_cfg": "model",
        "dataset_cfg": "dataset",
        "lora_cfg": "lora",
        "wandb_cfg": "wandb",
        "rewards_cfg": "rewards",
        "pivot_cfg": "pivot",
        "profiling": "pivot.profiling",
        "filtering": "pivot.filtering",
        "cache": "pivot.cache",
    },
    (GRPO / "src" / "rewards.py", "build_combined_reward_function"): {
        "rewards_config": "rewards",
        "item": "rewards.items[]",
        "custom": "rewards.custom",
        "fn_cfg": "rewards.custom.functions[]",
    },
}
ENV_GRPO_READERS = {
    (TRAIN_ENV_GRPO, "run"): {
        "config": "",
        "model_cfg": "model",
        "dataset_cfg": "dataset",
        "training_cfg": "training",
        "lora_cfg": "lora",
        "env_cfg": "env_training",
        "runtime_cfg": "env_training.runtime",
    },
    (GRPO / "src" / "env_rollout.py", None): {"env_training_cfg": "env_training"},
    (GRPO / "src" / "env_runtime.py", None): {
        "config": "",
        "runtime_cfg": "env_training.runtime",
    },
    (GRPO / "src" / "env_dataset.py", None): {
        "prompt_augmentation": "env_training.prompt_augmentation",
    },
    (GRPO / "src" / "env_rewards.py", None): {"reward_cfg": "rewards"},
}


def _string_keys_read(node: ast.AST, variables: Iterable[str]) -> Set[Tuple[str, str]]:
    """``(variable, key)`` for ``v.get("k")``, ``v["k"]`` and ``"k" in v``.

    Stores (``v["k"] = ...``, e.g. CLI overrides) are not reads.
    """
    names = set(variables)
    reads: Set[Tuple[str, str]] = set()
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "get"
            and isinstance(item.func.value, ast.Name)
            and item.func.value.id in names
            and item.args
            and isinstance(item.args[0], ast.Constant)
            and isinstance(item.args[0].value, str)
        ):
            reads.add((item.func.value.id, item.args[0].value))
        elif (
            isinstance(item, ast.Subscript)
            and isinstance(item.ctx, ast.Load)
            and isinstance(item.value, ast.Name)
            and item.value.id in names
            and isinstance(item.slice, ast.Constant)
            and isinstance(item.slice.value, str)
        ):
            reads.add((item.value.id, item.slice.value))
        elif (
            isinstance(item, ast.Compare)
            and len(item.ops) == 1
            and isinstance(item.ops[0], ast.In)
            and isinstance(item.left, ast.Constant)
            and isinstance(item.left.value, str)
            and isinstance(item.comparators[0], ast.Name)
            and item.comparators[0].id in names
        ):
            reads.add((item.comparators[0].id, item.left.value))
    return reads


def _scope(path: Path, function: Optional[str]) -> ast.AST:
    tree = _module_tree(path)
    if function is None:
        return tree
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return node
    raise AssertionError(f"{function} not found in {path}")


def _schema_at(schema: Any, dotted: str) -> Any:
    node = schema
    for part in [p for p in dotted.split(".") if p]:
        is_list = part.endswith("[]")
        node = node[part[:-2] if is_list else part]
        if is_list:
            assert isinstance(node, list), dotted
            node = node[0]
    return node


def _declared_leaves(schema: Any, prefix: str = "") -> Set[Tuple[str, str]]:
    leaves: Set[Tuple[str, str]] = set()
    for key, child in schema.items():
        path = f"{prefix}.{key}" if prefix else key
        if child is None:
            leaves.add((prefix, key))
        elif isinstance(child, list):
            leaves |= _declared_leaves(child[0], f"{path}[]")
        else:
            leaves |= _declared_leaves(child, path)
    return leaves


def _reads_by_path(readers) -> Set[Tuple[str, str]]:
    reads: Set[Tuple[str, str]] = set()
    for (path, function), variables in readers.items():
        for variable, key in _string_keys_read(_scope(path, function), variables):
            reads.add((variables[variable], key))
    return reads


@pytest.mark.parametrize(
    "schema, readers",
    [(GRPO_SCHEMA, GRPO_READERS), (ENV_GRPO_SCHEMA, ENV_GRPO_READERS)],
    ids=["grpo", "env_grpo"],
)
def test_declared_schema_matches_keys_the_code_reads(schema, readers):
    reads = {(path, key) for path, key in _reads_by_path(readers) if key not in INTERNAL_KEYS}
    undeclared = sorted(
        f"{path}.{key}" if path else key
        for path, key in reads
        if key not in _schema_at(schema, path)
    )
    assert not undeclared, f"code reads keys the schema does not declare: {undeclared}"
    unread = sorted(
        f"{path}.{key}" if path else key
        for path, key in _declared_leaves(schema)
        if (path, key) not in reads
    )
    assert not unread, f"schema declares keys nothing reads: {unread}"


# ---------------------------------------------------------------------------
# _build_grpo_config refuses user settings the installed GRPOConfig rejects
# ---------------------------------------------------------------------------


@dataclass
class FakeGRPOConfig:
    """Stands in for a TRL GRPOConfig without importance_sampling_level/top_p."""

    output_dir: str = ""
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 2
    max_prompt_length: int = 512
    max_completion_length: int = 256
    temperature: float = 1.0
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    logging_steps: int = 1
    save_steps: int = 1
    save_total_limit: int = 1
    num_train_epochs: int = 1
    max_steps: int = -1
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    seed: int = 0
    report_to: str = "none"
    run_name: Optional[str] = None
    beta: float = 0.04


def _build_grpo_config_function():
    tree = _module_tree(TRAIN_GRPO)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_grpo_config"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "Path": Path,
        "GRPOConfig": FakeGRPOConfig,
        "build_trainer_config": build_trainer_config,
        "is_bfloat16_supported": lambda: True,
    }
    exec(compile(module, str(TRAIN_GRPO), "exec"), namespace)
    return namespace["_build_grpo_config"]


def test_build_grpo_config_applies_checked_in_config():
    build = _build_grpo_config_function()
    config = _load_yaml("config.yaml")
    args = build(config, checkpoints_dir=Path("ckpt"))
    assert args.num_generations == config["training"]["num_generations"]
    assert args.max_prompt_length == config["training"]["max_prompt_length"]
    assert args.bf16 is True


def test_build_grpo_config_refuses_unsupported_method_switch_and_extra_args():
    build = _build_grpo_config_function()
    config = copy.deepcopy(_load_yaml("config.yaml"))
    config["training"]["use_gspo"] = True
    config["training"]["extra_args"] = {"top_p": 0.9}
    with pytest.raises(UnsupportedTrainerArgumentError) as excinfo:
        build(config, checkpoints_dir=Path("ckpt"))
    message = str(excinfo.value)
    assert excinfo.value.arguments == ["importance_sampling_level", "top_p"]
    assert "importance_sampling_level (set by training.use_gspo)" in message
    assert "top_p (set by training.extra_args.top_p)" in message
    assert "installed trl" in message


def test_build_grpo_config_wires_max_grad_norm_and_wandb_run_name():
    build = _build_grpo_config_function()
    config = copy.deepcopy(_load_yaml("config.yaml"))
    config["training"]["max_grad_norm"] = 0.5
    config["training"]["report_to"] = "wandb"
    config["wandb"]["run_name"] = "grpo-run"
    args = build(config, checkpoints_dir=Path("ckpt"))
    assert args.max_grad_norm == 0.5
    assert args.run_name == "grpo-run"


def test_env_grpo_cli_wires_checkpoint_cadence_and_has_no_dead_length_flag():
    import argparse

    function = next(
        node for node in _module_tree(TRAIN_ENV_GRPO).body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    namespace: Dict[str, Any] = {"argparse": argparse}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TRAIN_ENV_GRPO), "exec"), namespace)
    args = namespace["parse_args"](["--save-steps", "10", "--save-total-limit", "3"])
    assert (args.save_steps, args.save_total_limit) == (10, 3)
    with pytest.raises(SystemExit):
        namespace["parse_args"](["--max-seq-length", "4096"])
    run_source = ast.unparse(next(
        node for node in _module_tree(TRAIN_ENV_GRPO).body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    ))
    assert "training_cfg['save_steps'] = args.save_steps" in run_source
    assert "training_cfg['save_total_limit'] = args.save_total_limit" in run_source


def test_env_grpo_version_dependent_defaults_are_named_and_minimal():
    assert _literal(TRAIN_ENV_GRPO, "_VERSION_DEPENDENT_GRPO_DEFAULTS") == {"max_prompt_length"}
    source = TRAIN_ENV_GRPO.read_text(encoding="utf-8")
    assert "inspect.signature" not in source
    assert "version_dependent_defaults=_VERSION_DEPENDENT_GRPO_DEFAULTS" in source
