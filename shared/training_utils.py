"""
shared/training_utils.py

Shared training utilities for SFT, KTO, and GRPO trainers.

Consolidates duplicated functions from train_sft.py, train_kto.py,
and train_grpo.py. Each function previously existed in two or three
trainer scripts with cosmetic or minor behavioral drift.

Used by: Trainers/sft/train_sft.py, Trainers/kto/train_kto.py,
         Trainers/dpo/train_dpo.py, Trainers/grpo/train_grpo.py,
         Trainers/grpo/train_env_grpo.py, the trainer config loaders, and the
         SFT/KTO/DPO data loaders (validation split helpers).

Config strictness lives here too: ``reject_unknown_config_keys`` refuses YAML
keys a trainer does not declare, and ``build_trainer_config`` refuses trainer
arguments the installed TRL config class does not accept. Both exist so a typo
or an unsupported setting fails loudly instead of silently changing a run.
"""

from __future__ import annotations

import dataclasses
import difflib
import inspect
import json
import math
import os
import random
import re
import typing
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def setup_wandb() -> bool:
    """Auto-setup W&B if WANDB_API_KEY is in environment.

    Returns True if W&B login succeeded, False otherwise.
    """
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        return False

    try:
        import wandb

        wandb.login(key=wandb_key, relogin=True, force=True)
        print("[OK] W&B: Logged in automatically (using WANDB_API_KEY from .env)")
        return True
    except ImportError:
        print("[WARN] W&B: API key found but wandb not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        print(f"[WARN] W&B: Login failed ({e})")
        return False


def apply_wandb_destination(project: Optional[str], entity: Optional[str]) -> None:
    """Route the run's W&B logging to the configured project and entity.

    Trainers report to W&B through transformers' ``WandbCallback``, which takes
    the project from ``WANDB_PROJECT`` and lets ``wandb.init`` take the entity
    from ``WANDB_ENTITY``. Without this, ``wandb.project`` / ``wandb.entity``
    in a trainer YAML were accepted but never reached W&B. A null value leaves
    the corresponding environment variable untouched.
    """
    if project:
        os.environ["WANDB_PROJECT"] = str(project)
    if entity:
        os.environ["WANDB_ENTITY"] = str(entity)


def extract_previous_log_entries(checkpoint_path: str) -> List[dict]:
    """Extract log entries from a previous run when resuming from checkpoint.

    Parses the checkpoint path to determine the resume step, finds the
    most recent training log file, and returns entries up to that step.

    This is the canonical implementation adopted from KTO's version, which
    provides step-filtered extraction (vs SFT's unfiltered glob approach).

    Args:
        checkpoint_path: Path to checkpoint directory
            (e.g., "output/20251114_135227/checkpoints/checkpoint-50")

    Returns:
        List of log entry dicts (empty list on failure)
    """
    checkpoint_path = Path(checkpoint_path)

    # Extract step number from checkpoint name (e.g., "checkpoint-50" -> 50)
    checkpoint_name = checkpoint_path.name
    step_match = re.search(r"checkpoint-(\d+)", checkpoint_name)
    if not step_match:
        print(f"[WARN] Could not extract step number from checkpoint path: {checkpoint_path}")
        return []

    resume_step = int(step_match.group(1))

    # Navigate up to find the run directory (timestamp directory)
    # checkpoint_path is like: output/20251114_135227/checkpoints/checkpoint-50
    # We want: output/20251114_135227
    run_dir = checkpoint_path.parent.parent

    # Find log files in the logs subdirectory
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        print(f"[WARN] Logs directory not found: {logs_dir}")
        return []

    # Find training log files (there may be multiple if resuming multiple times)
    log_files = list(logs_dir.glob("training_*.jsonl"))
    if not log_files:
        print(f"[WARN] No log files found in: {logs_dir}")
        return []

    # Use the most recent log file (sorted by name, which includes timestamp)
    log_file = sorted(log_files)[-1]

    print(f"\n[OK] Found previous run log: {log_file}")
    print(f"  Extracting entries from steps 0 to {resume_step}")

    # Read log entries up to the resume step
    previous_entries: List[dict] = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                step = entry.get("step", 0)

                # Include entries up to and including the resume step
                if step <= resume_step:
                    previous_entries.append(entry)
                else:
                    break

        print(f"  Extracted {len(previous_entries)} log entries\n")
        return previous_entries

    except Exception as e:
        print(f"[WARN] Failed to read log file: {e}")
        return []


def save_training_lineage(lineage: Dict[str, Any], run_dir: Path) -> Path:
    """Save training lineage to JSON file, plus capacity features.

    Args:
        lineage: Training lineage dictionary
        run_dir: Path to training run directory

    Returns:
        Path to saved lineage file
    """
    from shared.training_capacity import build_capacity_feature_row

    lineage_path = run_dir / "training_lineage.json"

    with open(lineage_path, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2, default=str)

    print(f"[OK] Training lineage saved to: {lineage_path}")

    feature_row = build_capacity_feature_row(lineage)
    if feature_row:
        features_path = run_dir / "capacity_features.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(feature_row, f, indent=2, default=str)
        print(f"[OK] Capacity features saved to: {features_path}")

    return lineage_path


def build_base_lineage(
    training_type: str,
    model_info: Dict[str, Any],
    lora_info: Dict[str, Any],
    training_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    run_dir: Path,
    trainer: Any,
    training_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the shared portion of training lineage.

    Constructs the common lineage structure used by all trainers.
    Callers merge trainer-specific fields (e.g., KTO beta, SFT evolutionary)
    via dict.update() on the returned structure.

    This follows the Open/Closed Principle: the shared function is closed
    for modification, but open for extension via the caller's merge.

    Args:
        training_type: "SFT", "KTO", or "GRPO"
        model_info: Dict with keys: base_model, max_seq_length, load_in_4bit, dtype
        lora_info: Dict with keys: rank, alpha, dropout, target_modules, bias
        training_info: Dict with keys: batch_size, gradient_accumulation_steps,
            effective_batch_size, learning_rate, num_epochs, max_steps,
            warmup_ratio, lr_scheduler, optimizer, max_grad_norm,
            gradient_checkpointing, fp16, bf16, seed
        dataset_info: Dict with keys: source, train_examples, eval_examples
            (plus any trainer-specific keys)
        run_dir: Path to training run directory
        trainer: Trainer object after training (for state extraction)
        training_time_seconds: Total training time in seconds

    Returns:
        Lineage dict with shared fields populated. Callers extend this.
    """
    from datetime import datetime

    import torch

    from shared.training_capacity import capture_hardware_info, summarize_capacity_from_logs

    hardware_info = capture_hardware_info(torch)

    lineage: Dict[str, Any] = {
        "training_type": training_type,
        "timestamp": datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "model": dict(model_info),
        "lora": dict(lora_info),
        "training": dict(training_info),
        "dataset": dict(dataset_info),
        "hardware": hardware_info,
        "capacity_profile": summarize_capacity_from_logs(run_dir / "logs"),
        "results": {},
    }

    # Add training results if available
    if hasattr(trainer, "state") and trainer.state is not None:
        lineage["results"]["final_step"] = trainer.state.global_step
        lineage["results"]["total_epochs"] = trainer.state.epoch

        if hasattr(trainer.state, "log_history") and trainer.state.log_history:
            for entry in reversed(trainer.state.log_history):
                if "loss" in entry:
                    lineage["results"]["final_loss"] = entry["loss"]
                    break

    if training_time_seconds:
        lineage["results"]["training_time_seconds"] = round(training_time_seconds, 1)
        hours = training_time_seconds // 3600
        minutes = (training_time_seconds % 3600) // 60
        seconds = training_time_seconds % 60
        lineage["results"]["training_time_formatted"] = f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"

    return lineage


def apply_tier_preset(
    config: Any,
    tier_name: str,
    tier_config_map: Dict[str, tuple],
    args: Any,
    configs_dir: Path,
) -> Dict[str, Any]:
    """Apply a tier preset from YAML config to the training configuration.

    Each trainer defines its own tier_config_map that maps tier YAML keys
    to (section, attribute) pairs on the config dataclass.

    Args:
        config: Training configuration dataclass
        tier_name: Name of the tier (e.g., "quick", "standard", "thorough")
        tier_config_map: Dict mapping tier key names to (section, attr) tuples
            on the config dataclass
        args: Command-line arguments namespace (for max_steps routing)
        configs_dir: Path to the configs directory containing tiers/ subdirectory

    Returns:
        The parsed tier_config dict (for logging)

    Raises:
        FileNotFoundError: If the tier YAML file does not exist
        UnknownConfigKeysError: If the tier YAML names a key the trainer's
            tier_config_map does not route (it would otherwise be ignored)
    """
    import yaml as _yaml

    tier_path = configs_dir / "tiers" / f"{tier_name}.yaml"
    if not tier_path.exists():
        raise FileNotFoundError(f"Tier config not found: {tier_path}")

    with open(tier_path) as f:
        tier_config = _yaml.safe_load(f) or {}

    tier_schema: Dict[str, Any] = {key: None for key in tier_config_map}
    tier_schema["max_steps"] = None  # routed through args, see below
    reject_unknown_config_keys(tier_schema, tier_config, source=str(tier_path))

    for key, value in tier_config.items():
        if key == "max_steps":
            # max_steps is handled via args, not config
            if getattr(args, "max_steps", None) is None:
                args.max_steps = value
        else:
            section, attr = tier_config_map[key]
            target = getattr(config, section)
            field_type = typing.get_type_hints(type(target))[attr]
            setattr(target, attr, coerce_config_value(field_type, value))

    print(f"Applied '{tier_name}' tier preset: {tier_config}")
    return tier_config


# ---------------------------------------------------------------------------
# Config key strictness
# ---------------------------------------------------------------------------
#
# A config *schema* is either a config dataclass (its fields are the declared
# keys; fields typed as another dataclass are walked as nested sections) or a
# plain literal:
#
#   * ``None``            a leaf; its value is not walked (scalars, lists, and
#                         free-form mappings such as ``training.extra_args``)
#   * ``{key: schema}``   a closed mapping; only these keys are accepted
#   * ``[schema]``        a list whose mapping items each follow ``schema``
#
# Literal schemas stay ``ast.literal_eval``-able so tests can read them from a
# trainer module without importing its GPU stack.

MAX_REPORTED_UNKNOWN_KEYS = 20


class UnknownConfigKeysError(ValueError):
    """A config names keys its trainer schema does not declare."""

    def __init__(self, source: str, unknown: List[Tuple[str, Optional[str]]]):
        self.source = source
        self.unknown = list(unknown)
        self.paths = [path for path, _ in self.unknown]
        shown = self.unknown[:MAX_REPORTED_UNKNOWN_KEYS]
        lines = [
            f"{source}: {len(self.unknown)} config key(s) are not declared by the "
            "trainer schema:"
        ]
        for path, suggestion in shown:
            hint = f" (did you mean '{suggestion}'?)" if suggestion else ""
            lines.append(f"  - {path}{hint}")
        hidden = len(self.unknown) - len(shown)
        if hidden > 0:
            lines.append(f"  ... and {hidden} more")
        lines.append(
            "Unknown keys are refused so a typo or unsupported setting cannot "
            "silently change a training run. Fix the spelling or remove the key."
        )
        super().__init__("\n".join(lines))


def _unwrap_optional(annotation: Any) -> Any:
    args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    origin = typing.get_origin(annotation)
    is_union = origin is typing.Union or type(annotation).__name__ == "UnionType"
    if is_union and len(args) == 1:
        return args[0]
    return annotation


def config_schema_from_dataclass(cls: type) -> Dict[str, Any]:
    """Return the literal key schema a (nested) config dataclass declares."""
    hints = typing.get_type_hints(cls)
    schema: Dict[str, Any] = {}
    for item in dataclasses.fields(cls):
        annotation = _unwrap_optional(hints.get(item.name, item.type))
        if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
            schema[item.name] = config_schema_from_dataclass(annotation)
        else:
            schema[item.name] = None
    return schema


def find_unknown_config_keys(
    schema: Any, data: Any, *, prefix: str = ""
) -> List[Tuple[str, Optional[str]]]:
    """List ``(dotted_path, suggestion)`` for every undeclared key in ``data``.

    Values whose type does not match the schema shape (e.g. a scalar where a
    section is expected) are not walked; type errors belong to the loader.
    """
    if isinstance(schema, type) and dataclasses.is_dataclass(schema):
        schema = config_schema_from_dataclass(schema)
    if schema is None:
        return []
    if isinstance(schema, list):
        if len(schema) != 1:
            raise ValueError("a list schema must hold exactly one item schema")
        if not isinstance(data, list):
            return []
        found: List[Tuple[str, Optional[str]]] = []
        for index, item in enumerate(data):
            found.extend(find_unknown_config_keys(schema[0], item, prefix=f"{prefix}[{index}]"))
        return found
    if not isinstance(schema, Mapping):
        raise TypeError(f"unsupported config schema node: {schema!r}")
    if not isinstance(data, Mapping):
        return []
    declared = [str(key) for key in schema]
    found = []
    for key, value in data.items():
        name = str(key)
        path = f"{prefix}.{name}" if prefix else name
        if key not in schema:
            close = difflib.get_close_matches(name, declared, n=1)
            found.append((path, close[0] if close else None))
            continue
        found.extend(find_unknown_config_keys(schema[key], value, prefix=path))
    return found


def reject_unknown_config_keys(
    schema: Any,
    data: Any,
    *,
    source: str,
    prefix: str = "",
    extra_top_level_keys: Iterable[str] = (),
) -> None:
    """Raise ``UnknownConfigKeysError`` if ``data`` has any undeclared key.

    ``extra_top_level_keys`` admits top-level keys that another component owns
    and validates (the trainer does not read them); callers must name that
    owner next to the set they pass.
    """
    if isinstance(schema, type) and dataclasses.is_dataclass(schema):
        schema = config_schema_from_dataclass(schema)
    extra = tuple(extra_top_level_keys)
    if extra:
        if not isinstance(schema, Mapping):
            raise TypeError("extra_top_level_keys requires a mapping schema")
        schema = {**schema, **{key: None for key in extra}}
    unknown = find_unknown_config_keys(schema, data, prefix=prefix)
    if unknown:
        raise UnknownConfigKeysError(source, unknown)


def dict_to_dataclass(cls, data: Dict[str, Any], *, section: str = ""):
    """Convert one config section mapping to its dataclass, refusing unknown keys.

    Numeric fields given as strings (YAML parses ``5e-6`` as a string) are
    coerced to ``int``/``float``. ``section`` prefixes reported key paths.
    """
    reject_unknown_config_keys(cls, data, source=cls.__name__, prefix=section)
    fieldtypes = typing.get_type_hints(cls)
    return cls(**{k: coerce_config_value(fieldtypes[k], v) for k, v in data.items()})


def coerce_config_value(field_type: Any, value: Any) -> Any:
    """Coerce a YAML value to a numeric config field's declared type.

    PyYAML loads exponent-only floats such as ``5e-4`` as strings, so a
    string given for an ``int``/``float`` (or ``Optional`` of one) field is
    converted; every other value is returned unchanged. The single conversion
    path for ``dict_to_dataclass`` and ``apply_tier_preset``.
    """
    # Handle Optional types
    if hasattr(field_type, '__origin__') and field_type.__origin__ is typing.Union:
        # Get the non-None type from Optional
        types = [t for t in field_type.__args__ if t is not type(None)]
        if types:
            field_type = types[0]

    # Convert strings to appropriate numeric types
    if field_type == float and isinstance(value, str):
        return float(value)
    if field_type == int and isinstance(value, str):
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Trainer (TRL) config construction
# ---------------------------------------------------------------------------


class UnsupportedTrainerArgumentError(ValueError):
    """The installed trainer library does not accept a configured argument."""

    def __init__(self, message: str, arguments: List[str]):
        self.arguments = list(arguments)
        super().__init__(message)


def installed_package_version(package: str) -> str:
    """Return the installed distribution version of ``package`` or ``unknown``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def accepted_config_parameters(config_cls: type) -> frozenset:
    """Keyword arguments ``config_cls(...)`` accepts.

    Explicit ``__init__`` parameters, plus the dataclass fields when ``__init__``
    forwards ``**kwargs`` (Unsloth's patched TRL configs subclass the TRL
    dataclass this way).
    """
    parameters = inspect.signature(config_cls.__init__).parameters
    keyword_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    named = {
        name
        for name, parameter in parameters.items()
        if name != "self" and parameter.kind in keyword_kinds
    }
    forwards_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if forwards_kwargs and dataclasses.is_dataclass(config_cls):
        named |= {item.name for item in dataclasses.fields(config_cls) if item.init}
    return frozenset(named)


def partition_trainer_kwargs(
    config_cls: type,
    kwargs: Mapping[str, Any],
    *,
    origins: Mapping[str, str],
    version_dependent_defaults: Iterable[str] = (),
    library: str = "trl",
    installed_version: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Split ``kwargs`` into what ``config_cls`` accepts and what may be omitted.

    ``origins`` maps every argument that came from the user (YAML keys, method
    switches, CLI overrides) to the config path that set it. An unsupported
    argument raises ``UnsupportedTrainerArgumentError`` naming it and the
    installed ``library`` version, unless it is an internal default (absent
    from ``origins``) listed in ``version_dependent_defaults``; only those are
    omitted. Returns ``(accepted_kwargs, omitted_names)``.
    """
    accepted = accepted_config_parameters(config_cls)
    optional_defaults = set(version_dependent_defaults)
    unsupported = [name for name in kwargs if name not in accepted]
    rejected = [name for name in unsupported if name in origins or name not in optional_defaults]
    if rejected:
        version = installed_version or installed_package_version(library)
        lines = [
            f"{config_cls.__name__} in the installed {library} {version} does not "
            f"accept {len(rejected)} configured argument(s):"
        ]
        for name in rejected:
            origin = origins.get(name)
            lines.append(f"  - {name} (set by {origin})" if origin else f"  - {name} (internal default)")
        lines.append(
            f"Remove the setting or install a {library} version that supports it; "
            "unsupported settings are refused instead of silently dropped."
        )
        raise UnsupportedTrainerArgumentError("\n".join(lines), rejected)
    omitted = [name for name in unsupported if name not in rejected]
    return {k: v for k, v in kwargs.items() if k not in omitted}, omitted


def build_trainer_config(
    config_cls: type,
    kwargs: Mapping[str, Any],
    *,
    origins: Mapping[str, str],
    version_dependent_defaults: Iterable[str] = (),
    library: str = "trl",
):
    """Construct ``config_cls`` after ``partition_trainer_kwargs`` validation."""
    accepted, omitted = partition_trainer_kwargs(
        config_cls,
        kwargs,
        origins=origins,
        version_dependent_defaults=version_dependent_defaults,
        library=library,
    )
    if omitted:
        print(
            f"[INFO] {config_cls.__name__} in {library} "
            f"{installed_package_version(library)} has no {omitted}; omitting "
            "these version-dependent internal defaults"
        )
    return config_cls(**accepted)


# ---------------------------------------------------------------------------
# Train/validation split (SFT, KTO, DPO data loaders)
# ---------------------------------------------------------------------------
#
# A plain random row split puts near-duplicate variants of one scenario,
# template, seed or transcript on both sides, which makes validation loss look
# better than it is. With ``validation_group_key`` (a dot-path into the raw row,
# e.g. ``metadata.scenario``) every row sharing a group value lands on the same
# side and ``test_size`` is applied over groups. The assignment is deterministic
# for a given seed and set of group values, independent of row order. Without a
# group key the split is the historical ``train_test_split(seed=42)``.
#
# Kept in this module (a member of the offline SFT worker closure) and free of
# non-stdlib imports so the packaged trainer can execute it without widening
# the closure.

DEFAULT_SPLIT_SEED = 42


_MISSING = object()


def _get_dotted(record: Any, dotted_path: str) -> Any:
    current = record
    for segment in dotted_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def _canonical_group_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def extract_group_values(rows: Iterable[Mapping[str, Any]], group_key: str) -> List[str]:
    """Resolve ``group_key`` on every row; fail loudly on the first missing value.

    A value that is absent, ``None`` (HF ``datasets`` fills absent struct keys
    with ``None``) or an empty/whitespace string counts as missing.
    """
    if not isinstance(group_key, str) or not group_key.strip():
        raise ValueError("validation_group_key must be a non-empty dot-path string.")
    values: List[str] = []
    for index, row in enumerate(rows):
        value = _get_dotted(row, group_key)
        if value is _MISSING or value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(
                f"validation_group_key={group_key!r} is missing, null or empty on row "
                f"index {index}. Every row must carry a group value when grouped "
                "validation splitting is enabled."
            )
        values.append(_canonical_group_value(value))
    return values


def extract_dataset_group_values(dataset: Any, group_key: str) -> List[str]:
    """Like :func:`extract_group_values` for a HF ``datasets.Dataset``.

    Only the top-level column the dot-path starts with is materialized.
    """
    top_level = group_key.split(".", 1)[0] if isinstance(group_key, str) else ""
    if top_level not in dataset.column_names:
        if len(dataset) == 0:
            return []
        raise ValueError(
            f"validation_group_key={group_key!r} is missing on row index 0: the dataset "
            f"has no {top_level!r} column (columns: {dataset.column_names})."
        )
    return extract_group_values(dataset.select_columns([top_level]).to_list(), group_key)


def grouped_split_indices(
    group_values: Sequence[str],
    test_size: float,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[int], List[int]]:
    """Assign whole groups to train or validation.

    ``ceil(test_size * n_groups)`` groups (at least one, at most ``n_groups - 1``)
    go to validation. Groups are sorted before the seeded shuffle so the result
    depends only on the seed and the set of group values.

    Returns:
        ``(train_indices, validation_indices)``, each in ascending row order.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1) for a grouped split, got {test_size!r}.")
    unique_groups = sorted(set(group_values))
    if len(unique_groups) < 2:
        raise ValueError(
            f"Grouped validation split needs at least 2 distinct groups; found "
            f"{len(unique_groups)}. Choose a finer validation_group_key or disable it."
        )
    rng = random.Random(seed)
    shuffled = list(unique_groups)
    rng.shuffle(shuffled)
    n_test = min(len(shuffled) - 1, max(1, math.ceil(test_size * len(shuffled))))
    test_groups = set(shuffled[:n_test])
    train_indices: List[int] = []
    test_indices: List[int] = []
    for index, value in enumerate(group_values):
        (test_indices if value in test_groups else train_indices).append(index)
    return train_indices, test_indices


def split_train_validation(
    dataset: Any,
    *,
    test_size: float,
    seed: int = DEFAULT_SPLIT_SEED,
    group_values: Optional[Sequence[str]] = None,
) -> Tuple[Any, Any]:
    """Split a HF ``Dataset`` into train/validation.

    Args:
        dataset: ``datasets.Dataset`` to split.
        test_size: Validation fraction (of rows when ungrouped, of groups when grouped).
        seed: Split seed.
        group_values: Optional per-row group values aligned with ``dataset`` rows
            (see :func:`extract_dataset_group_values`). ``None`` keeps the random
            row split.
    """
    if group_values is None:
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        return split["train"], split["test"]
    if len(group_values) != len(dataset):
        raise ValueError(
            f"group_values has {len(group_values)} entries but the dataset has "
            f"{len(dataset)} rows; they must be aligned."
        )
    train_indices, test_indices = grouped_split_indices(group_values, test_size, seed)
    n_groups = len(set(group_values))
    n_test_groups = len({group_values[i] for i in test_indices})
    print(
        f"  Grouped split: {n_groups} groups -> {n_groups - n_test_groups} train / "
        f"{n_test_groups} validation"
    )
    return dataset.select(train_indices), dataset.select(test_indices)
