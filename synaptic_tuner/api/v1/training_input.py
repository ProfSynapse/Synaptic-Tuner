"""Provider-neutral, canonical training input contract."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from ._contract import contract_digest


_TRAINING_SCHEMA = "synaptic-training-input/v1"
_SFT_SCHEMA = "synaptic-sft-hyperparameters/v1"
_ENV_GRPO_SCHEMA = "synaptic-env-grpo-hyperparameters/v1"
_MAX_JSON_BYTES = 64 * 1024
_MAX_REF_BYTES = 512
_MAX_ITEM_BYTES = 128
_MAX_TARGET_MODULES = 256
_MAX_REQUIRED_KINDS = 64
_MAX_STEPS = 10_000_000
_MAX_EPOCHS = 1000.0
_MAX_LEARNING_RATE = 1.0
_MAX_BATCH_SIZE = 4096
_MAX_GRADIENT_ACCUMULATION_STEPS = 4096
_MAX_SEQ_LENGTH = 1_048_576
_MAX_SAVE_STEPS = 10_000_000
_MAX_SAVE_TOTAL_LIMIT = 10_000
_MAX_LORA_RANK = 4096
_MAX_LORA_ALPHA = 65_536
_MAX_SEED = 4_294_967_295
_MAX_CHAT_TEMPLATE_KWARGS_BYTES = 4096
_MAX_NUM_GENERATIONS = 1024
_MAX_TEMPERATURE = 10.0
_MAX_BETA = 10.0
_MAX_TURNS = 1024
_MAX_TOOL_STEPS = 4096
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
# Env-GRPO rollouts run in-process against the local environment validator;
# remote sandboxes (for example e2b) need credentials and network egress.
_ENV_GRPO_BACKENDS = frozenset({"local"})
_ENV_GRPO_CONTEXT_TOKEN_POLICIES = frozenset({"mask", "drop"})
_RESERVED_CHAT_TEMPLATE_KWARGS = frozenset({
    "messages", "tokenize", "add_generation_prompt", "return_dict",
    "return_tensors", "continue_final_message", "chat_template",
})
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_ASCII_COMPONENT = re.compile(r"[A-Za-z0-9]+")
_CREDENTIAL_KEYS = frozenset(
    {
        "key", "token", "accesstoken", "refreshtoken", "idtoken", "apikey",
        "secret", "clientsecret", "password", "passwd", "pwd", "authorization",
        "auth", "bearer", "signature", "sig", "credential", "credentials",
        "session", "sessionid", "sessiontoken", "cookie",
    }
)
_CREDENTIAL_SUFFIXES = (
    "token", "secret", "password", "passwd", "apikey", "signature",
    "credential", "credentials",
)
_ENCODED_PATH_BYTES = frozenset({0x2E, 0x2F, 0x5C, 0x7E})


def _text(value: object, field: str, *, maximum_bytes: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    if not value:
        raise ValueError(f"{field} is required")
    if value != value.strip():
        raise ValueError(f"{field} has invalid whitespace")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field} must be NFC")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError(f"{field} contains a control character")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError(f"{field} is not valid UTF-8 text") from None
    if size > maximum_bytes:
        raise ValueError(f"{field} exceeds its byte limit")
    return value


def _query_keys(query: str, field: str) -> None:
    if query == "":
        raise ValueError(f"{field} contains an empty query key")
    seen: set[str] = set()
    for pair in query.split("&"):
        raw_key = pair.partition("=")[0]
        if not raw_key:
            raise ValueError(f"{field} contains an empty query key")
        folded = raw_key.casefold()
        if folded in seen:
            raise ValueError(f"{field} contains duplicate query keys")
        seen.add(folded)
        components = tuple(item.casefold() for item in _ASCII_COMPONENT.findall(raw_key))
        compact = "".join(components)
        adjacent_key = any(
            pair in {("access", "key"), ("private", "key")}
            for pair in zip(components, components[1:])
        )
        if (
            compact in _CREDENTIAL_KEYS
            or compact.endswith(_CREDENTIAL_SUFFIXES)
            or "accesskey" in compact
            or "privatekey" in compact
            or adjacent_key
        ):
            raise ValueError(f"{field} must not contain credential query keys")


def _validate_ref_lexical(value: str, field: str, *, validate_query: bool) -> None:
    if "#" in value:
        raise ValueError(f"{field} must not contain a fragment")
    base, separator, query = value.partition("?")
    if (
        base.startswith(("/", "//", "\\", "./", "../", "~/", ".\\", "..\\", "~\\"))
        or "\\" in base
        or _WINDOWS_DRIVE.match(base) is not None
        or any(segment in {".", "..", "~"} for segment in base.split("/"))
    ):
        raise ValueError(f"{field} must be a logical reference")
    try:
        parsed = urlsplit(base)
        if parsed.scheme.casefold() == "file":
            raise ValueError(f"{field} must not use the file scheme")
        if "://" in base and (not parsed.netloc or "@" in parsed.netloc):
            raise ValueError(f"{field} has invalid URI authority")
        if parsed.netloc and (
            parsed.username is not None or parsed.password is not None or "@" in parsed.netloc
        ):
            raise ValueError(f"{field} must not contain URI userinfo")
    except ValueError:
        raise ValueError(f"{field} is not a valid logical reference") from None
    if separator and validate_query:
        _query_keys(query, field)


def _project_ref(value: str, field: str) -> str:
    projected = bytearray()
    cursor = 0
    while cursor < len(value):
        character = value[cursor]
        if character != "%":
            try:
                projected.extend(character.encode("utf-8"))
            except UnicodeEncodeError:
                raise ValueError(f"{field} is not valid UTF-8 text") from None
            cursor += 1
            continue
        if cursor + 2 >= len(value):
            raise ValueError(f"{field} contains an invalid percent escape")
        encoded = value[cursor + 1:cursor + 3]
        try:
            byte = int(encoded, 16)
        except ValueError:
            raise ValueError(f"{field} contains an invalid percent escape") from None
        if byte in _ENCODED_PATH_BYTES:
            raise ValueError(f"{field} contains an encoded path character")
        projected.append(byte)
        cursor += 3
    try:
        result = bytes(projected).decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{field} contains invalid projected UTF-8") from None
    if "%" in result:
        raise ValueError(f"{field} contains residual percent encoding")
    return result


def _logical_ref(value: object, field: str) -> str:
    original = _text(value, field, maximum_bytes=_MAX_REF_BYTES)
    _validate_ref_lexical(original, field, validate_query=False)
    projected = _project_ref(original, field)
    projected = _text(projected, field, maximum_bytes=_MAX_REF_BYTES)
    _validate_ref_lexical(projected, field, validate_query=True)
    return projected


def _exact_integer(
    value: object, field: str, *, minimum: int, maximum: int | None = None
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{field} is outside its allowed range")
    return value


def _finite_float(
    value: object, field: str, *, minimum_exclusive: float,
    maximum_inclusive: float | None = None,
) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{field} must be a number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{field} is outside its allowed range") from None
    if (
        not math.isfinite(normalized)
        or normalized <= minimum_exclusive
        or (maximum_inclusive is not None and normalized > maximum_inclusive)
    ):
        raise ValueError(f"{field} is outside its allowed range")
    return normalized


def _dropout(value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError("lora_dropout must be a number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("lora_dropout is outside its allowed range") from None
    if not math.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError("lora_dropout is outside its allowed range")
    return normalized


def _exact_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a boolean")
    return value


def _canonical_items(
    value: object, field: str, *, maximum_items: int
) -> tuple[str, ...]:
    if type(value) not in (tuple, list):
        raise TypeError(f"{field} must be an array")
    items = tuple(
        _text(item, field, maximum_bytes=_MAX_ITEM_BYTES) for item in value
    )
    if not items or len(items) > maximum_items:
        raise ValueError(f"{field} has invalid cardinality")
    if len(items) != len(set(items)):
        raise ValueError(f"{field} must contain unique values")
    if items != tuple(sorted(items)):
        raise ValueError(f"{field} must be ascending")
    return items


def _fields(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object")
    try:
        snapshot = value.copy()
    except (RuntimeError, TypeError, ValueError):
        raise ValueError(f"{name} could not be snapshotted") from None
    if type(snapshot) is not dict:
        raise ValueError(f"{name} could not be snapshotted")
    if any(type(key) is not str for key in snapshot):
        raise TypeError(f"{name} field names must be strings")
    if frozenset(snapshot) != expected:
        raise ValueError(f"{name} has invalid fields")
    return snapshot


class TrainingMethodV1(str, Enum):
    SFT = "sft"
    GRPO = "grpo"


@dataclass(frozen=True, slots=True)
class TrainingModelInputV1:
    ref: str
    revision: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _logical_ref(self.ref, "model.ref"))
        object.__setattr__(self, "revision", _logical_ref(self.revision, "model.revision"))
        object.__setattr__(
            self,
            "tokenizer_revision",
            _logical_ref(self.tokenizer_revision, "model.tokenizer_revision"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainingModelInputV1":
        value = _fields(
            value, frozenset({"ref", "revision", "tokenizer_revision"}), "model"
        )
        return cls(value["ref"], value["revision"], value["tokenizer_revision"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TrainingDatasetInputV1:
    ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _logical_ref(self.ref, "dataset.ref"))

    def to_dict(self) -> dict[str, object]:
        return {"ref": self.ref}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainingDatasetInputV1":
        value = _fields(value, frozenset({"ref"}), "dataset")
        return cls(value["ref"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TrainingDurationV1:
    max_steps: int | None
    num_epochs: float | None

    def __post_init__(self) -> None:
        if (self.max_steps is None) == (self.num_epochs is None):
            raise ValueError("duration requires exactly one limit")
        if self.max_steps is not None:
            object.__setattr__(
                self,
                "max_steps",
                _exact_integer(
                    self.max_steps, "duration.max_steps", minimum=1, maximum=_MAX_STEPS
                ),
            )
        if self.num_epochs is not None:
            object.__setattr__(
                self,
                "num_epochs",
                _finite_float(
                    self.num_epochs,
                    "duration.num_epochs",
                    minimum_exclusive=0.0,
                    maximum_inclusive=_MAX_EPOCHS,
                ),
            )

    def to_dict(self) -> dict[str, object]:
        return {"max_steps": self.max_steps, "num_epochs": self.num_epochs}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainingDurationV1":
        value = _fields(value, frozenset({"max_steps", "num_epochs"}), "duration")
        return cls(value["max_steps"], value["num_epochs"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SFTTrainingHyperparametersV1:
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    duration: TrainingDurationV1
    max_seq_length: int
    seed: int
    save_steps: int
    save_total_limit: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    use_dora: bool
    use_rslora: bool
    init_lora_weights: bool
    split_dataset: bool
    dataset_format: str | None = None
    completion_only_loss: bool | None = None
    assistant_only_loss: bool | None = None
    use_preassigned_splits: bool | None = None
    prompt_render: str | None = None
    packing: bool | None = None
    require_memory_efficient_loss: bool | None = None
    chat_template_kwargs: dict[str, object] | None = None

    def __post_init__(self) -> None:
        integer_bounds = {
            "batch_size": _MAX_BATCH_SIZE,
            "gradient_accumulation_steps": _MAX_GRADIENT_ACCUMULATION_STEPS,
            "max_seq_length": _MAX_SEQ_LENGTH,
            "save_steps": _MAX_SAVE_STEPS,
            "save_total_limit": _MAX_SAVE_TOTAL_LIMIT,
            "lora_rank": _MAX_LORA_RANK,
            "lora_alpha": _MAX_LORA_ALPHA,
        }
        for field, maximum in integer_bounds.items():
            object.__setattr__(
                self,
                field,
                _exact_integer(getattr(self, field), field, minimum=1, maximum=maximum),
            )
        object.__setattr__(
            self,
            "seed",
            _exact_integer(self.seed, "seed", minimum=0, maximum=_MAX_SEED),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _finite_float(
                self.learning_rate,
                "learning_rate",
                minimum_exclusive=0.0,
                maximum_inclusive=_MAX_LEARNING_RATE,
            ),
        )
        if type(self.duration) is not TrainingDurationV1:
            raise TypeError("duration must be exact TrainingDurationV1")
        object.__setattr__(self, "lora_dropout", _dropout(self.lora_dropout))
        object.__setattr__(
            self,
            "lora_target_modules",
            _canonical_items(
                self.lora_target_modules,
                "lora_target_modules",
                maximum_items=_MAX_TARGET_MODULES,
            ),
        )
        for field in ("use_dora", "use_rslora", "init_lora_weights", "split_dataset"):
            object.__setattr__(self, field, _exact_bool(getattr(self, field), field))
        prepared_fields = (
            "dataset_format",
            "completion_only_loss",
            "assistant_only_loss",
            "use_preassigned_splits",
            "prompt_render",
            "packing",
            "require_memory_efficient_loss",
        )
        present = tuple(getattr(self, field) is not None for field in prepared_fields)
        if any(present) and not all(present):
            raise ValueError("prepared SFT controls must be supplied together")
        if all(present):
            if self.dataset_format not in {"raw_text", "messages"}:
                raise ValueError("dataset_format is unsupported")
            if self.prompt_render not in {"full_conversation", "prompt_completion"}:
                raise ValueError("prompt_render is unsupported")
            for field in (
                "completion_only_loss",
                "assistant_only_loss",
                "use_preassigned_splits",
                "packing",
                "require_memory_efficient_loss",
            ):
                object.__setattr__(
                    self, field, _exact_bool(getattr(self, field), field)
                )
        if self.chat_template_kwargs is not None:
            if self.dataset_format == "raw_text":
                raise ValueError("chat_template_kwargs does not apply to raw text")
            object.__setattr__(self, "chat_template_kwargs",
                               validate_chat_template_kwargs(self.chat_template_kwargs))

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema_version": _SFT_SCHEMA,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "duration": self.duration.to_dict(),
            "max_seq_length": self.max_seq_length,
            "seed": self.seed,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
            "use_dora": self.use_dora,
            "use_rslora": self.use_rslora,
            "init_lora_weights": self.init_lora_weights,
            "split_dataset": self.split_dataset,
        }
        if self.dataset_format is not None:
            result.update(
                {
                    "dataset_format": self.dataset_format,
                    "completion_only_loss": self.completion_only_loss,
                    "assistant_only_loss": self.assistant_only_loss,
                    "use_preassigned_splits": self.use_preassigned_splits,
                    "prompt_render": self.prompt_render,
                    "packing": self.packing,
                    "require_memory_efficient_loss": self.require_memory_efficient_loss,
                }
            )
        if self.chat_template_kwargs is not None:
            result["chat_template_kwargs"] = validate_chat_template_kwargs(self.chat_template_kwargs)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SFTTrainingHyperparametersV1":
        base_fields = frozenset(
            {
                "schema_version", "batch_size", "gradient_accumulation_steps",
                "learning_rate", "duration", "max_seq_length", "seed", "save_steps",
                "save_total_limit", "lora_rank", "lora_alpha", "lora_dropout",
                "lora_target_modules", "use_dora", "use_rslora",
                "init_lora_weights", "split_dataset",
            }
        )
        prepared_fields = frozenset(
            {
                "dataset_format", "completion_only_loss", "assistant_only_loss",
                "use_preassigned_splits", "prompt_render", "packing",
                "require_memory_efficient_loss",
            }
        )
        supplied = frozenset(value) if type(value) is dict else frozenset()
        expected = base_fields | prepared_fields if supplied & prepared_fields else base_fields
        if "chat_template_kwargs" in supplied:
            expected = expected | {"chat_template_kwargs"}
        value = _fields(value, expected, "hyperparameters")
        if value["schema_version"] != _SFT_SCHEMA:
            raise ValueError("hyperparameters schema is unsupported")
        duration = value["duration"]
        if type(duration) is not dict:
            raise TypeError("hyperparameters.duration must be an object")
        return cls(
            batch_size=value["batch_size"],  # type: ignore[arg-type]
            gradient_accumulation_steps=value["gradient_accumulation_steps"],  # type: ignore[arg-type]
            learning_rate=value["learning_rate"],  # type: ignore[arg-type]
            duration=TrainingDurationV1.from_dict(duration),
            max_seq_length=value["max_seq_length"],  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            save_steps=value["save_steps"],  # type: ignore[arg-type]
            save_total_limit=value["save_total_limit"],  # type: ignore[arg-type]
            lora_rank=value["lora_rank"],  # type: ignore[arg-type]
            lora_alpha=value["lora_alpha"],  # type: ignore[arg-type]
            lora_dropout=value["lora_dropout"],  # type: ignore[arg-type]
            lora_target_modules=value["lora_target_modules"],  # type: ignore[arg-type]
            use_dora=value["use_dora"],  # type: ignore[arg-type]
            use_rslora=value["use_rslora"],  # type: ignore[arg-type]
            init_lora_weights=value["init_lora_weights"],  # type: ignore[arg-type]
            split_dataset=value["split_dataset"],  # type: ignore[arg-type]
            dataset_format=value.get("dataset_format"),  # type: ignore[arg-type]
            completion_only_loss=value.get("completion_only_loss"),  # type: ignore[arg-type]
            assistant_only_loss=value.get("assistant_only_loss"),  # type: ignore[arg-type]
            use_preassigned_splits=value.get("use_preassigned_splits"),  # type: ignore[arg-type]
            prompt_render=value.get("prompt_render"),  # type: ignore[arg-type]
            packing=value.get("packing"),  # type: ignore[arg-type]
            require_memory_efficient_loss=value.get("require_memory_efficient_loss"),  # type: ignore[arg-type]
            chat_template_kwargs=value.get("chat_template_kwargs"),  # type: ignore[arg-type]
        )


def _bounded_nonnegative_float(value: object, field: str, *, maximum_inclusive: float) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{field} must be a number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{field} is outside its allowed range") from None
    if not math.isfinite(normalized) or not 0.0 <= normalized <= maximum_inclusive:
        raise ValueError(f"{field} is outside its allowed range")
    return normalized


@dataclass(frozen=True, slots=True)
class EnvGRPOHyperparametersV1:
    """Environment-backed GRPO controls (``Trainers/grpo/train_env_grpo.py``).

    Rollouts are generated in-process by transformers (``use_vllm`` is false and
    ``allow_transformers_rollout_func`` is true) against the local environment
    validator. Reward weights are referenced by a logical ref plus content
    digest rather than inlined, so the reward table stays a reviewed artifact.
    """

    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_steps: int
    seed: int
    save_steps: int
    save_total_limit: int
    num_generations: int
    max_completion_length: int
    temperature: float
    beta: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    env_backend: str
    max_turns: int
    max_tool_steps: int
    token_faithful: bool
    context_token_policy: str
    reward_config_ref: str
    reward_config_digest: str
    use_vllm: bool
    allow_transformers_rollout_func: bool

    def __post_init__(self) -> None:
        integer_bounds = {
            "batch_size": (1, _MAX_BATCH_SIZE),
            "gradient_accumulation_steps": (1, _MAX_GRADIENT_ACCUMULATION_STEPS),
            "max_steps": (1, _MAX_STEPS),
            "seed": (0, _MAX_SEED),
            "save_steps": (1, _MAX_SAVE_STEPS),
            "save_total_limit": (1, _MAX_SAVE_TOTAL_LIMIT),
            # Group-relative advantages need at least two completions per prompt.
            "num_generations": (2, _MAX_NUM_GENERATIONS),
            "max_completion_length": (1, _MAX_SEQ_LENGTH),
            "lora_rank": (1, _MAX_LORA_RANK),
            "lora_alpha": (1, _MAX_LORA_ALPHA),
            "max_turns": (1, _MAX_TURNS),
            "max_tool_steps": (1, _MAX_TOOL_STEPS),
        }
        for field, (minimum, maximum) in integer_bounds.items():
            object.__setattr__(
                self,
                field,
                _exact_integer(getattr(self, field), field, minimum=minimum, maximum=maximum),
            )
        # TRL's GRPOConfig derives generation_batch_size as per-device batch x
        # gradient accumulation x world size and requires whole prompt groups.
        # Packaged Modal runs use exactly one accelerator (world size 1).
        if (self.batch_size * self.gradient_accumulation_steps) % self.num_generations:
            raise ValueError(
                "batch_size * gradient_accumulation_steps must be divisible by num_generations"
            )
        object.__setattr__(
            self,
            "learning_rate",
            _finite_float(
                self.learning_rate, "learning_rate",
                minimum_exclusive=0.0, maximum_inclusive=_MAX_LEARNING_RATE,
            ),
        )
        object.__setattr__(
            self,
            "temperature",
            _finite_float(
                self.temperature, "temperature",
                minimum_exclusive=0.0, maximum_inclusive=_MAX_TEMPERATURE,
            ),
        )
        object.__setattr__(
            self, "beta",
            _bounded_nonnegative_float(self.beta, "beta", maximum_inclusive=_MAX_BETA),
        )
        object.__setattr__(self, "lora_dropout", _dropout(self.lora_dropout))
        object.__setattr__(
            self,
            "lora_target_modules",
            _canonical_items(
                self.lora_target_modules, "lora_target_modules",
                maximum_items=_MAX_TARGET_MODULES,
            ),
        )
        for field in ("token_faithful", "use_vllm", "allow_transformers_rollout_func"):
            object.__setattr__(self, field, _exact_bool(getattr(self, field), field))
        backend = _text(self.env_backend, "env_backend", maximum_bytes=_MAX_ITEM_BYTES)
        if backend not in _ENV_GRPO_BACKENDS:
            raise ValueError("env_backend is unsupported")
        policy = _text(
            self.context_token_policy, "context_token_policy", maximum_bytes=_MAX_ITEM_BYTES
        )
        if policy not in _ENV_GRPO_CONTEXT_TOKEN_POLICIES:
            raise ValueError("context_token_policy is unsupported")
        if self.use_vllm:
            raise ValueError("env-GRPO requires transformers rollouts; use_vllm must be false")
        if not self.allow_transformers_rollout_func:
            raise ValueError("env-GRPO requires allow_transformers_rollout_func")
        object.__setattr__(
            self, "reward_config_ref", _logical_ref(self.reward_config_ref, "reward_config_ref")
        )
        digest = _text(self.reward_config_digest, "reward_config_digest", maximum_bytes=64)
        if _SHA256_HEX.fullmatch(digest) is None:
            raise ValueError("reward_config_digest must be a lowercase SHA-256 hex digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _ENV_GRPO_SCHEMA,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "num_generations": self.num_generations,
            "max_completion_length": self.max_completion_length,
            "temperature": self.temperature,
            "beta": self.beta,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
            "env_backend": self.env_backend,
            "max_turns": self.max_turns,
            "max_tool_steps": self.max_tool_steps,
            "token_faithful": self.token_faithful,
            "context_token_policy": self.context_token_policy,
            "reward_config_ref": self.reward_config_ref,
            "reward_config_digest": self.reward_config_digest,
            "use_vllm": self.use_vllm,
            "allow_transformers_rollout_func": self.allow_transformers_rollout_func,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "EnvGRPOHyperparametersV1":
        names = tuple(field for field in cls.__dataclass_fields__)
        value = _fields(value, frozenset(names) | {"schema_version"}, "hyperparameters")
        if value["schema_version"] != _ENV_GRPO_SCHEMA:
            raise ValueError("hyperparameters schema is unsupported")
        return cls(**{name: value[name] for name in names})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TrainingArtifactRequirementsV1:
    required_kinds: tuple[str, ...]
    retain_checkpoints: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_kinds",
            _canonical_items(
                self.required_kinds,
                "required_kinds",
                maximum_items=_MAX_REQUIRED_KINDS,
            ),
        )
        object.__setattr__(
            self,
            "retain_checkpoints",
            _exact_bool(self.retain_checkpoints, "retain_checkpoints"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "required_kinds": list(self.required_kinds),
            "retain_checkpoints": self.retain_checkpoints,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainingArtifactRequirementsV1":
        value = _fields(
            value, frozenset({"required_kinds", "retain_checkpoints"}), "artifacts"
        )
        return cls(value["required_kinds"], value["retain_checkpoints"])  # type: ignore[arg-type]


class _DuplicateJSONKey(ValueError):
    pass


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey
        result[key] = value
    return result


def _reject_constant(_value: str) -> object:
    raise ValueError


def validate_chat_template_kwargs(value: object) -> dict[str, object]:
    """Snapshot bounded, finite JSON kwargs without assuming a model template."""
    if type(value) is not dict or not value:
        raise ValueError("chat_template_kwargs must be a nonempty object")
    nodes = 0

    def check(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 64 or depth > 4:
            raise ValueError("chat_template_kwargs exceeds its structure limit")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str or not key:
                    raise ValueError("chat_template_kwargs contains an invalid key")
                try:
                    key_size = len(key.encode("utf-8"))
                except UnicodeEncodeError:
                    raise ValueError("chat_template_kwargs contains an invalid key") from None
                if key_size > 128:
                    raise ValueError("chat_template_kwargs contains an invalid key")
                check(child, depth + 1)
        elif type(item) is list:
            for child in item:
                check(child, depth + 1)
        elif type(item) is str:
            try:
                size = len(item.encode("utf-8"))
            except UnicodeEncodeError:
                raise ValueError("chat_template_kwargs contains invalid text") from None
            if size > 1024:
                raise ValueError("chat_template_kwargs contains an oversized string")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("chat_template_kwargs must be finite JSON")
        elif type(item) not in (int, bool, type(None)):
            raise TypeError("chat_template_kwargs must contain only JSON values")

    if set(value) & _RESERVED_CHAT_TEMPLATE_KWARGS:
        raise ValueError("chat_template_kwargs overrides renderer controls")
    check(value, 0)
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        raise ValueError("chat_template_kwargs must be valid JSON") from None
    if len(encoded) > _MAX_CHAT_TEMPLATE_KWARGS_BYTES:
        raise ValueError("chat_template_kwargs exceeds its byte limit")
    return json.loads(encoded)


# Each method binds exactly one hyperparameter contract.
_HYPERPARAMETERS: dict[TrainingMethodV1, type] = {
    TrainingMethodV1.SFT: SFTTrainingHyperparametersV1,
    TrainingMethodV1.GRPO: EnvGRPOHyperparametersV1,
}


@dataclass(frozen=True, slots=True)
class TrainingInputV1:
    schema_version: str
    method: TrainingMethodV1
    model: TrainingModelInputV1
    dataset: TrainingDatasetInputV1
    hyperparameters: SFTTrainingHyperparametersV1 | EnvGRPOHyperparametersV1
    artifacts: TrainingArtifactRequirementsV1

    def __post_init__(self) -> None:
        if self.schema_version != _TRAINING_SCHEMA:
            raise ValueError("training input schema is unsupported")
        if type(self.method) is not TrainingMethodV1:
            raise TypeError("method must be exact TrainingMethodV1")
        expected = (
            (self.model, TrainingModelInputV1, "model"),
            (self.dataset, TrainingDatasetInputV1, "dataset"),
            (self.hyperparameters, _HYPERPARAMETERS[self.method], "hyperparameters"),
            (self.artifacts, TrainingArtifactRequirementsV1, "artifacts"),
        )
        for value, expected_type, field in expected:
            if type(value) is not expected_type:
                raise TypeError(f"{field} has an invalid exact type")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": self.method.value,
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "artifacts": self.artifacts.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainingInputV1":
        value = _fields(
            value,
            frozenset(
                {"schema_version", "method", "model", "dataset", "hyperparameters", "artifacts"}
            ),
            "training_input",
        )
        if value["schema_version"] != _TRAINING_SCHEMA:
            raise ValueError("training input schema is unsupported")
        method = next(
            (item for item in TrainingMethodV1 if item.value == value["method"]), None
        )
        if type(value["method"]) is not str or method is None:
            raise ValueError("training method is unsupported")
        nested = {}
        for field in ("model", "dataset", "hyperparameters", "artifacts"):
            item = value[field]
            if type(item) is not dict:
                raise TypeError(f"{field} must be an object")
            nested[field] = item
        return cls(
            schema_version=_TRAINING_SCHEMA,
            method=method,
            model=TrainingModelInputV1.from_dict(nested["model"]),
            dataset=TrainingDatasetInputV1.from_dict(nested["dataset"]),
            hyperparameters=_HYPERPARAMETERS[method].from_dict(nested["hyperparameters"]),
            artifacts=TrainingArtifactRequirementsV1.from_dict(nested["artifacts"]),
        )

    @classmethod
    def from_json(cls, value: str) -> "TrainingInputV1":
        if type(value) is not str:
            raise TypeError("training input JSON must be a string")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("training input JSON is malformed") from None
        if len(encoded) > _MAX_JSON_BYTES:
            raise ValueError("training input JSON exceeds its size limit")
        try:
            document = json.loads(
                value,
                object_pairs_hook=_json_object,
                parse_constant=_reject_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("training input JSON is malformed") from None
        if type(document) is not dict:
            raise TypeError("training input JSON must encode an object")
        try:
            return cls.from_dict(document)
        except (TypeError, ValueError) as error:
            raise type(error)(str(error)) from None

    def canonical_bytes(self) -> bytes:
        try:
            return json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError):
            raise ValueError("training input cannot be canonicalized") from None

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    def input_digest(self) -> str:
        return contract_digest(_TRAINING_SCHEMA, self.to_dict())


__all__ = [
    "EnvGRPOHyperparametersV1",
    "SFTTrainingHyperparametersV1",
    "TrainingArtifactRequirementsV1",
    "TrainingDatasetInputV1",
    "TrainingDurationV1",
    "TrainingInputV1",
    "TrainingMethodV1",
    "TrainingModelInputV1",
]
