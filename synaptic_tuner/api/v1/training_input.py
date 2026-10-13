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


@dataclass(frozen=True, slots=True)
class GRPOTrainingHyperparametersV1:
    batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_steps: int
    max_seq_length: int
    max_prompt_length: int
    max_completion_length: int
    learning_rate: float
    temperature: float
    beta: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    seed: int
    logging_steps: int
    save_steps: int
    save_total_limit: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    use_dora: bool
    use_rslora: bool
    load_in_4bit: bool
    bf16: bool
    fp16: bool
    prompt_column: str
    chat_template_kwargs: dict[str, object] | None = None

    def __post_init__(self) -> None:
        bounds = {
            "batch_size": 4096, "gradient_accumulation_steps": 4096,
            "num_generations": 4096, "max_steps": 100000,
            "max_seq_length": 100000, "max_prompt_length": 100000,
            "max_completion_length": 100000, "logging_steps": 100000,
            "save_steps": 100000, "save_total_limit": 100000,
            "lora_rank": 4096, "lora_alpha": 65536, "seed": _MAX_SEED,
        }
        for name, maximum in bounds.items():
            minimum = 0 if name == "seed" else 2 if name == "num_generations" else 1
            object.__setattr__(self, name, _exact_integer(
                getattr(self, name), name, minimum=minimum, maximum=maximum))
        if self.batch_size * self.gradient_accumulation_steps % self.num_generations:
            raise ValueError("effective batch must be divisible by num_generations")
        if self.max_prompt_length + self.max_completion_length > self.max_seq_length:
            raise ValueError("prompt and completion exceed sequence length")
        for name, maximum in (("learning_rate", 1), ("temperature", 10),
                              ("max_grad_norm", 100)):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name,
                minimum_exclusive=0, maximum_inclusive=maximum))
        for name in ("beta", "weight_decay", "warmup_ratio"):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a number")
            try:
                value = float(value)
            except OverflowError:
                raise ValueError(f"{name} is outside its allowed range") from None
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} is outside its allowed range")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "lora_dropout", _dropout(self.lora_dropout))
        object.__setattr__(self, "lora_target_modules", _canonical_items(
            self.lora_target_modules, "lora_target_modules", maximum_items=256))
        object.__setattr__(self, "prompt_column", _text(
            self.prompt_column, "prompt_column", maximum_bytes=128))
        for name in ("use_dora", "use_rslora", "load_in_4bit", "bf16", "fp16"):
            object.__setattr__(self, name, _exact_bool(getattr(self, name), name))
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled")
        if self.chat_template_kwargs is not None:
            object.__setattr__(self, "chat_template_kwargs",
                               validate_chat_template_kwargs(self.chat_template_kwargs))

    def to_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__
                  if name != "chat_template_kwargs"}
        result["schema_version"] = "synaptic-grpo-hyperparameters/v1"
        result["lora_target_modules"] = list(self.lora_target_modules)
        if self.chat_template_kwargs is not None:
            result["chat_template_kwargs"] = validate_chat_template_kwargs(self.chat_template_kwargs)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "GRPOTrainingHyperparametersV1":
        names = set(cls.__dataclass_fields__) - {"chat_template_kwargs"}
        expected = names | {"schema_version"}
        if type(value) is dict and "chat_template_kwargs" in value:
            expected.add("chat_template_kwargs")
        snapshot = _fields(value, frozenset(expected), "grpo hyperparameters")
        if snapshot.pop("schema_version") != "synaptic-grpo-hyperparameters/v1":
            raise ValueError("GRPO hyperparameter schema is unsupported")
        return cls(**snapshot)


def validate_grpo_rewards(value: object) -> dict[str, object]:
    """Detach the first raw-response declarative reward lane."""
    value = _fields(value, frozenset({"items", "documents", "schema"}), "rewards")
    if type(value["schema"]) is not dict or value["schema"]:
        raise ValueError("raw GRPO rewards require an empty schema")
    items, documents = value["items"], value["documents"]
    if type(items) is not list or not 1 <= len(items) <= 32:
        raise ValueError("rewards.items exceeds its bounds")
    if type(documents) is not dict or any(type(k) is not str for k in documents):
        raise TypeError("rewards.documents must be an object")
    selected = set()
    positive = False
    for item in items:
        item = _fields(item, frozenset({"name", "weight"}), "reward item")
        name = _text(item["name"], "reward name", maximum_bytes=128)
        if name in selected:
            raise ValueError("duplicate reward name")
        selected.add(name)
        weight = item["weight"]
        if type(weight) not in (int, float) or not 0 <= weight <= 100:
            raise ValueError("reward weight is outside its bounds")
        positive |= weight > 0
    if not positive or set(documents) != selected:
        raise ValueError("reward documents must exactly match positive selections")
    for name, document in documents.items():
        required = {"name", "scope", "validation", "scoring"}
        allowed = required | {"type", "description"}
        if type(document) is not dict or any(type(k) is not str for k in document):
            raise TypeError("reward document must be an object")
        if not required <= set(document) <= allowed:
            raise ValueError("reward document has invalid fields")
        if document["name"] != name or document["scope"] != "response":
            raise ValueError("reward identity or scope is unsupported")
        if "type" in document and document["type"] != "rubric":
            raise ValueError("reward type is unsupported")
        if "description" in document:
            _text(document["description"], "reward description", maximum_bytes=4096)
        rules = document["validation"]
        if type(rules) is not list or not 1 <= len(rules) <= 32:
            raise ValueError("reward validation exceeds its bounds")
        for rule in rules:
            rule = _fields(rule, frozenset({"type", "match"}), "reward rule")
            if rule["type"] != "regex":
                raise ValueError("reward rule is unsupported")
            pattern = _text(rule["match"], "reward pattern", maximum_bytes=4096)
            try:
                re.compile(pattern)
            except re.error:
                raise ValueError("reward pattern is invalid") from None
        scoring = _fields(document["scoring"],
                          frozenset({"strategy", "on_pass", "on_fail"}), "scoring")
        if scoring["strategy"] != "binary":
            raise ValueError("reward scoring is unsupported")
        for key in ("on_pass", "on_fail"):
            number = scoring[key]
            if type(number) not in (int, float) or not -100 <= number <= 100:
                raise ValueError("reward score is outside its bounds")
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        raise ValueError("rewards must be finite JSON") from None
    if len(raw) > 65536:
        raise ValueError("rewards exceeds its byte bound")
    return json.loads(raw)


@dataclass(frozen=True, slots=True)
class TrainingInputV1:
    schema_version: str
    method: TrainingMethodV1
    model: TrainingModelInputV1
    dataset: TrainingDatasetInputV1
    hyperparameters: SFTTrainingHyperparametersV1 | GRPOTrainingHyperparametersV1
    artifacts: TrainingArtifactRequirementsV1
    rewards: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != _TRAINING_SCHEMA:
            raise ValueError("training input schema is unsupported")
        if type(self.method) is not TrainingMethodV1:
            raise TypeError("method must be exact TrainingMethodV1")
        expected = (
            (self.model, TrainingModelInputV1, "model"),
            (self.dataset, TrainingDatasetInputV1, "dataset"),
            (self.hyperparameters, SFTTrainingHyperparametersV1 if self.method is TrainingMethodV1.SFT
             else GRPOTrainingHyperparametersV1, "hyperparameters"),
            (self.artifacts, TrainingArtifactRequirementsV1, "artifacts"),
        )
        for value, expected_type, field in expected:
            if type(value) is not expected_type:
                raise TypeError(f"{field} has an invalid exact type")

        if self.method is TrainingMethodV1.SFT:
            if self.rewards is not None:
                raise ValueError("SFT does not accept GRPO rewards")
        else:
            object.__setattr__(self, "rewards", validate_grpo_rewards(self.rewards))

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema_version": self.schema_version,
            "method": self.method.value,
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "artifacts": self.artifacts.to_dict(),
        }
        if self.method is TrainingMethodV1.GRPO:
            result["rewards"] = validate_grpo_rewards(self.rewards)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainingInputV1":
        fields = {"schema_version", "method", "model", "dataset", "hyperparameters", "artifacts"}
        if type(value) is dict and value.get("method") == "grpo":
            fields.add("rewards")
        value = _fields(value, frozenset(fields), "training_input")
        if value["schema_version"] != _TRAINING_SCHEMA:
            raise ValueError("training input schema is unsupported")
        if value["method"] not in {method.value for method in TrainingMethodV1}:
            raise ValueError("training method is unsupported")
        nested = {}
        for field in ("model", "dataset", "hyperparameters", "artifacts"):
            item = value[field]
            if type(item) is not dict:
                raise TypeError(f"{field} must be an object")
            nested[field] = item
        return cls(
            schema_version=_TRAINING_SCHEMA,
            method=TrainingMethodV1(value["method"]),
            model=TrainingModelInputV1.from_dict(nested["model"]),
            dataset=TrainingDatasetInputV1.from_dict(nested["dataset"]),
            hyperparameters=(SFTTrainingHyperparametersV1 if value["method"] == "sft"
                             else GRPOTrainingHyperparametersV1).from_dict(nested["hyperparameters"]),
            artifacts=TrainingArtifactRequirementsV1.from_dict(nested["artifacts"]),
            rewards=value.get("rewards"),
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
    "SFTTrainingHyperparametersV1",
    "TrainingArtifactRequirementsV1",
    "TrainingDatasetInputV1",
    "TrainingDurationV1",
    "TrainingInputV1",
    "TrainingMethodV1",
    "TrainingModelInputV1",
]
