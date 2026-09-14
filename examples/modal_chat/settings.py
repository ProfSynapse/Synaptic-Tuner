"""Strict declarative settings for the standalone Modal chat example."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from pathlib import PurePosixPath

from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    exact_fields,
    safe_ref,
)
from tuner.execution.providers.modal.config import (
    ModalProviderProfileV1,
    ModalRuntimeLockV1,
)
from tuner.execution.providers.modal.deployment_v1 import ModalDeploymentSpecV1
from tuner.project.secrets import reject_literal_secrets

from .remote import ModalChatGitRemote

_SCHEMA = "synaptic-modal-chat-example/v1"
_TOP = frozenset(
    {
        "schema_version",
        "attempt_ref",
        "environment_name",
        "dataset_project_path",
        "training_input",
        "load_in_4bit",
        "allowed_refs",
        "profile",
        "runtime_environment",
        "training_timeout_seconds",
        "maximum_training_cost_minor_units",
        "maximum_chat_cost_minor_units",
        "log_terminal_policy",
        "reviewed_runtime",
        "inference",
        "prompt",
    }
)
_INFERENCE = {
    "provider": frozenset({"provider_id", "profile_ref"}),
    "application": frozenset(
        {"app_name", "app_ref", "sandbox_entrypoint", "worker_ref"}
    ),
    "resources": frozenset(
        {
            "accelerator",
            "accelerator_count",
            "cpu_millicores",
            "memory_mb",
            "service_port",
            "provider_timeout_seconds",
            "provider_idle_timeout_seconds",
            "max_retries",
        }
    ),
    "serving": frozenset(
        {
            "served_model_name",
            "gpu_memory_utilization_milli",
            "enforce_eager",
            "tokenizer_mode",
            "max_lora_rank",
            "readiness_request_timeout_milliseconds",
            "max_tokens",
            "temperature_milli",
            "top_p_milli",
        }
    ),
    "policy": frozenset(
        {
            "startup_timeout_seconds",
            "request_timeout_seconds",
            "idle_timeout_seconds",
            "absolute_lifetime_seconds",
            "max_turns",
            "max_history_bytes",
            "max_request_bytes",
            "max_response_bytes",
        }
    ),
}
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_CREDENTIAL_KEYS = frozenset({"HF_TOKEN", "SYNAPTIC_EVIDENCE_MAC_KEY"})


class ModalChatSettings:
    """Parser-minted immutable settings retaining canonical owned bytes."""

    __slots__ = ("_raw", "_training", "_profile")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("ModalChatSettings is final")

    def __init__(self, *args, **kwargs):
        raise TypeError("settings are parser minted")

    def __setattr__(self, name, value):
        raise AttributeError("settings are immutable")

    @classmethod
    def build(cls, document: dict[str, object]) -> "ModalChatSettings":
        return cls.parse(canonical_bytes(document))

    @classmethod
    def parse(cls, raw: bytes) -> "ModalChatSettings":
        if type(raw) is not bytes or not raw or len(raw) > 256 * 1024:
            raise ValueError("settings exceed their bound")
        try:

            def closed_pairs(pairs):
                result = {}
                for key, item in pairs:
                    if key in result:
                        raise ValueError("duplicate settings field")
                    result[key] = item
                return result

            document = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=closed_pairs,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("invalid number")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise ValueError("Modal chat settings are malformed") from None
        if type(document) is not dict or canonical_bytes(document) != raw:
            raise ValueError("Modal chat settings must be canonical")
        exact_fields(document, _TOP, "Modal chat settings")
        if (
            document["schema_version"] != _SCHEMA
            or document["environment_name"] != "synaptic-smoke-v1"
        ):
            raise ValueError("unsupported Modal chat settings")
        safe_ref(document["attempt_ref"], "attempt_ref")
        cls._path(document["dataset_project_path"], "dataset_project_path")
        if type(document["load_in_4bit"]) is not bool:
            raise TypeError("load_in_4bit must be an exact boolean")
        training = TrainingInputV1.from_dict(deepcopy(document["training_input"]))
        if training.dataset.ref != "project://" + document["dataset_project_path"]:
            raise ValueError("training dataset does not match dataset_project_path")
        if (
            _REVISION.fullmatch(training.model.revision) is None
            or _REVISION.fullmatch(training.model.tokenizer_revision) is None
        ):
            raise ValueError("model and tokenizer revisions must be pinned")
        refs = document["allowed_refs"]
        if type(refs) is not list or len(refs) != 2:
            raise ValueError("exact consumer and engine refs are required")
        pairs = []
        for item in refs:
            exact_fields(item, frozenset({"url", "ref"}), "allowed ref")
            pairs.append((item["url"], item["ref"]))
        if len(set(pairs)) != 2:
            raise ValueError("consumer and engine refs must be distinct")
        ModalChatGitRemote(frozenset(pairs))
        profile = ModalProviderProfileV1.from_mapping(document["profile"])
        if (
            len(profile.secrets) != 1
            or set(profile.secrets[0].required_keys) != _CREDENTIAL_KEYS
        ):
            raise ValueError("Modal credential key names are not exact")
        environment = document["runtime_environment"]
        if (
            type(environment) is not dict
            or not environment
            or any(
                type(key) is not str or not key or type(value) is not str or not value
                for key, value in environment.items()
            )
        ):
            raise ValueError("runtime_environment must be a nonempty string map")
        reject_literal_secrets(
            {"runtime_environment": environment, "inference": document["inference"]}
        )
        for name in (
            "training_timeout_seconds",
            "maximum_training_cost_minor_units",
            "maximum_chat_cost_minor_units",
        ):
            value = document[name]
            maximum = 3600 if name == "training_timeout_seconds" else 1000
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} is outside its bound")
        lock = ModalRuntimeLockV1.packaged()
        secret = profile.secrets[0]
        ModalDeploymentSpecV1(
            profile.deployment_ref,
            profile.function_name,
            lock.registry_reference,
            profile.control_volume_ref,
            profile.artifact_volume_ref,
            secret.name,
            secret.required_keys,
            environment,
            document["training_timeout_seconds"],
        )
        cls._log_policy(document["log_terminal_policy"])
        reviewed = document["reviewed_runtime"]
        exact_fields(reviewed, frozenset({"capture", "readback"}), "reviewed_runtime")
        for name in ("capture", "readback"):
            item = reviewed[name]
            exact_fields(item, frozenset({"path", "sha256"}), name)
            cls._path(item["path"], f"{name}.path")
            digest_text(item["sha256"], f"{name}.sha256")
        cls._inference(document["inference"])
        prompt = document["prompt"]
        if (
            type(prompt) is not str
            or not prompt.strip()
            or prompt != prompt.strip()
            or len(prompt.encode("utf-8")) > 16 * 1024
        ):
            raise ValueError("prompt is invalid")
        value = object.__new__(cls)
        object.__setattr__(value, "_raw", bytes(raw))
        object.__setattr__(
            value, "_training", TrainingInputV1.from_json(training.canonical_json())
        )
        object.__setattr__(value, "_profile", profile)
        return value

    @staticmethod
    def _path(value: object, name: str) -> None:
        if type(value) is not str or not value or "\\" in value or "\0" in value:
            raise ValueError(f"{name} is invalid")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"{name} must be a canonical relative path")

    @staticmethod
    def _log_policy(value: object) -> None:
        fields = frozenset(
            {
                "schema_version",
                "generation",
                "max_log_chunks",
                "max_chunk_bytes",
                "max_terminal_bytes",
            }
        )
        exact_fields(value, fields, "log_terminal_policy")
        if value["schema_version"] != "synaptic-modal-log-terminal-policy/v2":
            raise ValueError("unsupported log terminal policy")
        for name, maximum in (
            ("generation", 2**31 - 1),
            ("max_log_chunks", 1_000_000),
            ("max_chunk_bytes", 1_048_576),
            ("max_terminal_bytes", 1_048_576),
        ):
            if type(value[name]) is not int or not 1 <= value[name] <= maximum:
                raise ValueError(f"invalid {name}")

    @staticmethod
    def _inference(value: object) -> None:
        exact_fields(value, frozenset(_INFERENCE), "inference")
        for name, fields in _INFERENCE.items():
            section = value[name]
            exact_fields(section, fields, f"inference.{name}")
        if (
            value["provider"]["provider_id"] != "modal"
            or value["application"]["app_name"] != "synaptic-training-v1"
        ):
            raise ValueError("unsupported inference provider or application")
        if value["resources"]["max_retries"] != 0:
            raise ValueError("inference retries must be disabled")
        resources = value["resources"]
        if (
            resources["accelerator"] != "A10"
            or resources["accelerator_count"] != 1
            or resources["cpu_millicores"] != 4000
            or resources["memory_mb"] != 16384
            or not 1 <= resources["provider_timeout_seconds"] <= 900
            or resources["provider_idle_timeout_seconds"]
            > resources["provider_timeout_seconds"]
            or not 1 <= resources["service_port"] <= 65535
        ):
            raise ValueError(
                "inference resources differ from the selected smoke bounds"
            )
        if type(value["serving"]["enforce_eager"]) is not bool:
            raise TypeError("enforce_eager must be an exact boolean")
        if value["serving"]["tokenizer_mode"] not in (None, "mistral"):
            raise ValueError("invalid tokenizer_mode")
        serving_bounds = {
            "gpu_memory_utilization_milli": (10, 1000),
            "max_lora_rank": (1, 1024),
            "readiness_request_timeout_milliseconds": (10, 30000),
            "max_tokens": (1, 32768),
            "temperature_milli": (0, 2000),
            "top_p_milli": (1, 1000),
        }
        for name, (minimum, maximum) in serving_bounds.items():
            item = value["serving"][name]
            if type(item) is not int or not minimum <= item <= maximum:
                raise ValueError(f"invalid inference {name}")
        for section_name in ("resources", "policy"):
            for name, item in value[section_name].items():
                if name == "accelerator":
                    safe_ref(item, name)
                elif type(item) is not int or item < (
                    0 if name == "max_retries" else 1
                ):
                    raise ValueError(f"invalid inference {name}")

    def _document(self) -> dict[str, object]:
        return json.loads(self._raw)

    @property
    def canonical_bytes(self) -> bytes:
        return bytes(self._raw)

    @property
    def training_input(self) -> TrainingInputV1:
        return TrainingInputV1.from_json(self._training.canonical_json())

    @property
    def training_json(self) -> str:
        return self._training.canonical_json()

    @property
    def profile(self) -> ModalProviderProfileV1:
        return self._profile

    def __getattr__(self, name: str):
        if name in _TOP - {"schema_version", "training_input", "profile"}:
            return deepcopy(self._document()[name])
        raise AttributeError(name)


__all__ = ["ModalChatSettings"]
