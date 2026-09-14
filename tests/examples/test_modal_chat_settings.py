from copy import deepcopy

import pytest

from examples.modal_chat.settings import ModalChatSettings
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.deployment_identity import modal_function_name


def _document():
    return {
        "schema_version": "synaptic-modal-chat-example/v1",
        "attempt_ref": "attempt-1",
        "environment_name": "synaptic-smoke-v1",
        "dataset_project_path": "data/train.jsonl",
        "load_in_4bit": False,
        "training_input": {
            "schema_version": "synaptic-training-input/v1",
            "method": "sft",
            "model": {
                "ref": "org/model",
                "revision": "c" * 40,
                "tokenizer_revision": "d" * 40,
            },
            "dataset": {"ref": "project://data/train.jsonl"},
            "hyperparameters": {
                "schema_version": "synaptic-sft-hyperparameters/v1",
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0002,
                "duration": {"max_steps": 1, "num_epochs": None},
                "max_seq_length": 128,
                "seed": 1,
                "save_steps": 1,
                "save_total_limit": 1,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_modules": ["q_proj"],
                "use_dora": False,
                "use_rslora": False,
                "init_lora_weights": True,
                "split_dataset": False,
            },
            "artifacts": {
                "required_kinds": ["final_model", "training_lineage"],
                "retain_checkpoints": False,
            },
        },
        "allowed_refs": [
            {"url": "https://github.com/acme/consumer.git", "ref": "refs/heads/main"},
            {"url": "https://github.com/acme/engine.git", "ref": "refs/heads/main"},
        ],
        "profile": {
            "schema_version": "synaptic-modal-provider/v1",
            "profile": "smoke",
            "deployment": {
                "app_name": "synaptic-training-v1",
                "function_name": modal_function_name("modal-deployment-" + "1" * 32),
                "deployment_ref": "modal-deployment-" + "1" * 32,
            },
            "runtime_lock": "engine://tuner/execution/providers/modal/modal-runtime-v1.lock.json",
            "volumes": {"control_ref": "control", "artifact_ref": "artifacts"},
            "secrets": [
                {
                    "provider": "modal",
                    "name": "runtime-secret",
                    "required_keys": ["HF_TOKEN", "SYNAPTIC_EVIDENCE_MAC_KEY"],
                }
            ],
        },
        "runtime_environment": {
            "LANG": "C.UTF-8",
            "PATH": "/opt/conda/bin:/usr/bin:/bin",
        },
        "training_timeout_seconds": 3600,
        "maximum_training_cost_minor_units": 1000,
        "maximum_chat_cost_minor_units": 1000,
        "log_terminal_policy": {
            "schema_version": "synaptic-modal-log-terminal-policy/v2",
            "generation": 1,
            "max_log_chunks": 10,
            "max_chunk_bytes": 1024,
            "max_terminal_bytes": 1024,
        },
        "reviewed_runtime": {
            "capture": {"path": "docs/review/capture.json", "sha256": "a" * 64},
            "readback": {"path": "docs/review/readback.json", "sha256": "b" * 64},
        },
        "inference": {
            "provider": {"provider_id": "modal", "profile_ref": "smoke"},
            "application": {
                "app_name": "synaptic-training-v1",
                "app_ref": "app",
                "sandbox_entrypoint": "serve",
                "worker_ref": "worker",
            },
            "resources": {
                "accelerator": "A10",
                "accelerator_count": 1,
                "cpu_millicores": 4000,
                "memory_mb": 16384,
                "service_port": 8000,
                "provider_timeout_seconds": 600,
                "provider_idle_timeout_seconds": 300,
                "max_retries": 0,
            },
            "serving": {
                "served_model_name": "trained",
                "gpu_memory_utilization_milli": 900,
                "enforce_eager": True,
                "tokenizer_mode": None,
                "max_lora_rank": 64,
                "readiness_request_timeout_milliseconds": 1000,
                "max_tokens": 32,
                "temperature_milli": 0,
                "top_p_milli": 1000,
            },
            "policy": {
                "startup_timeout_seconds": 300,
                "request_timeout_seconds": 30,
                "idle_timeout_seconds": 300,
                "absolute_lifetime_seconds": 600,
                "max_turns": 2,
                "max_history_bytes": 4096,
                "max_request_bytes": 4096,
                "max_response_bytes": 4096,
            },
        },
        "prompt": "Say hello.",
    }


def test_settings_parse_real_contracts_and_return_owned_values():
    document = _document()
    settings = ModalChatSettings.parse(canonical_bytes(document))
    assert settings.training_input.model.revision == "c" * 40
    assert settings.training_json == settings.training_input.canonical_json()
    assert settings.profile.app_name == "synaptic-training-v1"
    inference = settings.inference
    inference["provider"]["provider_id"] = "changed"
    assert settings.inference["provider"]["provider_id"] == "modal"
    assert settings.canonical_bytes == canonical_bytes(document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(project_root="/tmp"),
        lambda value: value.update(environment_name="other"),
        lambda value: value.update(dataset_project_path="../train.jsonl"),
        lambda value: value["training_input"]["model"].update(revision="main"),
        lambda value: value["training_input"]["dataset"].update(
            ref="project://data/other.jsonl"
        ),
        lambda value: value["profile"]["secrets"][0].update(required_keys=["HF_TOKEN"]),
        lambda value: value["profile"]["secrets"].append(
            {
                "provider": "modal",
                "name": "other",
                "required_keys": ["HF_TOKEN", "SYNAPTIC_EVIDENCE_MAC_KEY"],
            }
        ),
        lambda value: value["runtime_environment"].update(HF_TOKEN="literal"),
        lambda value: value.update(maximum_chat_cost_minor_units=1001),
        lambda value: value["inference"]["resources"].update(cpu_millicores=8000),
        lambda value: value["inference"]["resources"].update(
            provider_timeout_seconds=901
        ),
    ],
)
def test_settings_reject_invalid_or_secret_configuration(mutation):
    document = _document()
    mutation(document)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        ModalChatSettings.build(document)


def test_settings_reject_duplicate_json_keys():
    raw = b'{"schema_version":"synaptic-modal-chat-example/v1","schema_version":"synaptic-modal-chat-example/v1"}'
    with pytest.raises(ValueError):
        ModalChatSettings.parse(raw)
