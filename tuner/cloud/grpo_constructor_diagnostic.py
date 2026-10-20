"""Provider-free GRPO constructor input and pinned model metadata admission.

Execution belongs to the separate Modal runtime qualification boundary. This
module never imports the ML stack or downloads model files.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from tuner.cloud.derived_training_image import DerivedTrainingImageError, _read_object
from tuner.runtime.packaged_worker_closure import stable_read


INPUT_SCHEMA = "synaptic-grpo-constructor-probe-input/v3"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def derive_probe_input(*, recipe_path: Path, plan_ready_path: Path,
                       dataset_path: Path, profiles_root: Path | None = None) -> bytes:
    """Retain the normal compiler's four rows and the original JSONL bytes."""
    try:
        plan_object, plan_raw = _read_object(plan_ready_path)
        plan = plan_object["data"]
        recipe, recipe_raw = _read_object(recipe_path)
        if len(plan_raw) > 65536 or len(recipe_raw) > 65536:
            raise ValueError
        dataset_raw = stable_read(dataset_path, 65536)
        from synaptic_tuner.api.v1.training_input import TrainingInputV1
        from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
        from tuner.training.grpo_compilation import compile_grpo_input, admit_grpo_snapshot

        if (recipe["training_input"] != plan["training_input"]
                or recipe["job"]["runtime_profile"] != plan["runtime_profile"]
                or plan["resource_request"] != {
                    "accelerator": recipe["job"]["accelerator"],
                    "accelerator_count": recipe["job"]["accelerator_count"],
                    "timeout_seconds": recipe["job"]["timeout_seconds"]}
                or plan["limits"] != recipe["limits"]):
            raise ValueError
        if profiles_root is not None:
            from tuner.runtime_profiles import load_runtime_profile
            profile = load_runtime_profile(plan["runtime_profile"], profiles_root)
            if profile.profile_sha256 != plan["profile_sha256"]:
                raise ValueError
            model = plan["training_input"]["model"]
            profile.resolve(model=model["ref"], model_revision=model["revision"], method="grpo")
        training = TrainingInputV1.from_dict(plan["training_input"])
        prepared = PreparedTrainingInputIdentity.from_dict(plan["prepared_dataset"])
        compiled = compile_grpo_input(training, prepared)
        _, rows, _, _ = admit_grpo_snapshot(training, prepared, dataset_raw)
        if compiled.fingerprint != plan["core_workload_digest"] or len(rows) != 4:
            raise ValueError
        result = _canonical({
            "schema_version": INPUT_SCHEMA,
            "workload": compiled.document,
            "dataset_base64": base64.b64encode(dataset_raw).decode("ascii"),
            "qualification_policy": recipe["limits"]["qualification_evidence"],
            "row_count": len(rows),
            "runtime_profile": plan["runtime_profile"],
            "profile_sha256": plan["profile_sha256"],
            "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
            "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "dataset_sha256": hashlib.sha256(dataset_raw).hexdigest(),
        })
        if not 0 < len(result) <= 65536:
            raise ValueError
        return result
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            DerivedTrainingImageError) as exc:
        raise DerivedTrainingImageError("PROBE_INPUT_INVALID") from exc


def verify_probe_input(raw: bytes) -> dict[str, object]:
    """Validate the exact projection before giving it any execution authority."""
    try:
        from tuner.runtime.packaged_grpo_constructor_child import verify_probe_input as verify
        return verify(raw)
    except (ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise DerivedTrainingImageError("PROBE_INPUT_INVALID") from exc


def probe_input_sha256(raw: bytes) -> str:
    verify_probe_input(raw)
    return hashlib.sha256(raw).hexdigest()
