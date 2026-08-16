"""Declarative built-in capabilities; importing this module executes nothing."""

from __future__ import annotations

from synaptic_tuner.api.v1 import CapabilityDescriptor


def _effects(*, write: bool, network: bool, gpu: str, paid: bool | str = False, publish: bool = False) -> dict[str, object]:
    return {
        "filesystem_write": write,
        "network": network,
        "gpu": gpu,
        "paid_compute": paid,
        "external_publish": publish,
    }


def _supports(*, dry_run: bool = False, json_result: bool = False, jsonl_events: bool = False, available: bool = True) -> dict[str, bool]:
    return {
        "available": available,
        "dry_run": dry_run,
        "json_result": json_result,
        "jsonl_events": jsonl_events,
    }


def builtin_descriptors() -> tuple[CapabilityDescriptor, ...]:
    """Return current command metadata without importing command handlers."""

    return (
        CapabilityDescriptor(
            id="training.local-run",
            summary="Run a config-defined training or evaluation job in local Docker.",
            command=("local-run",),
            inputs=(
                {"name": "job_config", "type": "path", "access": "read", "required": True},
            ),
            outputs=(
                {"kind": "run_artifacts", "root": "artifact"},
            ),
            effects=_effects(write=True, network=True, gpu="required"),
            confirmation={"required": True, "reason": "gpu"},
            resumable=True,
            supports=_supports(dry_run=True),
        ),
        CapabilityDescriptor(
            id="experiment.run",
            summary="Run a config-defined train, evaluation, loss, analysis, and recommendation experiment.",
            command=("run-experiment",),
            inputs=(
                {"name": "experiment_spec", "type": "path", "access": "read", "required": True},
            ),
            outputs=(
                {"kind": "experiment_bundle", "root": "tracking"},
                {"kind": "stage_artifacts", "root": "artifact"},
            ),
            effects=_effects(write=True, network=True, gpu="required", paid="possible"),
            confirmation={"required": True, "reason": "gpu_or_paid_compute"},
            resumable=True,
            supports=_supports(),
        ),
        CapabilityDescriptor(
            id="mechinterp.steer",
            summary="Run a config-defined smoke-gated activation intervention cell.",
            command=("mechinterp", "steer"),
            inputs=(
                {"name": "config", "type": "path", "access": "read", "required": True},
                {"name": "model", "type": "string", "required": True},
            ),
            outputs=(
                {"kind": "rows_jsonl", "root": "artifact"},
                {"kind": "smoke_state", "root": "state"},
            ),
            effects=_effects(write=True, network=True, gpu="required", paid="possible"),
            confirmation={"required": True, "reason": "gpu"},
            resumable=True,
            supports=_supports(),
        ),
        CapabilityDescriptor(
            id="evaluation.run",
            summary="Evaluate a model with configured scenarios, rubrics, and backends.",
            command=("eval",),
            inputs=(),
            outputs=(
                {"kind": "evaluation_results", "root": "artifact"},
            ),
            effects=_effects(write=True, network=True, gpu="optional", paid="possible"),
            confirmation={"required": True, "reason": "evaluation_execution"},
            resumable=False,
            supports=_supports(),
        ),
        CapabilityDescriptor(
            id="generation.batch",
            summary="Generate a crash-safe JSONL completion batch from configured prompts and inference settings.",
            command=("batch-generate",),
            inputs=(
                {"name": "prompts", "type": "path", "access": "read", "required": True},
                {"name": "model", "type": "string", "required": True},
                {"name": "out_dir", "type": "path", "access": "write", "required": True},
                {"name": "json_schema", "type": "path", "access": "read", "required": False},
            ),
            outputs=(
                {"kind": "completions_jsonl", "root": "artifact"},
                {"kind": "generation_provenance", "root": "artifact"},
            ),
            effects=_effects(write=True, network=True, gpu="optional"),
            confirmation={"required": False},
            resumable=True,
            supports=_supports(),
        ),
        CapabilityDescriptor(
            id="cloud.launch",
            summary="Submit a config-defined training job to Hugging Face Jobs.",
            command=("cloud-run",),
            inputs=(
                {"name": "job_config", "type": "path", "access": "read", "required": True},
            ),
            outputs=(
                {"kind": "cloud_run", "root": "external"},
            ),
            effects=_effects(write=True, network=True, gpu="required", paid="required"),
            confirmation={"required": True, "reason": "paid_compute"},
            resumable=False,
            supports=_supports(available=False),
        ),
        CapabilityDescriptor(
            id="cloud.inspect",
            summary="Inspect existing Hugging Face cloud evaluation artifacts without launching compute.",
            command=("cloud-inspect",),
            inputs=(
                {"name": "run", "type": "string", "required": False},
                {"name": "eval_run", "type": "string", "required": False},
            ),
            outputs=(
                {"kind": "evaluation_inspection", "root": "stdout"},
            ),
            effects=_effects(write=True, network=True, gpu="none"),
            confirmation={"required": False},
            resumable=False,
            supports=_supports(),
        ),
    )


__all__ = ["builtin_descriptors"]
