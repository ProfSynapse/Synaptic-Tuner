"""Orchestration for ``mechinterp weights-inspect`` / ``weights-patch``.

``validate`` = open the LOCAL checkpoint, verify the declared source binding and
the targeted base bytes, load/verify the payload, and plan the patch against the
checkpoint's own tensors (budget enforced on the realized result). Nothing is
written. ``export`` does the same and then writes the exact modified checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from MechInterp.weights.checkpoint import (
    LocalCheckpoint,
    inspect_checkpoint,
    open_checkpoint,
    verify_binding,
)
from MechInterp.weights.errors import WeightPatchError
from MechInterp.weights.export import export_checkpoint
from MechInterp.weights.patch import (
    PatchPlan,
    Payload,
    buffer_to_tensor,
    build_plan,
    load_payload,
)
from MechInterp.weights.safetensors_io import read_region
from MechInterp.weights.spec import (
    InspectRequest,
    WeightPatchSpec,
    load_inspect_request,
    load_patch_spec,
)


def inspect_from_config(path: str | Path, *, context=None) -> dict:
    request: InspectRequest = load_inspect_request(path, context=context)
    return inspect_checkpoint(request)


def _read_from_checkpoint(ckpt: LocalCheckpoint):
    def read_tensor(name: str):
        sf, entry = ckpt.entry(name)
        return buffer_to_tensor(read_region(sf, entry), entry.torch_dtype, entry.shape)

    return read_tensor


def plan_against_checkpoint(
    spec: WeightPatchSpec, spec_sha256: str
) -> tuple[LocalCheckpoint, dict, Payload, PatchPlan]:
    if not spec.base.checkpoint:
        raise WeightPatchError("schema", "base.checkpoint is required for weights-patch")
    ckpt = open_checkpoint(
        spec.base.checkpoint, allow_symlinked_files=spec.base.allow_symlinked_files
    )
    binding = verify_binding(spec, ckpt)
    payload = load_payload(spec)
    plan = build_plan(spec, spec_sha256, payload, _read_from_checkpoint(ckpt))
    return ckpt, binding, payload, plan


def validate_patch_config(path: str | Path, *, context=None) -> dict:
    spec, digest = load_patch_spec(path, context=context)
    _, binding, _, plan = plan_against_checkpoint(spec, digest)
    return {
        "mode": "validate",
        "written": False,
        "binding": binding,
        "budget": spec.budget.model_dump(mode="json"),
        "receipt": plan.receipt(),
    }


def export_patch_config(path: str | Path, *, context=None) -> dict:
    spec, digest = load_patch_spec(path, context=context)
    if spec.export is None:
        raise WeightPatchError("schema", "the spec has no export block (export.output_dir)")
    ckpt, binding, _, plan = plan_against_checkpoint(spec, digest)
    manifest = export_checkpoint(spec, digest, plan, ckpt, binding)
    return {
        "mode": "export",
        "written": True,
        "output_dir": spec.export.output_dir,
        "patch_digest": manifest["patch_digest"],
        "totals": manifest["totals"],
        "manifest": manifest,
    }
