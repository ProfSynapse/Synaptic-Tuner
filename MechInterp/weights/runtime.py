"""Transactional in-memory application of a weight patch to a loaded module.

This is the path ``steer`` / ``extract`` / ``dose-calibrate`` use: no checkpoint
is exported per candidate. The patch is planned against the loaded module's own
bytes (so a wrong base is refused by hash, whatever the model's config claims),
then committed all-or-nothing with the originals retained for exact revert.

    prepared = prepare_patch(ref_path, ref_sha256, context=ctx)
    check_declared_revision(prepared, revision=cli_revision)       # pre-load
    ...load the model...
    with patched(model, prepared) as applied:                      # reverts on exit
        ...
    applied = apply_patch(model, prepared)                          # keep applied
    applied.revert()                                                # exact undo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from MechInterp.weights.errors import WeightPatchError
from MechInterp.weights.patch import (
    PatchPlan,
    Payload,
    build_plan,
    load_payload,
    tensor_sha256,
)
from MechInterp.weights.spec import WeightPatchSpec, load_patch_spec


@dataclass
class PreparedPatch:
    """A spec whose file bytes and payload bytes have been verified (not yet applied)."""

    spec: WeightPatchSpec
    spec_sha256: str
    payload: Payload


def prepare_patch(path: str | Path, expected_sha256: str, *, context=None) -> PreparedPatch:
    """Load a patch spec, require its file sha256 to equal ``expected_sha256``,
    and verify + load its payload. Cheap; call before loading any model."""
    spec, digest = load_patch_spec(path, context=context)
    if digest != expected_sha256:
        raise WeightPatchError(
            "binding", "weight patch spec sha256 does not match the pinned weight_patch.sha256"
        )
    return PreparedPatch(spec, digest, load_payload(spec))


def check_declared_revision(prepared: PreparedPatch, revision: Optional[str]) -> None:
    """The run's declared model revision must equal the patch's base.revision."""
    declared = prepared.spec.base.revision
    if revision is None:
        raise WeightPatchError(
            "binding",
            f"the patch binds base.revision {declared}; pass --model-revision {declared} "
            "(an unpinned load cannot be tied to the patch's base)",
        )
    if revision != declared:
        raise WeightPatchError(
            "binding", f"declared model revision {revision!r} != patch base.revision {declared!r}"
        )


def check_loaded_commit(prepared: PreparedPatch, model) -> Optional[str]:
    """If the loaded model carries a resolved commit hash, it must match.

    This is corroboration only (transformers records ``config._commit_hash`` for
    Hub loads). The binding that matters is the targeted-tensor byte hash check
    performed when the patch is planned against the module.
    """
    commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    if isinstance(commit, str) and commit and commit != prepared.spec.base.revision:
        raise WeightPatchError(
            "binding",
            f"loaded model resolved commit {commit!r} != patch base.revision "
            f"{prepared.spec.base.revision!r}",
        )
    return commit if isinstance(commit, str) and commit else None


def runtime_name(spec: WeightPatchSpec, checkpoint_name: str) -> str:
    for rule in spec.base.runtime_prefix_rules:
        if checkpoint_name.startswith(rule.checkpoint_prefix):
            return rule.runtime_prefix + checkpoint_name[len(rule.checkpoint_prefix) :]
    return checkpoint_name


def _module_tensors(model) -> dict[str, torch.Tensor]:
    tensors = dict(model.named_parameters(remove_duplicate=False))
    for name, buf in model.named_buffers(remove_duplicate=False):
        tensors.setdefault(name, buf)
    return tensors


class AppliedPatch:
    """A committed patch with its originals retained for exact, verified revert."""

    def __init__(self, plan: PatchPlan, targets: dict[str, torch.Tensor], originals: dict[str, torch.Tensor]):
        self.plan = plan
        self._targets = targets
        self._originals = originals
        self.reverted = False

    @property
    def receipt(self) -> dict:
        return self.plan.receipt()

    def revert(self) -> None:
        if self.reverted:
            return
        with torch.no_grad():
            for name, original in self._originals.items():
                self._targets[name].copy_(original.to(self._targets[name].device))
        for name in self._originals:
            if tensor_sha256(self._targets[name]) != self.plan.targeted[name].before_sha256:
                raise WeightPatchError("apply", f"{name}: revert did not restore the original bytes")
        self._originals = {}
        self.reverted = True

    def __enter__(self) -> "AppliedPatch":
        return self

    def __exit__(self, *exc) -> None:
        self.revert()


def apply_patch(model, prepared: PreparedPatch) -> AppliedPatch:
    """Plan against the module's own bytes, then commit all-or-nothing."""
    spec = prepared.spec
    tensors = _module_tensors(model)
    targets: dict[str, torch.Tensor] = {}
    for name in spec.targeted_tensors():
        rname = runtime_name(spec, name)
        target = tensors.get(rname)
        if target is None:
            raise WeightPatchError("binding", f"loaded module has no tensor {rname!r} (for {name!r})")
        if target.device.type == "meta":
            raise WeightPatchError("binding", f"{rname!r} is on the meta device (not materialized)")
        if not target.is_floating_point():
            raise WeightPatchError("binding", f"{rname!r} is not a floating-point tensor")
        targets[name] = target

    # A write through one name must not modify an unbudgeted alias, including
    # overlapping views. Refuse shared storage conservatively (even disjoint views).
    for name, target in targets.items():
        target_storage = (target.device, target.untyped_storage().data_ptr())
        aliases = [other for other, tensor in tensors.items()
                   if other != runtime_name(spec, name) and tensor.device.type != "meta"
                   and tensor.numel() and target.numel()
                   and (tensor.device, tensor.untyped_storage().data_ptr()) == target_storage]
        if aliases:
            raise WeightPatchError("binding", f"{name}: target shares storage with another parameter or buffer")

    def read_tensor(name: str) -> torch.Tensor:
        return targets[name].detach().to("cpu").contiguous().clone()

    plan = build_plan(spec, prepared.spec_sha256, prepared.payload, read_tensor)

    originals: dict[str, torch.Tensor] = {}
    written: list[str] = []
    try:
        with torch.no_grad():
            for name, tp in plan.modified.items():
                originals[name] = targets[name].detach().clone()
                written.append(name)
                targets[name].copy_(tp.new_tensor.to(device=targets[name].device, dtype=targets[name].dtype))
        for name, tp in plan.modified.items():
            if tensor_sha256(targets[name]) != tp.after_sha256:
                raise WeightPatchError("apply", f"{name}: committed bytes differ from the plan")
    except BaseException:
        try:
            with torch.no_grad():
                for name in written:
                    targets[name].copy_(originals[name])
            for name in written:
                if tensor_sha256(targets[name]) != plan.targeted[name].before_sha256:
                    raise WeightPatchError("apply", f"{name}: rollback did not restore the original bytes")
        except BaseException as rollback_exc:
            raise WeightPatchError("apply", "patch failed and rollback verification failed") from rollback_exc
        raise
    # Free the planned copies; revert uses the retained originals.
    for tp in plan.targeted.values():
        tp.new_tensor = None
    return AppliedPatch(plan, targets, originals)


def patched(model, prepared: PreparedPatch) -> AppliedPatch:
    """Context manager form: ``with patched(model, prepared) as applied: ...``"""
    return apply_patch(model, prepared)
