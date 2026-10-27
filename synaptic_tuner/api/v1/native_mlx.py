"""Explicit native MLX composition for the public TrainingAPI and RunsAPI.

Import this optional module deliberately, as with ``api.v1.modal``. Imports do
not load ML frameworks, inspect hardware, open storage, or admit execution.
The host owns source/runtime/model preparation, qualification or restricted
admission, signing keys, command catalogs and durable record/stream ports.
The separate GRPO compiler/profile is an unqualified source candidate; SFT
qualification does not admit GRPO or its reward/objective/runtime selection.

Submission is bounded and synchronous on the host store's owning thread.
Known-run reads reuse the original bindings and cannot resubmit. Public
logs/cancel/reconcile are unavailable; an expired restricted start admission
does not prevent authenticated retained reads. A desktop should isolate its
host process and retain original state rather than signal a persisted PID.
"""

from tuner.runtime.native_files import NativeFile
from tuner.runtime.native_mlx_input import NativeModelSnapshot
from tuner.runtime.native_mlx_publication import NativeArtifactBudget
from tuner.runtime.native_process import NativeInvocation, NativeProcessLimits
from tuner.training.native_mlx_authority import NativeEvidenceAuthority, NativeMaterialAuthority
from tuner.training.native_mlx_compilation import NativeMLXProfile, compile_native_sft
from tuner.training.native_mlx_grpo_compilation import NativeMLXGRPOProfile, compile_native_grpo
from tuner.training.native_mlx_family import (
    NativeCommandCatalogV1, NativeCommandMaterialV1, native_mlx_family,
)
from tuner.training.native_mlx_planning import (
    NativePlanningV1, NativeQualificationV1, NativeRestrictedAdmissionV1,
)

__all__ = [
    "NativeArtifactBudget",
    "NativeCommandCatalogV1",
    "NativeCommandMaterialV1",
    "NativeEvidenceAuthority",
    "NativeFile",
    "NativeInvocation",
    "NativeMLXProfile",
    "NativeMLXGRPOProfile",
    "NativeMaterialAuthority",
    "NativeModelSnapshot",
    "NativePlanningV1",
    "NativeProcessLimits",
    "NativeQualificationV1",
    "NativeRestrictedAdmissionV1",
    "compile_native_sft",
    "compile_native_grpo",
    "native_mlx_family",
]
