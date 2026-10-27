"""Explicit native MLX composition for the public TrainingAPI and RunsAPI.

Import this optional module deliberately, as with ``api.v1.modal``. Imports do
not load ML frameworks, inspect hardware, open storage, or admit execution.
The host owns source/runtime/model preparation, qualification or restricted
admission, signing keys, command catalogs and durable record/stream ports.
The separate GRPO compiler/profile is an unqualified source candidate; SFT
qualification does not admit GRPO or its reward/objective/runtime selection.

Submission defaults to bounded synchronous operation on the store-owning thread.
The versioned lifecycle candidate requires a host-owned session and separately
authenticated lifecycle qualification or exact restricted-test grant. Selecting
it alone enables no capability. Its public progress/cancel/reconcile joins still
require actual Darwin/MLX qualification; CPU tests are not hardware evidence.
Known-run reads retain original authority after start expiry and never resubmit.
Keep the session alive and pump it through run reads on its owning thread. A
desktop should isolate that host and retain state rather than signal saved PIDs.
"""

from tuner.runtime.native_files import NativeFile
from tuner.runtime.native_mlx_input import NativeModelSnapshot
from tuner.runtime.native_mlx_publication import NativeArtifactBudget
from tuner.runtime.native_process import NativeInvocation, NativeProcessLimits
from tuner.runtime.native_lifecycle_protocol import NativeLifecyclePolicyV1
from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
from tuner.training.native_mlx_authority import NativeEvidenceAuthority, NativeMaterialAuthority
from tuner.training.native_mlx_compilation import NativeMLXProfile, compile_native_sft
from tuner.training.native_mlx_grpo_compilation import NativeMLXGRPOProfile, compile_native_grpo
from tuner.training.native_mlx_family import (
    NativeCommandCatalogV1, NativeCommandMaterialV1, NativeCancellationMaterialV1, native_mlx_family,
)
from tuner.training.native_mlx_planning import (
    NativePlanningV1, NativeQualificationV1, NativeRestrictedAdmissionV1, NativeLifecycleGrantV1,
)


def native_lifecycle_session(*, records, streams, authority, launch):
    """Construct owned transport without opening storage or admitting execution.

    The consumer owns this context manager's lifetime. POSIX-only imports remain
    lazy so deliberate optional API imports stay neutral on other platforms.
    """
    import os
    if os.name != "posix":
        raise ValueError("native lifecycle session requires a POSIX host")
    from tuner.runtime.native_process_session import NativeProcessSession
    return NativeProcessSession(records, streams=streams, lifecycle_authority=authority, lifecycle=launch)

__all__ = [
    "NativeArtifactBudget",
    "NativeCommandCatalogV1",
    "NativeCommandMaterialV1",
    "NativeCancellationMaterialV1",
    "NativeEvidenceAuthority",
    "NativeFile",
    "NativeInvocation",
    "NativeLifecycleGrantV1",
    "NativeLifecycleLaunchV1",
    "NativeLifecyclePolicyV1",
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
    "native_lifecycle_session",
]
