"""Provider-neutral engine dispatch and artifact verification."""

from importlib import import_module


_EXPORTS = {
    "ArtifactContract": ".artifacts",
    "ArtifactEntry": ".artifacts",
    "ArtifactIntegrity": ".artifacts",
    "ArtifactInventory": ".artifacts",
    "ArtifactRequirement": ".artifacts",
    "IntegrityVerification": ".artifacts",
    "InventoryVerification": ".artifacts",
    "verify_inventory": ".artifacts",
    "CanonicalWorkloadBytesV1": ".dispatch",
    "CanonicalWorkloadFileLocationV1": ".dispatch",
    "CanonicalWorkloadFileV1": ".dispatch",
    "DispatchInvocation": ".dispatch",
    "EngineDispatcher": ".dispatch",
    "ProcessResult": ".dispatch",
    "ProcessRunner": ".dispatch",
    "SubprocessRunner": ".dispatch",
    "WorkerInvocationV1": ".dispatch",
    "build_dispatch_invocation": ".dispatch",
    "build_worker_invocation": ".dispatch",
    "materialize_worker_invocation": ".dispatch",
    "ArtifactReader": ".verification",
    "ArtifactReadError": ".verification",
    "ArtifactReadLimitExceeded": ".verification",
    "MAX_ARTIFACT_BYTES": ".verification",
    "SemanticCheck": ".verification",
    "SemanticVerifier": ".verification",
    "VerificationReport": ".verification",
    "VerificationService": ".verification",
    "VerificationStatus": ".verification",
    "WorkloadBindingVerifier": ".verification",
    "authenticate_artifacts": ".verification",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "ArtifactContract",
    "ArtifactEntry",
    "ArtifactIntegrity",
    "ArtifactInventory",
    "ArtifactReadError",
    "ArtifactReadLimitExceeded",
    "ArtifactReader",
    "ArtifactRequirement",
    "CanonicalWorkloadBytesV1",
    "CanonicalWorkloadFileLocationV1",
    "CanonicalWorkloadFileV1",
    "DispatchInvocation",
    "EngineDispatcher",
    "InventoryVerification",
    "IntegrityVerification",
    "MAX_ARTIFACT_BYTES",
    "ProcessResult",
    "ProcessRunner",
    "SemanticCheck",
    "SemanticVerifier",
    "SubprocessRunner",
    "WorkerInvocationV1",
    "VerificationReport",
    "VerificationService",
    "VerificationStatus",
    "WorkloadBindingVerifier",
    "authenticate_artifacts",
    "build_dispatch_invocation",
    "build_worker_invocation",
    "materialize_worker_invocation",
    "verify_inventory",
]
