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
    "DispatchInvocation": ".dispatch",
    "DispatchSpec": ".dispatch",
    "EngineDispatcher": ".dispatch",
    "ProcessResult": ".dispatch",
    "ProcessRunner": ".dispatch",
    "SubprocessRunner": ".dispatch",
    "build_dispatch_invocation": ".dispatch",
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
    "DispatchInvocation",
    "DispatchSpec",
    "EngineDispatcher",
    "InventoryVerification",
    "IntegrityVerification",
    "MAX_ARTIFACT_BYTES",
    "ProcessResult",
    "ProcessRunner",
    "SemanticCheck",
    "SemanticVerifier",
    "SubprocessRunner",
    "VerificationReport",
    "VerificationService",
    "VerificationStatus",
    "WorkloadBindingVerifier",
    "authenticate_artifacts",
    "build_dispatch_invocation",
    "verify_inventory",
]
