"""Stable errors for the versioned host-project contract."""

from __future__ import annotations


class ProjectError(RuntimeError):
    """Base class carrying a machine-stable error code."""

    code = "PROJECT_ERROR"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "details": self.details}


class ProjectRootAmbiguousError(ProjectError):
    code = "PROJECT_ROOT_AMBIGUOUS"


class ManifestNotFoundError(ProjectError):
    code = "PROJECT_MANIFEST_NOT_FOUND"


class ManifestValidationError(ProjectError):
    code = "PROJECT_MANIFEST_INVALID"


class ManifestVersionError(ProjectError):
    code = "PROJECT_SCHEMA_UNSUPPORTED"


class PathReferenceError(ProjectError):
    code = "PROJECT_PATH_INVALID"


class PathEscapeError(PathReferenceError):
    code = "PROJECT_PATH_ESCAPE"


class WriteAccessError(PathReferenceError):
    code = "PROJECT_WRITE_DENIED"


class ExternalPathError(PathReferenceError):
    code = "PROJECT_EXTERNAL_PATH_DENIED"


class SourceLockError(ProjectError):
    code = "PROJECT_SOURCE_LOCK_INVALID"


class RepositoryUrlError(SourceLockError):
    code = "PROJECT_REPOSITORY_URL_INVALID"


class SecretReferenceError(ProjectError):
    code = "PROJECT_SECRET_REF_INVALID"


class SecretUnavailableError(ProjectError):
    code = "PROJECT_SECRET_UNAVAILABLE"
