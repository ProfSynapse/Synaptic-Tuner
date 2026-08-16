"""Internal contracts for embedding Synaptic Tuner in a host project."""

from .config_layers import (
    ConfigDocument,
    ConfigOverride,
    ResolvedConfig,
    load_config_document,
    resolve_config_layers,
)
from .context import ProjectContext, discover_project_context, find_nearest_manifest
from .manifest import ProjectManifest, load_project_manifest
from .path_refs import PathRef, resolve_path
from .secrets import SecretRef, redact_secrets, reject_literal_secrets, resolve_secret
from .source_bundle import (
    GitSource,
    RepositoryLocation,
    SourceLock,
    canonicalize_repository_url,
    inspect_git_source,
    resolve_relative_repository_url,
)

__all__ = [
    "ConfigDocument",
    "ConfigOverride",
    "GitSource",
    "PathRef",
    "ProjectContext",
    "ProjectManifest",
    "RepositoryLocation",
    "ResolvedConfig",
    "SecretRef",
    "SourceLock",
    "canonicalize_repository_url",
    "discover_project_context",
    "find_nearest_manifest",
    "inspect_git_source",
    "load_config_document",
    "load_project_manifest",
    "redact_secrets",
    "reject_literal_secrets",
    "resolve_config_layers",
    "resolve_path",
    "resolve_relative_repository_url",
    "resolve_secret",
]
