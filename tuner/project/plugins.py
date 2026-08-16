"""Import-light discovery and explicit trusted loading for project plug-ins.

Discovery in this module never imports project plug-in code.  Loading a
callable is a separate, deliberately named operation because a plug-in runs
with the full authority of the Synaptic Tuner process; it is not sandboxed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

PLUGIN_ENTRY_POINT_GROUP_PREFIX = "synaptic_tuner.plugins"
SUPPORTED_PLUGIN_API_VERSIONS = frozenset({"v1"})
# Retained as a compatibility alias for callers that used the original public
# constant.  The unversioned group itself is not a valid installed plug-in
# declaration; installed contracts use ``<prefix>.<kind>.<version>``.
PLUGIN_ENTRY_POINT_GROUP = PLUGIN_ENTRY_POINT_GROUP_PREFIX
TRUST_NOTICE = (
    "Trusted plug-in code runs with full process authority and may access "
    "environment variables, credentials, memory, and accessible files."
)

PluginOrigin = Literal["manifest", "entry_point", "legacy"]
_LOGICAL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_TARGET_RE = re.compile(
    r"^(?P<module>[A-Za-z_][A-Za-z0-9_.]*):(?P<attribute>[A-Za-z_][A-Za-z0-9_.]*)$"
)
_KIND_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_API_RE = re.compile(
    r"^synaptic\.plugin/(?P<kind>[A-Za-z][A-Za-z0-9_]*)/(?P<version>v[1-9][0-9]*)$"
)
_ENTRY_POINT_GROUP_RE = re.compile(
    rf"^{re.escape(PLUGIN_ENTRY_POINT_GROUP_PREFIX)}\."
    r"(?P<kind>[A-Za-z][A-Za-z0-9_]*)\.(?P<version>v[1-9][0-9]*)$"
)


class PluginError(RuntimeError):
    """Base error with a stable code for plug-in contract failures."""

    code = "PLUGIN_ERROR"

    def __init__(self, message: str, *, details: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "details": self.details}


class PluginReferenceError(PluginError):
    code = "PLUGIN_REFERENCE_INVALID"


class PluginNotFoundError(PluginError):
    code = "PLUGIN_NOT_FOUND"


class PluginConflictError(PluginError):
    code = "PLUGIN_BINDING_CONFLICT"


class PluginContractError(PluginError):
    code = "PLUGIN_CONTRACT_MISMATCH"


class PluginLoadError(PluginError):
    code = "PLUGIN_TRUSTED_LOAD_FAILED"


@dataclass(frozen=True)
class PluginBinding:
    """Serializable plug-in identity discovered without importing its target."""

    name: str
    kind: str
    api: str
    target: str
    origin: PluginOrigin
    distribution: str | None = None
    trusted: bool = True
    _entry_point: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _LOGICAL_NAME_RE.fullmatch(self.name):
            raise PluginReferenceError(f"Invalid plug-in name: {self.name!r}")
        if not self.kind or not self.api:
            raise PluginContractError("Plug-in kind and api must be non-empty")
        if not _TARGET_RE.fullmatch(self.target):
            raise PluginReferenceError(
                "Plug-in target must use the form 'module:callable'",
                details={"target": self.target},
            )
        if not self.trusted:
            raise PluginContractError("Project plug-ins cannot be represented as sandboxed")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "api": self.api,
            "target": self.target,
            "origin": self.origin,
            "trusted": True,
            "trust_notice": TRUST_NOTICE,
        }
        if self.distribution:
            result["distribution"] = self.distribution
        return result

    def load_trusted(self) -> object:
        """Import and return the target with full process authority.

        Calling this method is the explicit trust boundary.  It performs no
        sandboxing and must never be used during metadata-only inspection.
        """

        try:
            if self._entry_point is not None:
                loaded = self._entry_point.load()
            else:
                module_name, attribute_path = self.target.split(":", 1)
                loaded = importlib.import_module(module_name)
                for component in attribute_path.split("."):
                    loaded = getattr(loaded, component)
        except Exception as exc:
            raise PluginLoadError(
                f"Trusted plug-in load failed for {self.name!r}: {exc}",
                details={"name": self.name, "origin": self.origin},
            ) from exc
        if not callable(loaded):
            raise PluginLoadError(
                f"Plug-in target for {self.name!r} is not callable",
                details={"name": self.name, "target": self.target},
            )
        return loaded


def _entry_point_distribution(entry_point: object) -> str | None:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return None
    name = getattr(distribution, "name", None)
    if isinstance(name, str) and name:
        return name
    metadata = getattr(distribution, "metadata", None)
    if metadata is not None:
        try:
            value = metadata.get("Name")
        except (AttributeError, TypeError):
            value = None
        if isinstance(value, str) and value:
            return value
    return None


def plugin_entry_point_group(kind: str, api: str) -> str:
    """Return the authoritative entry-point group for a plug-in contract."""

    if not isinstance(kind, str) or not _KIND_RE.fullmatch(kind):
        raise PluginContractError(
            "Installed plug-in kind must be a portable identifier",
            details={"kind": kind},
        )
    match = _API_RE.fullmatch(api) if isinstance(api, str) else None
    if match is None or match.group("kind") != kind:
        raise PluginContractError(
            "Installed plug-in API must identify the requested kind and version",
            details={"kind": kind, "api": api},
        )
    version = match.group("version")
    if version not in SUPPORTED_PLUGIN_API_VERSIONS:
        raise PluginContractError(
            "Installed plug-in API version is not supported",
            details={"kind": kind, "api": api, "version": version},
        )
    return f"{PLUGIN_ENTRY_POINT_GROUP_PREFIX}.{kind}.{version}"


def _entry_point_contract(entry_point: object) -> tuple[str, str]:
    group = getattr(entry_point, "group", None)
    match = _ENTRY_POINT_GROUP_RE.fullmatch(group) if isinstance(group, str) else None
    if match is None:
        raise PluginContractError(
            "Installed plug-in is missing authoritative kind/API metadata",
            details={"group": group},
        )
    kind = match.group("kind")
    version = match.group("version")
    if version not in SUPPORTED_PLUGIN_API_VERSIONS:
        raise PluginContractError(
            "Installed plug-in declares an unsupported API version",
            details={"group": group, "version": version},
        )
    return kind, f"synaptic.plugin/{kind}/{version}"


def _is_plugin_group(group: object) -> bool:
    return isinstance(group, str) and (
        group == PLUGIN_ENTRY_POINT_GROUP_PREFIX
        or group.startswith(f"{PLUGIN_ENTRY_POINT_GROUP_PREFIX}.")
    )


def _all_entry_points(entry_points: object | None = None) -> list[object]:
    available = importlib.metadata.entry_points() if entry_points is None else entry_points
    if isinstance(available, Mapping):
        flattened: list[object] = []
        for group, members in available.items():
            if _is_plugin_group(group):
                flattened.extend(members)
        return flattened
    return [
        item
        for item in available  # type: ignore[union-attr]
        if _is_plugin_group(getattr(item, "group", None))
    ]


def discover_entry_points(
    entry_points: object | None = None,
    *,
    kind: str,
    api: str,
) -> Mapping[str, object]:
    """Return matching authoritative metadata without calling ``load``."""

    expected_group = plugin_entry_point_group(kind, api)

    discovered: dict[str, object] = {}
    for entry_point in _all_entry_points(entry_points):
        declared_kind, declared_api = _entry_point_contract(entry_point)
        if getattr(entry_point, "group", None) != expected_group:
            continue
        if declared_kind != kind or declared_api != api:
            raise PluginContractError(
                "Installed plug-in metadata does not match the requested contract",
                details={
                    "expected_kind": kind,
                    "expected_api": api,
                    "actual_kind": declared_kind,
                    "actual_api": declared_api,
                },
            )
        name = getattr(entry_point, "name", None)
        value = getattr(entry_point, "value", None)
        if not isinstance(name, str) or not _LOGICAL_NAME_RE.fullmatch(name):
            raise PluginReferenceError(f"Invalid installed plug-in name: {name!r}")
        if not isinstance(value, str) or not _TARGET_RE.fullmatch(value):
            raise PluginReferenceError(
                f"Installed plug-in {name!r} has an invalid target",
                details={"target": value},
            )
        if name in discovered:
            raise PluginConflictError(
                f"Duplicate installed plug-in binding: {name}",
                details={"name": name},
            )
        discovered[name] = entry_point
    return discovered


def _resolve_installed_entry_point(
    name: str,
    *,
    kind: str,
    api: str,
    entry_points: object | None,
) -> object | None:
    available = _all_entry_points(entry_points)
    matching = discover_entry_points(available, kind=kind, api=api)
    installed = matching.get(name)
    if installed is not None:
        return installed

    declared_contracts: list[dict[str, object]] = []
    for candidate in available:
        if getattr(candidate, "name", None) != name:
            continue
        candidate_kind, candidate_api = _entry_point_contract(candidate)
        declared_contracts.append(
            {
                "kind": candidate_kind,
                "api": candidate_api,
                "group": getattr(candidate, "group", None),
            }
        )
    if declared_contracts:
        raise PluginContractError(
            f"Installed plug-in {name!r} does not match the requested contract",
            details={
                "expected_kind": kind,
                "expected_api": api,
                "declared_contracts": declared_contracts,
            },
        )
    return None


def manifest_bindings(manifest: Mapping[str, object] | object | None) -> Mapping[str, Mapping[str, object]]:
    """Extract binding mappings from a manifest object or raw manifest data."""

    if manifest is None:
        return {}
    data = getattr(manifest, "data", manifest)
    if not isinstance(data, Mapping):
        raise PluginContractError("Project manifest must be a mapping")
    plugins = data.get("plugins", {})
    if not isinstance(plugins, Mapping):
        raise PluginContractError("Project manifest plugins must be a mapping")
    bindings = plugins.get("bindings", {})
    if not isinstance(bindings, Mapping):
        raise PluginContractError("Project manifest plugin bindings must be a mapping")
    result: dict[str, Mapping[str, object]] = {}
    for name, binding in bindings.items():
        if not isinstance(name, str) or not isinstance(binding, Mapping):
            raise PluginContractError("Each manifest plug-in binding must be a named mapping")
        if name in result:
            raise PluginConflictError(f"Duplicate manifest plug-in binding: {name}")
        result[name] = binding
    return result


def resolve_plugin(
    reference: str,
    *,
    kind: str,
    api: str,
    manifest: Mapping[str, object] | object | None = None,
    entry_points: object | None = None,
) -> PluginBinding:
    """Resolve manifest, installed, or explicit legacy metadata without import.

    Installed entry points carry identity, target, kind, and API version in
    standard package metadata. The caller supplies only the expected contract;
    it can never assign a different contract to installed code.
    """

    if not isinstance(reference, str) or not reference:
        raise PluginReferenceError("Plug-in reference must be a non-empty string")
    if not kind or not api:
        raise PluginContractError("Expected plug-in kind and api are required")

    if reference.startswith("plugin://"):
        name = reference[len("plugin://") :]
        if not _LOGICAL_NAME_RE.fullmatch(name):
            raise PluginReferenceError(f"Invalid logical plug-in reference: {reference!r}")
        declared = manifest_bindings(manifest).get(name)
        if declared is not None:
            declared_kind = declared.get("kind")
            declared_api = declared.get("api")
            target = declared.get("target")
            if declared_kind != kind or declared_api != api:
                raise PluginContractError(
                    f"Manifest plug-in {name!r} does not match the requested contract",
                    details={
                        "expected_kind": kind,
                        "expected_api": api,
                        "actual_kind": declared_kind,
                        "actual_api": declared_api,
                    },
                )
            if not isinstance(target, str):
                raise PluginReferenceError(f"Manifest plug-in {name!r} requires a target")
            return PluginBinding(
                name=name, kind=kind, api=api, target=target, origin="manifest"
            )

        installed = _resolve_installed_entry_point(
            name, kind=kind, api=api, entry_points=entry_points
        )
        if installed is not None:
            return PluginBinding(
                name=name,
                kind=kind,
                api=api,
                target=str(getattr(installed, "value")),
                origin="entry_point",
                distribution=_entry_point_distribution(installed),
                _entry_point=installed,
            )
        raise PluginNotFoundError(
            f"No plug-in binding found for {name!r}", details={"name": name}
        )

    if _TARGET_RE.fullmatch(reference):
        return PluginBinding(
            name=reference.replace(":", "."),
            kind=kind,
            api=api,
            target=reference,
            origin="legacy",
        )
    raise PluginReferenceError(
        "Plug-ins must use plugin://name or the legacy module:callable form",
        details={"reference": reference},
    )


def discover_plugins(
    *,
    kind: str,
    api: str,
    manifest: Mapping[str, object] | object | None = None,
    entry_points: object | None = None,
) -> tuple[PluginBinding, ...]:
    """List metadata with manifest bindings overriding installed names."""

    installed = discover_entry_points(entry_points, kind=kind, api=api)
    combined: dict[str, PluginBinding] = {}
    for name, entry_point in installed.items():
        combined[name] = PluginBinding(
            name=name,
            kind=kind,
            api=api,
            target=str(getattr(entry_point, "value")),
            origin="entry_point",
            distribution=_entry_point_distribution(entry_point),
            _entry_point=entry_point,
        )
    for name, declared in manifest_bindings(manifest).items():
        # A host declaration owns its logical name even when this listing is
        # filtered to a different kind. Never reveal an installed fallback
        # behind a manifest binding.
        combined.pop(name, None)
        declared_kind = declared.get("kind")
        declared_api = declared.get("api")
        target = declared.get("target")
        if declared_kind != kind:
            continue
        if declared_api != api:
            raise PluginContractError(
                f"Manifest plug-in {name!r} has incompatible API {declared_api!r}"
            )
        if not isinstance(target, str):
            raise PluginReferenceError(f"Manifest plug-in {name!r} requires a target")
        combined[name] = PluginBinding(
            name=name, kind=kind, api=api, target=target, origin="manifest"
        )
    return tuple(combined[name] for name in sorted(combined))


__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "PLUGIN_ENTRY_POINT_GROUP_PREFIX",
    "SUPPORTED_PLUGIN_API_VERSIONS",
    "TRUST_NOTICE",
    "PluginBinding",
    "PluginConflictError",
    "PluginContractError",
    "PluginError",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginReferenceError",
    "discover_entry_points",
    "discover_plugins",
    "manifest_bindings",
    "plugin_entry_point_group",
    "resolve_plugin",
]
