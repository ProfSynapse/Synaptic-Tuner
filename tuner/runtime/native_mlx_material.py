"""Reviewable native runtime bytes and linked-library closure.

Capture never grants execution authority. The consuming host must independently
review and bind the exact capture digest in its grant and compiled workload.
OS shared-cache libraries are trusted through the exact observed OS build;
runtime-loaded plug-ins and hardware behavior still require qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import sysconfig

from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_macho import macho_links, resolve_links
from tuner.runtime.native_mlx_runtime import MLXRuntimeExpectation, inspect_mlx_runtime_candidate


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def strict_json(raw: bytes, *, maximum: int = 16 * 1024 * 1024):
    if type(raw) is not bytes or not 1 <= len(raw) <= maximum:
        raise ValueError("native document size is invalid")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("native document has duplicate keys")
            value[key] = item
        return value

    def constant(_):
        raise ValueError("native document has nonfinite values")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _absolute(value):
    if (type(value) is not str or not value.startswith("/") or len(value.encode()) > 4096
            or str(Path(value)) != value or any(p == ".." for p in Path(value).parts)
            or any(ord(c) < 32 for c in value)):
        raise ValueError("native capture path is invalid")
    return value


@dataclass(frozen=True)
class NativeCaptureConfig:
    """Named root trees plus explicit external interpreter/library files.

    ``stdlib`` and ``packages`` must be exact real directories; ``sources`` is
    one or more narrow code trees, never a project root with secrets. Every file
    in the selected trees participates, including .pth and .pyc files.
    """

    stdlib: str
    packages: str
    sources: tuple[str, ...]
    files: tuple[str, ...]
    system_libraries: tuple[str, ...]
    aliases: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        _absolute(self.stdlib)
        _absolute(self.packages)
        for name in ("sources", "files", "system_libraries"):
            values = getattr(self, name)
            if type(values) is not tuple or not 1 <= len(values) <= 4096:
                raise ValueError("native capture path set is invalid")
            if values != tuple(sorted(set(values))):
                raise ValueError("native capture paths must be unique and sorted")
            for value in values:
                _absolute(value)
        trees = (self.stdlib, self.packages, *self.sources)
        if len(set(trees)) != len(trees):
            raise ValueError("native capture trees overlap")
        for first in trees:
            for second in trees:
                if first != second and Path(first).is_relative_to(second):
                    raise ValueError("native capture trees overlap")
        if any(Path(file).is_relative_to(tree) for file in self.files for tree in trees):
            raise ValueError("native explicit file duplicates a tree")
        if (type(self.aliases) is not tuple or len(self.aliases) > 128
                or any(type(item) is not tuple or len(item) != 2 for item in self.aliases)):
            raise ValueError("native aliases are invalid")
        paths = []
        for path, target in self.aliases:
            _absolute(path)
            if type(target) is not str or not target or len(target.encode()) > 4096:
                raise ValueError("native alias target is invalid")
            if not any(Path(path).is_relative_to(tree) for tree in trees):
                raise ValueError("native alias is outside capture")
            paths.append(path)
        if paths != sorted(set(paths)):
            raise ValueError("native aliases must be unique and sorted")

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("native capture configuration fields differ")
        if any(type(value[name]) is not list for name in ("sources", "files", "system_libraries", "aliases")):
            raise ValueError("native capture arrays are invalid")
        return cls(value["stdlib"], value["packages"], tuple(value["sources"]),
                   tuple(value["files"]), tuple(value["system_libraries"]),
                   tuple(tuple(item) for item in value["aliases"]))


def capture_native_material(config: NativeCaptureConfig, expected: MLXRuntimeExpectation,
                            *, dependency_lock: bytes) -> bytes:
    if type(config) is not NativeCaptureConfig or type(expected) is not MLXRuntimeExpectation:
        raise TypeError("native capture requires exact configuration")
    if str(Path(sysconfig.get_path("stdlib")).resolve(strict=True)) != config.stdlib:
        raise ValueError("native standard library root differs")
    executable = str(Path(sys.executable).resolve(strict=True))
    if executable not in config.files:
        raise ValueError("native interpreter is absent from capture")
    # Distribution discovery must use the same package tree whose bytes we bind.
    from importlib import metadata
    observed_roots = {
        str(Path(item.locate_file("")).resolve(strict=True))
        for item in metadata.distributions()
    }
    if observed_roots != {config.packages}:
        raise ValueError("native package discovery roots differ")
    runtime = inspect_mlx_runtime_candidate(expected, dependency_lock=dependency_lock)
    roots = (config.stdlib, config.packages, *config.sources)
    alias_targets = []
    for path, target in config.aliases:
        resolved = str(Path(path).resolve(strict=True))
        if path == str(Path(config.stdlib) / "site-packages"):
            # Homebrew's shared site-packages link is explicitly excluded from
            # the isolated -S child's search path, never followed or admitted.
            if resolved == config.packages:
                raise ValueError("native isolated package root aliases global site-packages")
            disposition = "excluded-global-site-packages"
        elif resolved in config.files:
            disposition = "captured-file-alias"
        else:
            raise ValueError("native alias does not resolve to captured file bytes")
        alias_targets.append({"path": path, "spelling": target, "resolved": resolved,
                              "disposition": disposition})
    inventories = []
    native = {}
    all_files = set()
    total = 0
    # Captured at one pass, then rechecked by the eventual child. Exclusive host
    # ownership is required; this is not a filesystem-wide atomic snapshot.
    for root in roots:
        with NativeTree(Path(root)) as tree:
            aliases = {str(Path(path).relative_to(root)): target for path, target in config.aliases
                       if Path(path).is_relative_to(root)}
            names = tree.members(aliases=aliases)
            members = []
            for member in names:
                raw = tree.read(member)
                total += len(raw)
                if total > 4 * 1024**3:
                    raise ValueError("native capture aggregate budget exceeded")
                absolute = str(Path(root) / member)
                all_files.add(absolute)
                members.append({"path": member, "size": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest()})
                links = macho_links(raw)
                if Path(member).suffix in (".so", ".dylib") and links is None:
                    raise ValueError("native extension has unsupported executable format")
                if links is not None:
                    native[absolute] = links
            if tree.members(aliases=aliases) != names:
                raise ValueError("native capture membership changed")
            inventories.append({"root": root, "members": members})
    files = []
    for filename in config.files:
        path = Path(filename)
        with NativeTree(path.parent) as tree:
            raw = tree.read(path.name)
        total += len(raw)
        if total > 4 * 1024**3:
            raise ValueError("native capture aggregate budget exceeded")
        files.append({"path": filename, "size": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()})
        all_files.add(filename)
        links = macho_links(raw)
        if path.suffix in (".so", ".dylib") and links is None:
            raise ValueError("native external library has unsupported executable format")
        if links is not None:
            native[filename] = links
    if executable not in native:
        raise ValueError("native interpreter is not arm64 Mach-O")
    graph = resolve_links(native, executable=executable, admitted_files=all_files,
                          system_libraries=config.system_libraries)
    return canonical({
        "schema": "synaptic-native-mlx-capture/v1",
        "status": "OBSERVED_NOT_ADMITTED",
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "macos": platform.mac_ver()[0], "kernel_build": platform.version()},
        "config": asdict(config), "runtime": runtime,
        "executable": executable, "trees": inventories, "files": files,
        "native_dependencies": graph,
        "aliases": alias_targets,
        "trust_boundary": "exact-os-build-shared-cache-and-exclusive-host-ownership",
    })


def verify_native_material(raw: bytes, *, expected_digest: str,
                           dependency_lock: bytes) -> None:
    """Compare to a digest bound by the host, never to a self-issued expectation."""
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("native material digest differs")
    value = strict_json(raw)
    if (type(value) is not dict or value.get("schema") != "synaptic-native-mlx-capture/v1"
            or canonical(value) != raw):
        raise ValueError("native capture schema or canonical bytes differ")
    runtime = value["runtime"]
    expected = MLXRuntimeExpectation.from_dict({
        name: runtime[name] for name in MLXRuntimeExpectation.__dataclass_fields__
    })
    actual = capture_native_material(NativeCaptureConfig.from_dict(value["config"]), expected,
                                     dependency_lock=dependency_lock)
    if actual != raw:
        raise ValueError("native runtime bytes or library closure differ")


def native_runtime_spec(raw: bytes):
    """Represent captured native bytes in the existing compiler contract.

    Structural representation grants no execution or hardware authority. The
    selected host admission must independently authenticate and revalidate the
    capture; no OCI image or Modal build is fabricated.
    """
    from tuner.training.contracts import RuntimeSpec

    value = strict_json(raw)
    fields = {
        "schema", "status", "platform", "config", "runtime", "executable",
        "trees", "files", "native_dependencies", "aliases", "trust_boundary",
    }
    if (type(value) is not dict or set(value) != fields
            or value["schema"] != "synaptic-native-mlx-capture/v1"
            or value["status"] != "OBSERVED_NOT_ADMITTED" or canonical(value) != raw):
        raise ValueError("native material capture is invalid")
    platform_value = value["platform"]
    if (type(platform_value) is not dict or set(platform_value) != {"system", "machine", "macos", "kernel_build"}
            or platform_value["system"] != "Darwin" or platform_value["machine"] != "arm64"
            or any(type(platform_value[key]) is not str or not platform_value[key]
                   for key in ("macos", "kernel_build"))):
        raise ValueError("native material platform is unsupported")
    NativeCaptureConfig.from_dict(value["config"])
    runtime = value["runtime"]
    if type(runtime) is not dict or set(runtime) != set(MLXRuntimeExpectation.__dataclass_fields__) | {"status"}:
        raise ValueError("native material runtime fields differ")
    if runtime["status"] != "CANDIDATE_MATCH_ONLY":
        raise ValueError("native material comparison status differs")
    expected = MLXRuntimeExpectation.from_dict({
        name: runtime[name] for name in MLXRuntimeExpectation.__dataclass_fields__
    })
    return RuntimeSpec(None, expected.dependency_lock_sha256, expected.python_version,
                       "native_environment", hashlib.sha256(raw).hexdigest())
