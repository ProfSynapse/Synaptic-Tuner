"""Reviewable native runtime bytes and linked-library closure.

Capture never grants execution authority. The consuming host must independently
review and bind the exact capture digest in its grant and compiled workload.
Static OS shared-cache libraries are trusted through the exact observed OS
build. V2 separately records source-observed literal ctypes load names; it
does not attest actual future dlopen behavior or loaded image origins.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
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


_V1 = "synaptic-native-mlx-capture/v1"
_V2 = "synaptic-native-mlx-capture/v2"
_DYNAMIC_LIMIT = 32
_DYNAMIC_SOURCE_LIMIT = 1024 * 1024
_DYNAMIC_AST_LIMIT = 20000
_LEGACY_CONFIG_FIELDS = {
    "stdlib", "packages", "sources", "files", "system_libraries", "aliases",
}
_DYNAMIC_CONFIG_FIELDS = _LEGACY_CONFIG_FIELDS | {
    "dynamic_system_libraries", "required_builtin_modules", "expected_os_build",
}


@dataclass(frozen=True)
class NativeOSBuild:
    macos: str
    kernel_build: str

    def __post_init__(self):
        if (type(self.macos) is not str or not 1 <= len(self.macos) <= 64
                or type(self.kernel_build) is not str
                or not 1 <= len(self.kernel_build) <= 512
                or any(ord(char) < 32 for char in self.macos + self.kernel_build)):
            raise ValueError("native reviewed OS build is invalid")

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("native reviewed OS build fields differ")
        return cls(**value)


@dataclass(frozen=True)
class NativeDynamicOSLibrary:
    """A literal load named in captured source, not a dlopen attestation."""

    source: str
    source_sha256: str
    call: str
    library: str

    def __post_init__(self):
        _absolute(self.source)
        _absolute(self.library)
        if (not self.source.endswith(".py") or len(self.source.encode()) > 4096
                or type(self.source_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None
                or self.call != "ctypes.CDLL" or len(self.library.encode()) > 1024
                or re.fullmatch(r"/(?:[A-Za-z0-9_.+-]+/)*[A-Za-z0-9_.+-]+",
                                self.library) is None
                or not (
                    self.library.startswith("/usr/lib/") and self.library.endswith(".dylib")
                    or self.library.startswith("/System/Library/Frameworks/")
                    and ".framework/Versions/" in self.library
                    or self.library.startswith("/System/Library/PrivateFrameworks/")
                    and ".framework/Versions/" in self.library
                )):
            raise ValueError("native dynamic OS library declaration is invalid")

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("native dynamic OS library fields differ")
        return cls(**value)


def _literal_ctypes_loads(raw: bytes) -> tuple[str, ...]:
    """Recognize only direct literal ctypes.CDLL calls in one reviewed source."""
    if not 0 < len(raw) <= _DYNAMIC_SOURCE_LIMIT:
        raise ValueError("native dynamic source size differs")
    try:
        syntax = ast.parse(raw)
    except (SyntaxError, UnicodeError, RecursionError, ValueError):
        raise ValueError("native dynamic source syntax differs") from None
    nodes = []
    for node in ast.walk(syntax):
        if len(nodes) >= _DYNAMIC_AST_LIMIT:
            raise ValueError("native dynamic source syntax exceeds bound")
        nodes.append(node)
    if not any(isinstance(node, ast.Import) and any(
        item.name == "ctypes" and item.asname is None for item in node.names
    ) for node in nodes):
        raise ValueError("native dynamic source lacks plain ctypes import")
    loads = []
    direct_calls = set()
    for node in nodes:
        if isinstance(node, ast.Import) and any(
            item.name == "ctypes" and item.asname is not None for item in node.names
        ) or isinstance(node, ast.ImportFrom) and node.module == "ctypes":
            raise ValueError("native dynamic ctypes alias is unsupported")
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id in {"CDLL", "PyDLL", "WinDLL", "OleDLL"}:
            raise ValueError("native dynamic library call alias is unsupported")
        if (isinstance(function, ast.Name) and function.id == "getattr" and node.args
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "ctypes"):
            raise ValueError("native dynamic ctypes lookup is unsupported")
        if not isinstance(function, ast.Attribute) or function.attr not in {
            "CDLL", "PyDLL", "WinDLL", "OleDLL", "LoadLibrary",
        }:
            continue
        if (function.attr != "CDLL" or not isinstance(function.value, ast.Name)
                or function.value.id != "ctypes" or len(node.args) != 1 or node.keywords
                or not isinstance(node.args[0], ast.Constant)
                or type(node.args[0].value) is not str):
            raise ValueError("native dynamic library call is unsupported")
        direct_calls.add(id(function))
        loads.append(node.args[0].value)
    if any(isinstance(node, ast.Attribute) and node.attr == "CDLL"
           and isinstance(node.value, ast.Name) and node.value.id == "ctypes"
           and id(node) not in direct_calls for node in nodes):
        raise ValueError("native dynamic ctypes alias is unsupported")
    if len(loads) != len(set(loads)) or len(loads) > _DYNAMIC_LIMIT:
        raise ValueError("native dynamic library calls are duplicated or excessive")
    return tuple(sorted(loads))


def _source_absolute_ctypes_loads(raw: bytes) -> tuple[str, ...]:
    """Discover explicit OS-path calls across captured project source trees."""
    if b"CDLL" not in raw:
        return ()
    if len(raw) > _DYNAMIC_SOURCE_LIMIT:
        raise ValueError("native potential dynamic source exceeds bound")
    try:
        syntax = ast.parse(raw)
    except (SyntaxError, UnicodeError, RecursionError, ValueError):
        raise ValueError("native potential dynamic source syntax differs") from None
    found = []
    for count, node in enumerate(ast.walk(syntax), start=1):
        if count > _DYNAMIC_AST_LIMIT:
            raise ValueError("native potential dynamic source syntax exceeds bound")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "CDLL" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ctypes" and node.args
                and isinstance(node.args[0], ast.Constant)
                and type(node.args[0].value) is str
                and node.args[0].value.startswith("/")):
            found.append(node.args[0].value)
    return tuple(found)


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
    dynamic_system_libraries: tuple[NativeDynamicOSLibrary, ...] = ()
    required_builtin_modules: tuple[str, ...] = ()
    expected_os_build: NativeOSBuild | None = None

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
        dynamic = self.dynamic_system_libraries
        builtins = self.required_builtin_modules
        if type(dynamic) is not tuple or type(builtins) is not tuple:
            raise ValueError("native dynamic closure arrays are invalid")
        if (bool(dynamic) != bool(builtins) or bool(dynamic) != bool(self.expected_os_build)):
            raise ValueError("native dynamic closure requires built-in and OS selections")
        if dynamic:
            if (type(dynamic) is not tuple or len(dynamic) > _DYNAMIC_LIMIT
                    or any(type(item) is not NativeDynamicOSLibrary for item in dynamic)
                    or len({(item.source, item.library) for item in dynamic}) != len(dynamic)
                    or dynamic != tuple(sorted(set(dynamic), key=lambda item: (
                        item.source, item.library, item.call,
                    )))):
                raise ValueError("native dynamic OS library set is invalid")
            alias_paths = {path for path, _ in self.aliases}
            if any(not any(Path(item.source).is_relative_to(root) for root in self.sources)
                   or item.source in alias_paths for item in dynamic):
                raise ValueError("native dynamic source is not a captured regular member")
            if (type(builtins) is not tuple or not 1 <= len(builtins) <= 32
                    or builtins != tuple(sorted(set(builtins)))
                    or any(type(name) is not str or not name.isidentifier()
                           for name in builtins) or "_ctypes" not in builtins):
                raise ValueError("native dynamic built-in selection is invalid")
            if type(self.expected_os_build) is not NativeOSBuild:
                raise ValueError("native dynamic OS build selection is invalid")

    @classmethod
    def from_dict(cls, value):
        if (type(value) is not dict
                or set(value) not in (_LEGACY_CONFIG_FIELDS, _DYNAMIC_CONFIG_FIELDS)):
            raise ValueError("native capture configuration fields differ")
        if any(type(value[name]) is not list for name in ("sources", "files", "system_libraries", "aliases")):
            raise ValueError("native capture arrays are invalid")
        dynamic = value.get("dynamic_system_libraries", [])
        builtins = value.get("required_builtin_modules", [])
        if type(dynamic) is not list or type(builtins) is not list:
            raise ValueError("native dynamic capture arrays are invalid")
        return cls(value["stdlib"], value["packages"], tuple(value["sources"]),
                   tuple(value["files"]), tuple(value["system_libraries"]),
                   tuple(tuple(item) for item in value["aliases"]),
                   tuple(NativeDynamicOSLibrary.from_dict(item) for item in dynamic),
                   tuple(builtins),
                   NativeOSBuild.from_dict(value["expected_os_build"])
                   if value.get("expected_os_build") is not None else None)


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
    dynamic_sources = {item.source for item in config.dynamic_system_libraries}
    source_bytes = {}
    observed_dynamic = []
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
                if absolute in dynamic_sources:
                    source_bytes[absolute] = raw
                if root in config.sources and member.endswith(".py"):
                    observed_dynamic.extend(
                        (absolute, library) for library in _source_absolute_ctypes_loads(raw)
                    )
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
    declared_dynamic = [(item.source, item.library) for item in config.dynamic_system_libraries]
    if sorted(observed_dynamic) != sorted(declared_dynamic):
        raise ValueError("native dynamic source calls differ from declaration")
    document = {
        "schema": _V1,
        "status": "OBSERVED_NOT_ADMITTED",
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "macos": platform.mac_ver()[0], "kernel_build": platform.version()},
        "config": asdict(config), "runtime": runtime,
        "executable": executable, "trees": inventories, "files": files,
        "native_dependencies": graph,
        "aliases": alias_targets,
        "trust_boundary": "exact-os-build-shared-cache-and-exclusive-host-ownership",
    }
    if config.dynamic_system_libraries:
        if asdict(config.expected_os_build) != {
            "macos": document["platform"]["macos"],
            "kernel_build": document["platform"]["kernel_build"],
        }:
            raise ValueError("native observed OS build differs from selection")
        selected = {}
        for item in config.dynamic_system_libraries:
            raw = source_bytes.get(item.source)
            if raw is None or hashlib.sha256(raw).hexdigest() != item.source_sha256:
                raise ValueError("native dynamic source member or hash differs")
            selected.setdefault(item.source, []).append(item.library)
        for source, libraries in selected.items():
            if _literal_ctypes_loads(source_bytes[source]) != tuple(sorted(libraries)):
                raise ValueError("native dynamic source calls differ from declaration")
        observed = tuple(sorted(sys.builtin_module_names))
        if not set(config.required_builtin_modules) <= set(observed):
            raise ValueError("native required built-in module is absent")
        document["schema"] = _V2
        document["dynamic_system_libraries"] = [
            asdict(item) for item in config.dynamic_system_libraries
        ]
        document["builtin_modules"] = {
            "required": list(config.required_builtin_modules),
            "observed_inventory_sha256": hashlib.sha256(canonical(observed)).hexdigest(),
        }
        document["dynamic_os_build"] = {
            "macos": document["platform"]["macos"],
            "kernel_build": document["platform"]["kernel_build"],
        }
    else:
        # Preserve exact v1 configuration/document bytes for legacy captures.
        del document["config"]["dynamic_system_libraries"]
        del document["config"]["required_builtin_modules"]
        del document["config"]["expected_os_build"]
    return canonical(document)


def verify_native_material(raw: bytes, *, expected_digest: str,
                           dependency_lock: bytes) -> None:
    """Compare to a digest bound by the host, never to a self-issued expectation."""
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("native material digest differs")
    value = strict_json(raw)
    if (type(value) is not dict or value.get("schema") not in (_V1, _V2)
            or canonical(value) != raw):
        raise ValueError("native capture schema or canonical bytes differ")
    config = NativeCaptureConfig.from_dict(value["config"])
    if (value["schema"] == _V2) != bool(config.dynamic_system_libraries):
        raise ValueError("native capture version and dynamic closure differ")
    if set(value["config"]) != (
        _DYNAMIC_CONFIG_FIELDS if value["schema"] == _V2 else _LEGACY_CONFIG_FIELDS
    ):
        raise ValueError("native capture configuration version differs")
    runtime = value["runtime"]
    expected = MLXRuntimeExpectation.from_dict({
        name: runtime[name] for name in MLXRuntimeExpectation.__dataclass_fields__
    })
    actual = capture_native_material(config, expected,
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
    if type(value) is not dict or value.get("schema") not in (_V1, _V2):
        raise ValueError("native material capture is invalid")
    if value["schema"] == _V2:
        fields |= {"dynamic_system_libraries", "builtin_modules", "dynamic_os_build"}
    if (set(value) != fields
            or value["status"] != "OBSERVED_NOT_ADMITTED" or canonical(value) != raw):
        raise ValueError("native material capture is invalid")
    platform_value = value["platform"]
    if (type(platform_value) is not dict or set(platform_value) != {"system", "machine", "macos", "kernel_build"}
            or platform_value["system"] != "Darwin" or platform_value["machine"] != "arm64"
            or any(type(platform_value[key]) is not str or not platform_value[key]
                   for key in ("macos", "kernel_build"))):
        raise ValueError("native material platform is unsupported")
    config = NativeCaptureConfig.from_dict(value["config"])
    if set(value["config"]) != (
        _DYNAMIC_CONFIG_FIELDS if value["schema"] == _V2 else _LEGACY_CONFIG_FIELDS
    ):
        raise ValueError("native capture configuration version differs")
    if value["schema"] == _V2:
        builtins = value["builtin_modules"]
        if (not config.dynamic_system_libraries
                or value["dynamic_system_libraries"] != [
                    asdict(item) for item in config.dynamic_system_libraries
                ] or type(builtins) is not dict
                or set(builtins) != {"required", "observed_inventory_sha256"}
                or builtins["required"] != list(config.required_builtin_modules)
                or type(builtins["observed_inventory_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", builtins["observed_inventory_sha256"]) is None
                or value["dynamic_os_build"] != {
                    "macos": platform_value["macos"],
                    "kernel_build": platform_value["kernel_build"],
                } or value["dynamic_os_build"] != asdict(config.expected_os_build)):
            raise ValueError("native dynamic material structure differs")
    elif config.dynamic_system_libraries:
        raise ValueError("native v1 material cannot declare dynamic libraries")
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
