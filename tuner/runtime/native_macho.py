"""Read arm64 Mach-O dependency commands without loading native code.

The closure checker deliberately rejects unsupported runpath inheritance and
ambiguous resolutions. OS shared-cache paths are an explicit reviewed platform
trust boundary, not falsely hashed filesystem files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class MachOLinks:
    dependencies: tuple[str, ...]
    rpaths: tuple[str, ...]


def macho_links(raw: bytes) -> MachOLinks | None:
    if raw[:4] in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
        wide = raw[:4] == b"\xca\xfe\xba\xbf"
        if len(raw) < 8:
            raise ValueError("truncated native fat header")
        count = struct.unpack_from(">I", raw, 4)[0]
        width = 32 if wide else 20
        if not 1 <= count <= 32 or 8 + count * width > len(raw):
            raise ValueError("invalid native fat inventory")
        slices = []
        for index in range(count):
            offset = 8 + width * index
            cpu = struct.unpack_from(">I", raw, offset)[0]
            start, size = struct.unpack_from(">QQ" if wide else ">II", raw, offset + 8)
            if start < 8 + count * width or start + size > len(raw):
                raise ValueError("invalid native fat slice")
            if cpu == 0x0100000C:
                slices.append(raw[start:start + size])
        if len(slices) != 1:
            raise ValueError("native fat file requires one arm64 slice")
        return _thin_links(slices[0])
    if raw[:4] == b"\xcf\xfa\xed\xfe":
        return _thin_links(raw)
    if raw[:4] in (b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xce",
                   b"\xbe\xba\xfe\xca", b"\xbf\xba\xfe\xca"):
        raise ValueError("unsupported native Mach-O architecture")
    return None


def _thin_links(raw):
    if len(raw) < 32 or raw[:4] != b"\xcf\xfa\xed\xfe":
        raise ValueError("invalid native Mach-O header")
    cpu, _, _, count, size = struct.unpack_from("<IIIII", raw, 4)
    if cpu != 0x0100000C or count > 4096 or size > 1024 * 1024 or 32 + size > len(raw):
        raise ValueError("invalid native Mach-O load commands")
    dependencies, rpaths = [], []
    position = 32
    for _ in range(count):
        if position + 8 > 32 + size:
            raise ValueError("truncated native load command")
        command, length = struct.unpack_from("<II", raw, position)
        if length < 8 or length % 8 or position + length > 32 + size:
            raise ValueError("invalid native load command length")
        # LC_LOAD_DYLIB, WEAK, REEXPORT, LAZY, UPWARD and LC_RPATH.
        if command in (0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023, 0x8000001C):
            minimum = 12 if command == 0x8000001C else 24
            if length < minimum:
                raise ValueError("truncated native dependency command")
            name_offset = struct.unpack_from("<I", raw, position + 8)[0]
            if not minimum <= name_offset < length:
                raise ValueError("invalid native dependency name")
            name_raw = raw[position + name_offset:position + length]
            if b"\0" not in name_raw:
                raise ValueError("unterminated native dependency name")
            name = name_raw.split(b"\0", 1)[0].decode("utf-8")
            if not name or any(ord(c) < 32 for c in name):
                raise ValueError("invalid native dependency name")
            (rpaths if command == 0x8000001C else dependencies).append(name)
        position += length
    if position != 32 + size:
        raise ValueError("native load command byte count differs")
    return MachOLinks(tuple(dependencies), tuple(rpaths))


def resolve_links(
    inventories: dict[str, MachOLinks],
    *, executable: str, admitted_files: set[str], system_libraries: tuple[str, ...],
) -> tuple[tuple[str, str, str], ...]:
    """Resolve captured load chains, including their inherited dyld runpaths.

    dyld builds the run path list from LC_RPATH commands in the dependency
    chain. Expand each entry relative to the image that *declared* it, then
    carry that directory into its dependent images. We reject ambiguous
    targets even where dyld's search order might choose one: this observation
    cannot approve an environmental or order-dependent library choice.
    """
    if (len(system_libraries) != len(set(system_libraries))
            or any(not (p.startswith("/usr/lib/") or p.startswith("/System/Library/"))
                   for p in system_libraries)):
        raise ValueError("invalid native OS library trust set")
    if executable not in inventories or not set(inventories) <= admitted_files:
        raise ValueError("native executable or inventory is outside captured files")
    if (len(inventories) > 4096
            or sum(len(links.dependencies) for links in inventories.values()) > 16384
            or sum(len(links.rpaths) for links in inventories.values()) > 4096):
        raise ValueError("native dependency inventory exceeds bound")
    edges = set()
    used_system = set()

    def expand(value, loader):
        if value == "@loader_path" or value.startswith("@loader_path/"):
            return str(Path(loader).parent) + value[len("@loader_path"):]
        if value == "@executable_path" or value.startswith("@executable_path/"):
            return str(Path(executable).parent) + value[len("@executable_path"):]
        if not value.startswith("/"):
            raise ValueError("unsupported native dependency resolution")
        return value

    declared = {
        loader: tuple(expand(path, loader) for path in links.rpaths)
        for loader, links in inventories.items()
    }
    if any(len(paths) > 128 for paths in declared.values()):
        raise ValueError("native runpath inventory exceeds bound")

    def targets(candidates):
        resolved = set()
        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                # Capture records the load spelling and canonical alias target.
                # Revalidation repeats this resolution against the exact files.
                canonical = str(path.resolve(strict=True))
                if canonical not in admitted_files:
                    raise ValueError("native dependency is outside captured files")
                resolved.add(canonical)
        return resolved

    # An extension module can be a load-chain root even though the executable
    # has no LC_LOAD_DYLIB edge to it. Determine possible incoming images only
    # from declared paths and captured bytes; unreachable cycles then reject.
    all_paths = tuple(sorted({path for paths in declared.values() for path in paths}))
    possible_incoming = set()
    for loader, links in inventories.items():
        for dependency in links.dependencies:
            if dependency in system_libraries:
                continue
            candidates = (
                (path + dependency[len("@rpath"):] for path in all_paths)
                if dependency.startswith("@rpath/")
                else (expand(dependency, loader),)
            )
            possible_incoming.update(targets(candidates) & inventories.keys())

    roots = sorted((set(inventories) - possible_incoming) | {executable})
    seen = set()
    bindings = {}
    pending = [(root, declared[root]) for root in roots]
    while pending:
        loader, inherited = pending.pop()
        context = (loader, inherited)
        if context in seen:
            continue
        seen.add(context)
        if len(seen) > 8192 or len(inherited) > 4096:
            raise ValueError("native load-chain context exceeds bound")
        for dependency in inventories[loader].dependencies:
            if dependency in system_libraries:
                used_system.add(dependency)
                edges.add((loader, dependency, dependency))
                continue
            candidates = (
                (path + dependency[len("@rpath"):] for path in inherited)
                if dependency.startswith("@rpath/")
                else (expand(dependency, loader),)
            )
            resolved = targets(candidates)
            if len(resolved) != 1:
                raise ValueError("native dependency is unresolved or ambiguous")
            target = resolved.pop()
            if target not in inventories:
                raise ValueError("native dependency target is not arm64 Mach-O")
            binding = (loader, dependency)
            if binding in bindings and bindings[binding] != target:
                raise ValueError("native dependency differs across load chains")
            bindings[binding] = target
            edges.add((loader, dependency, target))
            child_paths = tuple(dict.fromkeys((*declared[target], *inherited)))
            pending.append((target, child_paths))
            if len(pending) > 8192:
                raise ValueError("native load-chain context exceeds bound")
    if {name for name, _ in seen} != set(inventories):
        raise ValueError("native dependency has no proven load chain")
    if used_system != set(system_libraries):
        raise ValueError("native OS library trust set contains unused members")
    return tuple(sorted(edges))
