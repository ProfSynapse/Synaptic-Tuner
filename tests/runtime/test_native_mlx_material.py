from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import pytest

from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_macho import macho_links, resolve_links
from tuner.runtime import native_mlx_material as material
from tuner.runtime.native_mlx_runtime import MLXRuntimeExpectation


@pytest.fixture
def private_tree():
    # Real runtime roots intentionally reject /tmp's public writable ancestor.
    previous = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-contract-", dir=Path.home()) as value:
            yield Path(value)
    finally:
        os.umask(previous)


def macho(*dependencies, rpaths=()):
    commands = []
    for command, name, minimum in [
        *((0xC, name, 24) for name in dependencies),
        *((0x8000001C, name, 12) for name in rpaths),
    ]:
        value = name.encode() + b"\0"
        size = (minimum + len(value) + 7) // 8 * 8
        commands.append(struct.pack("<III", command, size, minimum)
                        + bytes(minimum - 12) + value + bytes(size - minimum - len(value)))
    body = b"".join(commands)
    return struct.pack("<IIIIIIII", 0xFEEDFACF, 0x0100000C, 0, 2, len(commands), len(body), 0, 0) + body


def test_tree_retains_real_file_bytes_and_bytecode(private_tree):
    (private_tree / "__pycache__").mkdir()
    (private_tree / "__pycache__/candidate.pyc").write_bytes(b"bytecode")
    (private_tree / "_virtualenv.pth").write_text("import _virtualenv")
    with NativeTree(private_tree, private=True) as tree:
        assert tree.members() == ("__pycache__/candidate.pyc", "_virtualenv.pth")
        inventory = tree.inventory()
        assert inventory[0].sha256 == hashlib.sha256(b"bytecode").hexdigest()
        with pytest.raises(ValueError):
            tree.read("_virtualenv.pth", maximum=2)


def test_tree_rejects_symlink_and_parent_substitution(private_tree):
    (private_tree / "file").write_bytes(b"abc")
    (private_tree / "alias").symlink_to("file")
    with NativeTree(private_tree) as tree:
        with pytest.raises(OSError):
            tree.read("alias")
        with pytest.raises(ValueError):
            tree.members()
    (private_tree / "alias").unlink()
    directory = private_tree / "owned"
    directory.mkdir()
    with NativeTree(directory) as held:
        directory.rename(private_tree / "previous")
        directory.mkdir()
        with pytest.raises(ValueError, match="directory changed"):
            held.check()


@pytest.mark.parametrize("path", ["../file", "/file", "a//b", "./file", "a\\b"])
def test_member_escape_rejected(private_tree, path):
    with NativeTree(private_tree) as tree:
        with pytest.raises(ValueError):
            tree.read(path)


def test_writable_ancestor_rejected(private_tree):
    (private_tree / "writable").mkdir(mode=0o700)
    os.chmod(private_tree / "writable", 0o777)
    with pytest.raises(ValueError):
        NativeTree(private_tree / "writable")


def test_macho_reads_real_command_shapes():
    result = macho_links(macho("/usr/lib/libSystem.B.dylib", rpaths=("@loader_path/lib",)))
    assert result.dependencies == ("/usr/lib/libSystem.B.dylib",)
    assert result.rpaths == ("@loader_path/lib",)
    assert macho_links(b"ordinary Python or metallib bytes") is None


@pytest.mark.parametrize("corrupt", [
    lambda data: data[:15],
    lambda data: data[:-1],
    lambda data: data[:16] + struct.pack("<I", 10000) + data[20:],
    lambda data: data[:4] + struct.pack("<I", 0x01000007) + data[8:],
    lambda data: data[:40] + struct.pack("<I", 0) + data[44:],
])
def test_macho_rejects_malformed_commands(corrupt):
    with pytest.raises(ValueError):
        macho_links(corrupt(macho("/usr/lib/libSystem.B.dylib")))


def test_universal_slice_is_explicitly_arm64():
    arm = macho("/usr/lib/libSystem.B.dylib")
    raw = b"\xca\xfe\xba\xbe" + struct.pack(">I", 1)
    raw += struct.pack(">IIIII", 0x0100000C, 0, 28, len(arm), 0) + arm
    assert macho_links(raw).dependencies == ("/usr/lib/libSystem.B.dylib",)
    with pytest.raises(ValueError):
        macho_links(raw[:-1])


def test_native_dependency_resolution_requires_captured_target(private_tree):
    executable = private_tree / "python"
    library = private_tree / "Python"
    executable.write_bytes(macho("@loader_path/Python"))
    library.write_bytes(macho("/usr/lib/libSystem.B.dylib"))
    links = {str(p): macho_links(p.read_bytes()) for p in (executable, library)}
    args = dict(executable=str(executable), system_libraries=("/usr/lib/libSystem.B.dylib",))
    with pytest.raises(ValueError, match="outside captured"):
        resolve_links(links, admitted_files={str(executable)}, **args)
    edges = resolve_links(links, admitted_files=set(links), **args)
    assert (str(executable), "@loader_path/Python", str(library)) in edges


def test_native_runpath_follows_declaring_loader_through_dependencies(private_tree):
    package = private_tree / "package"
    library_dir = package / "lib"
    library_dir.mkdir(parents=True)
    core = package / "core.so"
    engine = library_dir / "libmlx.dylib"
    helper = library_dir / "libjaccl.dylib"
    core.write_bytes(macho("@rpath/libmlx.dylib", rpaths=("@loader_path/lib",)))
    engine.write_bytes(macho("@rpath/libjaccl.dylib"))
    helper.write_bytes(macho("/usr/lib/libSystem.B.dylib"))
    links = {str(p): macho_links(p.read_bytes()) for p in (core, engine, helper)}
    edges = resolve_links(links, executable=str(core), admitted_files=set(links),
                          system_libraries=("/usr/lib/libSystem.B.dylib",))
    assert (str(core), "@rpath/libmlx.dylib", str(engine)) in edges
    assert (str(engine), "@rpath/libjaccl.dylib", str(helper)) in edges


def test_native_runpath_expands_relative_to_its_declarer(private_tree):
    entry = private_tree / "entry"
    plugin = private_tree / "plugin"
    libs = private_tree / "libs"
    for path in (entry, plugin, libs):
        path.mkdir()
    executable = entry / "python"
    core = plugin / "core.so"
    engine = libs / "libengine.dylib"
    helper = libs / "libhelper.dylib"
    executable.write_bytes(macho("@loader_path/../plugin/core.so"))
    core.write_bytes(macho("@rpath/libengine.dylib", rpaths=("@loader_path/../libs",)))
    engine.write_bytes(macho("@rpath/libhelper.dylib"))
    helper.write_bytes(macho())
    links = {str(p): macho_links(p.read_bytes()) for p in (executable, core, engine, helper)}
    edges = resolve_links(links, executable=str(executable), admitted_files=set(links),
                          system_libraries=())
    assert (str(engine), "@rpath/libhelper.dylib", str(helper)) in edges


def test_native_runpath_rejects_different_ancestor_bindings(private_tree):
    first = private_tree / "first"
    second = private_tree / "second"
    for directory in (first, second):
        directory.mkdir()
    roots = (private_tree / "first.so", private_tree / "second.so")
    shared = private_tree / "shared.dylib"
    leaves = (first / "leaf.dylib", second / "leaf.dylib")
    for root, directory in zip(roots, (first, second), strict=True):
        root.write_bytes(macho("@loader_path/shared.dylib",
                               rpaths=(f"@loader_path/{directory.name}",)))
    shared.write_bytes(macho("@rpath/leaf.dylib"))
    for leaf in leaves:
        leaf.write_bytes(macho())
    links = {str(p): macho_links(p.read_bytes()) for p in (*roots, shared, *leaves)}
    with pytest.raises(ValueError, match="differs across load chains"):
        resolve_links(links, executable=str(roots[0]), admitted_files=set(links),
                      system_libraries=())


def test_native_runpath_rejects_outside_captured_target(private_tree):
    executable = private_tree / "python"
    library = private_tree / "libengine.dylib"
    executable.write_bytes(macho("@rpath/libengine.dylib", rpaths=("@loader_path",)))
    library.write_bytes(macho())
    links = {str(p): macho_links(p.read_bytes()) for p in (executable, library)}
    with pytest.raises(ValueError, match="outside captured"):
        resolve_links(links, executable=str(executable), admitted_files={str(executable)},
                      system_libraries=())


def test_native_runpath_cycle_is_bounded_and_orphan_cycle_rejects(private_tree):
    executable = private_tree / "python"
    first = private_tree / "first.dylib"
    second = private_tree / "second.dylib"
    executable.write_bytes(macho("@loader_path/first.dylib"))
    first.write_bytes(macho("@loader_path/second.dylib"))
    second.write_bytes(macho("@loader_path/first.dylib"))
    links = {str(p): macho_links(p.read_bytes()) for p in (executable, first, second)}
    edges = resolve_links(links, executable=str(executable), admitted_files=set(links),
                          system_libraries=())
    assert len(edges) == 3
    executable.write_bytes(macho())
    links[str(executable)] = macho_links(executable.read_bytes())
    with pytest.raises(ValueError, match="no proven load chain"):
        resolve_links(links, executable=str(executable), admitted_files=set(links),
                      system_libraries=())


@pytest.fixture
def capture(private_tree, monkeypatch):
    trees = [private_tree / name for name in ("stdlib", "packages", "source")]
    for path in trees:
        path.mkdir()
        (path / "module.py").write_text("# pinned\n")
    executable = private_tree / "python"
    library = private_tree / "Python"
    executable.write_bytes(macho("@loader_path/Python"))
    library.write_bytes(macho("/usr/lib/libSystem.B.dylib"))
    lock = b"reviewed-original-hash-lock"
    expected = MLXRuntimeExpectation(
        "3.11.14", hashlib.sha256(executable.read_bytes()).hexdigest(),
        hashlib.sha256(lock).hexdigest(), (("mlx", "0.32.2"), ("mlx-lm", "0.32.2")),
    )
    config = material.NativeCaptureConfig(
        str(trees[0]), str(trees[1]), (str(trees[2]),),
        tuple(sorted((str(executable), str(library)))), ("/usr/lib/libSystem.B.dylib",),
    )
    runtime = asdict(expected)
    runtime["distributions"] = dict(expected.distributions)
    runtime["status"] = "CANDIDATE_MATCH_ONLY"
    monkeypatch.setattr(material.sys, "executable", str(executable))
    monkeypatch.setattr(material.sysconfig, "get_path", lambda key: str(trees[0]))
    monkeypatch.setattr(material, "inspect_mlx_runtime_candidate", lambda *a, **kw: runtime)
    from importlib import metadata

    class Distribution:
        def locate_file(self, _):
            return trees[1]
    monkeypatch.setattr(metadata, "distributions", lambda: [Distribution()])
    return config, expected, lock, library


@pytest.fixture
def dynamic_capture(capture, monkeypatch):
    config, expected, lock, _ = capture
    source = Path(config.sources[0]) / "module.py"
    source.write_bytes(
        b'import ctypes\nctypes.CDLL("/usr/lib/libSystem.B.dylib")\n'
        b'ctypes.CDLL("/usr/lib/libproc.dylib")\n'
    )
    monkeypatch.setattr(material.sys, "builtin_module_names", (
        *material.sys.builtin_module_names, "_ctypes",
    ))
    monkeypatch.setattr(material.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(material.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(material.platform, "mac_ver", lambda: ("27.0.1", (), ""))
    monkeypatch.setattr(material.platform, "version", lambda: "Darwin xnu-13432.1.9~1")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = replace(config, dynamic_system_libraries=tuple(
        material.NativeDynamicOSLibrary(str(source), digest, "ctypes.CDLL", library)
        for library in ("/usr/lib/libSystem.B.dylib", "/usr/lib/libproc.dylib")
    ), required_builtin_modules=("_ctypes",),
        expected_os_build=material.NativeOSBuild("27.0.1", "Darwin xnu-13432.1.9~1"))
    return config, expected, lock, source


def test_v1_capture_keeps_legacy_canonical_document_and_config(capture):
    config, expected, lock, _ = capture
    raw = material.capture_native_material(config, expected, dependency_lock=lock)
    value = material.strict_json(raw)
    assert value["schema"] == "synaptic-native-mlx-capture/v1"
    assert set(value["config"]) == {
        "stdlib", "packages", "sources", "files", "system_libraries", "aliases",
    }
    assert "dynamic_system_libraries" not in value
    assert "builtin_modules" not in value
    assert "dynamic_os_build" not in value
    assert material.canonical(value) == raw
    assert material.NativeCaptureConfig.from_dict(value["config"]) == config
    material.verify_native_material(raw, expected_digest=hashlib.sha256(raw).hexdigest(),
                                    dependency_lock=lock)


def test_v2_dynamic_source_builtin_and_host_profile_recheck(dynamic_capture, monkeypatch):
    from examples.training_qualification import native_host

    config, expected, lock, source = dynamic_capture
    raw = material.capture_native_material(config, expected, dependency_lock=lock)
    digest = hashlib.sha256(raw).hexdigest()
    value = material.strict_json(raw)
    assert value["schema"] == "synaptic-native-mlx-capture/v2"
    assert [item["library"] for item in value["dynamic_system_libraries"]] == [
        "/usr/lib/libSystem.B.dylib", "/usr/lib/libproc.dylib",
    ]
    assert all(item["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
               for item in value["dynamic_system_libraries"])
    assert value["builtin_modules"]["required"] == ["_ctypes"]
    assert value["dynamic_os_build"] == {
        "macos": "27.0.1", "kernel_build": "Darwin xnu-13432.1.9~1",
    }
    material.verify_native_material(raw, expected_digest=digest, dependency_lock=lock)
    assert material.native_runtime_spec(raw).material_digest == digest

    root = Path(config.stdlib).parent
    capture_file, lock_file, proposal_file = (
        root / "capture.json", root / "lock", root / "proposal.json",
    )
    capture_file.write_bytes(raw)
    lock_file.write_bytes(lock)
    proposal = json.loads((Path(native_host.ENGINE_ROOT) /
        "Trainers/mlx_sft_mac/runtime-candidate/sft-qualification-proposal.json").read_bytes())
    proposal["runtime"].update(hardware="fixture-M4", physical_memory_bytes=24 * 1024**3,
                               python_version=expected.python_version,
                               dependency_lock_sha256=expected.dependency_lock_sha256,
                               mlx_version="0.32.2", mlx_lm_version="0.32.2")
    proposal_file.write_bytes(material.canonical(proposal))
    monkeypatch.setattr(native_host, "source_record", lambda _: ({}, "a" * 64))
    monkeypatch.setattr(native_host, "hardware_record", lambda: (
        {"cpu": "fixture-M4", "memory_bytes": 24 * 1024**3}, "b" * 64,
    ))
    profile = native_host.profile_candidate({
        "proposal_file": str(proposal_file), "runtime_capture_file": str(capture_file),
        "dependency_lock_file": str(lock_file), "use_fast_tokenizer": True,
    })
    assert profile["runtime_digest"] == digest and profile["source_digest"] == "a" * 64
    source.write_bytes(source.read_bytes() + b"# drift\n")
    with pytest.raises(ValueError, match="dynamic source|runtime bytes"):
        native_host.profile_candidate({
            "proposal_file": str(proposal_file), "runtime_capture_file": str(capture_file),
            "dependency_lock_file": str(lock_file), "use_fast_tokenizer": True,
        })


def test_v1_cannot_omit_explicit_dynamic_os_calls(dynamic_capture):
    config, expected, lock, _ = dynamic_capture
    legacy = replace(config, dynamic_system_libraries=(), required_builtin_modules=(),
                     expected_os_build=None)
    with pytest.raises(ValueError, match="dynamic source calls"):
        material.capture_native_material(legacy, expected, dependency_lock=lock)


def test_v2_rejects_unselected_absolute_load_in_another_captured_member(dynamic_capture):
    config, expected, lock, source = dynamic_capture
    source.with_name("another.py").write_bytes(
        b'import ctypes\nctypes.CDLL("/usr/lib/libother.dylib")\n'
    )
    with pytest.raises(ValueError, match="dynamic source calls"):
        material.capture_native_material(config, expected, dependency_lock=lock)


@pytest.mark.parametrize("change", [
    "missing", "extra", "wrong_hash", "wrong_source", "nonliteral", "alias",
    "duplicate_call", "unsupported_call", "assignment_alias", "computed_getattr",
    "duplicate_declaration",
])
def test_v2_dynamic_declaration_rejects_source_or_selection_drift(dynamic_capture, change):
    config, expected, lock, source = dynamic_capture
    chosen = list(config.dynamic_system_libraries)
    if change == "missing":
        chosen.pop()
    elif change == "extra":
        chosen.append(material.NativeDynamicOSLibrary(
            str(source), chosen[0].source_sha256, "ctypes.CDLL", "/usr/lib/libother.dylib",
        ))
    elif change == "wrong_hash":
        chosen = [replace(item, source_sha256="0" * 64) for item in chosen]
    elif change == "wrong_source":
        chosen = [replace(item, source=str(source.with_name("missing.py"))) for item in chosen]
    elif change in {"nonliteral", "alias", "duplicate_call", "unsupported_call",
                    "assignment_alias", "computed_getattr"}:
        additions = {
            "nonliteral": b'ctypes.CDLL(variable)\n',
            "alias": b'from ctypes import CDLL\n',
            "duplicate_call": b'ctypes.CDLL("/usr/lib/libproc.dylib")\n',
            "unsupported_call": b'ctypes.PyDLL("/usr/lib/libother.dylib")\n',
            "assignment_alias": b'load = ctypes.CDLL\n',
            "computed_getattr": b'getattr(ctypes, "CDLL")\n',
        }
        source.write_bytes(source.read_bytes() + additions[change])
        chosen = [replace(item, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                  for item in chosen]
    elif change == "duplicate_declaration":
        chosen.append(chosen[-1])
    with pytest.raises(ValueError, match="native dynamic"):
        material.capture_native_material(
            replace(config, dynamic_system_libraries=tuple(chosen)), expected,
            dependency_lock=lock,
        )


def test_v2_builtin_os_build_and_static_graph_remain_independent(dynamic_capture, monkeypatch):
    config, expected, lock, _ = dynamic_capture
    raw = material.capture_native_material(config, expected, dependency_lock=lock)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(material.sys, "builtin_module_names", ("other",))
    with pytest.raises(ValueError, match="built-in"):
        material.verify_native_material(raw, expected_digest=digest, dependency_lock=lock)
    monkeypatch.setattr(material.sys, "builtin_module_names", ("_ctypes",))
    monkeypatch.setattr(material.platform, "version", lambda: "different-kernel-build")
    with pytest.raises(ValueError, match="OS build"):
        material.verify_native_material(raw, expected_digest=digest, dependency_lock=lock)
    monkeypatch.setattr(material.platform, "version", lambda: "Darwin xnu-13432.1.9~1")
    with pytest.raises(ValueError, match="OS build"):
        material.capture_native_material(replace(
            config, expected_os_build=material.NativeOSBuild("27.0.1", "other-build"),
        ), expected, dependency_lock=lock)
    with pytest.raises(ValueError, match="unused"):
        material.capture_native_material(replace(
            config, system_libraries=("/usr/lib/libSystem.B.dylib", "/usr/lib/libproc.dylib"),
        ), expected, dependency_lock=lock)
    with pytest.raises(ValueError, match="unresolved"):
        material.capture_native_material(replace(config, system_libraries=("/usr/lib/libproc.dylib",)),
                                         expected, dependency_lock=lock)


@pytest.mark.parametrize("name", [
    "/Users/owner/lib/libproc.dylib", "/usr/lib/../private/libproc.dylib",
    "/usr/lib/libproc.so", "/System/Library/Other/libproc.dylib",
])
def test_dynamic_os_trust_rejects_noncanonical_or_non_os_names(name):
    with pytest.raises(ValueError):
        material.NativeDynamicOSLibrary("/owned/source.py", "a" * 64, "ctypes.CDLL", name)


def test_actual_guardian_has_exact_reviewed_literal_os_loads():
    guardian = Path(material.__file__).with_name("native_process_guardian.py")
    assert material._literal_ctypes_loads(guardian.read_bytes()) == (
        "/usr/lib/libSystem.B.dylib", "/usr/lib/libproc.dylib",
    )


def test_material_rechecks_framework_not_only_python_stub(capture):
    config, expected, lock, framework = capture
    raw = material.capture_native_material(config, expected, dependency_lock=lock)
    digest = hashlib.sha256(raw).hexdigest()
    material.verify_native_material(raw, expected_digest=digest, dependency_lock=lock)
    framework.write_bytes(framework.read_bytes() + b"tampered native bytes")
    with pytest.raises(ValueError, match="runtime bytes"):
        material.verify_native_material(raw, expected_digest=digest, dependency_lock=lock)


def test_material_rechecks_added_importable_file_and_startup(capture):
    config, expected, lock, _ = capture
    raw = material.capture_native_material(config, expected, dependency_lock=lock)
    (Path(config.packages) / "unreviewed.pth").write_text("import unexpected")
    with pytest.raises(ValueError):
        material.verify_native_material(raw, expected_digest=hashlib.sha256(raw).hexdigest(),
                                        dependency_lock=lock)


def test_canonical_capture_and_host_digest_are_required(capture):
    config, expected, lock, _ = capture
    raw = material.capture_native_material(config, expected, dependency_lock=lock)
    with pytest.raises(ValueError, match="digest"):
        material.verify_native_material(raw, expected_digest="0" * 64, dependency_lock=lock)
    pretty = json.dumps(json.loads(raw), indent=2).encode()
    with pytest.raises(ValueError, match="canonical"):
        material.verify_native_material(pretty, expected_digest=hashlib.sha256(pretty).hexdigest(),
                                        dependency_lock=lock)


def test_homebrew_aliases_bind_spelling_and_captured_target(capture):
    config, expected, lock, framework = capture
    alias = Path(config.stdlib) / "framework.dylib"
    alias.symlink_to(framework)
    config = replace(config, aliases=((str(alias), str(framework)),))
    raw = material.capture_native_material(config, expected, dependency_lock=lock)
    assert json.loads(raw)["aliases"][0]["disposition"] == "captured-file-alias"
    alias.unlink()
    alias.symlink_to(Path(config.packages) / "module.py")
    with pytest.raises(ValueError, match="captured file"):
        material.verify_native_material(raw, expected_digest=hashlib.sha256(raw).hexdigest(),
                                        dependency_lock=lock)


def test_global_site_packages_alias_is_pinned_but_not_followed(capture):
    config, expected, lock, framework = capture
    global_packages = framework.parent / "global-packages"
    global_packages.mkdir(mode=0o700)
    (global_packages / "must_not_capture.py").write_text("not on isolated child path")
    alias = Path(config.stdlib) / "site-packages"
    alias.symlink_to(global_packages)
    config = replace(config, aliases=((str(alias), str(global_packages)),))
    raw = material.capture_native_material(config, expected, dependency_lock=lock)
    parsed = json.loads(raw)
    assert parsed["aliases"][0]["disposition"] == "excluded-global-site-packages"
    assert "must_not_capture.py" not in raw.decode()


def test_existing_runtime_spec_represents_native_without_invented_oci(capture):
    from tuner.training.contracts import RuntimeSpec
    config, expected, lock, _ = capture
    captured = json.loads(material.capture_native_material(config, expected, dependency_lock=lock))
    captured["platform"] = {
        "system": "Darwin", "machine": "arm64", "macos": "15.0",
        "kernel_build": "reviewed-kernel-build",
    }
    raw = material.canonical(captured)
    spec = material.native_runtime_spec(raw)
    assert spec.image is None and spec.material_kind == "native_environment"
    assert spec.material_digest == hashlib.sha256(raw).hexdigest()
    assert spec.dependency_lock_digest == expected.dependency_lock_sha256
    with pytest.raises(ValueError):
        RuntimeSpec("repo/image@sha256:" + "a" * 64, "b" * 64, "3.11.14",
                    "native_environment", "c" * 64)
    captured["platform"]["system"] = "Linux"
    with pytest.raises(ValueError, match="platform"):
        material.native_runtime_spec(material.canonical(captured))


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'])
def test_strict_capture_parser(raw):
    with pytest.raises(ValueError):
        material.strict_json(raw)
