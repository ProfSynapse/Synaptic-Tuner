from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "scripts" / "regenerate_modal_inference_lock.py"
_spec = importlib.util.spec_from_file_location(
    "modal_inference_lock_maintenance", _SCRIPT
)
assert _spec is not None and _spec.loader is not None
maintenance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(maintenance)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _put(root: Path, relative: str, payload: bytes) -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fixture(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    for index, relative in enumerate(maintenance.SOURCE_MEMBERS):
        _put(root, relative, f"source-{index}:{relative}\n".encode())
    dependency = b"dependency lock bytes\n"
    _put(root, maintenance.DEPENDENCY_RELATIVE, dependency)
    stale = "0" * 64
    closure: dict[str, object] = {
        "schema_version": "synaptic-modal-inference-worker-closure/v1",
        "entrypoint": "tuner/execution/providers/modal/inference_bootstrap.py",
        "member_count": maintenance.SOURCE_MEMBER_COUNT,
        "payload_bytes": 0,
        "members": [
            {"path": path, "size_bytes": 0, "sha256": stale}
            for path in maintenance.SOURCE_MEMBERS
        ],
        "closure_digest": stale,
    }
    unsigned = dict(closure)
    unsigned.pop("closure_digest")
    closure["closure_digest"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    closure_bytes = _canonical(closure)
    _put(root, maintenance.CLOSURE_RELATIVE, closure_bytes)
    inventory_paths = sorted(
        (
            *maintenance.SOURCE_MEMBERS,
            maintenance.DEPENDENCY_RELATIVE,
            maintenance.CLOSURE_RELATIVE,
        )
    )
    runtime: dict[str, object] = {
        "schema_version": "synaptic-modal-inference-runtime-lock/v1",
        "base_registry_reference": "registry.example/vllm@sha256:" + "1" * 64,
        "sdk_version": "1.5.4",
        "python": {
            "implementation": "cpython",
            "version": "3.12.13",
            "executable": "/opt/python/bin/python",
            "executable_sha256": "2" * 64,
        },
        "dependency_lock_path": maintenance.DEPENDENCY_RELATIVE,
        "worker_closure_manifest_path": maintenance.CLOSURE_RELATIVE,
        "distributions": {
            "modal": "1.5.4",
            "safetensors": "0.7.0",
            "tokenizers": "0.22.2",
            "torch": "2.10.0",
            "transformers": "5.3.0",
            "vllm": "0.17.1",
        },
        "source_inventory": [
            {"path": path, "size_bytes": 0, "sha256": stale} for path in inventory_paths
        ],
    }
    _put(root, maintenance.RUNTIME_RELATIVE, _canonical(runtime))
    return runtime, closure


def test_reviewed_source_inventory_is_exact_and_fixed() -> None:
    assert maintenance.SOURCE_MEMBER_COUNT == 119
    assert len(maintenance.SOURCE_MEMBERS) == maintenance.SOURCE_MEMBER_COUNT
    assert maintenance.SOURCE_MEMBERS == tuple(sorted(maintenance.SOURCE_MEMBERS))
    assert len(set(maintenance.SOURCE_MEMBERS)) == maintenance.SOURCE_MEMBER_COUNT
    assert (
        hashlib.sha256(
            ("\n".join(maintenance.SOURCE_MEMBERS) + "\n").encode()
        ).hexdigest()
        == "5109950e21ca195c25ce1bc1c8769210bd4c173876bfccbb75bc67ae03dd7ab9"
    )


def _module_name(relative: str) -> tuple[str, str]:
    parts = list(Path(relative).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    module = ".".join(parts)
    package = module if is_package else module.rpartition(".")[0]
    return module, package


def _local_module_path(module: str) -> str | None:
    if not module or module.split(".", 1)[0] not in {
        "Evaluator", "synaptic_tuner", "tuner",
    }:
        return None
    relative = Path(*module.split("."))
    module_file = _ROOT / relative.with_suffix(".py")
    package_file = _ROOT / relative / "__init__.py"
    if module_file.is_file():
        return module_file.relative_to(_ROOT).as_posix()
    if package_file.is_file():
        return package_file.relative_to(_ROOT).as_posix()
    return None


def test_reviewed_source_inventory_is_closed_over_static_local_imports() -> None:
    declared = set(maintenance.SOURCE_MEMBERS)
    missing: set[tuple[str, str]] = set()
    for relative in maintenance.SOURCE_MEMBERS:
        if not relative.endswith(".py"):
            continue
        _, package = _module_name(relative)
        tree = ast.parse((_ROOT / relative).read_bytes(), filename=relative)
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if not package:
                        continue
                    anchor = "." * node.level + (node.module or "")
                    try:
                        base = importlib.util.resolve_name(anchor, package)
                    except (ImportError, ValueError):
                        continue
                else:
                    base = node.module or ""
                candidates.append(base)
                candidates.extend(
                    f"{base}.{alias.name}" if base else alias.name
                    for alias in node.names
                    if alias.name != "*"
                )
            for candidate in candidates:
                imported = _local_module_path(candidate)
                if imported is not None and imported not in declared:
                    missing.add((relative, imported))
    assert not missing


def test_check_then_write_refreshes_only_content_commitments(tmp_path: Path) -> None:
    runtime_before, closure_before = _fixture(tmp_path)
    dependency_before = (tmp_path / maintenance.DEPENDENCY_RELATIVE).read_bytes()

    assert maintenance.main([], root=tmp_path) == 3
    assert maintenance.main(["--write"], root=tmp_path) == 0
    assert maintenance.main([], root=tmp_path) == 0

    runtime = json.loads((tmp_path / maintenance.RUNTIME_RELATIVE).read_text())
    closure = json.loads((tmp_path / maintenance.CLOSURE_RELATIVE).read_text())
    pinned_runtime = copy.deepcopy(runtime)
    pinned_runtime.pop("source_inventory")
    expected_pins = copy.deepcopy(runtime_before)
    expected_pins.pop("source_inventory")
    assert pinned_runtime == expected_pins
    assert {
        key: closure[key] for key in ("schema_version", "entrypoint", "member_count")
    } == {
        key: closure_before[key]
        for key in ("schema_version", "entrypoint", "member_count")
    }
    assert (
        tmp_path / maintenance.DEPENDENCY_RELATIVE
    ).read_bytes() == dependency_before
    assert closure["payload_bytes"] == sum(
        item["size_bytes"] for item in closure["members"]
    )
    unsigned = dict(closure)
    unsigned.pop("closure_digest")
    assert closure["closure_digest"] == hashlib.sha256(_canonical(unsigned)).hexdigest()
    by_path = {item["path"]: item for item in runtime["source_inventory"]}
    closure_bytes = (tmp_path / maintenance.CLOSURE_RELATIVE).read_bytes()
    assert by_path[maintenance.CLOSURE_RELATIVE] == {
        "path": maintenance.CLOSURE_RELATIVE,
        "size_bytes": len(closure_bytes),
        "sha256": hashlib.sha256(closure_bytes).hexdigest(),
    }

    first = maintenance.SOURCE_MEMBERS[0]
    _put(tmp_path, first, b"changed\n")
    assert maintenance.main([], root=tmp_path) == 3


def test_exact_reviewed_predecessor_migrates_only_the_additive_source(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    closure_path = tmp_path / maintenance.CLOSURE_RELATIVE
    runtime_path = tmp_path / maintenance.RUNTIME_RELATIVE
    dependency_path = tmp_path / maintenance.DEPENDENCY_RELATIVE
    closure = json.loads(closure_path.read_bytes())
    closure["members"] = [
        item
        for item in closure["members"]
        if item["path"] != maintenance.ADDITIVE_SOURCE_MIGRATION
    ]
    closure["member_count"] = maintenance.PREVIOUS_SOURCE_MEMBER_COUNT
    closure["payload_bytes"] = sum(
        item["size_bytes"] for item in closure["members"]
    )
    unsigned = dict(closure)
    unsigned.pop("closure_digest")
    closure["closure_digest"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    closure_bytes = _canonical(closure)
    closure_path.write_bytes(closure_bytes)

    runtime = json.loads(runtime_path.read_bytes())
    runtime["source_inventory"] = [
        item
        for item in runtime["source_inventory"]
        if item["path"] != maintenance.ADDITIVE_SOURCE_MIGRATION
    ]
    by_path = {item["path"]: item for item in runtime["source_inventory"]}
    by_path[maintenance.CLOSURE_RELATIVE].update(
        size_bytes=len(closure_bytes),
        sha256=hashlib.sha256(closure_bytes).hexdigest(),
    )
    dependency = dependency_path.read_bytes()
    by_path[maintenance.DEPENDENCY_RELATIVE].update(
        size_bytes=len(dependency),
        sha256=hashlib.sha256(dependency).hexdigest(),
    )
    runtime_path.write_bytes(_canonical(runtime))

    assert maintenance.main([], root=tmp_path) == 3
    assert maintenance.main(["--write"], root=tmp_path) == 0
    assert maintenance.main([], root=tmp_path) == 0
    migrated = json.loads(closure_path.read_bytes())
    assert migrated["member_count"] == maintenance.SOURCE_MEMBER_COUNT
    assert [item["path"] for item in migrated["members"]] == list(
        maintenance.SOURCE_MEMBERS
    )


@pytest.mark.parametrize("kind", ["addition", "removal", "path_escape", "schema"])
def test_refuses_structural_or_schema_changes(tmp_path: Path, kind: str) -> None:
    _fixture(tmp_path)
    path = tmp_path / maintenance.CLOSURE_RELATIVE
    closure = json.loads(path.read_text())
    if kind == "addition":
        closure["members"].append(
            {"path": "tuner/extra.py", "size_bytes": 0, "sha256": "0" * 64}
        )
    elif kind == "removal":
        closure["members"].pop()
    elif kind == "path_escape":
        closure["members"][0]["path"] = "../escape.py"
    else:
        closure["unexpected"] = True
    path.write_bytes(_canonical(closure))
    before = path.read_bytes()

    assert maintenance.main(["--write"], root=tmp_path) == 125
    assert path.read_bytes() == before


def test_refuses_missing_lock_files_without_creating_them(tmp_path: Path) -> None:
    assert maintenance.main(["--write"], root=tmp_path) == 125
    assert not (tmp_path / maintenance.RUNTIME_RELATIVE).exists()
    assert not (tmp_path / maintenance.CLOSURE_RELATIVE).exists()


def test_refuses_malformed_existing_runtime_pin(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / maintenance.RUNTIME_RELATIVE
    runtime = json.loads(path.read_text())
    runtime["python"]["unexpected"] = "widened"
    path.write_bytes(_canonical(runtime))
    before = path.read_bytes()

    assert maintenance.main(["--write"], root=tmp_path) == 125
    assert path.read_bytes() == before


def test_rechecks_all_inputs_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path)
    runtime_before = (tmp_path / maintenance.RUNTIME_RELATIVE).read_bytes()
    closure_before = (tmp_path / maintenance.CLOSURE_RELATIVE).read_bytes()
    source = tmp_path / maintenance.SOURCE_MEMBERS[0]
    original_refresh = maintenance.refresh
    calls = 0

    def changing_refresh(root: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            source.write_bytes(b"concurrent source edit\n")
        return original_refresh(root)

    monkeypatch.setattr(maintenance, "refresh", changing_refresh)
    assert maintenance.main(["--write"], root=tmp_path) == 125
    assert (tmp_path / maintenance.RUNTIME_RELATIVE).read_bytes() == runtime_before
    assert (tmp_path / maintenance.CLOSURE_RELATIVE).read_bytes() == closure_before


def test_second_stage_failure_cleans_temporary_files(
    tmp_path: Path, monkeypatch
) -> None:
    _fixture(tmp_path)
    original = maintenance._stage
    calls = 0

    def fail_second(path, payload, mode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")
        return original(path, payload, mode)

    monkeypatch.setattr(maintenance, "_stage", fail_second)
    assert maintenance.main(["--write"], root=tmp_path) == 125
    parent = (tmp_path / maintenance.CLOSURE_RELATIVE).parent
    assert not list(parent.glob(".inference-*"))


def test_interrupted_pair_requires_consistent_pair_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    _fixture(tmp_path)
    runtime_before = (tmp_path / maintenance.RUNTIME_RELATIVE).read_bytes()
    closure_before = (tmp_path / maintenance.CLOSURE_RELATIVE).read_bytes()
    original = maintenance.os.replace

    def fail_runtime(source, destination):
        if destination == tmp_path / maintenance.RUNTIME_RELATIVE:
            raise OSError("injected replacement failure")
        return original(source, destination)

    monkeypatch.setattr(maintenance.os, "replace", fail_runtime)
    assert maintenance.main(["--write"], root=tmp_path) == 125
    assert (tmp_path / maintenance.RUNTIME_RELATIVE).read_bytes() == runtime_before
    assert (tmp_path / maintenance.CLOSURE_RELATIVE).read_bytes() != closure_before
    monkeypatch.setattr(maintenance.os, "replace", original)
    assert maintenance.main(["--write"], root=tmp_path) == 125
    _put(tmp_path, maintenance.CLOSURE_RELATIVE, closure_before)
    assert maintenance.main(["--write"], root=tmp_path) == 0
    assert maintenance.main([], root=tmp_path) == 0
