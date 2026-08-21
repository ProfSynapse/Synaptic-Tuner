from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tuner.cloud import bootstrap_capsule as capsule_module
from tuner.cloud.bootstrap_capsule import (
    CAPSULE_MANIFEST,
    CAPSULE_MODULE_PATHS,
    CapsuleError,
    authenticate_external_input,
    build_capsule,
    invoke_verified_capsule,
    verified_capsule_scratch,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bootstrap_capsule_remains_minimal_and_excludes_training_runtime() -> None:
    assert CAPSULE_MODULE_PATHS == (
        "tuner/cloud/bootstrap_core.py",
        "tuner/cloud/bootstrap_capsule.py",
    )
    assert "tuner/cloud/hf_training_smoke_remote_entry.py" not in CAPSULE_MODULE_PATHS


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _literal_git_bytes(repository: Path, commit: str, member: str) -> bytes:
    environment = dict(os.environ)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{member}"],
        check=True, capture_output=True, env=environment,
    ).stdout


def _directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True, capture_output=True, text=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def _capsule_repository(tmp_path: Path, *, stub_core: bool = False) -> tuple[Path, str]:
    repository = tmp_path / "capsule-source"
    module_root = repository / "tuner" / "cloud"
    module_root.mkdir(parents=True)
    if stub_core:
        (module_root / "bootstrap_core.py").write_text(
            "import json\n"
            "class BootstrapError(RuntimeError): pass\n"
            "def reconstruct_source_lock_json(lock, policy, destination, environment=None):\n"
            "    return {'schema_version':'synaptic-bootstrap-result/v1',"
            "'lock':json.loads(lock),'policy':json.loads(policy),'destination':destination}\n",
            encoding="utf-8",
        )
    else:
        shutil.copyfile(REPO_ROOT / "tuner/cloud/bootstrap_core.py", module_root / "bootstrap_core.py")
    shutil.copyfile(REPO_ROOT / "tuner/cloud/bootstrap_capsule.py", module_root / "bootstrap_capsule.py")
    _git(repository, "init")
    _git(repository, "config", "user.name", "Synaptic Test")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "add", "tuner/cloud/bootstrap_core.py", "tuner/cloud/bootstrap_capsule.py")
    _git(repository, "commit", "-m", "bootstrap")
    return repository, _git(repository, "rev-parse", "HEAD")


def _built_capsule(tmp_path: Path, *, stub_core: bool = False) -> tuple[Path, str, str]:
    repository, commit = _capsule_repository(tmp_path, stub_core=stub_core)
    build = build_capsule(repository, tmp_path / "capsule")
    return build.root, build.manifest_sha256, commit


def _rewrite_manifest(root: Path, mutate) -> str:
    path = root / CAPSULE_MANIFEST
    document = json.loads(path.read_text(encoding="ascii"))
    mutate(document)
    content = (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_capsule_build_is_deterministic_committed_code_only_and_manifest_bound(tmp_path: Path) -> None:
    repository, commit = _capsule_repository(tmp_path)
    committed_members = {
        member: _literal_git_bytes(repository, commit, member)
        for member in CAPSULE_MODULE_PATHS
    }
    for member in CAPSULE_MODULE_PATHS:
        (repository / member).write_text(f"dirty {member}\n", encoding="utf-8")
    first = build_capsule(repository, tmp_path / "first", revision=commit)
    second = build_capsule(repository, tmp_path / "second", revision=commit)

    assert first.engine_commit == second.engine_commit == commit
    assert first.manifest_sha256 == second.manifest_sha256
    for member in (*CAPSULE_MODULE_PATHS, CAPSULE_MANIFEST):
        assert (first.root / member).read_bytes() == (second.root / member).read_bytes()
    for member, committed in committed_members.items():
        assert (first.root / member).read_bytes() == committed
    manifest = json.loads(first.manifest_path.read_text(encoding="ascii"))
    schema = json.loads(
        (REPO_ROOT / "schemas/synaptic-bootstrap-capsule-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    assert manifest["schema_version"] == "synaptic-bootstrap-capsule/v1"
    assert manifest["engine_commit"] == commit
    assert [item["path"] for item in manifest["files"]] == list(CAPSULE_MODULE_PATHS)
    serialized = first.manifest_path.read_text(encoding="ascii").lower()
    for forbidden in ("source_lock", "checkout_policy", "credential", "plugin", "input"):
        assert forbidden not in serialized


def test_capsule_literal_commit_and_both_blobs_ignore_replacement_refs(tmp_path: Path) -> None:
    repository, commit_a = _capsule_repository(tmp_path)
    expected = {member: _literal_git_bytes(repository, commit_a, member) for member in CAPSULE_MODULE_PATHS}
    for member in CAPSULE_MODULE_PATHS:
        (repository / member).write_text(f"replacement B for {member}\n", encoding="utf-8")
    _git(repository, "add", *CAPSULE_MODULE_PATHS)
    _git(repository, "commit", "-m", "replacement target")
    commit_b = _git(repository, "rev-parse", "HEAD")
    _git(repository, "replace", commit_a, commit_b)

    build = build_capsule(repository, tmp_path / "literal-a", revision=commit_a)

    assert build.engine_commit == commit_a
    assert commit_a != commit_b
    for member, literal_a in expected.items():
        assert (build.root / member).read_bytes() == literal_a
        assert (build.root / member).read_bytes() != _literal_git_bytes(repository, commit_b, member)


def test_verified_capsule_copies_to_private_scratch_and_cleans_success_and_failure(tmp_path: Path) -> None:
    root, digest, _ = _built_capsule(tmp_path)
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    with verified_capsule_scratch(root, digest, scratch_parent=scratch_parent) as scratch:
        assert scratch.parent == scratch_parent
        if os.name != "nt":
            assert scratch.stat().st_mode & 0o777 == 0o700
        assert (scratch / CAPSULE_MANIFEST).is_file()
    assert list(scratch_parent.iterdir()) == []

    changed = False
    def race(path: Path) -> None:
        nonlocal changed
        if not changed:
            path.write_bytes(path.read_bytes() + b"changed")
            changed = True

    with pytest.raises(CapsuleError, match="changed before private copy"):
        with verified_capsule_scratch(root, digest, scratch_parent=scratch_parent, after_member_read=race):
            pass
    assert list(scratch_parent.iterdir()) == []


def test_capsule_rejects_manifest_and_member_tampering(tmp_path: Path) -> None:
    root, digest, _ = _built_capsule(tmp_path)
    (root / CAPSULE_MANIFEST).write_bytes(b"{}\n")
    with pytest.raises(CapsuleError, match="manifest digest mismatch"):
        with verified_capsule_scratch(root, digest):
            pass


def test_capsule_rejects_reordered_or_shape_drifted_manifest(tmp_path: Path) -> None:
    schema = json.loads(
        (REPO_ROOT / "schemas/synaptic-bootstrap-capsule-v1.schema.json").read_text(encoding="utf-8")
    )
    root, _digest, _ = _built_capsule(tmp_path / "order")
    digest = _rewrite_manifest(root, lambda value: value["files"].reverse())
    reordered = json.loads((root / CAPSULE_MANIFEST).read_text(encoding="ascii"))
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(reordered)
    with pytest.raises(CapsuleError, match="canonical order"):
        with verified_capsule_scratch(root, digest):
            pass


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["limits"].update(max_file_bytes=2097152.0),
        lambda value: value["limits"].update(max_total_bytes=4194304.0),
        lambda value: value["limits"].update(max_total_bytes=True),
        lambda value: value["files"][0].update(size=2097152.0),
        lambda value: value["files"][1].update(size=False),
        lambda value: value["files"][0].update(mode=420.0),
        lambda value: value["files"][1].update(mode=True),
    ],
)
def test_manifest_numeric_scalars_require_exact_ints_before_member_read_or_scratch_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate,
) -> None:
    root, _digest, _ = _built_capsule(tmp_path / "capsule-case")
    digest = _rewrite_manifest(root, mutate)
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    marker = scratch_parent / "marker.txt"
    marker.write_text("unchanged\n", encoding="utf-8")
    reads: list[Path] = []
    original_read = capsule_module._read_regular_file

    def recording_read(path: Path, *, maximum: int):
        reads.append(Path(path))
        return original_read(path, maximum=maximum)

    monkeypatch.setattr(capsule_module, "_read_regular_file", recording_read)
    with pytest.raises(CapsuleError, match="must be an integer"):
        with verified_capsule_scratch(root, digest, scratch_parent=scratch_parent):
            pass
    assert reads == [root / CAPSULE_MANIFEST]
    assert marker.read_text(encoding="utf-8") == "unchanged\n"
    assert list(scratch_parent.iterdir()) == [marker]

    root, _digest, _ = _built_capsule(tmp_path / "extra")
    digest = _rewrite_manifest(root, lambda value: value.update(unexpected=True))
    with pytest.raises(CapsuleError, match="canonical wire shape"):
        with verified_capsule_scratch(root, digest):
            pass

    root, _digest, _ = _built_capsule(tmp_path / "member-extra")
    digest = _rewrite_manifest(root, lambda value: value["files"][0].update(unexpected=True))
    with pytest.raises(CapsuleError, match="canonical wire shape"):
        with verified_capsule_scratch(root, digest):
            pass

    root, digest, _ = _built_capsule(tmp_path / "member")
    member = root.joinpath(*Path(CAPSULE_MODULE_PATHS[0]).parts)
    member.write_bytes(member.read_bytes() + b"tamper")
    with pytest.raises(CapsuleError, match="integrity"):
        with verified_capsule_scratch(root, digest):
            pass


@pytest.mark.parametrize("unsafe", ["../escape.py", "/absolute.py", "tuner\\cloud\\core.py"])
def test_capsule_rejects_traversal_and_noncanonical_paths(tmp_path: Path, unsafe: str) -> None:
    root, _digest, _ = _built_capsule(tmp_path)
    digest = _rewrite_manifest(root, lambda value: value["files"][0].update(path=unsafe))
    with pytest.raises(CapsuleError, match="unsafe member path"):
        with verified_capsule_scratch(root, digest):
            pass


def test_capsule_rejects_duplicate_oversize_and_nonregular_members(tmp_path: Path) -> None:
    root, _digest, _ = _built_capsule(tmp_path / "duplicate")
    digest = _rewrite_manifest(root, lambda value: value["files"][1].update(path=value["files"][0]["path"]))
    with pytest.raises(CapsuleError, match="duplicate"):
        with verified_capsule_scratch(root, digest):
            pass

    root, _digest, _ = _built_capsule(tmp_path / "oversize")
    digest = _rewrite_manifest(root, lambda value: value["files"][0].update(size=2 * 1024 * 1024 + 1))
    with pytest.raises(CapsuleError, match="invalid member size"):
        with verified_capsule_scratch(root, digest):
            pass

    root, digest, _ = _built_capsule(tmp_path / "directory")
    member = root.joinpath(*Path(CAPSULE_MODULE_PATHS[0]).parts)
    member.unlink()
    member.mkdir()
    with pytest.raises(CapsuleError, match="links or reparse|regular files"):
        with verified_capsule_scratch(root, digest):
            pass


def test_capsule_rejects_symlink_member_when_platform_permits(tmp_path: Path) -> None:
    root, digest, _ = _built_capsule(tmp_path)
    member = root.joinpath(*Path(CAPSULE_MODULE_PATHS[0]).parts)
    target = tmp_path / "target.py"
    target.write_bytes(member.read_bytes())
    member.unlink()
    try:
        member.symlink_to(target)
    except OSError:
        pytest.skip("platform does not permit an unprivileged file symlink")
    with pytest.raises(CapsuleError, match="links or reparse|regular files"):
        with verified_capsule_scratch(root, digest):
            pass


def test_capsule_rejects_real_junction_or_symlink_components_before_io(tmp_path: Path) -> None:
    root, digest, _ = _built_capsule(tmp_path / "root-case")
    linked_root = tmp_path / "linked-root"
    _directory_link(linked_root, root)
    with pytest.raises(CapsuleError, match="links or reparse"):
        with verified_capsule_scratch(linked_root, digest):
            pass

    root, digest, _ = _built_capsule(tmp_path / "ancestor-case")
    original = root / "tuner"
    moved = tmp_path / "moved-tuner"
    original.rename(moved)
    _directory_link(original, moved)
    with pytest.raises(CapsuleError, match="links or reparse"):
        with verified_capsule_scratch(root, digest):
            pass

    real_scratch = tmp_path / "real-scratch"
    real_scratch.mkdir()
    linked_scratch = tmp_path / "linked-scratch"
    _directory_link(linked_scratch, real_scratch)
    root, digest, _ = _built_capsule(tmp_path / "scratch-case")
    with pytest.raises(CapsuleError, match="links or reparse"):
        with verified_capsule_scratch(root, digest, scratch_parent=linked_scratch):
            pass
    assert not list(real_scratch.iterdir())

    inputs = tmp_path / "real-inputs"
    inputs.mkdir()
    source = inputs / "lock.json"
    source.write_bytes(b"{}\n")
    linked_inputs = tmp_path / "linked-inputs"
    _directory_link(linked_inputs, inputs)
    with pytest.raises(CapsuleError, match="links or reparse"):
        authenticate_external_input(linked_inputs / "lock.json", hashlib.sha256(b"{}\n").hexdigest())


def test_private_scratch_cleanup_failure_is_fail_closed_without_masking_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, digest, _ = _built_capsule(tmp_path)

    def fail_cleanup(_scratch: Path) -> None:
        raise OSError("sensitive-cleanup-detail")

    monkeypatch.setattr(capsule_module, "_cleanup_private_scratch", fail_cleanup)
    with pytest.raises(CapsuleError, match="scratch cleanup failed") as cleanup_only:
        with verified_capsule_scratch(root, digest, scratch_parent=tmp_path):
            pass
    assert "sensitive-cleanup-detail" not in str(cleanup_only.value)

    primary = CapsuleError("primary integrity failure")
    with pytest.raises(CapsuleError, match="primary integrity failure") as combined:
        with verified_capsule_scratch(root, digest, scratch_parent=tmp_path):
            raise primary
    assert combined.value is primary
    assert any("cleanup also failed" in note for note in getattr(primary, "__notes__", ()))


def test_external_inputs_are_separate_hash_bound_regular_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source-lock.json"
    content = b'{"opaque":"not interpreted by verifier"}\n'
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    assert authenticate_external_input(source, digest) == content
    with pytest.raises(CapsuleError, match="digest mismatch"):
        authenticate_external_input(source, "0" * 64)


def test_verified_entrypoint_authenticates_external_inputs_then_calls_shared_core(tmp_path: Path) -> None:
    root, manifest_digest, _ = _built_capsule(tmp_path, stub_core=True)
    lock_document = {"schema_version": "synaptic-source-lock/v1", "run_id": "separate"}
    policy_document = {"allowed_hosts": ["git.example.test"], "allowed_schemes": ["https"]}
    lock_bytes = (json.dumps(lock_document) + "\n").encode()
    policy_bytes = (json.dumps(policy_document) + "\n").encode()
    lock_path = tmp_path / "source-lock.json"
    policy_path = tmp_path / "checkout-policy.json"
    lock_path.write_bytes(lock_bytes)
    policy_path.write_bytes(policy_bytes)
    result = invoke_verified_capsule(
        root, manifest_digest,
        source_lock_path=lock_path, source_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        checkout_policy_path=policy_path, checkout_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        destination=tmp_path / "checkout", scratch_parent=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["lock"] == lock_document
    assert payload["policy"] == policy_document
    assert payload["destination"] == str(tmp_path / "checkout")
    assert "huggingface" not in result.stdout.lower() + result.stderr.lower()


def test_bootstrap_modules_are_stdlib_only_and_have_no_provider_or_publication_surface() -> None:
    core = (REPO_ROOT / "tuner/cloud/bootstrap_core.py").read_text(encoding="utf-8")
    capsule = (REPO_ROOT / "tuner/cloud/bootstrap_capsule.py").read_text(encoding="utf-8")
    assert "from tuner" not in core and "import tuner" not in core
    for forbidden in ("huggingface_hub", "run_job(", "volume(", "modal", "runpod", "twine", "publish"):
        assert forbidden not in core.lower()
        assert forbidden not in capsule.lower()
