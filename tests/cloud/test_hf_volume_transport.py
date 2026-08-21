from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from tuner.cloud.hf_volume_transport import (
    _INLINE_VERIFIER,
    FIXED_STDLIB_TRAINING_LAUNCHER,
    HFArtifactVolumeSpec,
    HFVerifiedVolume,
    HFVerifiedVolumeSpec,
    build_verified_bootstrap_step,
    build_training_provider_command,
    build_runtime_projection_step,
    prove_read_only_volume,
    prove_writable_artifact_volume,
    project_runtime_layout,
    transport_metadata,
    validate_disjoint_volume_prefixes,
)
from tuner.core.exceptions import CloudProviderError
from tuner.cloud.hf_jobs import CloudJobSpec, HFJobExecutor


class RecordingVolume:
    def __init__(self, **kwargs):
        self.type = kwargs["type"]
        self.source = kwargs["source"]
        self.mount_path = kwargs["mount_path"]
        self.read_only = kwargs.get("read_only")
        self.path = kwargs.get("path")

    def to_dict(self):
        value = {
            "type": self.type,
            "source": self.source,
            "mountPath": self.mount_path,
            "readOnly": self.read_only,
        }
        if self.path is not None:
            value["path"] = self.path
        return value


def _run_job(*, image, command, volumes=None):
    raise AssertionError("feature detection must not invoke run_job")


def _write(root: Path, relative: str, content: bytes) -> str:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _spec(tmp_path: Path) -> HFVerifiedVolumeSpec:
    capsule_digest = _write(tmp_path, "capsule/synaptic-bootstrap-capsule.json", b"manifest\n")
    lock_digest = _write(tmp_path, "run/source-lock.json", b"lock\n")
    policy_digest = _write(tmp_path, "run/checkout-policy.json", b"policy\n")
    return HFVerifiedVolumeSpec(
        source="owner/bootstrap-bucket",
        path="prepared/run-123",
        capsule_path="capsule",
        capsule_manifest_sha256=capsule_digest,
        source_lock_path="run/source-lock.json",
        source_lock_sha256=lock_digest,
        checkout_policy_path="run/checkout-policy.json",
        checkout_policy_sha256=policy_digest,
        local_root=tmp_path.resolve(),
    )


def test_proves_exact_read_only_volume_without_run_job(tmp_path: Path) -> None:
    client = SimpleNamespace(Volume=RecordingVolume, run_job=_run_job)
    proven = prove_read_only_volume(client, _spec(tmp_path))
    assert proven.provider_volume.to_dict() == {
        "type": "bucket",
        "source": "owner/bootstrap-bucket",
        "mountPath": "/workspace/synaptic-bootstrap-input",
        "readOnly": True,
        "path": "prepared/run-123",
    }
    assert inspect.signature(client.run_job).parameters["volumes"]


def test_training_provider_command_is_fixed_no_shell_and_exactly_bound(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    remote = ("--recipe", "/workspace/project/recipe.yaml")
    command = build_training_provider_command(
        spec, remote_argv=remote,
        expected_project_root="/workspace/source/project",
        expected_engine_root="/workspace/source/engine",
        expected_project_commit="1" * 40, expected_engine_commit="2" * 40,
        expected_mode="dual_clone",
    )
    assert command[:4] == ("python", "-I", "-c", FIXED_STDLIB_TRAINING_LAUNCHER)
    assert command[-3:] == ("--", *remote)
    assert all(value not in {"sh", "bash", "cmd", "powershell"} for value in command)
    assert "HF_TOKEN" not in "".join(command)


def test_training_provider_command_rejects_unbound_inputs(tmp_path: Path) -> None:
    with pytest.raises(CloudProviderError, match="provider argv"):
        build_training_provider_command(
            _spec(tmp_path), remote_argv=["--recipe", "x"],  # type: ignore[arg-type]
            expected_project_root="/workspace/source/project",
            expected_engine_root="/workspace/source/engine",
            expected_project_commit="1" * 40, expected_engine_commit="2" * 40,
            expected_mode="dual_clone",
        )

def test_proves_exact_writable_artifact_volume_without_run_job() -> None:
    client = SimpleNamespace(Volume=RecordingVolume, run_job=_run_job)
    spec = HFArtifactVolumeSpec(
        source="owner/artifacts", path="synaptic/training-smoke/v1/experiment/slot",
    )
    volume = prove_writable_artifact_volume(client, spec)
    assert volume.provider_volume.to_dict() == {
        "type": "bucket", "source": "owner/artifacts",
        "path": "synaptic/training-smoke/v1/experiment/slot",
        "mountPath": "/workspace/artifacts", "readOnly": False,
    }


@pytest.mark.parametrize(
    ("source_prefix", "artifact_prefix"),
    [
        ("prepared/run", "prepared/run"),
        ("prepared/run", "prepared/run/artifacts"),
        ("prepared/run/source", "prepared/run"),
    ],
)
def test_rejects_source_artifact_prefix_overlap(tmp_path: Path, source_prefix: str, artifact_prefix: str) -> None:
    source = _spec(tmp_path)
    object.__setattr__(source, "path", source_prefix)
    artifact = HFArtifactVolumeSpec(source=source.source, path=artifact_prefix)
    with pytest.raises(CloudProviderError, match="overlap"):
        validate_disjoint_volume_prefixes(source, artifact)


def test_accepts_disjoint_source_artifact_prefixes(tmp_path: Path) -> None:
    source = _spec(tmp_path)
    artifact = HFArtifactVolumeSpec(source=source.source, path="artifacts/run-123")
    validate_disjoint_volume_prefixes(source, artifact)


@pytest.mark.parametrize("missing", ["Volume", "run_job"])
def test_missing_client_feature_fails_closed_before_run_job(tmp_path: Path, missing: str) -> None:
    values = {"Volume": RecordingVolume, "run_job": _run_job}
    values.pop(missing)
    with pytest.raises(CloudProviderError, match="lacks verified Jobs volume support"):
        prove_read_only_volume(SimpleNamespace(**values), _spec(tmp_path))


def test_kwargs_only_run_job_is_not_accepted_as_semantic_proof(tmp_path: Path) -> None:
    calls = []

    def opaque(**kwargs):
        calls.append(kwargs)

    with pytest.raises(CloudProviderError, match="explicitly support volumes"):
        prove_read_only_volume(SimpleNamespace(Volume=RecordingVolume, run_job=opaque), _spec(tmp_path))
    assert calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("readOnly"),
        lambda value: value.update(readOnly=False),
        lambda value: value.update(readOnly=1),
        lambda value: value.update(mount_path=value.pop("mountPath")),
        lambda value: value.update(unexpected=True),
    ],
)
def test_volume_wire_semantic_drift_fails_closed(tmp_path: Path, mutate) -> None:
    class DriftVolume(RecordingVolume):
        def to_dict(self):
            value = super().to_dict()
            mutate(value)
            return value

    with pytest.raises(CloudProviderError, match="semantics have drifted"):
        prove_read_only_volume(SimpleNamespace(Volume=DriftVolume, run_job=_run_job), _spec(tmp_path))


def test_absent_or_mismatched_preprovisioned_member_fails_before_client_probe(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    (tmp_path / spec.source_lock_path).unlink()
    with pytest.raises(CloudProviderError, match="member is absent"):
        prove_read_only_volume(SimpleNamespace(Volume=RecordingVolume, run_job=_run_job), spec)

    spec = _spec(tmp_path)
    (tmp_path / spec.checkout_policy_path).write_bytes(b"changed\n")
    with pytest.raises(CloudProviderError, match="digest mismatch"):
        prove_read_only_volume(SimpleNamespace(Volume=RecordingVolume, run_job=_run_job), spec)


def test_inline_verifier_is_deterministic_digest_bound_and_verifies_before_import(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    first = build_verified_bootstrap_step(spec)
    second = build_verified_bootstrap_step(spec)
    assert first == second
    assert spec.capsule_manifest_sha256 in first
    assert spec.source_lock_sha256 in first
    assert spec.checkout_policy_sha256 in first
    assert first.index("sha256(raw).hexdigest()!=expected") < first.index("spec_from_file_location")
    assert "subprocess" not in first
    assert "git clone" not in first
    assert "source-lock/v1" not in first
    assert ".synaptic-bootstrap-result.json" in first

    projection = build_runtime_projection_step(
        expected_project_root="/workspace/source/project",
        expected_engine_root="/workspace/source/project/deps/tuner",
        expected_project_commit="1" * 40,
        expected_engine_commit="2" * 40,
        expected_mode="superproject",
    )
    assert "_project-layout" in projection
    assert "--logical-project-root /workspace/project" in projection
    assert "--logical-engine-root /workspace/engine" in projection
    assert "git clone" not in projection


def _execute_inline_verifier(tmp_path: Path, capsule_source: str) -> None:
    capsule_root = tmp_path / "capsule"
    capsule_member = capsule_root / "tuner" / "cloud" / "bootstrap_capsule.py"
    capsule_member.parent.mkdir(parents=True)
    capsule_bytes = capsule_source.encode("utf-8")
    capsule_member.write_bytes(capsule_bytes)
    manifest = {
        "schema_version": "synaptic-bootstrap-capsule/v1",
        "engine_commit": "1" * 40,
        "files": [
            {
                "path": "tuner/cloud/bootstrap_capsule.py",
                "size": len(capsule_bytes),
                "sha256": hashlib.sha256(capsule_bytes).hexdigest(),
                "mode": "0644",
            },
            {
                "path": "tuner/cloud/bootstrap_core.py",
                "size": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
                "mode": "0644",
            },
        ],
        "limits": {},
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    (capsule_root / "synaptic-bootstrap-capsule.json").write_bytes(manifest_bytes)
    source_lock = tmp_path / "source-lock.json"
    checkout_policy = tmp_path / "checkout-policy.json"
    source_lock.write_bytes(b"lock\n")
    checkout_policy.write_bytes(b"policy\n")
    destination = tmp_path / "destination"
    destination.mkdir()
    arguments = [
        "inline-verifier",
        str(capsule_root),
        hashlib.sha256(manifest_bytes).hexdigest(),
        str(source_lock),
        hashlib.sha256(source_lock.read_bytes()).hexdigest(),
        str(checkout_policy),
        hashlib.sha256(checkout_policy.read_bytes()).hexdigest(),
        str(destination),
    ]
    original_argv = sys.argv
    try:
        sys.argv = arguments
        exec(_INLINE_VERIFIER, {"__name__": "__main__"})
    finally:
        sys.argv = original_argv


def test_inline_verifier_registers_dataclass_module_during_execution_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_parent))
    capsule_source = '''from dataclasses import dataclass
import json
import os

@dataclass(frozen=True)
class CapsuleResult:
    returncode: int
    stdout: str
    stderr: str = ""

def invoke_verified_capsule(root, expected, **kwargs):
    destination = kwargs["destination"]
    document = {
        "schema_version": "synaptic-bootstrap-result/v1",
        "project_root": os.path.join(destination, "project"),
        "engine_root": os.path.join(destination, "engine"),
        "project_commit": "1" * 40,
        "engine_commit": "2" * 40,
    }
    return CapsuleResult(0, json.dumps(document, sort_keys=True, separators=(",", ":")) + "\\n")
'''

    assert "synaptic_verified_capsule" not in sys.modules
    _execute_inline_verifier(tmp_path, capsule_source)

    assert "synaptic_verified_capsule" not in sys.modules
    assert not list(scratch_parent.glob("synaptic-hf-loader-*"))
    assert (tmp_path / "destination" / ".synaptic-bootstrap-result.json").is_file()


def test_inline_verifier_removes_dataclass_module_after_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_parent))
    capsule_source = '''from dataclasses import dataclass

@dataclass(frozen=True)
class CapsuleProbe:
    value: str

raise RuntimeError("capsule import failed")
'''

    assert "synaptic_verified_capsule" not in sys.modules
    with pytest.raises(RuntimeError, match="capsule import failed"):
        _execute_inline_verifier(tmp_path, capsule_source)

    assert "synaptic_verified_capsule" not in sys.modules
    assert not list(scratch_parent.glob("synaptic-hf-loader-*"))


def test_inline_verifier_preserves_replacement_module_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_parent))
    capsule_source = '''from dataclasses import dataclass
from types import SimpleNamespace
import json
import os
import sys

@dataclass(frozen=True)
class CapsuleResult:
    returncode: int
    stdout: str
    stderr: str = ""

sys.modules[__name__] = SimpleNamespace(marker="success replacement")

def invoke_verified_capsule(root, expected, **kwargs):
    destination = kwargs["destination"]
    document = {
        "schema_version": "synaptic-bootstrap-result/v1",
        "project_root": os.path.join(destination, "project"),
        "engine_root": os.path.join(destination, "engine"),
        "project_commit": "1" * 40,
        "engine_commit": "2" * 40,
    }
    return CapsuleResult(0, json.dumps(document, sort_keys=True, separators=(",", ":")) + "\\n")
'''

    assert "synaptic_verified_capsule" not in sys.modules
    try:
        _execute_inline_verifier(tmp_path, capsule_source)
        replacement = sys.modules.get("synaptic_verified_capsule")
        assert getattr(replacement, "marker", None) == "success replacement"
        assert not list(scratch_parent.glob("synaptic-hf-loader-*"))
    finally:
        sys.modules.pop("synaptic_verified_capsule", None)


def test_inline_verifier_preserves_replacement_module_after_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_parent))
    capsule_source = '''from dataclasses import dataclass
from types import SimpleNamespace
import sys

@dataclass(frozen=True)
class CapsuleProbe:
    value: str

sys.modules[__name__] = SimpleNamespace(marker="failure replacement")
raise RuntimeError("capsule import failed after replacement")
'''

    assert "synaptic_verified_capsule" not in sys.modules
    try:
        with pytest.raises(RuntimeError, match="capsule import failed after replacement"):
            _execute_inline_verifier(tmp_path, capsule_source)
        replacement = sys.modules.get("synaptic_verified_capsule")
        assert getattr(replacement, "marker", None) == "failure replacement"
        assert not list(scratch_parent.glob("synaptic-hf-loader-*"))
    finally:
        sys.modules.pop("synaptic_verified_capsule", None)


def test_inline_verifier_atomically_rejects_preexisting_module_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_parent))
    existing = SimpleNamespace(marker="preexisting")
    sys.modules["synaptic_verified_capsule"] = existing
    try:
        with pytest.raises(RuntimeError, match="module name is already registered"):
            _execute_inline_verifier(tmp_path, "raise AssertionError('must not execute')\n")
        assert sys.modules["synaptic_verified_capsule"] is existing
        assert not list(scratch_parent.glob("synaptic-hf-loader-*"))
    finally:
        sys.modules.pop("synaptic_verified_capsule", None)


def test_metadata_describes_preprovisioned_regular_member_contract_without_local_path(tmp_path: Path) -> None:
    metadata = transport_metadata(_spec(tmp_path))
    assert metadata["read_only"] is True
    assert metadata["profile"] == "hf_read_only_volume"
    assert "local_root" not in json.dumps(metadata)


def test_generic_job_keeps_no_volume_compatibility_and_redacts_exact_secret() -> None:
    secret = "provider-secret-without-hf-prefix"

    def run_job(**kwargs):
        assert "volumes" not in kwargs
        raise RuntimeError(f"provider echoed {secret}")

    spec = CloudJobSpec(
        provider="hf_jobs", image="image@sha256:test", command=["true"],
        flavor="cpu-basic", secrets={"TOKEN": secret},
    )
    with pytest.raises(CloudProviderError) as caught:
        HFJobExecutor(SimpleNamespace(run_job=run_job)).submit(spec)
    assert secret not in str(caught.value)


def test_executor_rejects_truthy_digest_wrapper_before_run_job(tmp_path: Path) -> None:
    calls = []
    volume = HFVerifiedVolume(
        spec=_spec(tmp_path),
        provider_volume=object(),
        descriptor_sha256="d" * 64,
        provisioning_evidence_sha256="e" * 64,
    )
    spec = CloudJobSpec(
        provider="hf_jobs", image="image@sha256:test", command=["true"],
        flavor="cpu-basic", volumes=(volume,),
    )
    with pytest.raises(CloudProviderError, match="complete closed CONSUMABLE"):
        HFJobExecutor(SimpleNamespace(run_job=lambda **kwargs: calls.append(kwargs))).submit(spec)
    assert calls == []


def test_executor_rejects_unbound_or_raw_source_volumes_before_run_job(tmp_path: Path) -> None:
    calls = []

    def run_job(**kwargs):
        calls.append(kwargs)

    executor = HFJobExecutor(SimpleNamespace(run_job=run_job))
    for volume in (object(), HFVerifiedVolume(_spec(tmp_path), object())):
        spec = CloudJobSpec(
            provider="hf_jobs", image="image@sha256:test", command=["true"],
            flavor="cpu-basic", volumes=(volume,),
        )
        with pytest.raises(CloudProviderError, match="CONSUMABLE"):
            executor.submit(spec)
    assert calls == []


def _bootstrap_result(path: Path, project: Path, engine: Path) -> Path:
    payload = {
        "schema_version": "synaptic-bootstrap-result/v1",
        "project_root": str(project),
        "engine_root": str(engine),
        "project_commit": "1" * 40,
        "engine_commit": "2" * 40,
    }
    path.write_bytes((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"))
    return path


@pytest.mark.parametrize("mode", ["standalone", "superproject", "dual_clone"])
def test_projects_authenticated_topologies_to_relative_logical_aliases_without_following_links(
    tmp_path: Path, mode: str,
) -> None:
    physical = tmp_path / "source"
    project = physical / "project"
    project.mkdir(parents=True)
    if mode == "standalone":
        engine = project
    elif mode == "superproject":
        engine = project / "deps" / "tuner"
        engine.mkdir(parents=True)
    else:
        engine = physical / "engine"
        engine.mkdir(parents=True)
    executable = engine / "entrypoint.py"
    executable.write_text("pass\n", encoding="utf-8")
    executable.chmod(0o755)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    link = engine / "outside-link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    logical_project = tmp_path / "project"
    logical_engine = tmp_path / "engine"
    project_runtime_layout(
        _bootstrap_result(physical / ".synaptic-bootstrap-result.json", project, engine),
        expected_project_root=project,
        expected_engine_root=engine,
        expected_project_commit="1" * 40,
        expected_engine_commit="2" * 40,
        expected_mode=mode,
        logical_project_root=logical_project,
        logical_engine_root=logical_engine,
    )
    assert logical_project.is_symlink() and logical_project.resolve() == project.resolve()
    assert logical_engine.is_symlink() and logical_engine.resolve() == engine.resolve()
    assert not (executable.stat().st_mode & 0o222)
    assert executable.stat().st_mode & 0o111
    assert outside.stat().st_mode & 0o200


def test_projection_rejects_result_drift_and_preexisting_logical_targets(tmp_path: Path) -> None:
    physical = tmp_path / "source"
    project = physical / "project"
    project.mkdir(parents=True)
    result = _bootstrap_result(physical / ".synaptic-bootstrap-result.json", project, project)
    logical_project = tmp_path / "project"
    logical_engine = tmp_path / "engine"
    logical_project.mkdir()
    with pytest.raises(CloudProviderError, match="absent absolute"):
        project_runtime_layout(
            result,
            expected_project_root=project,
            expected_engine_root=project,
            expected_project_commit="1" * 40,
            expected_engine_commit="2" * 40,
            expected_mode="standalone",
            logical_project_root=logical_project,
            logical_engine_root=logical_engine,
        )
    logical_project.rmdir()
    with pytest.raises(CloudProviderError, match="locked topology"):
        project_runtime_layout(
            result,
            expected_project_root=project,
            expected_engine_root=physical / "escape",
            expected_project_commit="1" * 40,
            expected_engine_commit="2" * 40,
            expected_mode="standalone",
            logical_project_root=logical_project,
            logical_engine_root=logical_engine,
        )
    with pytest.raises(CloudProviderError, match="locked commits"):
        project_runtime_layout(
            result,
            expected_project_root=project,
            expected_engine_root=project,
            expected_project_commit="3" * 40,
            expected_engine_commit="2" * 40,
            expected_mode="standalone",
            logical_project_root=logical_project,
            logical_engine_root=logical_engine,
        )
    with pytest.raises(CloudProviderError, match="mode topology"):
        project_runtime_layout(
            result,
            expected_project_root=project,
            expected_engine_root=project,
            expected_project_commit="1" * 40,
            expected_engine_commit="2" * 40,
            expected_mode="dual_clone",
            logical_project_root=logical_project,
            logical_engine_root=logical_engine,
        )


@pytest.mark.skipif(os.name == "nt", reason="hostile shell execution is exercised on POSIX CI")
@pytest.mark.parametrize(
    "hostile_template",
    [
        "engine path; printf injected > {marker}",
        "engine'path; printf injected > {marker}",
        "engine; printf injected > {marker}",
        "engine$(printf injected > {marker})",
        "engine\nprintf injected > {marker}",
    ],
    ids=["space", "quote", "semicolon", "substitution", "newline"],
)
def test_projection_shell_step_does_not_execute_hostile_submodule_path(
    tmp_path: Path, hostile_template: str,
) -> None:
    shell = shutil.which("bash") or shutil.which("sh")
    if shell is None:
        pytest.skip("a POSIX shell is unavailable")
    marker = tmp_path / "injected"
    hostile = hostile_template.format(marker=shlex.quote(str(marker)))
    step = build_runtime_projection_step(
        expected_project_root="/workspace/source/project",
        expected_engine_root=f"/workspace/source/project/deps/{hostile}",
        expected_project_commit="1" * 40,
        expected_engine_commit="2" * 40,
        expected_mode="superproject",
    )

    completed = subprocess.run(
        [shell, "-c", step],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not marker.exists()


def _assert_linked_physical_root_rejected(
    *, physical: Path, project: Path, engine: Path, mode: str, tmp_path: Path,
) -> None:
    with pytest.raises(CloudProviderError, match="links or reparse points"):
        project_runtime_layout(
            _bootstrap_result(physical / ".synaptic-bootstrap-result.json", project, engine),
            expected_project_root=project,
            expected_engine_root=engine,
            expected_project_commit="1" * 40,
            expected_engine_commit="2" * 40,
            expected_mode=mode,
            logical_project_root=tmp_path / "logical-project",
            logical_engine_root=tmp_path / "logical-engine",
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink test")
@pytest.mark.parametrize("linked_component", ["root", "intermediate"])
def test_projection_rejects_posix_linked_physical_roots(
    tmp_path: Path, linked_component: str,
) -> None:
    physical = tmp_path / "source"
    physical.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if linked_component == "root":
        project = physical / "project"
        project.symlink_to(outside, target_is_directory=True)
        engine = project
        mode = "standalone"
    else:
        project = physical / "project"
        project.mkdir()
        (outside / "tuner").mkdir()
        (project / "deps").symlink_to(outside, target_is_directory=True)
        engine = project / "deps" / "tuner"
        mode = "superproject"

    _assert_linked_physical_root_rejected(
        physical=physical, project=project, engine=engine, mode=mode, tmp_path=tmp_path,
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
@pytest.mark.parametrize("linked_component", ["root", "intermediate"])
def test_projection_rejects_windows_junction_physical_roots(
    tmp_path: Path, linked_component: str,
) -> None:
    physical = tmp_path / "source"
    physical.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    project = physical / "project"
    if linked_component == "root":
        link = project
        engine = project
        target = outside
        mode = "standalone"
    else:
        project.mkdir()
        (outside / "tuner").mkdir()
        link = project / "deps"
        engine = link / "tuner"
        target = outside
        mode = "superproject"
    created = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip("directory junction creation is unavailable")
    assert os.path.isjunction(link)

    _assert_linked_physical_root_rejected(
        physical=physical, project=project, engine=engine, mode=mode, tmp_path=tmp_path,
    )
