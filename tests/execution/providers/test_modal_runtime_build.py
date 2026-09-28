"""Provider-free tests for Modal-built runtime evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from tuner.execution.providers.modal.runtime_build import (
    ModalBuildCandidateV1,
    build_modal_runtime_release_v2,
    capture_modal_build_candidate,
    plan_modal_build_material,
)


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_REVISION = "1" * 40


def _candidate() -> ModalBuildCandidateV1:
    build_inputs = {
        "sdk_version": "1.5.4",
        "python_executable": "/opt/unsloth-venv/bin/python3",
        "commands": ["engine-owned-install-command"],
        "installer_flags": ["--no-index", "--no-deps", "--require-hashes"],
        "wheels": [
            {"filename": "synaptic_tuner-1.1.0-py3-none-any.whl", "sha256": _A},
            {"filename": "bootstrap-1.0-py3-none-any.whl", "sha256": _B},
        ],
        "builder_policy": "synaptic-modal-wheel-build/v1",
    }
    material = {
        "kind": "modal_build",
        "base_image": {"reference": f"docker.io/fixture/base@sha256:{_C}", "digest": _C},
        "build_inputs": build_inputs,
        "build_inputs_digest": hashlib.sha256(
            json.dumps(build_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
    }
    measured = {
        "package": {"name": "synaptic-tuner", "version": "1.1.0", "digest": _A, "source_provenance_digest": _D},
        "worker": {"entrypoint": "tuner.runtime.packaged_training_worker:main", "closure_digest": _B},
        "python": {"implementation": "cpython", "version": "3.12.3", "executable": "/opt/unsloth-venv/bin/python3", "executable_digest": _C},
        "installed_distributions": {"digest": _D, "count": 327},
        "platform": {"system": "linux", "machine": "x86_64", "cuda_version": None, "runtime_facts": {"python_cache_tag": "cpython-312"}},
        "contracts": {"workload_schema": "synaptic-sft-workload/v1", "prepared_input_schema": "synaptic-prepared-training-input/v1", "artifact_contract_schema": "synaptic-sft-artifacts/v1"},
        "capabilities": {"compatibility": {"methods": ["sft"], "models": [{"ref": "Qwen/Qwen3.5-4B", "revision": _REVISION}], "dataset_formats": ["syntunia-sft-row/v2"]}},
        "build_inputs_digest": _A,
    }
    document = {"schema_version": "synaptic-modal-build-capture/v1", "image_id": "im-exact", "inspector_sha256": _C, "measured": measured}
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    return ModalBuildCandidateV1.from_capture(material=material, raw=raw, expected_image_id="im-exact", expected_inspector_sha256=_C,
                                               expected_build_inputs_digest=_A)


def test_release_is_issued_after_capture_and_binds_measured_inventory() -> None:
    candidate = _candidate()
    release = build_modal_runtime_release_v2(candidate, release_ref="runtime:modal-fixture")
    assert release.material == candidate.material
    assert release.package_digest == _A
    assert release.installed_distribution_count == 327
    assert release.to_dict()["material"]["base_image"]["digest"] == _C
    assert "image" not in release.to_dict()
    candidate.validate_release(release, require_builder=False)
    with pytest.raises(ValueError, match="provider-build attestation"):
        candidate.validate_release(release)


def test_v2_builder_lock_is_required_and_bound_to_release() -> None:
    legacy = _candidate()
    material = legacy.material
    material["build_inputs"]["builder_policy"] = "synaptic-modal-wheel-build/v2"
    material["build_inputs"]["builder_lock_sha256"] = _D
    material["build_inputs_digest"] = hashlib.sha256(
        json.dumps(material["build_inputs"], sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode()
    ).hexdigest()
    candidate = ModalBuildCandidateV1.from_capture(
        material=material, raw=legacy._capture_raw, expected_image_id="im-exact",
        expected_inspector_sha256=_C, expected_build_inputs_digest=_A,
    )
    release = build_modal_runtime_release_v2(candidate, release_ref="runtime:modal-locked")
    assert release.material["build_inputs"]["builder_lock_sha256"] == _D
    missing = release.to_dict()
    del missing["material"]["build_inputs"]["builder_lock_sha256"]
    from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV2
    with pytest.raises(ValueError, match="build inputs"):
        PackagedTrainingRuntimeReleaseV2.from_dict(missing)


def test_capture_rejects_provider_image_substitution() -> None:
    candidate = _candidate()
    document = {"schema_version": "synaptic-modal-build-capture/v1", "image_id": "im-other", "inspector_sha256": _C, "measured": candidate.measured}
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError, match="identity differs"):
        ModalBuildCandidateV1.from_capture(material=candidate.material, raw=raw,
                                            expected_image_id="im-exact", expected_inspector_sha256=_C,
                                            expected_build_inputs_digest=_A)


def test_capture_rejects_build_input_digest_mismatch() -> None:
    candidate = _candidate()
    with pytest.raises(ValueError, match="build inputs"):
        ModalBuildCandidateV1.from_capture(material=candidate.material, raw=candidate._capture_raw,
                                            expected_image_id="im-exact", expected_inspector_sha256=_C,
                                            expected_build_inputs_digest=_B)


def test_candidate_nested_evidence_is_copy_on_read_and_capture_digest_is_rechecked() -> None:
    candidate = _candidate()
    candidate.material["build_inputs"]["commands"].append("untrusted")
    candidate.measured["package"]["digest"] = _B
    assert candidate.material["build_inputs"]["commands"] == ["engine-owned-install-command"]
    assert candidate.measured["package"]["digest"] == _A
    release = build_modal_runtime_release_v2(candidate, release_ref="runtime:modal-fixture")
    candidate.validate_release(release, require_builder=False)
    from dataclasses import replace
    tampered = replace(candidate, capture_digest=_B)
    with pytest.raises(ValueError, match="capture changed"):
        tampered.validate_release(release, require_builder=False)


@pytest.mark.parametrize("mutate_lock_after_capture", [False, True])
def test_build_claim_precedes_provider_effect_and_capture_cleans_up(
        monkeypatch, tmp_path, mutate_lock_after_capture) -> None:
    from pathlib import Path
    from tuner.execution.providers.modal import runtime_build

    profile = Path("Trainers/image_profiles/qwen35_4b_packaged_sft_3360351c/profile.yaml")
    source_wheel = profile.parent / "synaptic_tuner-1.1.0-py3-none-any.whl"
    events: list[str] = []

    def prepare(_source, destination, *, expected_source_commit, builder_cache_root):
        assert len(expected_source_commit) == 40
        assert builder_cache_root is None
        target = destination / source_wheel.name
        target.write_bytes(source_wheel.read_bytes())
        return target, hashlib.sha256(target.read_bytes()).hexdigest()

    monkeypatch.setattr(runtime_build, "prepare_current_source_wheel", prepare)

    class Image:
        object_id = "im-exact"
        files: dict[str, bytes] = {}

        @classmethod
        def from_registry(cls, reference):
            events.append("image")
            assert reference.startswith("docker.io/")
            return cls()

        def entrypoint(self, value):
            assert value == []
            return self

        def add_local_file(self, local, remote, *, copy):
            assert copy is True
            self.files[remote] = Path(local).read_bytes()
            return self

        def run_commands(self, *commands):
            events.append("install")
            assert len(commands) == 1 and "--require-hashes" in commands[0]
            return self

        def build(self, app):
            assert app == "scoped-app"
            events.append("build")
            return self

    class App:
        @staticmethod
        def lookup(name, **kwargs):
            events.append("lookup")
            assert kwargs["create_if_missing"] is False
            assert kwargs["client"] is client
            assert kwargs["environment_name"] == "production"
            return "scoped-app"

    class Sandbox:
        def __init__(self, raw):
            self.stdout = [raw]
            self.returncode = 0
            self.terminated = False

        @classmethod
        def create(cls, *argv, **kwargs):
            events.append("sandbox")
            assert kwargs["image"].object_id == "im-exact"
            assert kwargs["block_network"] is True
            assert kwargs["client"] is client
            measured = json.loads(json.dumps(_candidate().measured))
            inputs = Image.files["/opt/synaptic-runtime/build-inputs.json"]
            measured["build_inputs_digest"] = hashlib.sha256(inputs).hexdigest()
            measured["package"]["digest"] = hashlib.sha256(Image.files["/opt/synaptic-runtime/" + source_wheel.name]).hexdigest()
            document = {
                "schema_version": "synaptic-modal-build-capture/v1",
                "image_id": "im-exact",
                "inspector_sha256": hashlib.sha256(runtime_build.stable_read(
                    Path(runtime_build.__file__).resolve().parents[3] / "runtime" / "modal_build_inspector.py", 128 * 1024
                )).hexdigest(),
                "measured": measured,
            }
            raw = (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
            return cls(raw)

        def wait(self, *, raise_on_termination):
            assert raise_on_termination is False

        def terminate(self, *, wait):
            events.append("terminate")
            assert wait is False
            self.terminated = True

        def poll(self):
            events.append("poll")
            return 0 if self.terminated else None

    class SDK:
        __version__ = "1.5.4"

    SDK.Image = Image
    SDK.App = App
    SDK.Sandbox = Sandbox

    client = object()
    intent = plan_modal_build_material(profile)["intent_digest"]
    if mutate_lock_after_capture:
        original = runtime_build.builder_lock_bytes()
        reads = iter((original, original + b"changed", original + b"changed"))
        monkeypatch.setattr(runtime_build, "builder_lock_bytes", lambda: next(reads))
        with pytest.raises(runtime_build.ModalBuildStageFailure) as caught:
            capture_modal_build_candidate(
                sdk=SDK(), client=client, profile_path=profile,
                build_claim=lambda: events.append("claim"), app_name="training-build",
                environment_name="production", expected_intent_digest=intent,
            )
        assert (caught.value.stage, caught.value.reason) == ("BUILD_INPUTS", "INVALID")
        assert events == []
        return
    candidate = capture_modal_build_candidate(
        sdk=SDK(), client=client, profile_path=profile,
        build_claim=lambda: events.append("claim"), app_name="training-build",
        environment_name="production", expected_intent_digest=intent,
    )
    assert candidate.image_id == "im-exact"
    assert events == ["claim", "lookup", "image", "install", "build", "sandbox", "terminate", "poll"]


def test_ambiguous_late_sandbox_create_retains_cleanup_lease() -> None:
    import threading
    import time
    from tuner.execution.providers.modal.runtime_build import _bounded

    release = threading.Event()
    cleaned = threading.Event()
    sandbox = object()

    def create():
        assert release.wait(2)
        return sandbox

    with pytest.raises(RuntimeError, match="create_ambiguous"):
        _bounded(create, deadline=time.monotonic() + 0.01,
                 code="create_ambiguous", late_cleanup=lambda value: cleaned.set() if value is sandbox else None)
    release.set()
    assert cleaned.wait(2)


def test_bounded_failure_separates_timeout_and_operation_without_provider_text() -> None:
    import time
    from tuner.execution.providers.modal.runtime_build import (
        ModalBoundedOperationFailure, _bounded,
    )

    with pytest.raises(ModalBoundedOperationFailure) as timeout:
        _bounded(lambda: pytest.fail("expired call must not start"),
                 deadline=time.monotonic() - 1, code="fixed_image_build_code")
    assert (timeout.value.reason, str(timeout.value)) == ("TIMEOUT", "fixed_image_build_code")

    def hostile():
        raise ValueError("token=must-not-escape")

    with pytest.raises(ModalBoundedOperationFailure) as failed:
        _bounded(hostile, deadline=time.monotonic() + 2, code="fixed_image_build_code")
    assert (failed.value.reason, str(failed.value)) == ("OPERATION_FAILED", "fixed_image_build_code")
    assert "must-not-escape" not in str(failed.value)
    assert failed.value.__cause__ is None


def test_capture_cleanup_poll_deadline_reports_timeout(monkeypatch) -> None:
    from tuner.execution.providers.modal import runtime_build

    ticks = iter((0.0, 0.0, 30.0))
    monkeypatch.setattr(runtime_build.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runtime_build.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runtime_build, "_bounded", lambda operation, **_kwargs: operation())

    class Sandbox:
        def terminate(self, *, wait):
            assert wait is False

        def poll(self):
            return None

    with pytest.raises(runtime_build.ModalBoundedOperationFailure) as caught:
        runtime_build._cleanup_sandbox(Sandbox())
    assert caught.value.reason == "TIMEOUT"
    assert str(caught.value) == "modal_training_capture_cleanup_unresolved"


@pytest.mark.parametrize("output,returncode,reason", [
    (b"", 2, "INSPECTOR_REJECTED"),
    (b"x" * (128 * 1024 + 1), 0, "OUTPUT_INVALID"),
], ids=["inspector-exit", "oversized-output"])
def test_capture_output_reports_only_fixed_inspector_reason(output, returncode, reason) -> None:
    import time
    from tuner.execution.providers.modal.runtime_build import (
        ModalBoundedOperationFailure, _bounded, _capture_output,
    )

    class Sandbox:
        stdout = (output,)

        def wait(self, *, raise_on_termination):
            assert raise_on_termination is False

    sandbox = Sandbox()
    sandbox.returncode = returncode
    with pytest.raises(ModalBoundedOperationFailure) as caught:
        _bounded(lambda: _capture_output(sandbox), deadline=time.monotonic() + 2,
                 code="modal_training_capture_failed")
    assert (caught.value.reason, str(caught.value)) == (reason, "modal_training_capture_failed")


@pytest.mark.parametrize("phase,fault,reason", [
    ("HEAD_BEFORE", "unavailable", "HEAD_BEFORE_UNAVAILABLE"),
    ("HEAD_BEFORE", "mismatch", "HEAD_BEFORE_MISMATCH"),
    ("STATUS_BEFORE", "timeout", "STATUS_BEFORE_TIMEOUT"),
    ("STATUS_BEFORE", "unavailable", "STATUS_BEFORE_UNAVAILABLE"),
    ("STATUS_BEFORE", "dirty", "STATUS_BEFORE_DIRTY"),
    ("HEAD_AFTER", "unavailable", "HEAD_AFTER_UNAVAILABLE"),
    ("HEAD_AFTER", "mismatch", "HEAD_AFTER_MISMATCH"),
    ("STATUS_AFTER", "timeout", "STATUS_AFTER_TIMEOUT"),
    ("STATUS_AFTER", "unavailable", "STATUS_AFTER_UNAVAILABLE"),
    ("STATUS_AFTER", "dirty", "STATUS_AFTER_DIRTY"),
])
def test_source_state_substages_are_closed(monkeypatch, tmp_path, phase, fault, reason) -> None:
    from tuner.execution.providers.modal import runtime_build

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    output = tmp_path / "output"
    output.mkdir()
    commit = "a" * 40
    counts = {"HEAD": 0, "STATUS": 0}

    def run(command, **_kwargs):
        if "rev-parse" in command:
            assert _kwargs["timeout"] == 30
            counts["HEAD"] += 1
            current = "HEAD_BEFORE" if counts["HEAD"] == 1 else "HEAD_AFTER"
            if phase == current and fault == "unavailable":
                raise subprocess.TimeoutExpired(command, 30, stderr=b"token=must-not-escape")
            observed = "b" * 40 if phase == current and fault == "mismatch" else commit
            return subprocess.CompletedProcess(command, 0, observed.encode() + b"\n", b"")
        if "status" in command:
            assert _kwargs["timeout"] == 120
            counts["STATUS"] += 1
            current = "STATUS_BEFORE" if counts["STATUS"] == 1 else "STATUS_AFTER"
            if phase == current and fault == "timeout":
                raise subprocess.TimeoutExpired(command, 120, stderr=b"token=must-not-escape")
            if phase == current and fault == "unavailable":
                return subprocess.CompletedProcess(command, 2, b"", b"token=must-not-escape")
            dirty = b" M token=must-not-escape\n" if phase == current and fault == "dirty" else b""
            return subprocess.CompletedProcess(command, 0, dirty, b"")
        if "archive" in command:
            return subprocess.CompletedProcess(command, 0, b"not-needed", b"")
        pytest.fail("wheel build must not start after source-state failure")

    monkeypatch.setattr(runtime_build.subprocess, "run", run)
    with pytest.raises(runtime_build.SourceWheelFailure) as caught:
        runtime_build.prepare_current_source_wheel(
            source, output, expected_source_commit=commit,
        )
    assert caught.value.reason == reason
    assert str(caught.value) == "source_wheel_unavailable"
    assert caught.value.__cause__ is None
    assert "must-not-escape" not in str(caught.value)


def test_source_state_invalid_input_is_closed_before_git(monkeypatch, tmp_path) -> None:
    from tuner.execution.providers.modal import runtime_build

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(runtime_build.subprocess, "run",
                        lambda *_args, **_kwargs: pytest.fail("git must not start"))
    with pytest.raises(runtime_build.SourceWheelFailure) as caught:
        runtime_build.prepare_current_source_wheel(
            source, output, expected_source_commit="invalid-commit",
        )
    assert (caught.value.reason, str(caught.value)) == (
        "INPUT_INVALID", "source_wheel_unavailable",
    )
    assert caught.value.__cause__ is None


def test_wheel_archive_rejects_head_change_before_builder(monkeypatch, tmp_path) -> None:
    from tuner.execution.providers.modal import runtime_build

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    output = tmp_path / "output"
    output.mkdir()
    first, second = "a" * 40, "b" * 40
    heads = iter((first, second))
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, next(heads).encode() + b"\n", b"")
        if "status" in command:
            assert "--untracked-files=no" in command
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if "archive" in command:
            assert command[3:] == [
                "archive", "--format=tar", first,
                "pyproject.toml", "README.md", "LICENSE", "tuner", "synaptic_tuner",
                "shared", "SynthChat", "Evaluator", "MechInterp", "Trainers",
            ]
            return subprocess.CompletedProcess(command, 0, b"not-needed", b"")
        pytest.fail("builder must not start after HEAD changes")

    monkeypatch.setattr(runtime_build.subprocess, "run", run)
    with pytest.raises(runtime_build.SourceWheelFailure) as caught:
        runtime_build.prepare_current_source_wheel(
            source, output, expected_source_commit=first,
        )
    assert caught.value.reason == "HEAD_AFTER_MISMATCH"
    assert sum("rev-parse" in call for call in calls) == 2


def test_wheel_archive_failure_is_specific_and_builder_does_not_start(monkeypatch, tmp_path) -> None:
    from tuner.execution.providers.modal import runtime_build

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    output = tmp_path / "output"
    output.mkdir()
    commit = "a" * 40

    def run(command, **_kwargs):
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, commit.encode() + b"\n", b"")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if "archive" in command:
            assert command[5] == commit
            assert "tests" not in command and "Datasets" not in command
            return subprocess.CompletedProcess(command, 0, b"", b"")
        pytest.fail("builder must not start without an archive")

    monkeypatch.setattr(runtime_build.subprocess, "run", run)
    with pytest.raises(runtime_build.SourceArchiveInvalid, match="source archive failed"):
        runtime_build.prepare_current_source_wheel(source, output, expected_source_commit=commit)


def test_wheel_archive_rejects_tracked_dirty_source(monkeypatch, tmp_path) -> None:
    from tuner.execution.providers.modal import runtime_build

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    output = tmp_path / "output"
    output.mkdir()
    commit = "a" * 40
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, commit.encode() + b"\n", b"")
        if "status" in command:
            assert "--untracked-files=no" in command
            return subprocess.CompletedProcess(command, 0, b" M tuner/runtime/releases.py\n", b"")
        pytest.fail("dirty tracked source must not be archived")

    monkeypatch.setattr(runtime_build.subprocess, "run", run)
    with pytest.raises(runtime_build.SourceWheelFailure) as caught:
        runtime_build.prepare_current_source_wheel(
            source, output, expected_source_commit=commit,
        )
    assert caught.value.reason == "STATUS_BEFORE_DIRTY"
    assert len(calls) == 2


@pytest.mark.parametrize("failure,reason", [
    ("builder", "BUILDER_SETUP_FAILED"),
    ("pip_timeout", "OFFLINE_WHEEL_TIMEOUT"),
    ("pip_exit", "OFFLINE_WHEEL_FAILED"),
    ("inventory", "WHEEL_INVENTORY_INVALID"),
])
def test_source_wheel_local_substages_are_closed(monkeypatch, tmp_path, failure, reason) -> None:
    from contextlib import nullcontext
    from tuner.execution.providers.modal import runtime_build

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    output = tmp_path / "output"
    output.mkdir()
    commit = "a" * 40

    def run(command, **_kwargs):
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, commit.encode() + b"\n", b"")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if "archive" in command:
            return subprocess.CompletedProcess(command, 0, b"bounded-tar", b"")
        if failure == "pip_timeout":
            raise subprocess.TimeoutExpired(command, 300, stderr=b"token=must-not-escape")
        return subprocess.CompletedProcess(command, 2 if failure == "pip_exit" else 0,
                                           b"", b"token=must-not-escape")

    def builder(_scratch, _cache):
        if failure == "builder":
            raise ValueError("token=must-not-escape")
        return tmp_path / "builder-python"

    monkeypatch.setattr(runtime_build.subprocess, "run", run)
    monkeypatch.setattr(runtime_build.tarfile, "open", lambda **_kwargs: nullcontext(()))
    monkeypatch.setattr(runtime_build, "create_offline_wheel_builder", builder)
    with pytest.raises(runtime_build.SourceWheelFailure) as caught:
        runtime_build.prepare_current_source_wheel(source, output, expected_source_commit=commit)
    assert caught.value.reason == reason
    assert str(caught.value) == "source_wheel_unavailable"
    assert "must-not-escape" not in str(caught.value)


def test_source_wheel_invalid_local_output_is_setup_failure(monkeypatch, tmp_path) -> None:
    from tuner.execution.providers.modal import runtime_build

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    monkeypatch.setattr(runtime_build.subprocess, "run",
                        lambda *_args, **_kwargs: pytest.fail("git must not start"))
    with pytest.raises(runtime_build.SourceWheelFailure) as caught:
        runtime_build.prepare_current_source_wheel(
            source, tmp_path / "missing-output", expected_source_commit="a" * 40,
        )
    assert caught.value.reason == "BUILDER_SETUP_FAILED"


@pytest.mark.parametrize("failure", ["scratch_mkdir", "source_write"])
def test_source_wheel_scratch_io_is_not_archive_invalid(monkeypatch, tmp_path, failure) -> None:
    import io
    from contextlib import nullcontext
    from tuner.execution.providers.modal import runtime_build

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    output = tmp_path / "output"
    output.mkdir()
    commit = "a" * 40

    def run(command, **_kwargs):
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, commit.encode() + b"\n", b"")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if "archive" in command:
            return subprocess.CompletedProcess(command, 0, b"bounded-tar", b"")
        pytest.fail("builder must not start after local I/O failure")

    class Member:
        name = "member.txt"
        size = 4

        def isfile(self):
            return True

        def isdir(self):
            return False

    class Archive:
        def __iter__(self):
            return iter((Member(),))

        def extractfile(self, _member):
            return io.BytesIO(b"data")

    original_mkdir = runtime_build.Path.mkdir
    original_write = runtime_build.Path.write_bytes

    def mkdir(path, *args, **kwargs):
        if failure == "scratch_mkdir" and path.name == "source":
            raise OSError("token=must-not-escape")
        return original_mkdir(path, *args, **kwargs)

    def write(path, data):
        if failure == "source_write" and path.name == "member.txt":
            raise OSError("token=must-not-escape")
        return original_write(path, data)

    monkeypatch.setattr(runtime_build.subprocess, "run", run)
    monkeypatch.setattr(runtime_build.tarfile, "open", lambda **_kwargs: nullcontext(Archive()))
    monkeypatch.setattr(runtime_build.Path, "mkdir", mkdir)
    monkeypatch.setattr(runtime_build.Path, "write_bytes", write)
    with pytest.raises(runtime_build.SourceWheelFailure) as caught:
        runtime_build.prepare_current_source_wheel(source, output, expected_source_commit=commit)
    assert caught.value.reason == "BUILDER_SETUP_FAILED"
    assert str(caught.value) == "source_wheel_unavailable"
