"""Provider-free qualification of the fixed packaged Modal worker wrapper."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.packaged_dispatch import (
    ModalPackagedVolumeMarker, build_modal_packaged_dispatch,
)
from tuner.execution.providers.modal.coordinator_producer import MODAL_TRAINING_ARTIFACT_BOUNDS_V1
from tuner.execution.providers.modal.packaged_worker import (
    ModalPackagedWorker,
    ModalPackagedWorkerRoots,
    PACKAGED_WORKER_FAILURE_STAGES,
    packaged_worker_failure,
    _PackagedPhaseTrace,
    PACKAGED_PHASES,
)
from tuner.execution.providers.modal.model_snapshot import PERSISTENT_PUBLICATION_DIAGNOSTICS
from tuner.execution.providers.modal.contracts import operation_path
from tuner.execution.providers.modal import volume_root_binding as binding_module
from tuner.execution.providers.modal.volume_root_binding import (
    PUBLICATION_DIAGNOSTICS, VolumeRootBinding,
)
from tuner.runtime.packaged_sft_execution import (
    CHILD_FAILURE_STAGES, PackagedSFTExecutionError, PREPARATION_FAILURE_STAGES,
)

from tests.execution.providers.test_modal_packaged_dispatch import Auth, _case


EXACT_ROLES = (
    "workload_record", "training_lineage", "training_metrics", "final_model",
    "tokenizer",
)


def _assert_failure(result, stage):
    assert result == {
        "schema_version": "synaptic-modal-packaged-worker-result/v2",
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
        "failure_stage": stage,
    }


class Signer:
    def __init__(self) -> None:
        self.calls = []

    def sign(self, purpose, payload, key_ref):
        self.calls.append((purpose, payload, key_ref))
        return b"completion-tag"


class Executor:
    def __init__(
        self,
        *,
        roles=EXACT_ROLES,
        fail=False,
        declared_sizes=None,
        bad_digest=False,
        extra_output=False,
    ) -> None:
        self.roles, self.fail = roles, fail
        self.declared_sizes = declared_sizes
        self.bad_digest = bad_digest
        self.extra_output = extra_output
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("trainer detail must stay private")
        paths = kwargs["paths"]
        records = []
        for index, role in enumerate(self.roles):
            content = f"artifact-{index}".encode()
            name = f"artifact-{index}.bin"
            (paths.artifacts / name).write_bytes(content)
            records.append({
                "role": role,
                "path": name,
                "sha256": (
                    "0" * 64 if self.bad_digest
                    else hashlib.sha256(content).hexdigest()
                ),
                "size": (
                    self.declared_sizes[index]
                    if self.declared_sizes is not None else len(content)
                ),
            })
        if self.extra_output:
            (paths.artifacts / "unlisted.bin").write_bytes(b"unlisted")
        inventory = paths.state / "inventory.json"
        terminal = paths.state / "terminal.json"
        inventory.write_bytes(canonical_bytes({
            "schema_version": "synaptic-artifact-inventory/v1",
            "workload_fingerprint": kwargs["execution_binding"].workload_digest,
            "artifacts": records,
        }))
        terminal.write_bytes(canonical_bytes({"status": "completed"}))
        return SimpleNamespace(inventory_path=inventory, terminal_path=terminal)


def _worker(tmp_path: Path, *, executor=None):
    binding, receipt, workload, policy = _case()
    auth, signer = Auth(), Signer()
    dispatch = build_modal_packaged_dispatch(
        binding, receipt, workload, policy, auth, key_ref="dispatch-key",
        environment=(("PATH", "/usr/bin:/bin"),),
    )
    roots = ModalPackagedWorkerRoots(
        (tmp_path / "control").resolve(),
        (tmp_path / "artifacts").resolve(),
        (tmp_path / "cache").resolve(),
    )
    for root in (roots.control, roots.artifacts, roots.cache):
        root.mkdir()
    staged = roots.artifacts / receipt.path
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"packaged-prepared-input")
    executor = executor or Executor()
    worker = ModalPackagedWorker(
        expected_facts=binding.provider_facts,
        dispatch_verifier=auth,
        trainer_executor=executor,
        evidence_signer=signer,
        roots=roots,
    )
    return binding, dispatch, worker, executor, signer, roots


class _BoundVolume:
    def __init__(self, marker: ModalPackagedVolumeMarker):
        self.volume_id = marker.volume_id
        self.marker_name = marker.marker_name
        self.marker_sha256 = marker.value_sha256
        self.files: dict[str, bytes] = {}
        self.claims: set[str] = set()

    def read_regular(self, path, maximum):
        data = self.files[path]
        assert len(data) <= maximum
        return data

    def claim_directory(self, path):
        assert path not in self.claims
        self.claims.add(path)

    def copy_in_exclusive(self, path, source, *, expected_size, expected_sha256, maximum):
        assert type(source) is str
        data = Path(source).read_bytes()
        assert len(data) == expected_size <= maximum
        assert hashlib.sha256(data).hexdigest() == expected_sha256
        assert path not in self.files
        self.files[path] = data

    def list_regular(self, path, maximum_entries):
        prefix = path + "/"
        return tuple(sorted((name[len(prefix):], len(data)) for name, data in self.files.items()
                            if name.startswith(prefix) and "/" not in name[len(prefix):]))

    def write_exclusive(self, path, content, maximum):
        assert len(content) <= maximum and path not in self.files
        self.files[path] = content


def test_bound_v2_worker_runs_offline_trainer_in_private_scratch_and_publishes_exclusively(tmp_path) -> None:
    binding, receipt, workload, policy = _case()
    facts = binding.provider_facts
    roles = (("control", facts.control_volume_id), ("artifacts", facts.artifact_volume_id))
    if facts.model_cache_volume_id is not None:
        roles += (("model_cache", facts.model_cache_volume_id),)
    markers = tuple(ModalPackagedVolumeMarker(role, volume_id,
                                              ".synaptic-volume-marker-" + hashlib.md5(role.encode()).hexdigest(),
                                              hashlib.sha256(role.encode()).hexdigest())
                    for role, volume_id in roles)
    auth, signer, executor = Auth(), Signer(), Executor()
    dispatch = build_modal_packaged_dispatch(
        binding, receipt, workload, policy, auth, key_ref="dispatch-key",
        environment=(("PATH", "/usr/bin:/bin"),), volume_markers=markers,
    )
    bound = {marker.role: _BoundVolume(marker) for marker in markers}
    bound["artifacts"].files[receipt.path] = b"packaged-prepared-input"
    roots = ModalPackagedWorkerRoots(tmp_path / "control", tmp_path / "artifacts", tmp_path / "cache")
    private = tmp_path / "private"
    private.mkdir()
    worker = ModalPackagedWorker(
        expected_facts=facts, dispatch_verifier=auth, trainer_executor=executor,
        evidence_signer=signer, roots=roots, volume_bindings=bound,
        private_root=private,
    )
    commits = []
    result = worker(dispatch, "fc-1", commit_artifacts=lambda: commits.append("artifacts"),
                    commit_control=lambda: commits.append("control"))
    assert result["status_code"] == "completed"
    assert commits == ["artifacts", "control"]
    paths = executor.calls[0]["paths"]
    assert all(path.is_relative_to(private) for path in (
        paths.prepared_input, paths.artifacts, paths.state, paths.tracking, paths.cache, paths.tmp,
    ))
    effect = binding.command.operation.effect.effect_id
    assert len(bound["artifacts"].list_regular(f"operations/{effect}/output", 6)) == 5
    assert f"operations/{effect}/evidence/packaged-completion.json" in bound["control"].files


@pytest.mark.skipif(sys.platform != "linux", reason="Linux descriptor-bound Volume operations")
@pytest.mark.parametrize("preexisting_parent_role", [None, "artifacts", "control"])
def test_real_bound_completion_claims_only_fresh_submit_parent(
    monkeypatch, preexisting_parent_role: str | None,
) -> None:
    binding, receipt, workload, policy = _case()
    facts = binding.provider_facts
    roles = (("control", facts.control_volume_id), ("artifacts", facts.artifact_volume_id))
    if facts.model_cache_volume_id is not None:
        roles += (("model_cache", facts.model_cache_volume_id),)
    auth, signer, executor = Auth(), Signer(), Executor()
    with ExitStack() as stack:
        base = Path(stack.enter_context(tempfile.TemporaryDirectory(
            prefix="modal-packaged-completion-", dir=Path.home(),
        )))
        base.chmod(0o700)
        mount_root = base / "mounts"
        volume_root = base / "volumes"
        mount_root.mkdir(mode=0o700)
        volume_root.mkdir(mode=0o700)
        monkeypatch.setattr(binding_module, "_PROVIDER_VOLUME_ROOT", str(volume_root))
        markers = []
        targets = {}
        roots_by_role = {}
        bindings = {}
        for role, volume_id in roles:
            target = volume_root / volume_id
            target.mkdir(mode=0o700)
            targets[role] = target
            marker_value = (role.encode() * 32)[:32]
            marker_name = ".synaptic-volume-marker-" + hashlib.md5(role.encode()).hexdigest()
            (target / marker_name).write_bytes(marker_value)
            marker = ModalPackagedVolumeMarker(
                role, volume_id, marker_name, hashlib.sha256(marker_value).hexdigest(),
            )
            markers.append(marker)
            mounted = mount_root / role
            mounted.symlink_to(target, target_is_directory=True)
            roots_by_role[role] = mounted
            bindings[role] = stack.enter_context(VolumeRootBinding.bind(
                root_path=str(mounted), volume_id=volume_id,
                marker_name=marker_name, marker_sha256=marker.value_sha256,
            ))
        staged_input = targets["artifacts"] / receipt.path
        staged_input.parent.mkdir(parents=True)
        staged_input.write_bytes(b"packaged-prepared-input")
        stage_control = targets["control"] / operation_path(
            receipt.stage_effect_id, "control", "stage-claim.v2.json",
        )
        stage_control.parent.mkdir(parents=True)
        stage_control.write_bytes(b"stage-claim")
        submit_effect = binding.command.operation.effect.effect_id
        assert submit_effect != receipt.stage_effect_id
        prior = targets["artifacts"] / operation_path(submit_effect)
        existing = (
            targets[preexisting_parent_role] / operation_path(submit_effect)
            if preexisting_parent_role is not None else None
        )
        if existing is not None:
            existing.mkdir(mode=0o700)
            (existing / "sentinel").write_bytes(b"existing")
        dispatch = build_modal_packaged_dispatch(
            binding, receipt, workload, policy, auth, key_ref="dispatch-key",
            environment=(("PATH", "/usr/bin:/bin"),), volume_markers=tuple(markers),
        )
        private = base / "private"
        private.mkdir(mode=0o700)
        roots = ModalPackagedWorkerRoots(
            roots_by_role["control"], roots_by_role["artifacts"],
            roots_by_role.get("model_cache", base / "unused-cache"),
        )
        worker = ModalPackagedWorker(
            expected_facts=facts, dispatch_verifier=auth, trainer_executor=executor,
            evidence_signer=signer, roots=roots, volume_bindings=bindings,
            private_root=private,
        )
        commits = []
        result = worker(
            dispatch, "fc-1", commit_artifacts=lambda: commits.append("artifacts"),
            commit_control=lambda: commits.append("control"),
        )
        if existing is not None:
            _assert_failure(result, "COMPLETION")
            assert commits == signer.calls == []
            assert (existing / "sentinel").read_bytes() == b"existing"
            assert not (existing / ("output" if preexisting_parent_role == "artifacts" else "state")).exists()
        else:
            assert result["status_code"] == "completed", result
            assert commits == ["artifacts", "control"]
            assert len(signer.calls) == 1
            assert len(list((prior / "output").iterdir())) == 5
            control_submit = targets["control"] / operation_path(submit_effect)
            assert (control_submit / "state" / "runtime-v1-inventory.json").is_file()
            assert (control_submit / "evidence" / "packaged-completion.json").is_file()


def test_bound_worker_rejects_unsigned_v1_before_staged_read(tmp_path) -> None:
    binding, dispatch, _, executor, signer, roots = _worker(tmp_path)
    facts = binding.provider_facts
    markers = (ModalPackagedVolumeMarker("control", facts.control_volume_id,
                                         ".synaptic-volume-marker-" + "1" * 32, "1" * 64),
               ModalPackagedVolumeMarker("artifacts", facts.artifact_volume_id,
                                         ".synaptic-volume-marker-" + "2" * 32, "2" * 64))
    if facts.model_cache_volume_id is not None:
        markers += (ModalPackagedVolumeMarker("model_cache", facts.model_cache_volume_id,
                                             ".synaptic-volume-marker-" + "3" * 32, "3" * 64),)
    private = tmp_path / "private"
    private.mkdir()
    worker = ModalPackagedWorker(
        expected_facts=facts, dispatch_verifier=Auth(), trainer_executor=executor,
        evidence_signer=signer, roots=roots,
        volume_bindings={marker.role: _BoundVolume(marker) for marker in markers},
        private_root=private,
    )
    result = worker(dispatch, "fc-1", commit_artifacts=lambda: None, commit_control=lambda: None)
    _assert_failure(result, "DISPATCH_AUTH")
    assert executor.calls == []


def test_worker_calls_generic_trainer_once_without_modal_objects_or_provider_facts(tmp_path) -> None:
    binding, dispatch, worker, executor, signer, _ = _worker(tmp_path)
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    assert result["status_code"] == "completed"
    assert result["schema_version"] == "synaptic-modal-packaged-worker-result/v1"
    assert "failure_stage" not in result
    assert result["effect_id"] == binding.command.operation.effect.effect_id
    assert len(executor.calls) == len(signer.calls) == 1
    assert events == ["artifacts", "control"]
    call = executor.calls[0]
    assert set(call) == {
        "runtime_release", "provider_binding", "execution_binding",
        "workload_bytes", "artifact_policy", "paths", "environment",
    }
    assert "provider_facts" not in call
    assert all("modal" not in type(value).__module__.lower() for value in call.values())
    assert b'"provider"' not in call["workload_bytes"].lower()


def test_staged_input_mismatch_fails_closed_before_trainer_or_commits(tmp_path) -> None:
    _, dispatch, worker, executor, signer, roots = _worker(tmp_path)
    staged = next(path for path in roots.artifacts.rglob("payload.bin"))
    staged.write_bytes(b"substituted")
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    _assert_failure(result, "STAGED_INPUT")
    assert executor.calls == signer.calls == events == []


def test_trainer_failure_is_closed_and_never_commits_or_signs(tmp_path) -> None:
    executor = Executor(fail=True)
    _, dispatch, worker, _, signer, _ = _worker(tmp_path, executor=executor)
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    _assert_failure(result, "SFT_UNKNOWN")
    assert len(executor.calls) == 1
    assert signer.calls == events == []


def test_worker_rejects_noncanonical_artifact_roles_before_signing_or_commit(tmp_path) -> None:
    executor = Executor(roles=("one", "two", "three", "four", "five"))
    _, dispatch, worker, _, signer, _ = _worker(tmp_path, executor=executor)
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    assert result["status_code"] == "failed"
    assert signer.calls == events == []


def test_worker_rejects_member_above_policy_bound_before_signing_or_commit(tmp_path) -> None:
    executor = Executor(declared_sizes=(MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes + 1, 1, 1, 1, 1))
    _, dispatch, worker, _, signer, _ = _worker(tmp_path, executor=executor)
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    assert result["status_code"] == "failed"
    assert signer.calls == events == []


def test_worker_rejects_aggregate_above_policy_bound_before_signing_or_commit(tmp_path) -> None:
    executor = Executor(declared_sizes=(MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_total_bytes // 5 + 1,) * 5)
    _, dispatch, worker, _, signer, _ = _worker(tmp_path, executor=executor)
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    assert result["status_code"] == "failed"
    assert signer.calls == events == []


def test_worker_rejects_stable_read_digest_mismatch_before_signing_or_commit(
    tmp_path,
) -> None:
    executor = Executor(bad_digest=True)
    _, dispatch, worker, _, signer, _ = _worker(tmp_path, executor=executor)
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    assert result["status_code"] == "failed"
    assert signer.calls == events == []


def test_worker_rejects_unlisted_output_before_signing_or_commit(tmp_path) -> None:
    executor = Executor(extra_output=True)
    _, dispatch, worker, _, signer, _ = _worker(tmp_path, executor=executor)
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    assert result["status_code"] == "failed"
    assert signer.calls == events == []


def test_existing_operation_directories_prevent_automatic_replay(tmp_path) -> None:
    _, dispatch, worker, executor, _, _ = _worker(tmp_path)
    commits = []
    first = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: commits.append("artifacts"),
        commit_control=lambda: commits.append("control"),
    )
    second = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: commits.append("replayed-artifacts"),
        commit_control=lambda: commits.append("replayed-control"),
    )
    assert first["status_code"] == "completed"
    assert second["status_code"] == "failed"
    assert len(executor.calls) == 1
    assert commits == ["artifacts", "control"]


def test_failure_stage_contract_is_closed_and_rejects_dynamic_values() -> None:
    assert PUBLICATION_DIAGNOSTICS == PERSISTENT_PUBLICATION_DIAGNOSTICS
    assert PACKAGED_WORKER_FAILURE_STAGES == frozenset({
        "ENTRYPOINT_SETUP", "ENTRYPOINT_IMPORTS", "ENTRYPOINT_DISPATCH_AUTH",
        "ENTRYPOINT_PROVIDER_ID", "ENTRYPOINT_VOLUME_ID", "ENTRYPOINT_CALL_ID",
        "ENTRYPOINT_MOUNTS", "ENTRYPOINT_WORKER_SETUP",
        "ENTRYPOINT_MOUNT_CONTROL_DIR", "ENTRYPOINT_MOUNT_CONTROL_LINK",
        "ENTRYPOINT_MOUNT_ARTIFACTS_DIR", "ENTRYPOINT_MOUNT_ARTIFACTS_LINK",
        "ENTRYPOINT_MOUNT_MODEL_CACHE_DIR", "ENTRYPOINT_MOUNT_MODEL_CACHE_LINK",
        "DISPATCH_AUTH", "STAGED_INPUT", "PATH_CLAIM",
        "SFT_ADMISSION", "SFT_ADMISSION_CONTRACTS", "SFT_ADMISSION_RELEASE",
        "SFT_ADMISSION_PATHS", "SFT_ADMISSION_INPUT", "SFT_ADMISSION_ENVIRONMENT",
        "SFT_ADMISSION_INVOCATION", "SFT_ADMISSION_COMMITMENT",
        "SFT_PREPARATION", "SFT_PREPARATION_MODEL_UNAVAILABLE",
        "SFT_PREPARATION_MODEL_SDK_ADMISSION", "SFT_PREPARATION_MODEL_INPUT",
        "SFT_PREPARATION_MODEL_WORKSPACE_SETUP", "SFT_PREPARATION_MODEL_METADATA_FETCH",
        "SFT_PREPARATION_MODEL_METADATA_VALIDATION", "SFT_PREPARATION_MODEL_DOWNLOAD",
        "SFT_PREPARATION_MODEL_VERIFICATION", "SFT_PREPARATION_MODEL_PERSISTENT_PUBLICATION",
        "SFT_PREPARATION_MODEL_DESTINATION_COPY",
        "SFT_PREPARATION_MODEL_DESTINATION_VERIFICATION", "SFT_PREPARATION_CACHE_COMMIT",
        "SFT_PREPARATION_PATH", "SFT_PREPARATION_SNAPSHOT_INVENTORY",
        "SFT_REVALIDATION",
        "SFT_INVOCATION", "SFT_TRAINER", "SFT_EVIDENCE", "SFT_POST_TRAINING",
        "SFT_ARTIFACT", "SFT_EVIDENCE_PRIVATE_COPY", "SFT_EVIDENCE_DIRECTORIES",
        "SFT_EVIDENCE_OUTPUT_BINDING", "SFT_EVIDENCE_OUTPUT_INVENTORY",
        "SFT_EVIDENCE_DATASET_BINDING", "SFT_EVIDENCE_PROJECTION_BINDING",
        "SFT_EVIDENCE_OUTPUT_DIRECTORY", "SFT_EVIDENCE_METRICS",
        "SFT_UNKNOWN", "COMPLETION", "ARTIFACT_COMMIT", "CONTROL_COMMIT",
    }) | frozenset("SFT_PREPARATION_MODEL_PERSISTENT_PUBLICATION_" + code
                   for code in PUBLICATION_DIAGNOSTICS) | frozenset(
                       "SFT_" + stage for stage in CHILD_FAILURE_STAGES)
    assert {stage.removeprefix("MODEL_PERSISTENT_PUBLICATION_")
            for stage in PREPARATION_FAILURE_STAGES
            if stage.startswith("MODEL_PERSISTENT_PUBLICATION_")} == PUBLICATION_DIAGNOSTICS
    for stage in PACKAGED_WORKER_FAILURE_STAGES:
        _assert_failure(packaged_worker_failure(stage), stage)
    _assert_failure(packaged_worker_failure("secret/path"), "SFT_UNKNOWN")
    _assert_failure(packaged_worker_failure("ENTRYPOINT_MOUNT_CONTROL_DIR_EXTRA"), "SFT_UNKNOWN")


def test_dispatch_auth_failure_is_closed(tmp_path) -> None:
    _, _, worker, executor, signer, _ = _worker(tmp_path)
    _assert_failure(worker(
        b"private malformed dispatch", "fc-1",
        commit_artifacts=lambda: pytest.fail("unexpected artifact commit"),
        commit_control=lambda: pytest.fail("unexpected control commit"),
    ), "DISPATCH_AUTH")
    assert executor.calls == signer.calls == []


def test_path_claim_failure_is_closed(tmp_path, monkeypatch) -> None:
    from tuner.execution.providers.modal import packaged_worker

    _, dispatch, worker, executor, signer, _ = _worker(tmp_path)
    def refuse_claim(*args):
        raise OSError("private /mount/path")
    monkeypatch.setattr(packaged_worker, "claim_directory", refuse_claim)
    _assert_failure(worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: pytest.fail("unexpected artifact commit"),
        commit_control=lambda: pytest.fail("unexpected control commit"),
    ), "PATH_CLAIM")
    assert executor.calls == signer.calls == []


@pytest.mark.parametrize("sft_stage", [
    "ADMISSION", "ADMISSION_CONTRACTS", "ADMISSION_RELEASE", "ADMISSION_PATHS",
    "ADMISSION_INPUT", "ADMISSION_ENVIRONMENT", "ADMISSION_INVOCATION",
    "ADMISSION_COMMITMENT", "PREPARATION", "REVALIDATION", "INVOCATION", "TRAINER",
    "EVIDENCE", "ARTIFACT",
] + ["PREPARATION_" + stage for stage in sorted(PREPARATION_FAILURE_STAGES)]
  + sorted(CHILD_FAILURE_STAGES))
def test_generic_executor_exact_closed_stage_is_reported(tmp_path, sft_stage) -> None:
    class StageExecutor:
        def execute(self, **kwargs):
            raise PackagedSFTExecutionError(sft_stage)

    _, dispatch, worker, _, signer, _ = _worker(tmp_path, executor=StageExecutor())
    _assert_failure(worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: pytest.fail("unexpected artifact commit"),
        commit_control=lambda: pytest.fail("unexpected control commit"),
    ), "SFT_" + sft_stage)
    assert signer.calls == []


def test_completion_failure_is_closed(tmp_path) -> None:
    _, dispatch, worker, _, signer, _ = _worker(
        tmp_path, executor=Executor(bad_digest=True),
    )
    _assert_failure(worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: pytest.fail("unexpected artifact commit"),
        commit_control=lambda: pytest.fail("unexpected control commit"),
    ), "COMPLETION")
    assert signer.calls == []


@pytest.mark.parametrize("failed_commit,stage", [
    ("artifacts", "ARTIFACT_COMMIT"), ("control", "CONTROL_COMMIT"),
])
def test_commit_failure_is_closed(tmp_path, failed_commit, stage) -> None:
    _, dispatch, worker, _, signer, _ = _worker(tmp_path)
    events = []
    def commit(role):
        events.append(role)
        if role == failed_commit:
            raise OSError("private commit path")
    _assert_failure(worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: commit("artifacts"),
        commit_control=lambda: commit("control"),
    ), stage)
    assert len(signer.calls) == 1
    assert events == (["artifacts"] if failed_commit == "artifacts" else ["artifacts", "control"])


def test_serving_trace_reserves_phase_capacity_and_projects_exact_closed_fields():
    from tuner.execution.providers.modal.packaged_worker import _PackagedPhaseTrace
    lines = []
    trace = _PackagedPhaseTrace(clock=lambda: 1, sink=lines.append)
    metrics = {"running_requests": 2, "waiting_requests": 0, "generation_tokens": 3, "kv_cache_usage": .5}
    for _ in range(200): trace.emit_serving("METRICS", metrics)
    assert len(lines) == 64
    for _ in range(191): trace.emit("CHAT_BATCH", "RETURN")
    trace.emit_serving("CLEANUP", {"cleanup_resolved": False})
    trace.emit_serving("METRICS", metrics)
    assert len(lines) == 256
    record = json.loads(lines[0].removeprefix("SYNAPTIC_SERVING "))
    assert record == {"schema_version": "synaptic-modal-serving-diagnostic/v1", "kind": "METRICS",
        "elapsed_ms": 0, **metrics, "cleanup_resolved": None}
    assert all(len(line.encode()) <= 512 for line in lines)
    assert json.loads(lines[-1].removeprefix("SYNAPTIC_SERVING "))["cleanup_resolved"] is False


@pytest.mark.parametrize("values", [None, {"private": "secret"},
    {"running_requests": True, "waiting_requests": 0, "generation_tokens": 0, "kv_cache_usage": 0},
    {"running_requests": 0, "waiting_requests": 0, "generation_tokens": 0, "kv_cache_usage": float("nan")},
    {"running_requests": 0, "waiting_requests": 0, "generation_tokens": 2**53, "kv_cache_usage": 0}])
def test_serving_trace_hostile_values_never_emitted(values):
    from tuner.execution.providers.modal.packaged_worker import _PackagedPhaseTrace
    lines = []
    _PackagedPhaseTrace(clock=lambda: 0, sink=lines.append).emit_serving("METRICS", values)
    assert lines == []


def test_both_trace_emitters_drop_contention_and_release_after_sink_error():
    from tuner.execution.providers.modal.packaged_worker import _PackagedPhaseTrace
    metrics = {"running_requests": 0, "waiting_requests": 0, "generation_tokens": 1, "kv_cache_usage": 0}
    trace = _PackagedPhaseTrace(clock=lambda: 0, sink=lambda line: (_ for _ in ()).throw(OSError("private sink")))
    trace._lock.acquire()
    try:
        trace.emit("VLLM_CLEANUP", "START")
        trace.emit_serving("CLEANUP", {"cleanup_resolved": True})
        assert trace._count == 0
    finally:
        trace._lock.release()
    trace.emit("VLLM_CLEANUP", "START")
    trace.emit_serving("METRICS", metrics)
    assert trace._count == 2
    assert trace._lock.acquire(blocking=False)
    trace._lock.release()


def test_phase_trace_is_atomic_finite_private_and_capped():
    lines = []
    trace = _PackagedPhaseTrace(clock=lambda: 1.25, sink=lines.append)
    threads = [threading.Thread(target=lambda: [trace.emit("CHAT_REQUEST", "START", 1)
                                                for _ in range(100)]) for _ in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    # Contended diagnostics may be dropped; emitted records remain atomic.
    assert 0 < len(lines) <= 256
    for _ in range(256): trace.emit("CHAT_REQUEST", "START", 1)
    assert len(lines) == 256
    for line in lines:
        assert len(line.encode()) <= 512 and line.startswith("SYNAPTIC_PHASE ")
        document = json.loads(line.removeprefix("SYNAPTIC_PHASE "))
        assert document == {"schema_version": "synaptic-modal-packaged-phase/v1",
            "phase": "CHAT_REQUEST", "edge": "START", "request_ordinal": 1, "elapsed_ms": 0}


@pytest.mark.parametrize("phase,edge,ordinal", [("private path", "START", None),
    ("CHAT_BATCH", "private edge", None), ("CHAT_REQUEST", "START", 33),
    ("CHAT_REQUEST", "START", True), ("CHAT_REQUEST", "START", None),
    ("CHAT_BATCH", "START", 1)])
def test_phase_trace_rejects_unknown_and_hostile_fields(phase, edge, ordinal):
    lines = []
    _PackagedPhaseTrace(clock=lambda: 0, sink=lines.append).emit(phase, edge, ordinal)
    assert lines == []


@pytest.mark.parametrize("now", [float("nan"), float("inf"), -1, 86401, True, "private"])
def test_phase_trace_clock_failures_are_inconclusive(now):
    values = iter([0, now])
    lines = []
    _PackagedPhaseTrace(clock=lambda: next(values), sink=lines.append).emit("CHAT_BATCH", "START")
    assert lines == []


def test_phase_output_failure_preserves_worker_result(tmp_path, monkeypatch):
    from tuner.execution.providers.modal import packaged_worker as module
    _, dispatch, worker, _, _, _ = _worker(tmp_path)
    def reject(*_args, **_kwargs): raise OSError("private output failure")
    monkeypatch.setattr(module, "print", reject, raising=False)
    result = worker(dispatch, "fc-1", commit_artifacts=lambda: None, commit_control=lambda: None)
    assert result["status_code"] == "completed"


def test_same_job_phase_edges_bracket_commits_before_evaluation(tmp_path, monkeypatch, capsys):
    from tuner.execution.providers.modal import packaged_worker as module
    from tuner.runtime import post_training_eval
    events = []
    class CallbackExecutor(Executor):
        def execute(self, **kwargs):
            result = super().execute(**kwargs)
            context = SimpleNamespace(validate=lambda: events.append("validate"),
                base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
                environment={}, python_executable=sys.executable, bindings={})
            kwargs["on_training_complete"](result, context)
            return result
    _, raw_dispatch, worker, _, _, _ = _worker(tmp_path, executor=CallbackExecutor())
    dispatch = module.parse_modal_packaged_dispatch(raw_dispatch, Auth())
    workload = json.loads(dispatch.workload_bytes)
    workload["configuration"]["document"]["post_training"] = {"mode": "same_job"}
    # The fixture's immutable binding has no evaluation recipe. Isolate callback
    # sequencing after admission; dispatch authentication is tested separately.
    admitted = SimpleNamespace(**{field.name: getattr(dispatch, field.name) for field in fields(dispatch)})
    admitted.workload_bytes = canonical_bytes(workload)
    admitted.submit_command = dispatch.submit_command
    monkeypatch.setattr(module, "parse_modal_packaged_dispatch", lambda *_args: admitted)
    def evaluate(*_args, **kwargs):
        assert events == ["artifacts", "control", "validate"]
        assert callable(kwargs["phase_callback"])
        events.append("evaluate")
        return {}
    monkeypatch.setattr(post_training_eval, "execute_post_training_evaluation", evaluate)
    monkeypatch.setattr(ModalPackagedWorker, "_publish_evaluation", lambda *_args: events.append("publish"))
    result = worker(raw_dispatch, "fc-1", commit_artifacts=lambda: events.append("artifacts"),
                    commit_control=lambda: events.append("control"))
    assert result["status_code"] == "completed"
    assert events == ["artifacts", "control", "validate", "evaluate", "publish", "artifacts"]
    records = [json.loads(line.removeprefix("SYNAPTIC_PHASE "))
               for line in capsys.readouterr().out.splitlines()]
    pairs = [(record["phase"], record["edge"]) for record in records]
    assert pairs == [("TRAINER_EXECUTE", "START"), *[(phase, edge) for phase in (
        "TRAINING_PUBLICATION", "TRAINING_ARTIFACT_COMMIT", "TRAINING_CONTROL_COMMIT",
        "EVALUATION_IDENTITY_VALIDATE", "EVALUATION_PUBLICATION", "EVALUATION_ARTIFACT_COMMIT")
        for edge in ("START", "RETURN")], ("TRAINER_EXECUTE", "RETURN")]
    assert all(record["phase"] in PACKAGED_PHASES for record in records)
