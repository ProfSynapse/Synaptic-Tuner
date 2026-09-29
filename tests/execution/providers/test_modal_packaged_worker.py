"""Provider-free qualification of the fixed packaged Modal worker wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.packaged_dispatch import (
    ModalPackagedVolumeMarker, build_modal_packaged_dispatch,
)
from tuner.execution.providers.modal.packaged_worker import (
    ModalPackagedWorker,
    ModalPackagedWorkerRoots,
    PACKAGED_WORKER_FAILURE_STAGES,
    packaged_worker_failure,
)
from tuner.execution.providers.modal.model_snapshot import PERSISTENT_PUBLICATION_DIAGNOSTICS
from tuner.execution.providers.modal.volume_root_binding import PUBLICATION_DIAGNOSTICS
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
        data = source.read_bytes()
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


def test_worker_rejects_member_above_192_mib_before_signing_or_commit(tmp_path) -> None:
    executor = Executor(declared_sizes=(192 * 1024 * 1024 + 1, 1, 1, 1, 1))
    _, dispatch, worker, _, signer, _ = _worker(tmp_path, executor=executor)
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    assert result["status_code"] == "failed"
    assert signer.calls == events == []


def test_worker_rejects_aggregate_above_256_mib_before_signing_or_commit(tmp_path) -> None:
    executor = Executor(declared_sizes=(60 * 1024 * 1024,) * 5)
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
        "SFT_INVOCATION", "SFT_TRAINER", "SFT_EVIDENCE", "SFT_ARTIFACT",
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
