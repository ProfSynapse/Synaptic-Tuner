"""Provider-free qualification of the fixed packaged Modal worker wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.packaged_dispatch import build_modal_packaged_dispatch
from tuner.execution.providers.modal.packaged_worker import (
    ModalPackagedWorker,
    ModalPackagedWorkerRoots,
)

from tests.execution.providers.test_modal_packaged_dispatch import Auth, _case


EXACT_ROLES = (
    "workload_record", "training_lineage", "training_metrics", "final_model",
    "tokenizer",
)


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


def test_worker_calls_generic_trainer_once_without_modal_objects_or_provider_facts(tmp_path) -> None:
    binding, dispatch, worker, executor, signer, _ = _worker(tmp_path)
    events = []
    result = worker(
        dispatch, "fc-1",
        commit_artifacts=lambda: events.append("artifacts"),
        commit_control=lambda: events.append("control"),
    )
    assert result["status_code"] == "completed"
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
    assert result == {
        "schema_version": "synaptic-modal-packaged-worker-result/v1",
        "effect_id": "unavailable", "status_code": "failed",
        "completion_sha256": "0" * 64,
    }
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
    assert result["status_code"] == "failed"
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
