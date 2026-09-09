from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.providers.modal.contracts import EXACT_ARTIFACT_ROLES, operation_path
from tuner.execution.providers.modal.contracts import TerminalEvidenceV1, _object
from tuner.execution.providers.modal.coordinator_dispatch import build_modal_worker_dispatch
from tuner.execution.providers.modal.coordinator_producer import MountedModalCoordinatorProducer
from tuner.execution.providers.modal.coordinator_wire import ModalWorkerLaunchExpectation
from tuner.execution.providers.modal.coordinator_worker import (
    ModalWorkerStaticExpectation, admit_modal_worker,
)
from tuner.execution.providers.modal.coordinator_logs import ModalCoordinatorLogChunk
from tuner.execution.providers.modal.manifest import CompletionManifestV1
from tuner.execution.providers.modal.mounted_io import list_regular_sizes
from tuner.execution.providers.modal.worker_ports import ModalProcessResult
from tests.execution.providers.modal_coordinator_fixtures import real_launch_bundle_case


def _invocation(monkeypatch):
    shared = real_launch_bundle_case(monkeypatch)
    envelope, staged, bundle = shared["envelope"], shared["material"], shared["bundle"]
    submit = parse_exact_command(envelope.submit_binding.command_bytes)
    prep, selection = submit.preparation, envelope.submit_binding.deployment.selection
    volumes = json.loads(envelope.submit_binding.preparation_snapshot)["configuration"]["profile"]["volumes"]
    expectation = ModalWorkerLaunchExpectation(
        submit.canonical_bytes, envelope.submit_binding.deployment_bytes,
        prep.provider.provider_id, prep.provider.profile_ref, prep.scope.account_ref,
        prep.scope.namespace_ref, selection.app_name, selection.function_name,
        staged.control_volume_id, staged.artifact_volume_id,
        volumes["control_ref"], volumes["artifact_ref"], staged.key_ref,
        hashlib.sha256(staged.claim).hexdigest(), hashlib.sha256(staged.bundle).hexdigest(),
        len(staged.bundle), submit.executor.executor_id, submit.executor.implementation_version,
    )
    dispatch = build_modal_worker_dispatch(expectation, envelope.claim, envelope.claim_tag)
    static = ModalWorkerStaticExpectation(
        canonical_bytes(selection.to_dict()), prep.provider.provider_id, prep.provider.profile_ref,
        submit.executor.executor_id, submit.executor.implementation_version,
        staged.control_volume_id, staged.artifact_volume_id,
        volumes["control_ref"], volumes["artifact_ref"], staged.key_ref,
        "/workspace/control", "/workspace/run", "/workspace/worker-control",
    )
    invocation = admit_modal_worker(
        dispatch.canonical_bytes, stage_claim=staged.claim,
        stage_claim_tag=staged.claim_tag, bundle_transport=bundle.transport_bytes,
        verifier=shared["authenticator"], recipes=shared["recipes"], static=static,
    )
    return invocation, shared["authenticator"]


def _inventory(invocation, *, mutate=None):
    records = []
    files = {}
    for index, role in enumerate(sorted(EXACT_ARTIFACT_ROLES, key=lambda value: value.value)):
        content = f"artifact-{index}".encode()
        name = f"artifact-{index}.bin"
        files[name] = content
        records.append({
            "role": role.value, "path": name, "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    document = {
        "schema_version": "synaptic-artifact-inventory/v1",
        "workload_fingerprint": invocation.submit_command.preparation.workload_digest,
        "artifacts": records,
    }
    if mutate is not None:
        mutate(document)
    return canonical_bytes(document), files


def _mounted_fakes(monkeypatch, inventory, files):
    writes = []
    copied = {}
    reads = []

    def read_regular(root, path, maximum):
        reads.append(str(path))
        if path.name == "runtime-v1-inventory.json":
            return inventory
        raise ValueError("unexpected mounted read")

    def copy_regular(source_root, source, destination_root, destination, *, maximum):
        content = files[source.name]
        copied[str(destination)] = content
        return len(content), hashlib.sha256(content).hexdigest()

    def write_exclusive(root, path, content):
        writes.append((str(path), bytes(content)))

    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.read_regular", read_regular,
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.copy_regular", copy_regular,
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.hash_regular",
        lambda root, path, maximum: (
            len(copied[str(path)]), hashlib.sha256(copied[str(path)]).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.claim_directory",
        lambda root, path: None,
    )
    def listing(root, path, maximum_entries):
        if path.name == "output":
            return tuple(sorted(
                (member.value, len(files[f"artifact-{index}.bin"]))
                for index, member in enumerate(sorted(EXACT_ARTIFACT_ROLES, key=lambda value: value.value))
            ))
        prefix = str(path) + "/"
        return tuple(sorted(
            (__import__("pathlib").Path(name).name, len(content))
            for name, content in writes
            if name.startswith(prefix) and "/" not in name[len(prefix):]
        ))
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.list_regular_sizes", listing,
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.write_exclusive", write_exclusive,
    )
    return writes, copied, reads


def test_success_uses_exact_submit_identity_and_writes_completion_last(monkeypatch):
    invocation, signer = _invocation(monkeypatch)
    inventory, files = _inventory(invocation)
    writes, copied, reads = _mounted_fakes(monkeypatch, inventory, files)
    result = MountedModalCoordinatorProducer(signer).finalize(
        invocation, ModalProcessResult(0), job_ref="job-a",
    )
    assert result.status_code == "completed" and len(copied) == 5
    assert len(reads) == 1 and reads[0].endswith("runtime-v1-inventory.json")
    assert writes[-1][0].endswith("completion-manifest.v1.mac")
    manifest = json.loads(writes[-2][1])
    submit = invocation.submit_command
    assert manifest["effect_id"] == submit.operation.effect.effect_id
    assert manifest["command_digest"] == submit.digest
    assert manifest["plan_digest"] == submit.preparation.plan_fingerprint
    assert manifest["deployment_attestation_digest"] == invocation.deployment.attestation_digest
    assert manifest["invocation_nonce"] == submit.operation.invocation_nonce
    assert "launch_claim_sha256" not in manifest and "bundle_sha256" not in manifest


def test_failed_process_reads_no_inventory_and_publishes_no_completion(monkeypatch):
    invocation, signer = _invocation(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.read_regular",
        lambda *args, **kwargs: calls.append("read") or b"",
    )
    writes = []
    sizes = {}
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.write_exclusive",
        lambda root, path, content: (
            writes.append(str(path)), sizes.__setitem__(str(path), len(content))
        ),
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.claim_directory",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.list_regular_sizes",
        lambda root, path, maximum_entries: tuple(sorted(
            (__import__("pathlib").Path(name).name, size)
            for name, size in sizes.items()
            if __import__("pathlib").Path(name).parent == path
        )),
    )
    result = MountedModalCoordinatorProducer(signer).finalize(
        invocation, ModalProcessResult(123, diagnostic_code="trainer_nonzero"), job_ref="job-a",
    )
    assert result.status_code == "failed" and calls == []
    assert not any("completion-manifest" in path for path in writes)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc["artifacts"].append(dict(doc["artifacts"][0])),
        lambda doc: doc["artifacts"].__setitem__(1, dict(doc["artifacts"][1], role=doc["artifacts"][0]["role"])),
        lambda doc: doc["artifacts"][0].__setitem__("size", True),
        lambda doc: doc["artifacts"][0].__setitem__("sha256", "not-a-digest"),
        lambda doc: doc["artifacts"][0].__setitem__("path", "nested/member"),
    ],
)
def test_entire_inventory_shape_is_rejected_before_copy(monkeypatch, mutation):
    invocation, signer = _invocation(monkeypatch)
    inventory, files = _inventory(invocation, mutate=mutation)
    writes, copied, _ = _mounted_fakes(monkeypatch, inventory, files)
    with pytest.raises((TypeError, ValueError)):
        MountedModalCoordinatorProducer(signer).finalize(
            invocation, ModalProcessResult(0), job_ref="job-a",
        )
    assert copied == {} and writes == []


def test_mutated_invocation_does_not_touch_mounted_state(monkeypatch):
    invocation, signer = _invocation(monkeypatch)
    inventory, files = _inventory(invocation)
    object.__setattr__(invocation, "workload", b"{}")
    writes, _, _ = _mounted_fakes(monkeypatch, inventory, files)
    with pytest.raises(ValueError):
        MountedModalCoordinatorProducer(signer).finalize(
            invocation, ModalProcessResult(0), job_ref="job-a",
        )
    assert writes == []


def test_copy_content_mismatch_emits_no_control_success(monkeypatch):
    invocation, signer = _invocation(monkeypatch)
    inventory, files = _inventory(invocation)
    writes, _, _ = _mounted_fakes(monkeypatch, inventory, files)
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.copy_regular",
        lambda *args, **kwargs: (1, "0" * 64),
    )
    with pytest.raises(ValueError, match="content mismatch"):
        MountedModalCoordinatorProducer(signer).finalize(
            invocation, ModalProcessResult(0), job_ref="job-a",
        )
    assert writes == []


def test_signer_failure_writes_no_control_success(monkeypatch):
    invocation, _ = _invocation(monkeypatch)
    inventory, files = _inventory(invocation)
    writes, copied, _ = _mounted_fakes(monkeypatch, inventory, files)

    class Broken:
        def sign(self, *args):
            raise RuntimeError("secret detail")

    with pytest.raises(ValueError, match="authentication unavailable"):
        MountedModalCoordinatorProducer(Broken()).finalize(
            invocation, ModalProcessResult(0), job_ref="job-a",
        )
    assert writes == [] and copied == {}


def test_preexisting_or_extra_output_directory_cannot_complete(monkeypatch):
    invocation, signer = _invocation(monkeypatch)
    inventory, files = _inventory(invocation)
    writes, copied, _ = _mounted_fakes(monkeypatch, inventory, files)
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.claim_directory",
        lambda *args: (_ for _ in ()).throw(FileExistsError("collision")),
    )
    with pytest.raises(FileExistsError):
        MountedModalCoordinatorProducer(signer).finalize(
            invocation, ModalProcessResult(0), job_ref="job-a",
        )
    assert copied == {} and writes == []


def test_invalid_remote_returncode_stops_before_mounted_io(monkeypatch):
    invocation, signer = _invocation(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.read_regular",
        lambda *args, **kwargs: calls.append("read") or b"",
    )
    with pytest.raises(ValueError, match="closed remote protocol"):
        MountedModalCoordinatorProducer(signer).finalize(
            invocation, ModalProcessResult(1), job_ref="job-a",
        )
    assert calls == []


def test_real_mounted_io_publishes_exact_authenticated_evidence(tmp_path, monkeypatch):
    import tests.execution.providers.test_modal_coordinator_bundle as bundle_tests

    original_source = bundle_tests._execution_source

    def temporary_source():
        source = original_source()
        writable = tmp_path / "run"
        roots = {
            name: str(
                (tmp_path / "sources" / name)
                if name in {"engine", "project"}
                else (writable / name)
            )
            for name in source.roots
        }
        replacements = sorted(source.roots.items(), key=lambda item: len(item[1]), reverse=True)
        environment = {}
        for name, value in source.environment.items():
            for root_name, old in replacements:
                value = value.replace(old, roots[root_name])
            environment[name] = value
        return replace(
            source, roots=roots, writable_capability_root=str(writable),
            environment=environment,
        )

    monkeypatch.setattr(bundle_tests, "_execution_source", temporary_source)
    invocation, signer = _invocation(monkeypatch)
    inventory, files = _inventory(invocation)
    source = invocation.source
    state_root = Path(source.roots["state"])
    artifact_source = Path(source.roots["artifacts"])
    state_root.mkdir(parents=True)
    artifact_source.mkdir(parents=True)
    (state_root / "runtime-v1-inventory.json").write_bytes(inventory)
    for name, content in files.items():
        (artifact_source / name).write_bytes(content)
    control_root, artifact_root = tmp_path / "control", tmp_path / "run"
    result = MountedModalCoordinatorProducer(
        signer, control_root=str(control_root), artifact_root=str(artifact_root),
        clock=lambda: "2026-09-09T12:00:00+00:00",
    ).finalize(invocation, ModalProcessResult(0), job_ref="job-real")
    effect_id = invocation.submit_command.operation.effect.effect_id
    evidence = control_root / operation_path(effect_id, "evidence")
    logs = control_root / operation_path(effect_id, "logs")
    terminal_bytes = (evidence / "terminal-evidence.v1.json").read_bytes()
    terminal_tag = (evidence / "terminal-evidence.v1.mac").read_bytes()
    manifest_bytes = (evidence / "completion-manifest.v1.json").read_bytes()
    manifest_tag = (evidence / "completion-manifest.v1.mac").read_bytes()
    metadata_bytes = (logs / "log-metadata.v1.json").read_bytes()
    metadata_tag = (logs / "log-metadata.v1.mac").read_bytes()
    terminal = TerminalEvidenceV1.parse(terminal_bytes)
    manifest = CompletionManifestV1.parse(manifest_bytes)
    metadata = _object(metadata_bytes, 65536)
    chunk_path = logs / "chunks" / "000.json"
    chunk = ModalCoordinatorLogChunk.parse(chunk_path.read_bytes())
    assert signer.verify("modal-terminal/v1", terminal_bytes, terminal_tag, invocation.key_ref)
    assert signer.verify("modal-completion/v1", manifest_bytes, manifest_tag, invocation.key_ref)
    assert signer.verify("modal-log-metadata/v1", metadata_bytes, metadata_tag, invocation.key_ref)
    assert terminal.status_code == result.status_code == "completed"
    assert terminal.artifact_set_digest == manifest.artifact_set_digest
    assert metadata["chain_digest"] == chunk.chunk_digest == result.log_chain_digest
    assert chunk.records[0].timestamp == "2026-09-09T12:00:00+00:00"
    assert metadata["chunks"][0]["sha256"] == hashlib.sha256(
        chunk.canonical_bytes
    ).hexdigest()
    changed = json.loads(chunk.canonical_bytes)
    changed["records"][0]["timestamp"] = "2026-09-09T12:00:01+00:00"
    changed["payload_digest"] = hashlib.sha256(
        canonical_bytes(changed["records"])
    ).hexdigest()
    assert metadata["chunks"][0]["sha256"] != hashlib.sha256(
        canonical_bytes(changed)
    ).hexdigest()
    output = artifact_root / operation_path(effect_id, "output")
    assert list_regular_sizes(artifact_root, output, maximum_entries=6) == tuple(sorted(
        (role.value, len(files[f"artifact-{index}.bin"]))
        for index, role in enumerate(sorted(EXACT_ARTIFACT_ROLES, key=lambda value: value.value))
    ))


@pytest.mark.parametrize("field", ["max_chunk_bytes", "max_terminal_bytes"])
def test_admitted_evidence_policy_is_enforced_before_artifact_copy(monkeypatch, field):
    invocation, signer = _invocation(monkeypatch)
    policy = invocation.log_policy
    policy[field] = 1
    object.__setattr__(invocation, "log_policy_bytes", canonical_bytes(policy))
    inventory, files = _inventory(invocation)
    writes, copied, _ = _mounted_fakes(monkeypatch, inventory, files)
    with pytest.raises(ValueError, match="exceeds admitted policy"):
        MountedModalCoordinatorProducer(signer).finalize(
            invocation, ModalProcessResult(0), job_ref="job-a",
        )
    assert copied == {} and writes == []


def test_invalid_clock_timestamp_fails_before_mounted_io(monkeypatch):
    invocation, signer = _invocation(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.read_regular",
        lambda *args, **kwargs: calls.append("read") or b"",
    )
    with pytest.raises(ValueError):
        MountedModalCoordinatorProducer(
            signer, clock=lambda: "not-rfc3339",
        ).finalize(invocation, ModalProcessResult(0), job_ref="job-a")
    assert calls == []


def test_extra_log_chunk_stops_before_completion_mac(monkeypatch):
    invocation, signer = _invocation(monkeypatch)
    inventory, files = _inventory(invocation)
    writes, _, _ = _mounted_fakes(monkeypatch, inventory, files)
    original = __import__(
        "tuner.execution.providers.modal.coordinator_producer", fromlist=["list_regular_sizes"],
    ).list_regular_sizes

    def extra(root, path, maximum_entries):
        value = original(root, path, maximum_entries)
        if path.name == "chunks":
            return value + (("001.json", 1),)
        return value

    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.list_regular_sizes", extra,
    )
    with pytest.raises(ValueError, match="log chunk set"):
        MountedModalCoordinatorProducer(
            signer, clock=lambda: "2026-09-09T12:00:00+00:00",
        ).finalize(invocation, ModalProcessResult(0), job_ref="job-a")
    assert not any(path.endswith("completion-manifest.v1.mac") for path, _ in writes)


def test_preexisting_control_prefix_denies_before_artifact_copy(monkeypatch):
    invocation, signer = _invocation(monkeypatch)
    inventory, files = _inventory(invocation)
    writes, copied, _ = _mounted_fakes(monkeypatch, inventory, files)
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_producer.claim_directory",
        lambda *args: (_ for _ in ()).throw(FileExistsError("collision")),
    )
    with pytest.raises(FileExistsError):
        MountedModalCoordinatorProducer(signer).finalize(
            invocation, ModalProcessResult(0), job_ref="job-a",
        )
    assert copied == {} and writes == []
