"""Provider-free boundary tests for the Foundation-native Modal read transport."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from synaptic_tuner.api.v1.runs_facade import RunLogEntry, RunLogLevel

from tests.execution.providers.modal_coordinator_fixtures import real_launch_bundle_case
from tests.execution.providers.test_modal_coordinator_producer import (
    _inventory, _invocation, _mounted_fakes,
)
from tuner.execution.coordinator_v1.model import ProviderLogQueryV1, ProviderRunPhaseV1
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.providers.modal.contracts import BoundsPolicyV1
from tuner.execution.providers.modal.contracts import canonical_json, sha
from tuner.execution.providers.modal.coordinator_logs import ModalCoordinatorLogChunk
from tuner.execution.providers.modal.coordinator_read_transport import ModalFoundationReadTransport
from tuner.execution.providers.modal.coordinator_producer import MountedModalCoordinatorProducer
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade, ModalFunctionCallState
from tuner.execution.providers.modal.worker_ports import ModalProcessResult


ROOT = Path(__file__).resolve().parents[3]


def test_read_transport_import_does_not_load_legacy_read_or_training_modules():
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import tuner.execution.providers.modal.coordinator_read_transport
for name in ('tuner.execution.providers.modal.run_reads',
             'tuner.execution.providers.modal.training'):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode()


def test_read_transport_requires_exact_provider_configuration():
    with pytest.raises(TypeError, match="exact explicit Modal facade"):
        ModalFoundationReadTransport(
            facade=object(), deployment=object(), launch_source=object(),
            binding_authority=object(), foundation_authenticator=object(),
            assessment_authenticator=object(), stage_verifier=object(),
            launch_verifier=object(), evidence_verifier=object(),
            recipes=object(), bounds=BoundsPolicyV1(),
        )


def test_read_transport_has_only_the_existing_four_reader_operations():
    operations = {
        name for name in ("observe", "logs", "artifact_inventory", "iter_artifact")
        if callable(getattr(ModalFoundationReadTransport, name, None))
    }
    assert operations == {"observe", "logs", "artifact_inventory", "iter_artifact"}


class _SDK:
    __version__ = "1.5.4"


class _Source:
    def __init__(self, value): self.value = value
    def resolve(self, digest): return self.value


def _completed_transport(monkeypatch):
    case = real_launch_bundle_case(monkeypatch)
    invocation, signer = _invocation(monkeypatch)
    assert invocation.submit_command.canonical_bytes == case["envelope"].submit_binding.command_bytes
    inventory, files = _inventory(invocation)
    writes, copied, _ = _mounted_fakes(monkeypatch, inventory, files)
    MountedModalCoordinatorProducer(
        signer, clock=lambda: "2026-09-09T00:00:00Z",
    ).finalize(invocation, ModalProcessResult(0), job_ref="job-a")
    stored = {
        path.removeprefix("/workspace/control/").removeprefix("/workspace/run/"): content
        for path, content in writes
    }
    stored.update({
        path.removeprefix("/workspace/run/"): content for path, content in copied.items()
    })
    binding = case["envelope"].submit_binding
    selection = binding.deployment.selection
    profile = __import__("json").loads(binding.preparation_snapshot)["configuration"]["profile"]
    facade = ExplicitModal154ReadFacade(
        binding.client_binding, sdk=_SDK, client=object(),
        scope_observer=lambda client: (
            selection.account_ref, selection.workspace_ref,
            selection.environment_ref, selection.client_ref,
        ), deployment_observer=lambda **kwargs: selection,
        volume_names={invocation.control_volume_id: profile["volumes"]["control_ref"],
                      invocation.artifact_volume_id: profile["volumes"]["artifact_ref"]},
    )
    body_reads = []
    def read_complete(self, volume, path, *, max_bytes):
        value = stored[path]
        if "/output/" in path: body_reads.append(path)
        if len(value) > max_bytes: raise ValueError("bound")
        return value
    def iter_complete(self, volume, path, *, max_bytes):
        value = stored[path]
        if len(value) > max_bytes: raise ValueError("bound")
        yield value[:2]
        if value[2:]: yield value[2:]
    def list_prefix(self, volume, prefix, *, max_entries):
        prefix = prefix if prefix.endswith("/") else prefix + "/"
        rows = tuple(sorted(
            (path, len(value), __import__("hashlib").sha256(
                b"synaptic.modal-volume-entry/v1\0" + volume.encode() + b"\0" +
                path.encode() + b"\0" + str(len(value)).encode()).hexdigest())
            for path, value in stored.items() if path.startswith(prefix)
        ))
        if len(rows) >= max_entries: raise ValueError("overflow")
        return rows
    monkeypatch.setattr(ExplicitModal154ReadFacade, "read_complete", read_complete)
    monkeypatch.setattr(ExplicitModal154ReadFacade, "iter_complete", iter_complete)
    monkeypatch.setattr(ExplicitModal154ReadFacade, "list_prefix", list_prefix)
    monkeypatch.setattr(ExplicitModal154ReadFacade, "bound_scope", lambda self: (
        selection.account_ref, selection.workspace_ref,
        selection.environment_ref, selection.client_ref,
    ))
    monkeypatch.setattr(ExplicitModal154ReadFacade, "inspect_deployment", lambda self, **kwargs: selection)
    monkeypatch.setattr(ExplicitModal154ReadFacade, "observe_known_call_pending",
                        lambda self, value: ModalFunctionCallState.UNKNOWN)
    transport = ModalFoundationReadTransport(
        facade=facade, deployment=binding.deployment,
        launch_source=_Source(case["envelope"]), binding_authority=case["authority"],
        foundation_authenticator=case["harness"].authenticator,
        assessment_authenticator=case["harness"].foundation,
        stage_verifier=case["authenticator"], launch_verifier=case["authenticator"],
        evidence_verifier=signer, recipes=case["recipes"],
    )
    return transport, binding, stored, body_reads, signer, case["material"].key_ref


def test_real_signed_evidence_observe_logs_inventory_then_stream(monkeypatch):
    transport, binding, stored, body_reads, _, _ = _completed_transport(monkeypatch)
    observed = transport.observe(binding, provider_job_ref="job-a")
    assert observed.phase is ProviderRunPhaseV1.SUCCEEDED
    page = transport.logs(
        binding, ProviderLogQueryV1(None, 10, 4096), provider_job_ref="job-a",
    )
    assert len(page.entries) == 1 and page.entries[0].timestamp == "2026-09-09T00:00:00Z"
    inventory = transport.artifact_inventory(binding, provider_job_ref="job-a")
    assert len(inventory.manifest.members) == 5 and body_reads == []
    member = inventory.manifest.members[0]
    assert b"".join(transport.iter_artifact(
        binding, member, provider_job_ref="job-a", maximum_bytes=member.size,
    )) == stored[member.path]


@pytest.mark.parametrize("fault", [
    "terminal_mac", "terminal_identity", "log_mac", "log_generation_bool",
    "extra_output", "truncated", "drift",
])
def test_adversarial_evidence_and_artifact_streams_fail_closed(monkeypatch, fault):
    transport, binding, stored, _, signer, key_ref = _completed_transport(monkeypatch)
    command = parse_exact_command(binding.command_bytes)
    effect = command.operation.effect.effect_id
    if fault == "terminal_mac":
        stored[f"operations/{effect}/evidence/terminal-evidence.v1.mac"] = b"wrong"
        with pytest.raises(ValueError, match="authentication"):
            transport.observe(binding, provider_job_ref="job-a")
        return
    if fault == "terminal_identity":
        path = f"operations/{effect}/evidence/terminal-evidence.v1.json"
        document = __import__("json").loads(stored[path])
        document["job_ref"] = "job-other"
        stored[path] = __import__(
            "tuner.execution.foundation_v2.canonical", fromlist=["canonical_bytes"],
        ).canonical_bytes(document)
        stored[path.removesuffix(".json") + ".mac"] = signer.sign(
            "modal-terminal/v1", stored[path], key_ref,
        )
        with pytest.raises(ValueError):
            transport.observe(binding, provider_job_ref="job-a")
        return
    if fault == "log_mac":
        stored[f"operations/{effect}/logs/log-metadata.v1.mac"] = b"wrong"
        with pytest.raises(ValueError, match="authentication"):
            transport.logs(binding, ProviderLogQueryV1(None, 10, 4096), provider_job_ref="job-a")
        return
    if fault == "log_generation_bool":
        path = f"operations/{effect}/logs/log-metadata.v1.json"
        document = __import__("json").loads(stored[path]); document["generation"] = True
        stored[path] = canonical_json(document)
        stored[path.removesuffix(".json") + ".mac"] = signer.sign(
            "modal-log-metadata/v1", stored[path], key_ref,
        )
        with pytest.raises(ValueError, match="generation"):
            transport.logs(binding, ProviderLogQueryV1(None, 10, 4096), provider_job_ref="job-a")
        return
    if fault == "extra_output":
        stored[f"operations/{effect}/output/extra"] = b"x"
        with pytest.raises(ValueError):
            transport.artifact_inventory(binding, provider_job_ref="job-a")
        return
    inventory = transport.artifact_inventory(binding, provider_job_ref="job-a")
    member = inventory.manifest.members[0]
    original = ExplicitModal154ReadFacade.iter_complete
    if fault == "truncated":
        monkeypatch.setattr(ExplicitModal154ReadFacade, "iter_complete",
                            lambda self, volume, path, *, max_bytes: iter((stored[path][:-1],)))
    else:
        calls = {"n": 0}
        original_list = ExplicitModal154ReadFacade.list_prefix
        def changed(self, volume, prefix, *, max_entries):
            rows = original_list(self, volume, prefix, max_entries=max_entries)
            calls["n"] += 1
            if calls["n"] >= 3 and "/output" in prefix:
                return rows[:-1]
            return rows
        monkeypatch.setattr(ExplicitModal154ReadFacade, "list_prefix", changed)
    with pytest.raises(ValueError):
        b"".join(transport.iter_artifact(
            binding, member, provider_job_ref="job-a", maximum_bytes=member.size,
        ))


def test_log_page_budget_counts_exact_utf8_messages_not_json_overhead(monkeypatch):
    transport, binding, stored, _, signer, key_ref = _completed_transport(monkeypatch)
    effect = parse_exact_command(binding.command_bytes).operation.effect.effect_id
    chunk_path = f"operations/{effect}/logs/chunks/000.json"
    original = ModalCoordinatorLogChunk.parse(stored[chunk_path])
    records = tuple(RunLogEntry(
        index, "2026-09-09T00:00:00Z", RunLogLevel.INFO, "progress",
        "é" * 40, 80,
    ) for index in range(40))
    chunk = ModalCoordinatorLogChunk(
        original.generation, original.sequence, original.previous_digest,
        sha(canonical_json([entry.to_dict() for entry in records])),
        original.job_ref, original.effect_id, original.plan_digest,
        original.invocation_nonce, records,
    )
    stored[chunk_path] = chunk.canonical_bytes
    meta_path = f"operations/{effect}/logs/log-metadata.v1.json"
    metadata = __import__("json").loads(stored[meta_path])
    metadata["chain_digest"] = chunk.chunk_digest
    metadata["chunks"][0].update(size=len(chunk.canonical_bytes), sha256=sha(chunk.canonical_bytes))
    metadata["chunks"][0]["provider_entry_id"] = __import__(
        "tuner.execution.providers.modal.contracts", fromlist=["provider_entry_identity"],
    ).provider_entry_identity(metadata["control_volume_id"], chunk_path, len(chunk.canonical_bytes))
    stored[meta_path] = canonical_json(metadata)
    stored[meta_path.removesuffix(".json") + ".mac"] = signer.sign(
        "modal-log-metadata/v1", stored[meta_path], key_ref,
    )
    for name in (
        "terminal-evidence.v1.json", "terminal-evidence.v1.mac",
        "completion-manifest.v1.json", "completion-manifest.v1.mac",
    ):
        del stored[f"operations/{effect}/evidence/{name}"]
    page = transport.logs(
        binding, ProviderLogQueryV1(None, 200, 4096), provider_job_ref="job-a",
    )
    assert len(page.entries) == 40 and sum(x.size_bytes for x in page.entries) == 3200


def test_absent_terminal_never_fabricates_queued_or_running(monkeypatch):
    transport, binding, stored, _, _, _ = _completed_transport(monkeypatch)
    effect = parse_exact_command(binding.command_bytes).operation.effect.effect_id
    del stored[f"operations/{effect}/evidence/terminal-evidence.v1.json"]
    del stored[f"operations/{effect}/evidence/terminal-evidence.v1.mac"]
    del stored[f"operations/{effect}/evidence/completion-manifest.v1.json"]
    del stored[f"operations/{effect}/evidence/completion-manifest.v1.mac"]
    with pytest.raises(ValueError, match="phase is unavailable"):
        transport.observe(binding, provider_job_ref="job-a")


def test_unpatched_explicit_facade_reads_exact_fake_sdk_volumes(monkeypatch):
    from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume, SDK

    with monkeypatch.context() as patched:
        prior, binding, stored, _, _, _ = _completed_transport(patched)
    material = prior._launch_source.value.stage_material
    profile = __import__("json").loads(binding.preparation_snapshot)["configuration"]["profile"]
    control = FakeVolume(material.control_volume_id)
    artifact = FakeVolume(material.artifact_volume_id)
    for path, content in stored.items():
        (artifact if "/output/" in path else control).files[path] = content
    FakeVolume.registry = {
        profile["volumes"]["control_ref"]: control,
        profile["volumes"]["artifact_ref"]: artifact,
    }
    FakeVolume.calls = []
    body_reads = []
    original_read = FakeVolume.read_file
    def counted(self, path):
        if "/output/" in path: body_reads.append(path)
        yield from original_read(self, path)
    monkeypatch.setattr(FakeVolume, "read_file", counted)
    selection = binding.deployment.selection
    client = object()
    facade = ExplicitModal154ReadFacade(
        binding.client_binding, sdk=SDK, client=client,
        scope_observer=lambda supplied: (
            selection.account_ref, selection.workspace_ref,
            selection.environment_ref, selection.client_ref,
        ) if supplied is client else (),
        deployment_observer=lambda **kwargs: selection,
        volume_names={
            material.control_volume_id: profile["volumes"]["control_ref"],
            material.artifact_volume_id: profile["volumes"]["artifact_ref"],
        },
    )
    transport = ModalFoundationReadTransport(
        facade=facade, deployment=prior._deployment,
        launch_source=prior._launch_source, binding_authority=prior._authority,
        foundation_authenticator=prior._foundation,
        assessment_authenticator=prior._assessments, stage_verifier=prior._stage,
        launch_verifier=prior._launch, evidence_verifier=prior._evidence,
        recipes=prior._recipes,
    )
    assert transport.observe(binding, provider_job_ref="job-a").phase is ProviderRunPhaseV1.SUCCEEDED
    assert len(transport.logs(
        binding, ProviderLogQueryV1(None, 10, 4096), provider_job_ref="job-a",
    ).entries) == 1
    inventory = transport.artifact_inventory(binding, provider_job_ref="job-a")
    assert body_reads == []
    member = inventory.manifest.members[0]
    assert b"".join(transport.iter_artifact(
        binding, member, provider_job_ref="job-a", maximum_bytes=member.size,
    )) == stored[member.path]
    assert body_reads == [member.path]
    assert all(call[1]["client"] is client for call in FakeVolume.calls)


def test_invalid_stream_arguments_fail_before_inventory_io(monkeypatch):
    transport, binding, _, _, _, _ = _completed_transport(monkeypatch)
    monkeypatch.setattr(
        ModalFoundationReadTransport, "artifact_inventory",
        lambda *args, **kwargs: pytest.fail("inventory I/O reached"),
    )
    with pytest.raises(ValueError, match="stream request invalid"):
        b"".join(transport.iter_artifact(
            binding, object(), provider_job_ref="job-a", maximum_bytes=True,
        ))
