from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO
from dataclasses import replace
from pathlib import Path

import pytest

from tuner.execution.providers.modal.coordinator_bundle import (
    BUNDLE_SCHEMA,
    MOUNTED_PREPARED_BUNDLE_SCHEMA,
    MOUNTED_PREPARED_MEMBER_NAMES,
    MEMBER_NAMES,
    PREPARED_BUNDLE_SCHEMA,
    PREPARED_MEMBER_NAMES,
    ModalCoordinatorBundle,
)
from tuner.execution.providers.modal.coordinator_retention import (
    ModalRetainedPreparation,
)
from tuner.execution.providers.modal.prepared_input import (
    MAX_MOUNTED_PREPARED_DATASET_BYTES,
    MAX_PRIVATE_DATASET_BYTES,
    MountedPreparedInputDescriptor,
    bind_private_dataset,
    materialize_private_dataset,
    prepared_dataset_path,
)
from tuner.training.contracts import (
    PreparedTrainingInputIdentity,
    RetainedTrainingInputStreamLease,
)
from tuner.execution.providers.modal.worker_ports import ModalProcessResult
from tuner.runtime.verification import _expected_dataset_path
from tests.execution.providers.test_modal_coordinator_bundle import _fixture
from tests.training.test_sft_compilation import _config


def _payload() -> bytes:
    return (
        b'{"format":"raw_text","schema_version":"syntunia-sft-row/v1",'
        b'"split":"train","text":"PRIVATE_REPR_SENTINEL"}\n'
    )


def _prepared_config(payload: bytes) -> dict[str, object]:
    document = _config().to_dict()
    prepared_digest = "a" * 64
    document["dataset"] = {
        "ref": f"prepared://sha256/{prepared_digest}",
        "revision": prepared_digest,
        "content_digest": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "format": "syntunia-sft-row/v1",
    }
    document["sft"] = dict(document["sft"]) | {
        "dataset_format": "raw_text",
        "completion_only_loss": False,
        "assistant_only_loss": False,
        "use_preassigned_splits": True,
        "split_dataset": False,
    }
    return document


def _prepared_fixture(payload: bytes):
    prepared = _prepared_config(payload)
    return _fixture(
        config_extra={
            "dataset": prepared["dataset"],
            "sft": prepared["sft"],
        }
    )


def test_project_bundle_remains_v2_and_has_no_private_member():
    binding, material, recipes, policy, closure = _fixture()
    bundle = ModalCoordinatorBundle.build(
        binding,
        material,
        recipes,
        log_terminal_policy=policy,
        worker_closure_manifest=closure,
    )
    assert json.loads(bundle.canonical_bytes)["schema_version"] == BUNDLE_SCHEMA
    assert tuple(member.name for member in bundle.members) == MEMBER_NAMES


def test_prepared_bundle_v3_binds_exact_opaque_dataset_bytes():
    payload = _payload()
    binding, material, recipes, policy, closure = _prepared_fixture(payload)
    bundle = ModalCoordinatorBundle.build(
        binding,
        material,
        recipes,
        log_terminal_policy=policy,
        worker_closure_manifest=closure,
        private_dataset_bytes=payload,
    )
    document = json.loads(bundle.canonical_bytes)
    assert document["schema_version"] == PREPARED_BUNDLE_SCHEMA
    assert tuple(member.name for member in bundle.members) == PREPARED_MEMBER_NAMES
    private = next(
        member for member in bundle.members if member.name == "private-dataset.jsonl"
    )
    assert private.content == payload
    assert "PRIVATE_REPR_SENTINEL" not in repr(private)
    assert "PRIVATE_REPR_SENTINEL" not in repr(bundle)
    plan = json.loads(
        next(member.content for member in bundle.members if member.name == "stage-plan.json")
    )
    assert plan["members"][private.name] == {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    assert plan["stage_command_digest"] == binding.command_digest
    assert plan["workload_digest"] == material.planning_request.workload_digest
    assert ModalCoordinatorBundle.parse_transport(
        bundle.transport_bytes, binding=binding, recipes=recipes,
    ) == bundle


def test_large_prepared_bundle_uses_descriptor_only_v4_without_payload():
    payload = b"PRIVATE_LARGE_SENTINEL" + b"x" * MAX_PRIVATE_DATASET_BYTES
    binding, material, recipes, policy, closure = _prepared_fixture(payload)
    identity = PreparedTrainingInputIdentity.from_mapping(
        _prepared_config(payload)["dataset"]
    )
    effect_id = __import__(
        "tuner.execution.foundation_v2.commands", fromlist=["parse_exact_command"]
    ).parse_exact_command(binding.command_bytes).operation.effect.effect_id
    descriptor = MountedPreparedInputDescriptor.create(
        identity, stage_effect_id=effect_id,
    )
    bundle = ModalCoordinatorBundle.build(
        binding,
        material,
        recipes,
        log_terminal_policy=policy,
        worker_closure_manifest=closure,
        mounted_prepared_input=descriptor,
    )
    assert json.loads(bundle.canonical_bytes)["schema_version"] == MOUNTED_PREPARED_BUNDLE_SCHEMA
    assert tuple(member.name for member in bundle.members) == MOUNTED_PREPARED_MEMBER_NAMES
    assert payload not in bundle.canonical_bytes
    assert b"PRIVATE_LARGE_SENTINEL" not in bundle.transport_bytes
    member = next(
        item for item in bundle.members if item.name == "prepared-input.json"
    )
    assert json.loads(member.content)["relative_path"].endswith(
        f"/{identity.content_digest}/payload.bin"
    )
    assert ModalCoordinatorBundle.parse_transport(
        bundle.transport_bytes, binding=binding, recipes=recipes,
    ) == bundle


def test_mounted_prepared_input_bounds_are_strict():
    payload = b"x" * (MAX_PRIVATE_DATASET_BYTES + 1)
    identity = PreparedTrainingInputIdentity.from_mapping(
        _prepared_config(payload)["dataset"]
    )
    assert identity.size_bytes == len(payload)
    with pytest.raises(ValueError, match="size"):
        PreparedTrainingInputIdentity(
            identity.ref,
            identity.revision,
            identity.content_digest,
            MAX_MOUNTED_PREPARED_DATASET_BYTES + 1,
            identity.format,
        )


def test_reconciled_production_identity_selects_the_mounted_range():
    identity = PreparedTrainingInputIdentity(
        ref=f"prepared://sha256/{'a' * 64}",
        revision="a" * 64,
        content_digest=(
            "e59327bfd5e7b6baec3b8725879d4488fa42c759b32d350180ecd1e02854c030"
        ),
        size_bytes=10_838_758,
        format="syntunia-sft-row/v2",
    )
    descriptor = MountedPreparedInputDescriptor.create(
        identity, stage_effect_id="effect-production",
    )
    assert identity.size_bytes > MAX_PRIVATE_DATASET_BYTES
    assert identity.size_bytes <= MAX_MOUNTED_PREPARED_DATASET_BYTES
    assert descriptor.relative_path.endswith(
        f"/{identity.content_digest}/payload.bin"
    )


def test_private_dataset_binding_preserves_authoritative_message_format():
    payload = _payload()
    dataset = dict(_prepared_config(payload)["dataset"])
    dataset["format"] = "syntunia-sft-row/v2"
    binding = bind_private_dataset(dataset, payload)
    assert binding.dataset_format == "syntunia-sft-row/v2"


def test_prepared_bundle_rejects_absent_private_bytes():
    payload = _payload()
    config = _prepared_config(payload)
    binding, material, recipes, policy, closure = _fixture(
        config_extra={"dataset": config["dataset"], "sft": config["sft"]},
    )
    with pytest.raises(ValueError, match="prepared dataset"):
        ModalCoordinatorBundle.build(
            binding,
            material,
            recipes,
            log_terminal_policy=policy,
            worker_closure_manifest=closure,
            private_dataset_bytes=None,
        )


@pytest.mark.parametrize("mutation", ["size", "digest", "revision", "format"])
def test_private_bytes_reject_mismatched_prepared_identity(mutation):
    payload = _payload()
    dataset = dict(_prepared_config(payload)["dataset"])
    if mutation == "size":
        dataset["size_bytes"] += 1
    elif mutation == "digest":
        dataset["content_digest"] = "f" * 64
    elif mutation == "revision":
        dataset["revision"] = "b" * 64
    elif mutation == "format":
        dataset["format"] = "other/v1"
    with pytest.raises(ValueError, match="prepared dataset"):
        bind_private_dataset(dataset, payload)


def test_private_member_has_two_mib_cap_but_ordinary_members_keep_one_mib():
    payload = b"x" * (1024 * 1024 + 1)
    binding = bind_private_dataset(_prepared_config(payload)["dataset"], payload)
    assert binding is not None and binding.size_bytes == len(payload)
    with pytest.raises(ValueError, match="byte bound"):
        bind_private_dataset(
            _prepared_config(b"x" * (MAX_PRIVATE_DATASET_BYTES + 1))["dataset"],
            b"x" * (MAX_PRIVATE_DATASET_BYTES + 1),
        )


def test_retained_preparation_keeps_private_bytes_internal_and_exact():
    payload = _payload()
    binding, material, recipes, policy, closure = _prepared_fixture(payload)
    retained = ModalRetainedPreparation(
        binding.preparation_snapshot,
        binding.deployment_bytes,
        material.canonical_bytes,
        recipes,
        policy,
        closure,
        "control-id",
        "artifact-id",
        "stage-key",
        private_dataset_bytes=payload,
    )
    assert retained.private_dataset_bytes == payload
    assert "private_dataset_bytes" not in repr(retained)
    assert "PRIVATE_REPR_SENTINEL" not in repr(retained)
    with pytest.raises(ValueError, match="identity|content digest"):
        replace(retained, private_dataset_bytes=payload + b"x")


def test_retained_preparation_keeps_large_source_out_of_value_equality():
    payload = b"x" * (MAX_PRIVATE_DATASET_BYTES + 1)
    binding, material, recipes, policy, closure = _prepared_fixture(payload)
    identity = PreparedTrainingInputIdentity.from_mapping(
        _prepared_config(payload)["dataset"]
    )

    class Source:
        def __init__(self, marker):
            self.identity = identity
            self.marker = marker

        def open_lease(self):
            return RetainedTrainingInputStreamLease(identity, BytesIO(payload))

    retained = ModalRetainedPreparation(
        binding.preparation_snapshot,
        binding.deployment_bytes,
        material.canonical_bytes,
        recipes,
        policy,
        closure,
        "control-id",
        "artifact-id",
        "stage-key",
        prepared_input_source=Source("first"),
    )
    rebuilt = replace(retained, prepared_input_source=Source("reopened"))
    assert rebuilt == retained
    assert "prepared_input_source" not in repr(retained)
    with pytest.raises(ValueError, match="source does not match"):
        replace(
            retained,
            prepared_input_source=type("BadSource", (), {
                "identity": PreparedTrainingInputIdentity(
                    identity.ref,
                    identity.revision,
                    "f" * 64,
                    identity.size_bytes,
                    identity.format,
                ),
                "open_lease": lambda self: None,
            })(),
        )


def test_materialization_is_deterministic_exclusive_and_reverified(tmp_path: Path):
    payload = _payload()
    binding = bind_private_dataset(_prepared_config(payload)["dataset"], payload)
    assert binding is not None
    state = (tmp_path / "state").resolve()
    state.mkdir()
    expected = prepared_dataset_path(state, binding)
    assert materialize_private_dataset(state, binding, payload) == expected
    assert expected.read_bytes() == payload
    with pytest.raises((FileExistsError, ValueError)):
        materialize_private_dataset(state, binding, payload)


def test_materialization_rejects_a_short_write(tmp_path: Path, monkeypatch):
    import tuner.execution.providers.modal.prepared_input as prepared_input

    payload = _payload()
    binding = bind_private_dataset(_prepared_config(payload)["dataset"], payload)
    assert binding is not None
    state = (tmp_path / "state").resolve()
    state.mkdir()

    def short_write(root, destination, content):
        destination.write_bytes(content[:-1])

    monkeypatch.setattr(prepared_input, "write_exclusive", short_write)
    with pytest.raises(ValueError, match="does not match"):
        materialize_private_dataset(state, binding, payload)


def test_materialization_rejects_redirected_parent(tmp_path: Path):
    payload = _payload()
    binding = bind_private_dataset(_prepared_config(payload)["dataset"], payload)
    assert binding is not None
    state = (tmp_path / "state").resolve()
    outside = (tmp_path / "outside").resolve()
    state.mkdir()
    outside.mkdir()
    try:
        (state / "prepared-inputs").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory redirects are unavailable")
    with pytest.raises((FileExistsError, ValueError)):
        materialize_private_dataset(state, binding, payload)
    assert tuple(outside.iterdir()) == ()


def test_verifier_accepts_only_canonical_positive_retained_descriptor_path():
    payload = _payload()
    dataset = _prepared_config(payload)["dataset"]
    roots = {"project": "/workspace/project", "state": "/workspace/run/run-1/state"}
    assert _expected_dataset_path(dataset, roots, "/proc/self/fd/7") == "/proc/self/fd/7"
    for rejected in (
        "/proc/self/fd/0", "/proc/self/fd/07", "/proc/self/fd/-1",
        "/workspace/run/run-1/state/prepared-inputs/private-dataset.jsonl",
    ):
        assert _expected_dataset_path(dataset, roots, rejected) is None


def test_worker_carries_and_materializes_private_bytes_before_process(monkeypatch):
    from tests.execution.providers.test_modal_coordinator_worker import admit, case
    from tuner.execution.providers.modal.coordinator_worker import _execute_modal_worker

    payload = _payload()
    config = _prepared_config(payload)
    invocation = admit(case(
        config_extra={"dataset": config["dataset"], "sft": config["sft"]},
        private_dataset_bytes=payload,
    ))
    assert invocation.private_dataset_bytes == payload
    assert "private_dataset_bytes" not in repr(invocation)
    assert "PRIVATE_REPR_SENTINEL" not in repr(invocation)
    events: list[str] = []

    class Sources:
        def prepare_and_verify(self, *args):
            events.append("source")

    class Processes:
        def run(self, argv, **kwargs):
            events.append("process")
            assert kwargs["stdin"] == invocation.workload
            return ModalProcessResult(0)

    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_worker.materialize_private_dataset",
        lambda state, binding, content: events.append("materialize")
        or state / binding.relative_path,
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_worker.worker_source.read_locked_closure_manifest",
        lambda source: events.append("closure") or invocation.closure_manifest,
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_worker.worker_source.write_runtime_closure_manifest",
        lambda *args: events.append("write"),
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_worker.worker_source.stage_runtime_worker",
        lambda *args: events.append("stage"),
    )
    assert _execute_modal_worker(
        invocation,
        sources=Sources(),
        processes=Processes(),
        commit_prepared=lambda: None,
    ).returncode == 0
    assert events == ["source", "materialize", "closure", "write", "stage", "process"]


def test_v3_transport_rejects_private_member_above_its_bound_before_decode():
    payload = _payload()
    binding, material, recipes, policy, closure = _prepared_fixture(payload)
    bundle = ModalCoordinatorBundle.build(
        binding,
        material,
        recipes,
        log_terminal_policy=policy,
        worker_closure_manifest=closure,
        private_dataset_bytes=payload,
    )
    outer = json.loads(base64.b64decode(bundle.transport_bytes))
    private = next(
        member for member in outer["members"]
        if member["name"] == "private-dataset.jsonl"
    )
    oversized = b"x" * (MAX_PRIVATE_DATASET_BYTES + 1)
    private["content_base64"] = base64.b64encode(oversized).decode("ascii")
    private["size"] = len(oversized)
    private["sha256"] = hashlib.sha256(oversized).hexdigest()
    transport = base64.b64encode(json.dumps(
        outer, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8"))
    with pytest.raises(ValueError, match="bound|transport"):
        ModalCoordinatorBundle.parse_transport(
            transport, binding=binding, recipes=recipes,
        )
