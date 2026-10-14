"""Native-verified GRPO bytes -> admitted model -> public scripted HF publication.

The trusted source port follows actual G2 native verification, not a fabricated
TrainingAPI outcome. Synthetic CPU model files do not qualify ML or live Hub use.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.runtime.test_packaged_grpo_native import native as native  # noqa: F401
from tests.runtime.test_packaged_grpo_native import _assemble
from tests.runtime.test_packaged_grpo_compilation import _fixture
from tests.training.test_hf_model_publication import Hub, Registry, PARENT, OID, REF
from synaptic_tuner.api.v1.artifacts_facade import ArtifactsAPI, PublicationRequest
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.reference.authority import (
    ReferencePublicationEvidenceAuthorityV1,
)
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference.publication_store import DurablePublicationStoreV1
from synaptic_tuner.api.v1.reference.local_publication import LocalArtifactSpoolV1
from synaptic_tuner.api.v1.reference.hf_publication import (
    HFModelPublicationDestinationV1,
)
from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.execution.coordinator_v1.publication import (
    AuthenticatedVerifiedSourceV1,
    PublicationOperationsV1,
)
from tuner.inference import retrieved_model
from tuner.runtime.packaged_grpo_verification import verify_packaged_grpo_artifacts
from tuner.training.packaged_grpo_compilation import (
    compile_packaged_grpo_workload,
    parse_packaged_grpo_workload,
)
from tuner.training.recipes import canonical_json_bytes

ADDITIONS = {
    "tuner/runtime/releases.py",
    "tuner/training/grpo_compilation.py",
    "tuner/training/packaged_compilation.py",
    "tuner/training/packaged_grpo_compilation.py",
}


class NativeSource:
    """Trusted test-owned source admitted through the real G2 native verifier."""

    def __init__(self, native, authority):
        admitted, _, _, _ = native
        _, inventory, terminal = _assemble(native)
        assert verify_packaged_grpo_artifacts(
            admitted=admitted, inventory_bytes=inventory, terminal_bytes=terminal
        )
        self.run = TrainingRunRef(admitted.execution.run_ref, "fixture/grpo")
        self.values = {}
        items = []
        for item in json.loads(inventory)["artifacts"]:
            raw = (admitted.paths.artifacts / item["path"]).read_bytes()
            assert (
                len(raw) == item["size"]
                and hashlib.sha256(raw).hexdigest() == item["sha256"]
            )
            artifact = VerifiedArtifact(item["role"], item["sha256"], item["size"])
            self.values[artifact.role] = raw
            items.append(artifact)
        value = AuthenticatedVerifiedSourceV1(
            "synaptic-publication-verified-source/v1",
            self.run,
            tuple(sorted(items, key=lambda item: item.role)),
            hashlib.sha256(inventory + terminal).hexdigest(),
            "host-publication",
            authority.key_ref,
            "0" * 64,
        )
        self.value = replace(
            value, tag=authority.sign("publication-verified-source/v1", value.payload)
        )
        self.opens = 0
        self.read_only = False

    def describe(self, run):
        assert run == self.run
        return self.value

    def open(self, request):
        assert not self.read_only, "cold publication must not reopen native artifacts"
        assert request.run == self.run
        artifact = next(
            item for item in self.value.artifacts if item.role == request.role
        )
        raw = self.values[artifact.role]
        assert len(raw) <= request.maximum_bytes
        self.opens += 1

        class Stream:
            run = request.run
            maximum_bytes = request.maximum_bytes

            def iter_bytes(self):
                for offset in range(0, len(raw), 1 << 20):
                    yield raw[offset : offset + (1 << 20)]

        stream = Stream()
        stream.artifact = artifact
        return stream


@contextmanager
def publication(root, records, source, authority, hub, *, create):
    if create:
        for name in ("locks", "spool", "render"):
            (root / name).mkdir(mode=0o700)
    deadline = time.monotonic() + 60
    store = DurablePublicationStoreV1(records, root / "locks", create=create)
    spool = (
        LocalArtifactSpoolV1(
            root / "spool",
            maximum_artifact_bytes=4 << 20,
            maximum_total_bytes=8 << 20,
            deadline=deadline,
        )
        if create
        else None
    )
    clock = lambda: "2026-10-14T12:00:00Z"
    destination = HFModelPublicationDestinationV1(
        destination_ref=REF,
        display_name="Scripted verified GRPO model",
        repo_id="fixture/model",
        branch="main",
        parent_commit=PARENT,
        secret_ref=SecretRef("env", "TEST_HF_PUBLICATION_TOKEN"),
        secrets=hub,
        authority_ref="host-publication",
        authority=authority,
        clock=clock,
        deadline=deadline,
        maximum_artifact_bytes=4 << 20,
        maximum_total_bytes=8 << 20,
        maximum_upload_bytes=16 << 20,
        render_root=root / "render" if create else None,
        spool=spool,
        api_factory=hub.api,
        http_factory=hub.http,
    )
    api = ArtifactsAPI(
        PublicationOperationsV1(
            store=store,
            spool=spool,
            destinations=Registry(destination),
            sources=source,
            authority=authority,
            clock=clock,
        )
    )
    try:
        yield api, store, destination
    finally:
        destination.close()
        if spool:
            spool.close()
        store.close()


@pytest.mark.parametrize("lost_reply", [False, True])
def test_native_grpo_public_hf_projection_and_cold_snapshot(
    native, tmp_path, lost_reply
):
    authority = ReferencePublicationEvidenceAuthorityV1(
        "fixture-publication-key", b"x" * 32
    )
    source = NativeSource(native, authority)
    hub = Hub()
    hub.lose_reply = lost_reply
    root = tmp_path / "publication"
    root.mkdir(mode=0o700)
    database = root / "store.sqlite3"
    with (
        SQLiteDurableStoreV1(database) as records,
        publication(root, records, source, authority, hub, create=True) as (
            api,
            store,
            destination,
        ),
    ):
        first = api.publish(PublicationRequest(source.run, REF))
        assert first.state.value == ("ambiguous" if lost_reply else "verified")
        assert source.opens == 5 and len(hub.calls) == 1
        record = store.get(first.publication.publication_id)
        prefix = "releases/" + record.command.publication_id
        marker = json.loads(hub.files[OID][prefix + "/publication.json"])
        model = parse_packaged_grpo_workload(source.values["workload_record"]).document[
            "core_workload"
        ]["model"]
        assert marker["model"] == {
            "kind": "lora",
            "model_ref": model["model_id"],
            "model_revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"],
        }
        assert any(
            item["path"].endswith("/adapter_model.safetensors")
            for item in marker["rendered"]
        )
        assert any(
            item["path"].endswith("/tokenizer.json") for item in marker["rendered"]
        )
        for item in marker["native"]:
            assert hub.files[OID][item["path"]] == source.values[item["role"]]
            assert (
                item["sha256"]
                == hashlib.sha256(source.values[item["role"]]).hexdigest()
            )
    source.read_only = True
    hub.lose_reply = False
    with (
        SQLiteDurableStoreV1(database, create=False) as records,
        publication(root, records, source, authority, hub, create=False) as (
            api,
            store,
            destination,
        ),
    ):
        recovered = api.publish(PublicationRequest(source.run, REF))
        assert recovered.state.value == "verified"
        assert api.verify(first.publication).verified
        receipt = store.get(first.publication.publication_id).receipt
        for item in receipt.inventory.inventory.artifacts:
            assert (
                b"".join(destination.iter_bytes(record.command, item, 4 << 20))
                == source.values[item.role]
            )
        assert source.opens == 5 and len(hub.calls) == 1
        assert (
            receipt.ownership_id
            == store.get(first.publication.publication_id)
            .ownership_history[0]
            .ownership_id
        )


def test_parser_imports_only_four_reviewed_inference_members(tmp_path):
    core, context = _fixture()
    policy = {
        "max_trainable_bytes": 16,
        "max_trainable_tensors": 2,
        "max_reward_values": 3,
    }
    compiled = compile_packaged_grpo_workload(
        core_workload=core,
        context=context,
        child_max_seconds=60,
        qualification_evidence=policy,
    )
    path = tmp_path / "workload.json"
    path.write_bytes(compiled.canonical_bytes)
    root = Path(retrieved_model.__file__).resolve().parents[2]
    code = """import sys,json
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from tuner.training.packaged_grpo_compilation import parse_packaged_grpo_workload
result=parse_packaged_grpo_workload(Path(sys.argv[2]).read_bytes())
assert result.canonical_bytes==Path(sys.argv[2]).read_bytes()
paths=set()
for m in list(sys.modules.values()):
 f=getattr(m,'__file__',None)
 if f and Path(f).resolve().is_relative_to(root):
  paths.add(Path(f).resolve().relative_to(root).as_posix())
print(json.dumps(sorted(paths)))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root), str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    loaded = set(json.loads(completed.stdout))
    # The locked119 predecessor remains unchanged until root-reviewed refresh.
    locked = {
        item["path"]
        for item in json.loads(
            (
                root / "tuner/execution/providers/modal/inference-worker-closure.json"
            ).read_bytes()
        )["members"]
    }
    assert loaded - locked <= ADDITIONS
    assert ADDITIONS <= loaded
    assert not any(path.startswith(("Trainers/", "shared/")) for path in loaded)
    assert not any(
        "packaged_sft_execution" in path or "packaged_training_worker" in path
        for path in loaded
    )


@pytest.mark.parametrize(
    "mutation", ["method", "model", "context", "extra", "noncanonical"]
)
def test_grpo_model_selector_rejects_mixed_or_malformed_declarations(
    tmp_path, mutation
):
    core, context = _fixture()
    compiled = compile_packaged_grpo_workload(
        core_workload=core, context=context, child_max_seconds=60
    )
    doc = compiled.document
    if mutation == "method":
        doc["method"] = "sft"
    elif mutation == "model":
        doc["core_workload"]["model"]["model_id"] = "wrong/model"
    elif mutation == "context":
        doc["context"]["runtime_release"]["compatible_methods"] = ["sft"]
    elif mutation == "extra":
        doc["extra"] = True
    raw = (
        canonical_json_bytes(doc)
        if mutation != "noncanonical"
        else b" " + compiled.canonical_bytes
    )
    (tmp_path / "workload_record.tar").write_bytes(raw)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises((ValueError, TypeError, KeyError)):
            retrieved_model._workload_model(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "policy,expected",
    [
        (None, "29e2d227e9c803a6bb7c75e09088a0463a618c067031bf9a82822388727a9c68"),
        (
            {
                "max_trainable_bytes": 16,
                "max_trainable_tensors": 2,
                "max_reward_values": 3,
            },
            "e9888bd1c1b1f80d4319346160de15cedf8e3807cec528a516713902c0c391c5",
        ),
    ],
)
def test_grpo_parser_refactor_preserves_pre_refactor_canonical_bytes(policy, expected):
    # Captured from accepted pre-refactor compiler, with the same explicit fixture.
    core, context = _fixture()
    compiled = compile_packaged_grpo_workload(
        core_workload=core,
        context=context,
        child_max_seconds=60,
        qualification_evidence=policy,
    )
    assert hashlib.sha256(compiled.canonical_bytes).hexdigest() == expected
    assert (
        parse_packaged_grpo_workload(compiled.canonical_bytes).canonical_bytes
        == compiled.canonical_bytes
    )


@pytest.mark.parametrize("role", ["final_model", "tokenizer"])
def test_hash_consistent_native_archive_fault_refuses_before_credentials(
    native, tmp_path, role
):
    import io
    import tarfile

    authority = ReferencePublicationEvidenceAuthorityV1(
        "fixture-publication-key", b"x" * 32
    )
    source = NativeSource(native, authority)
    # Deliberate trusted-source corruption after real native admission. The
    # independent renderer must reject these rehashed malformed archives too.
    output = io.BytesIO()
    target = "adapter_config.json" if role == "final_model" else "tokenizer.json"
    changed = False
    with (
        tarfile.open(fileobj=io.BytesIO(source.values[role]), mode="r:*") as original,
        tarfile.open(fileobj=output, mode="w") as rewritten,
    ):
        for member in original.getmembers():
            data = original.extractfile(member).read() if member.isfile() else None
            if member.name.endswith(target):
                data = (
                    b'{"peft_type":"LORA","base_model_name_or_path":"wrong/model"}'
                    if role == "final_model"
                    else b"[]"
                )
                member.size = len(data)
                changed = True
            rewritten.addfile(member, io.BytesIO(data) if data is not None else None)
    assert changed
    raw = output.getvalue()
    source.values[role] = raw
    items = tuple(
        VerifiedArtifact(item.role, hashlib.sha256(raw).hexdigest(), len(raw))
        if item.role == role
        else item
        for item in source.value.artifacts
    )
    value = replace(source.value, artifacts=items, tag="0" * 64)
    source.value = replace(
        value, tag=authority.sign("publication-verified-source/v1", value.payload)
    )
    root = tmp_path / "publication-fault"
    root.mkdir(mode=0o700)
    hub = Hub()
    with (
        SQLiteDurableStoreV1(root / "db") as records,
        publication(root, records, source, authority, hub, create=True) as (api, _, _),
    ):
        result = api.publish(PublicationRequest(source.run, REF))
        assert result.state.value != "verified"
        assert hub.secrets == 0 and not hub.calls
