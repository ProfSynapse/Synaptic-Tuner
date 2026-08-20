from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.cloud.hf_provider_adapter import HFProviderAdapter, PINNED_HF_HUB_VERSION
from tuner.cloud.hf_provisioning import canonical_json_bytes, document_sha256
from tuner.cloud.hf_provisioning_operator import provision_hf_source_transport
from tuner.core.exceptions import CloudProviderError


MEMBERS = (
    "checkout-policy.json",
    "source-lock.json",
    "capsule/synaptic-bootstrap-capsule.json",
    "capsule/tuner/cloud/bootstrap_capsule.py",
    "capsule/tuner/cloud/bootstrap_core.py",
)


class BucketClient:
    def __init__(self, store=None, *, upload_error: bool = False) -> None:
        self.store = dict(store or {})
        self.upload_error = upload_error
        self.calls: list[str] = []

    def create_bucket(self, bucket_id, *, private=None, resource_group_id=None, region=None, exist_ok=False, token=None):
        self.calls.append("create")
        return SimpleNamespace(bucket_id=bucket_id)

    def bucket_info(self, bucket_id, *, token=None):
        self.calls.append("info")
        return SimpleNamespace(id=bucket_id, private=True)

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
        self.calls.append("list")
        return [SimpleNamespace(type="file", path=path) for path in sorted(self.store) if path.startswith(prefix)]

    def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None, token=None):
        self.calls.append("upload")
        if self.upload_error:
            raise RuntimeError(f"uncertain {token}")
        for source, destination in add:
            self.store[destination] = (
                bytes(source) if isinstance(source, bytes) else Path(source).read_bytes()
            )

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False, token=None):
        self.calls.append("download")
        for remote, local in files:
            Path(local).write_bytes(self.store[remote.path])


def _prepared(tmp_path: Path):
    root = tmp_path / "transport"
    bundle = root / "bundle"
    for index, member in enumerate(MEMBERS):
        path = bundle.joinpath(*member.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"member-{index}".encode())
    inventory = [
        {"path": member, "sha256": hashlib.sha256(bundle.joinpath(*member.split("/")).read_bytes()).hexdigest()}
        for member in sorted(MEMBERS)
    ]
    digest = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    descriptor_uri = "tracking://runs/run-test/source-transport/descriptor.json"
    descriptor = {
        "schema_version": "synaptic-hf-source-transport/v1",
        "run_id": "run-test",
        "profile": "C",
        "provider": "hf_jobs",
        "source_lock": {"uri": "tracking://runs/run-test/source-lock.json", "sha256": "1" * 64, "path": "source-lock.json"},
        "capsule": {
            "engine_commit": "2" * 40,
            "uri": "tracking://runs/run-test/source-transport/bundle/capsule",
            "root": "capsule",
            "manifest": {"path": "capsule/synaptic-bootstrap-capsule.json", "sha256": "3" * 64},
        },
        "checkout_policy": {
            "uri": "tracking://runs/run-test/source-transport/bundle/checkout-policy.json",
            "path": "checkout-policy.json",
            "sha256": "4" * 64,
        },
        "bundle": {"uri": "tracking://runs/run-test/source-transport/bundle", "content_sha256": digest},
        "volume": {
            "type": "bucket",
            "source": "owner/bucket",
            "path": f"prepared/run-test/{digest}",
            "mount_path": "/workspace/synaptic-bootstrap-input",
            "read_only": True,
        },
    }
    return SimpleNamespace(
        root=root,
        bundle_root=bundle,
        descriptor=descriptor,
        descriptor_uri=descriptor_uri,
        descriptor_sha256=document_sha256(descriptor),
    )


def _run(monkeypatch, prepared, client):
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning_operator.load_hf_source_transport",
        lambda *args, **kwargs: prepared,
    )
    provider = HFProviderAdapter(client, token="secret-value", client_version=PINNED_HF_HUB_VERSION)
    return provision_hf_source_transport(
        SimpleNamespace(),
        transport_root=prepared.root,
        descriptor_uri=prepared.descriptor_uri,
        source_lock_uri="tracking://runs/run-test/source-lock.json",
        provider=provider,
        actor="operator-1",
        asserted_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )


def _remote_store(prepared):
    prefix = prepared.descriptor["volume"]["path"]
    return {
        f"{prefix}/{member}": prepared.bundle_root.joinpath(*member.split("/")).read_bytes()
        for member in MEMBERS
    }


def test_empty_prefix_uploads_once_then_lists_downloads_and_emits_evidence(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    client = BucketClient()
    outcome = _run(monkeypatch, prepared, client)
    assert outcome.succeeded and outcome.mutated
    assert client.calls == ["create", "info", "list", "upload", "list", "download"]
    assert outcome.evidence["provider_receipt_id"].startswith("hf-prefix-")
    assert outcome.evidence_sha256 == document_sha256(outcome.evidence)


def test_identical_prefix_is_read_only_idempotent_verification(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    client = BucketClient(_remote_store(prepared))
    outcome = _run(monkeypatch, prepared, client)
    assert outcome.succeeded and not outcome.mutated
    assert "upload" not in client.calls
    assert client.calls[-1] == "download"


def test_nonempty_divergent_prefix_stops_without_mutation(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    prefix = prepared.descriptor["volume"]["path"]
    client = BucketClient({f"{prefix}/unexpected": b"x"})
    with pytest.raises(CloudProviderError, match="not exactly"):
        _run(monkeypatch, prepared, client)
    assert "upload" not in client.calls


def test_identical_paths_with_wrong_bytes_stop_without_mutation(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    store = _remote_store(prepared)
    store[next(iter(store))] = b"wrong"
    client = BucketClient(store)
    with pytest.raises(CloudProviderError, match="digest"):
        _run(monkeypatch, prepared, client)
    assert "upload" not in client.calls


def test_upload_ambiguity_returns_bounded_failure_without_evidence_or_retry(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    client = BucketClient(upload_error=True)
    outcome = _run(monkeypatch, prepared, client)
    assert not outcome.succeeded
    assert outcome.evidence is None and outcome.evidence_sha256 is None
    assert outcome.failure.code == "mutation_ambiguous"
    assert not outcome.failure.retryable
    assert "secret-value" not in outcome.failure.message
    assert client.calls.count("upload") == 1


def test_bucket_creation_ambiguity_returns_no_evidence_and_no_upload(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)

    class CreateFailure(BucketClient):
        def create_bucket(self, bucket_id, *, private=None, resource_group_id=None, region=None, exist_ok=False, token=None):
            self.calls.append("create")
            raise RuntimeError(f"uncertain {token}")

    client = CreateFailure()
    outcome = _run(monkeypatch, prepared, client)
    assert outcome.failure.code == "mutation_ambiguous"
    assert outcome.evidence is None
    assert client.calls == ["create"]


def test_invalid_evidence_identity_fails_before_any_provider_call(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    client = BucketClient()
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning_operator.load_hf_source_transport",
        lambda *args, **kwargs: prepared,
    )
    provider = HFProviderAdapter(client, token="secret", client_version=PINNED_HF_HUB_VERSION)
    with pytest.raises(CloudProviderError, match="invalid"):
        provision_hf_source_transport(
            SimpleNamespace(),
            transport_root=prepared.root,
            descriptor_uri=prepared.descriptor_uri,
            source_lock_uri="tracking://runs/run-test/source-lock.json",
            provider=provider,
            actor="invalid actor with spaces",
        )
    assert client.calls == []


def test_documented_directory_entries_are_validated_not_silently_filtered(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)

    class DirectoryClient(BucketClient):
        def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
            entries = super().list_bucket_tree(
                bucket_id, prefix=prefix, recursive=recursive, token=token
            )
            entries.extend(
                SimpleNamespace(type="directory", path=path)
                for path in (
                    prefix,
                    f"{prefix}/capsule",
                    f"{prefix}/capsule/tuner",
                    f"{prefix}/capsule/tuner/cloud",
                )
            )
            return entries

    outcome = _run(monkeypatch, prepared, DirectoryClient(_remote_store(prepared)))
    assert outcome.succeeded and not outcome.mutated


@pytest.mark.parametrize(
    "extra",
    [
        SimpleNamespace(type="symlink", path="placeholder"),
        SimpleNamespace(type="directory", path="placeholder"),
    ],
)
def test_unknown_or_unrelated_tree_entries_fail_closed(tmp_path, monkeypatch, extra) -> None:
    prepared = _prepared(tmp_path)

    class ExtraEntryClient(BucketClient):
        def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
            entries = super().list_bucket_tree(
                bucket_id, prefix=prefix, recursive=recursive, token=token
            )
            path = (
                f"{prefix}/unknown"
                if extra.type == "directory"
                else f"{prefix}/source-lock.json"
            )
            entries.append(SimpleNamespace(type=extra.type, path=path))
            return entries

    with pytest.raises(CloudProviderError, match="inspect|descriptor-identical"):
        _run(monkeypatch, prepared, ExtraEntryClient(_remote_store(prepared)))


def test_file_directory_path_collision_fails_closed(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)

    class CollisionClient(BucketClient):
        def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
            entries = super().list_bucket_tree(
                bucket_id, prefix=prefix, recursive=recursive, token=token
            )
            entries.append(
                SimpleNamespace(type="directory", path=f"{prefix}/source-lock.json")
            )
            return entries

    with pytest.raises(CloudProviderError, match="colliding"):
        _run(monkeypatch, prepared, CollisionClient(_remote_store(prepared)))


def test_upload_handoff_uses_authenticated_immutable_bytes_not_mutable_paths(tmp_path, monkeypatch) -> None:
    prepared = _prepared(tmp_path)
    original = _remote_store(prepared)

    class MutatingClient(BucketClient):
        def __init__(self):
            super().__init__()
            self.sources = []

        def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None, token=None):
            self.sources = [source for source, _destination in add]
            for member in MEMBERS:
                prepared.bundle_root.joinpath(*member.split("/")).write_bytes(b"changed")
            return super().batch_bucket_files(
                bucket_id,
                add=add,
                copy=copy,
                delete=delete,
                token=token,
            )

    client = MutatingClient()
    outcome = _run(monkeypatch, prepared, client)
    assert outcome.succeeded
    assert all(isinstance(source, bytes) for source in client.sources)
    assert client.store == original
