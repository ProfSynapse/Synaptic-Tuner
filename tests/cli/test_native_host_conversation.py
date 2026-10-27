"""Real conversation bytes through the Mac host adapter; no ML imports."""

from dataclasses import replace
import os
from pathlib import Path
import tempfile

import pytest

from examples.training_qualification import native_host
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tests.dataset_prep.test_conversation_messages import build
from tuner.runtime.native_mlx_input import read_retained_native_source


@pytest.fixture
def conversation():
    prepared = build()
    semantic = prepared.identity
    with tempfile.TemporaryDirectory(prefix="native-conversation-", dir=Path.home()) as name:
        root = Path(name) / semantic.dataset_id
        root.mkdir(mode=0o700)
        for member, raw in (("manifest.json", prepared.manifest_raw), ("dataset.jsonl", prepared.dataset_raw)):
            path = root / member
            path.write_bytes(raw)
            path.chmod(0o600)
        identity = PreparedTrainingInputIdentity(
            ref=f"prepared://sha256/{semantic.dataset_digest}", revision=semantic.dataset_digest,
            content_digest=semantic.dataset_sha256, size_bytes=semantic.dataset_bytes,
            format="synaptic-conversation-sft-row/v1",
        )
        yield root, identity, prepared


def test_retained_conversation_uses_shared_semantic_verifier_and_exact_bytes(conversation, monkeypatch):
    from tuner.dataset_prep import conversation_messages
    root, identity, prepared = conversation
    monkeypatch.setattr(conversation_messages, "snapshot_conversation_dataset",
                        lambda *a, **k: pytest.fail("Linux-only snapshot must not be used"))
    source = native_host.NativePreparedTrainingInputSource(root, identity)
    assert source.identity == identity
    assert read_retained_native_source(source, identity) == prepared.dataset_raw
    # The lease owns the verified copy, not a later reopening of mutable paths.
    lease = source.open_lease()
    try:
        (root / "dataset.jsonl").write_bytes(b"replacement")
        assert lease.take_stream().read() == prepared.dataset_raw
    finally:
        lease.close()
    with pytest.raises(ValueError):
        source.open_lease()


@pytest.mark.parametrize("change", ["manifest", "dataset", "extra", "missing", "symlink", "hardlink",
                                   "root_mode", "member_mode", "root_link", "name", "oversized"])
def test_conversation_rejects_changed_or_unsafe_publication(conversation, change):
    root, identity, _ = conversation
    dataset = root / "dataset.jsonl"
    selected = root
    if change == "manifest":
        (root / "manifest.json").write_bytes(b"{}")
    elif change == "dataset":
        dataset.write_bytes(dataset.read_bytes().replace(b"answer a", b"answer z"))
    elif change == "extra":
        (root / "extra").mkdir()
    elif change == "missing":
        dataset.unlink()
    elif change == "symlink":
        dataset.rename(root.parent / "other")
        dataset.symlink_to(root.parent / "other")
    elif change == "hardlink":
        os.link(dataset, root.parent / "other")
    elif change == "root_mode":
        root.chmod(0o750)
    elif change == "member_mode":
        dataset.chmod(0o644)
    elif change == "root_link":
        selected = root.parent / "alias"
        selected.symlink_to(root, target_is_directory=True)
    elif change == "name":
        selected = root.rename(root.parent / "different")
    else:
        with dataset.open("ab") as stream:
            stream.truncate(native_host.CONVERSATION_MAX_BYTES + 1)
    with pytest.raises((ValueError, OSError)):
        native_host.NativePreparedTrainingInputSource(selected, identity).open_lease()


def test_conversation_rejects_member_replacement_during_semantic_verification(conversation, monkeypatch):
    root, identity, _ = conversation
    verify = native_host.verify_conversation_bytes

    def replace_member(*args, **kwargs):
        result = verify(*args, **kwargs)
        dataset = root / "dataset.jsonl"
        replacement = root.parent / "replacement"
        replacement.write_bytes(dataset.read_bytes())
        replacement.chmod(0o600)
        replacement.replace(dataset)
        return result

    monkeypatch.setattr(native_host, "verify_conversation_bytes", replace_member)
    with pytest.raises(ValueError, match="changed during verification"):
        native_host.NativePreparedTrainingInputSource(root, identity).open_lease()


@pytest.mark.parametrize("field,value", [("revision", "f" * 64), ("content_digest", "f" * 64),
                                        ("size_bytes", 1)])
def test_conversation_rejects_selected_identity_mismatch(conversation, field, value):
    root, identity, _ = conversation
    changes = {field: value}
    if field == "revision":
        changes["ref"] = f"prepared://sha256/{value}"
    changed = replace(identity, **changes)
    with pytest.raises(ValueError, match="changed before staging"):
        native_host.NativePreparedTrainingInputSource(root, changed).open_lease()


@pytest.mark.parametrize("schema", ["syntunia-sft-row/v1", "syntunia-sft-row/v2"])
def test_legacy_publications_keep_existing_reader(conversation, monkeypatch, schema):
    root, identity, _ = conversation
    sentinel = object()
    monkeypatch.setattr(native_host.LocalPreparedTrainingInputSource, "open_lease", lambda self: sentinel)
    assert native_host.NativePreparedTrainingInputSource(root, replace(identity, format=schema)).open_lease() is sentinel
