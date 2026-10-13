from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import tuner

from synaptic_tuner.api.v1.data_facade import DataOutcome, DataRunRef, DataRunState, DataScenarioOutcome
from synaptic_tuner.api.v1.results import VerifiedArtifact
from tuner.dataset_prep import DatasetPrepValidationError, DatasetPrepCollisionError
from tuner.dataset_prep.conversation_messages import (
    CONFIG_SCHEMA, MANIFEST_SCHEMA, ConversationNormalizerConfigV1,
    build_conversation_publication, canonical, domain, publish_conversation_dataset,
    read_source_members, sha, snapshot_conversation_dataset, verify_bytes,
)


def fixture(*, alternate=False):
    names = ("speaker", "body", "dialog") if alternate else ("role", "content", "conversations")
    role, content, messages = names
    rows = []
    for group, split in (("a", "learning"), ("b", "heldout")):
        rows.append({messages: [{role: "instruction", content: "system " + group},
                                {role: "human", content: "question " + group},
                                {role: "model", content: "previous " + group},
                                {role: "human", content: "follow up " + group},
                                {role: "model", content: "answer " + group}],
                     "group": group, "partition": split, "unrelated": "id " + group})
    dataset = b"".join(canonical(row) + b"\n" for row in rows)
    metadata = canonical({"count": 2, "private_note": "not public"})
    document = {
        "schema_version": CONFIG_SCHEMA,
        "source": {"dataset": {"member": "raw.jsonl", "sha256": sha(dataset), "size_bytes": len(dataset)},
                   "metadata": {"member": "run.json", "sha256": sha(metadata), "size_bytes": len(metadata),
                                "row_count_path": ["count"]},
                   "rows": 2, "provenance": {"origin": "host-declared"}, "data_outcome": None},
        "projection": {"messages_path": [messages], "role_field": role, "content_field": content,
                       "role_map": {"instruction": "system", "human": "user", "model": "assistant"}},
        "selection": {"rows": None, "target_index": None},
        "grouping": {"path": ["group"]},
        "split": {"kind": "field", "path": ["partition"], "values": {"learning": "train", "heldout": "validation"}},
    }
    return document, dataset, metadata


def outcome(document):
    source = document["source"]
    return DataOutcome(DataRunRef("run-generated", "project-a"), DataRunState.SUCCEEDED, 2, 2,
                       (DataScenarioOutcome("scenario-a", 2, 2),),
                       tuple(VerifiedArtifact(role, source[name]["sha256"], source[name]["size_bytes"])
                             for role, name in (("dataset_jsonl", "dataset"), ("dataset_metadata", "metadata"))))


def build(document=None, dataset=None, metadata=None):
    if document is None:
        document, dataset, metadata = fixture()
    return build_conversation_publication(ConversationNormalizerConfigV1.from_dict(document), dataset, metadata)


def rehash(manifest, rows):
    from tuner.dataset_prep.conversation_messages import _row_id
    for row, entry in zip(rows, manifest["provenance"]):
        body = {key: value for key, value in row.items() if key != "row_id"}
        row["row_id"] = _row_id(manifest["configuration"], body, entry)
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    manifest.update(dataset_sha256=sha(raw), dataset_bytes=len(raw),
                    row_ids_sha256=sha(canonical([row["row_id"] for row in rows])))
    semantic = {k: v for k, v in manifest.items() if k not in {"dataset_digest", "dataset_id"}}
    manifest["dataset_digest"] = domain(MANIFEST_SCHEMA, semantic)
    manifest["dataset_id"] = "dataset-" + manifest["dataset_digest"]
    return canonical(manifest), raw


@pytest.mark.parametrize("alternate", [False, True])
def test_generic_mapping_preserves_order_target_and_provenance(alternate):
    document, dataset, metadata = fixture(alternate=alternate)
    prepared = build(document, dataset, metadata)
    rows = [json.loads(line) for line in prepared.dataset_raw.splitlines()]
    assert [message["role"] for message in rows[0]["messages"]] == ["system", "user", "assistant", "user", "assistant"]
    assert rows[0]["messages"][-1]["content"] == "answer a"
    manifest = json.loads(prepared.manifest_raw)
    assert manifest["configuration"]["source"]["data_outcome"] is None
    assert manifest["configuration"]["source"]["dataset"]["sha256"] == sha(dataset)
    assert manifest["provenance"][0]["target_message_index"] == 4
    assert verify_bytes(Path(prepared.identity.dataset_id), prepared.manifest_raw, prepared.dataset_raw).semantic_identity == prepared.identity


def test_earlier_target_explicitly_excludes_suffix_and_changes_identity():
    document, dataset, metadata = fixture()
    original = build(document, dataset, metadata)
    document["selection"]["target_index"] = 2
    selected = build(document, dataset, metadata)
    rows = [json.loads(line) for line in selected.dataset_raw.splitlines()]
    assert len(rows[0]["messages"]) == 3
    assert rows[0]["messages"][-1]["content"] == "previous a"
    assert original.identity != selected.identity
    assert json.loads(selected.manifest_raw)["provenance"][0]["message_count"] == 5


def test_optional_existing_data_outcome_is_retained_and_fully_checked():
    document, dataset, metadata = fixture()
    public = outcome(document)
    config = ConversationNormalizerConfigV1.from_data_outcome(document, public)
    prepared = build_conversation_publication(config, dataset, metadata)
    assert json.loads(prepared.manifest_raw)["configuration"]["source"]["data_outcome"] == public.to_dict()
    assert config.to_dict()["source"]["data_outcome"]["run"] == public.run.to_dict()
    assert document["source"]["data_outcome"] is None  # caller copy isolated


@pytest.mark.parametrize("change", ["state", "count", "artifact", "metadata_missing", "extra_artifact"])
def test_data_origin_cannot_silently_drop_or_change_commitments(change):
    document, dataset, metadata = fixture()
    public = outcome(document).to_dict()
    if change == "state":
        public["state"] = "partially_succeeded"
    elif change == "count":
        public["rows_requested"] = public["rows_written"] = 3
        public["scenarios"][0]["rows_requested"] = public["scenarios"][0]["rows_written"] = 3
    elif change == "artifact":
        public["artifacts"][0]["sha256"] = "0" * 64
    elif change == "metadata_missing":
        document["source"]["metadata"] = None
    else:
        public["artifacts"].append({"role": "other", "sha256": "0" * 64, "size_bytes": 1})
    document["source"]["data_outcome"] = public
    with pytest.raises((ValueError, TypeError)):
        build(document, dataset, None if change == "metadata_missing" else metadata)


@pytest.mark.parametrize("case", ["unknown_heldout", "one_split", "same_group", "duplicate_conversation", "duplicate_prefix", "metadata_count", "hash", "structured", "tool_calls", "target", "late_system"])
def test_invalid_projection_split_or_source_rejects(case):
    document, dataset, metadata = fixture()
    rows = [json.loads(line) for line in dataset.splitlines()]
    if case == "unknown_heldout":
        document["split"] = {"kind": "heldout_groups", "validation_groups": ["b", "missing"]}
    elif case == "one_split":
        rows[1]["partition"] = "learning"
    elif case == "same_group":
        rows[1]["group"] = "a"
    elif case == "duplicate_conversation":
        rows[1]["conversations"] = copy.deepcopy(rows[0]["conversations"])
    elif case == "duplicate_prefix":
        rows[1]["conversations"][:3] = copy.deepcopy(rows[0]["conversations"][:3])
        document["selection"]["target_index"] = 2
    elif case == "metadata_count":
        metadata = canonical({"count": 3})
        document["source"]["metadata"].update(sha256=sha(metadata), size_bytes=len(metadata))
    elif case == "hash":
        document["source"]["dataset"]["sha256"] = "0" * 64
    elif case == "structured":
        rows[0]["conversations"][-1]["content"] = {"text": "answer"}
    elif case == "tool_calls":
        rows[0]["conversations"][-1]["content"] = [{"tool_calls": [{"name": "x"}]}]
    elif case == "target":
        document["selection"]["target_index"] = 3
    else:
        rows[0]["conversations"][1]["role"] = "instruction"
    if case != "hash":
        dataset = b"".join(canonical(row) + b"\n" for row in rows)
        document["source"]["dataset"].update(sha256=sha(dataset), size_bytes=len(dataset))
    with pytest.raises(DatasetPrepValidationError):
        build(document, dataset, metadata)


@pytest.mark.parametrize("case", ["group", "count", "target", "coverage", "split_value", "prefix_hash"])
def test_independent_verifier_rejects_hash_consistent_semantic_corruption(case):
    prepared = build()
    manifest = json.loads(prepared.manifest_raw)
    rows = [json.loads(line) for line in prepared.dataset_raw.splitlines()]
    if case == "group":
        rows[1]["group_id"] = manifest["provenance"][1]["group_id"] = "a"
    elif case == "count":
        manifest["split_counts"] = {"train": 0, "validation": 2}
    elif case == "target":
        rows[0]["target_message_index"] = manifest["provenance"][0]["target_message_index"] = 3
    elif case == "coverage":
        rows[1]["source_row_number"] = manifest["provenance"][1]["source_row_number"] = 1
    elif case == "split_value":
        manifest["provenance"][1]["declared_split_value"] = "learning"
    else:
        manifest["provenance"][0]["prefix_sha256"] = "0" * 64
    raw_manifest, raw_rows = rehash(manifest, rows)
    with pytest.raises(DatasetPrepValidationError):
        verify_bytes(Path(manifest["dataset_id"]), raw_manifest, raw_rows)


def test_publication_reuse_collision_and_corruption(tmp_path):
    prepared = build()
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    result = publish_conversation_dataset(root, prepared)
    assert publish_conversation_dataset(root, prepared) == result
    verified, raw = snapshot_conversation_dataset(result.path)
    assert verified == result and raw == prepared.dataset_raw
    (result.path / "dataset.jsonl").chmod(0o600)
    (result.path / "dataset.jsonl").write_bytes(b"bad\n")
    with pytest.raises(DatasetPrepCollisionError):
        publish_conversation_dataset(root, prepared)


@pytest.mark.parametrize("member_kind", ["symlink", "hardlink", "fifo"])
def test_unsafe_source_members_reject_before_read(tmp_path, member_kind):
    import os
    document, dataset, metadata = fixture()
    (tmp_path / "run.json").write_bytes(metadata)
    external = tmp_path / "external"
    external.write_bytes(dataset)
    member = tmp_path / "raw.jsonl"
    if member_kind == "symlink":
        member.symlink_to(external)
    elif member_kind == "hardlink":
        os.link(external, member)
    else:
        os.mkfifo(member)
    with pytest.raises(ValueError):
        read_source_members(tmp_path, ConversationNormalizerConfigV1.from_dict(document))


def test_declared_message_projection_omits_extra_fields_without_coercing_content():
    document, dataset, metadata = fixture()
    rows = [json.loads(line) for line in dataset.splitlines()]
    rows[0]["conversations"][0]["auxiliary"] = {"timestamp": 123, "opaque": [1, 2]}
    dataset = b"".join(canonical(row) + b"\n" for row in rows)
    document["source"]["dataset"].update(sha256=sha(dataset), size_bytes=len(dataset))
    prepared = build(document, dataset, metadata)
    published = json.loads(prepared.dataset_raw.splitlines()[0])
    assert set(published["messages"][0]) == {"role", "content"}
    assert json.loads(prepared.manifest_raw)["provenance"][0]["source_row_sha256"] == sha(canonical(rows[0]))


def test_explicit_selection_and_heldout_groups_cover_declared_rows():
    document, dataset, metadata = fixture()
    rows = [json.loads(line) for line in dataset.splitlines()]
    rows.insert(1, {"ignored": "not a conversation by declared row selection"})
    dataset = b"".join(canonical(row) + b"\n" for row in rows)
    document["source"]["dataset"].update(sha256=sha(dataset), size_bytes=len(dataset))
    document["source"]["rows"] = 3
    document["source"]["metadata"] = None
    document["selection"]["rows"] = [1, 3]
    document["split"] = {"kind": "heldout_groups", "validation_groups": ["b"]}
    prepared = build(document, dataset, None)
    assert [json.loads(line)["source_row_number"] for line in prepared.dataset_raw.splitlines()] == [1, 3]
    verify_bytes(Path(prepared.identity.dataset_id), prepared.manifest_raw, prepared.dataset_raw)


@pytest.mark.parametrize("case", ["duplicate_key", "nonfinite", "bool_index", "depth", "tuple", "cycle", "path", "unknown_field"])
def test_bounded_closed_config_and_json(case):
    document, _, _ = fixture()
    if case == "duplicate_key":
        raw = canonical(document).decode().replace('"rows":2', '"rows":2,"rows":2')
        with pytest.raises(ValueError):
            ConversationNormalizerConfigV1(raw)
        return
    if case == "nonfinite":
        document["source"]["provenance"]["bad"] = float("inf")
    elif case == "bool_index":
        document["selection"]["target_index"] = True
    elif case == "depth":
        child = {}
        document["source"]["provenance"] = child
        for _ in range(33):
            child["nested"] = {}
            child = child["nested"]
    elif case == "tuple":
        document["selection"]["rows"] = (1, 2)
    elif case == "cycle":
        document["source"]["provenance"]["cycle"] = document
    elif case == "path":
        document["source"]["dataset"]["member"] = "../raw.jsonl"
    else:
        document["unknown"] = 1
    with pytest.raises(ValueError):
        ConversationNormalizerConfigV1.from_dict(document)


def test_schema_assets_match_actual_configuration_rows_and_manifest():
    import jsonschema
    prepared = build()
    root = Path(tuner.__file__).resolve().parent.parent / "schemas"
    values = {
        "conversation-preparation": json.loads(prepared.manifest_raw)["configuration"],
        "conversation-prepared-dataset": json.loads(prepared.manifest_raw),
        "conversation-sft-row": json.loads(prepared.dataset_raw.splitlines()[0]),
    }
    for name, value in values.items():
        schema = json.loads((root / f"synaptic-{name}-v1.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(value, schema)


@pytest.mark.parametrize("unsafe", ["root_mode", "member_mode", "hardlink", "symlink"])
def test_publication_reader_refuses_unsafe_custody(tmp_path, unsafe):
    import os
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    verified = publish_conversation_dataset(root, build())
    dataset = verified.path / "dataset.jsonl"
    if unsafe == "root_mode":
        verified.path.chmod(0o755)
    elif unsafe == "member_mode":
        dataset.chmod(0o644)
    elif unsafe == "hardlink":
        os.link(dataset, tmp_path / "link")
    else:
        dataset.rename(tmp_path / "elsewhere")
        dataset.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError):
        snapshot_conversation_dataset(verified.path)


def test_source_parent_replacement_during_retained_read_rejects(tmp_path, monkeypatch):
    import os
    source = tmp_path / "source"
    source.mkdir()
    document, dataset, metadata = fixture()
    (source / "raw.jsonl").write_bytes(dataset)
    (source / "run.json").write_bytes(metadata)
    original = os.read
    changed = False
    def replace_parent(descriptor, length):
        nonlocal changed
        result = original(descriptor, length)
        if not changed:
            changed = True
            source.rename(tmp_path / "old")
            source.mkdir()
        return result
    monkeypatch.setattr(os, "read", replace_parent)
    with pytest.raises(ValueError):
        read_source_members(source, ConversationNormalizerConfigV1.from_dict(document))
    assert (tmp_path / "old" / "raw.jsonl").read_bytes() == dataset
    assert not list(source.iterdir())
