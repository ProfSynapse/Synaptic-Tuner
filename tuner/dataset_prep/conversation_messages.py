"""Immutable text-conversation preparation; no trainer/release admission.

One declared assistant target and its ordered prefix form each row. Provenance
is an exact host declaration, not an assertion of chronological independence.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
from types import MappingProxyType

from synaptic_tuner.api.v1.data_facade import DataOutcome, DataRunState
from .models import DatasetPrepValidationError, DatasetSemanticIdentityV1, VerifiedPreparedDatasetV1
from . import publication as storage

CONFIG_SCHEMA = "synaptic-conversation-preparation/v1"
ROW_SCHEMA = "synaptic-conversation-sft-row/v1"
MANIFEST_SCHEMA = "synaptic-conversation-prepared-dataset/v1"
NORMALIZER_REF = "synaptic.conversation-preparation/v1"
MAX_BYTES = 64 * 1024 * 1024
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_ROW_BYTES = 1024 * 1024
MAX_ROWS = 100_000
MAX_EXECUTABLE_BYTES = 2 * 1024 * 1024


def invalid() -> None:
    raise DatasetPrepValidationError("conversation preparation input is invalid") from None


def canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError):
        invalid()


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def domain(name: str, value: object) -> str:
    return sha(name.encode("ascii") + b"\0" + canonical(value))


def _tree(value: object, depth: int = 0) -> None:
    if depth > 32:
        invalid()
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                invalid()
            _tree(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _tree(child, depth + 1)
    elif type(value) not in (str, int, float, bool, type(None)):
        invalid()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            invalid()
        result[key] = value
    return result


def _json_budget(raw: bytes) -> None:
    depth = nodes = 0
    quoted = escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            nodes += 1
            if depth > 32:
                invalid()
        elif byte in (93, 125):
            depth -= 1
        elif byte in (44, 58):
            nodes += 1
        if nodes > 500_000:
            invalid()


def parse(raw: bytes, maximum: int) -> object:
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        invalid()
    _json_budget(raw)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda _: invalid())
        _tree(value)
        canonical(value)
        return value
    except (ValueError, TypeError, RecursionError, UnicodeError):
        invalid()


def exact(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        invalid()
    return value


def count(value, minimum=1, maximum=MAX_ROWS):
    if type(value) is not int or not minimum <= value <= maximum:
        invalid()
    return value


def text(value):
    if type(value) is not str or not value or len(value.encode("utf-8")) > 256:
        invalid()
    if any(ord(c) < 32 for c in value):
        invalid()
    return value


def digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        invalid()
    return value


def path(value):
    if type(value) is not list or not 1 <= len(value) <= 16:
        invalid()
    for part in value:
        text(part)
    return value


def at(value, keys):
    for key in keys:
        if type(value) is not dict or key not in value:
            invalid()
        value = value[key]
    return value


def _member(value, *, metadata=False):
    fields = {"member", "sha256", "size_bytes"}
    if metadata:
        fields.add("row_count_path")
    exact(value, fields)
    name = text(value["member"])
    # Deliberately single-leaf source members: no ancestor writes or traversal.
    if name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
        invalid()
    digest(value["sha256"])
    count(value["size_bytes"], maximum=MAX_DOCUMENT_BYTES if metadata else MAX_BYTES)
    if metadata and value["row_count_path"] is not None:
        path(value["row_count_path"])


def _validate_config(value):
    exact(value, {"schema_version", "source", "projection", "selection", "grouping", "split"})
    if value["schema_version"] != CONFIG_SCHEMA:
        invalid()
    source = exact(value["source"], {"dataset", "metadata", "rows", "provenance", "data_outcome"})
    _member(source["dataset"])
    if source["metadata"] is not None:
        _member(source["metadata"], metadata=True)
        if source["metadata"]["member"] == source["dataset"]["member"]:
            invalid()
    count(source["rows"])
    if type(source["provenance"]) is not dict:
        invalid()
    outcome = source["data_outcome"]
    if outcome is not None:
        admitted = DataOutcome.from_dict(outcome)
        if admitted.state is not DataRunState.SUCCEEDED or admitted.rows_written != admitted.rows_requested:
            invalid()
        if canonical(admitted.to_dict()) != canonical(outcome) or admitted.rows_written != source["rows"]:
            invalid()
        artifacts = {item.role: item for item in admitted.artifacts}
        if set(artifacts) != {"dataset_jsonl", "dataset_metadata"} or source["metadata"] is None:
            invalid()
        for role, member in (("dataset_jsonl", source["dataset"]), ("dataset_metadata", source["metadata"])):
            observed = artifacts[role]
            if observed.sha256 != member["sha256"] or observed.size_bytes != member["size_bytes"]:
                invalid()
    projection = exact(value["projection"], {"messages_path", "role_field", "content_field", "role_map"})
    path(projection["messages_path"])
    text(projection["role_field"])
    text(projection["content_field"])
    if projection["role_field"] == projection["content_field"]:
        invalid()
    roles = projection["role_map"]
    if type(roles) is not dict or not 1 <= len(roles) <= 16:
        invalid()
    for key, role in roles.items():
        text(key)
        if role not in {"system", "user", "assistant"}:
            invalid()
    selection = exact(value["selection"], {"rows", "target_index"})
    rows = selection["rows"]
    if rows is not None:
        if type(rows) is not list or not 1 <= len(rows) <= MAX_ROWS:
            invalid()
        if rows != sorted(set(count(row) for row in rows)) or rows[-1] > source["rows"]:
            invalid()
    if selection["target_index"] is not None:
        count(selection["target_index"], minimum=0, maximum=127)
    group = exact(value["grouping"], {"path"})
    if group["path"] is not None:
        path(group["path"])
    split = value["split"]
    if type(split) is not dict:
        invalid()
    if split.get("kind") == "heldout_groups":
        exact(split, {"kind", "validation_groups"})
        groups = split["validation_groups"]
        if type(groups) is not list or not 1 <= len(groups) <= MAX_ROWS:
            invalid()
        for group in groups:
            text(group)
        if groups != sorted(set(groups)):
            invalid()
    elif split.get("kind") == "field":
        exact(split, {"kind", "path", "values"})
        path(split["path"])
        values = split["values"]
        if type(values) is not dict or not 1 <= len(values) <= 16:
            invalid()
        for key, name in values.items():
            text(key)
            if name not in {"train", "validation"}:
                invalid()
    else:
        invalid()
    return value


def validate_config(value):
    try:
        return _validate_config(value)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        invalid()


@dataclass(frozen=True, slots=True)
class ConversationNormalizerConfigV1:
    """One canonical configuration; generic provenance or optional DataOutcome."""
    canonical_json: str = field(repr=False)

    def __post_init__(self):
        if type(self.canonical_json) is not str:
            invalid()
        raw = self.canonical_json.encode("utf-8")
        value = parse(raw, MAX_DOCUMENT_BYTES)
        validate_config(value)
        if canonical(value) != raw:
            invalid()

    @property
    def normalizer_ref(self):
        return NORMALIZER_REF

    def to_dict(self):
        return parse(self.canonical_json.encode("utf-8"), MAX_DOCUMENT_BYTES)

    @classmethod
    def from_dict(cls, value):
        _tree(value)
        return cls(canonical(value).decode("utf-8"))

    @classmethod
    def from_data_outcome(cls, value, outcome):
        if type(outcome) is not DataOutcome:
            invalid()
        # Derive commitments into the single document; configured contradictions reject.
        document = parse(canonical(value), MAX_DOCUMENT_BYTES)
        source = document["source"]
        if source.get("data_outcome") is not None and source["data_outcome"] != outcome.to_dict():
            invalid()
        source["data_outcome"] = outcome.to_dict()
        return cls.from_dict(document)


def _source_rows(dataset):
    rows = []
    stream = BytesIO(dataset)
    while line := stream.readline(MAX_ROW_BYTES + 1):
        if len(line) > MAX_ROW_BYTES or len(rows) >= MAX_ROWS:
            invalid()
        value = parse(line, MAX_ROW_BYTES)
        if type(value) is not dict:
            invalid()
        rows.append(value)
    if not rows or not dataset.endswith(b"\n"):
        invalid()
    return rows


def _messages(source, projection):
    original = at(source, projection["messages_path"])
    if type(original) is not list or not 2 <= len(original) <= 128:
        invalid()
    result = []
    for item in original:
        if type(item) is not dict or not {projection["role_field"], projection["content_field"]} <= set(item):
            invalid()
        role = item[projection["role_field"]]
        if type(role) is not str or role not in projection["role_map"]:
            invalid()
        content = item[projection["content_field"]]
        if type(content) is not str or not content:
            invalid()
        result.append({"role": projection["role_map"][role], "content": content})
    return result


def _row_body(config, number, original):
    messages = _messages(original, config["projection"])
    target = config["selection"]["target_index"]
    target = len(messages) - 1 if target is None else target
    if target < 1 or target >= len(messages) or messages[target]["role"] != "assistant":
        invalid()
    selected = messages[:target + 1]
    if any(msg["role"] == "system" for msg in selected[1:]) or selected[0]["role"] == "assistant":
        invalid()
    original_digest = sha(canonical(original))
    group = config["grouping"]["path"]
    conversation_digest = sha(canonical(messages))
    group = conversation_digest if group is None else text(at(original, group))
    split = config["split"]
    if split["kind"] == "heldout_groups":
        split_name = "validation" if group in split["validation_groups"] else "train"
    else:
        split_value = at(original, split["path"])
        if type(split_value) is not str or split_value not in split["values"]:
            invalid()
        split_name = split["values"][split_value]
    row = {"schema_version": ROW_SCHEMA, "format": "messages", "source_row_number": number,
           "target_message_index": target, "group_id": group, "split": split_name, "messages": selected}
    provenance = {"source_row_number": number, "source_row_sha256": original_digest, "conversation_sha256": conversation_digest,
                  "declared_split_value": None if config["split"]["kind"] == "heldout_groups" else split_value,
                  "message_count": len(messages), "target_message_index": target,
                  "prefix_sha256": sha(canonical(selected[:-1])), "target_sha256": sha(canonical(selected[-1])),
                  "group_id": group, "split": split_name}
    return row, provenance


def _row_id(config, row, provenance):
    return "row-" + domain("synaptic-conversation-row/v1", {"config_digest": domain(CONFIG_SCHEMA, config),
                                                          "row": row, "provenance": provenance})


def _identity(manifest):
    return DatasetSemanticIdentityV1(
        manifest["dataset_id"], manifest["dataset_digest"], manifest["row_count"],
        manifest["dataset_bytes"], manifest["dataset_sha256"], manifest["row_ids_sha256"],
        MappingProxyType(manifest["split_counts"]),
    )


@dataclass(frozen=True, slots=True)
class ConversationPublicationV1:
    identity: DatasetSemanticIdentityV1
    dataset_raw: bytes = field(repr=False)
    manifest_raw: bytes = field(repr=False)

    def __post_init__(self):
        if (type(self.identity) is not DatasetSemanticIdentityV1
                or type(self.dataset_raw) is not bytes or not 0 < len(self.dataset_raw) <= MAX_BYTES
                or type(self.manifest_raw) is not bytes or not 0 < len(self.manifest_raw) <= MAX_DOCUMENT_BYTES):
            invalid()


def build_conversation_publication(config: ConversationNormalizerConfigV1, dataset: bytes,
                                   metadata: bytes | None = None) -> ConversationPublicationV1:
    if type(config) is not ConversationNormalizerConfigV1:
        invalid()
    document = config.to_dict()
    source = document["source"]
    for member, payload in ((source["dataset"], dataset), (source["metadata"], metadata)):
        if member is None:
            if payload is not None:
                invalid()
        elif type(payload) is not bytes or len(payload) != member["size_bytes"] or sha(payload) != member["sha256"]:
            invalid()
    original = _source_rows(dataset)
    if len(original) != source["rows"]:
        invalid()
    if metadata is not None:
        meta = parse(metadata, MAX_DOCUMENT_BYTES)
        count_path = source["metadata"]["row_count_path"]
        if count_path is not None and count(at(meta, count_path)) != len(original):
            invalid()
    selection = document["selection"]["rows"]
    selection = list(range(1, len(original) + 1)) if selection is None else selection
    rows, provenance = [], []
    for number in selection:
        row, entry = _row_body(document, number, original[number - 1])
        row["row_id"] = _row_id(document, row, entry)
        rows.append(row)
        provenance.append(entry)
    _check_rows(document, rows, provenance)
    dataset_raw = b"".join(canonical(row) + b"\n" for row in rows)
    if len(dataset_raw) > MAX_BYTES or any(len(canonical(row)) + 1 > MAX_ROW_BYTES for row in rows):
        invalid()
    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation")}
    manifest = {"schema_version": MANIFEST_SCHEMA, "configuration": document,
                "row_count": len(rows), "dataset_bytes": len(dataset_raw), "dataset_sha256": sha(dataset_raw),
                "row_ids_sha256": sha(canonical([row["row_id"] for row in rows])), "split_counts": counts,
                "provenance": provenance}
    digest_value = domain(MANIFEST_SCHEMA, manifest)
    manifest.update(dataset_digest=digest_value, dataset_id="dataset-" + digest_value)
    manifest_raw = canonical(manifest)
    if len(manifest_raw) > MAX_DOCUMENT_BYTES:
        invalid()
    return ConversationPublicationV1(_identity(manifest), dataset_raw, manifest_raw)


def _check_rows(config, rows, provenance):
    if not 1 <= len(rows) <= MAX_ROWS or len(rows) != len(provenance):
        invalid()
    groups, contents, selected_contents, previous = {}, {}, {}, 0
    counts = {"train": 0, "validation": 0}
    for row, entry in zip(rows, provenance):
        exact(row, {"schema_version", "format", "row_id", "source_row_number", "target_message_index",
                    "group_id", "split", "messages"})
        exact(entry, {"source_row_number", "source_row_sha256", "conversation_sha256", "declared_split_value", "message_count", "target_message_index",
                      "prefix_sha256", "target_sha256", "group_id", "split"})
        if row["schema_version"] != ROW_SCHEMA or row["format"] != "messages":
            invalid()
        number = count(row["source_row_number"], maximum=config["source"]["rows"])
        if number <= previous:
            invalid()
        previous = number
        target = count(row["target_message_index"], minimum=1, maximum=127)
        count(entry["message_count"], minimum=target + 1, maximum=128)
        messages = row["messages"]
        if type(messages) is not list or len(messages) != target + 1:
            invalid()
        for index, message in enumerate(messages):
            exact(message, {"role", "content"})
            if message["role"] not in {"system", "user", "assistant"} or type(message["content"]) is not str or not message["content"]:
                invalid()
            if message["role"] == "system" and index != 0:
                invalid()
        if messages[0]["role"] == "assistant" or messages[-1]["role"] != "assistant":
            invalid()
        if config["selection"]["target_index"] is None:
            if entry["message_count"] != target + 1:
                invalid()
        elif config["selection"]["target_index"] != target:
            invalid()
        group = text(row["group_id"])
        split_name = row["split"]
        if split_name not in counts:
            invalid()
        for key in ("source_row_number", "target_message_index", "group_id", "split"):
            if entry[key] != row[key]:
                invalid()
        digest(entry["source_row_sha256"])
        digest(entry["conversation_sha256"])
        if entry["message_count"] == target + 1 and entry["conversation_sha256"] != sha(canonical(messages)):
            invalid()
        if config["grouping"]["path"] is None and group != entry["conversation_sha256"]:
            invalid()
        if config["split"]["kind"] == "heldout_groups":
            if entry["declared_split_value"] is not None:
                invalid()
            expected = "validation" if group in config["split"]["validation_groups"] else "train"
            if split_name != expected:
                invalid()
        else:
            declared_value = entry["declared_split_value"]
            if type(declared_value) is not str or config["split"]["values"].get(declared_value) != split_name:
                invalid()
        if entry["prefix_sha256"] != sha(canonical(messages[:-1])) or entry["target_sha256"] != sha(canonical(messages[-1])):
            invalid()
        body = {key: value for key, value in row.items() if key != "row_id"}
        if row["row_id"] != _row_id(config, body, entry):
            invalid()
        if group in groups and groups[group] != split_name:
            invalid()
        groups[group] = split_name
        content = entry["conversation_sha256"]
        if content in contents and contents[content] != split_name:
            invalid()
        contents[content] = split_name
        selected_digest = sha(canonical(messages))
        if selected_digest in selected_contents and selected_contents[selected_digest] != split_name:
            invalid()
        selected_contents[selected_digest] = split_name
        counts[split_name] += 1
    if not all(counts.values()):
        invalid()
    expected_numbers = config["selection"]["rows"]
    if expected_numbers is None:
        expected_numbers = list(range(1, config["source"]["rows"] + 1))
    if [row["source_row_number"] for row in rows] != expected_numbers:
        invalid()
    if config["split"]["kind"] == "heldout_groups" and not set(config["split"]["validation_groups"]) <= set(groups):
        invalid()
    return counts


def _verify_bytes(path: Path, manifest_raw: bytes, dataset_raw: bytes, *, require_content_addressed_name=True,
                 expected_artifact_version=None) -> VerifiedPreparedDatasetV1:
    manifest = parse(manifest_raw, MAX_DOCUMENT_BYTES)
    exact(manifest, {"schema_version", "configuration", "row_count", "dataset_bytes", "dataset_sha256",
                     "row_ids_sha256", "split_counts", "provenance", "dataset_digest", "dataset_id"})
    if manifest["schema_version"] != MANIFEST_SCHEMA or canonical(manifest) != manifest_raw:
        invalid()
    config = validate_config(manifest["configuration"])
    if type(dataset_raw) is not bytes or not 0 < len(dataset_raw) <= MAX_BYTES:
        invalid()
    rows = _source_rows(dataset_raw)
    if any(canonical(row) + b"\n" != line for row, line in zip(rows, dataset_raw.splitlines(keepends=True))):
        invalid()
    if type(manifest["provenance"]) is not list:
        invalid()
    counts = _check_rows(config, rows, manifest["provenance"])
    if (type(manifest["row_count"]) is not int or manifest["row_count"] != len(rows)
            or type(manifest["dataset_bytes"]) is not int or manifest["dataset_bytes"] != len(dataset_raw)
            or manifest["dataset_sha256"] != sha(dataset_raw)
            or manifest["row_ids_sha256"] != sha(canonical([row["row_id"] for row in rows]))
            or canonical(manifest["split_counts"]) != canonical(counts)):
        invalid()
    semantic = {key: value for key, value in manifest.items() if key not in {"dataset_digest", "dataset_id"}}
    expected = domain(MANIFEST_SCHEMA, semantic)
    if manifest["dataset_digest"] != expected or manifest["dataset_id"] != "dataset-" + expected:
        invalid()
    if require_content_addressed_name and path.name != manifest["dataset_id"]:
        invalid()
    return VerifiedPreparedDatasetV1(path, _identity(manifest))


def verify_bytes(path: Path, manifest_raw: bytes, dataset_raw: bytes, *, require_content_addressed_name=True):
    try:
        return _verify_bytes(path, manifest_raw, dataset_raw,
                             require_content_addressed_name=require_content_addressed_name)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        invalid()


def _private_directory(path):
    if type(path) is not type(Path()) or not path.is_absolute():
        invalid()
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        invalid()


def snapshot_conversation_dataset(path: Path, *, require_content_addressed_name=True):
    # Existing reader retains member handles and rechecks the named directory.
    if sys.platform != "linux":
        invalid()
    _private_directory(path)
    authority = storage._RetainedDatasetDirectory(path)
    retained = []
    try:
        for name, maximum in (("manifest.json", MAX_DOCUMENT_BYTES), ("dataset.jsonl", MAX_BYTES)):
            retained.append(storage._RetainedMember(path / name, maximum, authority.member_identities[name]))
        manifest_raw, dataset_raw = (member.read() for member in retained)
        verified = verify_bytes(path, manifest_raw, dataset_raw,
                                require_content_addressed_name=require_content_addressed_name)
        for member in retained:
            member.verify_final_identity()
        authority.verify_final_inventory()
        _private_directory(path)
        return verified, dataset_raw
    finally:
        for member in reversed(retained):
            if not member.close():
                invalid()


def _verify(path, *, require_content_addressed_name=True):
    return snapshot_conversation_dataset(path, require_content_addressed_name=require_content_addressed_name)[0]


def publish_conversation_dataset(output_root, prepared):
    if sys.platform != "linux" or type(prepared) is not ConversationPublicationV1:
        invalid()
    _private_directory(output_root)
    return storage._publish(output_root, prepared, verify=_verify, verify_bytes=verify_bytes)


def read_source_members(source_root: Path, config: ConversationNormalizerConfigV1):
    """One retained parent, bounded regular single-link reads, no SDK or writes."""
    if sys.platform != "linux":
        invalid()
    root = storage._OutputRootAuthority(source_root)
    payloads = []
    try:
        assert root.descriptor is not None
        document = config.to_dict()
        for member in (document["source"]["dataset"], document["source"]["metadata"]):
            if member is None:
                payloads.append(None)
                continue
            root.verify()
            name = member["member"]
            declared = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
            if not stat.S_ISREG(declared.st_mode) or declared.st_nlink != 1 or declared.st_size != member["size_bytes"]:
                invalid()
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root.descriptor)
            try:
                def observation(info):
                    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)
                before = observation(declared)
                if observation(os.fstat(descriptor)) != before:
                    invalid()
                chunks, total = [], 0
                while total <= member["size_bytes"]:
                    chunk = os.read(descriptor, min(1024 * 1024, member["size_bytes"] + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                raw = b"".join(chunks)
                if (total != member["size_bytes"] or sha(raw) != member["sha256"]
                        or observation(os.fstat(descriptor)) != before
                        or observation(os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)) != before):
                    invalid()
                payloads.append(raw)
            finally:
                os.close(descriptor)
            root.verify()
        return tuple(payloads)
    finally:
        root.close()
