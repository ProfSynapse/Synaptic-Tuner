"""Config-first, bounded host dataset selection; never loads a model or starts training.

Uses the existing conversation normalizer/publisher. Corpus names, columns,
labels, split quotas and provenance live in JSON, not the training runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from shared.sft_preprocessing import sanitize_messages_for_chat_template
from tuner.dataset_prep.conversation_messages import (
    ConversationNormalizerConfigV1, build_conversation_publication, canonical,
    publish_conversation_dataset, sha, verify_bytes,
)
from tuner.runtime.packaged_worker_closure import stable_read

MAX_SOURCE = 16 << 20
MAX_TOTAL = 64 << 20


def private(path):
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("owner-private directory required")


def write(root, name, raw):
    private(root)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=descriptor)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if stable_read(root / name, MAX_TOTAL) != raw or root.stat().st_ino != before.st_ino:
            raise ValueError("written bytes differ")
    finally:
        os.close(descriptor)


def load_config(path):
    raw = stable_read(path, 1 << 20)
    config = json.loads(raw)
    if set(config) != {"schema", "provenance", "sources", "projection", "selection"}:
        raise ValueError("configuration fields differ")
    if config["schema"] != "synaptic-sft-subset-preparation/v1":
        raise ValueError("configuration schema differs")
    projection = config["projection"]
    if (set(projection) != {"messages_path", "normalization", "invalid_rows"}
            or projection["normalization"] not in {"text_only", "existing_sft_tool_text"}
            or projection["invalid_rows"] not in {"reject", "exclude_and_report"}
            or type(projection["messages_path"]) is not list
            or not 1 <= len(projection["messages_path"]) <= 16
            or any(type(key) is not str or not key for key in projection["messages_path"])):
        raise ValueError("projection policy differs")
    if not 1 <= len(config["sources"]) <= 32:
        raise ValueError("source count exceeds bound")
    for source in config["sources"]:
        if set(source) != {"id", "path", "url", "sha256", "size_bytes", "rows", "format",
                           "partition", "quotas", "required_label", "group_paths"}:
            raise ValueError("source fields differ")
        if (not re.fullmatch(r"[a-zA-Z0-9_-]+", source["id"])
                or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])
                or type(source["size_bytes"]) is not int
                or not 0 < source["size_bytes"] <= MAX_SOURCE
                or type(source["rows"]) is not int or not 1 <= source["rows"] <= 10000
                or source["format"] not in {"jsonl", "parquet"}
                or set(source["quotas"]) != {"train", "validation"}
                or any(type(n) is not int or n < 0 for n in source["quotas"].values())
                or source["partition"] not in {"train", "validation", "grouped"}):
            raise ValueError("source bounds differ")
        if source["partition"] != "grouped":
            other = "train" if source["partition"] == "validation" else "validation"
            if source["quotas"][other]:
                raise ValueError("declared source split must be preserved")
        member = Path(source["path"])
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("source path must be relative")
    if len({s["id"] for s in config["sources"]}) != len(config["sources"]):
        raise ValueError("source IDs must be unique")
    if sum(s["size_bytes"] for s in config["sources"]) > MAX_TOTAL:
        raise ValueError("aggregate source bound exceeded")
    if config["selection"]["kind"] != "group_hash_rank" or set(config["selection"]) != {"kind", "seed"}:
        raise ValueError("selection policy differs")
    return config


class Redirects(HTTPRedirectHandler):
    def __init__(self):
        self.count = 0

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        self.count += 1
        if self.count > 3:
            raise ValueError("redirect bound exceeded")
        check_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def check_url(url):
    value = urlsplit(url)
    if (value.scheme != "https" or value.hostname not in {
            "huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co",
            "cas-bridge.xethub.hf.co", "us.aws.cdn.hf.co"}
            or value.port not in {None, 443} or value.username or value.password or value.fragment):
        raise ValueError("unapproved HTTPS source")


def acquire(config, output):
    """One create-only acquisition; failures and partial members remain for review."""
    output.mkdir(mode=0o700)
    private(output)
    for source in config["sources"]:
        if source["url"] is None:
            continue
        check_url(source["url"])
        # Corpus configs pin immutable Hub revision URLs; never send credentials.
        opener = build_opener(ProxyHandler({}), Redirects())
        request = Request(source["url"], headers={"Accept-Encoding": "identity"})
        deadline = time.monotonic() + 60
        with opener.open(request, timeout=20) as response:
            check_url(response.geturl())
            if response.status != 200 or response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("HTTP response differs")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > source["size_bytes"]:
                raise ValueError("source response exceeds bound")
            chunks, count = [], 0
            while True:
                if time.monotonic() >= deadline:
                    raise ValueError("source deadline exceeded")
                chunk = response.read(min(65536, source["size_bytes"] + 1 - count))
                if not chunk:
                    break
                count += len(chunk)
                if count > source["size_bytes"]:
                    raise ValueError("source byte bound exceeded")
                chunks.append(chunk)
        raw = b"".join(chunks)
        if len(raw) != source["size_bytes"] or sha(raw) != source["sha256"]:
            raise ValueError("source commitment differs")
        write(output, source["id"] + ".source", raw)


def at(value, path):
    for key in path:
        value = value[key]
    return value


def normalized(value):
    return " ".join(value.casefold().split())


def select(config, source_root, acquired):
    """Union related groups before sampling; validation source groups win."""
    candidates, audit, reserved_keys = [], {}, set()
    projection = config["projection"]
    for source in config["sources"]:
        member = (acquired / (source["id"] + ".source") if source["url"]
                  else source_root / source["path"])
        raw = stable_read(member, MAX_SOURCE)
        if len(raw) != source["size_bytes"] or sha(raw) != source["sha256"]:
            raise ValueError("source commitment differs")
        if source["format"] == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pq.read_table(pa.BufferReader(raw))
            if table.num_rows != source["rows"] or table.nbytes > MAX_TOTAL:
                raise ValueError("decoded parquet exceeds bounds")
            rows = table.to_pylist()
        else:
            rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != source["rows"]:
            raise ValueError("source row count differs")
        labels = Counter()
        excluded_shape = []
        for number, row in enumerate(rows, 1):
            if source["partition"] == "validation":
                reserved_keys.update("declared:" + sha(canonical(at(row, path)))
                                     for path in source["group_paths"])
            label = source["required_label"]
            if label is not None:
                value = row.get(label["field"])
                labels["missing" if label["field"] not in row else str(value)] += 1
                if type(value) is not type(label["equals"]) or value != label["equals"]:
                    continue
            messages = deepcopy(at(row, projection["messages_path"]))
            if projection["normalization"] == "existing_sft_tool_text":
                messages = sanitize_messages_for_chat_template(messages)
            elif projection["normalization"] != "text_only":
                raise ValueError("projection normalization differs")
            if (not messages or messages[-1]["role"] != "assistant"
                    or any(m["role"] not in {"system", "user", "assistant"}
                           or type(m["content"]) is not str or not m["content"].strip() for m in messages)):
                if projection.get("invalid_rows") != "exclude_and_report":
                    raise ValueError("selected conversation is not text with final assistant")
                excluded_shape.append(number)
                continue
            messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            # Global exact prompt/target/conversation equivalence joins aliases,
            # including aliases in different source files or declared splits.
            keys = ["conversation:" + sha(canonical(messages)),
                    "users:" + sha(canonical([normalized(m["content"]) for m in messages if m["role"] == "user"])),
                    "target:" + sha(canonical(normalized(messages[-1]["content"])))]
            keys += ["declared:" + sha(canonical(at(row, path))) for path in source["group_paths"]]
            candidates.append({"source": source["id"], "source_row": number, "source_row_sha256": sha(canonical(row)),
                               "messages": messages, "keys": keys, "partition": source["partition"]})
        audit[source["id"]] = {"raw_rows": len(rows), "labels": dict(labels),
                                 "excluded_shape_rows": excluded_shape,
                                 "eligible_rows": sum(c["source"] == source["id"] for c in candidates)}
    parent = list(range(len(candidates)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, candidate in enumerate(candidates):
        for key in candidate["keys"]:
            if key in owners:
                parent[root(i)] = root(owners[key])
            owners[key] = i
    # A selected target reused as earlier context is the same target family.
    # Join it before assigning splits; common greetings only join a group when
    # they are themselves candidate targets.
    targets = {normalized(c["messages"][-1]["content"]): i for i, c in enumerate(candidates)}
    for i, candidate in enumerate(candidates):
        for message in candidate["messages"][:-1]:
            other = targets.get(normalized(message["content"]))
            if other is not None:
                parent[root(i)] = root(other)
    groups = {}
    for i, candidate in enumerate(candidates):
        groups.setdefault(root(i), []).append(candidate)
    group_ids = {r: sha(canonical(sorted({k for c in cs for k in c["keys"]}))) for r, cs in groups.items()}
    heldout = {r for r, cs in groups.items() if any(c["partition"] == "validation"
        or reserved_keys.intersection(c["keys"]) for c in cs)}
    used, selected = set(), []
    seed = config["selection"]["seed"]
    for split in ("validation", "train"):
        for source in config["sources"]:
            count = 0
            for r in sorted(groups, key=lambda r: sha(canonical([seed, split, source["id"], group_ids[r]]))):
                if count == source["quotas"][split]:
                    break
                if r in used or (split == "train" and r in heldout):
                    continue
                eligible = [c for c in groups[r] if c["source"] == source["id"]
                            and c["partition"] in {"grouped", split}]
                if not eligible:
                    continue
                item = min(eligible, key=lambda c: c["source_row"])
                selected.append({**{k: v for k, v in item.items() if k not in {"keys", "partition"}},
                                 "group": group_ids[r], "split": split})
                used.add(r)
                count += 1
            if count != source["quotas"][split]:
                raise ValueError("insufficient independent source groups for exact quota")
    selected.sort(key=lambda c: (c["source"], c["source_row"]))
    validation_targets = [normalized(c["messages"][-1]["content"])
                          for c in selected if c["split"] == "validation"]
    for candidate in selected:
        if candidate["split"] == "train" and any(
                target in normalized(message["content"])
                for target in validation_targets for message in candidate["messages"]):
            raise ValueError("held-out target prose appears in selected training conversation")
    audit["groups"] = {"eligible": len(groups), "reserved_source_validation": len(heldout),
                       "selected": len(used), "selection_one_row_per_group": True}
    return selected, audit


def prepare(config, source_root, acquired, output):
    selected, audit = select(config, source_root, acquired)
    output.mkdir(mode=0o700)
    private(output)
    raw = b"".join(canonical(row) + b"\n" for row in selected)
    document = {
        "schema_version": "synaptic-conversation-preparation/v1",
        "source": {"dataset": {"member": "selected.jsonl", "sha256": sha(raw), "size_bytes": len(raw)},
                   "metadata": None, "rows": len(selected), "data_outcome": None,
                   "provenance": {"subset_configuration": config}},
        "projection": {"messages_path": ["messages"], "role_field": "role", "content_field": "content",
                       "role_map": {r: r for r in ("system", "user", "assistant")}},
        "selection": {"rows": None, "target_index": None}, "grouping": {"path": ["group"]},
        "split": {"kind": "field", "path": ["split"], "values": {"train": "train", "validation": "validation"}},
    }
    write(output, "selected.jsonl", raw)
    write(output, "normalizer.json", canonical(document))
    prepared = build_conversation_publication(ConversationNormalizerConfigV1.from_dict(document), raw)
    verified = publish_conversation_dataset(output, prepared)
    semantic = verified.semantic_identity
    identity = {"ref": "prepared://sha256/" + semantic.dataset_digest, "revision": semantic.dataset_digest,
                "content_digest": semantic.dataset_sha256, "size_bytes": semantic.dataset_bytes,
                "format": "synaptic-conversation-sft-row/v1"}
    write(output, "identity.json", canonical(identity))
    write(output, "coverage.json", canonical({"sources": audit, "split_counts": dict(semantic.split_counts),
        "selected": [{k: v for k, v in row.items() if k != "messages"} for row in selected]}))
    return {"identity": identity, "publication": str(verified.path), "split_counts": dict(semantic.split_counts)}


def profile(config, publication, tokenizer_snapshot, output):
    """Tokenizer-only shared preprocessing with full per-token evidence, no MLX."""
    from tuner.runtime.native_files import NativeFile, NativeTree
    from tuner.runtime.native_mlx_input import NativeModelSnapshot
    from tuner.runtime.native_mlx_sft import MLXSFTConfig, prepare_mlx_sft, iter_prepared_batches
    from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
    from shared.sft_preprocessing import materialize_sft_example
    from transformers import AutoTokenizer

    if (set(config) != {"schema", "controls", "chat_template_kwargs", "representative_rows", "maximum_output_bytes"}
            or config["schema"] != "synaptic-sft-token-profile/v1"
            or type(config["maximum_output_bytes"]) is not int
            or not 1 <= config["maximum_output_bytes"] <= MAX_TOTAL):
        raise ValueError("token profile configuration differs")
    controls = MLXSFTConfig.from_dict(config["controls"])
    value = json.loads(stable_read(tokenizer_snapshot, 1 << 20))
    snapshot = NativeModelSnapshot(value["repository"], value["revision"], value["root"],
                                   tuple(NativeFile(**member) for member in value["files"]))
    with NativeTree(publication, private=True) as tree:
        if tuple(member.path for member in tree.inventory()) != ("dataset.jsonl", "manifest.json"):
            raise ValueError("prepared publication inventory differs")
        raw = tree.read("dataset.jsonl", maximum=2 << 20)
        manifest = tree.read("manifest.json", maximum=2 << 20)
        verified = verify_bytes(publication, manifest, raw)
        tree.check()
    rows = [json.loads(line) for line in raw.splitlines()]
    records, examples, splits = [], [], {"train": [], "validation": []}
    with NativeTree(Path(snapshot.root), private=True) as tree:
        snapshot.verify(tree)
        tokenizer = AutoTokenizer.from_pretrained(snapshot.root, local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
        for index, row in enumerate(rows):
            prepared = materialize_sft_example(tokenizer=tokenizer, record=row,
                max_seq_length=controls.max_seq_length, assistant_only_loss=True,
                chat_template_kwargs=config["chat_template_kwargs"], prompt_render="prompt_completion")
            if (prepared.truncation_applied or prepared.drop_reason
                    or not any(label != -100 for label in prepared.labels)):
                raise ValueError("tokenized row cannot be trained intact")
            records.append({"index": index, "row_id": row["row_id"], "split": row["split"],
                "length": len(prepared.input_ids), "trained_tokens": sum(label != -100 for label in prepared.labels),
                "masked_prefix_length": prepared.masked_prefix_length,
                "input_ids": prepared.input_ids, "labels": prepared.labels,
                "generation_prefix": tokenizer.apply_chat_template(
                    row["messages"][:-1], tokenize=False, add_generation_prompt=True,
                    **config["chat_template_kwargs"])})
            splits[row["split"]].append((row["row_id"], prepared))
            if index in config["representative_rows"]:
                examples.append({"row": row, "tokens": records[-1]})
        snapshot.verify(tree)
        tree.check()
    semantic = verified.semantic_identity
    identity = PreparedTrainingInputIdentity("prepared://sha256/" + semantic.dataset_digest,
        semantic.dataset_digest, semantic.dataset_sha256, semantic.dataset_bytes, "synaptic-conversation-sft-row/v1")
    native = prepare_mlx_sft(source=identity, tokenizer_revision=snapshot.revision,
        train=tuple(splits["train"]), validation=tuple(splits["validation"]), config=controls)
    token_bytes = b"".join(canonical(record) + b"\n" for record in records)
    summary = {"schema": "synaptic-sft-token-profile-result/v1", "configuration": config,
        "dataset_digest": verified.semantic_identity.dataset_digest,
        "tokenizer_snapshot": value, "token_evidence_sha256": sha(token_bytes), "splits": {}}
    for split in ("train", "validation"):
        selected = [record for record in records if record["split"] == split]
        batches = list(iter_prepared_batches(native, split=split, loop=False))
        summary["splits"][split] = {"rows": len(selected), "total_tokens": sum(r["length"] for r in selected),
            "supervised_tokens": sum(r["trained_tokens"] for r in selected),
            "maximum_length": max(r["length"] for r in selected), "batches": len(batches),
            "padded_batch_tokens": sum(len(b.input_ids) * len(b.input_ids[0]) for b in batches),
            "maximum_padded_batch_tokens": max(len(b.input_ids) * len(b.input_ids[0]) for b in batches),
            "batch_row_ids": [list(b.row_ids) for b in batches]}
    if summary["splits"]["train"]["batches"] != controls.iterations:
        raise ValueError("profile controls do not consume exactly one complete training pass")
    summary["failures"] = 0
    payloads = {"tokens.jsonl": token_bytes, "complete-examples.json": canonical(examples),
                "token-profile.json": canonical(summary)}
    if sum(map(len, payloads.values())) > config["maximum_output_bytes"]:
        raise ValueError("token evidence exceeds explicit aggregate bound")
    output.mkdir(mode=0o700)
    for name, raw in payloads.items():
        write(output, name, raw)
    return {"dataset_digest": summary["dataset_digest"], "splits": {
        split: {k: v for k, v in values.items() if k != "batch_row_ids"}
        for split, values in summary["splits"].items()}, "failures": 0,
        "token_evidence_sha256": sha(token_bytes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "prepare", "profile"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--acquired-root", type=Path)
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory under an existing private parent")
    args = parser.parse_args()
    private(args.output.absolute().parent)
    if args.action == "profile":
        result = profile(json.loads(stable_read(args.config.absolute(), 1 << 20)),
                         args.publication, args.tokenizer_snapshot, args.output.absolute())
        print(json.dumps(result, sort_keys=True))
        return
    config = load_config(args.config.absolute())
    if args.action == "acquire":
        acquire(config, args.output.absolute())
        result = {"status": "acquired_not_training_authority"}
    else:
        result = prepare(config, args.source_root.absolute(), args.acquired_root, args.output.absolute())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"status": "failed_preserve_attempt", "error_code": "dataset_preparation_rejected"}))
        raise SystemExit(2) from None
