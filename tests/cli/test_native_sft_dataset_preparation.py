"""Real config-first selection/publication: no providers, tokenizer, or ML imports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


PATH = Path(__file__).resolve().parents[2] / "examples/training_qualification/native_sft_datasets/prepare.py"
spec = importlib.util.spec_from_file_location("sft_subset_example", PATH)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def fixture(tmp_path, *, heldout=False):
    rows = [{"approved": True, "topic": str(i), "dialog": [
        {"role": "user", "content": "question " + str(i)},
        {"role": "assistant", "content": "response " + str(i)},
    ]} for i in range(12)]
    rows += [{**copy.deepcopy(rows[0]), "approved": "true"},
             {"dialog": rows[0]["dialog"]}, copy.deepcopy(rows[0])]
    raw = b"".join(prep.canonical(row) + b"\n" for row in rows)
    (tmp_path / "corpus.jsonl").write_bytes(raw)
    source = {"id": "generic", "path": "corpus.jsonl", "url": None, "sha256": prep.sha(raw),
              "size_bytes": len(raw), "rows": len(rows), "format": "jsonl", "partition": "grouped",
              "quotas": {"train": 3, "validation": 2}, "required_label": {"field": "approved", "equals": True},
              "group_paths": [["topic"]]}
    return {"schema": "synaptic-sft-subset-preparation/v1", "sources": [source],
            "provenance": {"origin": "test"}, "projection": {"messages_path": ["dialog"],
            "normalization": "text_only", "invalid_rows": "exclude_and_report"},
            "selection": {"kind": "group_hash_rank", "seed": "proof"}}


def test_exact_label_filter_dedup_quota_and_determinism(tmp_path):
    config = fixture(tmp_path)
    first, audit = prep.select(config, tmp_path, None)
    second, _ = prep.select(copy.deepcopy(config), tmp_path, None)
    assert first == second
    assert len(first) == 5 and len({r["group"] for r in first}) == 5
    assert sum(r["split"] == "train" for r in first) == 3
    assert all(r["source_row"] not in {13, 14} for r in first)
    assert audit["generic"]["eligible_rows"] == 13
    assert audit["groups"]["eligible"] == 12


def test_declared_heldout_alias_cannot_leak_into_training(tmp_path):
    config = fixture(tmp_path)
    original = config["sources"][0]
    raw = (tmp_path / "corpus.jsonl").read_bytes()
    # Separate declared test source with an identical conversation and a changed
    # nonsemantic source attribute. All aliases must become validation-owned.
    row = json.loads(raw.splitlines()[0])
    row["aux"] = "test source"
    heldout = prep.canonical(row) + b"\n"
    (tmp_path / "test.jsonl").write_bytes(heldout)
    config["sources"].append({**original, "id": "heldout", "path": "test.jsonl",
        "sha256": prep.sha(heldout), "size_bytes": len(heldout), "rows": 1,
        "partition": "validation", "quotas": {"train": 0, "validation": 1}})
    config["sources"][0] = {**original, "partition": "train", "quotas": {"train": 3, "validation": 0}}
    selected, _ = prep.select(config, tmp_path, None)
    assert len(selected) == 4
    assert all(r["messages"] != row["dialog"] for r in selected if r["split"] == "train")


def test_real_publication_identity_and_tamper_refusal(tmp_path):
    config = fixture(tmp_path)
    result = prep.prepare(config, tmp_path, None, tmp_path / "publication")
    from tuner.training.conversation_preparation import ConversationPreparedInputFormatVerifierV1
    verifier = ConversationPreparedInputFormatVerifierV1()
    observed = verifier.verify(Path(result["publication"]))
    assert observed.content_digest == result["identity"]["content_digest"]
    assert json.loads((tmp_path / "publication/coverage.json").read_bytes())["split_counts"] == {"train": 3, "validation": 2}
    (tmp_path / "corpus.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="commitment"):
        prep.prepare(config, tmp_path, None, tmp_path / "another")
    assert not (tmp_path / "another").exists()
    with pytest.raises(FileExistsError):
        prep.write(tmp_path / "publication", "identity.json", b"replace")


def test_missing_final_response_is_explicitly_audited(tmp_path):
    config = fixture(tmp_path)
    raw = (tmp_path / "corpus.jsonl").read_bytes()
    extra = {"approved": True, "topic": "missing", "dialog": [{"role": "user", "content": "unanswered"}]}
    raw += prep.canonical(extra) + b"\n"
    (tmp_path / "corpus.jsonl").write_bytes(raw)
    config["sources"][0].update(sha256=prep.sha(raw), size_bytes=len(raw), rows=16)
    _, audit = prep.select(config, tmp_path, None)
    assert audit["generic"]["excluded_shape_rows"] == [16]
    config["projection"]["invalid_rows"] = "reject"
    with pytest.raises(ValueError, match="final assistant"):
        prep.select(config, tmp_path, None)


def test_https_destination_is_checked_before_transfer():
    for url in ("http://huggingface.co/file", "https://evil.test/file", "https://secret@huggingface.co/file"):
        with pytest.raises(ValueError, match="HTTPS"):
            prep.check_url(url)


def test_token_profile_preserves_native_ids_masks_and_padded_batches(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    config["sources"][0]["quotas"] = {"train": 4, "validation": 2}
    result = prep.prepare(config, tmp_path, None, tmp_path / "publication")
    from tuner.runtime.native_files import NativeFile
    from tuner.runtime import native_files

    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    tokenizer_raw = b"test tokenizer bytes"
    (tokenizer_root / "tokenizer.json").write_bytes(tokenizer_raw)
    snapshot = {"repository": "example/tokenizer", "revision": "a" * 40, "root": str(tokenizer_root),
                "files": [{"path": "tokenizer.json", "size": len(tokenizer_raw), "sha256": prep.sha(tokenizer_raw)}]}
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_bytes(prep.canonical(snapshot))

    class Tree:
        # Filesystem custody has independent NativeTree/host tests. This double
        # permits pytest's shared /tmp ancestor while exercising real verifier,
        # shared preprocessing, native admission, and native batching here.
        def __init__(self, path, **kwargs):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, name, *, maximum):
            raw = (self.path / name).read_bytes()
            assert len(raw) <= maximum
            return raw

        def check(self):
            pass

        def inventory(self):
            return tuple(NativeFile(p.name, p.stat().st_size, prep.sha(p.read_bytes()))
                         for p in sorted(self.path.iterdir()))

    class Tokenizer:
        eos_token_id = 126

        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(tokenizer_root)
            assert kwargs == {"local_files_only": True, "trust_remote_code": False, "use_fast": True}
            return Tokenizer()

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False and kwargs == {"enable_thinking": False}
            return "".join(m["role"] + ":" + m["content"] + "~" for m in messages) + (
                "assistant:" if add_generation_prompt else "")

        def encode(self, text, add_special_tokens=False):
            return list(text.encode("ascii"))

        def decode(self, ids, **kwargs):
            return bytes(ids).decode("ascii")

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(native_files, "NativeTree", Tree)
    profile = {"schema": "synaptic-sft-token-profile/v1", "controls": {
        "batch_size": 2, "max_seq_length": 512, "max_batch_tokens": 1024, "iterations": 2,
        "seed": 17, "steps_per_report": 1, "steps_per_eval": 2, "steps_per_save": 2},
        "chat_template_kwargs": {"enable_thinking": False}, "representative_rows": [0],
        "maximum_output_bytes": 1 << 20}
    observed = prep.profile(profile, Path(result["publication"]), snapshot_path, tmp_path / "profile")
    assert observed["failures"] == 0
    assert observed["splits"]["train"]["rows"] == 4
    assert observed["splits"]["train"]["batches"] == 2
    assert observed["splits"]["validation"]["rows"] == 2
    tokens = [json.loads(line) for line in (tmp_path / "profile/tokens.jsonl").read_bytes().splitlines()]
    assert len({r["row_id"] for r in tokens}) == 6
    for row in tokens:
        boundary = row["masked_prefix_length"]
        assert row["labels"][:boundary] == [-100] * boundary
        assert row["labels"][boundary:] == row["input_ids"][boundary:]
        assert row["input_ids"][-1] == 126
        assert row["generation_prefix"].endswith("assistant:")
    assert "mlx" not in sys.modules and "mlx_lm" not in sys.modules
    profile["maximum_output_bytes"] = 1
    with pytest.raises(ValueError, match="aggregate bound"):
        prep.profile(profile, Path(result["publication"]), snapshot_path, tmp_path / "over-budget")
    assert not (tmp_path / "over-budget").exists()
