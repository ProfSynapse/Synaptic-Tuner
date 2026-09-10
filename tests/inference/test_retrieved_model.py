from __future__ import annotations
import hashlib, io, json, os, tarfile
from dataclasses import replace
from pathlib import Path
import pytest
from synaptic_tuner.api.v1.results import (
    TrainingRunRef,
    TrainingRunState,
    VerifiedArtifact,
)
from synaptic_tuner.api.v1.runs_facade import RunOutcome, RunVerification, RunsAPI
from tuner.inference import retrieved_model
from tuner.inference.retrieved_model import (
    RetrievedSFTModel,
    materialize_verified_sft_model,
)


def _safe(payload=b"\1\0\0\0"):
    h = json.dumps(
        {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    return len(h).to_bytes(8, "little") + h + payload


def _tar(items):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as t:
        for name, data in items:
            i = tarfile.TarInfo(name)
            i.size = len(data)
            t.addfile(i, io.BytesIO(data))
    return out.getvalue()


def _fixture(tmp_path):
    run = TrainingRunRef("run-1", "project-1")
    model = {
        "ref": "example/model",
        "revision": "c" * 40,
        "tokenizer_revision": "c" * 40,
    }
    workload = json.dumps(
        {
            "configuration": {"document": {"model": model}},
            "identities": {"model": model},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    values = {
        "final_model": _tar(
            (
                (
                    "adapter_config.json",
                    b'{"base_model_name_or_path":"example/model","peft_type":"LORA"}',
                ),
                ("adapter_model.safetensors", _safe()),
            )
        ),
        "tokenizer": _tar(
            (
                ("tokenizer_config.json", b'{"tokenizer_class":"Fixture"}'),
                (
                    "tokenizer.json",
                    b'{"model":{"type":"BPE","vocab":{"x":0}},"version":"1"}',
                ),
            )
        ),
        "training_lineage": b"{}",
        "training_metrics": b"{}",
        "workload_record": workload,
    }
    artifacts = tuple(
        VerifiedArtifact(k, hashlib.sha256(v).hexdigest(), len(v))
        for k, v in sorted(values.items())
    )

    class Stream:
        def __init__(self, x, owner):
            self.run = run
            self.artifact = x
            self.maximum_bytes = x.size_bytes
            self.owner = owner

        def iter_bytes(self):
            yield self.owner.values[self.artifact.role]

    class Ops:
        fail = None

        def __init__(self):
            self.values = values
            self.inventory = artifacts

        def reverify(self, r):
            return RunVerification(r, True, "2026-09-09T00:00:00Z")

        def outcome(self, r):
            return RunOutcome(
                "synaptic-run-outcome/v1", r, TrainingRunState.SUCCEEDED, self.inventory
            )

        def artifacts(self, q):
            x = next(x for x in self.inventory if x.role == q.role)
            if self.fail == q.role:

                class Bad(Stream):
                    def iter_bytes(self):
                        yield self.owner.values[x.role][:-1]

                return Bad(x, self)
            return Stream(x, self)

    return run, values, Ops()


def test_materializes_and_revalidates_exact_files(tmp_path: Path):
    run, _, ops = _fixture(tmp_path)
    result = materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    assert result.model_path.is_dir() and result.tokenizer_path.is_dir()
    assert (
        result.model_ref,
        result.model_revision,
        result.tokenizer_revision,
        result.model_kind,
    ) == ("example/model", "c" * 40, "c" * 40, "lora")
    (result.model_path / "adapter_config.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file (?:identity invalid|changed)"):
        result.validate()


def test_receipt_cannot_be_constructed():
    with pytest.raises(TypeError):
        RetrievedSFTModel()


def test_truncation_cleans_only_owned_attempt(tmp_path: Path):
    keep = tmp_path / "keep"
    keep.write_text("safe")
    run, _, ops = _fixture(tmp_path)
    ops.fail = "tokenizer"
    with pytest.raises(ValueError, match="integrity"):
        materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    assert keep.read_text() == "safe"
    assert not list(tmp_path.glob(".retrieved-*"))


@pytest.mark.parametrize("kind", ("role", "run", "digest"))
def test_substituted_authenticated_surface_fails_before_success(tmp_path: Path, kind):
    run, _, ops = _fixture(tmp_path)
    if kind == "role":
        old = ops.outcome
        ops.outcome = lambda r: replace(old(r), artifacts=old(r).artifacts[:-1])
    elif kind == "run":
        ops.reverify = lambda r: RunVerification(
            TrainingRunRef("other", r.project_ref), True, "2026-09-09T00:00:00Z"
        )
    else:
        old = ops.artifacts

        def substituted(q):
            stream = old(q)
            stream.artifact = replace(stream.artifact, sha256="0" * 64)
            return stream

        ops.artifacts = substituted
    with pytest.raises((ValueError, TypeError)):
        materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)


@pytest.mark.parametrize("typeflag", (b"x", b"g", b"L", b"S", b"2", b"5"))
def test_raw_tar_scan_rejects_extensions_sparse_links_and_directories(typeflag):
    value = bytearray(_tar((("entry", b"x"),)))
    value[156:157] = typeflag
    value[148:156] = b"        "
    checksum = sum(value[:512])
    value[148:156] = f"{checksum:06o}\0 ".encode()
    with pytest.raises(ValueError, match="extension"):
        retrieved_model._scan(io.BytesIO(value))


@pytest.mark.parametrize("attack", ("checksum", "base256", "trailing"))
def test_raw_tar_scan_rejects_noncanonical_framing(attack):
    value = bytearray(_tar((("entry", b"x"),)))
    if attack == "checksum":
        value[0] = ord("X")
    elif attack == "base256":
        value[124] |= 128
    else:
        value[-1] = 1
    with pytest.raises(ValueError):
        retrieved_model._scan(io.BytesIO(value))


def test_raw_tar_scan_rejects_nonzero_name_suffix_after_nul():
    value = bytearray(_tar((("entry", b"x"),)))
    value[6] = ord("X")
    value[148:156] = b"        "
    checksum = sum(value[:512])
    value[148:156] = f"{checksum:06o}\0 ".encode()
    with pytest.raises(ValueError, match="suffix"):
        retrieved_model._scan(io.BytesIO(value))


@pytest.mark.parametrize("path", (Path("relative"), Path("/tmp/../tmp")))
def test_private_root_must_be_absolute_and_lexically_canonical(path):
    with pytest.raises(ValueError, match="lexically canonical"):
        retrieved_model._open_root(path)


def test_invalid_private_root_fails_before_runs_api(tmp_path: Path):
    _, _, ops = _fixture(tmp_path)
    calls = []
    ops.reverify = lambda run: calls.append("reverify")
    with pytest.raises(OSError):
        materialize_verified_sft_model(
            RunsAPI(ops), TrainingRunRef("run-1", "project-1"), tmp_path / "missing"
        )
    assert calls == []


def test_symlink_private_root_fails_before_runs_api(tmp_path: Path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    _, _, ops = _fixture(actual)
    calls = []
    ops.reverify = lambda run: calls.append("reverify")
    with pytest.raises(OSError):
        materialize_verified_sft_model(
            RunsAPI(ops), TrainingRunRef("run-1", "project-1"), alias
        )
    assert calls == []


def test_attempt_collision_is_not_adopted_or_removed(tmp_path: Path, monkeypatch):
    occupied = tmp_path / (".retrieved-" + "a" * 32)
    occupied.mkdir()
    (occupied / "owner").write_text("other")
    monkeypatch.setattr(retrieved_model.secrets, "token_hex", lambda n: "a" * 32)
    run, _, ops = _fixture(tmp_path)
    with pytest.raises(FileExistsError):
        materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    assert (occupied / "owner").read_text() == "other"


def test_factory_receipt_metadata_substitution_is_detected(tmp_path: Path):
    run, _, ops = _fixture(tmp_path)
    result = materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    object.__setattr__(result, "model_revision", "0" * 40)
    with pytest.raises(ValueError, match="target changed"):
        result.validate()


@pytest.mark.parametrize("field", ("revision", "tokenizer_revision"))
def test_malformed_workload_revision_is_rejected_and_cleaned(tmp_path: Path, field):
    run, _, ops = _fixture(tmp_path)
    document = json.loads(ops.values["workload_record"])
    document["configuration"]["document"]["model"][field] = "not-a-revision"
    document["identities"]["model"][field] = "not-a-revision"
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ops.values["workload_record"] = raw
    ops.inventory = tuple(
        (
            VerifiedArtifact(x.role, hashlib.sha256(raw).hexdigest(), len(raw))
            if x.role == "workload_record"
            else x
        )
        for x in ops.inventory
    )
    with pytest.raises(ValueError, match="target invalid"):
        materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    assert not list(tmp_path.glob(".retrieved-*"))


def test_mismatched_model_and_tokenizer_revisions_are_rejected(tmp_path: Path):
    run, _, operations = _fixture(tmp_path)
    document = json.loads(operations.values["workload_record"])
    document["configuration"]["document"]["model"]["tokenizer_revision"] = "d" * 40
    document["identities"]["model"]["tokenizer_revision"] = "d" * 40
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    operations.values["workload_record"] = raw
    operations.inventory = tuple(
        (
            VerifiedArtifact(item.role, hashlib.sha256(raw).hexdigest(), len(raw))
            if item.role == "workload_record"
            else item
        )
        for item in operations.inventory
    )
    with pytest.raises(ValueError, match="revisions must match"):
        materialize_verified_sft_model(RunsAPI(operations), run, tmp_path)


def test_revalidation_rejects_fifo_leaf_without_blocking(tmp_path: Path):
    run, _, ops = _fixture(tmp_path)
    result = materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    target = result.model_path / "adapter_config.json"
    target.unlink()
    os.mkfifo(target)
    with pytest.raises(ValueError):
        result.validate()


def test_partial_local_writes_are_completed(tmp_path: Path, monkeypatch):
    real = retrieved_model.os.write
    monkeypatch.setattr(
        retrieved_model.os,
        "write",
        lambda fd, data: real(fd, data[: max(1, len(data) // 2)]),
    )
    run, _, ops = _fixture(tmp_path)
    assert materialize_verified_sft_model(
        RunsAPI(ops), run, tmp_path
    ).model_path.is_dir()


def test_failed_local_write_cleans_owned_attempt(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(retrieved_model.os, "write", lambda fd, data: 0)
    run, _, ops = _fixture(tmp_path)
    with pytest.raises(OSError, match="short artifact write"):
        materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    assert not list(tmp_path.glob(".retrieved-*"))


def test_archive_aba_restore_between_validation_and_extraction_is_rejected(
    tmp_path: Path, monkeypatch
):
    real = retrieved_model._validate_sft_archive_stream
    changed = False

    def validate(stream, *args, **kwargs):
        nonlocal changed
        result = real(stream, *args, **kwargs)
        if args[0] == "model" and not changed:
            changed = True
            path = next(tmp_path.glob(".retrieved-*/final_model.tar"))
            original = path.read_bytes()
            altered = bytes([original[0] ^ 1]) + original[1:]
            path.write_bytes(altered)
            path.write_bytes(original)
        return result

    monkeypatch.setattr(retrieved_model, "_validate_sft_archive_stream", validate)
    run, _, ops = _fixture(tmp_path)
    with pytest.raises(ValueError, match="archive changed"):
        materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    assert not list(tmp_path.glob(".retrieved-*"))


def test_cleanup_preserves_injected_future_role_leaf(tmp_path: Path, monkeypatch):
    real_stream = retrieved_model._stream

    def inject_before_stream(runs, run, artifact, attempt, owned_files):
        if artifact.role == "tokenizer":
            path = next(tmp_path.glob(".retrieved-*")) / "tokenizer.tar"
            path.write_bytes(b"unowned future role")
        return real_stream(runs, run, artifact, attempt, owned_files)

    monkeypatch.setattr(retrieved_model, "_stream", inject_before_stream)
    run, _, operations = _fixture(tmp_path)
    with pytest.raises(FileExistsError):
        materialize_verified_sft_model(RunsAPI(operations), run, tmp_path)
    injected = next(tmp_path.glob(".retrieved-*/tokenizer.tar"))
    assert injected.read_bytes() == b"unowned future role"


def test_cleanup_preserves_replaced_owned_role_leaf(tmp_path: Path, monkeypatch):
    real_stream = retrieved_model._stream

    def replace_after_stream(runs, run, artifact, attempt, owned_files):
        real_stream(runs, run, artifact, attempt, owned_files)
        if artifact.role == "final_model":
            path = next(tmp_path.glob(".retrieved-*")) / "final_model.tar"
            path.unlink()
            path.write_bytes(b"unowned replacement")
            raise RuntimeError("forced failure")

    monkeypatch.setattr(retrieved_model, "_stream", replace_after_stream)
    run, _, operations = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="forced failure"):
        materialize_verified_sft_model(RunsAPI(operations), run, tmp_path)
    replacement = next(tmp_path.glob(".retrieved-*/final_model.tar"))
    assert replacement.read_bytes() == b"unowned replacement"


def test_cleanup_preserves_unknown_extracted_child(tmp_path: Path, monkeypatch):
    real_extract = retrieved_model._extract

    def inject_after_extract(*args, **kwargs):
        result = real_extract(*args, **kwargs)
        if args[3] == "model":
            path = next(tmp_path.glob(".retrieved-*/model")) / "unknown.bin"
            path.write_bytes(b"unowned extracted child")
            raise RuntimeError("forced failure")
        return result

    monkeypatch.setattr(retrieved_model, "_extract", inject_after_extract)
    run, _, operations = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="forced failure"):
        materialize_verified_sft_model(RunsAPI(operations), run, tmp_path)
    unknown = next(tmp_path.glob(".retrieved-*/model/unknown.bin"))
    assert unknown.read_bytes() == b"unowned extracted child"


def test_cleanup_preserves_substituted_attempt_directory(tmp_path: Path, monkeypatch):
    real_write = retrieved_model.os.write
    changed = False

    def fail_after_substitution(fd, data):
        nonlocal changed
        if not changed:
            changed = True
            attempt = next(tmp_path.glob(".retrieved-*"))
            attempt.rename(tmp_path / "original-attempt")
            attempt.mkdir()
            (attempt / "replacement").write_text("unowned")
            return 0
        return real_write(fd, data)

    monkeypatch.setattr(retrieved_model.os, "write", fail_after_substitution)
    run, _, ops = _fixture(tmp_path)
    with pytest.raises(OSError):
        materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    replacement = next(tmp_path.glob(".retrieved-*/replacement"))
    assert replacement.read_text() == "unowned"


def test_traversal_mutation_after_validation_writes_nothing(
    tmp_path: Path, monkeypatch
):
    real = retrieved_model._validate_sft_archive_stream
    changed = False

    def validate(stream, *args, **kwargs):
        nonlocal changed
        result = real(stream, *args, **kwargs)
        if args[0] == "model" and not changed:
            changed = True
            path = next(tmp_path.glob(".retrieved-*/final_model.tar"))
            raw = bytearray(path.read_bytes())
            raw[:19] = b"../evil_config.json"
            raw[148:156] = b"        "
            checksum = sum(raw[:512])
            raw[148:156] = f"{checksum:06o}\0 ".encode()
            path.write_bytes(raw)
        return result

    monkeypatch.setattr(retrieved_model, "_validate_sft_archive_stream", validate)
    run, _, ops = _fixture(tmp_path)
    with pytest.raises(ValueError, match="archive (?:plan )?changed"):
        materialize_verified_sft_model(RunsAPI(ops), run, tmp_path)
    assert not (tmp_path / "evil_config.json").exists()
