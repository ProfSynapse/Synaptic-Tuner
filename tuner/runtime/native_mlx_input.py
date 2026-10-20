"""Native child join from retained prepared publication to shared tokenization.

No dataset normalization, re-split, mask repair or network access. Publication
lineage/format validation remains the existing host preparation service's job.
This join verifies that exact retained bytes are what the tokenizer consumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re

from shared.sft_preprocessing import materialize_sft_example
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.native_files import NativeFile, NativeTree
from tuner.runtime.native_mlx_material import canonical, strict_json, verify_native_material
from tuner.runtime.native_mlx_sft import MLXSFTConfig, PreparedMLXSFT, prepare_mlx_sft
from tuner.training.contracts import RetainedTrainingInputStreamLease


@dataclass(frozen=True)
class NativeModelSnapshot:
    """Reviewed host material, not self-issued download or model provenance."""

    repository: str
    revision: str
    root: str
    files: tuple[NativeFile, ...]

    def __post_init__(self):
        if (type(self.repository) is not str or not re.fullmatch(
                r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository)
                or type(self.revision) is not str or not re.fullmatch(r"[a-f0-9]{40}", self.revision)):
            raise ValueError("native model source pins are invalid")
        if type(self.root) is not str or not Path(self.root).is_absolute():
            raise ValueError("native model root must be absolute")
        if (type(self.files) is not tuple or not self.files
                or any(type(member) is not NativeFile for member in self.files)
                or tuple(f.path for f in self.files) != tuple(sorted({f.path for f in self.files}))):
            raise ValueError("native model inventory is invalid")

    @property
    def digest(self):
        return hashlib.sha256(canonical(asdict(self))).hexdigest()

    def verify(self, tree: NativeTree):
        if str(tree.path) != self.root or tree.inventory() != self.files:
            raise ValueError("native model snapshot bytes differ")


@dataclass(frozen=True)
class NativeTokenization:
    assistant_only_loss: bool
    prompt_render: str
    chat_template_json: bytes | None
    use_fast: bool

    def __post_init__(self):
        if type(self.assistant_only_loss) is not bool or type(self.use_fast) is not bool:
            raise ValueError("native tokenizer switches must be explicit booleans")
        if self.prompt_render not in ("full_conversation", "prompt_completion"):
            raise ValueError("native prompt render is unsupported")
        if self.chat_template_json is not None:
            parsed = strict_json(self.chat_template_json, maximum=65536)
            if type(parsed) is not dict or canonical(parsed) != self.chat_template_json:
                raise ValueError("native template arguments must be canonical JSON")

    def to_dict(self):
        return {
            "assistant_only_loss": self.assistant_only_loss,
            "prompt_render": self.prompt_render,
            "chat_template_kwargs": (
                None if self.chat_template_json is None else strict_json(self.chat_template_json)
            ),
            "use_fast": self.use_fast,
        }


@dataclass(frozen=True)
class BoundNativeInput:
    prepared: PreparedMLXSFT
    tokenizer_snapshot_digest: str
    tokenization: NativeTokenization
    workload_digest: str
    runtime_digest: str

    @property
    def digest(self):
        return hashlib.sha256(canonical({
            "schema": "synaptic-native-tokenization-binding/v1",
            "prepared_digest": self.prepared.digest,
            "source": self.prepared.source.to_dict(),
            "tokenizer_snapshot_digest": self.tokenizer_snapshot_digest,
            "tokenization": self.tokenization.to_dict(),
            "workload_digest": self.workload_digest,
            "runtime_digest": self.runtime_digest,
        })).hexdigest()


def read_retained_native_source(source, identity: PreparedTrainingInputIdentity) -> bytes:
    if (type(identity) is not PreparedTrainingInputIdentity or source.identity != identity
            or not 1 <= identity.size_bytes <= 64 * 1024 * 1024):
        raise ValueError("native prepared source identity differs")
    lease = source.open_lease()
    if type(lease) is not RetainedTrainingInputStreamLease:
        raise TypeError("native source must return the existing retained stream lease")
    try:
        if lease.identity != identity:
            raise ValueError("native prepared lease identity differs")
        stream = lease.take_stream()
        chunks, size = [], 0
        while size <= identity.size_bytes:
            chunk = stream.read(min(1024 * 1024, identity.size_bytes + 1 - size))
            if type(chunk) is not bytes:
                raise TypeError("native source must stream bytes")
            if not chunk:
                break
            size += len(chunk)
            chunks.append(chunk)
        raw = b"".join(chunks)
        if len(raw) != identity.size_bytes or hashlib.sha256(raw).hexdigest() != identity.content_digest:
            raise ValueError("native prepared source bytes differ")
        return raw
    finally:
        lease.close()


def tokenize_native_source(
    *, source, identity: PreparedTrainingInputIdentity, tokenizer,
    tokenizer_snapshot: NativeModelSnapshot, tokenization: NativeTokenization,
    config: MLXSFTConfig, workload_digest: str, runtime_digest: str,
) -> BoundNativeInput:
    """Internal algebra boundary. The child wrapper owns tokenizer admission."""
    for digest in (workload_digest, runtime_digest):
        if type(digest) is not str or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("native input workload/runtime binding is invalid")
    if type(tokenizer_snapshot) is not NativeModelSnapshot or type(tokenization) is not NativeTokenization:
        raise TypeError("native input requires exact tokenizer controls")
    raw = read_retained_native_source(source, identity)
    splits = {"train": [], "validation": []}
    lines = raw.splitlines()
    if not 2 <= len(lines) <= 100000:
        raise ValueError("native prepared row count is invalid")
    for line in lines:
        record = strict_json(line, maximum=1024 * 1024)
        if type(record) is not dict or record.get("schema_version") != identity.format:
            raise ValueError("native prepared format differs from publication")
        # These are the established publication fields, not an MLX text schema.
        if record.get("split") not in splits:
            raise ValueError("native prepared row has an unsupported declared split")
        row = materialize_sft_example(
            tokenizer=tokenizer, record=record, max_seq_length=config.max_seq_length,
            assistant_only_loss=tokenization.assistant_only_loss,
            source_hash=hashlib.sha256(line).hexdigest(),
            chat_template_kwargs=tokenization.to_dict()["chat_template_kwargs"],
            prompt_render=tokenization.prompt_render,
        )
        splits[record["split"]].append((record.get("row_id"), row))
    prepared = prepare_mlx_sft(
        source=identity, tokenizer_revision=tokenizer_snapshot.revision,
        train=tuple(splits["train"]), validation=tuple(splits["validation"]), config=config,
    )
    return BoundNativeInput(prepared, tokenizer_snapshot.digest, tokenization,
                            workload_digest, runtime_digest)


def prepare_native_input(
    *, source, identity, tokenizer_snapshot, tokenization, config,
    workload_digest: str, runtime_material: bytes, runtime_digest: str, dependency_lock: bytes,
) -> BoundNativeInput:
    """Run only in the isolated admitted child, before model/Metal imports."""
    verify_native_material(runtime_material, expected_digest=runtime_digest,
                           dependency_lock=dependency_lock)
    with NativeTree(Path(tokenizer_snapshot.root), private=True) as tree:
        tokenizer_snapshot.verify(tree)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_snapshot.root, local_files_only=True, trust_remote_code=False,
            use_fast=tokenization.use_fast,
        )
        result = tokenize_native_source(
            source=source, identity=identity, tokenizer=tokenizer,
            tokenizer_snapshot=tokenizer_snapshot, tokenization=tokenization, config=config,
            workload_digest=workload_digest, runtime_digest=runtime_digest,
        )
        # Hold the root identity across preparation, then recheck bytes.
        tokenizer_snapshot.verify(tree)
        return result
