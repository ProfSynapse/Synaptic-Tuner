"""Bounded native artifact projection for the existing public artifact roles.

No ML imports, file extraction, network, database, or capability promotion.
The provider authenticates the command and terminated child before calling this
verifier. Numeric model-change proof is recomputed from the actual saved files.
"""

from dataclasses import asdict, dataclass
import hashlib
import io
import math
from pathlib import Path
import tarfile

from synaptic_tuner.api.v1.results import VerifiedArtifact
from tuner.runtime.native_files import NativeTree, relative_member
from tuner.runtime.native_mlx_artifacts import adapter_tensors, finite_adapter_tensors, verify_adapter_delta
from tuner.runtime.native_mlx_input import NativeModelSnapshot
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.training.native_mlx_compilation import parse_native_workload


@dataclass(frozen=True)
class NativeArtifactBudget:
    per_file: int
    aggregate: int

    def __post_init__(self):
        if (type(self.per_file) is not int or type(self.aggregate) is not int
                or not 1 <= self.per_file <= self.aggregate <= 512 * 1024 * 1024):
            raise ValueError("native artifact byte budget is invalid")


@dataclass(frozen=True)
class NativePublishedArtifact:
    descriptor: VerifiedArtifact
    content: bytes


def _archive(members, budget):
    """Deterministic POSIX tar with bounded, regular, relative members only."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw in sorted(members.items()):
            relative_member(name)
            # Account for headers, padding and tarfile's final record padding
            # before allocating the next member.
            if (len(raw) > budget.per_file
                    or output.tell() + len(raw) + 512 + 10240 > budget.aggregate):
                raise ValueError("native artifact archive exceeds budget")
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(raw), 0o600, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def verify_native_publication(
    *, output: Path, workload: bytes, model: NativeModelSnapshot,
    tokenizer: NativeModelSnapshot, tokenizer_members: tuple[str, ...],
    budget: NativeArtifactBudget,
) -> tuple[NativePublishedArtifact, ...]:
    """Verify exact native output and return the five portable role payloads.

    Final-model payload is an MLX adapter tar, not merged weights or PEFT. Base
    repository/revision and snapshot identity travel in its manifest. Tokenizer
    member selection is explicit host config, checked against its whole pinned
    snapshot. Final-evaluation numbers remain trusted admitted-worker evidence;
    this CPU verifier does not rerun model inference.
    """
    projection = parse_native_workload(workload)
    if (type(model) is not NativeModelSnapshot or type(tokenizer) is not NativeModelSnapshot
            or type(budget) is not NativeArtifactBudget
            or type(tokenizer_members) is not tuple or not tokenizer_members
            or tokenizer_members != tuple(sorted(set(tokenizer_members)))):
        raise ValueError("native artifact projection inputs differ")
    profile = projection.profile
    if (model.repository, model.revision, tokenizer.repository, tokenizer.revision) != (
        profile.model_ref, profile.model_revision, profile.model_ref, profile.tokenizer_revision,
    ):
        raise ValueError("native artifact snapshot pins differ")
    with NativeTree(output, private=True) as tree, NativeTree(Path(tokenizer.root), private=True) as token_tree:
        tokenizer.verify(token_tree)
        # Checkpoint spelling is the pinned upstream MLX-LM trainer contract.
        checkpoints = tuple(f"{step:07d}_adapters.safetensors" for step in range(
            projection.controls.steps_per_save, projection.controls.iterations + 1,
            projection.controls.steps_per_save,
        ))
        expected = tuple(sorted((
            "adapter_config.json", "adapters.safetensors", "initial-adapters.safetensors",
            "input_binding.json", "native_result.json", "workload.json", *checkpoints,
        )))
        if tree.members() != expected:
            raise ValueError("native artifact inventory differs")
        files, total = {}, 0
        for name in expected:
            raw = tree.read(name, maximum=budget.per_file)
            total += len(raw)
            if total > budget.aggregate:
                raise ValueError("native artifact aggregate exceeds budget")
            files[name] = raw
        if files["workload.json"] != workload:
            raise ValueError("native artifact workload differs")
        result = strict_json(files["native_result.json"])
        binding = strict_json(files["input_binding.json"])
        if (canonical(result) != files["native_result.json"]
                or canonical(binding) != files["input_binding.json"]
                or set(binding) != {
                    "schema", "prepared_digest", "source", "tokenizer_snapshot_digest",
                    "tokenization", "workload_digest", "runtime_digest",
                }
                or binding["schema"] != "synaptic-native-tokenization-binding/v1"
                or binding["source"] != projection.source.to_dict()
                or binding["tokenizer_snapshot_digest"] != tokenizer.digest
                or binding["tokenization"] != projection.tokenization.to_dict()
                or binding["workload_digest"] != projection.workload.fingerprint
                or binding["runtime_digest"] != profile.runtime_digest):
            raise ValueError("native artifact input binding differs")
        prepared_digest = binding["prepared_digest"]
        if (type(prepared_digest) is not str or len(prepared_digest) != 64
                or any(c not in "0123456789abcdef" for c in prepared_digest)):
            raise ValueError("native prepared token digest is invalid")
        if (set(result) != {
                "schema", "status", "input_binding_digest", "workload_digest",
                "runtime_digest", "model_snapshot_digest", "lora", "optimizer",
                "completed_updates", "adapter_format", "parameter_delta", "final_validation",
            } or result["schema"] != "synaptic-native-mlx-sft-result/v1"
                or result["status"] != "CANDIDATE_WORKER_COMPLETE"
                or result["input_binding_digest"] != hashlib.sha256(files["input_binding.json"]).hexdigest()
                or result["workload_digest"] != projection.workload.fingerprint
                or result["runtime_digest"] != profile.runtime_digest
                or result["model_snapshot_digest"] != model.digest
                or canonical(result["lora"]) != canonical(asdict(projection.lora))
                or canonical(result["optimizer"]) != canonical(asdict(projection.optimizer))
                or type(result["completed_updates"]) is not int
                or result["completed_updates"] != projection.controls.iterations
                or result["adapter_format"] != "mlx-lm-lora-safetensors"):
            raise ValueError("native artifact result binding differs")
        config = {
            "fine_tune_type": "lora", "num_layers": projection.lora.layers,
            "lora_parameters": {
                "rank": projection.lora.rank, "scale": projection.lora.scale,
                "dropout": projection.lora.dropout, "keys": list(projection.lora.targets),
            },
        }
        if files["adapter_config.json"] != canonical(config):
            raise ValueError("native adapter configuration differs")
        initial, final = files["initial-adapters.safetensors"], files["adapters.safetensors"]
        tensors = adapter_tensors(initial)
        shapes = tuple((tensor.name, tensor.shape) for tensor in tensors)
        # Actual target realization was checked inside the admitted worker.
        # Check pair coverage and explicit layer/target count independently.
        names = {tensor.name for tensor in tensors}
        stems = {name.rsplit(".", 1)[0] for name in names}
        if (len(stems) != projection.lora.layers * len(projection.lora.targets)
                or any({stem + ".lora_a", stem + ".lora_b"} - names for stem in stems)
                or any(projection.lora.rank not in shape for _, shape in shapes)
                or any(not any(stem.endswith("." + target) for target in projection.lora.targets)
                       for stem in stems)):
            raise ValueError("native saved adapter target layout differs")
        delta = verify_adapter_delta(initial, final, expected_shapes=shapes)
        if result["parameter_delta"] != delta:
            raise ValueError("native parameter delta evidence differs")
        for name in checkpoints:
            checkpoint = finite_adapter_tensors(files[name])
            if tuple((tensor.name, tensor.shape, tensor.dtype) for tensor in checkpoint) != tuple(
                (tensor.name, tensor.shape, tensor.dtype) for tensor in tensors
            ):
                raise ValueError("native checkpoint tensor inventory differs")
        validation = result["final_validation"]
        if (type(validation) is not dict or set(validation) != {
                "adapter_sha256", "phase", "loss", "rows", "supervised_tokens",
            } or validation["adapter_sha256"] != hashlib.sha256(final).hexdigest()
                or validation["phase"] != "reloaded_saved_final_adapter"
                or type(validation["loss"]) not in (int, float)
                or not math.isfinite(validation["loss"]) or validation["loss"] < 0
                or any(type(validation[key]) is not int or validation[key] < 1
                       for key in ("rows", "supervised_tokens"))):
            raise ValueError("native final-adapter validation differs")
        model_members = {
            name: files[name] for name in ("adapters.safetensors", "adapter_config.json", *checkpoints)
        }
        model_members["native_model.json"] = canonical({
            "schema": "synaptic-native-mlx-model/v1", "format": "mlx-lm-lora-safetensors",
            "repository": model.repository, "revision": model.revision,
            "snapshot_digest": model.digest, "workload_digest": projection.workload.fingerprint,
            "runtime_digest": profile.runtime_digest,
        })
        tokenizer_files = {}
        available = {item.path for item in tokenizer.files}
        for name in tokenizer_members:
            if name not in available or name == "native_tokenizer.json":
                raise ValueError("native tokenizer publication member differs")
            raw = token_tree.read(name, maximum=budget.per_file)
            total += len(raw)
            if total > budget.aggregate:
                raise ValueError("native tokenizer aggregate exceeds budget")
            tokenizer_files[name] = raw
        tokenizer_files["native_tokenizer.json"] = canonical({
            "schema": "synaptic-native-tokenizer/v1", "repository": tokenizer.repository,
            "revision": tokenizer.revision, "snapshot_digest": tokenizer.digest,
            "tokenization": projection.tokenization.to_dict(),
        })
        roles = {
            "final_model": _archive(model_members, budget),
            "tokenizer": _archive(tokenizer_files, budget),
            "workload_record": workload,
            "training_metrics": files["native_result.json"],
            "training_lineage": canonical({
                "schema": "synaptic-native-training-lineage/v1",
                "source_digest": profile.source_digest, "input_binding": binding,
                "model_snapshot_digest": model.digest,
            }),
        }
        if (sum(len(raw) for raw in roles.values()) > budget.aggregate
                or any(len(raw) > budget.per_file for raw in roles.values())):
            raise ValueError("native public role byte budget exceeded")
        token_tree.check()
        tree.check()
        return tuple(NativePublishedArtifact(
            VerifiedArtifact(role, hashlib.sha256(raw).hexdigest(), len(raw)), raw,
        ) for role, raw in sorted(roles.items()))
