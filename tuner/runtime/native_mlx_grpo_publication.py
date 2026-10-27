"""Framework-free native GRPO artifact verification, never hardware authority."""
import hashlib
import math
from pathlib import Path
import re
import struct

from synaptic_tuner.api.v1.results import VerifiedArtifact
from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_artifacts import adapter_tensors, verify_adapter_delta
from tuner.runtime.native_mlx_grpo import response_reward
from tuner.runtime.native_mlx_input import NativeModelSnapshot
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.training.native_mlx_compilation import parse_native_workload


def _integer(value, maximum, minimum=0):
    return type(value) is int and minimum <= value <= maximum


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_rollout_evidence(evidence, binding, projection):
    """Check complete groups and configured budgets, not arbitrary summaries."""
    controls, profile = projection.controls, projection.profile
    prompts = binding["prompt_tokens"]
    if (type(prompts) is not list or not 1 <= len(prompts) <= 128
            or any(type(p) is not list or not 1 <= len(p) <= controls.max_prompt_length
                   or any(not _integer(t, 2**31 - 1) for t in p) for p in prompts)
            or type(evidence) is not list or len(evidence) != controls.iterations
            or len(canonical(evidence)) > profile.maximum_evidence_bytes):
        raise ValueError("native GRPO evidence inventory differs")
    training = projection.workload.document["training_input"]
    total_tokens, reward_calls, varied_groups, total_groups = 0, 0, 0, 0
    per_step = controls.batch_size // controls.num_generations
    for step, record in enumerate(evidence):
        if (type(record) is not dict or set(record) != {
                "step", "prompt_indices", "completions", "texts", "rewards", "old", "reference"}
                or type(record["step"]) is not int or record["step"] != step + 1
                or type(record["prompt_indices"]) is not list
                or any(not _integer(i, len(prompts) - 1) for i in record["prompt_indices"])
                or record["prompt_indices"] != [((step * per_step + i) % len(prompts)) for i in range(per_step)]):
            raise ValueError("native GRPO rollout order differs")
        for key in ("completions", "texts", "rewards", "old", "reference"):
            if type(record[key]) is not list or len(record[key]) != controls.batch_size:
                raise ValueError("native GRPO rollout batch differs")
        lengths = []
        for index, tokens in enumerate(record["completions"]):
            if (type(tokens) is not list or not 1 <= len(tokens) <= controls.max_completion_length
                    or any(not _integer(t, 2**31 - 1) for t in tokens)):
                raise ValueError("native GRPO generated tokens differ")
            prompt = prompts[record["prompt_indices"][index // controls.num_generations]]
            if len(prompt) + len(tokens) > controls.max_seq_length:
                raise ValueError("native GRPO sequence budget exceeded")
            lengths.append(len(tokens))
            reward = record["rewards"][index]
            if not _finite(reward) or reward != response_reward(record["texts"][index], training["rewards"],
                    timeout_ms=profile.reward_timeout_ms):
                raise ValueError("native GRPO reward evidence differs")
        width = max(lengths)
        for key in ("old", "reference"):
            for length, row in zip(lengths, record[key]):
                if (type(row) is not list or len(row) != width
                        or any(not _finite(v) or v > 1e-5 for v in row)
                        or any(v != 0 for v in row[length:])):
                    raise ValueError("native GRPO generated logprob mask differs")
        for start in range(0, controls.batch_size, controls.num_generations):
            group = record["rewards"][start:start + controls.num_generations]
            total_groups += 1
            # The differentiated objective normalizes float32 rewards. Double
            # precision metadata differences that round away do not qualify it.
            realized = [struct.unpack("<f", struct.pack("<f", value))[0] for value in group]
            varied_groups += max(realized) > min(realized)
        total_tokens += sum(lengths)
        reward_calls += controls.batch_size * len(training["rewards"]["items"])
    if (total_tokens > profile.maximum_rollout_tokens or reward_calls > profile.maximum_reward_calls
            or varied_groups == 0):
        raise ValueError("native GRPO variation is inconclusive or budget exceeded")
    return {"groups": total_groups, "varied_groups": varied_groups,
            "generated_tokens": total_tokens, "reward_calls": reward_calls}


def verify_native_grpo_publication(*, output, workload, model, tokenizer, tokenizer_members, budget):
    from tuner.runtime.native_mlx_publication import (
        NativeArtifactBudget, NativePublishedArtifact, _archive, tokenizer_publication_bytes,
    )
    projection = parse_native_workload(workload)
    profile, controls = projection.profile, projection.controls
    if (projection.workload.method != "grpo" or type(model) is not NativeModelSnapshot
            or type(tokenizer) is not NativeModelSnapshot or type(budget) is not NativeArtifactBudget
            or (model.repository, model.revision, tokenizer.repository, tokenizer.revision) != (
                profile.model_ref, profile.model_revision, profile.model_ref, profile.tokenizer_revision)):
        raise ValueError("native GRPO snapshot/profile differs")
    with (NativeTree(Path(output), private=True) as tree,
          NativeTree(Path(model.root), private=True) as model_tree,
          NativeTree(Path(tokenizer.root), private=True) as token_tree):
        model.verify(model_tree)
        tokenizer.verify(token_tree)
        expected = tuple(sorted(("adapter_config.json", "adapters.safetensors", "initial-adapters.safetensors",
            "input_binding.json", "native_result.json", "rollout_evidence.json", "workload.json")))
        if tree.members() != expected:
            raise ValueError("native GRPO artifact inventory differs")
        files = {name: tree.read(name, maximum=budget.per_file) for name in expected}
        if sum(map(len, files.values())) > budget.aggregate or files["workload.json"] != workload:
            raise ValueError("native GRPO artifact budget/workload differs")
        binding, result, evidence = (strict_json(files[name]) for name in (
            "input_binding.json", "native_result.json", "rollout_evidence.json"))
        if any(canonical(value) != files[name] for value, name in (
            (binding, "input_binding.json"), (result, "native_result.json"), (evidence, "rollout_evidence.json"))):
            raise ValueError("native GRPO artifact canonical bytes differ")
        if (type(binding) is not dict or set(binding) != {"schema", "source", "tokenizer_snapshot_digest",
                "tokenization", "prompt_tokens", "workload_digest", "runtime_digest"}
                or binding["schema"] != "synaptic-native-grpo-input-binding/v1"
                or binding["source"] != projection.source.to_dict()
                or binding["tokenizer_snapshot_digest"] != tokenizer.digest
                or binding["tokenization"] != projection.tokenization.to_dict()
                or binding["workload_digest"] != projection.workload.fingerprint
                or binding["runtime_digest"] != profile.runtime_digest):
            raise ValueError("native GRPO input binding differs")
        variation = verify_rollout_evidence(evidence, binding, projection)
        if (type(result) is not dict or set(result) != {"schema", "status", "input_binding_digest",
                "workload_digest", "runtime_digest", "model_snapshot_digest", "completed_updates",
                "adapter_format", "parameter_delta", "final_validation", "metrics", "evidence_sha256"}
                or result["schema"] != "synaptic-native-mlx-grpo-result/v1"
                or result["status"] != "CANDIDATE_WORKER_COMPLETE"
                or result["input_binding_digest"] != hashlib.sha256(files["input_binding.json"]).hexdigest()
                or result["workload_digest"] != projection.workload.fingerprint
                or result["runtime_digest"] != profile.runtime_digest
                or result["model_snapshot_digest"] != model.digest
                or type(result["completed_updates"]) is not int or result["completed_updates"] != controls.iterations
                or result["adapter_format"] != "mlx-lm-lora-safetensors"
                or result["evidence_sha256"] != hashlib.sha256(files["rollout_evidence.json"]).hexdigest()):
            raise ValueError("native GRPO result binding differs")
        metrics = result["metrics"]
        expected_steps = sorted(set(range(controls.logging_steps, controls.iterations + 1,
            controls.logging_steps)) | {controls.iterations})
        if (type(metrics) is not list or len(metrics) != len(expected_steps)
                or any(type(m) is not dict or set(m) != {"step", "loss", "gradient_norm"}
                       or type(m["step"]) is not int or m["step"] != step
                       or not _finite(m["loss"]) or not _finite(m["gradient_norm"]) or m["gradient_norm"] < 0
                       for m, step in zip(metrics, expected_steps))):
            raise ValueError("native GRPO metrics differ")
        lora = projection.lora
        config = {"fine_tune_type": "lora", "num_layers": lora.layers, "lora_parameters": {
            "rank": lora.rank, "scale": lora.scale, "dropout": lora.dropout, "keys": list(lora.targets)}}
        if files["adapter_config.json"] != canonical(config):
            raise ValueError("native GRPO adapter config differs")
        model_config = strict_json(model_tree.read("config.json", maximum=1024 * 1024))
        layer_count = model_config.get("num_hidden_layers")
        quantization = model_config.get("quantization", model_config.get("quantization_config", {}))
        if type(quantization) is not dict or quantization.get("bits") != profile.quantization_bits:
            raise ValueError("native GRPO model quantization differs")
        if not _integer(layer_count, 4096, lora.layers):
            raise ValueError("native GRPO model layer count differs")
        tensors = adapter_tensors(files["initial-adapters.safetensors"])
        shapes = tuple((t.name, t.shape) for t in tensors)
        names = {t.name for t in tensors}
        stems = {n.rsplit(".", 1)[0] for n in names}
        parsed = [re.fullmatch(r"((?:[A-Za-z_][A-Za-z0-9_]*\.)*layers)\.(0|[1-9][0-9]*)\.(.+)", s) for s in stems]
        prefixes = {m.group(1) for m in parsed if m is not None}
        expected_stems = {f"{p}.{i}.{t}" for p in prefixes
            for i in range(layer_count - lora.layers, layer_count) for t in lora.targets}
        if (len(prefixes) != 1 or any(m is None for m in parsed) or stems != expected_stems
                or names != {s + "." + suffix for s in expected_stems for suffix in ("lora_a", "lora_b")}
                or any(lora.rank not in shape for _, shape in shapes)):
            raise ValueError("native GRPO adapter targets differ")
        delta = verify_adapter_delta(files["initial-adapters.safetensors"], files["adapters.safetensors"], expected_shapes=shapes)
        validation = result["final_validation"]
        if (result["parameter_delta"] != delta or type(validation) is not dict or set(validation) != {
                "phase", "adapter_sha256", "loss"}
                or validation["phase"] != "reloaded_saved_final_adapter_rollout_replay"
                or validation["adapter_sha256"] != delta["final_sha256"] or not _finite(validation["loss"])):
            raise ValueError("native GRPO saved-final evidence differs")
        model_files = {name: files[name] for name in ("adapter_config.json", "adapters.safetensors")}
        model_files["native_model.json"] = canonical({"schema": "synaptic-native-mlx-model/v1",
            "format": "mlx-lm-lora-safetensors", "repository": model.repository, "revision": model.revision,
            "snapshot_digest": model.digest, "workload_digest": projection.workload.fingerprint,
            "runtime_digest": profile.runtime_digest})
        roles = {"final_model": _archive(model_files, budget),
            "tokenizer": tokenizer_publication_bytes(tokenizer, tokenizer_members, projection.tokenization, budget, token_tree),
            "workload_record": workload, "training_metrics": files["native_result.json"],
            "training_lineage": canonical({"schema": "synaptic-native-grpo-training-lineage/v1",
                "source_digest": profile.source_digest, "input_binding": binding,
                "model_snapshot_digest": model.digest, "grouped_reward_evidence": variation,
                "objective": profile.objective, "upstream_commit": profile.upstream_commit,
                "rollouts": evidence})}
        if sum(map(len, roles.values())) > budget.aggregate or any(len(r) > budget.per_file for r in roles.values()):
            raise ValueError("native GRPO public role budget exceeded")
        model_tree.check()
        token_tree.check()
        tree.check()
        return tuple(NativePublishedArtifact(VerifiedArtifact(role, hashlib.sha256(raw).hexdigest(), len(raw)), raw)
                     for role, raw in sorted(roles.items()))
