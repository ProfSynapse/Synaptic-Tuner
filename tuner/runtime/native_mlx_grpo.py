"""Native GRPO inside the existing admitted child; never an operational launcher."""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path

from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_artifacts import adapter_tensors, verify_adapter_delta
from tuner.runtime.native_mlx_composition import attach_native_lora
from tuner.runtime.native_mlx_grpo_math import completion_logprobs, group_advantages, grpo_objective
from tuner.runtime.native_mlx_material import canonical, strict_json, verify_native_material
from tuner.training.grpo_compilation import grpo_prepared_semantic_digest


def prepared_prompts(raw, projection):
    """Admit the existing publication; only its selected prompt can be rendered."""
    source = projection.source
    if (type(raw) is not bytes or len(raw) != source.size_bytes or len(raw) > 65536
            or hashlib.sha256(raw).hexdigest() != source.content_digest
            or raw.startswith(b"\xef\xbb\xbf")):
        raise ValueError("native GRPO prepared bytes differ")
    lines = raw.splitlines()
    if not 1 <= len(lines) <= 128:
        raise ValueError("native GRPO row count exceeded")
    column = projection.workload.document["training_input"]["hyperparameters"]["prompt_column"]
    prompts = []
    for line in lines:
        row = strict_json(line, maximum=65536)
        canonical(row)  # Reject exponent-overflow/nonfinite metadata as well.
        if type(row) is not dict or column not in row or (column != "prompt" and "prompt" in row):
            raise ValueError("native GRPO prompt column differs")
        prompt = row[column]
        if type(prompt) is str and prompt:
            prompt = [{"role": "user", "content": prompt}]
        if (type(prompt) is not list or not prompt
                or any(type(m) is not dict or set(m) != {"role", "content"}
                       or m["role"] not in ("system", "user", "assistant")
                       or type(m["content"]) is not str or not m["content"] for m in prompt)
                or prompt[-1]["role"] != "user"):
            raise ValueError("native GRPO prompt must end at a user generation boundary")
        prompts.append(prompt)
    semantic = grpo_prepared_semantic_digest(content_digest=source.content_digest,
        size_bytes=source.size_bytes, row_count=len(prompts), prompt_column=column)
    if source.revision != semantic or source.ref != "prepared://sha256/" + semantic:
        raise ValueError("native GRPO publication identity differs")
    return tuple(prompts)


def response_reward(text, rewards, *, timeout_ms):
    # Same raw-response binary regex contract as public GRPO. No row metadata,
    # ground truth, Python import, network or environment is visible here.
    if type(text) is not str or len(text.encode("utf-8")) > 1024 * 1024:
        raise ValueError("native GRPO response byte budget exceeded")
    # Already present in the separate 54-package native closure. A per-search
    # deadline also protects framework-free host verification of generated text.
    import regex

    result = 0.0
    for item in rewards["items"]:
        document = rewards["documents"][item["name"]]
        passed = all(regex.search(rule["match"], text, flags=regex.VERSION0,
            timeout=timeout_ms / 1000) is not None for rule in document["validation"])
        result += item["weight"] * document["scoring"]["on_pass" if passed else "on_fail"]
    if not math.isfinite(result):
        raise ValueError("native GRPO reward is nonfinite")
    return result


@contextmanager
def base_reference(model, *, lora_type):
    modules = [module for _, module in model.named_modules() if type(module) is lora_type]
    if not modules:
        raise ValueError("native GRPO base reference has no verified LoRA modules")
    scales = [module.scale for module in modules]
    try:
        for module in modules:
            module.scale = 0.0
        yield
    finally:
        for module, scale in zip(modules, scales):
            module.scale = scale


def generate_completion(model, prompt, eos, maximum, *, mx, make_cache):
    cache, current, generated = make_cache(model), mx.array(prompt, dtype=mx.int32), []
    for _ in range(maximum):
        logits = model(current[None], cache=cache)[:, -1, :].astype(mx.float32)
        token = mx.random.categorical(logits)
        mx.eval(token)
        value = int(token.item())
        generated.append(value)
        if value in eos:
            break
        current = token
    return tuple(generated)


def execute_native_grpo(*, projection, dataset, model_snapshot, tokenizer_snapshot,
                        runtime_material, dependency_lock, output):
    """All inputs must already be host admitted; verify bytes before ML imports."""
    profile, controls = projection.profile, projection.controls
    verify_native_material(runtime_material, expected_digest=profile.runtime_digest,
                           dependency_lock=dependency_lock)
    prompts = prepared_prompts(dataset, projection)
    with (NativeTree(Path(model_snapshot.root), private=True) as model_tree,
          NativeTree(Path(tokenizer_snapshot.root), private=True) as token_tree,
          NativeTree(Path(output), private=True) as destination):
        model_snapshot.verify(model_tree)
        tokenizer_snapshot.verify(token_tree)
        if destination.members():
            raise ValueError("native GRPO output must be fresh")
        config = strict_json(model_tree.read("config.json", maximum=1024 * 1024))
        quantization = config.get("quantization", config.get("quantization_config", {}))
        if profile.quantization_bits == 4 and quantization.get("bits") != 4:
            raise ValueError("native GRPO quantization differs")
        if profile.quantization_bits != 4:
            raise ValueError("native GRPO initial candidate requires a pinned 4-bit snapshot")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot.root, local_files_only=True,
            trust_remote_code=False, use_fast=projection.tokenization.use_fast)
        kwargs = projection.tokenization.to_dict()["chat_template_kwargs"] or {}
        encoded = tuple(tuple(tokenizer.apply_chat_template(prompt, tokenize=True,
            add_generation_prompt=True, **kwargs)) for prompt in prompts)
        if any(not row or len(row) > controls.max_prompt_length
               or any(type(t) is not int or t < 0 for t in row) for row in encoded):
            raise ValueError("native GRPO prompt token budget exceeded")
        eos = tokenizer.eos_token_id
        if type(eos) is not int or eos < 0:
            raise ValueError("native GRPO requires an explicit tokenizer EOS")
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optimizers
        from mlx.utils import tree_flatten, tree_map
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.tuner.lora import LoRALinear
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.utils import load_model

        mx.random.seed(controls.seed)
        model, _ = load_model(Path(model_snapshot.root), lazy=False, strict=True, trust_remote_code=False)
        parameters, shapes = attach_native_lora(model, projection.lora,
            convert=linear_to_lora_layers, flatten=tree_flatten)
        model.eval()  # Dropout is rejected; rollout and gradient modes agree.
        mx.eval(parameters)
        initial_path, final_path = Path(output) / "initial-adapters.safetensors", Path(output) / "adapters.safetensors"
        mx.save_safetensors(str(initial_path), parameters)
        initial = destination.read(initial_path.name)
        adapter_tensors(initial)
        optimizer = optimizers.AdamW(**projection.optimizer.arguments())
        training = projection.workload.document["training_input"]
        evidence, metrics = [], []
        per_step_prompts = controls.batch_size // controls.num_generations
        for step in range(controls.iterations):
            indices = [((step * per_step_prompts + i) % len(encoded)) for i in range(per_step_prompts)]
            selected = [encoded[index] for index in indices for _ in range(controls.num_generations)]
            completions = [generate_completion(model, prompt, {eos}, controls.max_completion_length,
                mx=mx, make_cache=make_prompt_cache) for prompt in selected]
            texts = [tokenizer.decode(list(tokens)) for tokens in completions]
            rewards = [response_reward(text, training["rewards"], timeout_ms=profile.reward_timeout_ms) for text in texts]
            old, mask = completion_logprobs(model, selected, completions, mx=mx)
            old = mx.stop_gradient(old)
            mx.eval(old, mask)
            with base_reference(model, lora_type=LoRALinear):
                reference, _ = completion_logprobs(model, selected, completions, mx=mx)
                reference = mx.stop_gradient(reference)
                mx.eval(reference)
            advantages = group_advantages(mx.array(rewards, dtype=mx.float32), controls.num_generations, mx=mx)
            mx.eval(advantages)

            def loss_fn(current_model):
                policy, current_mask = completion_logprobs(current_model, selected, completions, mx=mx)
                return grpo_objective(policy, old, reference, advantages, current_mask, controls.beta, mx=mx)

            loss, gradients = nn.value_and_grad(model, loss_fn)(model)
            norm = mx.sqrt(sum(mx.sum(mx.square(g.astype(mx.float32))) for _, g in tree_flatten(gradients)))
            mx.eval(loss, norm)
            if not math.isfinite(float(loss.item())) or not math.isfinite(float(norm.item())):
                raise ValueError("native GRPO gradient/loss is nonfinite")
            scale = min(1.0, controls.max_grad_norm / max(float(norm.item()), 1e-12))
            optimizer.update(model, tree_map(lambda g: g * scale, gradients))
            mx.eval(model.parameters(), optimizer.state)
            record = {"step": step + 1, "prompt_indices": indices,
                "completions": [list(tokens) for tokens in completions], "texts": texts, "rewards": rewards,
                "old": old.tolist(), "reference": reference.tolist()}
            evidence.append(record)
            if len(canonical(evidence)) > profile.maximum_evidence_bytes:
                raise ValueError("native GRPO evidence byte budget exceeded")
            if (step + 1) % controls.logging_steps == 0 or step + 1 == controls.iterations:
                metrics.append({"step": step + 1, "loss": float(loss.item()), "gradient_norm": float(norm.item())})
        completed = optimizer.step.item()
        if type(completed) is not int or completed != controls.iterations:
            raise ValueError("native GRPO optimizer count differs")
        mx.save_safetensors(str(final_path), dict(tree_flatten(model.trainable_parameters())))
        final = destination.read(final_path.name)
        delta = verify_adapter_delta(initial, final, expected_shapes=shapes)
        # Fresh-base reload verifies saved bytes rather than in-memory success.
        reloaded, _ = load_model(Path(model_snapshot.root), lazy=False, strict=True, trust_remote_code=False)
        _, reload_shapes = attach_native_lora(reloaded, projection.lora,
            convert=linear_to_lora_layers, flatten=tree_flatten)
        if reload_shapes != shapes:
            raise ValueError("native GRPO reload layout differs")
        saved = mx.load(str(final_path))
        reloaded.load_weights(list(saved.items()), strict=False)
        reloaded.eval()
        actual = dict(tree_flatten(reloaded.trainable_parameters()))
        if set(actual) != set(saved) or any(actual[k].shape != saved[k].shape
                or actual[k].dtype != saved[k].dtype or not bool(mx.all(actual[k] == saved[k]).item()) for k in saved):
            raise ValueError("native GRPO final adapter reload differs")
        final_loss = float(loss_fn(reloaded).item())
        if not math.isfinite(final_loss) or destination.read(final_path.name) != final:
            raise ValueError("native GRPO final replay differs")
        binding = {"schema": "synaptic-native-grpo-input-binding/v1", "source": projection.source.to_dict(),
            "tokenizer_snapshot_digest": tokenizer_snapshot.digest,
            "tokenization": projection.tokenization.to_dict(), "prompt_tokens": encoded,
            "workload_digest": projection.workload.fingerprint, "runtime_digest": profile.runtime_digest}
        result = {"schema": "synaptic-native-mlx-grpo-result/v1", "status": "CANDIDATE_WORKER_COMPLETE",
            "input_binding_digest": hashlib.sha256(canonical(binding)).hexdigest(),
            "workload_digest": projection.workload.fingerprint, "runtime_digest": profile.runtime_digest,
            "model_snapshot_digest": model_snapshot.digest, "completed_updates": completed,
            "adapter_format": "mlx-lm-lora-safetensors", "parameter_delta": delta,
            "final_validation": {"phase": "reloaded_saved_final_adapter_rollout_replay",
                "adapter_sha256": hashlib.sha256(final).hexdigest(), "loss": final_loss},
            "metrics": metrics, "evidence_sha256": hashlib.sha256(canonical(evidence)).hexdigest()}
        for name, value in (("input_binding.json", binding), ("native_result.json", result),
                            ("rollout_evidence.json", evidence), ("adapter_config.json", {
                                "fine_tune_type": "lora", "num_layers": projection.lora.layers,
                                "lora_parameters": {"rank": projection.lora.rank, "scale": projection.lora.scale,
                                    "dropout": projection.lora.dropout, "keys": list(projection.lora.targets)}})):
            destination.write_exclusive(name, canonical(value))
        destination.write_exclusive("workload.json", projection.workload.canonical_bytes)
        model_snapshot.verify(model_tree)
        tokenizer_snapshot.verify(token_tree)
        destination.check()
        return canonical(result)
