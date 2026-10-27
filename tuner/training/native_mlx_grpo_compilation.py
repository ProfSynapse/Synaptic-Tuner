"""Pure public GRPO compilation for a separately admitted native runtime.

No framework imports, profile fallback, preparation effects or launch authority.
The first objective is token-mean, population-normalized, unclipped on-policy
GRPO with one update per rollout. Unsupported public controls fail closed.
"""
from dataclasses import asdict, dataclass

from synaptic_tuner.api.v1.training_input import TrainingInputV1, TrainingMethodV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.native_mlx_composition import NativeAdamW, NativeLoRA
from tuner.runtime.native_mlx_input import NativeTokenization
from tuner.runtime.native_mlx_material import canonical
from tuner.training.grpo_compilation import GRPO_PREPARED_FORMAT, normalized_grpo_configuration
from tuner.training.native_mlx_compilation import NativeMLXProfile
from tuner.training.recipes import CompiledWorkload


@dataclass(frozen=True)
class NativeMLXGRPOProfile(NativeMLXProfile):
    objective: str
    upstream_commit: str
    maximum_rollout_tokens: int
    maximum_reward_calls: int
    maximum_evidence_bytes: int
    quantization_bits: int
    reward_engine: str
    reward_timeout_ms: int

    def __post_init__(self):
        super().__post_init__()
        if (self.objective != "mlxrl-token-mean-on-policy/v1"
                or self.upstream_commit != "5ee8a6bd7764b0585094d78c530ac87d26364034"):
            raise ValueError("native GRPO objective/source is unsupported")
        for value, maximum in ((self.maximum_rollout_tokens, 1048576),
                               (self.maximum_reward_calls, 65536),
                               (self.maximum_evidence_bytes, 16 * 1024 * 1024)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("native GRPO budget is invalid")
        if type(self.quantization_bits) is not int or self.quantization_bits != 4:
            raise ValueError("native GRPO quantization selection is invalid")
        if (self.reward_engine != "regex-v0/v1" or type(self.reward_timeout_ms) is not int
                or not 1 <= self.reward_timeout_ms <= 1000):
            raise ValueError("native GRPO reward engine/budget is invalid")


@dataclass(frozen=True)
class NativeGRPOControls:
    iterations: int
    batch_size: int
    num_generations: int
    max_seq_length: int
    max_prompt_length: int
    max_completion_length: int
    seed: int
    beta: float
    max_grad_norm: float
    logging_steps: int


@dataclass(frozen=True)
class NativeGRPOProjection:
    workload: CompiledWorkload
    source: PreparedTrainingInputIdentity
    controls: NativeGRPOControls
    tokenization: NativeTokenization
    lora: NativeLoRA
    optimizer: NativeAdamW
    profile: NativeMLXGRPOProfile


def compile_native_grpo(training, source, profile):
    if (type(training) is not TrainingInputV1 or training.method is not TrainingMethodV1.GRPO
            or type(source) is not PreparedTrainingInputIdentity
            or type(profile) is not NativeMLXGRPOProfile):
        raise ValueError("native compilation requires exact public GRPO inputs")
    training = TrainingInputV1.from_json(training.canonical_json())
    # Reuse the public prepared format/reward/role contract, not a new schema.
    normalized_grpo_configuration(training, source)
    if source.format != GRPO_PREPARED_FORMAT or (
        training.model.ref, training.model.revision, training.model.tokenizer_revision
    ) != (profile.model_ref, profile.model_revision, profile.tokenizer_revision):
        raise ValueError("native GRPO model/publication differs from profile")
    hp = training.hyperparameters
    if (hp.gradient_accumulation_steps != 1 or hp.use_dora or hp.use_rslora
            or hp.lora_dropout != 0 or hp.warmup_ratio != 0 or hp.temperature != 1
            or hp.bf16 or hp.fp16 or hp.weight_decay != profile.weight_decay
            or hp.load_in_4bit != (profile.quantization_bits == 4)
            or hp.save_steps <= hp.max_steps or hp.save_total_limit != 1):
        raise ValueError("native GRPO request contains unsupported controls")
    if (hp.max_steps > profile.maximum_updates
            or hp.batch_size * hp.max_seq_length > profile.maximum_batch_tokens
            or hp.max_steps * hp.batch_size * hp.max_completion_length > profile.maximum_rollout_tokens
            or hp.max_steps * hp.batch_size * len(training.rewards["items"]) > profile.maximum_reward_calls):
        raise ValueError("native GRPO execution budget exceeded")
    searches = hp.max_steps * hp.batch_size * sum(
        len(document["validation"]) for document in training.rewards["documents"].values())
    if searches * profile.reward_timeout_ms > 10000:
        raise ValueError("native GRPO aggregate reward time budget exceeded")
    controls = NativeGRPOControls(hp.max_steps, hp.batch_size, hp.num_generations,
        hp.max_seq_length, hp.max_prompt_length, hp.max_completion_length, hp.seed,
        hp.beta, hp.max_grad_norm, hp.logging_steps)
    tokenization = NativeTokenization(False, "prompt_completion",
        None if hp.chat_template_kwargs is None else canonical(hp.chat_template_kwargs),
        profile.use_fast_tokenizer)
    lora = NativeLoRA(profile.lora_layers, hp.lora_rank, hp.lora_alpha / hp.lora_rank,
                      hp.lora_dropout, hp.lora_target_modules)
    optimizer = NativeAdamW(hp.learning_rate, profile.beta1, profile.beta2, profile.epsilon,
        hp.weight_decay, profile.bias_correction, profile.schedule)
    document = {
        "schema_version": "synaptic-native-mlx-grpo-workload/v1", "method": "grpo",
        "entrypoint": "tuner.runtime.native_mlx_child:main",
        "training_input": training.to_dict(), "prepared_input": source.to_dict(),
        "profile": asdict(profile), "controls": asdict(controls),
        "tokenization": tokenization.to_dict(), "lora": asdict(lora),
        "optimizer": asdict(optimizer),
    }
    workload = CompiledWorkload("grpo", document["schema_version"], document["entrypoint"], canonical(document))
    return NativeGRPOProjection(workload, source, controls, tokenization, lora, optimizer, profile)
