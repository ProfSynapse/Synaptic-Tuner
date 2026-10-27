"""Pure projection of public SFT input into the internal native worker contract.

This adds no public verb, registration, grant, or execution-mode fallback. The
host binds this compiled workload and its reviewed profile through the existing
coordinator; hardware/provenance qualification remains a separate gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from synaptic_tuner.api.v1.training_input import TrainingInputV1, TrainingMethodV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.runtime.native_mlx_composition import NativeAdamW, NativeLoRA
from tuner.runtime.native_mlx_input import NativeTokenization
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.runtime.native_mlx_sft import MLXSFTConfig
from tuner.training.recipes import CompiledWorkload


@dataclass(frozen=True)
class NativeMLXProfile:
    profile_ref: str
    model_ref: str
    model_revision: str
    tokenizer_revision: str
    runtime_digest: str
    source_digest: str
    lora_layers: int
    lora_scale_policy: str
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    bias_correction: bool
    schedule: str
    use_fast_tokenizer: bool
    steps_per_report: int
    steps_per_eval: int
    maximum_updates: int
    maximum_batch_tokens: int

    def __post_init__(self):
        for value in (self.profile_ref, self.model_ref):
            if type(value) is not str or not 1 <= len(value) <= 256 or any(ord(c) < 32 for c in value):
                raise ValueError("native profile reference is invalid")
        for value in (self.model_revision, self.tokenizer_revision):
            if type(value) is not str or not re.fullmatch("[0-9a-f]{40}", value):
                raise ValueError("native profile model/tokenizer pin is invalid")
        for value in (self.runtime_digest, self.source_digest):
            if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
                raise ValueError("native profile material/source pin is invalid")
        if self.lora_scale_policy != "alpha_over_rank":
            raise ValueError("native LoRA scale conversion must be explicitly alpha_over_rank")
        for value in (self.lora_layers, self.steps_per_report, self.steps_per_eval,
                      self.maximum_updates, self.maximum_batch_tokens):
            if type(value) is not int or not 1 <= value <= 1000000:
                raise ValueError("native profile bound is invalid")
        if type(self.use_fast_tokenizer) is not bool:
            raise ValueError("native tokenizer implementation must be explicit")
        NativeAdamW(.001, self.beta1, self.beta2, self.epsilon, self.weight_decay,
                    self.bias_correction, self.schedule)

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("native profile fields differ")
        return cls(**value)


@dataclass(frozen=True)
class NativeSFTProjection:
    workload: CompiledWorkload
    source: PreparedTrainingInputIdentity
    controls: MLXSFTConfig
    tokenization: NativeTokenization
    lora: NativeLoRA
    optimizer: NativeAdamW
    profile: NativeMLXProfile


def compile_native_sft(
    training: TrainingInputV1, source: PreparedTrainingInputIdentity, profile: NativeMLXProfile,
) -> NativeSFTProjection:
    if (type(training) is not TrainingInputV1 or type(source) is not PreparedTrainingInputIdentity
            or type(profile) is not NativeMLXProfile or training.method is not TrainingMethodV1.SFT):
        raise ValueError("native compilation requires exact public prepared SFT inputs")
    # Detach mutable nested chat-template data through the existing public parser.
    training = TrainingInputV1.from_json(training.canonical_json())
    if training.dataset.ref != source.ref:
        raise ValueError("native public request does not bind prepared publication")
    if (training.model.ref, training.model.revision, training.model.tokenizer_revision) != (
        profile.model_ref, profile.model_revision, profile.tokenizer_revision,
    ):
        raise ValueError("native request differs from selected model profile")
    hp = training.hyperparameters
    if (hp.gradient_accumulation_steps != 1 or hp.use_dora or hp.use_rslora
            or hp.init_lora_weights is not True or hp.split_dataset
            or hp.use_preassigned_splits is not True or hp.packing is not False
            or hp.require_memory_efficient_loss is not False
            or hp.duration.max_steps is None or hp.duration.num_epochs is not None):
        raise ValueError("native SFT request contains unsupported controls")
    if hp.duration.max_steps > profile.maximum_updates:
        raise ValueError("native SFT update bound exceeded")
    # Upstream retains every periodic checkpoint. Do not silently reinterpret a
    # requested pruning policy or drop checkpoints claimed by the public request.
    checkpoints = hp.duration.max_steps // hp.save_steps
    if not training.artifacts.retain_checkpoints or checkpoints > hp.save_total_limit:
        raise ValueError("native checkpoint retention policy is unsupported")
    if not set(training.artifacts.required_kinds) <= {
        "workload_record", "training_lineage", "training_metrics", "final_model", "tokenizer",
    }:
        raise ValueError("native artifact role is unsupported")
    formats = {
        "syntunia-sft-row/v1": "raw_text", "syntunia-sft-row/v2": "messages",
        "synaptic-conversation-sft-row/v1": "messages",
    }
    if source.format not in formats or hp.dataset_format != formats[source.format]:
        raise ValueError("native prepared format differs from public request")
    if hp.dataset_format == "raw_text":
        if hp.completion_only_loss or hp.assistant_only_loss or hp.prompt_render != "full_conversation":
            raise ValueError("native raw text requires full sequence supervision")
        mask = False
    else:
        if (hp.completion_only_loss is not True or hp.assistant_only_loss is not False
                or hp.prompt_render != "prompt_completion"):
            raise ValueError("native prepared messages require final completion supervision")
        # The shared preprocessor's assistant_only_loss names this semantic
        # masking operation. Public completion_only_loss is the final-target lane.
        mask = True
    controls = MLXSFTConfig(
        hp.batch_size, hp.max_seq_length, profile.maximum_batch_tokens,
        hp.duration.max_steps, hp.seed, profile.steps_per_report, profile.steps_per_eval,
        hp.save_steps,
    )
    tokenization = NativeTokenization(
        mask, hp.prompt_render,
        None if hp.chat_template_kwargs is None else canonical(hp.chat_template_kwargs),
        profile.use_fast_tokenizer,
    )
    lora = NativeLoRA(profile.lora_layers, hp.lora_rank, hp.lora_alpha / hp.lora_rank,
                     hp.lora_dropout, hp.lora_target_modules)
    optimizer = NativeAdamW(hp.learning_rate, profile.beta1, profile.beta2, profile.epsilon,
                            profile.weight_decay, profile.bias_correction, profile.schedule)
    document = {
        "schema_version": "synaptic-native-mlx-workload/v1", "method": "sft",
        "entrypoint": "tuner.runtime.native_mlx_child:main",
        "training_input": training.to_dict(), "prepared_input": source.to_dict(),
        "profile": asdict(profile),
        "controls": asdict(controls), "tokenization": tokenization.to_dict(),
        "lora": asdict(lora), "optimizer": asdict(optimizer),
    }
    workload = CompiledWorkload("sft", document["schema_version"], document["entrypoint"], canonical(document))
    return NativeSFTProjection(workload, source, controls, tokenization, lora, optimizer, profile)


def parse_native_workload(raw: bytes):
    """Recompile an exact method-specific native document; never infer a profile."""
    document = strict_json(raw, maximum=1024 * 1024)
    if type(document) is not dict:
        raise ValueError("native workload must be an object")
    if document.get("schema_version") == "synaptic-native-mlx-grpo-workload/v1":
        from tuner.training.native_mlx_grpo_compilation import NativeMLXGRPOProfile, compile_native_grpo
        compiler, profile_type = compile_native_grpo, NativeMLXGRPOProfile
    else:
        compiler, profile_type = compile_native_sft, NativeMLXProfile
    projected = compiler(
        TrainingInputV1.from_dict(document["training_input"]),
        PreparedTrainingInputIdentity.from_dict(document["prepared_input"]),
        profile_type.from_dict(document["profile"]),
    )
    if projected.workload.canonical_bytes != raw:
        raise ValueError("native workload differs from public-input compilation")
    return projected
