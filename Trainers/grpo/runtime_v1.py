"""Explicit retained-input GRPO core; importing this module loads no ML stack."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import re
from pathlib import Path
from typing import Any, Dict

from shared.training_utils import build_trainer_config, reject_unknown_config_keys
from tuner.training.grpo_compilation import (
    _finite_tree, _EVIDENCE_CAPS as _EVIDENCE_CAPS, validate_qualification_evidence_policy,
)

GRPO_CONFIG_SCHEMA = {
    "model": {
        "model_name": None,
        "lora_path": None,
        "max_seq_length": None,
        "dtype": None,
        "load_in_4bit": None,
        "chat_template": None,
    },
    "lora": {
        "r": None,
        "lora_alpha": None,
        "lora_dropout": None,
        "bias": None,
        "target_modules": None,
        "use_gradient_checkpointing": None,
        "random_state": None,
        "use_rslora": None,
        "use_dora": None,
    },
    "training": {
        "output_dir": None,
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
        "num_generations": None,
        "max_prompt_length": None,
        "max_completion_length": None,
        "temperature": None,
        "learning_rate": None,
        "weight_decay": None,
        "warmup_ratio": None,
        "lr_scheduler_type": None,
        "num_train_epochs": None,
        "max_steps": None,
        "max_grad_norm": None,
        "beta": None,
        "fp16": None,
        "bf16": None,
        "optim": None,
        "logging_steps": None,
        "save_steps": None,
        "save_total_limit": None,
        "report_to": None,
        "use_gspo": None,
        "chat_template_kwargs": None,
        "extra_args": None,
    },
    "dataset": {
        "dataset_name": None,
        "dataset_file": None,
        "local_file": None,
        "num_proc": None,
        "prompt_column": None,
    },
    "rewards": {
        "items": [{"name": None, "weight": None}],
        "custom": {
            "enabled": None,
            "file": None,
            "functions": [{"name": None, "weight": None}],
        },
    },
    "pivot": {
        "enabled": None,
        "profiled_file": None,
        "sft_source": None,
        "profiling": {
            "n_rollouts": None,
            "temperature": None,
            "max_completion_length": None,
            "batch_size": None,
        },
        "filtering": {
            "variance_threshold": None,
            "min_candidates": None,
            "max_candidates": None,
            "mean_reward_range": None,
        },
        "cache": {"enabled": None, "cache_dir": None},
    },
    "wandb": {"enabled": None, "project": None, "run_name": None, "entity": None},
    "seed": None,
}


def build_grpo_config(config: dict, checkpoints_dir: Path, *, config_class, bf16_supported: bool):
    training = config['training']
    wandb_cfg = config.get('wandb') or {}
    bf16 = bool(training.get('bf16') and bf16_supported)
    fp16 = bool(training.get('fp16') and not bf16)
    if not bf16 and not fp16:
        fp16 = not bf16_supported

    args: Dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "per_device_train_batch_size": int(training['per_device_train_batch_size']),
        "gradient_accumulation_steps": int(training['gradient_accumulation_steps']),
        "num_generations": int(training['num_generations']),
        "max_completion_length": int(training['max_completion_length']),
        "temperature": float(training['temperature']),
        "learning_rate": float(training['learning_rate']),
        "weight_decay": float(training['weight_decay']),
        "warmup_ratio": float(training['warmup_ratio']),
        "lr_scheduler_type": str(training['lr_scheduler_type']),
        "optim": str(training['optim']),
        "logging_steps": int(training['logging_steps']),
        "save_steps": int(training['save_steps']),
        "save_total_limit": int(training['save_total_limit']),
        "num_train_epochs": int(training['num_train_epochs']),
        "fp16": fp16,
        "bf16": bf16,
        "seed": int(config.get('seed', 42)),
        "report_to": str(training.get('report_to', 'none')),
        # KL penalty coefficient - critical for preventing divergence
        "beta": float(training.get('beta', 0.1)),
    }
    # Every argument the user's YAML (or a CLI override written into it) set,
    # mapped to the config path that set it. build_trainer_config refuses any
    # of these the installed TRL GRPOConfig does not accept.
    origins: Dict[str, str] = {
        name: f"training.{name}" for name in args if name in training
    }
    if 'seed' in config:
        origins["seed"] = "seed"

    # Optional: GRPOConfig dropped max_prompt_length in trl 0.28; set it only
    # when the YAML does (TRL's own default applies otherwise).
    if 'max_prompt_length' in training:
        args["max_prompt_length"] = int(training['max_prompt_length'])
        origins["max_prompt_length"] = "training.max_prompt_length"
    if training.get('max_grad_norm') is not None:
        args["max_grad_norm"] = float(training['max_grad_norm'])
        origins["max_grad_norm"] = "training.max_grad_norm"

    max_steps = training.get('max_steps', 0)
    if max_steps and int(max_steps) > 0:
        args["max_steps"] = int(max_steps)
        origins["max_steps"] = "training.max_steps"

    # GSPO toggle uses sequence-level importance sampling.
    if training.get('use_gspo'):
        args["importance_sampling_level"] = "sequence"
        origins["importance_sampling_level"] = "training.use_gspo"

    if args["report_to"] == "wandb" and wandb_cfg.get('run_name'):
        args["run_name"] = str(wandb_cfg['run_name'])
        origins["run_name"] = "wandb.run_name"

    # Pass-through GRPOConfig args; each must be accepted by the installed TRL.
    extra_args = training.get('extra_args') or {}
    if not isinstance(extra_args, dict):
        raise TypeError("training.extra_args must be a mapping/dict")
    args.update(extra_args)
    origins.update({name: f"training.extra_args.{name}" for name in extra_args})

    # Every internal default above is accepted by all supported TRL releases,
    # so nothing is version-dependent here: any unsupported argument raises.
    return build_trainer_config(config_class, args, origins=origins)



def train_grpo_core(*, model, tokenizer, reward_fn, formatted_dataset,
                    training_args, callbacks, trainer_class, resume=None, _runtime_milestone=None):
    if _runtime_milestone is not None:
        _runtime_milestone("TRAINER_SETUP")
    trainer = create_grpo_trainer(trainer_class=trainer_class, model=model, processing_class=tokenizer,
                            reward_funcs=reward_fn, args=training_args,
                            train_dataset=formatted_dataset, callbacks=callbacks)
    if _runtime_milestone is not None:
        _runtime_milestone("TRAIN_CALL")
    result = train_grpo_trainer(trainer, resume=resume)
    return trainer, result


def create_grpo_trainer(*, trainer_class, **kwargs):
    return trainer_class(**kwargs)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

def admit_grpo_inputs(*, config, dataset_bytes, reward_documents, reward_schema,
                      model_identity, workload):
    """Check caller-supplied compilation commitments before loading ML.

    Dataset digest binds original JSONL bytes; rows are parsed from those SAME
    bytes. Other commitments use canonical finite JSON, not an assertion that
    reserialization reconstructs a historical publication. Compiler/closure
    authentication remains the caller's responsibility.
    """
    for value in (config, reward_documents, reward_schema, model_identity, workload):
        _finite_tree(value)
        if len(_canonical(value)) > 2 * 1024 * 1024:
            raise ValueError("input record bound exceeded")
    config = json.loads(_canonical(config))
    reject_unknown_config_keys(GRPO_CONFIG_SCHEMA, config, source="retained GRPO")
    if type(dataset_bytes) is not bytes or not 0 < len(dataset_bytes) <= 65536:
        raise ValueError("invalid retained dataset")
    rows = []
    for line in dataset_bytes.splitlines():
        if not line.strip():
            raise ValueError("blank dataset row")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate dataset key")
                result[key] = value
            return result
        row = json.loads(line, object_pairs_hook=unique,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite dataset")))
        _finite_tree(row)
        if type(row) is not dict:
            raise ValueError("invalid dataset row")
        prompt = row.get(config['dataset'].get('prompt_column', 'prompt'))
        if type(prompt) not in (str, list) or not prompt:
            raise ValueError("invalid prompt")
        if type(prompt) is list:
            for message in prompt:
                if (type(message) is not dict or set(message) != {'role', 'content'}
                        or type(message['role']) is not str or not message['role']
                        or type(message['content']) is not str):
                    raise ValueError('invalid text message prompt')
        rows.append(row)
    if not 1 <= len(rows) <= 128:
        raise ValueError("invalid dataset count")
    model = config['model']
    training = config['training']
    if (model.get('lora_path') or model.get('chat_template') != 'native'
            or model.get('model_name') != model_identity.get('model_id')
            or type(model_identity.get('revision')) is not str
            or re.fullmatch(r'[0-9a-f]{40}', model_identity['revision']) is None):
        raise ValueError("invalid offline model binding")
    if 'tokenizer_revision' in model_identity and (type(model_identity['tokenizer_revision']) is not str
            or re.fullmatch(r'[0-9a-f]{40}', model_identity['tokenizer_revision']) is None):
        raise ValueError('invalid tokenizer revision')
    if (config.get('pivot', {}).get('enabled')
            or config.get('wandb', {}).get('enabled')
            or config['rewards'].get('custom', {}).get('enabled')
            or training.get('report_to', 'none') != 'none'
            or training.get('extra_args') or training.get('use_gspo')):
        raise ValueError("unsupported retained feature")
    for name in ('per_device_train_batch_size', 'gradient_accumulation_steps',
                 'num_generations', 'max_completion_length', 'max_steps', 'logging_steps',
                 'save_steps', 'save_total_limit', 'num_train_epochs'):
        value = training.get(name)
        if type(value) is not int or not 0 < value <= 100000:
            raise ValueError("invalid finite training control")
    if (training['num_generations'] < 2
            or (training['per_device_train_batch_size'] * training['gradient_accumulation_steps'])
            % training['num_generations']):
        raise ValueError("generation batch incompatible")
    if 'max_prompt_length' in training and (type(training['max_prompt_length']) is not int
            or not 0 < training['max_prompt_length'] <= 100000):
        raise ValueError('invalid prompt bound')
    for name in ('fp16', 'bf16'):
        if type(training.get(name)) is not bool:
            raise ValueError('invalid precision selection')
    for name in ('temperature', 'learning_rate', 'beta', 'weight_decay', 'warmup_ratio', 'max_grad_norm'):
        value = training.get(name, 1.0 if name == 'max_grad_norm' else None)
        if type(value) not in (int, float) or value < 0:
            raise ValueError("invalid numeric training control")
    for name in ('max_seq_length',):
        if type(model.get(name)) is not int or not 0 < model[name] <= 100000:
            raise ValueError('invalid model control')
    if type(model.get('load_in_4bit')) is not bool or model.get('dtype') not in (None, 'float16', 'bfloat16', 'float32'):
        raise ValueError('invalid model precision')
    for name in ('r', 'lora_alpha', 'random_state'):
        if type(config['lora'].get(name)) is not int or config['lora'][name] < (0 if name == 'random_state' else 1):
            raise ValueError('invalid LoRA control')
    lora = config['lora']
    if (type(lora.get('lora_dropout')) not in (int, float) or not 0 <= lora['lora_dropout'] < 1
            or type(lora.get('target_modules')) is not list or not lora['target_modules']
            or any(type(name) is not str or not name for name in lora['target_modules'])
            or len(set(lora['target_modules'])) != len(lora['target_modules'])
            or lora.get('bias') not in ('none', 'all', 'lora_only')
            or type(lora.get('use_gradient_checkpointing')) not in (bool, str)
            or any(type(lora.get(name, False)) is not bool for name in ('use_rslora', 'use_dora'))):
        raise ValueError('invalid LoRA selection')
    if type(reward_documents) is not dict or not reward_documents:
        raise ValueError("missing rewards")
    for name, document in reward_documents.items():
        if (type(document) is not dict or document.get('name') != name
                or document.get('type') not in (None, 'rubric')
                or document.get('scoring', {}).get('strategy') != 'binary'
                or not document.get('validation')):
            raise ValueError("unsupported reward")
        for score in ('on_pass', 'on_fail'):
            if type(document['scoring'].get(score, 1.0 if score == 'on_pass' else 0.0)) not in (int, float):
                raise ValueError('invalid reward score')
        for rule in document['validation']:
            if type(rule) is not dict or rule.get('type') != 'regex' or type(rule.get('match')) is not str:
                raise ValueError("unsupported raw reward rule")
            re.compile(rule['match'])
    items = config['rewards'].get('items')
    if type(items) is not list or not items:
        raise ValueError("missing reward selection")
    for item in items:
        if (item.get('name') not in reward_documents
                or type(item.get('weight')) not in (int, float) or item['weight'] < 0):
            raise ValueError("invalid reward selection")
    expected = {
        'method': 'grpo', 'model': model_identity,
        'config_sha256': _sha(_canonical(config)),
        'dataset_sha256': _sha(dataset_bytes),
        'rewards_sha256': _sha(_canonical({'documents': reward_documents, 'schema': reward_schema})),
    }
    if any(workload.get(key) != value for key, value in expected.items()):
        raise ValueError("workload input commitment differs")
    return config, tuple(rows), json.loads(_canonical(reward_documents)), json.loads(_canonical(reward_schema))


def _dependencies():
    # Effectful imports happen only after pure admission.
    import torch  # noqa: F401
    from unsloth import is_bfloat16_supported
    from trl import GRPOConfig, GRPOTrainer
    from Trainers.grpo.src import model_loader
    from datasets import Dataset
    return {'config_class': GRPOConfig, 'trainer_class': GRPOTrainer,
            'bf16_supported': is_bfloat16_supported(),
            'model_loader': model_loader, 'dataset_class': Dataset}


def _stamp_packaged_adapter(final_model, model_ref):
    """Selected offline artifacts name the verified logical model, not scratch."""
    from contextlib import ExitStack
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                info.st_ctime_ns, info.st_mode, info.st_uid, info.st_nlink)
    if (not isinstance(final_model, Path) or not final_model.is_absolute()
            or ".." in final_model.parts):
        raise ValueError("invalid packaged adapter directory")
    with ExitStack() as stack:
        parent = os.open(final_model.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, parent)
        chain = []
        for part in final_model.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, child)
            chain.append((parent, part, child))
            parent = child
        def check_parents():
            for ancestor, name, retained in chain:
                named = os.stat(name, dir_fd=ancestor, follow_symlinks=False)
                opened = os.fstat(retained)
                if (not stat.S_ISDIR(named.st_mode)
                        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)):
                    raise ValueError("changed packaged adapter directory")
        check_parents()
        descriptor = os.open("adapter_config.json", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        stack.callback(os.close, descriptor)
        before = os.fstat(descriptor)
        named = os.stat("adapter_config.json", dir_fd=parent, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > 65536
                or identity(before) != identity(named)):
            raise ValueError("invalid packaged adapter config")
        raw = os.pread(descriptor, 65537, 0)
        if len(raw) != before.st_size:
            raise ValueError("invalid packaged adapter config")
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate packaged adapter config")
                value[key] = item
            return value
        document = json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite adapter config")))
        _finite_tree(document)
        if type(document) is not dict:
            raise ValueError("invalid packaged adapter config")
        document["base_model_name_or_path"] = model_ref
        content = _canonical(document)
        check_parents()
        if (len(content) > 65536 or identity(os.fstat(descriptor)) != identity(before)
                or identity(os.stat("adapter_config.json", dir_fd=parent,
                                    follow_symlinks=False)) != identity(named)):
            raise ValueError("changed packaged adapter config")
        os.ftruncate(descriptor, 0)
        offset = 0
        while offset < len(content):
            check_parents()
            count = os.pwrite(descriptor, content[offset:], offset)
            if count <= 0:
                raise OSError("packaged adapter write rejected")
            offset += count
        os.fsync(descriptor)
        check_parents()
        after = os.fstat(descriptor)
        if ((after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                or identity(os.stat("adapter_config.json", dir_fd=parent,
                                    follow_symlinks=False)) != identity(after)
                or os.pread(descriptor, len(content) + 1, 0) != content):
            raise ValueError("changed packaged adapter config")


def run_grpo(*, config, dataset_bytes, reward_documents, reward_schema,
             model_snapshot, model_identity, output_root, workload,
             metric_sink=None, _dependency_factory=None, _packaged_artifacts=False,
             qualification_evidence=None, _child_deadline=None, _runtime_milestone=None):
    """Run the existing GRPO core with explicit immutable admitted inputs.

    This entry does not authenticate a public plan or attest model member
    inventory. The packaged caller must do both before invoking it.
    It never claims parameter updates or reward dispersion from step count.
    """
    if _runtime_milestone is not None and not callable(_runtime_milestone):
        raise ValueError("invalid private runtime milestone callback")
    def mark(value):
        if _runtime_milestone is not None:
            _runtime_milestone(value)
    mark("INPUT_ADMISSION")
    config, rows, documents, schema = admit_grpo_inputs(
        config=config, dataset_bytes=dataset_bytes, reward_documents=reward_documents,
        reward_schema=reward_schema, model_identity=model_identity, workload=workload)
    if type(_packaged_artifacts) is not bool:
        raise ValueError("invalid packaged artifact mode")
    if _packaged_artifacts and model_identity.get("tokenizer_revision") != model_identity["revision"]:
        raise ValueError("packaged GRPO requires one model/tokenizer snapshot")
    measurements = None
    if qualification_evidence is not None:
        if not _packaged_artifacts:
            raise ValueError("qualification measurements require packaged custody")
        measurements = _QualificationMeasurements(qualification_evidence, _child_deadline)
    mark("REWARDS")
    from Trainers.grpo.src.rewards import build_combined_reward_function
    reward_fn, reward_plan = build_combined_reward_function(
        config['rewards'], Path('.'), documents=documents, schema_config=schema, raw_only=True)
    if measurements is not None:
        reward_fn = measurements.reward_wrapper(reward_fn)
    model_identity = json.loads(_canonical(model_identity))
    workload = json.loads(_canonical(workload))
    if not isinstance(model_snapshot, Path):
        raise ValueError("explicit local model path required")
    if not model_snapshot.is_absolute() or not model_snapshot.is_dir():
        raise ValueError("local model unavailable")
    # Owned output root creation/custody is the packaged caller's responsibility.
    if not isinstance(output_root, Path) or not output_root.is_absolute() or not output_root.is_dir():
        raise ValueError("explicit output root required")
    mark("DEPENDENCIES")
    deps = (_dependency_factory or _dependencies)()
    mark("CONFIG")
    args = build_grpo_config(config, output_root / 'checkpoints',
                           config_class=deps['config_class'], bf16_supported=deps['bf16_supported'])
    helper = deps['model_loader']
    mark("MODEL_LIBRARY_LOAD")
    model, tokenizer, is_vl = helper.load_model_and_tokenizer(
        str(model_snapshot), max_seq_length=config['model']['max_seq_length'],
        dtype=config['model'].get('dtype'), load_in_4bit=config['model'].get('load_in_4bit', True),
        offline=True)
    if is_vl:
        raise ValueError("retained GRPO requires text model")
    mark("LORA_ATTACH")
    model = helper.apply_lora_adapters(model=model, is_vision_model=False, **config['lora'])
    mark("DATA_PREP")
    from Trainers.grpo.src.data_loader import dataset_from_rows, format_dataset_for_grpo
    dataset = dataset_from_rows(rows, dataset_class=deps['dataset_class'])
    dataset = format_dataset_for_grpo(dataset, tokenizer,
        prompt_column=config['dataset'].get('prompt_column', 'prompt'), num_proc=1,
        chat_template_kwargs=config['training'].get('chat_template_kwargs'))
    callbacks = []
    if metric_sink is not None:
        # Trainer callbacks need the actual transformers base only at execution.
        callback_class = deps.get('callback_class')
        if callback_class is None:
            from transformers import TrainerCallback
            callback_class = TrainerCallback
        class Metrics(callback_class):
            def on_log(self, args, state, control, logs=None, **kwargs):
                record = {'step': state.global_step}
                record.update({key: value for key, value in (logs or {}).items()
                               if type(value) in (int, float) and math.isfinite(value)})
                metric_sink(record)
        callbacks.append(Metrics())
    if measurements is not None:
        callback_class = deps.get('callback_class')
        if callback_class is None:
            from transformers import TrainerCallback
            callback_class = TrainerCallback
        class Evidence(callback_class):
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                mark("MEASUREMENT_BEGIN")
                measurements.begin(model, args)
                mark("TRAIN_CALL")
                return control

            def on_train_end(self, args, state, control, model=None, **kwargs):
                mark("MEASUREMENT_END")
                measurements.end(model)
                mark("TRAIN_CALL")
                return control

            def on_log(self, args, state, control, logs=None, **kwargs):
                measurements.log(state, logs)
                return control
        callbacks.append(Evidence())
    trainer, result = train_grpo_core(model=model, tokenizer=tokenizer, reward_fn=reward_fn,
        formatted_dataset=dataset, training_args=args, callbacks=callbacks,
        trainer_class=deps['trainer_class'],
        **({"_runtime_milestone": mark} if _runtime_milestone is not None else {}))
    mark("MEASUREMENT_RECORD")
    evidence = measurements.record(workload) if measurements is not None else None
    mark("SAVE")
    final = output_root / 'final_model'
    token = output_root / 'tokenizer'
    final.mkdir(exist_ok=False)
    token.mkdir(exist_ok=False)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(token))
    if _packaged_artifacts:
        # The native model-bundle selector validates colocated tokenizer files.
        # Save explicitly; do not depend on trainer processing_class side effects.
        tokenizer.save_pretrained(str(final))
        _stamp_packaged_adapter(final, model_identity["model_id"])
    mark("POST_SAVE")
    metrics = {'global_step': trainer.state.global_step,
               'log_history': trainer.state.log_history,
               'train_metrics': getattr(result, 'metrics', {})}
    lineage = {'method': 'grpo', 'model': model_identity,
               'input_commitments': {key: workload[key] for key in
                   ('config_sha256', 'dataset_sha256', 'rewards_sha256')},
               'reward_plan': reward_plan, 'actual_steps': trainer.state.global_step}
    if evidence is not None:
        metrics['qualification_evidence'] = evidence
        lineage['qualification_evidence'] = evidence
    for role, document in (('training_metrics', metrics), ('training_lineage', lineage),
                           ('workload_record', workload)):
        raw = _canonical(document)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("native record too large")
        with (output_root / (role + '.json')).open('xb') as stream:
            stream.write(raw)
    return {'final_model': final, 'tokenizer': token,
            **{role: output_root / (role + '.json') for role in
               ('training_metrics', 'training_lineage', 'workload_record')}}


def train_grpo_trainer(trainer, *, resume=None):
    return trainer.train(resume_from_checkpoint=resume)


_EVIDENCE_SCHEMA = "synaptic-grpo-qualification-evidence/v1"
_EVIDENCE_CHUNK_BYTES = 1024 * 1024


class _QualificationMeasurements:
    """Trusted child measurements, not independent historical tensor witnesses."""

    def __init__(self, policy, deadline):
        import time
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("finite original child deadline required")
        self.policy = validate_qualification_evidence_policy(policy)
        self.deadline = deadline
        self._clock = time.monotonic
        self.before = self.after = None
        self._model = None
        self.scale_rewards = None
        self.callbacks = self.values = self.nonconstant = 0
        self.minimum = self.maximum = None
        self.mean = self.m2 = 0.0
        self.reward_digest = hashlib.sha256(b"synaptic-grpo-reward-values/v1\0")
        self.grouped = []
        self._check()

    def _check(self):
        if self._clock() >= self.deadline:
            raise TimeoutError("GRPO measurement deadline")

    def _copy_chunk(self, chunk):
        # chunk is a bounded byte view; never copy or cast a complete model.
        import torch
        cpu = chunk.to(device="cpu", copy=True)
        # Signed zeros compare numerically equal. Canonicalize only this bounded
        # detached copy, never mutate the model or promote its dtype.
        cpu[cpu == 0] = 0
        return cpu.view(torch.uint8).numpy().tobytes()

    def fingerprint(self, model):
        import torch
        self._check()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() != 1:
                raise ValueError("distributed GRPO evidence unsupported")
        metadata, parameters, names = [], [], set()
        total = 0
        device = None
        for examined, (name, parameter) in enumerate(model.named_parameters()):
            self._check()
            if examined >= 65536 or type(name) is not str or not 0 < len(name.encode("utf-8")) <= 512:
                raise ValueError("parameter enumeration exceeds bound")
            if not parameter.requires_grad:
                continue
            if name in names or len(parameters) >= self.policy["max_trainable_tensors"]:
                raise ValueError("trainable parameter count exceeds bound")
            if (type(parameter) is not torch.nn.Parameter
                    or getattr(parameter, "_is_flat_param", False) or hasattr(parameter, "ds_id")
                    or parameter.device.type not in {"cpu", "cuda"}
                    or parameter.layout != torch.strided or not parameter.is_contiguous()
                    or parameter.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                    or parameter.ndim > 32):
                raise ValueError("unsupported trainable parameter layout")
            current_device = str(parameter.device)
            if device is not None and device != current_device:
                raise ValueError("multiple trainable devices unsupported")
            device = current_device
            size = parameter.numel() * parameter.element_size()
            if size <= 0 or size > self.policy["max_trainable_bytes"] - total:
                raise ValueError("trainable parameter bytes exceed bound")
            names.add(name)
            total += size
            metadata.append({"name": name, "shape": list(parameter.shape),
                             "dtype": str(parameter.dtype), "device": current_device, "size_bytes": size})
            parameters.append((name, parameter))
        if not parameters:
            raise ValueError("no trainable parameters measured")
        ordered = sorted(zip(metadata, parameters), key=lambda item: item[0]["name"])
        inventory = hashlib.sha256(b"synaptic-grpo-trainable-inventory/v1\0")
        for item, _ in ordered:
            inventory.update(_canonical(item) + b"\n")
        inventory_sha = inventory.hexdigest()
        digest = hashlib.sha256(b"synaptic-grpo-trainable-values/v1\0" + inventory_sha.encode("ascii"))
        with torch.no_grad():
            for item, (_, parameter) in ordered:
                view = parameter.detach().view(-1)
                per_chunk = max(1, _EVIDENCE_CHUNK_BYTES // parameter.element_size())
                for offset in range(0, view.numel(), per_chunk):
                    self._check()
                    chunk = view[offset:offset + per_chunk]
                    if not bool(torch.isfinite(chunk).all().item()):
                        raise ValueError("nonfinite trainable parameter")
                    raw = self._copy_chunk(chunk)
                    if not 0 < len(raw) <= _EVIDENCE_CHUNK_BYTES:
                        raise ValueError("invalid parameter transfer")
                    digest.update(raw)
                    self._check()
        return {"inventory_sha256": inventory_sha, "tensor_count": len(parameters),
                "total_bytes": total, "sha256": digest.hexdigest()}

    def begin(self, model, args):
        self._check()
        if (self.before is not None or type(getattr(args, "world_size", None)) is not int
                or args.world_size != 1):
            raise ValueError("single training begin required")
        mode = getattr(args, "scale_rewards", None)
        if mode is not None and (type(mode) is not str or mode not in {"group", "none", "batch"}):
            raise ValueError("unknown trainer reward scale")
        self.scale_rewards = mode
        self.before = self.fingerprint(model)
        self._model = model

    def end(self, model):
        self._check()
        if self.before is None or self.after is not None or model is not self._model:
            raise ValueError("same model and single training end required")
        self.after = self.fingerprint(model)
        if any(self.before[key] != self.after[key] for key in
               ("inventory_sha256", "tensor_count", "total_bytes")):
            raise ValueError("trainable inventory changed")

    def reward_wrapper(self, actual):
        def observe(completions, prompts=None, **kwargs):
            import struct
            self._check()
            if self.before is None or self.after is not None:
                raise ValueError("reward outside measured training interval")
            count = len(completions)
            if count <= 0 or count > self.policy["max_reward_values"] - self.values:
                raise ValueError("reward observation bound exceeded")
            values = actual(completions, prompts=prompts, **kwargs)
            if type(values) is not list or len(values) != count:
                raise ValueError("invalid reward observations")
            low = high = None
            for item in values:
                self._check()
                if type(item) not in (int, float) or not math.isfinite(item):
                    raise ValueError("nonfinite reward observation")
                value = float(item)
                low = value if low is None else min(low, value)
                high = value if high is None else max(high, value)
                self.values += 1
                delta = value - self.mean
                self.mean += delta / self.values
                self.m2 += delta * (value - self.mean)
                if not math.isfinite(self.mean) or not math.isfinite(self.m2):
                    raise ValueError("reward moments overflow")
                self.minimum = value if self.minimum is None else min(self.minimum, value)
                self.maximum = value if self.maximum is None else max(self.maximum, value)
                self.reward_digest.update(struct.pack(">d", value))
            self.callbacks += 1
            self.nonconstant += int(low != high)
            self._check()
            return values
        return observe

    def log(self, state, logs):
        self._check()
        logs = logs or {}
        if "reward_std" not in logs and "frac_reward_zero_std" not in logs:
            return
        if len(self.grouped) >= min(10000, self.policy["max_reward_values"]) or type(state.global_step) is not int or state.global_step < 0:
            raise ValueError("grouped observation bound exceeded")
        item = {"step": state.global_step, "reward_std": None, "frac_reward_zero_std": None}
        for key in ("reward_std", "frac_reward_zero_std"):
            if key in logs:
                value = logs[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid grouped reward observation")
                if key == "frac_reward_zero_std" and value > 1:
                    raise ValueError("invalid grouped reward fraction")
                item[key] = float(value)
        self.grouped.append(item)

    def record(self, core):
        self._check()
        if self.before is None or self.after is None:
            raise ValueError("training boundaries not measured")
        parameters = {
            "before_checkpoint": "on_train_begin", "after_checkpoint": "on_train_end",
            "before_inventory_sha256": self.before["inventory_sha256"],
            "after_inventory_sha256": self.after["inventory_sha256"],
            "tensor_count": self.before["tensor_count"], "total_bytes": self.before["total_bytes"],
            "before_sha256": self.before["sha256"], "after_sha256": self.after["sha256"],
            "changed": self.before["sha256"] != self.after["sha256"],
        }
        rewards = {"callback_count": self.callbacks, "value_count": self.values,
                   "nonconstant_callbacks": self.nonconstant, "minimum": self.minimum,
                   "maximum": self.maximum, "mean": self.mean if self.values else None,
                   "variance": max(0.0, self.m2 / self.values) if self.values else None,
                   "values_sha256": self.reward_digest.hexdigest(),
                   "trainer_scale_rewards": self.scale_rewards,
                   "grouped_observations": list(self.grouped)}
        varied = (self.scale_rewards in {"group", "none"} and
                  any(item["reward_std"] is not None and item["reward_std"] > 0
                      and item["frac_reward_zero_std"] is not None and item["frac_reward_zero_std"] < 1
                      for item in self.grouped))
        return {"schema_version": _EVIDENCE_SCHEMA, "policy": dict(self.policy),
                "core_workload_sha256": _sha(_canonical(core)),
                "parameters": parameters, "rewards": rewards,
                "measurement_result": "changed_with_grouped_variation"
                if parameters["changed"] and self.nonconstant and varied else "inconclusive"}


def validate_qualification_evidence_record(value, *, policy, core, log_history):
    """Validate trusted measurement declarations; historical weights are absent."""
    policy = validate_qualification_evidence_policy(policy)
    _finite_tree(value)
    fields = {"schema_version", "policy", "core_workload_sha256", "parameters", "rewards", "measurement_result"}
    if (type(value) is not dict or set(value) != fields
            or value["schema_version"] != _EVIDENCE_SCHEMA
            or value["policy"] != policy
            or validate_qualification_evidence_policy(value["policy"]) != policy
            or value["core_workload_sha256"] != _sha(_canonical(core))):
        raise ValueError("invalid qualification measurement bindings")
    parameters = value["parameters"]
    if (type(parameters) is not dict or set(parameters) != {
            "before_checkpoint", "after_checkpoint", "before_inventory_sha256", "after_inventory_sha256",
            "tensor_count", "total_bytes", "before_sha256", "after_sha256", "changed"}
            or parameters["before_checkpoint"] != "on_train_begin"
            or parameters["after_checkpoint"] != "on_train_end"):
        raise ValueError("invalid parameter measurement")
    for key in ("before_inventory_sha256", "after_inventory_sha256", "before_sha256", "after_sha256"):
        if type(parameters[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", parameters[key]) is None:
            raise ValueError("invalid measurement digest")
    if (parameters["before_inventory_sha256"] != parameters["after_inventory_sha256"]
            or type(parameters["tensor_count"]) is not int
            or not 1 <= parameters["tensor_count"] <= policy["max_trainable_tensors"]
            or type(parameters["total_bytes"]) is not int
            or not parameters["tensor_count"] * 2 <= parameters["total_bytes"] <= policy["max_trainable_bytes"]
            or type(parameters["changed"]) is not bool
            or parameters["changed"] != (parameters["before_sha256"] != parameters["after_sha256"])):
        raise ValueError("inconsistent parameter measurement")
    rewards = value["rewards"]
    if type(rewards) is not dict or set(rewards) != {
            "callback_count", "value_count", "nonconstant_callbacks", "minimum", "maximum", "mean",
            "variance", "values_sha256", "trainer_scale_rewards", "grouped_observations"}:
        raise ValueError("invalid reward measurement")
    for key in ("callback_count", "value_count", "nonconstant_callbacks"):
        if type(rewards[key]) is not int or not 0 <= rewards[key] <= policy["max_reward_values"]:
            raise ValueError("invalid reward count")
    if (not rewards["nonconstant_callbacks"] <= rewards["callback_count"] <= rewards["value_count"]
            or rewards["value_count"] < rewards["callback_count"] + rewards["nonconstant_callbacks"]
            or (rewards["callback_count"] == 0) != (rewards["value_count"] == 0)
            or type(rewards["values_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", rewards["values_sha256"]) is None):
        raise ValueError("inconsistent reward count")
    for key in ("minimum", "maximum", "mean", "variance"):
        item = rewards[key]
        if rewards["value_count"] == 0:
            if item is not None:
                raise ValueError("unobserved reward statistic")
        elif type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError("invalid reward statistic")
    if rewards["value_count"]:
        if (rewards["minimum"] > rewards["maximum"] or rewards["variance"] < 0
                or not rewards["minimum"] <= rewards["mean"] <= rewards["maximum"]
                or (rewards["minimum"] == rewards["maximum"] and
                    (rewards["nonconstant_callbacks"] or rewards["variance"] != 0))):
            raise ValueError("inconsistent reward moments")
        if rewards["value_count"] == 1 and (rewards["minimum"] != rewards["maximum"]
                or rewards["mean"] != rewards["minimum"] or rewards["variance"] != 0):
            raise ValueError("inconsistent single reward value")
    elif rewards["values_sha256"] != hashlib.sha256(b"synaptic-grpo-reward-values/v1\0").hexdigest():
        raise ValueError("invalid unobserved reward digest")
    mode = rewards["trainer_scale_rewards"]
    if mode is not None and (type(mode) is not str or mode not in {"group", "none", "batch"}):
        raise ValueError("invalid trainer reward scale")
    groups = rewards["grouped_observations"]
    if type(groups) is not list or len(groups) > min(10000, policy["max_reward_values"]):
        raise ValueError("invalid grouped observations")
    if type(log_history) is not list or len(log_history) > 100000:
        raise ValueError("trainer log count exceeds bound")
    expected = []
    for log in log_history:
        if type(log) is not dict:
            raise ValueError("invalid trainer log")
        if "reward_std" not in log and "frac_reward_zero_std" not in log:
            continue
        if len(expected) >= min(10000, policy["max_reward_values"]):
            raise ValueError("trainer log bound exceeded")
        expected.append({"step": log.get("step"), "reward_std": log.get("reward_std"),
                         "frac_reward_zero_std": log.get("frac_reward_zero_std")})
    last = -1
    for item in groups:
        if (type(item) is not dict or set(item) != {"step", "reward_std", "frac_reward_zero_std"}
                or type(item["step"]) is not int or not max(0, last) <= item["step"] <= core["config"]["training"]["max_steps"]):
            raise ValueError("invalid grouped step")
        last = item["step"]
        if item["reward_std"] is None and item["frac_reward_zero_std"] is None:
            raise ValueError("empty grouped observation")
        for key in ("reward_std", "frac_reward_zero_std"):
            number = item[key]
            if number is not None and (type(number) not in (int, float) or not math.isfinite(number)
                                      or number < 0 or (key == "frac_reward_zero_std" and number > 1)):
                raise ValueError("invalid grouped value")
    if _canonical(groups) != _canonical(expected):
        raise ValueError("grouped observations differ from trainer logs")
    varied = (mode in {"group", "none"} and
              any(item["reward_std"] is not None and item["reward_std"] > 0
                  and item["frac_reward_zero_std"] is not None and item["frac_reward_zero_std"] < 1
                  for item in groups))
    result = ("changed_with_grouped_variation" if parameters["changed"]
              and rewards["nonconstant_callbacks"] and varied else "inconclusive")
    if value["measurement_result"] != result:
        raise ValueError("inconsistent measurement result")
