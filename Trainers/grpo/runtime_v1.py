"""Explicit retained-input GRPO core; importing this module loads no ML stack."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict

from shared.training_utils import build_trainer_config, reject_unknown_config_keys

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
                    training_args, callbacks, trainer_class, resume=None):
    trainer = create_grpo_trainer(trainer_class=trainer_class, model=model, processing_class=tokenizer,
                            reward_funcs=reward_fn, args=training_args,
                            train_dataset=formatted_dataset, callbacks=callbacks)
    result = train_grpo_trainer(trainer, resume=resume)
    return trainer, result


def create_grpo_trainer(*, trainer_class, **kwargs):
    return trainer_class(**kwargs)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _finite_tree(value, depth=0, budget=None):
    if budget is None:
        budget = [20000]
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("input node bound exceeded")
    if depth > 32:
        raise ValueError("input depth exceeded")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("nonfinite input")
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("invalid input key")
            _finite_tree(child, depth + 1, budget)
    elif type(value) is list:
        for child in value:
            _finite_tree(child, depth + 1, budget)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("invalid input type")


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


def run_grpo(*, config, dataset_bytes, reward_documents, reward_schema,
             model_snapshot, model_identity, output_root, workload,
             metric_sink=None, _dependency_factory=None):
    """Run the existing GRPO core with explicit immutable admitted inputs.

    This entry does not authenticate a public plan or attest model member
    inventory. The future packaged caller must do both before invoking it.
    It never claims parameter updates or reward dispersion from step count.
    """
    config, rows, documents, schema = admit_grpo_inputs(
        config=config, dataset_bytes=dataset_bytes, reward_documents=reward_documents,
        reward_schema=reward_schema, model_identity=model_identity, workload=workload)
    from Trainers.grpo.src.rewards import build_combined_reward_function
    reward_fn, reward_plan = build_combined_reward_function(
        config['rewards'], Path('.'), documents=documents, schema_config=schema, raw_only=True)
    model_identity = json.loads(_canonical(model_identity))
    workload = json.loads(_canonical(workload))
    if not isinstance(model_snapshot, Path):
        raise ValueError("explicit local model path required")
    if not model_snapshot.is_absolute() or not model_snapshot.is_dir():
        raise ValueError("local model unavailable")
    # Owned output root creation/custody is the packaged caller's responsibility.
    if not isinstance(output_root, Path) or not output_root.is_absolute() or not output_root.is_dir():
        raise ValueError("explicit output root required")
    deps = (_dependency_factory or _dependencies)()
    args = build_grpo_config(config, output_root / 'checkpoints',
                           config_class=deps['config_class'], bf16_supported=deps['bf16_supported'])
    helper = deps['model_loader']
    model, tokenizer, is_vl = helper.load_model_and_tokenizer(
        str(model_snapshot), max_seq_length=config['model']['max_seq_length'],
        dtype=config['model'].get('dtype'), load_in_4bit=config['model'].get('load_in_4bit', True),
        offline=True)
    if is_vl:
        raise ValueError("retained GRPO requires text model")
    model = helper.apply_lora_adapters(model=model, is_vision_model=False, **config['lora'])
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
    trainer, result = train_grpo_core(model=model, tokenizer=tokenizer, reward_fn=reward_fn,
        formatted_dataset=dataset, training_args=args, callbacks=callbacks,
        trainer_class=deps['trainer_class'])
    final = output_root / 'final_model'
    token = output_root / 'tokenizer'
    final.mkdir(exist_ok=False)
    token.mkdir(exist_ok=False)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(token))
    metrics = {'global_step': trainer.state.global_step,
               'log_history': trainer.state.log_history,
               'train_metrics': getattr(result, 'metrics', {})}
    lineage = {'method': 'grpo', 'model': model_identity,
               'input_commitments': {key: workload[key] for key in
                   ('config_sha256', 'dataset_sha256', 'rewards_sha256')},
               'reward_plan': reward_plan, 'actual_steps': trainer.state.global_step}
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
