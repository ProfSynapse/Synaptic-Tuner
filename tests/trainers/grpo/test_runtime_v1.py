"""Actual retained-input control flow with CPU dependency substitutes."""
import copy
from dataclasses import make_dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from Trainers.grpo import runtime_v1 as runtime


class Dataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    @property
    def column_names(self):
        return list(self[0])

    def map(self, function, **kwargs):
        return Dataset([function(row) for row in self])


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        return messages[0]['content']

    def save_pretrained(self, destination):
        Path(destination, 'tokenizer.json').write_text('{}')


def inputs(tmp_path):
    config = yaml.safe_load((Path(runtime.__file__).parent / 'configs/config.yaml').read_text())
    config['model'].update(model_name='declared/model', lora_path=None, chat_template='native')
    config['training'].update(max_steps=3, per_device_train_batch_size=2,
                             gradient_accumulation_steps=1, num_generations=2,
                             report_to='none', extra_args={}, use_gspo=False, learning_rate=5e-6, beta=0.04)
    config['pivot'] = {'enabled': False}
    config['wandb'] = {'enabled': False}
    config['rewards'] = {'items': [{'name': 'exact', 'weight': 1.0}]}
    documents = {'exact': {'name': 'exact', 'scope': 'response',
                'validation': [{'type': 'regex', 'match': '^OK$'}],
                'scoring': {'strategy': 'binary', 'on_pass': 1.0, 'on_fail': 0.0}}}
    raw = b'{"prompt":[{"role":"user","content":"say OK"}],"declared":"retained"}\n'
    identity = {'model_id': 'declared/model', 'revision': 'a' * 40}
    workload = {'method': 'grpo', 'model': copy.deepcopy(identity),
                'config_sha256': runtime._sha(runtime._canonical(config)),
                'dataset_sha256': runtime._sha(raw),
                'rewards_sha256': runtime._sha(runtime._canonical({'documents': documents, 'schema': {}}))}
    model = tmp_path / 'model'; model.mkdir()
    output = tmp_path / 'output'; output.mkdir()
    return dict(config=config, dataset_bytes=raw, reward_documents=documents,
                reward_schema={}, model_snapshot=model, model_identity=identity,
                output_root=output, workload=workload)


def dependencies(events, *, save_error=False):
    token = Tokenizer()
    class Trainer:
        def __init__(self, **kwargs):
            events.append(('trainer', kwargs))
            self.kwargs = kwargs
            self.state = SimpleNamespace(global_step=3, log_history=[{'reward': 0.5, 'loss': 0.2}])
        def train(self, **kwargs):
            events.append(('train', kwargs))
            assert self.kwargs['train_dataset'][0]['declared'] == 'retained'
            assert self.kwargs['reward_funcs'](['OK', 'bad']) == [1.0, 0.0]
            for callback in self.kwargs['callbacks']:
                callback.on_log(None, self.state, None, logs={'reward': 0.5})
            return SimpleNamespace(metrics={'train_loss': 0.2})
        def save_model(self, destination):
            if save_error:
                raise RuntimeError('save rejected')
            Path(destination, 'adapter_config.json').write_text('{}')
    def load(*args, **kwargs):
        events.append(('model', args, kwargs)); return object(), token, False
    names = 'output_dir per_device_train_batch_size gradient_accumulation_steps num_generations max_completion_length temperature learning_rate weight_decay warmup_ratio lr_scheduler_type optim logging_steps save_steps save_total_limit num_train_epochs fp16 bf16 seed report_to beta max_prompt_length max_steps'.split()
    config_class = make_dataclass('Config', [(name, object) for name in names])
    return lambda: dict(config_class=config_class,
            trainer_class=Trainer, bf16_supported=True, dataset_class=Dataset,
            callback_class=object, model_loader=SimpleNamespace(
                load_model_and_tokenizer=load, apply_lora_adapters=lambda **kwargs: kwargs['model']))


def test_real_flow_offline_rewards_metrics_and_mandatory_roles(tmp_path, monkeypatch):
    values = inputs(tmp_path); events = []; metrics = []
    from Trainers.grpo.src import rewards
    monkeypatch.setattr(rewards, 'extract', lambda *a, **k: pytest.fail('ambient tool parser'))
    result = runtime.run_grpo(**values, metric_sink=metrics.append,
                             _dependency_factory=dependencies(events))
    assert set(result) == {'final_model', 'tokenizer', 'training_lineage', 'training_metrics', 'workload_record'}
    assert events[0][2]['offline'] is True
    assert events[1][1]['args'].num_generations == 2
    assert events[2] == ('train', {'resume_from_checkpoint': None})
    assert metrics == [{'step': 3, 'reward': 0.5}]
    assert json.loads(result['training_metrics'].read_bytes())['train_metrics'] == {'train_loss': 0.2}
    assert json.loads(result['training_lineage'].read_bytes())['actual_steps'] == 3
    assert 'parameter_updated' not in result['training_lineage'].read_text()


@pytest.mark.parametrize('change', ['dataset', 'reward', 'config', 'model', 'method', 'nonfinite', 'bool', 'custom', 'pivot', 'empty'])
def test_rejects_before_dependency_effects(tmp_path, change):
    values = inputs(tmp_path)
    if change == 'dataset': values['dataset_bytes'] += b' '
    elif change == 'reward': values['reward_documents']['exact']['validation'][0]['match'] = 'bad'
    elif change == 'config': values['config']['training']['temperature'] = 0.2
    elif change == 'model': values['model_identity']['revision'] = 'b' * 40
    elif change == 'method': values['workload']['method'] = 'sft'
    elif change == 'nonfinite': values['config']['training']['beta'] = float('nan')
    elif change == 'bool': values['config']['training']['max_steps'] = True
    elif change == 'custom': values['config']['rewards']['custom'] = {'enabled': True}
    elif change == 'pivot': values['config']['pivot']['enabled'] = True
    elif change == 'empty': values['reward_documents'] = {}
    with pytest.raises((ValueError, TypeError)):
        runtime.run_grpo(**values, _dependency_factory=lambda: pytest.fail('ML effect'))


def test_failed_adapter_save_propagates(tmp_path):
    with pytest.raises(RuntimeError, match='save rejected'):
        runtime.run_grpo(**inputs(tmp_path), _dependency_factory=dependencies([], save_error=True))


def test_explicit_empty_rewards_never_load_default(tmp_path, monkeypatch):
    from Trainers.grpo.src import rewards
    monkeypatch.setattr(rewards.RewardEngine, '_load_schema_config', lambda *a: pytest.fail('default schema'))
    monkeypatch.setattr(rewards.RewardEngine, '_load_rubrics', lambda *a: pytest.fail('default rubric'))
    with pytest.raises(ValueError, match='missing'):
        rewards.build_combined_reward_function({'items': [{'name': 'missing'}]}, tmp_path, documents={})


def test_retained_values_isolated_from_caller_mutation_at_factory(tmp_path):
    values = inputs(tmp_path); original = copy.deepcopy(values['workload']); events = []
    factory = dependencies(events)
    def mutate():
        values['config']['training']['num_generations'] = 999
        values['reward_documents']['exact']['validation'][0]['match'] = 'bad'
        values['workload']['method'] = 'sft'
        values['model_identity']['revision'] = 'b' * 40
        return factory()
    result = runtime.run_grpo(**values, _dependency_factory=mutate)
    assert json.loads(result['workload_record'].read_bytes()) == original
    assert events[1][1]['args'].num_generations == 2


def test_tokenizer_save_failure_propagates(tmp_path):
    factory = dependencies([])
    def failing():
        deps = factory()
        load = deps['model_loader'].load_model_and_tokenizer
        def replacement(*args, **kwargs):
            model, token, vision = load(*args, **kwargs)
            token.save_pretrained = lambda *a: (_ for _ in ()).throw(RuntimeError('tokenizer save rejected'))
            return model, token, vision
        deps['model_loader'].load_model_and_tokenizer = replacement
        return deps
    with pytest.raises(RuntimeError, match='tokenizer save rejected'):
        runtime.run_grpo(**inputs(tmp_path), _dependency_factory=failing)


def test_legacy_and_new_argument_builder_identical(tmp_path):
    import ast
    path = Path(runtime.__file__).parent / 'train_grpo.py'
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_build_grpo_config')
    values = inputs(tmp_path)
    deps = dependencies([])()
    namespace = {'Path': Path, 'GRPOConfig': deps['config_class'],
                 'is_bfloat16_supported': lambda: True}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    original = namespace['_build_grpo_config'](values['config'], tmp_path)
    extracted = runtime.build_grpo_config(values['config'], tmp_path,
                    config_class=deps['config_class'], bf16_supported=True)
    assert original == extracted


@pytest.mark.parametrize('change', ['duplicate', 'message', 'nan_row', 'revision', 'tokenizer', 'r_zero', 'precision', 'prompt_length'])
def test_structured_controls_reject_before_ml(tmp_path, change):
    values = inputs(tmp_path)
    if change == 'duplicate': values['dataset_bytes'] = b'{"prompt":"a","prompt":"b"}\n'
    elif change == 'message': values['dataset_bytes'] = b'{"prompt":[{"role":"user","content":{"bad":"shape"}}]}\n'
    elif change == 'nan_row': values['dataset_bytes'] = b'{"prompt":"a","extra":NaN}\n'
    elif change == 'revision': values['model_identity']['revision'] = 'Z' * 40
    elif change == 'tokenizer': values['model_identity']['tokenizer_revision'] = 'not-pinned'
    elif change == 'r_zero': values['config']['lora']['r'] = 0
    elif change == 'precision': values['config']['training']['bf16'] = 1
    elif change == 'prompt_length': values['config']['training']['max_prompt_length'] = -1
    # Refresh commitments so tests locate input validation, not just hash mismatch.
    values['workload'].update(model=copy.deepcopy(values['model_identity']),
        config_sha256=runtime._sha(runtime._canonical(values['config'])),
        dataset_sha256=runtime._sha(values['dataset_bytes']))
    with pytest.raises(ValueError):
        runtime.run_grpo(**values, _dependency_factory=lambda: pytest.fail('ML effect'))


@pytest.mark.parametrize('offline', [False, True])
def test_actual_model_helper_preserves_legacy_or_explicit_offline_kwargs(tmp_path, offline):
    import ast
    from typing import Optional, Tuple
    source = Path(runtime.__file__).parent / 'src/model_loader.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'load_model_and_tokenizer')
    calls = []
    def load(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(config=SimpleNamespace()), Tokenizer()
    namespace = {'Optional': Optional, 'Tuple': Tuple,
                 'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                 'FastLanguageModel': SimpleNamespace(from_pretrained=load),
                 '_is_vision_model': lambda name: False, '_VISION_AVAILABLE': False}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    kwargs = {'offline': True} if offline else {'hf_token': 'test-owned-token'}
    namespace['load_model_and_tokenizer'](str(tmp_path), **kwargs)
    expected = dict(model_name=str(tmp_path), max_seq_length=2048, dtype=None,
                    load_in_4bit=True, token=False if offline else 'test-owned-token')
    if offline: expected['local_files_only'] = True
    assert calls == [expected]


def test_lazy_runtime_import_no_ml_in_fresh_process(tmp_path):
    import subprocess
    import sys
    script = "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from Trainers.grpo import runtime_v1; assert Path(runtime_v1.__file__).resolve() == Path(sys.argv[2]); assert not {'torch','trl','unsloth','datasets'} & set(sys.modules)"
    subprocess.run([sys.executable, '-I', '-B', '-c', script,
                    str(Path(runtime.__file__).resolve().parents[2]),
                    str(Path(runtime.__file__).resolve())], check=True, timeout=15, cwd=tmp_path)


def test_retained_runtime_with_real_dataset(tmp_path):
    datasets = pytest.importorskip("datasets")
    events = []
    factory = dependencies(events)
    def actual_dataset():
        deps = factory()
        deps["dataset_class"] = datasets.Dataset
        return deps
    result = runtime.run_grpo(**inputs(tmp_path), _dependency_factory=actual_dataset)
    assert isinstance(events[1][1]["train_dataset"], datasets.Dataset)
    assert events[1][1]["train_dataset"][0]["prompt"] == "say OK"
    assert json.loads(result["training_metrics"].read_bytes())["global_step"] == 3


def test_private_milestones_follow_actual_core_boundaries(tmp_path):
    values = inputs(tmp_path)
    marks = []
    runtime.run_grpo(**values, _dependency_factory=dependencies([]), _runtime_milestone=marks.append)
    assert marks == ['INPUT_ADMISSION', 'REWARDS', 'DEPENDENCIES', 'CONFIG',
                     'MODEL_LIBRARY_LOAD', 'LORA_ATTACH', 'DATA_PREP', 'TRAINER_SETUP',
                     'TRAIN_CALL', 'MEASUREMENT_RECORD', 'SAVE', 'POST_SAVE']


@pytest.mark.parametrize('boundary', ['config', 'trainer', 'train', 'save'])
def test_milestone_stays_at_actual_failing_operation(tmp_path, boundary):
    values = inputs(tmp_path)
    marks = []
    deps = dependencies([], save_error=boundary == 'save')()
    def reject(*args, **kwargs):
        raise RuntimeError('PRIVATE_SENTINEL')
    if boundary == 'config':
        deps['config_class'] = reject
    elif boundary == 'trainer':
        deps['trainer_class'] = reject
    elif boundary == 'train':
        deps['trainer_class'].train = reject
    with pytest.raises(Exception):
        runtime.run_grpo(**values, _dependency_factory=lambda: deps, _runtime_milestone=marks.append)
    assert marks[-1] == {'config':'CONFIG', 'trainer':'TRAINER_SETUP',
                          'train':'TRAIN_CALL', 'save':'SAVE'}[boundary]
