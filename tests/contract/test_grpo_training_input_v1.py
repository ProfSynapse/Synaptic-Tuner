"""First public GRPO declaration lane, without execution capability."""
import copy
import hashlib
import json

import pytest

from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from tuner.training.grpo_compilation import admit_grpo_snapshot, compile_grpo_input, grpo_prepared_semantic_digest


def document():
    return {
        "schema_version": "synaptic-training-input/v1", "method": "grpo",
        "model": {"ref": "model/example", "revision": "a" * 40, "tokenizer_revision": "a" * 40},
        "dataset": {"ref": "prepared://sha256/" + "b" * 64},
        "hyperparameters": {
            "schema_version": "synaptic-grpo-hyperparameters/v1",
            "batch_size": 2, "gradient_accumulation_steps": 1, "num_generations": 2,
            "max_steps": 3, "max_seq_length": 256, "max_prompt_length": 128,
            "max_completion_length": 32, "learning_rate": 5e-6, "temperature": .8,
            "beta": .04, "weight_decay": 0., "warmup_ratio": 0., "max_grad_norm": 1.,
            "seed": 3407, "logging_steps": 1, "save_steps": 10, "save_total_limit": 1,
            "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.,
            "lora_target_modules": ["q_proj", "v_proj"], "use_dora": False,
            "use_rslora": False, "load_in_4bit": False, "bf16": False, "fp16": False,
            "prompt_column": "prompt",
        },
        "rewards": {"items": [{"name": "exact", "weight": 1}],
            "documents": {"exact": {"name": "exact", "scope": "response",
                "validation": [{"type": "regex", "match": "^OK$"}],
                "scoring": {"strategy": "binary", "on_pass": 1., "on_fail": 0.}}}, "schema": {}},
        "artifacts": {"required_kinds": sorted(["workload_record", "training_lineage",
            "training_metrics", "final_model", "tokenizer"]), "retain_checkpoints": False},
    }


def prepared(raw):
    digest = grpo_prepared_semantic_digest(content_digest=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw), row_count=len(raw.splitlines()), prompt_column="prompt")
    return PreparedTrainingInputIdentity(ref="prepared://sha256/" + digest,
        revision=digest, content_digest=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw), format="synaptic-grpo-prompts/v1")


def bound(value, raw):
    detached = value.to_dict()
    detached["dataset"] = {"ref": prepared(raw).ref}
    return TrainingInputV1.from_dict(detached)


def test_roundtrip_detaches_rewards_and_compiles_actual_core():
    source = document()
    value = TrainingInputV1.from_dict(source)
    source["rewards"]["documents"]["exact"]["validation"][0]["match"] = "changed"
    assert value.to_dict() == document()
    assert TrainingInputV1.from_json(value.canonical_json()) == value
    raw = b'{"prompt":"Say OK","answer":"OK","weight":0.25}\n'
    workload = compile_grpo_input(bound(value, raw), prepared(raw)).document
    assert workload["dataset_sha256"] == hashlib.sha256(raw).hexdigest()
    config, rows, _, _ = admit_grpo_snapshot(bound(value, raw), prepared(raw), raw)
    assert rows[0]["weight"] == .25 and rows[0]["answer"] == "OK"
    assert config["training"]["optim"] == "adamw_8bit"


@pytest.mark.parametrize("field,bad", [("batch_size", True), ("num_generations", 1),
    ("max_steps", 100001), ("learning_rate", float("nan")), ("temperature", float("inf")),
    ("warmup_ratio", -1), ("max_prompt_length", 256), ("prompt_column", "")])
def test_hyperparameter_negative(field, bad):
    source = document(); source["hyperparameters"][field] = bad
    with pytest.raises((ValueError, TypeError)):
        TrainingInputV1.from_dict(source)


@pytest.mark.parametrize("mutation", [
    lambda d: d["rewards"].update(schema={"tool_schema_path": "ambient"}),
    lambda d: d["rewards"]["items"][0].update(weight=0),
    lambda d: d["rewards"]["documents"]["exact"].update(ground_truth="path"),
    lambda d: d["rewards"]["documents"]["exact"]["validation"][0].update(match="["),
    lambda d: d["hyperparameters"].update(extra_args={}),
])
def test_reward_and_extra_controls_reject(mutation):
    source = document(); mutation(source)
    with pytest.raises((ValueError, TypeError)):
        TrainingInputV1.from_dict(source)


@pytest.mark.parametrize("raw", [b'\xef\xbb\xbf{"prompt":"ok"}\n',
    '{"prompt":"ok"}\n'.encode("utf-16"), b'{"prompt":"ok","x":NaN}\n',
    b'{"prompt":"ok","prompt":"other"}\n', b'{"prompt":"ok"}\n\n'])
def test_actual_snapshot_invalid(raw):
    with pytest.raises((ValueError, TypeError)):
        admit_grpo_snapshot(bound(TrainingInputV1.from_dict(document()), raw), prepared(raw), raw)


def test_nondefault_prompt_column_collision_rejects():
    source = document(); source["hyperparameters"]["prompt_column"] = "question"
    raw = b'{"question":"Say OK","prompt":"reward column"}\n'
    with pytest.raises(ValueError, match="derived prompt"):
        admit_grpo_snapshot(bound(TrainingInputV1.from_dict(source), raw), prepared(raw), raw)


def test_reward_and_config_commitments_change_independently():
    first = document(); second = copy.deepcopy(first)
    second["rewards"]["documents"]["exact"]["validation"][0]["match"] = "^YES$"
    raw = b'{"prompt":"Say OK"}\n'
    a = compile_grpo_input(bound(TrainingInputV1.from_dict(first), raw), prepared(raw)).document
    b = compile_grpo_input(bound(TrainingInputV1.from_dict(second), raw), prepared(raw)).document
    assert a["config_sha256"] == b["config_sha256"]
    assert a["rewards_sha256"] != b["rewards_sha256"]
    assert json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)


def test_explicit_recipe_registry_compilation_preserves_default_sft():
    from tuner.training import GRPORecipe, default_recipe_registry
    from tuner.training.contracts import CanonicalDocument
    from tuner.training.recipes import RecipeRegistry, RecipeNotRegistered, compile_execution_workload
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / "training"))
    from tests.training.test_sft_compilation import _execution_source
    raw = b'{"prompt":"Say OK"}\n'
    value = TrainingInputV1.from_dict(document())
    resolved = CanonicalDocument.from_mapping({"schema_version": "synaptic-grpo-config/v1",
        "method": "grpo", "execution": {"mode": "developer_integration"},
        "training_input": bound(value, raw).to_dict(), "prepared_input": prepared(raw).to_dict()})
    registry = RecipeRegistry(); registry.register(GRPORecipe())
    compiled = compile_execution_workload(resolved_config=resolved,
        execution_source=_execution_source(), recipes=registry, require_mode=True)
    assert compiled.method == "grpo"
    assert compiled.document["execution_source"] == _execution_source().to_dict()
    with pytest.raises(RecipeNotRegistered):
        default_recipe_registry().resolve("grpo")


def test_checked_in_schemas_accept_actual_compilation_and_reject_extras():
    from pathlib import Path
    from jsonschema import Draft202012Validator
    root = Path(__file__).parents[2] / "schemas"
    raw = b'{"prompt":"Say OK"}\n'
    value = TrainingInputV1.from_dict(document())
    config = {"schema_version": "synaptic-grpo-config/v1", "method": "grpo",
        "execution": {"mode": "developer_integration"}, "training_input": bound(value, raw).to_dict(),
        "prepared_input": prepared(raw).to_dict()}
    for name, payload in (("hyperparameters", value.hyperparameters.to_dict()),
                          ("config", config), ("workload", compile_grpo_input(bound(value, raw), prepared(raw)).document)):
        schema = json.loads((root / f"synaptic-grpo-{name}-v1.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(payload)
        invalid = {**payload, "extra": "unsupported"}
        assert list(validator.iter_errors(invalid))


def test_default_import_does_not_load_grpo_or_ml():
    import subprocess
    import sys
    from pathlib import Path
    root = str(Path(__file__).parents[2])
    program = f'''
import sys, importlib.abc
sys.path.insert(0, {root!r})
class NoHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {{'torch','unsloth','transformers','datasets','modal','huggingface_hub'}}:
            raise AssertionError('unexpected heavy dependency')
sys.meta_path.insert(0, NoHeavy())
import tuner.training
registry = tuner.training.default_recipe_registry()
assert 'tuner.training.methods.grpo' not in sys.modules
assert 'Trainers.grpo.runtime_v1' not in sys.modules
from tuner.training import GRPORecipe
assert GRPORecipe.method == 'grpo'
assert 'Trainers.grpo.runtime_v1' not in sys.modules
'''
    subprocess.run([sys.executable, "-I", "-c", program], check=True, timeout=20,
                   capture_output=True)


def test_forged_prompt_semantic_identity_rejects_actual_admission():
    from dataclasses import replace
    raw = b'{"prompt":"Say OK","other":"different"}\n'
    identity = prepared(raw)
    source = bound(TrainingInputV1.from_dict(document()), raw)
    forged = replace(identity, ref="prepared://sha256/" + "c" * 64, revision="c" * 64)
    detached = source.to_dict(); detached["dataset"] = {"ref": forged.ref}
    with pytest.raises(ValueError, match="semantic identity"):
        admit_grpo_snapshot(TrainingInputV1.from_dict(detached), forged, raw)
