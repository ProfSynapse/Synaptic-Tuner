"""CPU conformance for conversation SFT; no release/GPU qualification."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from shared.sft_preprocessing import (
    CONVERSATION_SCHEMA_VERSION, detect_sft_record_format,
    is_authoritative_preassigned_sft_record, materialize_sft_example,
)
from Trainers.sft.runtime_v1 import _append_sft_arguments
from tuner.project.execution_source import ExecutionSourceV1
from tuner.runtime.offline_sft_worker import _validate_trainer_arguments, OfflineSFTWorkerError
from tuner.runtime.verification import _expected_dataset_path, _expected_trainer_argv
from tuner.training.contracts import CanonicalDocument
from tuner.training.methods.sft import compile_sft_workload
from tests.runtime.test_artifact_verification import _runtime_workload, _fixture, _lineage, _verify_current


ROW_SCHEMA = CONVERSATION_SCHEMA_VERSION


def row(split="train", index=1):
    return {"schema_version": ROW_SCHEMA, "format": "messages", "row_id": "row-" + f"{index:064x}",
            "source_row_number": index, "target_message_index": 4, "group_id": "group-" + str(index),
            "split": split, "messages": [
                {"role": "system", "content": "configured instruction"},
                {"role": "user", "content": "first prompt"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "last prompt"},
                {"role": "assistant", "content": "only this target"},
            ]}


class Tokenizer:
    eos_token_id = 999
    all_special_ids = [999]

    def __init__(self):
        self.renders = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        self.renders.append((deepcopy(messages), add_generation_prompt, deepcopy(kwargs)))
        text = "|".join(message["role"] + ":" + message["content"] for message in messages)
        return text + ("|assistant:anchor" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def workload(schema=ROW_SCHEMA, size=400):
    previous = _runtime_workload()
    config = previous.document["configuration"]["document"]
    config["dataset"] = {"ref": "prepared://sha256/" + "a" * 64, "revision": "a" * 64,
                         "content_digest": "d" * 64, "size_bytes": size, "format": schema}
    config["sft"].update(dataset_format="messages", completion_only_loss=True,
                        assistant_only_loss=False, use_preassigned_splits=True,
                        prompt_render="prompt_completion", packing=False,
                        require_memory_efficient_loss=True, chat_template_kwargs={"enable_thinking": False})
    return compile_sft_workload(resolved_config=CanonicalDocument.from_mapping(config),
                               execution_source=ExecutionSourceV1.from_dict(previous.document["execution_source"]))


def test_final_target_labels_preserve_ordered_prefix_and_template_kwargs():
    tokenizer = Tokenizer()
    original = row()
    kwargs = {"custom_template_option": False}
    result = materialize_sft_example(tokenizer=tokenizer, record=original, max_seq_length=1024,
        assistant_only_loss=True, prompt_render="prompt_completion", chat_template_kwargs=kwargs)
    assert is_authoritative_preassigned_sft_record(original) is True
    assert detect_sft_record_format(original) == "messages"
    intended = original["messages"][:-1]
    assert tokenizer.renders[0] == (intended, True, kwargs)
    assert original == row() and kwargs == {"custom_template_option": False}
    prompt = "|".join(message["role"] + ":" + message["content"] for message in intended) + "|assistant:anchor"
    target = tokenizer.encode(original["messages"][-1]["content"]) + [tokenizer.eos_token_id]
    assert result.input_ids == tokenizer.encode(prompt) + target
    assert result.labels == [-100] * len(prompt) + target
    assert result.masked_prefix_length == len(prompt)
    assert result.truncation_applied is False


@pytest.mark.parametrize("case", ["format", "missing_field", "bool_index", "target_index", "structured_content", "tool_role", "late_system", "empty_target", "extra_field"])
def test_malformed_new_rows_reject_at_actual_execution_parser(case):
    value = row()
    if case == "format":
        value["format"] = "legacy"
    elif case == "missing_field":
        del value["group_id"]
    elif case == "bool_index":
        value["source_row_number"] = True
    elif case == "target_index":
        value["target_message_index"] = 3
    elif case == "structured_content":
        value["messages"][-1]["content"] = {"text": "target"}
    elif case == "tool_role":
        value["messages"][1]["role"] = "tool"
    elif case == "late_system":
        value["messages"][1]["role"] = "system"
    elif case == "empty_target":
        value["messages"][-1]["content"] = ""
    else:
        value["unadmitted"] = "value"
    with pytest.raises(ValueError):
        materialize_sft_example(tokenizer=Tokenizer(), record=value, max_seq_length=1024,
                               assistant_only_loss=True, prompt_render="prompt_completion")


@pytest.mark.parametrize("case", ["truncation", "no_target_tokens", "full_render", "full_loss"])
def test_new_rows_never_truncate_or_fall_back_to_wrong_loss(case):
    tokenizer = Tokenizer()
    value = row()
    kwargs = dict(max_seq_length=1024, assistant_only_loss=True, prompt_render="prompt_completion")
    if case == "truncation":
        kwargs["max_seq_length"] = 4
    elif case == "no_target_tokens":
        original_encode = tokenizer.encode
        tokenizer.encode = lambda text, add_special_tokens=False: [] if text == "only this target" else original_encode(text, add_special_tokens)
    elif case == "full_render":
        kwargs["prompt_render"] = "full_conversation"
    else:
        kwargs["assistant_only_loss"] = False
    with pytest.raises(ValueError):
        materialize_sft_example(tokenizer=tokenizer, record=value, **kwargs)


@pytest.mark.parametrize("schema", ["syntunia-sft-row/v1", "syntunia-sft-row/v2", ROW_SCHEMA])
def test_native_prepared_paths_keep_two_mib_limit_and_exact_fd_identity(schema):
    dataset = {"ref": "prepared://sha256/" + "a" * 64, "revision": "a" * 64,
               "content_digest": "b" * 64, "size_bytes": 2 * 1024 * 1024, "format": schema}
    assert _expected_dataset_path(dataset, {}, "/proc/self/fd/9") == "/proc/self/fd/9"
    assert _expected_dataset_path(dataset, {}, "/private/path") is None
    dataset["size_bytes"] += 1
    assert _expected_dataset_path(dataset, {}, "/proc/self/fd/9") is None


@pytest.mark.parametrize("schema", ["syntunia-sft-row/v2", ROW_SCHEMA])
def test_offline_argv_schema_and_kwargs_match_native_verifier(schema):
    compiled = workload(schema)
    config = compiled.document["configuration"]["document"]
    args = []
    _append_sft_arguments(args, config["sft"], config["model"], dataset_schema=schema)
    assert args[args.index("--runtime-v1-dataset-schema") + 1] == schema
    assert args[args.index("--chat-template-kwargs") + 1] == '{"enable_thinking":false}'
    expected = _expected_trainer_argv("/usr/bin/python", compiled.document["execution_source"]["runtime"]["roots"],
                                     "/proc/self/fd/9", config, compiled)
    assert expected[expected.index("--batch-size"):] == args
    _validate_trainer_arguments(expected[4:])
    bad = list(expected[4:])
    bad[bad.index("--runtime-v1-dataset-schema") + 1] = "unreviewed/v1"
    with pytest.raises(OfflineSFTWorkerError):
        _validate_trainer_arguments(bad)


@pytest.mark.parametrize("field,value", [("completion_only_loss", False), ("assistant_only_loss", True),
    ("use_preassigned_splits", False), ("split_dataset", True), ("prompt_render", "full_conversation"),
    ("packing", True), ("require_memory_efficient_loss", False)])
def test_compiler_requires_exact_new_format_controls(field, value):
    compiled = workload()
    config = compiled.document["configuration"]["document"]
    config["sft"][field] = value
    with pytest.raises(ValueError, match="conversation SFT"):
        compile_sft_workload(resolved_config=CanonicalDocument.from_mapping(config),
            execution_source=ExecutionSourceV1.from_dict(compiled.document["execution_source"]))


def test_legacy_v2_default_argv_is_unchanged_when_schema_argument_absent():
    compiled = workload("syntunia-sft-row/v2")
    config = compiled.document["configuration"]["document"]
    first, explicit = [], []
    _append_sft_arguments(first, config["sft"], config["model"])
    _append_sft_arguments(explicit, config["sft"], config["model"], dataset_schema="syntunia-sft-row/v2")
    assert first == explicit


def test_new_compiler_refuses_large_executable_inputs():
    with pytest.raises(ValueError, match="2MiB"):
        workload(size=2 * 1024 * 1024 + 1)


@pytest.mark.parametrize("schema", ["syntunia-sft-row/v2", ROW_SCHEMA])
def test_full_native_verification_of_prepared_messages_uses_admitted_schema(schema):
    compiled = workload(schema)
    config = compiled.document["configuration"]["document"]
    previous, inventory, reader, values = _fixture()
    lineage = json.loads(_lineage(compiled))
    execution = lineage["execution_evidence"]
    dataset_path = "/proc/self/fd/9"
    execution["dataset"]["resolved_path"] = dataset_path
    argv = execution["argv"][:execution["argv"].index("--batch-size")]
    argv[argv.index("--local-file") + 1] = dataset_path
    argv[argv.index("--runtime-v1-dataset-revision") + 1] = config["dataset"]["revision"]
    _append_sft_arguments(argv, config["sft"], config["model"], dataset_schema=schema)
    execution["argv"] = argv
    projection = lineage["trainer_lineage"]["synaptic_runtime_projection"]
    projection["dataset"].update(resolved_path=dataset_path, schema_version=schema, format="messages")
    projection["training"].update(completion_only_loss=True, assistant_only_loss=False,
        use_preassigned_splits=True, prompt_render="prompt_completion", packing=False, require_memory_efficient_loss=True)
    lineage["trainer_lineage"]["dataset"]["source"] = dataset_path
    import hashlib
    lineage["execution_evidence_sha256"] = hashlib.sha256(json.dumps(execution, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    values["workload.json"] = compiled.canonical_bytes
    values["training_lineage.json"] = json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
    assert _verify_current(compiled, inventory, reader, values).success is True
    # A self-consistent row schema claim cannot disagree with admitted workload.
    projection["dataset"]["schema_version"] = "syntunia-sft-row/v1"
    values["training_lineage.json"] = json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
    assert _verify_current(compiled, inventory, reader, values).success is False


@pytest.mark.parametrize("raw", ['{"tokenize":false}', '{"x":1,"x":2}', '{"x":NaN}',
                                  '{"x":1e309}', '{}', '[]', '{"x":"' + 'a' * 5000 + '"}'])
def test_offline_template_kwargs_fail_closed(raw):
    compiled = workload()
    config = compiled.document["configuration"]["document"]
    args = _expected_trainer_argv("/usr/bin/python", compiled.document["execution_source"]["runtime"]["roots"],
                                 "/proc/self/fd/9", config, compiled)[4:]
    args[args.index("--chat-template-kwargs") + 1] = raw
    with pytest.raises(OfflineSFTWorkerError, match="chat template kwargs"):
        _validate_trainer_arguments(args)


def test_duplicate_offline_template_flag_never_replaces_bound_value():
    compiled = workload()
    config = compiled.document["configuration"]["document"]
    args = _expected_trainer_argv("/usr/bin/python", compiled.document["execution_source"]["runtime"]["roots"],
                                 "/proc/self/fd/9", config, compiled)[4:]
    args.extend(("--chat-template-kwargs", '{"another_option":false}'))
    with pytest.raises(OfflineSFTWorkerError, match="duplicated"):
        _validate_trainer_arguments(args)


def test_packaged_compiler_supports_format_without_promoting_old_release():
    from tests.training.test_packaged_execution_material import packaged_fixture
    from tuner.training.packaged_compilation import compile_packaged_sft_workload
    from tuner.runtime.releases import PackagedExecutionBindingV1
    from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
    public_input, components = packaged_fixture()
    old_config = components.resolved_config
    old_binding = components.execution_source
    from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1
    release = PackagedTrainingRuntimeReleaseV1.from_dict(components.execution_context.to_dict()["runtime_release"])
    document = old_config.to_dict()
    document["dataset"]["format"] = ROW_SCHEMA
    document["sft"]["require_memory_efficient_loss"] = True
    config = CanonicalDocument.from_mapping(document)
    compiled = compile_packaged_sft_workload(resolved_config=config)
    assert compiled.document["configuration"]["document"]["dataset"]["format"] == ROW_SCHEMA
    identity = PreparedTrainingInputIdentity.from_dict(document["dataset"])
    assert ROW_SCHEMA not in release.compatible_dataset_formats
    from dataclasses import replace
    from tests.runtime.test_packaged_runtime_releases import _provider
    from tuner.training.packaged_compilation import packaged_configuration_digest
    from tuner.training.packaged_boundary import validate_packaged_material
    from tuner.training.contracts import TrainingRequest
    binding = PackagedExecutionBindingV1.build(run_ref="unqualified-conversation", runtime_release=release,
        provider_runtime_binding=_provider(release), **identity.execution_binding_fields(),
        workload_digest=compiled.fingerprint, configuration_digest=packaged_configuration_digest(config),
        artifact_policy_digest=old_binding.artifact_policy_digest)
    admitted = replace(components, execution_source=binding, resolved_config=config)
    public = public_input.to_dict()
    public["hyperparameters"]["require_memory_efficient_loss"] = True
    with pytest.raises(ValueError, match="runtime release does not admit"):
        validate_packaged_material(request=TrainingRequest(CanonicalDocument.from_mapping(public)), material=admitted)


def test_offline_kwargs_integer_domain_matches_existing_public_syntax():
    from synaptic_tuner.api.v1.training_input import validate_chat_template_kwargs
    from tuner.runtime.offline_sft_worker import _validate_chat_template_argument
    value = {"custom_limit": 2**64}
    assert validate_chat_template_kwargs(value) == value
    _validate_chat_template_argument(json.dumps(value, separators=(",", ":")))
