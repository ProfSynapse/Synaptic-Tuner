"""Contract proofs for the provider-neutral TrainingInputV1 document."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import get_type_hints

import pytest

from synaptic_tuner.api.v1.training_input import (
    SFTTrainingHyperparametersV1,
    TrainingArtifactRequirementsV1,
    TrainingDatasetInputV1,
    TrainingDurationV1,
    TrainingInputV1,
    TrainingMethodV1,
    TrainingModelInputV1,
)


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_NAMES = [
    "SFTTrainingHyperparametersV1",
    "TrainingArtifactRequirementsV1",
    "TrainingDatasetInputV1",
    "TrainingDurationV1",
    "TrainingInputV1",
    "TrainingMethodV1",
    "TrainingModelInputV1",
]
FORBIDDEN_FIELDS = (
    "provider", "provider_profile", "profile", "runtime", "image", "accelerator",
    "device", "environment", "credentials", "secrets", "path", "source_path",
    "artifact_path", "destination", "destination_ref", "bucket", "volume", "database",
    "sqlite", "persistence", "state",
)


def _document() -> dict[str, object]:
    return {
        "schema_version": "synaptic-training-input/v1",
        "method": "sft",
        "model": {
            "ref": "organization/model",
            "revision": "revision-1",
            "tokenizer_revision": "tokenizer-1",
        },
        "dataset": {"ref": "dataset://organization/corpus"},
        "hyperparameters": {
            "schema_version": "synaptic-sft-hyperparameters/v1",
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0002,
            "duration": {"max_steps": 100, "num_epochs": None},
            "max_seq_length": 2048,
            "seed": 42,
            "save_steps": 25,
            "save_total_limit": 2,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["k_proj", "q_proj", "v_proj"],
            "use_dora": False,
            "use_rslora": True,
            "init_lora_weights": True,
            "split_dataset": False,
        },
        "artifacts": {
            "required_kinds": ["final_model", "training_lineage"],
            "retain_checkpoints": True,
        },
    }


def _input() -> TrainingInputV1:
    return TrainingInputV1.from_dict(_document())


def test_valid_document_is_exact_immutable_and_canonical() -> None:
    value = _input()
    assert type(value) is TrainingInputV1
    assert value.method is TrainingMethodV1.SFT
    assert value.to_dict() == _document()
    expected = json.dumps(
        _document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    assert value.canonical_json() == expected
    assert value.canonical_bytes() == expected.encode("utf-8")
    assert value.input_digest() == hashlib.sha256(
        b"synaptic-training-input/v1\0" + expected.encode("utf-8")
    ).hexdigest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.schema_version = "changed"  # type: ignore[misc]
    assert not hasattr(value, "__dict__")
    assert tuple(item.value for item in TrainingMethodV1) == ("sft",)
    assert "schema_version" not in {
        field.name for field in dataclasses.fields(SFTTrainingHyperparametersV1)
    }


def test_pretty_json_round_trip_is_stable_and_snapshots_arrays() -> None:
    document = _document()
    target_modules = document["hyperparameters"]["lora_target_modules"]  # type: ignore[index]
    required_kinds = document["artifacts"]["required_kinds"]  # type: ignore[index]
    value = TrainingInputV1.from_json(json.dumps(document, indent=2))
    target_modules.append("z_proj")  # type: ignore[union-attr]
    required_kinds.append("tokenizer")  # type: ignore[union-attr]
    assert value.hyperparameters.lora_target_modules == ("k_proj", "q_proj", "v_proj")
    assert value.artifacts.required_kinds == ("final_model", "training_lineage")
    assert TrainingInputV1.from_json(value.canonical_json()) == value
    assert TrainingInputV1.from_dict(value.to_dict()) == value


def test_duration_supports_exact_xor_and_emits_both_keys() -> None:
    steps = TrainingDurationV1(10_000_000, None)
    epochs = TrainingDurationV1(None, 1000)
    assert steps.to_dict() == {"max_steps": 10_000_000, "num_epochs": None}
    assert epochs.to_dict() == {"max_steps": None, "num_epochs": 1000.0}
    for values in ((None, None), (1, 1.0)):
        with pytest.raises(ValueError):
            TrainingDurationV1(*values)
    for values in ((0, None), (10_000_001, None), (None, 0.0), (None, 1000.1)):
        with pytest.raises(ValueError):
            TrainingDurationV1(*values)
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            TrainingDurationV1(None, value)


@pytest.mark.parametrize(
    ("path", "maximum"),
    [
        (("batch_size",), 4096),
        (("gradient_accumulation_steps",), 4096),
        (("max_seq_length",), 1_048_576),
        (("save_steps",), 10_000_000),
        (("save_total_limit",), 10_000),
        (("lora_rank",), 4096),
        (("lora_alpha",), 65_536),
        (("seed",), 4_294_967_295),
    ],
)
def test_integer_exact_maxima_and_one_over_apply_to_direct_and_dict_inputs(
    path: tuple[str], maximum: int
) -> None:
    field = path[0]
    hyperparameters = _input().hyperparameters
    assert getattr(dataclasses.replace(hyperparameters, **{field: maximum}), field) == maximum
    with pytest.raises(ValueError):
        dataclasses.replace(hyperparameters, **{field: maximum + 1})
    document = _document()
    document["hyperparameters"][field] = maximum + 1  # type: ignore[index]
    with pytest.raises(ValueError):
        TrainingInputV1.from_dict(document)


@pytest.mark.parametrize("value", [True, False, 1.0, "1"])
def test_integer_fields_reject_bool_and_non_exact_ints(value: object) -> None:
    for field in (
        "batch_size", "gradient_accumulation_steps", "max_seq_length", "seed",
        "save_steps", "save_total_limit", "lora_rank", "lora_alpha",
    ):
        document = _document()
        document["hyperparameters"][field] = value  # type: ignore[index]
        with pytest.raises(TypeError):
            TrainingInputV1.from_dict(document)


def test_integer_field_minimums_are_enforced() -> None:
    for field in (
        "batch_size", "gradient_accumulation_steps", "max_seq_length", "save_steps",
        "save_total_limit", "lora_rank", "lora_alpha",
    ):
        document = _document()
        document["hyperparameters"][field] = 0  # type: ignore[index]
        with pytest.raises(ValueError):
            TrainingInputV1.from_dict(document)
    document = _document()
    document["hyperparameters"]["seed"] = -1  # type: ignore[index]
    with pytest.raises(ValueError):
        TrainingInputV1.from_dict(document)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_learning_rate_requires_finite_positive_value(value: float) -> None:
    document = _document()
    document["hyperparameters"]["learning_rate"] = value  # type: ignore[index]
    with pytest.raises(ValueError):
        TrainingInputV1.from_dict(document)


def test_learning_rate_exact_upper_bound() -> None:
    assert dataclasses.replace(_input().hyperparameters, learning_rate=1).learning_rate == 1.0
    with pytest.raises(ValueError):
        dataclasses.replace(_input().hyperparameters, learning_rate=1.0000001)


@pytest.mark.parametrize("value", [-0.1, 1.0, math.nan, math.inf])
def test_dropout_requires_closed_finite_range(value: float) -> None:
    document = _document()
    document["hyperparameters"]["lora_dropout"] = value  # type: ignore[index]
    with pytest.raises(ValueError):
        TrainingInputV1.from_dict(document)


@pytest.mark.parametrize(
    "field", ["learning_rate", "num_epochs", "lora_dropout"]
)
def test_numeric_conversion_overflow_is_closed_across_all_ingress_paths(field: str) -> None:
    huge = 10**400
    if field == "num_epochs":
        with pytest.raises(ValueError) as direct:
            TrainingDurationV1(None, huge)
        document = _document()
        document["hyperparameters"]["duration"] = {"max_steps": None, "num_epochs": huge}  # type: ignore[index]
    else:
        with pytest.raises(ValueError) as direct:
            dataclasses.replace(_input().hyperparameters, **{field: huge})
        document = _document()
        document["hyperparameters"][field] = huge  # type: ignore[index]
    assert "10" not in str(direct.value)
    with pytest.raises(ValueError):
        TrainingInputV1.from_dict(document)
    with pytest.raises(ValueError):
        TrainingInputV1.from_json(json.dumps(document))


@pytest.mark.parametrize(
    "field", ["use_dora", "use_rslora", "init_lora_weights", "split_dataset"]
)
def test_hyperparameter_booleans_are_exact(field: str) -> None:
    document = _document()
    document["hyperparameters"][field] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        TrainingInputV1.from_dict(document)


@pytest.mark.parametrize(
    ("container", "field", "replacement"),
    [
        ("hyperparameters", "lora_target_modules", ["q_proj", "k_proj"]),
        ("hyperparameters", "lora_target_modules", ["q_proj", "q_proj"]),
        ("artifacts", "required_kinds", ["training_lineage", "final_model"]),
        ("artifacts", "required_kinds", ["final_model", "final_model"]),
    ],
)
def test_sequence_order_and_uniqueness_are_required(
    container: str, field: str, replacement: list[str]
) -> None:
    document = _document()
    document[container][field] = replacement  # type: ignore[index]
    with pytest.raises(ValueError):
        TrainingInputV1.from_dict(document)


def test_sequence_exact_bounds_are_enforced() -> None:
    targets = tuple(f"module-{index:03d}" for index in range(256))
    kinds = tuple(f"kind-{index:02d}" for index in range(64))
    assert SFTTrainingHyperparametersV1.from_dict(
        {**_document()["hyperparameters"], "lora_target_modules": targets}  # type: ignore[arg-type]
    ).lora_target_modules == targets
    assert TrainingArtifactRequirementsV1(kinds, False).required_kinds == kinds
    with pytest.raises(ValueError):
        TrainingArtifactRequirementsV1(tuple(f"kind-{index:03d}" for index in range(65)), False)
    with pytest.raises(ValueError):
        SFTTrainingHyperparametersV1.from_dict(
            {**_document()["hyperparameters"], "lora_target_modules": targets + ("z",)}  # type: ignore[arg-type]
        )
    assert TrainingArtifactRequirementsV1(("x" * 128,), False).required_kinds
    with pytest.raises(ValueError):
        TrainingArtifactRequirementsV1(("x" * 129,), False)
    assert TrainingArtifactRequirementsV1(("é" * 64,), False).required_kinds
    with pytest.raises(ValueError):
        TrainingArtifactRequirementsV1(("é" * 65,), False)
    for empty in ((), []):
        with pytest.raises(ValueError):
            TrainingArtifactRequirementsV1(empty, False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        "/absolute/path", "C:\\absolute\\path", "\\\\server\\share",
        "//server/share", "file:///tmp/data", "https://user:password@example.test/repo",
        "https://example.test/repo?token=secret", "dataset://repo#api_key=secret",
        "https://example.test/repo?X-Amz-Credential=secret",
        "./relative", "../relative", "~/relative", ".\\relative", "..\\relative",
        "~\\relative", "namespace/./item", "namespace/../item", "namespace/~/item",
        "namespace\\item", "namespace/%2e/item", "namespace/%2F/item",
        "namespace/%5c/item", "namespace/%7E/item", "namespace/%252e/item",
        "project://organization/data#fragment", "project://organization/data#",
        "C:relative", "%43%3Arelative", "C%3A/relative", "f%69le:///tmp/data",
        "project://organization%40evil/data", "project%3A///data",
        "project:///data", "project://@organization/data",
        "project://organization/data%23fragment", "namespace/%25/item",
        "namespace/%2525/item", "namespace/%", "namespace/%G0/item",
        "namespace/%C3/item", "dataset://organization/corpus?name=%2F",
        "dataset://organization/corpus?name=%ZZ",
        "dataset://organization/corpus?%74oken=opaque",
    ],
)
def test_refs_reject_filesystem_and_inline_credential_forms(value: str) -> None:
    with pytest.raises(ValueError) as caught:
        TrainingDatasetInputV1(value)
    assert "password" not in str(caught.value)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        "organization/model", "namespace/child", "project://organization/dataset",
        "dataset://organization/corpus?split=train",
        "dataset://organization/corpus?tokenizer=name",
        "dataset://organization/corpus?secretariat=name",
        "dataset://organization/corpus?session_name=name",
        "dataset://organization/corpus?name=token=secret",
    ],
)
def test_refs_accept_forward_namespaces_and_opaque_noncredential_queries(value: str) -> None:
    assert TrainingDatasetInputV1(value).ref == value


@pytest.mark.parametrize(
    "query",
    [
        "", "=value", "%ZZ=value", "%FF=value", "name=one&NAME=two",
        "na%6de=one&NAME=two",
    ],
)
def test_query_keys_are_nonempty_well_escaped_utf8_and_casefold_unique(query: str) -> None:
    with pytest.raises(ValueError):
        TrainingDatasetInputV1(f"dataset://organization/corpus?{query}")


@pytest.mark.parametrize(
    "key",
    [
        "key", "token", "access-token", "refresh_token", "id.token", "apikey",
        "secret", "client-secret", "password", "passwd", "pwd", "authorization",
        "auth", "bearer", "signature", "sig", "credential", "credentials",
        "session", "session-id", "session_token", "cookie", "deployment-token",
        "signing_secret", "db.password", "service_passwd", "vendor_api_key",
        "request_signature", "cloud_credential", "cloud.credentials",
    ],
)
def test_query_credential_keys_and_suffixes_are_rejected(key: str) -> None:
    with pytest.raises(ValueError) as caught:
        TrainingDatasetInputV1(f"dataset://organization/corpus?{key}=opaque")
    assert "opaque" not in str(caught.value)


@pytest.mark.parametrize(
    "key",
    [
        "access-key-id", "prefix.access_key.suffix", "myaccesskeyidentifier",
        "private-key", "prefix_private.key_suffix", "myprivatekeyidentifier",
    ],
)
def test_query_component_and_compact_access_private_key_forms_are_rejected(
    key: str,
) -> None:
    with pytest.raises(ValueError):
        TrainingDatasetInputV1(f"dataset://organization/corpus?{key}=opaque")


def test_one_round_projection_is_stored_and_digest_canonical() -> None:
    literal_document = _document()
    encoded_document = _document()
    encoded_document["model"] = {
        "ref": "organizati%6Fn/model",
        "revision": "revisi%6Fn-1",
        "tokenizer_revision": "tokenizer-1",
    }
    encoded_document["dataset"] = {
        "ref": "dataset://organization/corpus?spl%69t=tr%61in"
    }
    literal_document["dataset"] = {
        "ref": "dataset://organization/corpus?split=train"
    }
    literal = TrainingInputV1.from_dict(literal_document)
    encoded = TrainingInputV1.from_dict(encoded_document)
    assert encoded == literal
    assert TrainingInputV1.from_json(json.dumps(encoded_document)) == literal
    assert encoded.model.ref == "organization/model"
    assert encoded.dataset.ref == "dataset://organization/corpus?split=train"
    assert encoded.canonical_bytes() == literal.canonical_bytes()
    assert encoded.input_digest() == literal.input_digest()


def test_projection_combines_escaped_bytes_and_unescaped_utf8_strictly() -> None:
    assert TrainingDatasetInputV1("dataset://org/caf%C3%A9").ref == "dataset://org/café"
    assert TrainingDatasetInputV1("dataset://org/café").ref == "dataset://org/café"


@pytest.mark.parametrize("value", ["", " edge", "edge ", "bad\x00ref", "e\u0301"])
def test_strings_require_nonempty_nfc_without_edge_space_or_controls(value: str) -> None:
    with pytest.raises(ValueError):
        TrainingDatasetInputV1(value)


def test_exact_nested_types_reject_subclasses_and_lookalikes() -> None:
    class ModelSubclass(TrainingModelInputV1):
        pass

    valid = _input()
    with pytest.raises(TypeError):
        TrainingInputV1(
            valid.schema_version,
            valid.method,
            ModelSubclass("org/model", "rev", "tokenizer"),
            valid.dataset,
            valid.hyperparameters,
            valid.artifacts,
        )
    with pytest.raises(TypeError):
        TrainingInputV1(
            valid.schema_version, "sft", valid.model, valid.dataset,  # type: ignore[arg-type]
            valid.hyperparameters, valid.artifacts,
        )


def test_from_dict_rejects_mapping_and_dict_subclasses_at_every_nesting() -> None:
    class DictSubclass(dict):
        pass

    class MappingProxy(Mapping):
        def __init__(self, value):
            self._value = value

        def __getitem__(self, key):
            return self._value[key]

        def __iter__(self):
            return iter(self._value)

        def __len__(self):
            return len(self._value)

    with pytest.raises(TypeError):
        TrainingInputV1.from_dict(DictSubclass(_document()))
    with pytest.raises(TypeError):
        TrainingInputV1.from_dict(MappingProxy(_document()))
    for path in (
        ("model",), ("dataset",), ("hyperparameters",),
        ("hyperparameters", "duration"), ("artifacts",),
    ):
        document = _document()
        parent = document
        for part in path[:-1]:
            parent = parent[part]  # type: ignore[assignment,index]
        field = path[-1]
        parent[field] = DictSubclass(parent[field])  # type: ignore[index]
        with pytest.raises(TypeError):
            TrainingInputV1.from_dict(document)


def test_all_from_dict_annotations_are_exact_builtin_dicts() -> None:
    for contract in (
        TrainingModelInputV1,
        TrainingDatasetInputV1,
        TrainingDurationV1,
        SFTTrainingHyperparametersV1,
        TrainingArtifactRequirementsV1,
        TrainingInputV1,
    ):
        assert get_type_hints(contract.from_dict)["value"] == dict[str, object]


def test_sequence_subclasses_are_rejected_before_tuple_snapshot() -> None:
    class ListSubclass(list):
        pass

    document = _document()
    document["hyperparameters"]["lora_target_modules"] = ListSubclass(["q_proj"])  # type: ignore[index]
    with pytest.raises(TypeError):
        TrainingInputV1.from_dict(document)
    document = _document()
    document["artifacts"]["required_kinds"] = ListSubclass(["final_model"])  # type: ignore[index]
    with pytest.raises(TypeError):
        TrainingInputV1.from_dict(document)


def test_every_missing_field_and_unknown_fields_fail_at_every_nesting() -> None:
    paths = (
        (), ("model",), ("dataset",), ("hyperparameters",),
        ("hyperparameters", "duration"), ("artifacts",),
    )
    for path in paths:
        template = _document()
        target = template
        for part in path:
            target = target[part]  # type: ignore[assignment,index]
        for field in tuple(target):  # type: ignore[arg-type]
            document = _document()
            missing_target = document
            for part in path:
                missing_target = missing_target[part]  # type: ignore[assignment,index]
            del missing_target[field]  # type: ignore[index]
            with pytest.raises(ValueError):
                TrainingInputV1.from_dict(document)
        document = _document()
        unknown_target = document
        for part in path:
            unknown_target = unknown_target[part]  # type: ignore[assignment,index]
        unknown_target["unknown"] = True  # type: ignore[index]
        with pytest.raises(ValueError):
            TrainingInputV1.from_dict(document)


@pytest.mark.parametrize("field", FORBIDDEN_FIELDS)
def test_provider_runtime_storage_and_secret_fields_are_forbidden(field: str) -> None:
    document = _document()
    document[field] = "forbidden"
    with pytest.raises(ValueError):
        TrainingInputV1.from_dict(document)


def test_schema_method_and_duplicate_json_keys_fail_closed() -> None:
    for path, replacement in (
        (("schema_version",), "synaptic-training-input/v2"),
        (("method",), "grpo"),
        (("hyperparameters", "schema_version"), "synaptic-sft-hyperparameters/v2"),
    ):
        document = _document()
        if len(path) == 1:
            document[path[0]] = replacement
        else:
            document[path[0]][path[1]] = replacement  # type: ignore[index]
        with pytest.raises(ValueError):
            TrainingInputV1.from_dict(document)
    duplicate = _input().canonical_json().replace(
        '"method":"sft"', '"method":"sft","method":"sft"'
    )
    with pytest.raises(ValueError, match="malformed"):
        TrainingInputV1.from_json(duplicate)


def test_json_size_nonfinite_and_root_type_are_closed() -> None:
    canonical = _input().canonical_json()
    exact_limit = canonical + (" " * (65536 - len(canonical.encode("utf-8"))))
    assert TrainingInputV1.from_json(exact_limit) == _input()
    oversized = _input().canonical_json() + (" " * 65536)
    with pytest.raises(ValueError, match="size"):
        TrainingInputV1.from_json(oversized)
    with pytest.raises(ValueError, match="malformed"):
        TrainingInputV1.from_json(_input().canonical_json().replace("0.0002", "NaN"))
    with pytest.raises(TypeError):
        TrainingInputV1.from_json("[]")
    with pytest.raises(TypeError):
        TrainingInputV1.from_json(1)  # type: ignore[arg-type]


def test_public_exports_are_exact_lazy_identities() -> None:
    import synaptic_tuner.api.v1 as api
    import synaptic_tuner.api.v1.training_input as module

    assert module.__all__ == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert name in api.__all__
        assert getattr(api, name) is getattr(module, name)


def test_training_input_import_has_no_forbidden_dependency_consequence() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
before = set(sys.modules)
import synaptic_tuner.api.v1.training_input
after = set(sys.modules) - before
forbidden = sorted(name for name in after if name == 'tuner' or name.startswith(('tuner.', 'synaptic_host', 'docker', 'modal', 'huggingface_hub', 'runpod', 'sqlite3', 'pathlib', 'os', 'subprocess')))
print(json.dumps(forbidden))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    assert json.loads(completed.stdout) == []


def test_training_input_source_imports_only_stdlib_and_contract_primitives() -> None:
    path = ROOT / "synaptic_tuner/api/v1/training_input.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    )
    assert imports <= {
        "__future__", "collections", "dataclasses", "enum", "json", "math", "re",
        "unicodedata", "urllib",
    }
    relative = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }
    assert relative == {"_contract"}
