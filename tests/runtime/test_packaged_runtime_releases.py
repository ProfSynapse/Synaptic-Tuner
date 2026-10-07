from __future__ import annotations

import json
import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import jsonschema
import pytest

from tuner.runtime.releases import (
    PackagedExecutionBindingV1,
    PackagedTrainingRuntimeReleaseV1,
    ProviderRuntimeBindingV1,
)


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_REVISION = "1" * 40
_VECTOR_PATH = Path("tests/runtime/fixtures/packaged_runtime_release_vectors.json")


def _release_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "release_ref": "runtime:qwen35-sft-v1",
        "package_name": "synaptic-tuner",
        "package_version": "1.2.3",
        "package_digest": _A,
        "source_provenance_digest": _B,
        "worker_entrypoint": "tuner.runtime.packaged_worker:main",
        "worker_closure_digest": _C,
        "image_ref": f"ghcr.io/synaptic-labs/tuner@sha256:{_D}",
        "image_digest": _D,
        "python_implementation": "cpython",
        "python_version": "3.12.3",
        "python_executable": "/usr/local/bin/python",
        "python_executable_digest": _E,
        "installed_distributions_digest": _A,
        "installed_distribution_count": 327,
        "platform_system": "linux",
        "platform_machine": "x86_64",
        "cuda_version": "12.4.1",
        "runtime_facts": {"compute": {"dtype": "bfloat16"}, "gpu": True},
        "compatible_methods": ("sft",),
        "compatible_models": (("Qwen/Qwen3.5-4B", _REVISION),),
        "compatible_dataset_formats": ("syntunia-sft-row/v2",),
        "workload_schema": "synaptic-sft-workload/v1",
        "prepared_input_schema": "synaptic-prepared-training-input/v1",
        "artifact_contract_schema": "synaptic-sft-artifacts/v1",
    }
    values.update(changes)
    return values


def _release(**changes: object) -> PackagedTrainingRuntimeReleaseV1:
    return PackagedTrainingRuntimeReleaseV1.build(**_release_values(**changes))


def _provider(
    release: PackagedTrainingRuntimeReleaseV1 | None = None,
) -> ProviderRuntimeBindingV1:
    return ProviderRuntimeBindingV1.build(
        provider_ref="provider:fixture",
        runtime_release=release or _release(),
        provider_facts_schema="fixture-runtime-facts/v1",
        provider_facts_digest=_E,
    )


def _execution(
    release: PackagedTrainingRuntimeReleaseV1 | None = None,
    provider: ProviderRuntimeBindingV1 | None = None,
) -> PackagedExecutionBindingV1:
    selected = release or _release()
    return PackagedExecutionBindingV1.build(
        run_ref="run:fiction-001",
        runtime_release=selected,
        provider_runtime_binding=provider or _provider(selected),
        prepared_input_ref=f"prepared://sha256/{_A}",
        prepared_input_revision=_A,
        prepared_input_content_digest=_E,
        prepared_input_size_bytes=10_513_545,
        prepared_input_format="syntunia-sft-row/v2",
        workload_digest=_B,
        configuration_digest=_C,
        artifact_policy_digest=_D,
    )


@pytest.mark.parametrize(
    ("factory", "parser"),
    (
        (_release, PackagedTrainingRuntimeReleaseV1.from_json),
        (_provider, ProviderRuntimeBindingV1.from_json),
        (_execution, PackagedExecutionBindingV1.from_json),
    ),
)
def test_contracts_round_trip_canonical_json_deterministically(factory, parser) -> None:
    value = factory()
    rebuilt = parser(value.canonical_json())
    assert rebuilt == value
    assert rebuilt.canonical_bytes() == value.canonical_bytes()


def test_release_digest_is_independent_of_input_mapping_order() -> None:
    first = _release(runtime_facts={"alpha": 1, "nested": {"x": True, "y": "2.5"}})
    second = _release(runtime_facts={"nested": {"y": "2.5", "x": True}, "alpha": 1})
    assert first.manifest_digest == second.manifest_digest
    assert first.canonical_bytes() == second.canonical_bytes()


def test_checked_in_canonical_json_and_domain_digest_vectors() -> None:
    vectors = json.loads(_VECTOR_PATH.read_text(encoding="utf-8"))
    assert vectors["unicode_key_policy"] == {
        "input_key": "café", "admission": "rejected_ascii_keys_only"
    }
    with pytest.raises(ValueError, match="ASCII identifiers"):
        _release(runtime_facts={vectors["unicode_key_policy"]["input_key"]: "x"})
    release = _release(
        runtime_facts={
            "bool": True,
            "maximum": 9_007_199_254_740_991,
            "minimum": -9_007_199_254_740_991,
            "nested": {"array": [None, "café", "雪", {"flag": False}]},
        }
    )
    provider = _provider(release)
    execution = _execution(release, provider)
    values = {
        "release": (
            release,
            "manifest_digest",
            "synaptic-packaged-training-runtime-release/v1",
        ),
        "provider": (
            provider, "binding_digest", "synaptic-provider-runtime-binding/v1"
        ),
        "execution": (
            execution, "binding_digest", "synaptic-packaged-execution-binding/v1"
        ),
    }
    for name, (value, digest_field, domain) in values.items():
        assert value.canonical_json() == vectors[name]["canonical_json"]
        assert getattr(value, digest_field) == vectors[name]["digest"]
        unsigned = json.loads(value.canonical_json())
        unsigned.pop(digest_field)
        payload = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        expected = hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()
        assert expected == vectors[name]["digest"]
        other_domain = "synaptic-domain-separation-test/v1"
        assert hashlib.sha256(
            other_domain.encode("ascii") + b"\0" + payload
        ).hexdigest() != expected


def test_runtime_facts_are_detached_from_mutable_input() -> None:
    facts = {"native": {"version": "before"}}
    release = _release(runtime_facts=facts)
    facts["native"]["version"] = "after"
    assert release.to_dict()["platform"]["runtime_facts"] == {
        "native": {"version": "before"}
    }


def test_contracts_are_frozen() -> None:
    release = _release()
    with pytest.raises(FrozenInstanceError):
        release.release_ref = "runtime:changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "parser,factory",
    (
        (PackagedTrainingRuntimeReleaseV1.from_json, _release),
        (ProviderRuntimeBindingV1.from_json, _provider),
        (PackagedExecutionBindingV1.from_json, _execution),
    ),
)
def test_json_parsers_reject_duplicate_unknown_and_noncanonical_input(parser, factory) -> None:
    canonical = factory().canonical_json()
    with pytest.raises(ValueError, match="duplicate|malformed"):
        parser('{"schema_version":"duplicate",' + canonical[1:])

    document = json.loads(canonical)
    document["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        parser(json.dumps(document, sort_keys=True, separators=(",", ":")))

    with pytest.raises(ValueError, match="canonical"):
        parser(" " + canonical)


def test_json_parser_rejects_nonfinite_runtime_fact() -> None:
    canonical = _release().canonical_json()
    attacked = canonical.replace('"gpu":true', '"gpu":NaN')
    with pytest.raises(ValueError, match="malformed"):
        PackagedTrainingRuntimeReleaseV1.from_json(attacked)


class _DictSubclass(dict):
    pass


class _ListSubclass(list):
    pass


def test_contracts_reject_container_subclasses_and_float_facts() -> None:
    with pytest.raises(TypeError, match="exact canonical JSON"):
        _release(runtime_facts=_DictSubclass(ok=True))
    with pytest.raises(TypeError, match="exact array"):
        _release(compatible_methods=_ListSubclass(["sft"]))
    with pytest.raises(TypeError, match="fractional values"):
        _release(runtime_facts={"bad": float("inf")})
    with pytest.raises(TypeError, match="fractional values"):
        _release(runtime_facts={"negative_zero": -0.0})


def test_runtime_facts_enforce_depth_width_cycle_text_and_integer_bounds() -> None:
    deep: object = "leaf"
    for _ in range(9):
        deep = {"next": deep}
    with pytest.raises(ValueError, match="depth limit"):
        _release(runtime_facts=deep)

    with pytest.raises(ValueError, match="item limit"):
        _release(runtime_facts={f"k{index}": index for index in range(129)})

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="cycles"):
        _release(runtime_facts=cyclic)

    with pytest.raises(ValueError, match="byte limit"):
        _release(runtime_facts={"text": "x" * 4097})
    _release(runtime_facts={"text": "x" * 4096})
    with pytest.raises(ValueError, match="text limit"):
        _release(
            runtime_facts={f"text{index}": "x" * 4096 for index in range(9)}
        )
    with pytest.raises(ValueError, match="byte limit"):
        _release(runtime_facts={"k" * 129: "value"})

    _release(runtime_facts={"maximum": 9_007_199_254_740_991})
    with pytest.raises(ValueError, match="safe range"):
        _release(runtime_facts={"too_large": 9_007_199_254_740_992})
    with pytest.raises(ValueError, match="safe range"):
        _release(runtime_facts={"too_small": -9_007_199_254_740_992})


def test_runtime_fact_node_limit_is_closed_and_deterministic() -> None:
    # Each nested list contributes nodes while staying below the per-container bound.
    wide_tree = {f"group{index}": list(range(128)) for index in range(8)}
    with pytest.raises(ValueError, match="node limit"):
        _release(runtime_facts=wide_tree)


def test_direct_builders_reject_wide_or_huge_values_before_member_validation() -> None:
    with pytest.raises(ValueError, match="bounded"):
        _release(compatible_methods=tuple(object() for _ in range(257)))
    with pytest.raises(ValueError, match="bounded"):
        _release(compatible_models=tuple(object() for _ in range(257)))
    with pytest.raises(ValueError, match="item limit"):
        _release(runtime_facts={f"k{index}": object() for index in range(129)})
    with pytest.raises(ValueError, match="byte limit"):
        ProviderRuntimeBindingV1.build(
            provider_ref="provider:fixture", runtime_release=_release(),
            provider_facts_schema="x" * 1_000_000,
            provider_facts_digest=_E,
        )
    release = _release()
    with pytest.raises(ValueError, match="byte limit"):
        PackagedExecutionBindingV1.build(
            run_ref="run:huge-format", runtime_release=release,
            provider_runtime_binding=_provider(release),
            prepared_input_ref=f"prepared://sha256/{_A}",
            prepared_input_revision=_A,
            prepared_input_content_digest=_E,
            prepared_input_size_bytes=1,
            prepared_input_format="x" * 1_000_000,
            workload_digest=_B, configuration_digest=_C,
            artifact_policy_digest=_D,
        )


@pytest.mark.parametrize("shape", ("deep", "wide", "negative_zero", "unsafe_integer"))
def test_runtime_fact_json_parser_enforces_builder_boundaries(shape: str) -> None:
    document = _release().to_dict()
    if shape == "deep":
        value: object = "leaf"
        for _ in range(9):
            value = {"next": value}
    elif shape == "wide":
        value = {f"k{index}": index for index in range(129)}
    elif shape == "negative_zero":
        value = {"value": -0.0}
    else:
        value = {"value": 9_007_199_254_740_992}
    document["platform"]["runtime_facts"] = value
    attacked = json.dumps(document, sort_keys=True, separators=(",", ":"))
    with pytest.raises((TypeError, ValueError)):
        PackagedTrainingRuntimeReleaseV1.from_json(attacked)


def test_provider_binding_serializes_only_provider_facts_commitment() -> None:
    document = _provider().to_dict()
    assert set(document) == {
        "schema_version", "provider_ref", "runtime_release_digest",
        "provider_facts_schema", "provider_facts_digest", "binding_digest",
    }
    encoded = _provider().canonical_json()
    for forbidden in (
        "deployment", "object_id", "named_secret", "environment", "token",
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    "schema_identity",
    (
        "https://user@host/schema/v1",
        "file:///schema/v1",
        "../schema/v1",
        "fixture-runtime-facts/latest",
        "Fixture-Runtime-Facts/v1",
    ),
)
def test_provider_facts_schema_is_versioned_identity_only(
    schema_identity: str,
) -> None:
    with pytest.raises(ValueError, match="versioned provider-specific"):
        ProviderRuntimeBindingV1.build(
            provider_ref="provider:fixture", runtime_release=_release(),
            provider_facts_schema=schema_identity, provider_facts_digest=_E,
        )


def test_provider_binding_parser_rejects_embedded_raw_facts() -> None:
    document = _provider().to_dict()
    document["binding_facts"] = {
        "environment": {"TOKEN": "secret"}, "named_secret_refs": ["secret"]
    }
    with pytest.raises(ValueError, match="unknown"):
        ProviderRuntimeBindingV1.from_dict(document)


@pytest.mark.parametrize(
    "prepared_ref",
    (
        "file:///tmp/data.jsonl",
        "git://host/repo",
        "https://host/data",
        "https://user:secret@host/data",
        "../data.jsonl",
        "prepared://sha256/not-a-digest",
        f"prepared://sha256/{'A' * 64}",
        f"prepared://sha256/{_A}?token=secret",
        f"prepared://other/{_A}",
    ),
)
def test_prepared_input_ref_is_exact_non_dereference_identity(prepared_ref: str) -> None:
    release = _release()
    with pytest.raises(ValueError, match="exact prepared"):
        PackagedExecutionBindingV1.build(
            run_ref="run:bad-prepared-ref",
            runtime_release=release,
            provider_runtime_binding=_provider(release),
            prepared_input_ref=prepared_ref,
            prepared_input_revision=_A,
            prepared_input_content_digest=_E,
            prepared_input_size_bytes=1,
            prepared_input_format="syntunia-sft-row/v2",
            workload_digest=_B,
            configuration_digest=_C,
            artifact_policy_digest=_D,
        )


def test_prepared_input_semantic_digest_must_match_ref() -> None:
    release = _release()
    with pytest.raises(ValueError, match="semantic digest"):
        PackagedExecutionBindingV1.build(
            run_ref="run:mismatch",
            runtime_release=release,
            provider_runtime_binding=_provider(release),
            prepared_input_ref=f"prepared://sha256/{_A}",
            prepared_input_revision=_B,
            prepared_input_content_digest=_E,
            prepared_input_size_bytes=1,
            prepared_input_format="syntunia-sft-row/v2",
            workload_digest=_B,
            configuration_digest=_C,
            artifact_policy_digest=_D,
        )


def test_execution_binds_complete_public_prepared_identity() -> None:
    identity = {
        "ref": f"prepared://sha256/{_A}", "revision": _A,
        "content_digest": _E, "size_bytes": 10_513_545,
        "format": "syntunia-sft-row/v2",
    }
    release = _release()
    execution = PackagedExecutionBindingV1.build(
        run_ref="run:prepared-identity", runtime_release=release,
        provider_runtime_binding=_provider(release),
        prepared_input_ref=identity["ref"],
        prepared_input_revision=identity["revision"],
        prepared_input_content_digest=identity["content_digest"],
        prepared_input_size_bytes=identity["size_bytes"],
        prepared_input_format=identity["format"],
        workload_digest=_B, configuration_digest=_C,
        artifact_policy_digest=_D,
    )
    assert execution.to_dict()["prepared_input"] == identity


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("revision", "prepared-v1"),
        ("content_digest", "E" * 64),
        ("size_bytes", 0),
        ("size_bytes", 64 * 1024 * 1024 + 1),
        ("format", "../format"),
    ),
)
def test_execution_rejects_malformed_complete_prepared_identity(
    field: str, value: object,
) -> None:
    document = _execution().to_dict()
    document["prepared_input"][field] = value
    with pytest.raises((TypeError, ValueError)):
        PackagedExecutionBindingV1.from_dict(document)


def test_execution_run_ref_rejects_path_material() -> None:
    document = _execution().to_dict()
    document["run_ref"] = "../run"
    with pytest.raises(ValueError, match="opaque non-path"):
        PackagedExecutionBindingV1.from_dict(document)


@pytest.mark.parametrize(
    "changes,match",
    (
        ({"image_ref": "ghcr.io/synaptic-labs/tuner:latest"}, "immutable"),
        ({"image_ref": f"ghcr.io/synaptic-labs/tuner@sha256:{_A}"}, "match"),
        ({"package_digest": "A" * 64}, "lowercase SHA-256"),
        ({"python_version": "3.12"}, "semantic version"),
        ({"python_executable": "python"}, "absolute canonical POSIX"),
        ({"worker_entrypoint": "worker.py"}, "package entrypoint"),
        ({"installed_distribution_count": 0}, "allowed range"),
        ({"compatible_methods": ("sft", "sft")}, "unique"),
        ({"compatible_models": (("Qwen/B", _REVISION), ("Qwen/A", _REVISION))}, "ascending"),
        ({"compatible_models": (("Qwen/A", "main"),)}, "immutable lowercase revision"),
        ({"compatible_dataset_formats": ()}, "nonempty"),
    ),
)
def test_release_rejects_malformed_identity_and_compatibility_facts(changes, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _release(**changes)


def test_recorded_digests_are_verified() -> None:
    release_document = _release().to_dict()
    release_document["manifest_digest"] = _E
    with pytest.raises(ValueError, match="does not bind"):
        PackagedTrainingRuntimeReleaseV1.from_dict(release_document)

    provider_document = _provider().to_dict()
    provider_document["binding_digest"] = _E
    with pytest.raises(ValueError, match="does not bind"):
        ProviderRuntimeBindingV1.from_dict(provider_document)

    execution_document = _execution().to_dict()
    execution_document["binding_digest"] = _E
    with pytest.raises(ValueError, match="does not bind"):
        PackagedExecutionBindingV1.from_dict(execution_document)


def test_execution_build_and_revalidation_reject_cross_binding_mismatch() -> None:
    first = _release()
    second = _release(release_ref="runtime:qwen35-sft-v2")
    first_provider = _provider(first)
    with pytest.raises(ValueError, match="different runtime release"):
        _execution(second, first_provider)

    execution = _execution(first, first_provider)
    second_provider = _provider(second)
    with pytest.raises(ValueError, match="cross-binding mismatch"):
        execution.validate_bindings(second, second_provider)


def test_public_v1_module_exposes_release_contracts() -> None:
    from synaptic_tuner.api.v1 import runtime_releases

    assert runtime_releases.PackagedTrainingRuntimeReleaseV1 is PackagedTrainingRuntimeReleaseV1
    assert runtime_releases.ProviderRuntimeBindingV1 is ProviderRuntimeBindingV1
    assert runtime_releases.PackagedExecutionBindingV1 is PackagedExecutionBindingV1


def _schema(name: str) -> dict[str, object]:
    return json.loads((Path("schemas") / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "attack", ("float", "wide", "unicode_key", "path_release_ref")
)
def test_release_schema_and_parser_reject_same_structural_attacks(attack: str) -> None:
    document = _release().to_dict()
    if attack == "float":
        document["platform"]["runtime_facts"] = {"value": 1.5}
    elif attack == "wide":
        document["platform"]["runtime_facts"] = {
            f"k{index}": index for index in range(129)
        }
    elif attack == "unicode_key":
        document["platform"]["runtime_facts"] = {"café": "rejected-key"}
    else:
        document["release_ref"] = "../runtime"
    schema = _schema("synaptic-packaged-training-runtime-release-v1.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)
    with pytest.raises((TypeError, ValueError)):
        PackagedTrainingRuntimeReleaseV1.from_dict(document)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("python", "version"), "1." + "2" * 125 + ".3"),
        (("worker", "entrypoint"), "a" * 508 + ":main"),
        (("image", "reference"), "a" * 441 + "@sha256:" + _D),
    ),
)
def test_release_schema_and_parser_reject_just_over_length_limits(
    path: tuple[str, str], value: str,
) -> None:
    assert len(value) == (129 if path == ("python", "version") else 513)
    document = _release().to_dict()
    document[path[0]][path[1]] = value
    schema = _schema("synaptic-packaged-training-runtime-release-v1.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)
    with pytest.raises((TypeError, ValueError), match="byte limit"):
        PackagedTrainingRuntimeReleaseV1.from_dict(document)


@pytest.mark.parametrize("attack", ("raw_facts", "schema_uri", "bad_digest"))
def test_provider_schema_and_parser_reject_same_structural_attacks(attack: str) -> None:
    document = _provider().to_dict()
    if attack == "raw_facts":
        document["binding_facts"] = {"environment": {"TOKEN": "secret"}}
    elif attack == "schema_uri":
        document["provider_facts_schema"] = "https://user@host/schema/v1"
    else:
        document["provider_facts_digest"] = "A" * 64
    schema = _schema("synaptic-provider-runtime-binding-v1.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)
    with pytest.raises((TypeError, ValueError)):
        ProviderRuntimeBindingV1.from_dict(document)


@pytest.mark.parametrize("prepared_ref", ("file:///tmp/data", "https://host/data", "../data"))
def test_execution_schema_and_parser_reject_same_prepared_ref_attacks(
    prepared_ref: str,
) -> None:
    document = _execution().to_dict()
    document["prepared_input"]["ref"] = prepared_ref
    schema = _schema("synaptic-packaged-execution-binding-v1.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)
    with pytest.raises((TypeError, ValueError)):
        PackagedExecutionBindingV1.from_dict(document)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("revision", "prepared-v1"),
        ("content_digest", "E" * 64),
        ("size_bytes", 0),
        ("size_bytes", 67_108_865),
        ("format", "../format"),
    ),
)
def test_execution_schema_and_parser_reject_same_prepared_field_attacks(
    field: str, value: object,
) -> None:
    document = _execution().to_dict()
    document["prepared_input"][field] = value
    schema = _schema("synaptic-packaged-execution-binding-v1.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)
    with pytest.raises((TypeError, ValueError)):
        PackagedExecutionBindingV1.from_dict(document)


@pytest.mark.parametrize(
    ("schema_name", "factory"),
    (
        ("synaptic-packaged-training-runtime-release-v1.schema.json", _release),
        ("synaptic-provider-runtime-binding-v1.schema.json", _provider),
        ("synaptic-packaged-execution-binding-v1.schema.json", _execution),
    ),
)
def test_canonical_contracts_validate_against_published_schemas(
    schema_name: str, factory
) -> None:
    schema = _schema(schema_name)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(factory().to_dict())
