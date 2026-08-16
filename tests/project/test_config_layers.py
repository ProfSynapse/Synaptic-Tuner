import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from tuner.project.config_layers import ConfigDocument, ConfigOverride, resolve_config_layers
from tuner.project.secrets import SecretRef
from tuner.project.errors import SecretReferenceError


def document(uri: str, precedence: int, data: dict) -> ConfigDocument:
    return ConfigDocument.from_mapping(uri=uri, precedence=precedence, data=data)


def test_layers_merge_by_precedence_and_ties_keep_declaration_order() -> None:
    resolved = resolve_config_layers(
        [
            document("engine://defaults.yaml", 10, {"training": {"steps": 10, "lr": 1e-4}}),
            document("project://first.yaml", 30, {"training": {"steps": 20}}),
            document("project://second.yaml", 30, {"training": {"steps": 30}}),
        ],
        overrides=[ConfigOverride("training.steps", 40)],
    )
    assert resolved.config == {"training": {"steps": 40, "lr": 1e-4}}
    assert resolved.source_map["training.steps"]["uri"] == "cli"
    assert [source["uri"] for source in resolved.sources] == [
        "engine://defaults.yaml",
        "project://first.yaml",
        "project://second.yaml",
    ]


def test_hash_is_deterministic_for_mapping_order() -> None:
    first = resolve_config_layers([document("project://a.yaml", 1, {"b": 2, "a": 1})])
    second = resolve_config_layers([document("project://a.yaml", 1, {"a": 1, "b": 2})])
    assert first.resolved_sha256 == second.resolved_sha256


def test_resolved_record_serializes_only_secret_reference(tmp_path: Path) -> None:
    resolved = resolve_config_layers(
        [document("project://a.yaml", 1, {"auth": SecretRef("env", "HF_TOKEN")})]
    )
    payload = resolved.to_dict()
    assert payload["config"]["auth"] == {"provider": "env", "name": "HF_TOKEN"}
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "schemas" / "synaptic-resolved-config-v1.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(payload)


def test_config_rejects_literal_values_in_sensitive_fields() -> None:
    with pytest.raises(SecretReferenceError, match="SecretRef"):
        document("project://a.yaml", 1, {"provider": {"api_key": "literal-value"}})


def test_structural_replacement_removes_stale_descendant_provenance() -> None:
    resolved = resolve_config_layers(
        [
            document("engine://defaults.yaml", 10, {"training": {"steps": 10, "lr": 1e-4}}),
            document("project://override.yaml", 20, {"training": "disabled"}),
        ]
    )
    assert resolved.config == {"training": "disabled"}
    assert resolved.source_map == {
        "training": {
            "uri": "project://override.yaml",
            "sha256": resolved.sources[1]["sha256"],
            "precedence": 20,
        }
    }


def test_nested_override_removes_stale_ancestor_provenance() -> None:
    resolved = resolve_config_layers(
        [document("engine://defaults.yaml", 10, {"training": "disabled"})],
        overrides=[ConfigOverride("training.steps", 20)],
    )
    assert resolved.config == {"training": {"steps": 20}}
    assert "training" not in resolved.source_map
    assert resolved.source_map["training.steps"] == {"uri": "cli", "precedence": 100}
