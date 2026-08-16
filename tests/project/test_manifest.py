import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tuner.project.errors import ManifestValidationError, ManifestVersionError
from tuner.project.manifest import (
    ProjectManifest,
    load_project_manifest,
    validate_engine_requirement,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_example_manifest_validates_and_builds_context(tmp_path: Path) -> None:
    source = REPO_ROOT / "examples" / "host-project" / "synaptic.yaml"
    manifest_path = tmp_path / "host" / "synaptic.yaml"
    manifest_path.parent.mkdir()
    manifest_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = load_project_manifest(manifest_path)
    context = manifest.create_context(engine_root=tmp_path / "host" / "vendor" / "engine")

    assert manifest.project_id == "example-research"
    assert context.config_root == manifest_path.parent / "experiments"
    assert context.tmp_root == manifest_path.parent / ".synaptic" / "tmp"


def test_unknown_same_major_top_level_field_warns_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "synaptic.yaml"
    path.write_text(
        """schema_version: synaptic-project/v1
project: {id: test, name: Test}
engine: {requires: '>=1', api: v1}
future_optional: {enabled: true}
""",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="future_optional"):
        manifest = load_project_manifest(path)
    assert manifest.to_dict()["future_optional"] == {"enabled": True}


def test_unknown_major_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "synaptic.yaml"
    path.write_text("schema_version: synaptic-project/v2\n", encoding="utf-8")
    with pytest.raises(ManifestVersionError):
        load_project_manifest(path)


def test_writable_root_inside_engine_is_rejected(tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    path = host / "synaptic.yaml"
    path.write_text(
        """schema_version: synaptic-project/v1
project: {id: test, name: Test}
engine: {requires: '>=1', api: v1}
paths:
  artifacts: engine://generated
""",
        encoding="utf-8",
    )
    manifest = load_project_manifest(path)
    with pytest.raises(ManifestValidationError, match="inside the engine"):
        manifest.create_context(engine_root=host / "vendor" / "engine")


def test_host_source_writable_root_is_rejected(tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    path = host / "synaptic.yaml"
    path.write_text(
        f"""schema_version: synaptic-project/v1
project: {{id: test, name: Test}}
engine: {{requires: '>=1', api: v1}}
paths:
  artifacts: project://configs
""",
        encoding="utf-8",
    )
    manifest = load_project_manifest(path)
    with pytest.raises(ManifestValidationError, match="host .synaptic"):
        manifest.create_context(engine_root=host / "vendor" / "engine")


def test_external_writable_root_is_rejected(tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    external = (tmp_path / "external-artifacts").resolve().as_uri()
    path = host / "synaptic.yaml"
    path.write_text(
        f"""schema_version: synaptic-project/v1
project: {{id: test, name: Test}}
engine: {{requires: '>=1', api: v1}}
paths:
  artifacts: {external}
""",
        encoding="utf-8",
    )
    manifest = load_project_manifest(path)
    with pytest.raises(ManifestValidationError, match="host .synaptic"):
        manifest.create_context(engine_root=host / "vendor" / "engine")


def test_manifest_rejects_literal_secret_without_echoing_value(tmp_path: Path) -> None:
    literal = "do-not-echo-this-value"
    path = tmp_path / "synaptic.yaml"
    path.write_text(
        f"""schema_version: synaptic-project/v1
project: {{id: test, name: Test}}
engine: {{requires: '>=1', api: v1}}
provider:
  api_token: {literal}
""",
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError) as error:
        load_project_manifest(path)
    assert literal not in str(error.value)
    assert literal not in repr(error.value.details)


def test_manifest_accepts_opaque_secret_reference(tmp_path: Path) -> None:
    path = tmp_path / "synaptic.yaml"
    path.write_text(
        """schema_version: synaptic-project/v1
project: {id: test, name: Test}
engine: {requires: '>=1', api: v1}
provider:
  api_token:
    provider: env
    name: HF_TOKEN
""",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="provider"):
        manifest = load_project_manifest(path)
    assert manifest.to_dict()["provider"]["api_token"]["name"] == "HF_TOKEN"


def test_all_owned_schemas_are_well_formed() -> None:
    for path in sorted((REPO_ROOT / "schemas").glob("synaptic-*-v1.schema.json")):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "duplicate_yaml",
    [
        """schema_version: synaptic-project/v1
schema_version: duplicate-value-must-not-leak
project: {id: test, name: Test}
engine: {requires: '>=1', api: v1}
""",
        """schema_version: synaptic-project/v1
project: {id: test, name: Test}
engine: {requires: '>=1', api: v1}
plugins:
  bindings:
    same:
      kind: renderer
      api: v1
      target: safe.module:first
    same:
      kind: grader
      api: v1
      target: secret.module:value_must_not_leak
""",
    ],
)
def test_manifest_rejects_duplicate_mapping_keys_without_values(
    tmp_path: Path, duplicate_yaml: str
) -> None:
    path = tmp_path / "synaptic.yaml"
    path.write_text(duplicate_yaml, encoding="utf-8")

    with pytest.raises(ManifestValidationError) as error:
        load_project_manifest(path)

    assert error.value.details["reason"] == "duplicate_mapping_key"
    assert "duplicate-value-must-not-leak" not in str(error.value)
    assert "duplicate-value-must-not-leak" not in repr(error.value.details)
    assert "secret.module:value_must_not_leak" not in str(error.value)
    assert "secret.module:value_must_not_leak" not in repr(error.value.details)


def _manifest_with_requirement(tmp_path: Path, requirement: str) -> ProjectManifest:
    path = tmp_path / "synaptic.yaml"
    path.write_text(
        f"""schema_version: synaptic-project/v1
project: {{id: test, name: Test}}
engine: {{requires: {requirement!r}, api: v1}}
""",
        encoding="utf-8",
    )
    return load_project_manifest(path)


@pytest.mark.parametrize("engine_version", ["1.5.0", "1.9.0"])
def test_engine_requirement_accepts_compatible_version(
    tmp_path: Path, engine_version: str
) -> None:
    manifest = _manifest_with_requirement(tmp_path, ">=1.0,<2")
    validate_engine_requirement(manifest, engine_version)


def test_engine_requirement_rejects_incompatible_version(tmp_path: Path) -> None:
    manifest = _manifest_with_requirement(tmp_path, ">=1.0,<2")

    with pytest.raises(ManifestValidationError) as error:
        validate_engine_requirement(manifest, "2.0.0")

    assert error.value.details == {
        "reason": "engine_version_incompatible",
        "requires": ">=1.0,<2",
        "engine_version": "2.0.0",
    }


def test_engine_requirement_rejects_invalid_specifier(tmp_path: Path) -> None:
    manifest = _manifest_with_requirement(tmp_path, "definitely-not-a-specifier")

    with pytest.raises(ManifestValidationError) as error:
        validate_engine_requirement(manifest, "1.0.0")

    assert error.value.details["reason"] == "invalid_engine_requirement"


def test_engine_requirement_rejects_invalid_engine_version(tmp_path: Path) -> None:
    manifest = _manifest_with_requirement(tmp_path, ">=1")

    with pytest.raises(ManifestValidationError) as error:
        validate_engine_requirement(manifest, "not-a-version")

    assert error.value.details["reason"] == "invalid_engine_version"
