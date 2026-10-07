from __future__ import annotations

import ast
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


ROOT = Path(__file__).parents[4]
PACKAGE = ROOT / "tuner" / "execution" / "providers" / "docker_provider_v1"
FIXTURE = ROOT / "tests" / "fixtures" / "docker_provider_v1" / "cpu_fixture"


def test_package_exports_are_empty_and_forbidden_dependencies_are_absent():
    namespace = {}
    exec((PACKAGE / "__init__.py").read_text(encoding="utf-8"), namespace)
    assert namespace["__all__"] == []
    forbidden = {"subprocess", "docker", "sqlite3", "modal", "runpod", "huggingface_hub"}
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not (imports & forbidden), (path.name, imports & forbidden)


def test_no_legacy_or_cross_provider_runtime_branches():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py"))
    for forbidden in ("LocalRunHandler", "legacy uploader", "provider_id ==", "profile_ref =="):
        assert forbidden not in text


def test_checked_in_cpu_fixture_is_dependency_free_and_writes_explicit_artifact_root(tmp_path):
    spec = spec_from_file_location("docker_cpu_fixture", FIXTURE / "source" / "run_fixture.py")
    module = module_from_spec(spec); spec.loader.exec_module(module)
    artifact_root = tmp_path / "arbitrary-output"
    assert module.main(["run_fixture.py", str(FIXTURE / "source"), str(artifact_root)]) == 0
    value = json.loads((artifact_root / "result.json").read_text(encoding="utf-8"))
    assert value["message"] == "docker-provider-v1-cpu-fixture"
