"""Static and runtime isolation of the explicit host packaged boundary."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LEGACY = (
    "tuner/training/contracts.py", "tuner/training/recipes.py",
    "tuner/training/service.py", "tuner/training/coordinator_material.py",
)
HOST_MODULES = {
    "tuner.runtime.releases", "tuner.training.packaged_compilation",
    "tuner.training.packaged_boundary",
}


def static_imports(relative):
    package = relative[:-3].replace("/", ".").rsplit(".", 1)[0]
    imports = set()
    for node in ast.walk(ast.parse((ROOT / relative).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = package.split(".")[:len(package.split(".")) - node.level + 1]
                module = ".".join(parent + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            imports.add(module)
            imports.update(module + "." + alias.name for alias in node.names)
    return imports


def isolated(code):
    preamble = f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
    result = subprocess.run([sys.executable, "-I", "-c", preamble + code],
                            check=True, capture_output=True, text=True)
    return result.stdout


def test_legacy_members_have_no_static_packaged_boundary_edges():
    for path in LEGACY:
        assert static_imports(path).isdisjoint(HOST_MODULES), path
    assert "synaptic_tuner.api.v1.training_sources" not in static_imports(
        "synaptic_tuner/api/v1/training_facade.py"
    )
    assert "synaptic_tuner.api.v1.execution" not in static_imports(
        "tuner/training/contracts.py"
    )


def test_fixed_offline_inventory_gains_no_static_packaged_edges():
    manifest = json.loads((ROOT / "tuner/runtime/manifests/offline-sft-worker-v1.json").read_text())
    assert manifest["member_count"] == len(manifest["members"]) == 66
    for member in manifest["members"]:
        path = member["path"]
        if path.endswith(".py"):
            assert static_imports(path).isdisjoint(HOST_MODULES), path


def test_host_boundary_owns_release_and_packaged_compiler_static_edges():
    paths = (*LEGACY, "tuner/training/packaged_compilation.py",
             "tuner/training/packaged_boundary.py")
    for module in ("tuner.runtime.releases", "tuner.training.packaged_compilation"):
        assert [path for path in paths if module in static_imports(path)] == [
            "tuner/training/packaged_boundary.py"
        ]


def test_public_and_developer_imports_and_introspection_are_isolated():
    isolated("""
import typing
from synaptic_tuner.api.v1 import PreparedTrainingInputIdentity, TrainingAPI
from tuner.training import TrainingService, default_recipe_registry
from tuner.training.contracts import ResolvedTrainingRequest, ResolvedTrainingComponents, TrainingPlan
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
default_recipe_registry()
for value in (ResolvedTrainingRequest, ResolvedTrainingComponents, TrainingPlan):
    assert typing.get_type_hints(value)['execution_source'] is object
for module in ('tuner.runtime.releases', 'tuner.training.packaged_compilation',
               'tuner.training.packaged_boundary', 'synaptic_tuner.api.v1.training_sources'):
    assert module not in sys.modules, module
""")


def test_explicit_compatibility_exports_load_only_the_host_boundary():
    isolated("""
import tuner.training.contracts as contracts
assert 'tuner.training.packaged_boundary' not in sys.modules
from tuner.training.contracts import ExecutionMaterialV1, compile_training_plan_for_execution_v1
from tuner.training import packaged_boundary as boundary
assert ExecutionMaterialV1 == boundary.ExecutionMaterialV1
assert compile_training_plan_for_execution_v1 is boundary.compile_training_plan_for_execution_v1
for prefix in ('modal', 'huggingface_hub', 'runpod', 'torch', 'tuner.execution.providers'):
    assert not any(n == prefix or n.startswith(prefix + '.') for n in sys.modules)
""")


def test_prepared_identity_is_one_shared_class_with_historical_pickle_identity():
    isolated("""
import pickle
from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity as shared
assert 'synaptic_tuner.api.v1.execution' not in sys.modules
from synaptic_tuner.api.v1 import PreparedTrainingInputIdentity as public
from synaptic_tuner.api.v1.execution import PreparedTrainingInputIdentity as execution
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity as source
from tuner.training.contracts import PreparedTrainingInputIdentity as rich
assert shared is public is execution is source is rich
assert shared.__module__ == 'synaptic_tuner.api.v1.execution'
identity = shared('prepared://sha256/' + 'a'*64, 'a'*64, 'b'*64, 42, 'syntunia-sft-row/v2')
assert pickle.loads(pickle.dumps(identity)) == identity
assert shared.from_dict(identity.to_dict()) == identity
assert len(identity.to_dict()) == len(identity.execution_binding_fields()) == 5
""")


def test_prepare_delegation_does_not_import_host_sources_or_change_objects():
    isolated("""
from synaptic_tuner.api.v1.training_facade import TrainingAPI
source, config, output = object(), object(), object()
class Operations:
    def prepare(self, actual_source, actual_config):
        assert actual_source is source and actual_config is config
        return output
assert TrainingAPI(Operations(), clock=object()).prepare(source, config) is output
assert 'synaptic_tuner.api.v1.training_sources' not in sys.modules
class RejectingOperations:
    def prepare(self, source, config):
        raise ValueError('operation rejected invalid input')
try:
    TrainingAPI(RejectingOperations(), clock=object()).prepare(source, config)
except ValueError:
    pass
else:
    raise AssertionError('operation validation was bypassed')
""")


def test_default_service_and_generic_coordinator_cannot_admit_packaged_material(tmp_path):
    from tuner.project.context import ProjectContext
    from tuner.project.errors import SourceLockError
    from tuner.training import TrainingService, default_recipe_registry
    from tuner.training.contracts import CanonicalDocument, TrainingRequest
    from tuner.training.coordinator_material import CoordinatorResolvedMaterial, derive_coordinator_material
    from tuner.training.packaged_boundary import derive_packaged_coordinator_material
    from tests.training.test_packaged_execution_material import packaged_fixture, Resolver, resolved

    training_input, components = packaged_fixture()
    default = TrainingService(context=ProjectContext.standalone(engine_root=tmp_path),
                              resolver=Resolver(components), recipes=default_recipe_registry())
    request = TrainingRequest(CanonicalDocument(training_input.canonical_json()))
    with pytest.raises(ValueError, match="explicit host boundary"):
        default.resolve_selected(request)
    rich = resolved(tmp_path)
    with pytest.raises(ValueError, match="explicit host boundary"):
        default.plan(rich)
    with pytest.raises(TypeError, match="ExecutionSourceV1"):
        derive_coordinator_material(rich, default_recipe_registry(), request_id="request",
                                    project_ref="project", run_id="run-packaged")
    material = derive_packaged_coordinator_material(rich, default_recipe_registry(), request_id="request",
                                                    project_ref="project", run_id="run-packaged")
    with pytest.raises((TypeError, ValueError, SourceLockError)):
        CoordinatorResolvedMaterial.parse(material.canonical_bytes, default_recipe_registry())


def test_packaged_coordinator_has_no_git_fallback(tmp_path):
    from tuner.training import default_recipe_registry
    from tuner.training.packaged_boundary import parse_packaged_coordinator_material
    from tests.training.test_coordinator_material import material

    git_material = material(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        parse_packaged_coordinator_material(git_material.canonical_bytes, default_recipe_registry())
