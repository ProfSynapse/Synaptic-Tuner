from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_modal_extra_and_launcher_lock_are_exact():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["optional-dependencies"]["modal"] == ["modal==1.5.4"]
    lines = [
        line.strip()
        for line in (ROOT / "requirements" / "modal-launcher-v1.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pattern = re.compile(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^ ]+) --hash=sha256:([0-9a-f]{64})$"
    )
    parsed = [pattern.fullmatch(line) for line in lines]
    assert all(parsed) and len(lines) == 37
    packages = {match.group(1).lower().replace("_", "-") for match in parsed}
    assert len(packages) == len(lines)
    assert {
        "modal", "pyyaml", "jsonschema", "packaging", "python-dotenv",
        "aiohttp", "cbor2", "certifi", "click", "grpclib", "protobuf",
        "rich", "synchronicity", "toml", "types-certifi", "types-toml",
        "watchfiles", "typing-extensions",
    }.issubset(packages)
    modal = next(line for line in lines if line.startswith("modal=="))
    assert modal == (
        "modal==1.5.4 --hash=sha256:"
        "3e54e26037c445af42f9a9ef9862b66bdd2e0b1faeced5fcc7adf3e5f59e44ed"
    )


def test_engine_modal_modules_do_not_import_the_optional_sdk_at_module_scope():
    root = ROOT / "tuner" / "execution" / "providers" / "modal"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "modal" for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "modal" and not (node.module or "").startswith("modal."), path


def test_importing_public_api_does_not_materialize_modal_sdk():
    before = set(sys.modules)
    __import__("synaptic_tuner.api.v1")
    added = set(sys.modules) - before
    assert "modal" not in added


def test_provider_specific_public_contract_still_does_not_import_modal_sdk():
    before = set(sys.modules)
    module = __import__("synaptic_tuner.api.v1.modal", fromlist=["*"])
    added = set(sys.modules) - before
    assert "modal" not in added
    assert hasattr(module, "ModalTrainingRepository")
    assert hasattr(module, "ModalDurablePreparationV1")
    assert hasattr(module, "ModalPreparedRunV1")
    from tuner.execution.providers.modal.training import ModalPreparedRunV1
    assert module.ModalPreparedRunV1 is ModalPreparedRunV1
    assert hasattr(module, "ExplicitModal154ReadFacade")
    assert hasattr(module, "ModalDeploymentSelectionV1")
    assert hasattr(module, "ModalDeploymentSpecV1")
    assert hasattr(module, "ModalVerificationPolicyV1")
    assert hasattr(module, "MountedModalWorkerV1")
    assert hasattr(module, "build_modal_deployment")
    assert hasattr(module, "compose_modal_source_finalizer")
