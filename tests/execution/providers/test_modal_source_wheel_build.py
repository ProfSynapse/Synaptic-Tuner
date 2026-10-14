"""Checkout-only real offline source-wheel regression; no provider effects."""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import zipfile

import pytest

from tuner.execution.providers.modal import runtime_build


def test_real_current_source_wheel_contains_declared_public_assets(tmp_path):
    python = os.environ.get("SYNAPTIC_TEST_MODAL_SOURCE_WHEEL_PYTHON")
    if not python:
        pytest.skip("explicit CPython 3.11 source-build lane required")
    # Capture the complete tracked source independently of the production allowlist.
    checkout = Path(runtime_build.__file__).resolve().parents[4]
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", "HEAD"],
        capture_output=True, check=True, timeout=120,
    ).stdout
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as members:
        members.extractall(source, filter="data")
    for command in (
        ["git", "init", "-q", str(source)],
        ["git", "-C", str(source), "add", "."],
        ["git", "-C", str(source), "-c", "user.name=Source build regression",
         "-c", "user.email=source-build@localhost", "commit", "-qm", "source snapshot"],
    ):
        subprocess.run(command, capture_output=True, check=True, timeout=120)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True, check=True, timeout=30,
    ).stdout.decode("ascii").strip()
    output = tmp_path / "wheel"
    output.mkdir(mode=0o700)
    cache = tmp_path / "builder-cache"
    # The selected source is the Git snapshot. Import the actual tested controller
    # separately so this regression also exercises an uncommitted allowlist fix.
    program = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tuner.execution.providers.modal.runtime_build import prepare_current_source_wheel
wheel,digest=prepare_current_source_wheel(Path(sys.argv[2]),Path(sys.argv[3]),
    expected_source_commit=sys.argv[4],builder_cache_root=Path(sys.argv[5]))
print(json.dumps({'wheel':str(wheel),'sha256':digest}))
"""
    result = subprocess.run(
        [python, "-I", "-B", "-c", program, str(checkout), str(source),
         str(output), commit, str(cache)],
        capture_output=True, check=False, timeout=600,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
             "TMPDIR": str(tmp_path), "PYTHONNOUSERSITE": "1"},
    )
    assert result.returncode == 0, "real offline source-wheel build rejected"
    receipt = json.loads(result.stdout)
    wheel = Path(receipt["wheel"])
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == receipt["sha256"]
    with zipfile.ZipFile(wheel) as built:
        schemas = sorted((source / "schemas").glob("synaptic-*.schema.json"))
        assert schemas, "selected source must contain its declared public schemas"
        for schema in schemas:
            assert built.read("schemas/" + schema.name) == schema.read_bytes()
        for member in ("cli-first-tool-schemas.json", "Tools/tool_schemas.json"):
            assert built.read("synaptic_tuner/runtime/" + member) == (source / member).read_bytes()
        for member in (
            "Trainers/runtime_profiles/smollm2-135m-grpo-candidate-v1.yaml",
            "Trainers/runtime_profiles/smollm2-135m-grpo-candidate-v1.inventory.json",
            "tuner/runtime/manifests/offline-grpo-worker-v1.json",
            "tuner/runtime/manifests/packaged-grpo-worker-v1.json",
        ):
            assert built.read(member) == (source / member).read_bytes()
        assert "_build_support.py" not in built.namelist()
        assert not any(name.startswith("Trainers/image_profiles/") for name in built.namelist())
