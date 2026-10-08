from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tuner.runtime_profile_inventory import (
    PLATFORM_FACTS,
    PROBE_SCHEMA,
    PROBE_SOURCE,
    InventoryCaptureError,
    build_inventory,
    capture_dockerfile,
    diff_inventories,
    overlay_probe,
    probe_command,
)
from tuner.runtime_profiles import parse_runtime_inventory


ROOT = Path(__file__).resolve().parents[2]
SFT_INVENTORY = ROOT / "Trainers" / "runtime_profiles" / "qwen35-sft-v1.inventory.json"
SFT_IMAGE = (
    "unsloth/unsloth@sha256:"
    "1644d635bc7c5b57ed64cabbab1ae00647dfb583c2135fdfb6a461aeeba52739"
)
PYTHON = "/opt/unsloth-venv/bin/python3"


def _probe_from_inventory(path: Path, *, reverse: bool = True) -> dict[str, object]:
    """Probe output that an image with exactly this inventory would print."""
    inventory = json.loads(path.read_bytes())
    distributions = [
        {"name": item["name"], "version": item["version"],
         "location": f"/opt/unsloth-venv/lib/python3.12/site-packages/{item['name']}.dist-info"}
        for item in inventory["distributions"]
    ]
    if reverse:  # metadata enumeration order is arbitrary
        distributions.reverse()
    return {
        "schema_version": PROBE_SCHEMA,
        "executable": PYTHON,
        "distributions": distributions,
        "runtime": {fact: inventory["runtime"][fact] for fact in PLATFORM_FACTS},
    }


def _raw(document: dict[str, object]) -> bytes:
    return json.dumps(document).encode()


def test_reproduces_checked_in_sft_inventory_bytes_exactly() -> None:
    captured = build_inventory(
        _raw(_probe_from_inventory(SFT_INVENTORY)), image=SFT_IMAGE, python_executable=PYTHON
    )
    assert captured.payload == SFT_INVENTORY.read_bytes()
    assert captured.distribution_count == 327
    assert captured.sha256 == (
        "sha256:ecf4d0a27e5455b946237bf97790cba308765c2e5cf538e560e0de4c372f88e3"
    )
    assert captured.runtime["trl_version"] == "0.24.0"
    assert captured.runtime["cuda_build"] == "12.8"


def test_version_facts_follow_installed_distributions() -> None:
    probe = _probe_from_inventory(SFT_INVENTORY)
    for item in probe["distributions"]:  # type: ignore[union-attr]
        if item["name"] == "trl":
            item["version"] = "1.13.0"
    captured = build_inventory(_raw(probe), image=SFT_IMAGE, python_executable=PYTHON)
    assert captured.runtime["trl_version"] == "1.13.0"
    parse_runtime_inventory(captured.payload, image=SFT_IMAGE)


def test_duplicate_normalized_distribution_fails_closed() -> None:
    probe = _probe_from_inventory(SFT_INVENTORY)
    probe["distributions"].append(  # type: ignore[union-attr]
        {"name": "Unsloth-Zoo", "version": "2026.10.2", "location": "/usr/lib/python3/dist-packages/x"}
    )
    with pytest.raises(InventoryCaptureError, match="installed more than once"):
        build_inventory(_raw(probe), image=SFT_IMAGE, python_executable=PYTHON)


@pytest.mark.parametrize("package", ["torch", "transformers", "trl", "unsloth", "unsloth-zoo"])
def test_missing_fact_distribution_fails_closed(package: str) -> None:
    probe = _probe_from_inventory(SFT_INVENTORY)
    probe["distributions"] = [  # type: ignore[assignment]
        item for item in probe["distributions"]  # type: ignore[union-attr]
        if item["name"].replace("_", "-").lower() != package
    ]
    with pytest.raises(InventoryCaptureError, match=package):
        build_inventory(_raw(probe), image=SFT_IMAGE, python_executable=PYTHON)


@pytest.mark.parametrize("mutate", [
    lambda probe: probe.update(executable="/usr/bin/python3"),
    lambda probe: probe.update(schema_version="other/v1"),
    lambda probe: probe.update(extra=1),
    lambda probe: probe["runtime"].update(cuda_build=None),
    lambda probe: probe["runtime"].pop("libc"),
    lambda probe: probe["runtime"].update(trl_version="1.0"),
    lambda probe: probe["distributions"].append({"name": "x", "version": ""}),
    lambda probe: probe["distributions"].append({"name": None, "version": "1", "location": ""}),
])
def test_malformed_probe_output_fails_closed(mutate) -> None:
    probe = _probe_from_inventory(SFT_INVENTORY)
    mutate(probe)
    with pytest.raises(InventoryCaptureError):
        build_inventory(_raw(probe), image=SFT_IMAGE, python_executable=PYTHON)


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b"x" * (4 * 1024 * 1024 + 1)])
def test_unparseable_probe_output_fails_closed(raw: bytes) -> None:
    with pytest.raises(InventoryCaptureError):
        build_inventory(raw, image=SFT_IMAGE, python_executable=PYTHON)


@pytest.mark.parametrize("image", ["unsloth/unsloth:latest", "unsloth/unsloth@sha256:abc", ""])
def test_inventory_image_must_be_immutable(image: str) -> None:
    with pytest.raises(InventoryCaptureError):
        build_inventory(_raw(_probe_from_inventory(SFT_INVENTORY)), image=image,
                        python_executable=PYTHON)


def test_probe_command_never_pulls_and_disables_network() -> None:
    argv = probe_command(docker="docker", image="sha256:" + "a" * 64, python_executable=PYTHON)
    assert argv[:2] == ["docker", "run"]
    assert argv[argv.index("--pull") + 1] == "never"
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--entrypoint") + 1] == PYTHON
    assert argv[-3:] == ["-I", "-c", PROBE_SOURCE]


@pytest.mark.parametrize("python", ["python3", "/opt/../bin/python", "/opt/py thon"])
def test_probe_command_requires_absolute_clean_interpreter(python: str) -> None:
    with pytest.raises(InventoryCaptureError):
        probe_command(docker="docker", image="img@sha256:" + "a" * 64, python_executable=python)


def test_probe_source_runs_and_matches_its_schema() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", PROBE_SOURCE], capture_output=True, check=True, timeout=120
    )
    document = json.loads(completed.stdout)
    assert document["schema_version"] == PROBE_SCHEMA
    assert document["executable"] == sys.executable
    assert set(document["runtime"]) == set(PLATFORM_FACTS)
    assert document["distributions"]
    assert all(set(item) == {"name", "version", "location"} for item in document["distributions"])


def test_diff_names_changed_distributions_and_facts() -> None:
    expected = SFT_INVENTORY.read_bytes()
    probe = _probe_from_inventory(SFT_INVENTORY)
    for item in probe["distributions"]:  # type: ignore[union-attr]
        if item["name"] == "trl":
            item["version"] = "1.13.0"
    actual = build_inventory(_raw(probe), image=SFT_IMAGE, python_executable=PYTHON).payload
    lines = diff_inventories(expected, actual)
    assert "distribution trl: trl 0.24.0 -> trl 1.13.0" in lines
    assert "runtime trl_version: '0.24.0' -> '1.13.0'" in lines
    assert diff_inventories(expected, expected) == []


# --- Profile mode: base overlay -------------------------------------------

GRPO_INVENTORY = ROOT / "Trainers" / "runtime_profiles" / "qwen35-env-grpo-v1.inventory.json"
GRPO_STACK = {"trl": "1.13.0", "datasets": "4.8.5", "unsloth": "2026.10.2", "unsloth-zoo": "2026.10.2"}
RUNTIME_CLOSURE = {"modal": "1.5.4", "grpclib": "0.4.9", "h2": "4.4.1"}


def _derived_probe(stack: dict[str, str], additive: dict[str, str]) -> dict[str, object]:
    probe = _probe_from_inventory(SFT_INVENTORY, reverse=False)
    for item in probe["distributions"]:  # type: ignore[union-attr]
        normalized = item["name"].replace("_", "-").lower()
        if normalized in stack:
            item["version"] = stack[normalized]
    for name, version in additive.items():
        probe["distributions"].append(  # type: ignore[union-attr]
            {"name": name, "version": version, "location": f"/x/{name}.dist-info"})
    return probe


def test_overlay_reproduces_checked_in_grpo_inventory() -> None:
    restricted, changed = overlay_probe(
        _raw(_probe_from_inventory(SFT_INVENTORY)),
        _raw(_derived_probe(GRPO_STACK, RUNTIME_CLOSURE)),
        declared={**GRPO_STACK, **RUNTIME_CLOSURE},
    )
    assert changed == {
        "datasets": ("4.3.0", "4.8.5"), "trl": ("0.24.0", "1.13.0"),
        "unsloth": ("2026.9.7", "2026.10.2"), "unsloth-zoo": ("2026.9.6", "2026.10.2"),
    }
    captured = build_inventory(restricted, image=SFT_IMAGE, python_executable=PYTHON)
    assert captured.payload == GRPO_INVENTORY.read_bytes()


def test_overlay_of_additive_closure_only_is_the_base_inventory() -> None:
    restricted, changed = overlay_probe(
        _raw(_probe_from_inventory(SFT_INVENTORY)),
        _raw(_derived_probe({}, RUNTIME_CLOSURE)),
        declared=RUNTIME_CLOSURE,
    )
    assert changed == {}
    captured = build_inventory(restricted, image=SFT_IMAGE, python_executable=PYTHON)
    assert captured.payload == SFT_INVENTORY.read_bytes()


@pytest.mark.parametrize("case,match", [
    ("undeclared-addition", "added undeclared"),
    ("undeclared-change", "changed undeclared"),
    ("wrong-pin", "pinned to"),
    ("removed", "removed base"),
    ("interpreter", "different interpreters"),
])
def test_overlay_fails_closed(case: str, match: str) -> None:
    declared = {**GRPO_STACK, **RUNTIME_CLOSURE}
    stack, additive = dict(GRPO_STACK), dict(RUNTIME_CLOSURE)
    if case == "undeclared-addition":
        additive["surprise"] = "1.0"
    elif case == "undeclared-change":
        stack["peft"] = "0.22.0"
    elif case == "wrong-pin":
        stack["trl"] = "1.12.0"
    derived = _derived_probe(stack, additive)
    if case == "removed":
        derived["distributions"] = [  # type: ignore[assignment]
            item for item in derived["distributions"]  # type: ignore[union-attr]
            if item["name"] != "peft"
        ]
    elif case == "interpreter":
        derived["executable"] = "/usr/bin/python3"
    with pytest.raises(InventoryCaptureError, match=match):
        overlay_probe(_raw(_probe_from_inventory(SFT_INVENTORY)), _raw(derived), declared=declared)


def test_capture_dockerfile_uses_packaged_install_and_pip_check_gate() -> None:
    text = capture_dockerfile(
        base_image="docker.io/unsloth/unsloth@sha256:" + "a" * 64, python_executable=PYTHON,
    )
    assert text.startswith("FROM docker.io/unsloth/unsloth@sha256:")
    assert "-m pip install --no-index --no-deps --require-hashes --no-cache-dir" in text
    assert text.count("-m pip check") == 2
    assert 'test "$before" = "$after"' in text
    with pytest.raises(InventoryCaptureError):
        capture_dockerfile(base_image="x", python_executable="python3")
