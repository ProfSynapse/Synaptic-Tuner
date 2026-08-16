from __future__ import annotations

from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from synaptic_tuner.api.v1 import CapabilityDescriptor
from tuner.capabilities import CapabilityRegistry, builtin_registry, validate_descriptor
from tuner.handlers.capabilities_handler import CapabilitiesHandler


EXPECTED_IDS = {
    "training.local-run",
    "experiment.run",
    "mechinterp.steer",
    "evaluation.run",
    "generation.batch",
    "cloud.launch",
    "cloud.inspect",
}

EXPECTED_DESCRIPTOR_MATRIX = {
    "training.local-run": {
        "inputs": {"job_config": True},
        "effects": (True, True, "required", False, False),
        "confirmation": {"required": True, "reason": "gpu"},
        "resumable": True,
        "available": True,
        "dry_run": True,
    },
    "experiment.run": {
        "inputs": {"experiment_spec": True},
        "effects": (True, True, "required", "possible", False),
        "confirmation": {"required": True, "reason": "gpu_or_paid_compute"},
        "resumable": True,
        "available": True,
        "dry_run": False,
    },
    "mechinterp.steer": {
        "inputs": {"config": True, "model": True},
        "effects": (True, True, "required", "possible", False),
        "confirmation": {"required": True, "reason": "gpu"},
        "resumable": True,
        "available": True,
        "dry_run": False,
    },
    "evaluation.run": {
        "inputs": {},
        "effects": (True, True, "optional", "possible", False),
        "confirmation": {"required": True, "reason": "evaluation_execution"},
        "resumable": False,
        "available": True,
        "dry_run": False,
    },
    "generation.batch": {
        "inputs": {"prompts": True, "model": True, "out_dir": True, "json_schema": False},
        "effects": (True, True, "optional", False, False),
        "confirmation": {"required": False},
        "resumable": True,
        "available": True,
        "dry_run": False,
    },
    "cloud.launch": {
        "inputs": {"job_config": True},
        "effects": (True, True, "required", "required", False),
        "confirmation": {"required": True, "reason": "paid_compute"},
        "resumable": False,
        "available": False,
        "dry_run": False,
    },
    "cloud.inspect": {
        "inputs": {"run": False, "eval_run": False},
        "effects": (True, True, "none", False, False),
        "confirmation": {"required": False},
        "resumable": False,
        "available": True,
        "dry_run": False,
    },
}


def test_builtin_registry_is_public_v1_valid_and_deterministic() -> None:
    first = builtin_registry().list()
    second = builtin_registry().list()
    assert all(isinstance(item, CapabilityDescriptor) for item in first)
    assert [item.id for item in first] == sorted(EXPECTED_IDS)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    for item in first:
        assert validate_descriptor(item) == item.to_dict()


def test_builtin_effect_and_support_contracts_are_explicit() -> None:
    registry = builtin_registry()
    for descriptor in registry.list():
        assert set(descriptor.effects) >= {
            "filesystem_write",
            "network",
            "gpu",
            "paid_compute",
            "external_publish",
        }
        assert set(descriptor.supports) >= {
            "available",
            "dry_run",
            "json_result",
            "jsonl_events",
        }
        expected = EXPECTED_DESCRIPTOR_MATRIX[descriptor.id]
        assert {item["name"]: item["required"] for item in descriptor.inputs} == expected["inputs"]
        assert (
            descriptor.effects["filesystem_write"],
            descriptor.effects["network"],
            descriptor.effects["gpu"],
            descriptor.effects["paid_compute"],
            descriptor.effects["external_publish"],
        ) == expected["effects"]
        assert descriptor.confirmation == expected["confirmation"]
        assert descriptor.resumable is expected["resumable"]
        assert descriptor.supports == {
            "available": expected["available"],
            "dry_run": expected["dry_run"],
            "json_result": False,
            "jsonl_events": False,
        }


def test_cloud_launch_does_not_promise_unwired_source_lock() -> None:
    launch = builtin_registry().describe("cloud.launch")
    assert launch.supports["available"] is False
    assert {item["kind"] for item in launch.outputs} == {"cloud_run"}
    assert "source_lock" not in str(launch.to_dict())


def test_discovery_does_not_import_runtime_or_provider_handlers() -> None:
    import subprocess
    import sys

    script = """
import json, sys
from tuner.capabilities import builtin_registry
builtin_registry().list()
blocked = [name for name in sys.modules if name.startswith(('torch', 'transformers', 'huggingface_hub', 'tuner.handlers.cloud_'))]
print(json.dumps(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert completed.stdout.strip() == "[]"


def test_registry_rejects_duplicate_ids() -> None:
    descriptor = builtin_registry().describe("evaluation.run")
    with pytest.raises(ValueError, match="Duplicate capability id"):
        CapabilityRegistry((descriptor, descriptor))


def test_describe_unknown_id_fails_closed() -> None:
    with pytest.raises(KeyError, match="Unknown capability"):
        builtin_registry().describe("missing.operation")


def test_handler_lists_and_describes_without_executing_commands(capsys) -> None:
    list_args = Namespace(subcommand="list", json=True, events=None)
    assert CapabilitiesHandler(list_args).handle() == 0
    list_output = capsys.readouterr()
    assert list_output.err == ""
    import json

    listed = json.loads(list_output.out)
    assert listed["schema_version"] == "synaptic-result/v1"
    assert listed["data"]["count"] == len(EXPECTED_IDS)

    describe_args = Namespace(
        subcommand="describe",
        capability_id="mechinterp.steer",
        json=True,
        events=None,
    )
    assert CapabilitiesHandler(describe_args).handle() == 0
    described = json.loads(capsys.readouterr().out)
    assert described["data"]["capability"]["id"] == "mechinterp.steer"


def test_handler_unknown_descriptor_keeps_diagnostic_off_stdout(capsys) -> None:
    args = Namespace(subcommand="describe", capability_id="missing", json=True, events=None)
    assert CapabilitiesHandler(args).handle() == 2
    captured = capsys.readouterr()
    import json

    assert json.loads(captured.out)["success"] is False
    assert json.loads(captured.err)["details"]["code"] == "CAPABILITY_NOT_FOUND"


def test_handler_jsonl_mode_emits_only_one_final_event(capsys) -> None:
    args = Namespace(subcommand="list", json=False, events="jsonl")
    assert CapabilitiesHandler(args).handle() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    import json

    event = json.loads(lines[0])
    assert event["schema_version"] == "synaptic-event/v1"
    assert event["final"] is True
    assert event["result"]["schema_version"] == "synaptic-result/v1"


def test_describe_events_jsonl_preserves_descriptor_access_from_unrelated_cwd(
    tmp_path,
) -> None:
    engine_root = Path(__file__).parents[2].resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(engine_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tuner",
            "capabilities",
            "describe",
            "training.local-run",
            "--events",
            "jsonl",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["schema_version"] == "synaptic-event/v1"
    assert event["final"] is True
    descriptor = event["result"]["data"]["capability"]
    assert descriptor["id"] == "training.local-run"
    assert descriptor["inputs"][0]["access"] == "read"
