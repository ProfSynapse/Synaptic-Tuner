"""Native training diagnosis stays read-only through the real CLI router."""

import builtins
import importlib
import json
from importlib import metadata
from unittest.mock import patch

import pytest

from tuner.cli.parser import create_parser
from tuner.cli.router import route_command
from tuner.handlers import native_training_observation as observation


def _run(capsys, *argv):
    result = route_command(create_parser().parse_args(["doctor", "native-training", "--json", *argv]))
    return result, json.loads(capsys.readouterr().out)


def test_real_parser_and_router_on_current_host(capsys):
    code, report = _run(capsys)
    assert code == 0
    assert report["schema_version"] == 1
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["native_apple_candidate"] == (
        report["platform"]["system"].get("value") == "Darwin"
        and report["platform"]["architecture"].get("value", "").lower() == "arm64"
    )
    assert report["platform"]["system"]["status"] == "observed"
    assert set(report["dependencies"]) == {"mlx", "mlx-lm", "numpy", "transformers", "torch"}


def test_darwin_arm64_and_dependency_statuses(capsys, monkeypatch):
    monkeypatch.setattr(observation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(observation.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(observation.platform, "release", lambda: "27.0.0")
    monkeypatch.setattr(observation.platform, "version", lambda: "Darwin")
    monkeypatch.setattr(observation.platform, "mac_ver", lambda: ("27.0.1", (), ""))

    def version(name):
        if name == "mlx":
            return "0.30.0"
        if name == "mlx-lm":
            return "bad\nsecret"
        if name == "numpy":
            return "x" * 10000
        if name == "torch":
            raise RuntimeError("secret exception payload")
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(observation.metadata, "version", version)
    code, report = _run(capsys)
    assert code == 0
    assert report["native_apple_candidate"] is True
    assert report["platform"]["macos_version"] == {"status": "observed", "value": "27.0.1"}
    assert report["dependencies"] == {
        "mlx": {"status": "observed", "value": "0.30.0"},
        "mlx-lm": {"status": "malformed"},
        "numpy": {"status": "malformed"},
        "transformers": {"status": "absent"},
        "torch": {"status": "observation_failed"},
    }
    assert "secret" not in json.dumps(report)


def test_no_ml_import_or_network(capsys, monkeypatch):
    forbidden = {"mlx", "mlx_lm", "numpy", "transformers", "torch", "socket", "urllib", "requests"}
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError("forbidden import")
        return original_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", guarded):
        code, _ = _run(capsys)
    assert code == 0


def test_fix_and_env_file_rejected(capsys):
    for option in (("--fix",), ("--env-file", "unused.env"), ("--env-file", "")):
        code, payload = _run(capsys, *option)
        assert code == 2
        assert payload["code"] == "INVALID_ARGS"


def test_cli_bootstrap_does_not_select_or_load_dotenv(capsys, monkeypatch):
    cli_main = importlib.import_module("tuner.cli.main")

    def forbidden(*args, **kwargs):
        raise AssertionError("dotenv access")

    monkeypatch.setattr(cli_main, "_explicit_env_path", forbidden)
    monkeypatch.setattr(cli_main, "load_env_file", forbidden)
    monkeypatch.setattr(cli_main, "build_project_context", forbidden)
    monkeypatch.setattr(importlib.import_module("tuner.cli.router"), "_default_context", forbidden)
    with pytest.raises(SystemExit) as exit_info:
        cli_main.main(["doctor", "native-training", "--json", "--manifest", "missing/synaptic.yaml"])
    assert exit_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "DIAGNOSTIC_ONLY"


@pytest.mark.parametrize("option", [("--fix",), ("--env-file", "missing.env"), ("--env-file", "")])
def test_cli_rejects_mutation_and_env_selection_before_dotenv(capsys, monkeypatch, option):
    cli_main = importlib.import_module("tuner.cli.main")

    def forbidden(*args, **kwargs):
        raise AssertionError("dotenv access")

    monkeypatch.setattr(cli_main, "_explicit_env_path", forbidden)
    monkeypatch.setattr(cli_main, "load_env_file", forbidden)
    monkeypatch.setattr(cli_main, "build_project_context", forbidden)
    with pytest.raises(SystemExit) as exit_info:
        cli_main.main(["doctor", "native-training", "--json", *option])
    assert exit_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["code"] == "INVALID_ARGS"


def test_failed_platform_observation_does_not_claim_candidate(capsys, monkeypatch):
    def failed():
        raise RuntimeError("private exception")

    monkeypatch.setattr(observation.platform, "system", failed)
    code, report = _run(capsys)
    assert code == 0
    assert report["native_apple_candidate"] is False
    assert report["platform"]["system"] == {"status": "observation_failed"}
    assert "private exception" not in json.dumps(report)


@pytest.mark.parametrize(("system", "architecture", "release"), [
    ("Windows", "AMD64", "11"),
    ("Windows", "ARM64", "11"),
    ("Linux", "x86_64", "6.6.87.2-microsoft-standard-WSL2"),
])
def test_windows_and_wsl_are_reported_without_apple_or_gpu_claims(
    capsys, monkeypatch, system, architecture, release,
):
    monkeypatch.setattr(observation.platform, "system", lambda: system)
    monkeypatch.setattr(observation.platform, "machine", lambda: architecture)
    monkeypatch.setattr(observation.platform, "release", lambda: release)
    code, report = _run(capsys)
    assert code == 0
    assert report["platform"]["system"] == {"status": "observed", "value": system}
    assert report["platform"]["architecture"] == {"status": "observed", "value": architecture}
    assert report["platform"]["release"] == {"status": "observed", "value": release}
    assert report["platform"]["macos_version"] == {"status": "not_applicable"}
    assert report["native_apple_candidate"] is False
    assert report["status"] == "DIAGNOSTIC_ONLY"
