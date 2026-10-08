from __future__ import annotations

import json
from pathlib import Path

from scripts import capture_runtime_profile_inventory as cli


ROOT = Path(__file__).resolve().parents[2]
IMAGE = "unsloth/unsloth@sha256:1644d635bc7c5b57ed64cabbab1ae00647dfb583c2135fdfb6a461aeeba52739"
SFT_INVENTORY = ROOT / "Trainers/runtime_profiles/qwen35-sft-v1.inventory.json"
PYTHON = "/opt/unsloth-venv/bin/python3"


def _probe_output() -> bytes:
    inventory = json.loads(SFT_INVENTORY.read_bytes())
    return json.dumps({
        "schema_version": "synaptic-runtime-inventory-probe/v1",
        "executable": PYTHON,
        "distributions": [dict(item, location="/site") for item in inventory["distributions"]],
        "runtime": {key: inventory["runtime"][key] for key in (
            "architecture", "cuda_build", "libc", "os", "python_implementation", "python_version")},
    }).encode()


def test_image_mode_check_reports_current(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(cli, "_run", lambda argv, **_kw: calls.append(argv) or _probe_output())
    assert cli.main(["--image", IMAGE, "--python", PYTHON, "--check", str(SFT_INVENTORY)]) == 0
    assert "CURRENT" in capsys.readouterr().out
    assert calls[0][calls[0].index("--pull") + 1] == "never"


def test_image_mode_writes_new_file_and_refuses_overwrite(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "_run", lambda argv, **_kw: _probe_output())
    output = tmp_path / "x.inventory.json"
    assert cli.main(["--image", IMAGE, "--python", PYTHON, "--output", str(output)]) == 0
    assert output.read_bytes() == SFT_INVENTORY.read_bytes()
    assert cli.main(["--image", IMAGE, "--python", PYTHON, "--output", str(output)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_check_reports_stale_differences(monkeypatch, tmp_path, capsys) -> None:
    stale = tmp_path / "stale.json"
    document = json.loads(SFT_INVENTORY.read_bytes())
    document["runtime"]["trl_version"] = "9.9.9"
    stale.write_text(json.dumps(document))
    monkeypatch.setattr(cli, "_run", lambda argv, **_kw: _probe_output())
    assert cli.main(["--image", IMAGE, "--python", PYTHON, "--check", str(stale)]) == 1
    assert "runtime trl_version: '9.9.9' -> '0.24.0'" in capsys.readouterr().err


def test_mode_argument_errors_fail_without_docker(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "_run", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("ran")))
    output = str(tmp_path / "out.json")
    assert cli.main(["--image", IMAGE, "--output", output]) == 2
    assert cli.main(["--image", IMAGE, "--python", PYTHON, "--wheel-dir", ".", "--output", output]) == 2
    # The 32k profile installs index packages, not hash-pinned bootstrap wheels.
    legacy = ROOT / "Trainers/image_profiles/qwen35_4b_32k_prompt_completion.yaml"
    assert cli.main(["--image", IMAGE, "--image-profile", str(legacy), "--output", output]) == 2
    grpo = ROOT / "Trainers/image_profiles/qwen35_4b_packaged_env_grpo/profile.yaml"
    other = "unsloth/unsloth@sha256:" + "0" * 64
    assert cli.main(["--image", other, "--image-profile", str(grpo), "--output", output]) == 2
    assert cli.main(["--image", IMAGE, "--image-profile", str(grpo),
                     "--wheel-dir", str(tmp_path), "--output", output]) == 2
    err = capsys.readouterr().err
    assert "--python is required" in err
    assert "is not the profile base" in err
    assert "is not staged" in err
