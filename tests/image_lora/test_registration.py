"""image_lora is registered in the TRAINING_METHODS SSOT and the method-label layer."""
from __future__ import annotations

from pathlib import Path

import yaml

from shared.utilities.paths import CANONICAL_OUTPUT_DIRS, TRAINING_METHODS

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_image_lora_in_training_methods_ssot():
    assert "image_lora" in TRAINING_METHODS
    assert CANONICAL_OUTPUT_DIRS["image_lora"] == "image_lora_output"


def test_image_lora_has_menu_label():
    labels = yaml.safe_load((REPO_ROOT / "Trainers" / "methods.yaml").read_text())["method_labels"]
    assert "image_lora" in labels


def test_dispatch_convention_files_exist():
    trainer = REPO_ROOT / "Trainers" / "image_lora"
    assert (trainer / "train_image_lora.py").is_file()
    assert (trainer / "configs" / "config.yaml").is_file()


def test_output_dir_is_gitignored():
    assert "image_lora_output/" in (REPO_ROOT / ".gitignore").read_text().splitlines()


def test_no_arguments_only_plans(monkeypatch):
    """The local menu runs the script bare; that must never touch the cloud."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_image_lora", REPO_ROOT / "Trainers" / "image_lora" / "train_image_lora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called = []
    monkeypatch.setattr(module, "_cmd_plan", lambda args: called.append(args.command) or 0)
    for name in ("_cmd_launch", "_cmd_run", "_cmd_probe"):
        monkeypatch.setattr(module, name, lambda args: (_ for _ in ()).throw(AssertionError("cloud")))
    assert module.main([]) == 0
    assert called == ["plan"]
