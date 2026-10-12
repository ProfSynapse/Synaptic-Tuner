"""Config -> ai-toolkit job rendering, cost estimate and spend-cap timeout."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import run_config

CONFIG = Path(run_config.TRAINER_DIR) / "configs" / "config.yaml"


def _dataset(tmp_path: Path, prompts: list[str] | None = None) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    for stem in ("a", "b"):
        (images / f"{stem}.png").write_bytes(b"png")
        (images / f"{stem}.txt").write_text("trig, x")
    (tmp_path / "manifest.json").write_text(json.dumps({"trigger": "trig", "name": "demo"}))
    if prompts is not None:
        (tmp_path / "sample_prompts.yaml").write_text(yaml.safe_dump(prompts))
    return tmp_path


def test_default_config_resolves_qwen_image_2512():
    config = run_config.load_run_config(CONFIG)
    assert config["_model"]["repo"] == "Qwen/Qwen-Image-2512"
    assert config["_model"]["arch"] == "qwen_image"


def test_unknown_registry_name_is_rejected(tmp_path):
    bad = tmp_path / "config.yaml"
    data = yaml.safe_load(CONFIG.read_text())
    data["model"]["registry_name"] = "nope"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="unknown model.registry_name"):
        run_config.load_run_config(bad)


def test_rank_override_keeps_alpha_equal():
    config = run_config.load_run_config(CONFIG, overrides={"rank": 64, "steps": 1000})
    assert config["network"] == {**config["network"], "rank": 64, "alpha": 64}
    assert config["training"]["steps"] == 1000


def test_rendered_job_matches_ai_toolkit_schema(tmp_path):
    config = run_config.load_run_config(CONFIG)
    plan = run_config.plan_run(config, dataset_dir=_dataset(tmp_path, ["trig, go"]), run_name="r1")
    process = plan.job["config"]["process"][0]
    assert plan.job["job"] == "extension" and plan.job["config"]["name"] == "r1"
    assert process["type"] == "sd_trainer"
    assert process["model"]["name_or_path"] == "${MODEL_PATH}"
    assert process["model"]["arch"] == "qwen_image"
    assert process["model"]["quantize"] is False
    assert process["datasets"][0]["folder_path"] == "${DATASET_DIR}"
    assert process["datasets"][0]["resolution"] == [768, 1024, 1328]
    assert process["training_folder"] == "${OUTPUT_DIR}"
    assert process["network"] == {"type": "lora", "linear": 32, "linear_alpha": 32}
    assert process["sample"]["prompts"] == ["trig, go"]
    assert process["sample"]["walk_seed"] is False
    assert process["save"]["max_step_saves_to_keep"] == 8
    # Cached text embeddings cannot take an injected trigger; captions carry it.
    assert "trigger_word" not in process
    assert yaml.safe_load(plan.job_yaml) == plan.job


def test_no_sample_prompts_disables_sampling(tmp_path):
    config = run_config.load_run_config(CONFIG)
    plan = run_config.plan_run(config, dataset_dir=_dataset(tmp_path))
    assert plan.job["config"]["process"][0]["train"]["disable_sampling"] is True


def test_dataset_requires_captions(tmp_path):
    root = _dataset(tmp_path)
    (root / "images" / "c.png").write_bytes(b"png")
    with pytest.raises(ValueError, match="without captions"):
        run_config.load_dataset_info(root)


def test_estimate_arithmetic():
    config = run_config.load_run_config(CONFIG, overrides={"steps": 1000, "gpu": "H100"})
    config["estimate"] = {
        "usd_per_hour": {"gpu": {"H100": 4.0}, "cpu_core": 0.05, "memory_gib": 0.01},
        "seconds_per_step": {"H100": 3.6}, "seconds_per_sample_image": {"H100": 36},
        "overhead_minutes": 6,
    }
    config["modal"].update(cpu=8, memory_mib=65536)
    est = run_config.estimate_cost(config, sample_prompt_count=5)
    # 1000 steps * 3.6 s = 1 h; 3 rounds (0, 500, 1000) * 5 * 36 s = 0.15 h; 0.1 h overhead.
    assert est.breakdown["sample_rounds"] == 3
    assert est.hours == pytest.approx(1.25)
    assert est.usd_per_hour == pytest.approx(4.0 + 0.4 + 0.64)
    assert est.usd == pytest.approx(1.25 * 5.04, abs=0.01)


def test_timeout_caps_spend_and_refuses_over_budget():
    est = run_config.Estimate("H100", hours=3.0, usd_per_hour=5.0, usd=15.0)
    assert run_config.timeout_for_budget(est, 20.0) == 4 * 3600
    with pytest.raises(ValueError, match="exceeds"):
        run_config.timeout_for_budget(est, 10.0)
    with pytest.raises(ValueError):
        run_config.timeout_for_budget(est, 0)
