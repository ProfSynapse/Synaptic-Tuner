from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

from Trainers.cloud.scripts.manage_space import (
    _apply_runtime_settings,
    _apply_space_variables,
    _parse_key_value,
    render_space_template,
)
from Trainers.cloud.spaces.vllm_warm.sync_bucket_prefix import parse_bucket_uri


def test_parse_key_value_accepts_equals_in_value():
    assert _parse_key_value("BASE_MODEL=google/gemma-4-E4B-it") == ("BASE_MODEL", "google/gemma-4-E4B-it")
    assert _parse_key_value("LIMIT_MM_PER_PROMPT=image=0 audio=0") == ("LIMIT_MM_PER_PROMPT", "image=0 audio=0")


def test_parse_bucket_uri_splits_bucket_and_prefix():
    bucket_id, prefix = parse_bucket_uri(
        "hf://buckets/professorsynapse/toolset-training-artifacts/runs/hf_jobs/sft/example/final_model"
    )
    assert bucket_id == "professorsynapse/toolset-training-artifacts"
    assert prefix == "runs/hf_jobs/sft/example/final_model"


def test_render_space_template_writes_expected_files(tmp_path):
    render_space_template(
        template="vllm_warm",
        output_dir=tmp_path,
        base_image="ghcr.io/profsynapse/gemma4-vllm-eval:latest",
        app_port=7860,
        space_title="Warm vLLM Eval",
        space_emoji="🔥",
        color_from="indigo",
        color_to="blue",
    )

    dockerfile = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    entrypoint_contents = (tmp_path / "entrypoint.sh").read_text(encoding="utf-8")
    entrypoint = tmp_path / "entrypoint.sh"
    sync_script = tmp_path / "sync_bucket_prefix.py"

    assert "FROM ghcr.io/profsynapse/gemma4-vllm-eval:latest" in dockerfile
    assert "ENTRYPOINT []" in dockerfile
    assert '/opt/bucket-sync-venv/bin/pip install "huggingface_hub[hf_transfer]>=1.0.0"' in dockerfile
    assert 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"' in entrypoint_contents
    assert 'BUCKET_SYNC_PYTHON="${BUCKET_SYNC_PYTHON:-/opt/bucket-sync-venv/bin/python}"' in entrypoint_contents
    assert 'CMD+=(--limit-mm-per-prompt "$LIMIT_MM_JSON")' in entrypoint_contents
    assert "sdk: docker" in readme
    assert "app_port: 7860" in readme
    assert entrypoint.exists()
    assert sync_script.exists()


def test_apply_runtime_settings_can_disable_dev_mode():
    api = Mock()
    args = Namespace(
        space_id="professorsynapse/vllm-warm-eval",
        token="token",
        storage=None,
        hardware=None,
        sleep_time=None,
        dev_mode=False,
        disable_dev_mode=True,
    )

    _apply_runtime_settings(api, args)

    api.disable_space_dev_mode.assert_called_once_with("professorsynapse/vllm-warm-eval", token="token")


def test_apply_space_variables_can_unset_and_set_values():
    api = Mock()
    args = Namespace(
        space_id="professorsynapse/vllm-warm-eval",
        token="token",
        unset_var=["ADAPTER_BUCKET_URI"],
        var=[("MAX_MODEL_LEN", "4096")],
    )

    _apply_space_variables(api, args)

    api.delete_space_variable.assert_called_once_with("professorsynapse/vllm-warm-eval", "ADAPTER_BUCKET_URI", token="token")
    api.add_space_variable.assert_called_once_with("professorsynapse/vllm-warm-eval", "MAX_MODEL_LEN", "4096", token="token")
