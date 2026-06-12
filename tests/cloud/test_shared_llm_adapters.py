import json
import sys
import types

from Evaluator.config import UnslothSettings
from Evaluator.shared_llm_adapters import SharedUnslothAdapter
from shared.cloud_stage_logging import (
    ENV_STAGE_BUCKET_ID,
    ENV_STAGE_JOB_REF,
    ENV_STAGE_LOG_DIR,
    ENV_STAGE_NAME,
    ENV_STAGE_PROVIDER,
    ENV_STAGE_RUN_PREFIX,
    apply_stage_logging_env,
)


def test_shared_unsloth_adapter_emits_model_load_events(tmp_path, monkeypatch):
    calls = []
    fake_module = types.ModuleType("shared.llm.providers.unsloth")

    class _FakeUnslothClient:
        provider_name = "unsloth"

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def unload(self):
            pass

    fake_module.UnslothClient = _FakeUnslothClient
    monkeypatch.setitem(sys.modules, "shared.llm.providers.unsloth", fake_module)
    for key in (
        ENV_STAGE_NAME,
        ENV_STAGE_PROVIDER,
        ENV_STAGE_RUN_PREFIX,
        ENV_STAGE_JOB_REF,
        ENV_STAGE_BUCKET_ID,
        ENV_STAGE_LOG_DIR,
    ):
        monkeypatch.delenv(key, raising=False)
    apply_stage_logging_env(
        stage="evaluation",
        provider="hf_jobs",
        run_prefix="runs/demo",
        job_ref="job-123",
        bucket_id="bucket-1",
        log_dir=tmp_path / "logs",
    )

    adapter = SharedUnslothAdapter(
        UnslothSettings(
            model=str(tmp_path / "final_model"),
            max_seq_length=2048,
            load_in_4bit=False,
            top_p=0.8,
        )
    )
    adapter.unload()

    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "stage_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == ["model_load_started", "model_load_completed"]
    assert events[0]["details"]["backend"] == "unsloth"
    assert events[0]["details"]["adapter_path"] == str(tmp_path / "final_model")
    assert events[0]["details"]["max_seq_length"] == 2048
    assert events[0]["details"]["load_in_4bit"] is False
    assert "elapsed_seconds" in events[1]["details"]
    assert calls == [
        {
            "adapter_path": str(tmp_path / "final_model"),
            "max_seq_length": 2048,
            "load_in_4bit": False,
            "top_p": 0.8,
        }
    ]
