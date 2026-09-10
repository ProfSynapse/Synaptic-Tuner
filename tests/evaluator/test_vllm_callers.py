from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Evaluator import interactive_cli


class Lease:
    served_model_name = "finetuned"

    def __init__(self, close_result: bool = True) -> None:
        self.close_calls = 0
        self.close_result = close_result

    def __enter__(self):
        return self

    def close(self) -> bool:
        if self.close_calls == 0:
            self.close_calls = 1
        return self.close_result

    def __exit__(self, *args: object) -> bool:
        result = self.close()
        if not result and args[0] is None:
            raise RuntimeError("cleanup unresolved")
        return False


def _interactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    running: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
):
    prompt = tmp_path / "prompts.yaml"
    prompt.write_text("prompts: []")
    status = SimpleNamespace(
        is_installed=True,
        version="fixture",
        cuda_available=True,
        server_running=running,
        server_url="http://127.0.0.1:8000",
    )
    status_calls = []
    monkeypatch.setattr(
        "Evaluator.vllm_setup.get_vllm_status",
        lambda **kwargs: status_calls.append(kwargs) or status,
    )
    monkeypatch.setattr("Evaluator.vllm_setup.format_gpu_info", lambda value: "fixture")
    monkeypatch.setattr(
        "Evaluator.vllm_setup.resolve_tensor_parallel_size", lambda model: 1
    )
    monkeypatch.setattr(
        "Evaluator.vllm_setup.resolve_tokenizer_mode", lambda model: None
    )
    monkeypatch.setattr(
        "Evaluator.vllm_setup.network_runtime_environment", lambda: {"PATH": "/bin"}
    )
    monkeypatch.setattr(interactive_cli, "_resolve_input", lambda value, args: prompt)
    monkeypatch.setattr(
        interactive_cli, "_resolve_output_dir", lambda args: tmp_path / "results"
    )
    monkeypatch.setattr(
        interactive_cli,
        "_select_vllm_model",
        lambda args, value: ("org/base", "base", "/private/adapter"),
    )
    monkeypatch.setattr(
        interactive_cli,
        "build_settings_kwargs",
        lambda args: {"host": host, "port": port},
    )
    args = SimpleNamespace(prompt_set=str(prompt), runs=1, timeout=1, retries=0)
    args.status_calls = status_calls
    return args


@pytest.mark.parametrize(
    "failure", ["settings", "client", "evaluation", "interrupt", "success"]
)
def test_interactive_owned_runtime_closes_on_every_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    args = _interactive(tmp_path, monkeypatch)
    lease = Lease()
    captured = {}
    monkeypatch.setattr(
        interactive_cli,
        "start_vllm_runtime",
        lambda spec, **kwargs: captured.update(spec=spec, kwargs=kwargs) or lease,
    )
    if failure == "settings":
        monkeypatch.setattr(
            interactive_cli,
            "VLLMSettings",
            lambda **kwargs: (_ for _ in ()).throw(ValueError("settings")),
        )
    else:
        monkeypatch.setattr(interactive_cli, "VLLMSettings", lambda **kwargs: kwargs)
        monkeypatch.setattr(
            interactive_cli,
            "VLLMClient",
            lambda **kwargs: (
                (_ for _ in ()).throw(ValueError("client"))
                if failure == "client"
                else object()
            ),
        )
        outcome = 0 if failure == "success" else ValueError("evaluation")
        if failure == "interrupt":
            outcome = KeyboardInterrupt()
        monkeypatch.setattr(
            interactive_cli,
            "_run_evaluation_loop",
            lambda **kwargs: (
                (_ for _ in ()).throw(outcome)
                if isinstance(outcome, BaseException)
                else outcome
            ),
        )
    if failure == "success":
        assert interactive_cli._run_vllm_evaluation(args) == 0
    else:
        with pytest.raises((ValueError, KeyboardInterrupt)):
            interactive_cli._run_vllm_evaluation(args)
    assert lease.close_calls == 1
    assert captured["spec"].served_model_name == "finetuned"


def test_interactive_never_stops_existing_external_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _interactive(tmp_path, monkeypatch, running=True)
    starts = []
    monkeypatch.setattr(
        interactive_cli, "start_vllm_runtime", lambda *a, **k: starts.append(1)
    )
    monkeypatch.setattr(interactive_cli, "VLLMSettings", lambda **kwargs: kwargs)
    monkeypatch.setattr(interactive_cli, "VLLMClient", lambda **kwargs: object())
    monkeypatch.setattr(interactive_cli, "_run_evaluation_loop", lambda **kwargs: 0)
    assert interactive_cli._run_vllm_evaluation(args) == 0
    assert starts == []


def test_interactive_binds_status_and_runtime_to_configured_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _interactive(tmp_path, monkeypatch, host="127.0.0.1", port=9123)
    lease = Lease()
    specs = []
    monkeypatch.setattr(
        interactive_cli,
        "start_vllm_runtime",
        lambda spec, **kwargs: specs.append(spec) or lease,
    )
    monkeypatch.setattr(interactive_cli, "VLLMSettings", lambda **kwargs: kwargs)
    monkeypatch.setattr(interactive_cli, "VLLMClient", lambda **kwargs: object())
    monkeypatch.setattr(interactive_cli, "_run_evaluation_loop", lambda **kwargs: 0)
    assert interactive_cli._run_vllm_evaluation(args) == 0
    assert args.status_calls == [{"host": "127.0.0.1", "port": 9123}]
    assert (specs[0].host, specs[0].port) == ("127.0.0.1", 9123)


def test_vllm_setup_helpers_are_explicit_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Evaluator import vllm_setup

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("HF_API_KEY", "   ")
    expected = {
        "PATH": "/bin",
        "LD_LIBRARY_PATH": "/lib",
        "CUDA_HOME": "/cuda",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/cache/hf",
        "SSL_CERT_FILE": "/tls/ca.pem",
        "HTTPS_PROXY": "http://proxy",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PYTHONPATH", "/inject")
    monkeypatch.setenv("UNRELATED_SECRET", "never")
    environment = vllm_setup.network_runtime_environment()
    assert environment["HF_TOKEN"] == "token"
    assert "HF_API_KEY" not in environment
    assert "UNRELATED_SECRET" not in environment
    assert "PYTHONPATH" not in environment
    assert all(environment[name] == value for name, value in expected.items())
    assert vllm_setup.resolve_tensor_parallel_size("model", requested=3) == 3
    with pytest.raises(ValueError):
        vllm_setup.resolve_tensor_parallel_size("model", requested=-1)


def _cloud_args(tmp_path: Path):
    return SimpleNamespace(
        output_root=str(tmp_path / "output"),
        bucket_id="bucket",
        run_prefix="runs/run",
        eval_prefix="runs/run/eval",
        vllm_timeout=10,
        vllm_host="127.0.0.1",
        vllm_port=8000,
        vllm_gpu_memory_utilization=0.8,
        vllm_tensor_parallel_size=1,
        config_dir="config",
        preset=None,
        scenarios=None,
        tags=None,
        env_backend="none",
        env_template=None,
        env_tool_schema=None,
        env_exec_config=None,
        upload_to_hf="org/output",
        update_model_card=False,
        with_loss=True,
        loss_workers=0,
    )


@pytest.mark.parametrize("failure", ["stop", "sync", "close", "loss"])
def test_cloud_runtime_closes_even_when_progress_shutdown_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from Evaluator import cloud_hf_job_vllm as cloud

    lease = Lease(close_result=failure != "close")
    events: list[str] = []
    cli_calls: list[list[str]] = []
    runtime_specs = []

    class Logger:
        job_ref = "job"

        def emit(self, *args: object, **kwargs: object) -> None:
            pass

        def emit_failure(self, *args: object, **kwargs: object) -> None:
            pass

        def emit_sync(self, *args: object, **kwargs: object) -> None:
            pass

    class Syncer:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            events.append("stop")
            if failure == "stop":
                raise RuntimeError("stop failed")

        def sync_once(self) -> None:
            events.append("sync")
            if failure == "sync":
                raise RuntimeError("sync failed")

    args = _cloud_args(tmp_path)
    args.with_loss = failure in {"close", "loss"}
    monkeypatch.setattr(cloud, "_parse_args", lambda: args)
    monkeypatch.setattr(cloud, "get_hf_token", lambda: "token")
    for name in (
        "_log_runtime_versions",
        "_persist_source_lock",
        "apply_stage_logging_env",
        "_install_termination_handler",
    ):
        monkeypatch.setattr(cloud, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cloud, "detect_cloud_job_ref", lambda: "job")
    monkeypatch.setattr(cloud, "StageLogger", lambda *args, **kwargs: Logger())
    monkeypatch.setattr(cloud, "_sync_from_bucket", lambda *args, **kwargs: None)
    monkeypatch.setattr(cloud, "_sync_bucket", lambda *args, **kwargs: None)
    monkeypatch.setattr(cloud, "_finalize_cloud_exit_code", lambda code, path: code)
    monkeypatch.setattr(cloud, "_load_base_model_name", lambda path: "org/base")
    monkeypatch.setattr(
        cloud, "_PeriodicBucketSyncer", lambda *args, **kwargs: Syncer()
    )
    monkeypatch.setattr(
        cloud,
        "start_vllm_runtime",
        lambda spec, **kwargs: runtime_specs.append(spec) or lease,
    )
    monkeypatch.setattr(
        cloud.vllm_setup, "resolve_tensor_parallel_size", lambda *args: 1
    )
    monkeypatch.setattr(cloud.vllm_setup, "resolve_tokenizer_mode", lambda model: None)
    monkeypatch.setattr(cloud.vllm_setup, "network_runtime_environment", lambda: {})
    monkeypatch.setattr(
        cloud, "evaluator_main", lambda args: cli_calls.append(args) or 0
    )
    monkeypatch.setattr(
        cloud,
        "_compute_exact_loss_outputs",
        lambda **kwargs: events.append(f"loss-after-close-{lease.close_calls}"),
    )
    if failure == "loss":
        assert cloud.main() == 0
        assert "loss-after-close-1" in events
    elif failure == "close":
        with pytest.raises(RuntimeError, match="cleanup unresolved"):
            cloud.main()
        assert all(not event.startswith("loss-") for event in events)
    else:
        with pytest.raises(RuntimeError, match=failure):
            cloud.main()
        assert all(not event.startswith("loss-") for event in events)
    assert lease.close_calls == 1
    assert len(cli_calls) == 1
    assert runtime_specs[0].served_model_name == "finetuned"
    assert "--hf-token" not in cli_calls[0]
    assert "token" not in cli_calls[0]
