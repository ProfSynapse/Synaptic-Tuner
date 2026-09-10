from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from Evaluator import vllm_runtime as runtime


@pytest.fixture(autouse=True)
def _free_port(monkeypatch):
    monkeypatch.setattr(runtime, "_port_available", lambda *args: True)


class _Process:
    pid = 41
    _pgid = 41
    _start_time = 9

    def __init__(self, closes: bool = True):
        self.cleanup_pending = True
        self.closes = closes
        self.close_calls = 0

    def close(self, **kwargs):
        self.close_calls += 1
        if self.closes:
            self.cleanup_pending = False
        return self.closes

    def __exit__(self, *args):
        self.close()
        return False


def _network(**changes):
    values = dict(
        source=runtime.ExplicitNetworkVLLMSource("org/model", "a" * 40),
        served_model_name="served",
        startup_timeout_s=1,
        readiness_request_timeout_s=0.1,
    )
    values.update(changes)
    return runtime.VLLMStartupSpec(**values)


def test_network_runtime_projects_exact_args_and_returns_lease(tmp_path: Path, monkeypatch):
    process = _Process()
    captured = {}
    monkeypatch.setattr(runtime, "_spawn", lambda argv, **kw: captured.update(argv=argv, **kw) or process)
    monkeypatch.setattr(runtime, "_leader_alive", lambda value: value is process)
    monkeypatch.setattr(runtime, "_ready", lambda *args: args[:3] == ("127.0.0.1", 8000, ("served",)))
    lease = runtime.start_vllm_runtime(_network(), cwd=tmp_path, environment={"PATH": "/bin"})
    assert lease.served_model_name == "served"
    assert captured["cwd"] == tmp_path
    assert captured["environment"] == {"PATH": "/bin"}
    assert captured["argv"][-4:] == ("--model", "org/model", "--revision", "a" * 40)
    assert lease.close()


def test_projection_denials_are_before_spawn(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: calls.append(1))
    bad = (
        _network(host="0.0.0.0"),
        _network(port=True),
        _network(gpu_memory_utilization=float("nan")),
        _network(tensor_parallel_size=0),
        _network(tokenizer_mode="auto"),
        _network(max_lora_rank=True),
        _network(startup_timeout_s=1801),
        _network(readiness_request_timeout_s=0),
    )
    for spec in bad:
        with pytest.raises((TypeError, ValueError)):
            runtime.start_vllm_runtime(spec, cwd=tmp_path, environment={})
    assert calls == []


def test_local_source_requires_exact_serving_target(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: pytest.fail("spawned"))
    spec = _network(source=runtime.VerifiedLocalVLLMSource(object()))
    with pytest.raises(TypeError):
        runtime.start_vllm_runtime(spec, cwd=tmp_path, environment={})


@pytest.mark.parametrize(
    "name",
    ["HF_TOKEN", "HTTPS_PROXY", "PYTHONPATH", "RANDOM_SETTING", "VLLM_API_KEY", "VLLM_PLUGINS"],
)
def test_local_environment_rejects_credentials_proxy_injection_and_unknown(
    tmp_path: Path, monkeypatch, name: str
):
    class Target:
        pass
    monkeypatch.setattr(runtime, "ServingTarget", Target)
    target = Target()
    target.validate = lambda: None
    target.retrieved = type("Retrieved", (), {"root": tmp_path, "attempt": "attempt"})()
    target.base_snapshot = None
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: pytest.fail("spawned"))
    with pytest.raises(ValueError):
        runtime.start_vllm_runtime(
            _network(source=runtime.VerifiedLocalVLLMSource(target)),
            cwd=tmp_path,
            environment={name: "value"},
        )


def test_local_projection_forces_offline_and_binds_lora_paths(tmp_path: Path, monkeypatch):
    class Target:
        pass
    monkeypatch.setattr(runtime, "ServingTarget", Target)
    target = Target()
    validations = []
    target.validate = lambda: validations.append(1)
    target.retrieved = type("Retrieved", (), {"root": tmp_path, "attempt": "attempt"})()
    target.base_snapshot = type("Snapshot", (), {"root": tmp_path, "snapshot": "base"})()
    process = _Process()
    captured = {}
    monkeypatch.setattr(runtime, "_spawn", lambda argv, **kw: captured.update(argv=argv, **kw) or process)
    monkeypatch.setattr(runtime, "_leader_alive", lambda value: True)
    monkeypatch.setattr(runtime, "_ready", lambda *args: True)
    runtime.start_vllm_runtime(
        _network(source=runtime.VerifiedLocalVLLMSource(target)),
        cwd=tmp_path,
        environment={"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"},
    )
    assert captured["environment"]["HF_HUB_OFFLINE"] == "1"
    model_index = captured["argv"].index("--model")
    assert captured["argv"][model_index:model_index + 2] == ("--model", str(tmp_path / "base"))
    assert ("--max-lora-rank", "64") == captured["argv"][-4:-2]
    assert captured["argv"][-2:] == ("--lora-modules", f"served={tmp_path / 'attempt' / 'model'}")
    served_index = captured["argv"].index("--served-model-name")
    assert captured["argv"][served_index + 1] == runtime._LORA_BASE_ALIAS
    assert validations == [1]


def test_timeout_closes_and_unresolved_cleanup_is_retained(tmp_path: Path, monkeypatch):
    process = _Process(closes=False)
    ticks = iter((0.0, 0.0, 0.0, 2.0))
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: process)
    monkeypatch.setattr(runtime, "_leader_alive", lambda value: True)
    monkeypatch.setattr(runtime, "_ready", lambda *args: False)
    monkeypatch.setattr(runtime, "_monotonic", lambda: next(ticks))
    monkeypatch.setattr(runtime, "_sleep", lambda value: None)
    with pytest.raises(runtime.VLLMRuntimeError) as caught:
        runtime.start_vllm_runtime(_network(), cwd=tmp_path, environment={})
    assert caught.value.cleanup_lease is process
    assert process.close_calls == 1


def test_keyboard_interrupt_is_not_masked_by_cleanup(tmp_path: Path, monkeypatch):
    process = _Process(closes=False)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: process)
    monkeypatch.setattr(runtime, "_leader_alive", lambda value: True)
    monkeypatch.setattr(runtime, "_ready", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt) as caught:
        runtime.start_vllm_runtime(_network(), cwd=tmp_path, environment={})
    assert caught.value.cleanup_lease is process


def test_readiness_parser_requires_exact_bounded_model_document(monkeypatch):
    class Response:
        status = 200
        def getheader(self, name): return None
        def read(self, amount): return json.dumps({"object": "list", "data": [{"id": "served"}]}).encode()
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, *args, **kwargs): pass
        def getresponse(self): return Response()
        def close(self): pass
    monkeypatch.setattr(runtime.http.client, "HTTPConnection", Connection)
    assert runtime._ready("127.0.0.1", 8000, ("served",), 1)
    Response.read = lambda self, amount: b"{" + b"x" * runtime._MAX_RESPONSE_BYTES
    assert not runtime._ready("127.0.0.1", 8000, ("served",), 1)


def test_leader_check_reads_identity_without_polling(monkeypatch):
    process = _Process()
    monkeypatch.setattr(runtime, "_identity", lambda pid: ("S", 41, 9))
    assert runtime._leader_alive(process)
    monkeypatch.setattr(runtime, "_identity", lambda pid: ("Z", 41, 9))
    assert not runtime._leader_alive(process)


def test_existing_listener_is_denied_before_spawn(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "_port_available", lambda *args: False)
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: calls.append(1))
    with pytest.raises(runtime.VLLMRuntimeError, match="already in use"):
        runtime.start_vllm_runtime(_network(), cwd=tmp_path, environment={})
    assert calls == []


@pytest.mark.parametrize("mode", ["late", "dead"])
def test_late_or_dead_readiness_is_rejected_and_cleaned(tmp_path: Path, monkeypatch, mode: str):
    process = _Process()
    ticks = iter((0.0, 0.0, 2.0)) if mode == "late" else iter((0.0, 0.0, 0.5))
    alive = iter((True,)) if mode == "late" else iter((True, False))
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: process)
    monkeypatch.setattr(runtime, "_monotonic", lambda: next(ticks))
    monkeypatch.setattr(runtime, "_leader_alive", lambda value: next(alive))
    monkeypatch.setattr(runtime, "_ready", lambda *args: True)
    with pytest.raises(runtime.VLLMRuntimeError, match="timely and live"):
        runtime.start_vllm_runtime(_network(), cwd=tmp_path, environment={})
    assert process.close_calls == 1


def test_network_lora_alias_is_exact_chat_model(tmp_path: Path, monkeypatch):
    source = runtime.ExplicitNetworkVLLMSource(
        "org/model", lora=runtime.ExplicitNetworkLoRA("different", tmp_path / "adapter")
    )
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: pytest.fail("spawned"))
    with pytest.raises(ValueError, match="must equal"):
        runtime.start_vllm_runtime(_network(source=source), cwd=tmp_path, environment={})


@pytest.mark.parametrize("name", ["bad=name", "bad name", "bad\nname"])
def test_lora_alias_grammar_is_closed_before_spawn(tmp_path: Path, monkeypatch, name: str):
    source = runtime.ExplicitNetworkVLLMSource(
        "org/model", lora=runtime.ExplicitNetworkLoRA(name, tmp_path)
    )
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: pytest.fail("spawned"))
    with pytest.raises(ValueError, match="LoRA name"):
        runtime.start_vllm_runtime(
            _network(source=source, served_model_name=name), cwd=tmp_path, environment={}
        )


def test_local_lora_alias_grammar_is_closed_before_spawn(tmp_path: Path, monkeypatch):
    class Target:
        pass
    monkeypatch.setattr(runtime, "ServingTarget", Target)
    target = Target()
    target.validate = lambda: None
    target.retrieved = type("Retrieved", (), {"root": tmp_path, "attempt": "attempt"})()
    target.base_snapshot = type("Snapshot", (), {"root": tmp_path, "snapshot": "base"})()
    calls = []
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: calls.append(1))
    with pytest.raises(ValueError, match="LoRA name"):
        runtime.start_vllm_runtime(
            _network(
                source=runtime.VerifiedLocalVLLMSource(target),
                served_model_name="chat=alien",
            ),
            cwd=tmp_path,
            environment={},
        )
    assert calls == []


def test_runtime_lease_serializes_concurrent_close_and_properties_are_read_only():
    class Process(_Process):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.maximum = 0
        def close(self, **kwargs):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            time.sleep(0.01)
            self.active -= 1
            return super().close(**kwargs)
    process = Process()
    lease = runtime.VLLMRuntimeLease(process, host="127.0.0.1", port=8000, served_model_name="served")
    threads = [threading.Thread(target=lease.close) for _ in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert process.maximum == 1
    assert process.close_calls == 1
    with pytest.raises(AttributeError):
        lease.host = "elsewhere"


def test_unresolved_close_can_be_retried_after_state_improves():
    process = _Process(closes=False)
    lease = runtime.VLLMRuntimeLease(
        process, host="127.0.0.1", port=8000, served_model_name="served"
    )
    assert not lease.close()
    process.closes = True
    assert lease.close()
    assert process.close_calls == 2
