"""Exact interpreter selection for the owned vLLM process."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

from Evaluator import vllm_runtime as runtime


def _network(**changes) -> runtime.VLLMStartupSpec:
    values = {
        "source": runtime.ExplicitNetworkVLLMSource("org/model", "a" * 40),
        "served_model_name": "served",
        "startup_timeout_s": 1,
        "readiness_request_timeout_s": 0.1,
    }
    values.update(changes)
    return runtime.VLLMStartupSpec(**values)


def test_default_interpreter_is_current_interpreter(tmp_path: Path):
    spec = _network()
    projection = runtime._projection(spec, cwd=tmp_path, environment={})
    assert spec.python_executable == sys.executable
    assert projection.argv[:3] == (
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
    )


@pytest.mark.parametrize(
    "value",
    (
        True,
        "",
        "python",
        "./python",
        "/runtime/../python",
        "/runtime/./python",
        "/runtime//python",
        "/runtime/python\nother",
        "/runtime/py\0thon",
        "/" + "é" * 2048 + "x",
        "/runtime/\ud800",
    ),
)
def test_invalid_interpreter_is_rejected_before_port_or_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
):
    calls: list[str] = []
    monkeypatch.setattr(
        runtime, "_port_available", lambda *args: calls.append("port") or True
    )
    monkeypatch.setattr(
        runtime, "_spawn", lambda *args, **kwargs: calls.append("spawn")
    )
    with pytest.raises((TypeError, ValueError)):
        runtime.start_vllm_runtime(
            replace(_network(), python_executable=value),
            cwd=tmp_path,
            environment={},
        )
    assert calls == []


def test_selected_interpreter_is_exact_for_network_projection(tmp_path: Path):
    selected = "/opt/chat-runtime/bin/python3.11"
    projection = runtime._projection(
        _network(python_executable=selected), cwd=tmp_path, environment={}
    )
    assert projection.argv[0] == selected


@pytest.mark.parametrize("kind", ("full", "lora"))
def test_selected_interpreter_is_exact_for_local_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
):
    class Target:
        def validate(self):
            return None

    monkeypatch.setattr(runtime, "ServingTarget", Target)
    target = Target()
    target.retrieved = type("Retrieved", (), {"root": tmp_path, "attempt": "attempt"})()
    target.base_snapshot = (
        None
        if kind == "full"
        else type("Snapshot", (), {"root": tmp_path, "snapshot": "base"})()
    )
    selected = "/opt/chat-runtime/bin/python"
    projection = runtime._projection(
        _network(
            source=runtime.VerifiedLocalVLLMSource(target),
            python_executable=selected,
        ),
        cwd=tmp_path,
        environment={},
    )
    assert projection.argv[0] == selected
    if kind == "lora":
        assert "--enable-lora" in projection.argv
    else:
        assert "--enable-lora" not in projection.argv
