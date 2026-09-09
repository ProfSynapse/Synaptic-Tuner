"""Owned-process cleanup regressions for vLLM startup."""

import subprocess
import pytest
from Evaluator import vllm_setup


class Process:
    def __init__(self, *, wait_errors=(), terminate_error=None, kill_error=None, returncode=None):
        self.wait_errors = list(wait_errors)
        self.terminate_error, self.kill_error = terminate_error, kill_error
        self.returncode = returncode
        self.terminate_calls, self.wait_calls, self.kill_calls = 0, [], 0

    def terminate(self):
        self.terminate_calls += 1
        if self.terminate_error:
            raise self.terminate_error

    def wait(self, *, timeout):
        self.wait_calls.append(timeout)
        if self.wait_errors:
            error = self.wait_errors.pop(0)
            if error is not None:
                raise error

    def kill(self):
        self.kill_calls += 1
        if self.kill_error:
            raise self.kill_error

    def poll(self):
        return self.returncode


@pytest.fixture(autouse=True)
def clear_managed_process():
    previous = vllm_setup._server_process
    vllm_setup._server_process = None
    yield
    vllm_setup._server_process = previous


def _spawn(monkeypatch, process, readiness):
    monkeypatch.setattr(vllm_setup.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(vllm_setup, "_wait_for_server", readiness)


@pytest.mark.parametrize("timeout", [120, 600])
def test_timeout_cleans_exact_spawned_process(monkeypatch, timeout):
    process, seen = Process(), []
    _spawn(monkeypatch, process, lambda h, p, value, *, show_logs: seen.append(value) or False)
    assert vllm_setup.start_vllm_server("model", timeout=timeout) is False
    assert seen == [timeout]
    assert (process.terminate_calls, process.wait_calls, process.kill_calls) == (1, [10], 0)
    assert vllm_setup._server_process is None


@pytest.mark.parametrize("failure", [RuntimeError("health failure"), KeyboardInterrupt()])
def test_error_or_cancellation_cleans_before_return_or_raise(monkeypatch, failure):
    process = Process()
    def fail(*args, **kwargs):
        raise failure
    _spawn(monkeypatch, process, fail)
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            vllm_setup.start_vllm_server("model")
    else:
        assert vllm_setup.start_vllm_server("model") is False
    assert process.terminate_calls == 1
    assert vllm_setup._server_process is None


def test_success_does_not_stop(monkeypatch):
    process = Process()
    _spawn(monkeypatch, process, lambda *a, **k: True)
    assert vllm_setup.start_vllm_server("model") is True
    assert vllm_setup._server_process is process
    assert (process.terminate_calls, process.kill_calls) == (0, 0)


def test_spawn_error_creates_no_owned_process(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("spawn failure")
    monkeypatch.setattr(vllm_setup.subprocess, "Popen", fail)
    assert vllm_setup.start_vllm_server("model") is False
    assert vllm_setup._server_process is None


@pytest.mark.parametrize("terminate_error,wait_errors", [
    (None, (subprocess.TimeoutExpired("vllm", 10), None)),
    (RuntimeError("terminate failure"), (None,)),
])
def test_cleanup_kill_fallback(monkeypatch, terminate_error, wait_errors):
    process = Process(wait_errors=wait_errors, terminate_error=terminate_error)
    _spawn(monkeypatch, process, lambda *a, **k: False)
    assert vllm_setup.start_vllm_server("model") is False
    assert process.terminate_calls == 1 and process.kill_calls == 1
    assert vllm_setup._server_process is None


@pytest.mark.parametrize("terminate_error,wait_errors", [
    (RuntimeError("terminate failure"), ()),
    (None, (subprocess.TimeoutExpired("vllm", 10),)),
])
def test_failed_kill_retains_handle_for_later_stop_retry(
    monkeypatch, terminate_error, wait_errors,
):
    process = Process(
        wait_errors=wait_errors,
        terminate_error=terminate_error,
        kill_error=RuntimeError("kill failure"),
    )
    _spawn(monkeypatch, process, lambda *a, **k: False)
    assert vllm_setup.start_vllm_server("model") is False
    assert vllm_setup._server_process is process

    process.terminate_error = None
    process.kill_error = None
    process.wait_errors = []
    assert vllm_setup.stop_vllm_server() is True
    assert vllm_setup._server_process is None


def test_kill_reap_timeout_retains_handle(monkeypatch):
    timeout = subprocess.TimeoutExpired("vllm", 10)
    process = Process(wait_errors=(timeout, timeout))
    _spawn(monkeypatch, process, lambda *a, **k: False)
    assert vllm_setup.start_vllm_server("model") is False
    assert process.kill_calls == 1
    assert vllm_setup._server_process is process


def test_keyboard_interrupt_survives_cleanup_failure(monkeypatch):
    process = Process(
        terminate_error=RuntimeError("terminate failure"),
        kill_error=RuntimeError("kill failure"),
    )
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    _spawn(monkeypatch, process, interrupt)
    with pytest.raises(KeyboardInterrupt):
        vllm_setup.start_vllm_server("model")
    assert vllm_setup._server_process is process


def test_concurrent_replacement_does_not_redirect_cleanup(monkeypatch):
    spawned, replacement = Process(), Process()
    monkeypatch.setattr(vllm_setup.subprocess, "Popen", lambda *a, **k: spawned)
    def replace_then_fail(*args, **kwargs):
        vllm_setup._server_process = replacement
        return False
    monkeypatch.setattr(vllm_setup, "_wait_for_server", replace_then_fail)
    assert vllm_setup.start_vllm_server("model") is False
    assert spawned.terminate_calls == 1
    assert (replacement.terminate_calls, replacement.kill_calls) == (0, 0)
    assert vllm_setup._server_process is replacement


def test_existing_managed_process_is_not_replaced_or_stopped(monkeypatch):
    existing, calls = Process(), []
    vllm_setup._server_process = existing
    monkeypatch.setattr(vllm_setup.subprocess, "Popen", lambda *a, **k: calls.append(1))
    assert vllm_setup.start_vllm_server("model") is False
    assert calls == [] and vllm_setup._server_process is existing
    assert (existing.terminate_calls, existing.kill_calls) == (0, 0)


def test_completed_retained_child_is_replaced_without_signalling_it(monkeypatch):
    completed, replacement = Process(returncode=1), Process()
    vllm_setup._server_process = completed
    _spawn(monkeypatch, replacement, lambda *a, **k: True)
    assert vllm_setup.start_vllm_server("model") is True
    assert vllm_setup._server_process is replacement
    assert (completed.terminate_calls, completed.kill_calls) == (0, 0)
