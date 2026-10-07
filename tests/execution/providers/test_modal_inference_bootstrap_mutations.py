"""Mutation and API-surface regressions for the Modal chat bootstrap."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import inspect
import math
from pathlib import Path

import pytest

from tests.execution.providers.modal_inference_worker_fixtures import (
    mounted_launch_case,
)
from tests.execution.providers.test_modal_inference_bootstrap import (
    _Session,
    _install_runtime,
    _open,
)
from tuner.execution.providers.modal import inference_bootstrap as bootstrap
from tuner.execution.providers.modal.inference_bootstrap import (
    ModalInferenceBootstrapError,
    open_modal_chat_worker,
)


def _equal_target(target):
    return replace(target)


def test_equal_metadata_target_substitution_during_second_verifier_is_denied(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    now = [100.0]
    calls = _install_runtime(monkeypatch, now)
    original_worker = bootstrap.prepare_modal_chat_worker
    prepared = []

    def capture(*args, **kwargs):
        value = original_worker(*args, **kwargs)
        prepared.append(value)
        return value

    monkeypatch.setattr(bootstrap, "prepare_modal_chat_worker", capture)
    original_verify = bootstrap.verify_modal_inference_runtime

    def substitute(configuration):
        original_verify(configuration)
        if len(calls["verify"]) == 2:
            value = prepared[0]
            source = replace(
                value.startup.source,
                target=_equal_target(value.startup.source.target),
            )
            object.__setattr__(value, "startup", replace(value.startup, source=source))

    monkeypatch.setattr(bootstrap, "verify_modal_inference_runtime", substitute)
    with pytest.raises(
        ModalInferenceBootstrapError, match="^modal_inference_bootstrap_invalid$"
    ):
        with _open(case):
            pytest.fail("substituted target yielded")
    assert len(calls["verify"]) == 2
    assert calls["start"] == []


def test_equal_metadata_target_substitution_during_startup_is_denied_and_closed(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    now = [100.0]
    calls = _install_runtime(monkeypatch, now)

    @contextmanager
    def substitute(startup, policy, **kwargs):
        calls["start"].append((startup, policy, kwargs))
        source = replace(startup.source, target=_equal_target(startup.source.target))
        object.__setattr__(startup, "source", source)
        try:
            yield _Session()
        finally:
            calls["closed"] += 1

    monkeypatch.setattr(bootstrap, "verified_vllm_chat", substitute)
    with pytest.raises(
        ModalInferenceBootstrapError, match="^modal_inference_bootstrap_invalid$"
    ):
        with _open(case):
            pytest.fail("substituted startup yielded")
    assert len(calls["start"]) == 1
    assert calls["closed"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (("root", Path("/different-model-root")), ("model_kind", "lora")),
)
def test_in_place_retrieved_metadata_mutation_after_baseline_is_denied(
    tmp_path, monkeypatch, field, value
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    now = [100.0]
    calls = _install_runtime(monkeypatch, now)
    original_worker = bootstrap.prepare_modal_chat_worker
    prepared = []

    def capture(*args, **kwargs):
        result = original_worker(*args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(bootstrap, "prepare_modal_chat_worker", capture)
    original_verify = bootstrap.verify_modal_inference_runtime

    def mutate(configuration):
        original_verify(configuration)
        if len(calls["verify"]) == 2:
            object.__setattr__(
                prepared[0].startup.source.target.retrieved, field, value
            )

    monkeypatch.setattr(bootstrap, "verify_modal_inference_runtime", mutate)
    with pytest.raises(ModalInferenceBootstrapError):
        with _open(case):
            pytest.fail("mutated model metadata yielded")
    assert calls["start"] == []


@pytest.mark.parametrize(
    "current",
    [True, object(), float("nan"), float("inf"), 10**10000],
    ids=("bool", "object", "nan", "inf", "huge-int"),
)
def test_tick_rejects_invalid_or_nonfinite_monotonic_values(monkeypatch, current):
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: current)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        bootstrap._tick()


def test_tick_rejects_regression_and_elapsed_deadline(monkeypatch):
    for current, kwargs in (
        (9.0, {"previous": 10.0}),
        (10.0, {"deadline": 10.0}),
    ):
        monkeypatch.setattr(bootstrap.time, "monotonic", lambda: current)
        with pytest.raises(ValueError, match="invalid or expired"):
            bootstrap._tick(**kwargs)


def test_invalid_initial_monotonic_value_denies_before_admission_or_preparation(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: math.nan)
    monkeypatch.setattr(
        bootstrap,
        "admit_modal_chat_launch",
        lambda *args, **kwargs: calls.append("admission"),
    )
    monkeypatch.setattr(
        bootstrap,
        "prepare_modal_chat_worker",
        lambda *args, **kwargs: calls.append("preparation"),
    )
    with pytest.raises(ModalInferenceBootstrapError):
        with _open(case):
            pytest.fail("invalid clock yielded")
    assert calls == []


def test_real_absent_packaged_runtime_lock_denies_before_model_preparation(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        bootstrap,
        "prepare_modal_chat_worker",
        lambda *args, **kwargs: calls.append("preparation"),
    )
    with pytest.raises(
        ModalInferenceBootstrapError, match="^modal_inference_bootstrap_invalid$"
    ):
        with _open(case):
            pytest.fail("missing packaged lock yielded")
    assert calls == []


@pytest.mark.parametrize("name", ("runtime_verifier", "lock_path", "monotonic"))
def test_bootstrap_signature_rejects_injected_authority_or_timer_kwargs(name):
    signature = inspect.signature(open_modal_chat_worker)
    with pytest.raises(TypeError):
        signature.bind(
            b"argument",
            expectation=object(),
            verifier=object(),
            clock=object(),
            destination=Path("/destination"),
            cwd=Path("/cwd"),
            environment={},
            **{name: object()},
        )
