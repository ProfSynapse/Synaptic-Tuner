"""Provider-free acceptance tests for the bounded Modal chat bootstrap."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest

from tests.execution.providers.modal_inference_worker_fixtures import (
    mounted_launch_case,
)
from tuner.execution.foundation_v2.canonical import parse_canonical_object
from tuner.execution.providers.modal import inference_bootstrap as bootstrap
from tuner.execution.providers.modal.inference_bootstrap import (
    ModalInferenceBootstrapError,
    open_modal_chat_worker,
)
from Evaluator.vllm_runtime import VLLMRuntimeLease


class _Session:
    pass


def _install_runtime(monkeypatch, now):
    calls = {"verify": [], "start": [], "closed": 0}

    def verify(configuration):
        calls["verify"].append(configuration.canonical_bytes)

    @contextmanager
    def verified(startup, policy, **kwargs):
        calls["start"].append((startup, policy, kwargs))
        try:
            yield _Session()
        finally:
            calls["closed"] += 1

    monkeypatch.setattr(bootstrap, "verify_modal_inference_runtime", verify)
    monkeypatch.setattr(bootstrap, "verified_vllm_chat", verified)
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: now[0])
    return calls


def _open(case, **changes):
    kwargs = dict(
        expectation=case.kwargs["expectation"],
        verifier=case.kwargs["verifier"],
        clock=case.kwargs["clock"],
        destination=case.kwargs["destination"],
        cwd=case.roots["destination"],
        environment={},
        preparer=case.kwargs["preparer"],
    )
    kwargs.update(changes)
    return open_modal_chat_worker(case.envelope.argument_bytes, **kwargs)


@pytest.mark.parametrize("model_kind", ("full", "lora"))
def test_real_signed_mount_preparation_reaches_exact_bounded_runtime(
    tmp_path, monkeypatch, model_kind
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind=model_kind)
    now = [100.0]
    calls = _install_runtime(monkeypatch, now)
    with _open(case) as session:
        assert type(session) is _Session
    assert calls["verify"] == [
        case.kwargs["expectation"].configuration_bytes,
        case.kwargs["expectation"].configuration_bytes,
    ]
    assert len(calls["start"]) == 1
    startup, policy, runtime = calls["start"][0]
    assert startup.source.target.retrieved.model_kind == model_kind
    assert startup.served_model_name == "fixture-chat"
    assert startup.tensor_parallel_size == 1
    assert startup.startup_timeout_s == 600
    assert startup.readiness_request_timeout_s == 0.75
    assert runtime == {
        "cwd": case.roots["destination"],
        "environment": {},
        "max_tokens": 73,
        "temperature": 0.25,
        "top_p": 0.875,
        "max_request_bytes": 8192,
        "max_response_bytes": 65536,
        "deadline": 219.0,
    }
    assert policy.absolute_lifetime_seconds == 900
    assert calls["closed"] == 1
    assert case.preparer.calls == (
        []
        if model_kind == "full"
        else [(case.workload["model_ref"], case.workload["model_revision"])]
    )


@pytest.mark.parametrize("slow_phase", ("runtime", "preparation"))
def test_time_consumed_before_or_during_preparation_never_renews_deadline(
    tmp_path, monkeypatch, slow_phase
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="full")
    now = [100.0]
    calls = _install_runtime(monkeypatch, now)
    if slow_phase == "runtime":
        original = bootstrap.verify_modal_inference_runtime

        def slow(configuration):
            original(configuration)
            now[0] += 30

        monkeypatch.setattr(bootstrap, "verify_modal_inference_runtime", slow)
    else:
        original = bootstrap.prepare_modal_chat_worker

        def slow(*args, **kwargs):
            result = original(*args, **kwargs)
            now[0] += 30
            return result

        monkeypatch.setattr(bootstrap, "prepare_modal_chat_worker", slow)
    with _open(case):
        pass
    assert calls["start"][0][2]["deadline"] == 219.0


@pytest.mark.parametrize("slow_phase", ("runtime", "preparation"))
def test_budget_exhausted_during_preparation_denies_before_runtime_start(
    tmp_path, monkeypatch, slow_phase
):
    case = mounted_launch_case(tmp_path, monkeypatch, model_kind="full")
    now = [100.0]
    calls = _install_runtime(monkeypatch, now)
    if slow_phase == "runtime":
        original = bootstrap.verify_modal_inference_runtime

        def slow(configuration):
            original(configuration)
            now[0] = 219.0

        monkeypatch.setattr(bootstrap, "verify_modal_inference_runtime", slow)
    else:
        original = bootstrap.prepare_modal_chat_worker

        def slow(*args, **kwargs):
            result = original(*args, **kwargs)
            now[0] = 219.0
            return result

        monkeypatch.setattr(bootstrap, "prepare_modal_chat_worker", slow)
    with pytest.raises(
        ModalInferenceBootstrapError, match="^modal_inference_bootstrap_invalid$"
    ):
        with _open(case):
            pytest.fail("expired preparation yielded")
    assert calls["start"] == []


def test_invalid_launch_denies_before_runtime_verification_or_model_preparation(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        bootstrap,
        "verify_modal_inference_runtime",
        lambda value: calls.append("runtime"),
    )
    monkeypatch.setattr(
        bootstrap,
        "prepare_modal_chat_worker",
        lambda *args, **kwargs: calls.append("worker"),
    )
    argument = case.envelope.argument_bytes[:-1] + b"x"
    with pytest.raises(
        ModalInferenceBootstrapError, match="^modal_inference_bootstrap_invalid$"
    ):
        with open_modal_chat_worker(
            argument,
            expectation=case.kwargs["expectation"],
            verifier=case.kwargs["verifier"],
            clock=case.kwargs["clock"],
            destination=case.kwargs["destination"],
            cwd=case.roots["destination"],
            environment={},
            preparer=case.kwargs["preparer"],
        ):
            pytest.fail("invalid launch yielded")
    assert calls == []


def test_missing_packaged_runtime_lock_denies_before_model_preparation(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    worker_calls = []

    def missing(_):
        raise RuntimeError("private-lock-detail")

    monkeypatch.setattr(bootstrap, "verify_modal_inference_runtime", missing)
    monkeypatch.setattr(
        bootstrap,
        "prepare_modal_chat_worker",
        lambda *args, **kwargs: worker_calls.append(1),
    )
    with pytest.raises(
        ModalInferenceBootstrapError, match="^modal_inference_bootstrap_invalid$"
    ) as caught:
        with _open(case):
            pytest.fail("missing lock yielded")
    assert caught.value.__cause__ is None
    assert worker_calls == []


def test_pre_runtime_failure_never_reads_or_retains_hostile_cleanup_property(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    reads = []

    class Hostile(RuntimeError):
        @property
        def cleanup_lease(self):
            reads.append("cleanup_lease")
            raise AssertionError("property executed")

    monkeypatch.setattr(
        bootstrap,
        "verify_modal_inference_runtime",
        lambda value: (_ for _ in ()).throw(Hostile("private")),
    )
    with pytest.raises(ModalInferenceBootstrapError) as caught:
        with _open(case):
            pytest.fail("hostile verifier yielded")
    assert reads == []
    assert not hasattr(caught.value, "cleanup_lease")


@pytest.mark.parametrize("field", ("startup", "policy", "max_tokens", "admission"))
def test_tampered_worker_projection_denies_before_runtime_start(
    tmp_path, monkeypatch, field
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    now = [100.0]
    calls = _install_runtime(monkeypatch, now)
    original = bootstrap.prepare_modal_chat_worker

    def tampered(*args, **kwargs):
        value = original(*args, **kwargs)
        if field == "startup":
            return replace(
                value, startup=replace(value.startup, served_model_name="other")
            )
        if field == "policy":
            return replace(
                value,
                configured_policy=replace(
                    value.configured_policy, request_timeout_seconds=61
                ),
            )
        if field == "max_tokens":
            return replace(value, max_tokens=74)
        return replace(value, admission=object())

    monkeypatch.setattr(bootstrap, "prepare_modal_chat_worker", tampered)
    with pytest.raises(
        ModalInferenceBootstrapError, match="^modal_inference_bootstrap_invalid$"
    ):
        with _open(case):
            pytest.fail("tampered preparation yielded")
    assert calls["start"] == []


def test_expiry_after_runtime_start_closes_exactly_once_and_never_yields(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    now = [100.0]
    calls = _install_runtime(monkeypatch, now)

    @contextmanager
    def late(startup, policy, **kwargs):
        calls["start"].append((startup, policy, kwargs))
        try:
            now[0] = kwargs["deadline"]
            yield _Session()
        finally:
            calls["closed"] += 1

    monkeypatch.setattr(bootstrap, "verified_vllm_chat", late)
    with pytest.raises(
        ModalInferenceBootstrapError, match="^modal_inference_bootstrap_invalid$"
    ):
        with _open(case):
            pytest.fail("expired startup yielded")
    assert len(calls["start"]) == 1
    assert calls["closed"] == 1


@pytest.mark.parametrize("phase", ("runtime", "worker", "startup"))
@pytest.mark.parametrize("failure", (KeyboardInterrupt(), SystemExit(7)))
def test_control_failures_preserve_identity_and_close_acquired_runtime(
    tmp_path, monkeypatch, phase, failure
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    now = [100.0]
    calls = _install_runtime(monkeypatch, now)
    if phase == "runtime":
        monkeypatch.setattr(
            bootstrap,
            "verify_modal_inference_runtime",
            lambda value: (_ for _ in ()).throw(failure),
        )
    elif phase == "worker":
        monkeypatch.setattr(
            bootstrap,
            "prepare_modal_chat_worker",
            lambda *args, **kwargs: (_ for _ in ()).throw(failure),
        )
    else:

        @contextmanager
        def fail(*args, **kwargs):
            raise failure
            yield

        monkeypatch.setattr(bootstrap, "verified_vllm_chat", fail)
    with pytest.raises(type(failure)) as caught:
        with _open(case):
            pytest.fail("control failure yielded")
    assert caught.value is failure
    assert calls["closed"] == 0


def test_consumer_exception_identity_is_preserved_and_runtime_closes(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)
    calls = _install_runtime(monkeypatch, [100.0])
    failure = RuntimeError("consumer-detail")
    with pytest.raises(RuntimeError) as caught:
        with _open(case):
            raise failure
    assert caught.value is failure
    assert calls["closed"] == 1


def test_startup_failure_preserves_exact_unresolved_cleanup_lease(
    tmp_path, monkeypatch
):
    case = mounted_launch_case(tmp_path, monkeypatch)

    class Process:
        cleanup_pending = True

        def close(self, **kwargs):
            return False

    lease = VLLMRuntimeLease(
        Process(), host="127.0.0.1", port=8000, served_model_name="fixture-chat"
    )

    @contextmanager
    def unresolved(*args, **kwargs):
        error = RuntimeError("private-startup-detail")
        error.cleanup_lease = lease
        raise error
        yield

    monkeypatch.setattr(bootstrap, "verify_modal_inference_runtime", lambda value: None)
    monkeypatch.setattr(bootstrap, "verified_vllm_chat", unresolved)
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: 100.0)
    with pytest.raises(
        ModalInferenceBootstrapError, match="^modal_inference_bootstrap_invalid$"
    ) as caught:
        with _open(case):
            pytest.fail("failed startup yielded")
    assert caught.value.cleanup_lease is lease
    assert caught.value.__cause__ is None
