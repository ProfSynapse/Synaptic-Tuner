"""Remote-local chat composition; not a Modal deployment or public endpoint.

The deployment must authenticate physical mounts and supply the trusted launch
expectation and evidence verifier. This composition owns only its local runtime.
A missing or mismatched packaged inference lock denies before model preparation.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields
import inspect
import math
from pathlib import Path
import time
from typing import Iterator

from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from Evaluator.verified_vllm_chat import verified_vllm_chat
from Evaluator.vllm_runtime import (
    VerifiedLocalVLLMSource,
    VLLMRuntimeLease,
    VLLMStartupSpec,
)
from tuner.execution.evidence import parse_utc
from tuner.execution.foundation_v2.canonical import parse_canonical_object
from tuner.inference.retrieved_model import RetrievedSFTModel
from tuner.inference.run_chat import PreparedModelIdentity
from tuner.inference.serving_target import PinnedModelPreparer, ServingTarget

from .inference_preparation import _validate_preparation_snapshot
from .inference_runtime import verify_modal_inference_runtime
from .inference_wire import (
    ModalChatWorkerExpectation,
    _clock_now,
    admit_modal_chat_launch,
)
from .inference_worker import (
    ModalChatWorkerPreparation,
    _serving_preparation,
    prepare_modal_chat_worker,
)


class ModalInferenceBootstrapError(RuntimeError):
    """Closed setup failure; unresolved cleanup retains its exact lease."""


@dataclass(frozen=True, slots=True)
class ModalChatWorkerSession:
    """Internal result projection, not a grant or standalone verification proof."""

    session: ChatSession
    model: PreparedModelIdentity


def _closed(
    error: Exception, *, runtime_entry: bool = False
) -> ModalInferenceBootstrapError:
    result = ModalInferenceBootstrapError("modal_inference_bootstrap_invalid")
    if runtime_entry:
        lease = inspect.getattr_static(error, "cleanup_lease", None)
        if type(lease) is VLLMRuntimeLease:
            result.cleanup_lease = lease
    return result


def _tick(*, previous: float | None = None, deadline: float | None = None) -> float:
    now = time.monotonic()
    if type(now) not in (int, float):
        raise ValueError("invalid monotonic clock")
    now = float(now)
    if (
        not math.isfinite(now)
        or (previous is not None and now < previous)
        or (deadline is not None and now >= deadline)
    ):
        raise ValueError("invalid or expired monotonic clock")
    return now


def _values(value) -> tuple:
    return tuple(getattr(value, field.name) for field in fields(value))


def _same_fields(left, right, *, omit: str | None = None) -> bool:
    return type(left) is type(right) and all(
        type(getattr(left, field.name)) is type(getattr(right, field.name))
        and getattr(left, field.name) == getattr(right, field.name)
        for field in fields(right)
        if field.name != omit
    )


def _target_metadata(target: ServingTarget) -> tuple:
    """Copy retained path/identity metadata without touching model bytes."""
    retrieved = target.retrieved
    base = target.base_snapshot
    return (
        str(retrieved.root),
        retrieved.root_identity,
        retrieved.attempt,
        retrieved.identity,
        retrieved.model_kind,
        tuple(_values(item) for item in retrieved.model_files),
        tuple(_values(item) for item in retrieved.tokenizer_files),
        (
            None
            if base is None
            else (
                base.model_ref,
                base.revision,
                str(base.root),
                base.snapshot,
                base.root_identity,
                base.snapshot_identity,
                tuple(_values(item) for item in base.files),
            )
        ),
    )


def _rederive(prepared, admission) -> ModalChatWorkerPreparation:
    if (
        type(prepared) is not ModalChatWorkerPreparation
        or type(prepared.startup) is not VLLMStartupSpec
        or type(prepared.startup.source) is not VerifiedLocalVLLMSource
        or type(prepared.configured_policy) is not ChatSessionPolicy
        or prepared.admission.argument_bytes != admission.argument_bytes
    ):
        raise ValueError("worker preparation differs from fresh admission")
    target = prepared.startup.source.target
    if (
        type(target) is not ServingTarget
        or type(target.retrieved) is not RetrievedSFTModel
    ):
        raise TypeError("exact serving target required")
    # This is an authority/configuration comparison, not another model scan.
    # Preparation verifies bytes; the existing runtime independently rechecks
    # the target immediately before spawn. Do not rehash multi-GB weights at
    # every clock/evidence boundary here.
    snapshot = _validate_preparation_snapshot(admission.preparation_snapshot)
    source = snapshot["chat_input"]["source"]
    workload = snapshot["chat_input"]["workload"]
    retrieved = target.retrieved
    if (
        retrieved.run.to_dict() != source["run"]
        or tuple(item.to_dict() for item in retrieved.artifacts)
        != tuple(source["artifacts"])
        or (retrieved.model_ref, retrieved.model_revision, retrieved.tokenizer_revision)
        != (
            workload["model_ref"],
            workload["model_revision"],
            workload["tokenizer_revision"],
        )
    ):
        raise ValueError("prepared model differs from authenticated source")
    expected = _serving_preparation(admission, target)
    if not _same_fields(prepared.startup, expected.startup, omit="source") or not (
        _same_fields(prepared.configured_policy, expected.configured_policy)
    ):
        raise ValueError("serving settings differ from authenticated configuration")
    for name in (
        "max_tokens",
        "temperature",
        "top_p",
        "max_request_bytes",
        "max_response_bytes",
    ):
        if type(getattr(prepared, name)) is not type(getattr(expected, name)) or (
            getattr(prepared, name) != getattr(expected, name)
        ):
            raise ValueError(
                "generation settings differ from authenticated configuration"
            )
    return expected


@contextmanager
def open_modal_chat_worker(
    argument: bytes,
    *,
    expectation: ModalChatWorkerExpectation,
    verifier,
    clock,
    destination: Path,
    cwd: Path,
    environment: dict[str, str],
    preparer: PinnedModelPreparer | None = None,
) -> Iterator[ModalChatWorkerSession]:
    """Admit, verify the runtime, prepare and own one bounded local session.

    No caller-supplied runtime verifier/lock or alternate timer is accepted.
    Runtime verification uses the fixed packaged inference inventory, not the
    training lock. SDK creation/access and exact Sandbox teardown remain the
    responsibility of the future deployment/transport composition.
    """
    try:
        if (
            type(argument) is not bytes
            or type(expectation) is not ModalChatWorkerExpectation
        ):
            raise TypeError("exact launch bytes and expectation required")
        original = _values(expectation)
        owned = ModalChatWorkerExpectation(
            **{
                field.name: getattr(expectation, field.name)
                for field in fields(expectation)
            }
        )
        if type(environment) is not dict or any(
            type(key) is not str or type(value) is not str
            for key, value in environment.items()
        ):
            raise TypeError("explicit exact-string environment required")
        owned_environment = dict(environment)
        first = _tick()
        admission = admit_modal_chat_launch(
            argument, expectation=owned, verifier=verifier, clock=clock
        )
        claim = parse_canonical_object(admission.claim, name="chat launch claim")
        # Anchor before the UTC read (and admission), not afterward. Slow
        # callbacks consume the budget. Never derive another deadline later.
        now_utc = _clock_now(clock)
        remaining = (
            parse_utc(claim["expires_at"]) - parse_utc(now_utc)
        ).total_seconds()
        remaining = min(
            remaining,
            admission.configuration.document["policy"]["absolute_lifetime_seconds"],
        )
        deadline = first + remaining
        if not math.isfinite(deadline) or remaining <= 0:
            raise ValueError("launch deadline expired")
        previous = _tick(previous=first, deadline=deadline)

        def unchanged() -> None:
            if _values(expectation) != original or _values(owned) != original:
                raise ValueError("launch expectation changed")

        unchanged()
        verify_modal_inference_runtime(admission.configuration)
        unchanged()
        previous = _tick(previous=previous, deadline=deadline)
        prepared = prepare_modal_chat_worker(
            argument,
            expectation=owned,
            verifier=verifier,
            clock=clock,
            destination=destination,
            preparer=preparer,
        )
        unchanged()
        _rederive(prepared, admission)
        baseline_target = prepared.startup.source.target
        baseline_retrieved = baseline_target.retrieved
        baseline_base = baseline_target.base_snapshot
        baseline_metadata = _target_metadata(baseline_target)

        def same_target(value) -> None:
            target = value.startup.source.target
            if (
                target is not baseline_target
                or target.retrieved is not baseline_retrieved
                or target.base_snapshot is not baseline_base
                or _target_metadata(target) != baseline_metadata
            ):
                raise ValueError("prepared target identity changed")

        def fresh() -> ModalChatWorkerPreparation:
            checked = admit_modal_chat_launch(
                argument, expectation=owned, verifier=verifier, clock=clock
            )
            unchanged()
            same_target(prepared)
            return _rederive(prepared, checked)

        projected = fresh()
        verify_modal_inference_runtime(projected.admission.configuration)
        unchanged()
        # Recheck after the runtime verifier, before any serving process exists.
        projected = fresh()
        previous = _tick(previous=previous, deadline=deadline)
        unchanged()
        same_target(prepared)
        projected = _rederive(prepared, projected.admission)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise _closed(error) from None

    with ExitStack() as ownership:
        try:
            session = ownership.enter_context(
                verified_vllm_chat(
                    projected.startup,
                    projected.configured_policy,
                    cwd=cwd,
                    environment=owned_environment,
                    max_tokens=projected.max_tokens,
                    temperature=projected.temperature,
                    top_p=projected.top_p,
                    max_request_bytes=projected.max_request_bytes,
                    max_response_bytes=projected.max_response_bytes,
                    deadline=deadline,
                )
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise _closed(error, runtime_entry=True) from None
        try:
            checked = fresh()
            _tick(previous=previous, deadline=deadline)
            unchanged()
            same_target(prepared)
            same_target(projected)
            _rederive(prepared, checked.admission)
            _rederive(projected, checked.admission)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise _closed(error) from None
        # Consumer exceptions retain their own identity and existing owned
        # cleanup semantics; only bootstrap/setup failures are sanitized here.
        retrieved = baseline_retrieved
        yield ModalChatWorkerSession(
            session,
            PreparedModelIdentity(
                retrieved.model_ref,
                retrieved.model_revision,
                retrieved.tokenizer_revision,
                retrieved.model_kind,
            ),
        )


__all__: list[str] = []
