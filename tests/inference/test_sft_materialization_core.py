"""Acceptance tests for the trusted post-admission SFT materialization core."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import replace
import pytest
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import (
    RunArtifactRequest,
    RunVerification,
    RunsAPI,
)
from tuner.inference.retrieved_model import (
    _materialize_admitted_sft_model,
    _open_root,
    materialize_verified_sft_model,
)
from tests.inference.test_retrieved_model import _fixture


def _reader(operations, calls=None):
    def read(run, artifact):
        if calls is not None:
            calls.append((run, artifact))
        return operations.artifacts(
            RunArtifactRequest(run, artifact.role, artifact.size_bytes)
        )

    return read


@contextmanager
def _root(path):
    import os

    fd = _open_root(path)
    try:
        yield fd
    finally:
        os.close(fd)


def _core(run, operations, root, read=None, **kwargs):
    with _root(root) as root_fd:
        return _materialize_admitted_sft_model(
            run=run,
            artifacts=operations.inventory,
            root=root,
            root_fd=root_fd,
            read_artifact=read or _reader(operations),
            **kwargs,
        )


def _projection(value):
    return (
        value.run,
        value.artifacts,
        value.model_ref,
        value.model_revision,
        value.tokenizer_revision,
        value.model_kind,
        tuple((x.name, x.size_bytes, x.sha256) for x in value.model_files),
        tuple((x.name, x.size_bytes, x.sha256) for x in value.tokenizer_files),
    )


def test_core_matches_public_runs_admission_for_same_inputs(tmp_path):
    public_root, core_root = tmp_path / "public", tmp_path / "core"
    public_root.mkdir()
    core_root.mkdir()
    run, _, public_ops = _fixture(public_root)
    public = materialize_verified_sft_model(RunsAPI(public_ops), run, public_root)
    _, _, core_ops = _fixture(core_root)
    calls = []
    core = _core(run, core_ops, core_root, _reader(core_ops, calls))
    assert _projection(core) == _projection(public)
    assert [x.role for _, x in calls] == [x.role for x in core_ops.inventory]


def test_public_wrapper_denies_before_artifact_stream(tmp_path):
    run, _, operations = _fixture(tmp_path)
    calls = []
    operations.reverify = lambda value: RunVerification(
        value, False, "2026-09-10T00:00:00Z"
    )
    operations.artifacts = lambda request: calls.append(request)
    with pytest.raises(ValueError, match="reverification"):
        materialize_verified_sft_model(RunsAPI(operations), run, tmp_path)
    assert calls == [] and not list(tmp_path.glob(".retrieved-*"))


def test_public_success_orders_admission_before_first_stream(tmp_path):
    run, _, operations = _fixture(tmp_path)
    calls = []
    old_reverify, old_outcome, old_artifacts = (
        operations.reverify,
        operations.outcome,
        operations.artifacts,
    )
    operations.reverify = lambda value: (calls.append("reverify"), old_reverify(value))[
        1
    ]
    operations.outcome = lambda value: (calls.append("outcome"), old_outcome(value))[1]
    operations.artifacts = lambda request: (
        calls.append("read"),
        old_artifacts(request),
    )[1]
    materialize_verified_sft_model(RunsAPI(operations), run, tmp_path)
    assert calls[:3] == ["reverify", "outcome", "read"]


def test_core_borrows_root_fd_and_rejects_mismatched_root_without_writes(tmp_path):
    import os

    root, other = tmp_path / "root", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    run, _, operations = _fixture(root)
    fd = _open_root(root)
    try:
        _materialize_admitted_sft_model(
            run=run,
            artifacts=operations.inventory,
            root=root,
            root_fd=fd,
            read_artifact=_reader(operations),
        )
        os.fstat(fd)
    finally:
        os.close(fd)
    wrong = _open_root(other)
    calls = []
    try:
        with pytest.raises((OSError, TypeError, ValueError)):
            _materialize_admitted_sft_model(
                run=run,
                artifacts=operations.inventory,
                root=root,
                root_fd=wrong,
                read_artifact=_reader(operations, calls),
            )
        os.fstat(wrong)
    finally:
        os.close(wrong)
    assert calls == [] and not list(other.glob(".retrieved-*"))


def test_core_keeps_borrowed_root_fd_open_after_reader_failure(tmp_path):
    import os

    run, _, operations = _fixture(tmp_path)
    fd = _open_root(tmp_path)

    def fail(run, artifact):
        raise RuntimeError("private")

    try:
        with pytest.raises(RuntimeError):
            _materialize_admitted_sft_model(
                run=run,
                artifacts=operations.inventory,
                root=tmp_path,
                root_fd=fd,
                read_artifact=fail,
            )
        os.fstat(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("when", ("callback", "iterator"))
def test_core_rejects_caller_artifact_mutation_during_read(tmp_path, when):
    run, _, operations = _fixture(tmp_path)
    target = operations.inventory[0]
    triggers = []
    original = target.sha256

    def read(selected, artifact):
        stream = _reader(operations)(selected, artifact)
        if artifact.role == target.role and when == "callback":
            triggers.append(when)
            object.__setattr__(target, "sha256", "0" * 64)
        if artifact.role == target.role and when == "iterator":
            old = stream.iter_bytes

            def chunks():
                triggers.append(when)
                object.__setattr__(target, "sha256", "0" * 64)
                yield from old()

            stream.iter_bytes = chunks
        return stream

    try:
        with pytest.raises((TypeError, ValueError)):
            _core(run, operations, tmp_path, read)
    finally:
        object.__setattr__(target, "sha256", original)
    assert triggers == [when] and not list(tmp_path.glob(".retrieved-*"))


def test_core_rejects_stream_header_mutation_during_iteration(tmp_path):
    run, _, operations = _fixture(tmp_path)

    def read(selected, artifact):
        stream = _reader(operations)(selected, artifact)
        old = stream.iter_bytes

        def chunks():
            stream.artifact = replace(
                stream.artifact, size_bytes=stream.artifact.size_bytes + 1
            )
            yield from old()

        stream.iter_bytes = chunks
        return stream

    with pytest.raises((TypeError, ValueError)):
        _core(run, operations, tmp_path, read)
    assert not list(tmp_path.glob(".retrieved-*"))


@pytest.mark.parametrize("target", ("run", "artifact"))
def test_core_rejects_request_clone_mutation_inside_callback(tmp_path, target):
    run, _, operations = _fixture(tmp_path)
    triggers = []

    def read(selected, artifact):
        triggers.append(target)
        if target == "run":
            object.__setattr__(selected, "run_id", "other")
        else:
            object.__setattr__(artifact, "role", "other")
        return _reader(operations)(run, operations.inventory[0])

    with pytest.raises((TypeError, ValueError)):
        _core(run, operations, tmp_path, read)
    assert triggers == [target]
    assert not list(tmp_path.glob(".retrieved-*"))


def test_core_rejects_caller_run_mutation_during_iteration(tmp_path):
    run, _, operations = _fixture(tmp_path)
    original = run.run_id
    triggers = []

    def read(selected, artifact):
        stream = _reader(operations)(selected, artifact)
        old = stream.iter_bytes

        def chunks():
            triggers.append("mutated")
            object.__setattr__(run, "run_id", "other")
            yield from old()

        stream.iter_bytes = chunks
        return stream

    try:
        with pytest.raises((TypeError, ValueError)):
            _core(run, operations, tmp_path, read)
    finally:
        object.__setattr__(run, "run_id", original)
    assert triggers == ["mutated"]
    assert not list(tmp_path.glob(".retrieved-*"))


@pytest.mark.parametrize("mutation", ("presented", "caller", "both"))
def test_public_reverify_run_mutation_denies_before_outcome_or_stream(
    tmp_path, mutation
):
    run, _, operations = _fixture(tmp_path)
    original = run.run_id
    calls = []

    def reverify(selected):
        calls.append("reverify")
        if mutation in ("presented", "both"):
            object.__setattr__(selected, "run_id", "presented-other")
        if mutation in ("caller", "both"):
            object.__setattr__(run, "run_id", "caller-other")
        return RunVerification(selected, True, "2026-09-10T00:00:00Z")

    operations.reverify = reverify
    operations.outcome = lambda selected: calls.append("outcome")
    operations.artifacts = lambda request: calls.append("stream")
    try:
        with pytest.raises((TypeError, ValueError)):
            materialize_verified_sft_model(RunsAPI(operations), run, tmp_path)
    finally:
        object.__setattr__(run, "run_id", original)
    assert calls == ["reverify"]
    assert not list(tmp_path.glob(".retrieved-*"))


@pytest.mark.parametrize("change", ("run", "role", "hash", "size"))
def test_core_rejects_stream_metadata_substitution(tmp_path, change):
    run, _, operations = _fixture(tmp_path)

    def read(selected, artifact):
        stream = _reader(operations)(selected, artifact)
        if change == "run":
            stream.run = TrainingRunRef("other", run.project_ref)
        elif change == "role":
            stream.artifact = replace(stream.artifact, role="other")
        elif change == "hash":
            stream.artifact = replace(stream.artifact, sha256="0" * 64)
        else:
            stream.artifact = replace(
                stream.artifact, size_bytes=artifact.size_bytes + 1
            )
        return stream

    with pytest.raises(ValueError, match="substituted"):
        _core(run, operations, tmp_path, read)
    assert not list(tmp_path.glob(".retrieved-*"))


@pytest.mark.parametrize(
    "failure", ("empty", "nonbytes", "oversize", "truncated", "overflow")
)
def test_core_rejects_malformed_stream_chunks(tmp_path, failure):
    run, values, operations = _fixture(tmp_path)

    def read(selected, artifact):
        stream = _reader(operations)(selected, artifact)
        if artifact.role != "final_model":
            return stream

        def chunks():
            raw = values[artifact.role]
            if failure == "empty":
                yield b""
            elif failure == "nonbytes":
                yield bytearray(raw)
            elif failure == "oversize":
                yield b"x" * 1_048_577
            elif failure == "truncated":
                yield raw[:-1]
            else:
                yield raw + b"x"

        stream.iter_bytes = chunks
        return stream

    with pytest.raises((TypeError, ValueError)):
        _core(run, operations, tmp_path, read)
    assert not list(tmp_path.glob(".retrieved-*"))


@pytest.mark.parametrize("failure", (RuntimeError("private"), KeyboardInterrupt()))
def test_core_cleans_reader_failure_and_preserves_control_flow(tmp_path, failure):
    run, _, operations = _fixture(tmp_path)

    def read(selected, artifact):
        if artifact.role == "tokenizer":
            raise failure
        return _reader(operations)(selected, artifact)

    with pytest.raises(type(failure)):
        _core(run, operations, tmp_path, read)
    assert not list(tmp_path.glob(".retrieved-*"))


@pytest.mark.parametrize("invalid", ("duplicate", "order", "zero", "total"))
def test_core_rejects_manifest_or_bounds_before_callback(tmp_path, invalid):
    run, _, operations = _fixture(tmp_path)
    artifacts = operations.inventory
    kwargs = {}
    if invalid == "duplicate":
        artifacts = artifacts[:-1] + (artifacts[0],)
    elif invalid == "order":
        artifacts = tuple(reversed(artifacts))
    elif invalid == "zero":
        artifacts = (replace(artifacts[0], size_bytes=0),) + artifacts[1:]
    else:
        kwargs["maximum_total_bytes"] = 1
    calls = []
    with _root(tmp_path) as root_fd:
        with pytest.raises((TypeError, ValueError)):
            _materialize_admitted_sft_model(
                run=run,
                artifacts=artifacts,
                root=tmp_path,
                root_fd=root_fd,
                read_artifact=_reader(operations, calls),
                **kwargs,
            )
    assert calls == [] and not list(tmp_path.glob(".retrieved-*"))
