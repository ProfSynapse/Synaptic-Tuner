"""CPU-only lifecycle framing; no guardian capability or hardware admission."""

from dataclasses import replace
import hashlib
import hmac
import json
import os

import pytest

from tuner.runtime.native_lifecycle_protocol import (
    NativeLifecyclePolicyV1, NativeLifecycleWriter, encode_worker_progress,
    parse_worker_progress, validate_event, verify_lifecycle_journal,
)
from tuner.training.modal_host_authority import HMACAuthenticator
from tuner.training.native_mlx_authority import NativeEvidenceAuthority


KEY = b"f" * 32  # Fixed test fixture, never a host key.
INVOCATION = "a" * 64
POLICY = NativeLifecyclePolicyV1(20, 16384)
PROGRESS = {"kind": "progress", "phase": "training", "step": 1,
            "metrics": {"loss": 2.5, "tokens": 17}}
OUTCOME = {"schema": "synaptic-native-process-outcome/v1", "reason": "worker_exited",
           "exit_code": 0, "termination_confirmed": True, "stdout_bytes": 13, "stderr_bytes": 0}
TERMINAL = {"kind": "terminal", "outcome": OUTCOME}


def test_worker_wire_accepts_only_canonical_progress():
    raw = encode_worker_progress(PROGRESS)
    assert raw.endswith(b"\n") and parse_worker_progress(raw[:-1]) == PROGRESS
    with pytest.raises(ValueError):
        encode_worker_progress(TERMINAL)
    with pytest.raises(ValueError):
        parse_worker_progress(json.dumps(TERMINAL, sort_keys=True, separators=(",", ":")).encode())
    for invalid in (raw, b"", b"x" * 4096, b"{\"kind\":\"progress\"}"):
        with pytest.raises(ValueError):
            parse_worker_progress(invalid)


@pytest.fixture
def journal(tmp_path):
    path = tmp_path / "journal.jsonl"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        yield path, fd
    finally:
        os.close(fd)


def writer(journal, *, policy=POLICY):
    return NativeLifecycleWriter(journal[1], invocation_digest=INVOCATION, policy=policy, key=KEY)


def read(raw, **kwargs):
    return verify_lifecycle_journal(raw, invocation_digest=INVOCATION,
                                    policy=POLICY, key=KEY, **kwargs)


def test_complete_numeric_prefix_and_terminal_roundtrip(journal):
    output = writer(journal)
    output.append(PROGRESS)
    first = journal[0].read_bytes()
    initial = read(first)
    assert initial.terminal_bytes is None and initial.committed_bytes == len(first)
    assert json.loads(initial.event_bytes[0]) == PROGRESS
    output.append(TERMINAL)
    raw = journal[0].read_bytes()
    final = read(raw, previous_raw=first)
    assert final.committed_bytes == len(raw) and final.partial_tail is False
    assert final.prefix_digest == hashlib.sha256(raw).hexdigest()
    assert json.loads(final.terminal_bytes) == OUTCOME
    assert KEY not in raw
    with pytest.raises(ValueError, match="closed"):
        output.append(PROGRESS)


def test_partial_tail_does_not_commit_or_establish_termination(journal):
    output = writer(journal)
    output.append(PROGRESS)
    prefix = journal[0].read_bytes()
    output.append(TERMINAL)
    raw = journal[0].read_bytes()
    for cut in (len(prefix) + 1, len(raw) - 1):
        observed = read(raw[:cut], previous_raw=prefix + b"partial old tail")
        assert observed.partial_tail is True and observed.terminal_bytes is None
        assert observed.committed_bytes == len(prefix)
        assert observed.event_bytes == read(prefix).event_bytes


@pytest.mark.parametrize("change", [
    lambda raw: raw.replace(b'"loss":2.5', b'"loss":2.6'),
    lambda raw: raw + raw,
    lambda raw: raw.replace(b'"sequence":0', b'"sequence":1'),
    lambda raw: raw.replace(b'"tag":"', b'"tag":"x'),
    lambda raw: raw.replace(b'"tag":"', '"tag":"é'.encode()),
    lambda raw: raw.replace(b'"step":1', b'"step":true'),
    lambda raw: b"\n" + raw,
])
def test_tampering_reordering_and_malformed_frames_reject(journal, change):
    writer(journal).append(PROGRESS)
    with pytest.raises(ValueError):
        read(change(journal[0].read_bytes()))


def test_invocation_policy_and_key_are_independently_bound(journal):
    writer(journal).append(PROGRESS)
    raw = journal[0].read_bytes()
    for overrides in ({"invocation_digest": "b" * 64}, {"key": b"z" * 32},
                      {"policy": replace(POLICY, maximum_progress_records=21)}):
        selection = dict(invocation_digest=INVOCATION, key=KEY, policy=POLICY)
        selection.update(overrides)
        with pytest.raises(ValueError):
            verify_lifecycle_journal(raw, **selection)


@pytest.mark.parametrize("event", [
    {**PROGRESS, "private_text": "must reject even with an authentic MAC"},
    {**PROGRESS, "metrics": {"exception": 123}},
    {"kind": "terminal", "outcome": {**OUTCOME, "termination_confirmed": False}},
])
def test_reader_independently_rejects_authentic_but_invalid_event(event):
    # Construct the wire envelope independently of the production writer.
    content = {"schema": "synaptic-native-lifecycle-frame/v1", "invocation_digest": INVOCATION,
               "policy_digest": POLICY.digest, "sequence": 0, "previous_digest": "0" * 64,
               "event": event}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    tag = hmac.new(KEY, b"synaptic-native-lifecycle-frame/v1\0" + encoded, hashlib.sha256).hexdigest()
    raw = json.dumps({"content": content, "tag": tag}, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError):
        read(raw)


@pytest.mark.parametrize("event", [
    {**PROGRESS, "text": "private output"}, {**PROGRESS, "phase": "private path"},
    {**PROGRESS, "step": True}, {**PROGRESS, "step": -1},
    {**PROGRESS, "metrics": {"exception": 1}},
    *({**PROGRESS, "metrics": {"loss": value}} for value in (True, "2", float("nan"), float("inf"), 10**1000)),
    {**PROGRESS, "metrics": {"tokens": 1.5}}, {**PROGRESS, "metrics": {"tokens": -1}},
    {**PROGRESS, "metrics": {"learning_rate": -1}},
    {"kind": "other"},
    {"kind": "terminal", "outcome": {**OUTCOME, "termination_confirmed": False}},
    {"kind": "terminal", "outcome": {**OUTCOME, "exit_code": True}},
    {"kind": "terminal", "outcome": {**OUTCOME, "reason": "exception text"}},
    {"kind": "terminal", "outcome": {**OUTCOME, "stderr_bytes": (16 << 20) + 65537}},
])
def test_only_closed_numeric_events_are_admitted_before_writing(journal, event):
    output = writer(journal)
    with pytest.raises(ValueError):
        validate_event(event)
    with pytest.raises(ValueError):
        output.append(event)
    assert journal[0].read_bytes() == b""


def test_record_budget_reserves_terminal_space(journal):
    policy = NativeLifecyclePolicyV1(1, 8192)
    output = writer(journal, policy=policy)
    output.append(PROGRESS)
    with pytest.raises(ValueError, match="bound"):
        output.append(PROGRESS)
    output.append(TERMINAL)
    observed = verify_lifecycle_journal(journal[0].read_bytes(), invocation_digest=INVOCATION,
                                       policy=policy, key=KEY)
    assert json.loads(observed.terminal_bytes) == OUTCOME


def test_aggregate_budget_reserves_terminal_space(journal):
    policy = NativeLifecyclePolicyV1(65536, 8192)
    output = writer(journal, policy=policy)
    for count in range(100):
        try:
            output.append(PROGRESS)
        except ValueError:
            break
    assert 1 < count < 100
    output.append(TERMINAL)
    raw = journal[0].read_bytes()
    assert len(raw) <= policy.maximum_journal_bytes
    assert verify_lifecycle_journal(raw, invocation_digest=INVOCATION,
                                    policy=policy, key=KEY).terminal_bytes is not None


def test_partial_write_poisons_writer_and_preserves_valid_prefix(journal, monkeypatch):
    output = writer(journal)
    output.append(PROGRESS)
    prefix = journal[0].read_bytes()
    original_write = os.write
    calls = []

    def interrupted(fd, raw):
        calls.append(True)
        if len(calls) == 1:
            return original_write(fd, raw[:7])
        raise OSError("fixture interrupted write")

    monkeypatch.setattr(os, "write", interrupted)
    with pytest.raises(OSError):
        output.append(TERMINAL)
    with pytest.raises(ValueError, match="closed"):
        output.append(TERMINAL)
    observed = read(journal[0].read_bytes(), previous_raw=prefix)
    assert observed.committed_bytes == len(prefix) and observed.terminal_bytes is None


def test_retained_prefix_cannot_roll_back_even_to_an_authentic_shorter_snapshot(journal):
    output = writer(journal)
    output.append(PROGRESS)
    first = journal[0].read_bytes()
    output.append({**PROGRESS, "step": 2})
    with pytest.raises(ValueError, match="prefix changed"):
        read(first, previous_raw=journal[0].read_bytes())


def test_no_bytes_can_follow_terminal_even_as_a_partial_tail(journal):
    writer(journal).append(TERMINAL)
    raw = journal[0].read_bytes()
    for suffix in (b"x", b"\n", raw):
        with pytest.raises(ValueError):
            read(raw + suffix)


@pytest.mark.parametrize("kwargs", [
    {"maximum_progress_records": True}, {"maximum_progress_records": 0},
    {"maximum_progress_records": 65537}, {"maximum_journal_bytes": 8191},
    {"maximum_journal_bytes": (16 << 20) + 1},
])
def test_policy_bounds_are_finite(kwargs):
    with pytest.raises(ValueError):
        replace(POLICY, **kwargs)


def test_descriptor_is_exclusive_private_and_not_resumed(journal):
    writer(journal).append(PROGRESS)
    with pytest.raises(ValueError, match="fresh"):
        writer(journal)
    os.chmod(journal[0], 0o644)
    with pytest.raises(ValueError, match="private"):
        writer(journal)


@pytest.mark.parametrize("change", [
    lambda path, fd: os.chmod(path, 0o644),
    lambda path, fd: os.lseek(fd, 1, os.SEEK_SET),
    lambda path, fd: os.ftruncate(fd, 1),
    lambda path, fd: os.unlink(path),
    lambda path, fd: os.link(path, path.with_name("alias")),
])
def test_changed_descriptor_custody_poisoned_before_next_append(journal, change):
    output = writer(journal)
    output.append(PROGRESS)
    change(*journal)
    with pytest.raises(ValueError, match="descriptor changed"):
        output.append(TERMINAL)
    with pytest.raises(ValueError, match="closed"):
        output.append(TERMINAL)


def test_key_derivation_requires_explicit_purpose_and_exact_scope():
    purpose = "native-lifecycle-capability/v1"
    denied = NativeEvidenceAuthority("reader", "reader-key", HMACAuthenticator(
        {"reader-key": b"test-root-only-not-a-host-key!!!!"},
        allowed_purposes=frozenset({"provider-run-observation/v1"})))
    with pytest.raises(ValueError):
        denied.lifecycle_key(INVOCATION, POLICY.digest)
    authority = NativeEvidenceAuthority("reader", "reader-key", HMACAuthenticator(
        {"reader-key": b"test-root-only-not-a-host-key!!!!"}, allowed_purposes=frozenset({purpose})))
    selected = authority.lifecycle_key(INVOCATION, POLICY.digest)
    assert type(selected) is bytes and len(selected) == 32
    assert selected == authority.lifecycle_key(INVOCATION, POLICY.digest)
    assert selected != authority.lifecycle_key("b" * 64, POLICY.digest)
    assert selected != authority.lifecycle_key(INVOCATION, replace(POLICY, maximum_progress_records=21).digest)
    with pytest.raises((TypeError, ValueError)):
        authority.lifecycle_key("private invalid digest", POLICY.digest)
