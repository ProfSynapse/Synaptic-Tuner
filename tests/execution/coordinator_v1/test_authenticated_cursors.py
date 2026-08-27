from __future__ import annotations

import hmac
import hashlib

import pytest

from synaptic_tuner.api.v1.runs_facade import (
    RunListRequest,
    RunLogsRequest,
    RunOperationCode,
    RunOperationError,
)
from tuner.execution.coordinator_v1.cursors import (
    AuthenticatedCursorV1,
    CursorContentV1,
    CursorKindV1,
    HMACCursorAuthorityV1,
    decode_cursor,
    encode_cursor,
    encode_cursor_bytes,
)
from tuner.execution.coordinator_v1.operations import _log_run_digest, _project_digest

from .test_operational_cas import _queued_store
from .test_operations_service import operations


def authority(*, generation=1, keys=None, revoked=frozenset()):
    return HMACCursorAuthorityV1(
        "cursor-authority",
        keys or {1: b"a" * 32},
        active_generation=generation,
        revoked_generations=revoked,
    )


def test_exact_binary_layout_hmac_and_canonical_transport() -> None:
    content = CursorContentV1(
        CursorKindV1.RUN_LOGS,
        b"q" * 32,
        after_sequence=9,
    )
    owned = authority().issue(content)
    token = encode_cursor(owned)
    raw = owned.canonical_bytes
    assert len(raw) == 134
    assert len(token[4:]) == 179 and len(token) == 183
    assert raw[:2] == bytes((1, 2))
    assert raw[2:34] == owned.authority_digest
    assert raw[34:38] == (1).to_bytes(4, "big")
    assert raw[38:70] == b"q" * 32
    assert raw[70:94] == b"\0" * 24
    assert raw[94:102] == (9).to_bytes(8, "big")
    expected = hmac.new(
        b"a" * 32,
        b"synaptic-authenticated-cursor/v1\0" + raw[:102],
        hashlib.sha256,
    ).digest()
    assert raw[102:] == expected
    assert decode_cursor(token) == owned
    assert authority().verify(decode_cursor(token)) is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda token: token + "=",
        lambda token: "xx1." + token[4:],
        lambda token: token[:-1],
        lambda token: token[:10] + "+" + token[11:],
    ],
)
def test_transport_rejects_noncanonical_shapes(mutation) -> None:
    token = encode_cursor(
        authority().issue(
            CursorContentV1(CursorKindV1.RUN_LIST, b"q" * 32, after_run_key=b"r" * 32)
        )
    )
    with pytest.raises(ValueError):
        decode_cursor(mutation(token))


def test_restart_rotation_and_revocation_are_host_configured() -> None:
    old = authority()
    content = CursorContentV1(
        CursorKindV1.RUN_LIST, b"q" * 32, after_run_key=b"r" * 32
    )
    token = encode_cursor(old.issue(content))
    restarted = authority()
    assert restarted.verify(decode_cursor(token)) is True

    rotated = authority(generation=2, keys={1: b"a" * 32, 2: b"b" * 32})
    assert rotated.verify(decode_cursor(token)) is True
    assert decode_cursor(encode_cursor(rotated.issue(content))).key_generation == 2

    revoked = authority(
        generation=2,
        keys={1: b"a" * 32, 2: b"b" * 32},
        revoked=frozenset({1}),
    )
    assert revoked.verify(decode_cursor(token)) is False
    with pytest.raises(ValueError, match="must not be reused"):
        authority(generation=2, keys={1: b"a" * 32, 2: b"a" * 32})
    changed = authority(keys={1: b"z" * 32})
    assert changed.verify(decode_cursor(token)) is False


def test_forged_log_sequence_rejects_before_store_foundation_or_reader() -> None:
    store, queued, foundation_record = _queued_store()
    service, reader = operations(store, queued, foundation_record)
    first = service.logs(RunLogsRequest(queued.run, limit=10, maximum_bytes=4096))
    raw = bytearray(decode_cursor(first.next_cursor).canonical_bytes)
    raw[101] ^= 0xFF
    forged = encode_cursor_bytes(bytes(raw))

    class CountingStore:
        def __init__(self, inner):
            self.inner = inner
            self.get_calls = 0

        def get(self, run):
            self.get_calls += 1
            return self.inner.get(run)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    counting = CountingStore(store)
    service._workflows = counting
    service._foundation.get_calls = 0
    prior_reader_calls = reader.log_calls
    with pytest.raises(RunOperationError) as caught:
        service.logs(
            RunLogsRequest(
                queued.run, cursor=forged, limit=10, maximum_bytes=4096
            )
        )
    assert caught.value.code is RunOperationCode.CURSOR_INVALID
    assert counting.get_calls == service._foundation.get_calls == 0
    assert reader.log_calls == prior_reader_calls


def test_forged_list_boundary_rejects_before_store_access() -> None:
    base = authority()
    store, queued, foundation_record = _queued_store()
    service, _ = operations(
        store, queued, foundation_record, cursor_authority=base
    )
    token = encode_cursor(
        base.issue(
            CursorContentV1(
                CursorKindV1.RUN_LIST,
                _project_digest(queued.run.project_ref),
                after_run_key=b"r" * 32,
            )
        )
    )
    raw = bytearray(decode_cursor(token).canonical_bytes)
    raw[70] ^= 0xFF
    forged = encode_cursor_bytes(bytes(raw))

    class CountingStore:
        def __init__(self, inner):
            self.inner = inner
            self.list_calls = 0

        def list_page(self, *args, **kwargs):
            self.list_calls += 1
            return self.inner.list_page(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    counting = CountingStore(store)
    service._workflows = counting
    with pytest.raises(RunOperationError) as caught:
        service.list(RunListRequest(queued.run.project_ref, forged, 10))
    assert caught.value.code is RunOperationCode.CURSOR_INVALID
    assert counting.list_calls == 0


@pytest.mark.parametrize("decision", [False, None, "truthy"])
def test_incoming_cursor_requires_exact_true(decision) -> None:
    base = authority()

    class DecisionAuthority:
        def issue(self, content):
            return base.issue(content)

        def verify(self, cursor):
            return decision

    store, queued, foundation_record = _queued_store()
    valid = encode_cursor(
        base.issue(
            CursorContentV1(
                CursorKindV1.RUN_LOGS,
                _log_run_digest(queued.run),
                after_sequence=2,
            )
        )
    )
    service, _ = operations(
        store, queued, foundation_record, cursor_authority=DecisionAuthority()
    )
    with pytest.raises(RunOperationError) as caught:
        service.logs(RunLogsRequest(queued.run, valid, 10, 4096))
    assert caught.value.code is RunOperationCode.CURSOR_INVALID


def test_throwing_incoming_authority_closes_before_downstream_access() -> None:
    base = authority()

    class ThrowingAuthority:
        def issue(self, content):
            return base.issue(content)

        def verify(self, cursor):
            raise RuntimeError("secret verifier failure")

    store, queued, foundation_record = _queued_store()
    valid = encode_cursor(
        base.issue(
            CursorContentV1(
                CursorKindV1.RUN_LOGS,
                _log_run_digest(queued.run),
                after_sequence=2,
            )
        )
    )
    service, reader = operations(
        store, queued, foundation_record, cursor_authority=ThrowingAuthority()
    )
    service._foundation.get_calls = 0
    with pytest.raises(RunOperationError) as caught:
        service.logs(RunLogsRequest(queued.run, valid, 10, 4096))
    assert caught.value.code is RunOperationCode.CURSOR_INVALID
    assert service._foundation.get_calls == reader.log_calls == 0
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_cross_kind_or_query_cursor_rejects_before_downstream_access() -> None:
    base = authority()
    store, queued, foundation_record = _queued_store()
    service, reader = operations(
        store, queued, foundation_record, cursor_authority=base
    )
    service._foundation.get_calls = 0
    tokens = (
        encode_cursor(
            base.issue(
                CursorContentV1(
                    CursorKindV1.RUN_LIST,
                    _log_run_digest(queued.run),
                    after_run_key=b"r" * 32,
                )
            )
        ),
        encode_cursor(
            base.issue(
                CursorContentV1(
                    CursorKindV1.RUN_LOGS,
                    b"x" * 32,
                    after_sequence=2,
                )
            )
        ),
    )
    for token in tokens:
        with pytest.raises(RunOperationError) as caught:
            service.logs(RunLogsRequest(queued.run, token, 10, 4096))
        assert caught.value.code is RunOperationCode.CURSOR_INVALID
    assert service._foundation.get_calls == reader.log_calls == 0


@pytest.mark.parametrize("decision", [False, None, "truthy"])
def test_issued_cursor_requires_exact_true_self_verification(decision) -> None:
    base = authority()

    class DecisionAuthority:
        def issue(self, content):
            return base.issue(content)

        def verify(self, cursor):
            return decision

    store, queued, foundation_record = _queued_store()
    service, _ = operations(
        store, queued, foundation_record, cursor_authority=DecisionAuthority()
    )
    content = CursorContentV1(
        CursorKindV1.RUN_LOGS,
        _log_run_digest(queued.run),
        after_sequence=2,
    )
    with pytest.raises(RunOperationError) as caught:
        service._issue_cursor(content)
    assert caught.value.code is RunOperationCode.INTEGRITY_ERROR


def test_issued_cursor_must_echo_exact_content_and_type() -> None:
    base = authority()

    class WrongContentAuthority:
        def issue(self, content):
            return base.issue(
                CursorContentV1(
                    CursorKindV1.RUN_LOGS,
                    b"x" * 32,
                    after_sequence=content.after_sequence,
                )
            )

        def verify(self, cursor):
            return True

    class WrongTypeAuthority:
        def issue(self, content):
            return object()

        def verify(self, cursor):
            return True

    store, queued, foundation_record = _queued_store()
    content = CursorContentV1(
        CursorKindV1.RUN_LOGS,
        _log_run_digest(queued.run),
        after_sequence=2,
    )
    for cursor_authority in (WrongContentAuthority(), WrongTypeAuthority()):
        service, _ = operations(
            store,
            queued,
            foundation_record,
            cursor_authority=cursor_authority,
        )
        with pytest.raises(RunOperationError) as caught:
            service._issue_cursor(content)
        assert caught.value.code is RunOperationCode.INTEGRITY_ERROR
