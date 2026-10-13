"""Optional host-owned atomic head/history extension to the generic store ports.

Only a single object supplying both ports can advertise this method. Rejection
or failure never authorizes separate writes or lifecycle effect replay.
"""
from __future__ import annotations

from typing import Protocol
from inspect import getattr_static


class AtomicLifecycleStorePort(Protocol):
    def commit_record_and_append(
        self, *, record_partition: str, key: str, expected_revision: int | None,
        record_canonical: bytes, stream_partition: str, stream_key: str,
        expected_tail: int | None, sequence: int, stream_canonical: bytes,
    ) -> bool: ...


def atomic_commit_method(records, streams):
    if records is not streams:
        return None
    descriptor = getattr_static(records, 'commit_record_and_append', None)
    if not callable(descriptor):
        return None
    method = getattr(records, 'commit_record_and_append')
    if not callable(method):
        raise TypeError('atomic_store_method_invalid')

    def commit(**kwargs):
        result = method(**kwargs)
        if type(result) is not bool:
            raise TypeError('atomic_store_result_invalid')
        return result
    return commit
