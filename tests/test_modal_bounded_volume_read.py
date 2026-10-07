"""Provider-free tests for the pinned v1 ranged Volume transport."""
from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

pytest.importorskip("modal_proto", reason="pinned Modal protocol runtime required")

from tuner.execution.providers.modal.bounded_volume_read import (
    BoundedModalVolumeReader,
    BoundedVolumeReadError,
)


class FakeContent:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.read_calls = 0
        self.read_sizes = []

    async def read(self, size):
        assert 0 < size <= 64 * 1024
        self.read_calls += 1
        self.read_sizes.append(size)
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) > size:
            self.chunks.insert(0, chunk[size:])
            return chunk[:size]
        return chunk


class FakeGet:
    def __init__(self, chunks, *, status=200, headers=None):
        self.status = status
        self.headers = headers or {}
        self.content = FakeContent(chunks)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def get(self, url, *, allow_redirects):
        assert allow_redirects is False
        self.urls.append(url)
        return self.responses[len(self.urls) - 1]


class FakeStub:
    def __init__(self, response):
        self.response = response
        self.request = None

    async def VolumeGetFile2(self, request, *, retry, timeout):
        assert retry is None and timeout == 15
        self.request = request
        return self.response


def make_reader(data=b"hello", *, response=None, chunks=None):
    if response is None:
        response = SimpleNamespace(
            size=len(data), start=0, len=len(data), get_urls=("https://volume.example/block",)
        )
    stub = FakeStub(response)
    session = FakeSession([FakeGet(chunks if chunks is not None else [data])])
    reader = BoundedModalVolumeReader(
        sdk=SimpleNamespace(__version__="1.5.4"),
        client=SimpleNamespace(stub=stub),
        http_session_factory=lambda: session,
    )
    return reader, stub, session


def exact(reader, data=b"hello", *, max_bytes=16):
    return asyncio.run(
        reader.read_exact(
            volume_id="vo-ABC123", path="operations/run/output/file",
            expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest(),
            max_bytes=max_bytes,
        )
    )


def test_exact_read_requests_one_extra_byte_and_validates_digest():
    reader, stub, session = make_reader(chunks=[b"he", b"llo"])
    assert exact(reader) == b"hello"
    assert (stub.request.volume_id, stub.request.path, stub.request.start, stub.request.len) == (
        "vo-ABC123", "operations/run/output/file", 0, 17,
    )
    assert session.closed and session.responses[0].closed


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(size=17, start=0, len=17, get_urls=("https://volume.example/block",)),
        SimpleNamespace(size=5, start=1, len=5, get_urls=("https://volume.example/block",)),
        SimpleNamespace(size=5, start=0, len=0, get_urls=("https://volume.example/block",)),
        SimpleNamespace(size=5, start=0, len=5, get_urls=()),
        SimpleNamespace(size=5, start=0, len=5, get_urls=("http://volume.example/block",)),
    ],
)
def test_rejects_invalid_range_without_downloading(response):
    reader, _, session = make_reader(response=response)
    with pytest.raises(BoundedVolumeReadError):
        exact(reader)
    assert not session.urls


def test_rejects_oversize_chunk_immediately_and_closes_response():
    first = b"hello!"
    reader, _, session = make_reader(chunks=[first, b"never read"])
    with pytest.raises(BoundedVolumeReadError, match="modal_volume_size_mismatch"):
        exact(reader)
    assert session.responses[0].content.read_calls == 1
    assert session.responses[0].closed and session.closed


def test_rejects_response_length_larger_than_expected_without_downloading():
    response = SimpleNamespace(
        size=5, start=0, len=6, get_urls=("https://volume.example/block",),
    )
    reader, _, session = make_reader(response=response)
    with pytest.raises(BoundedVolumeReadError, match="modal_volume_range_invalid"):
        exact(reader)
    assert not session.urls


def test_rejects_multiple_urls_for_small_marker_without_downloading():
    response = SimpleNamespace(
        size=5,
        start=0,
        len=5,
        get_urls=("https://volume.example/first", "https://volume.example/second"),
    )
    reader, _, session = make_reader(response=response)
    with pytest.raises(BoundedVolumeReadError, match="modal_volume_range_invalid"):
        exact(reader)
    assert not session.urls


def test_32_byte_marker_reads_at_most_one_extra_byte_and_rejects_oversize_body():
    marker = b"m" * 32
    reader, stub, session = make_reader(data=marker, chunks=[b"x" * 33])
    with pytest.raises(BoundedVolumeReadError, match="modal_volume_size_mismatch"):
        exact(reader, data=marker, max_bytes=32)
    assert stub.request.len == 33
    assert session.responses[0].content.read_sizes == [33]
    assert max(session.responses[0].content.read_sizes) <= 33


def test_rejects_declared_oversize_before_body():
    reader, _, session = make_reader()
    session.responses[0].headers = {"Content-Length": "17"}
    with pytest.raises(BoundedVolumeReadError, match="modal_volume_size_mismatch"):
        exact(reader)
    assert session.responses[0].content.read_calls == 0


def test_rejects_bad_digest_and_bad_pre_network_bounds():
    reader, stub, _ = make_reader(data=b"HELLO")
    with pytest.raises(BoundedVolumeReadError, match="modal_volume_digest_mismatch"):
        exact(reader)
    stub.request = None
    with pytest.raises(BoundedVolumeReadError, match="modal_volume_bound_invalid"):
        exact(reader, max_bytes=4)
    assert stub.request is None


def test_rejects_oversized_volume_id_before_provider_call():
    reader, stub, _ = make_reader()
    with pytest.raises(BoundedVolumeReadError, match="modal_volume_identity_invalid"):
        asyncio.run(reader.read_exact(
            volume_id="vo-" + "A" * 254,
            path="operations/run/output/file", expected_size=5,
            expected_sha256=hashlib.sha256(b"hello").hexdigest(), max_bytes=16,
        ))
    assert stub.request is None
