"""Internal, exact-ID Modal v1 reads with an explicit byte budget.

This transport does not authenticate a Volume's ownership or its contents. Its
caller must supply the independently authenticated object ID, path, size and
digest. No named lookup, implicit client, or SDK ``Volume.read_file`` is used.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from .contracts import canonical_path, strict_int

_SDK_VERSION = "1.5.4"
_VOLUME_ID = re.compile(r"vo-[A-Za-z0-9]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHUNK_BYTES = 64 * 1024
_MAX_URLS = 64
_MAX_BYTES = 64 * 1024 * 1024
_RPC_TIMEOUT_SECONDS = 15


class BoundedVolumeReadError(RuntimeError):
    """Fixed, non-secret transport failure code."""


def _volume_id(value: str) -> str:
    if type(value) is not str or len(value) > 256 or _VOLUME_ID.fullmatch(value) is None:
        raise BoundedVolumeReadError("modal_volume_identity_invalid")
    return value


def _path(value: str) -> str:
    try:
        path = canonical_path(value)
        if len(path.encode("utf-8")) > 1024:
            raise ValueError
        return path
    except (TypeError, ValueError):
        raise BoundedVolumeReadError("modal_volume_path_invalid") from None


def _bound(value: int, *, maximum: int, minimum: int = 1) -> int:
    try:
        return strict_int(value, "bound", minimum=minimum, maximum=maximum)
    except ValueError:
        raise BoundedVolumeReadError("modal_volume_bound_invalid") from None


def _digest(value: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise BoundedVolumeReadError("modal_volume_digest_invalid")
    return value


def _https_url(value: str) -> str:
    if type(value) is not str or len(value) > 8192:
        raise BoundedVolumeReadError("modal_volume_url_invalid")
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.fragment:
        raise BoundedVolumeReadError("modal_volume_url_invalid")
    return value


class BoundedModalVolumeReader:
    """Low-level Modal 1.5.4 Volume v1 transport bound to one explicit client.

    ``http_session_factory`` is injected only to test the signed-URL transport
    without provider I/O. The default owns and closes a fresh aiohttp session.
    """

    def __init__(
        self,
        *,
        sdk: Any,
        client: Any,
        http_session_factory: Callable[[], Any] | None = None,
    ) -> None:
        if sdk is None or getattr(sdk, "__version__", None) != _SDK_VERSION:
            raise BoundedVolumeReadError("modal_sdk_version_mismatch")
        if client is None or getattr(client, "stub", None) is None:
            raise BoundedVolumeReadError("modal_client_unavailable")
        self._client = client
        if http_session_factory is None:
            import aiohttp

            def _new_session() -> Any:
                timeout = aiohttp.ClientTimeout(total=30, sock_read=10)
                return aiohttp.ClientSession(
                    timeout=timeout, read_bufsize=_CHUNK_BYTES,
                    auto_decompress=False,
                )

            http_session_factory = _new_session
        self._session_factory = http_session_factory

    async def read_exact(
        self,
        *,
        volume_id: str,
        path: str,
        expected_size: int,
        expected_sha256: str,
        max_bytes: int,
    ) -> bytes:
        """Return only a previously bound exact file, or fail with a fixed code.

        ``max_bytes + 1`` is requested so an oversized file can be detected at
        the RPC boundary. Each URL is downloaded serially and closed as soon as
        the cumulative count exceeds the smaller exact-size budget.
        """
        volume_id = _volume_id(volume_id)
        path = _path(path)
        max_bytes = _bound(max_bytes, maximum=_MAX_BYTES)
        expected_size = _bound(expected_size, maximum=max_bytes, minimum=0)
        expected_sha256 = _digest(expected_sha256)

        try:
            from modal_proto import api_pb2

            request = api_pb2.VolumeGetFile2Request(
                volume_id=volume_id, path=path, start=0, len=max_bytes + 1
            )
            response = await asyncio.wait_for(
                self._client.stub.VolumeGetFile2(
                    request, retry=None, timeout=_RPC_TIMEOUT_SECONDS,
                ), timeout=_RPC_TIMEOUT_SECONDS + 1,
            )
            size = response.size
            start = response.start
            length = response.len
            url_limit = 1 if expected_size <= _CHUNK_BYTES else _MAX_URLS
            if len(response.get_urls) > url_limit:
                raise BoundedVolumeReadError("modal_volume_range_invalid")
            urls = tuple(response.get_urls)
            if (
                type(size) is not int or size != expected_size
                or type(start) is not int or start != 0
                or type(length) is not int or length != expected_size
                or not (0 if expected_size == 0 else 1) <= len(urls) <= url_limit
            ):
                raise BoundedVolumeReadError("modal_volume_range_invalid")
            urls = tuple(_https_url(url) for url in urls)
        except BoundedVolumeReadError:
            raise
        except Exception:
            raise BoundedVolumeReadError("modal_volume_range_unavailable") from None

        result = bytearray()
        try:
            async with self._session_factory() as session:
                for url in urls:
                    async with session.get(url, allow_redirects=False) as response:
                        if response.status != 200:
                            raise BoundedVolumeReadError("modal_volume_download_failed")
                        declared = response.headers.get("Content-Length")
                        if declared is not None:
                            if not declared.isdecimal() or int(declared) > expected_size - len(result):
                                raise BoundedVolumeReadError("modal_volume_size_mismatch")
                        while True:
                            chunk = await response.content.read(
                                min(_CHUNK_BYTES, expected_size - len(result) + 1)
                            )
                            if not chunk:
                                break
                            if type(chunk) is not bytes or len(chunk) > _CHUNK_BYTES:
                                raise BoundedVolumeReadError("modal_volume_download_failed")
                            if len(chunk) > expected_size - len(result):
                                raise BoundedVolumeReadError("modal_volume_size_mismatch")
                            result.extend(chunk)
        except BoundedVolumeReadError:
            raise
        except Exception:
            raise BoundedVolumeReadError("modal_volume_download_failed") from None

        if len(result) != expected_size:
            raise BoundedVolumeReadError("modal_volume_size_mismatch")
        if hashlib.sha256(result).hexdigest() != expected_sha256:
            raise BoundedVolumeReadError("modal_volume_digest_mismatch")
        return bytes(result)
