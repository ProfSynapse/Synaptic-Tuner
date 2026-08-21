"""Bounded anonymous Docker Hub Distribution client for training-lock evidence."""

from __future__ import annotations

import hashlib
import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping


REGISTRY_HOST = "registry-1.docker.io"
TOKEN_HOST = "auth.docker.io"
REGISTRY_REPOSITORY = "docker.io/unsloth/unsloth"
PROVIDER_REPOSITORY = "unsloth/unsloth"
REGISTRY_PATH = "unsloth/unsloth"
PULL_SCOPE = "repository:unsloth/unsloth:pull"
TOKEN_SERVICE = "registry.docker.io"
TOKEN_URL = f"https://{TOKEN_HOST}/token?service={TOKEN_SERVICE}&scope={urllib.parse.quote(PULL_SCOPE, safe=':')}"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_TOKEN_BYTES = 64 * 1024
MAX_JSON_DEPTH = 64
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
INDEX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }
)
CHILD_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
CONFIG_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.config.v1+json",
        "application/vnd.docker.container.image.v1+json",
    }
)
ACCEPT = ", ".join(sorted(INDEX_MEDIA_TYPES | {CHILD_MEDIA_TYPE}))
REASON_CODES = frozenset(
    {
        "REFERENCE_INVALID",
        "AUTH_INVALID",
        "HTTP_INVALID",
        "CONTENT_INVALID",
        "DIGEST_INVALID",
        "DOCUMENT_INVALID",
    }
)


class OCIRegistryError(RuntimeError):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code if reason_code in REASON_CODES else "HTTP_INVALID"
        super().__init__(self.reason_code)


@dataclass(frozen=True)
class HTTPRequest:
    url: str
    headers: Mapping[str, str]
    maximum_bytes: int


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    url: str
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[HTTPRequest], HTTPResponse]


@dataclass(frozen=True)
class RegistryDocuments:
    requested_raw: bytes
    child_raw: bytes | None
    requested_digest: str
    requested_media_type: str
    requested_kind: str
    child_digest: str
    child_media_type: str
    config_digest: str
    config_media_type: str
    config_size: int


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _closed_url(url: str, *, host: str, allow_query: bool) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != host
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (parsed.query and not allow_query)
    ):
        raise OCIRegistryError("HTTP_INVALID")
    return parsed


def stdlib_transport(request: HTTPRequest) -> HTTPResponse:
    parsed = urllib.parse.urlsplit(request.url)
    if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise OCIRegistryError("HTTP_INVALID")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        _NoRedirect(),
    )
    req = urllib.request.Request(request.url, headers=dict(request.headers), method="GET")
    try:
        response = opener.open(req, timeout=30)
    except urllib.error.HTTPError as exc:
        response = exc
    except (OSError, urllib.error.URLError) as exc:
        raise OCIRegistryError("HTTP_INVALID") from exc
    try:
        raw = response.read(request.maximum_bytes + 1)
        headers: dict[str, str] = {}
        for key in response.headers:
            values = response.headers.get_all(key) or []
            lowered = key.lower()
            if lowered in headers or len(values) != 1:
                raise OCIRegistryError("HTTP_INVALID")
            headers[lowered] = values[0]
        result = HTTPResponse(int(response.status), str(response.geturl()), headers, raw)
    except (OSError, ValueError) as exc:
        raise OCIRegistryError("HTTP_INVALID") from exc
    finally:
        response.close()
    return result


def parse_reference(reference: str) -> str:
    prefix = REGISTRY_REPOSITORY + "@"
    if not reference.startswith(prefix) or reference.count("@") != 1:
        raise OCIRegistryError("REFERENCE_INVALID")
    digest = reference[len(prefix):]
    if not DIGEST.fullmatch(digest):
        raise OCIRegistryError("REFERENCE_INVALID")
    return digest


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _object(raw: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    _assert_json_depth(raw)

    def reject_constant(_value: str) -> object:
        raise ValueError("constant")

    def bounded_integer(value: str) -> int:
        if len(value.lstrip("-")) > 20:
            raise ValueError("integer")
        return int(value)

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=lambda _value: (_ for _ in ()).throw(ValueError("float")),
            parse_int=bounded_integer,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise OCIRegistryError("DOCUMENT_INVALID") from exc
    if not isinstance(value, dict):
        raise OCIRegistryError("DOCUMENT_INVALID")
    return value


def _assert_json_depth(raw: bytes) -> None:
    depth = 0
    quoted = False
    escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                quoted = False
        elif byte == 0x22:
            quoted = True
        elif byte in (0x5B, 0x7B):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise OCIRegistryError("DOCUMENT_INVALID")
        elif byte in (0x5D, 0x7D):
            depth -= 1
            if depth < 0:
                raise OCIRegistryError("DOCUMENT_INVALID")
    if quoted or escaped or depth:
        raise OCIRegistryError("DOCUMENT_INVALID")


def _header(response: HTTPResponse, name: str) -> str | None:
    return response.headers.get(name.lower())


def _content_type(response: HTTPResponse) -> str:
    value = _header(response, "content-type")
    if not value:
        raise OCIRegistryError("CONTENT_INVALID")
    return value.split(";", 1)[0].strip().lower()


def _bounded_response(
    response: HTTPResponse, *, requested_url: str, maximum: int,
    allow_chunked: bool = False,
) -> None:
    if response.url != requested_url or not response.body or len(response.body) > maximum:
        raise OCIRegistryError("HTTP_INVALID")
    length = _header(response, "content-length")
    encoding = _header(response, "content-encoding")
    transfer = _header(response, "transfer-encoding")
    try:
        parsed_length = int(length) if length is not None and length.isascii() and length.isdigit() and len(length) <= 20 else None
    except (ValueError, OverflowError):
        parsed_length = None
    content_length_framed = length is not None and parsed_length == len(response.body) and transfer is None
    chunked_framed = (
        allow_chunked and length is None and transfer is not None
        and transfer.lower() == "chunked"
    )
    if (
        encoding not in (None, "identity")
        or not (content_length_framed or chunked_framed)
    ):
        raise OCIRegistryError("CONTENT_INVALID")


def _challenge(response: HTTPResponse, *, requested_url: str) -> None:
    expected = (
        f'Bearer realm="https://{TOKEN_HOST}/token",service="{TOKEN_SERVICE}",'
        f'scope="{PULL_SCOPE}"'
    )
    length = _header(response, "content-length")
    try:
        parsed_length = int(length) if length is not None and length.isdigit() and len(length) <= 20 else None
    except (ValueError, OverflowError):
        parsed_length = None
    if (
        response.status != 401
        or response.url != requested_url
        or _header(response, "www-authenticate") != expected
        or parsed_length != len(response.body)
        or len(response.body) > MAX_TOKEN_BYTES
    ):
        raise OCIRegistryError("AUTH_INVALID")
    if response.body:
        if _content_type(response) != "application/json":
            raise OCIRegistryError("AUTH_INVALID")
        value = _object(response.body)
        errors = value.get("errors") if set(value) == {"errors"} else None
        if (
            not isinstance(errors, list) or len(errors) != 1
            or not isinstance(errors[0], dict)
            or not {"code", "message"} <= set(errors[0])
            or not set(errors[0]) <= {"code", "message", "detail"}
            or errors[0].get("code") != "UNAUTHORIZED"
        ):
            raise OCIRegistryError("AUTH_INVALID")


def _token(transport: Transport) -> str:
    _closed_url(TOKEN_URL, host=TOKEN_HOST, allow_query=True)
    response = transport(
        HTTPRequest(TOKEN_URL, {"Accept": "application/json", "User-Agent": "synaptic-hf-training-lock/1"}, MAX_TOKEN_BYTES)
    )
    _bounded_response(
        response, requested_url=TOKEN_URL, maximum=MAX_TOKEN_BYTES,
        allow_chunked=True,
    )
    if response.status != 200 or _content_type(response) != "application/json":
        raise OCIRegistryError("AUTH_INVALID")
    value = _object(response.body)
    if not set(value) <= {"token", "access_token", "expires_in", "issued_at"}:
        raise OCIRegistryError("AUTH_INVALID")
    token = value.get("token")
    if (
        not isinstance(token, str) or not 16 <= len(token) <= 8192
        or not re.fullmatch(r"[A-Za-z0-9._~-]+", token)
        or value.get("access_token") not in (None, token)
    ):
        raise OCIRegistryError("AUTH_INVALID")
    return token


def _manifest(
    digest: str, *, transport: Transport, token: str | None
) -> tuple[bytes, str, str]:
    if not DIGEST.fullmatch(digest):
        raise OCIRegistryError("REFERENCE_INVALID")
    url = f"https://{REGISTRY_HOST}/v2/{REGISTRY_PATH}/manifests/{digest}"
    _closed_url(url, host=REGISTRY_HOST, allow_query=False)
    headers = {"Accept": ACCEPT, "User-Agent": "synaptic-hf-training-lock/1"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    response = transport(HTTPRequest(url, headers, MAX_MANIFEST_BYTES))
    if token is None:
        _challenge(response, requested_url=url)
        return b"", "", ""
    _bounded_response(response, requested_url=url, maximum=MAX_MANIFEST_BYTES)
    if response.status != 200:
        raise OCIRegistryError("HTTP_INVALID")
    media_type = _content_type(response)
    if media_type not in INDEX_MEDIA_TYPES | {CHILD_MEDIA_TYPE}:
        raise OCIRegistryError("CONTENT_INVALID")
    if _header(response, "docker-content-digest") != digest or _sha256(response.body) != digest:
        raise OCIRegistryError("DIGEST_INVALID")
    return response.body, media_type, url


def fetch_registry_documents(reference: str, *, transport: Transport = stdlib_transport) -> RegistryDocuments:
    requested_digest = parse_reference(reference)
    empty, _, _ = _manifest(requested_digest, transport=transport, token=None)
    if empty:
        raise OCIRegistryError("AUTH_INVALID")
    token = _token(transport)
    requested_raw, requested_media, _ = _manifest(requested_digest, transport=transport, token=token)
    requested = _object(requested_raw)
    if requested.get("schemaVersion") != 2 or requested.get("mediaType") != requested_media:
        raise OCIRegistryError("DOCUMENT_INVALID")
    child_raw: bytes | None = None
    if requested_media in INDEX_MEDIA_TYPES:
        manifests = requested.get("manifests")
        if not isinstance(manifests, list) or not 1 <= len(manifests) <= 256:
            raise OCIRegistryError("DOCUMENT_INVALID")
        matches = []
        for item in manifests:
            if not isinstance(item, dict):
                raise OCIRegistryError("DOCUMENT_INVALID")
            platform = item.get("platform")
            if isinstance(platform, dict) and platform == {"os": "linux", "architecture": "amd64"}:
                matches.append(item)
        if len(matches) != 1:
            raise OCIRegistryError("DOCUMENT_INVALID")
        child_digest = matches[0].get("digest")
        if matches[0].get("mediaType") != CHILD_MEDIA_TYPE or not isinstance(child_digest, str):
            raise OCIRegistryError("DOCUMENT_INVALID")
        child_raw, child_media, _ = _manifest(child_digest, transport=transport, token=token)
        if len(requested_raw) + len(child_raw) > MAX_MANIFEST_BYTES:
            raise OCIRegistryError("CONTENT_INVALID")
        if child_media != CHILD_MEDIA_TYPE or matches[0].get("size") != len(child_raw):
            raise OCIRegistryError("DOCUMENT_INVALID")
        child = _object(child_raw)
        requested_kind = "index"
    else:
        child_digest = requested_digest
        child_media = requested_media
        child = requested
        requested_kind = "manifest"
    if child_media != CHILD_MEDIA_TYPE or child.get("schemaVersion") != 2 or child.get("mediaType") != CHILD_MEDIA_TYPE:
        raise OCIRegistryError("DOCUMENT_INVALID")
    config = child.get("config")
    if not isinstance(config, dict) or set(config) != {"mediaType", "digest", "size"}:
        raise OCIRegistryError("DOCUMENT_INVALID")
    config_digest, config_media, config_size = config.get("digest"), config.get("mediaType"), config.get("size")
    if (
        not isinstance(config_digest, str) or not DIGEST.fullmatch(config_digest)
        or config_media not in CONFIG_MEDIA_TYPES or not isinstance(config_size, int)
        or not 1 <= config_size <= MAX_CONFIG_BYTES
    ):
        raise OCIRegistryError("DOCUMENT_INVALID")
    return RegistryDocuments(
        requested_raw=requested_raw,
        child_raw=child_raw,
        requested_digest=requested_digest,
        requested_media_type=requested_media,
        requested_kind=requested_kind,
        child_digest=child_digest,
        child_media_type=child_media,
        config_digest=config_digest,
        config_media_type=config_media,
        config_size=config_size,
    )


__all__ = [
    "CHILD_MEDIA_TYPE", "CONFIG_MEDIA_TYPES", "HTTPRequest",
    "HTTPResponse", "OCIRegistryError", "PROVIDER_REPOSITORY", "PULL_SCOPE",
    "REGISTRY_HOST", "REGISTRY_REPOSITORY", "RegistryDocuments", "TOKEN_HOST",
    "fetch_registry_documents", "parse_reference", "stdlib_transport",
]
