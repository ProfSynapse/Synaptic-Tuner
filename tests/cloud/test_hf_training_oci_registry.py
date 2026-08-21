from __future__ import annotations

import hashlib
import json

import pytest

from tuner.cloud import hf_training_oci_registry as registry


def _raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _documents(kind: str = "manifest"):
    config = _raw(
        {"architecture": "amd64", "os": "linux", "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "5" * 64]}}
    )
    child = _raw(
        {
            "schemaVersion": 2,
            "mediaType": registry.CHILD_MEDIA_TYPE,
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": _digest(config),
                "size": len(config),
            },
            "layers": [
                {"mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip", "digest": "sha256:" + "4" * 64, "size": 200}
            ],
        }
    )
    if kind == "index":
        requested = _raw(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
                "manifests": [
                    {"mediaType": registry.CHILD_MEDIA_TYPE, "digest": _digest(child), "size": len(child), "platform": {"os": "linux", "architecture": "amd64"}}
                ],
            }
        )
    else:
        requested = child
    return requested, child, config


def _response(status: int, url: str, body: bytes, content_type: str, **headers) -> registry.HTTPResponse:
    normalized = {"content-length": str(len(body)), "content-type": content_type, **headers}
    return registry.HTTPResponse(status, url, normalized, body)


class FakeTransport:
    def __init__(self, kind: str = "manifest") -> None:
        self.requested, self.child, self.config = _documents(kind)
        self.kind = kind
        self.calls: list[registry.HTTPRequest] = []

    @property
    def reference(self) -> str:
        return f"{registry.REGISTRY_REPOSITORY}@{_digest(self.requested)}"

    def __call__(self, request: registry.HTTPRequest) -> registry.HTTPResponse:
        self.calls.append(request)
        if request.url == registry.TOKEN_URL:
            body = _raw({"token": "a" * 32})
            return _response(200, request.url, body, "application/json")
        if "/manifests/" in request.url:
            digest = request.url.rsplit("/", 1)[-1]
            if "Authorization" not in request.headers:
                challenge = (
                    f'Bearer realm="https://{registry.TOKEN_HOST}/token",service="{registry.TOKEN_SERVICE}",'
                    f'scope="{registry.PULL_SCOPE}"'
                )
                return registry.HTTPResponse(401, request.url, {"www-authenticate": challenge, "content-length": "0"}, b"")
            body = self.requested if digest == _digest(self.requested) else self.child
            media = json.loads(body)["mediaType"]
            return _response(200, request.url, body, media, **{"docker-content-digest": digest})
        raise AssertionError("unexpected registry request")


@pytest.mark.parametrize("kind", ["manifest", "index"])
def test_fetch_authenticates_direct_manifest_or_index_child_and_config(kind: str) -> None:
    transport = FakeTransport(kind)
    documents = registry.fetch_registry_documents(transport.reference, transport=transport)
    assert documents.requested_kind == kind
    assert documents.child_raw is (None if kind == "manifest" else transport.child)
    assert documents.config_digest == _digest(transport.config)
    assert documents.config_size == len(transport.config)
    assert documents.child_digest == _digest(transport.child)
    token_call = next(call for call in transport.calls if call.url == registry.TOKEN_URL)
    assert "Authorization" not in token_call.headers
    assert all("/blobs/" not in call.url for call in transport.calls)
    assert transport.calls[-1].url.endswith(f"/manifests/{documents.child_digest}")


def test_exact_repository_and_pull_scope_only() -> None:
    with pytest.raises(registry.OCIRegistryError, match="REFERENCE_INVALID"):
        registry.fetch_registry_documents("unsloth/unsloth@sha256:" + "a" * 64, transport=FakeTransport())
    transport = FakeTransport()
    registry.fetch_registry_documents(transport.reference, transport=transport)
    challenge_request = transport.calls[0]
    assert challenge_request.url.startswith(f"https://{registry.REGISTRY_HOST}/v2/unsloth/unsloth/manifests/")
    assert f"scope=repository:unsloth%2Funsloth:pull" in registry.TOKEN_URL


def test_bounded_closed_distribution_json_challenge_is_accepted() -> None:
    base = FakeTransport()

    def transport(request: registry.HTTPRequest) -> registry.HTTPResponse:
        response = base(request)
        if response.status == 401:
            body = _raw({"errors": [{"code": "UNAUTHORIZED", "message": "authentication required", "detail": []}]})
            headers = dict(response.headers)
            headers.update({"content-length": str(len(body)), "content-type": "application/json"})
            return registry.HTTPResponse(401, request.url, headers, body)
        return response

    assert registry.fetch_registry_documents(base.reference, transport=transport).requested_kind == "manifest"


def test_bounded_chunked_token_response_is_accepted() -> None:
    base = FakeTransport()

    def transport(request: registry.HTTPRequest) -> registry.HTTPResponse:
        response = base(request)
        if request.url == registry.TOKEN_URL:
            headers = dict(response.headers)
            headers.pop("content-length")
            headers["transfer-encoding"] = "Chunked"
            return registry.HTTPResponse(response.status, response.url, headers, response.body)
        return response

    assert registry.fetch_registry_documents(base.reference, transport=transport).requested_kind == "manifest"


@pytest.mark.parametrize(
    ("transfer_encoding", "include_content_length"),
    [("chunked", True), ("gzip", False), ("gzip, chunked", False)],
)
def test_token_rejects_ambiguous_or_nonexact_transfer_encoding(
    transfer_encoding: str, include_content_length: bool,
) -> None:
    base = FakeTransport()

    def transport(request: registry.HTTPRequest) -> registry.HTTPResponse:
        response = base(request)
        if request.url == registry.TOKEN_URL:
            headers = dict(response.headers)
            if not include_content_length:
                headers.pop("content-length")
            headers["transfer-encoding"] = transfer_encoding
            return registry.HTTPResponse(response.status, response.url, headers, response.body)
        return response

    with pytest.raises(registry.OCIRegistryError, match="CONTENT_INVALID"):
        registry.fetch_registry_documents(base.reference, transport=transport)


@pytest.mark.parametrize("tamper", ["digest", "media", "length", "challenge"])
def test_manifest_authentication_fails_closed(tamper: str) -> None:
    base = FakeTransport()

    def hostile(request: registry.HTTPRequest) -> registry.HTTPResponse:
        response = base(request)
        if "/manifests/" in request.url and "Authorization" in request.headers:
            headers = dict(response.headers)
            body = response.body
            if tamper == "digest":
                headers["docker-content-digest"] = "sha256:" + "0" * 64
            elif tamper == "media":
                headers["content-type"] = "text/plain"
            elif tamper == "length":
                headers["content-length"] = str(len(body) + 1)
            return registry.HTTPResponse(response.status, response.url, headers, body)
        if tamper == "challenge" and response.status == 401:
            headers = dict(response.headers)
            headers["www-authenticate"] = headers["www-authenticate"].replace(":pull", ":push")
            return registry.HTTPResponse(response.status, response.url, headers, response.body)
        return response

    with pytest.raises(registry.OCIRegistryError):
        registry.fetch_registry_documents(base.reference, transport=hostile)


def test_registry_capture_never_requests_config_blob_or_redirect_host() -> None:
    transport = FakeTransport("index")
    registry.fetch_registry_documents(transport.reference, transport=transport)
    assert all("/blobs/" not in call.url for call in transport.calls)
    assert {call.url.split("/v2/", 1)[0] for call in transport.calls if "/v2/" in call.url} == {
        f"https://{registry.REGISTRY_HOST}"
    }


def test_token_rejects_header_injection_characters() -> None:
    base = FakeTransport()

    def hostile(request: registry.HTTPRequest) -> registry.HTTPResponse:
        response = base(request)
        if request.url == registry.TOKEN_URL:
            body = _raw({"token": "safe-prefix\r\nInjected: value"})
            return _response(200, request.url, body, "application/json")
        return response

    with pytest.raises(registry.OCIRegistryError, match="AUTH_INVALID"):
        registry.fetch_registry_documents(base.reference, transport=hostile)


@pytest.mark.parametrize(
    "body",
    [b'{"token":NaN}', b'{"token":Infinity}', b"[" * 65 + b"]" * 65],
)
def test_registry_rejects_nonfinite_or_deep_json_without_recursion(body: bytes) -> None:
    base = FakeTransport()

    def hostile(request: registry.HTTPRequest) -> registry.HTTPResponse:
        response = base(request)
        if request.url == registry.TOKEN_URL:
            return _response(200, request.url, body, "application/json")
        return response

    with pytest.raises(registry.OCIRegistryError):
        registry.fetch_registry_documents(base.reference, transport=hostile)


def test_registry_rejects_enormous_content_length_closed() -> None:
    base = FakeTransport()

    def hostile(request: registry.HTTPRequest) -> registry.HTTPResponse:
        response = base(request)
        if request.url == registry.TOKEN_URL:
            headers = dict(response.headers)
            headers["content-length"] = "9" * 100000
            return registry.HTTPResponse(response.status, response.url, headers, response.body)
        return response

    with pytest.raises(registry.OCIRegistryError, match="CONTENT_INVALID"):
        registry.fetch_registry_documents(base.reference, transport=hostile)


def test_registry_enforces_aggregate_manifest_budget(monkeypatch) -> None:
    transport = FakeTransport("index")
    monkeypatch.setattr(
        registry, "MAX_MANIFEST_BYTES", len(transport.requested) + len(transport.child) - 1,
    )
    with pytest.raises(registry.OCIRegistryError, match="CONTENT_INVALID"):
        registry.fetch_registry_documents(transport.reference, transport=transport)
