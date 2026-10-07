"""Repository identities must be canonical, policy-bound, and secret-free."""

from __future__ import annotations

import pytest

from tuner.project.errors import RepositoryUrlError
from tuner.project.secrets import SecretRef
from tuner.project.source_bundle import RepositoryLocation


@pytest.mark.parametrize(
    ("unsafe_url", "secret_marker"),
    [
        ("https://token-value@example.test/org/repo.git", "token-value"),
        ("https://example.test/org/repo.git?token=token-value", "token-value"),
        ("https://example.test/org/repo.git#token-value", "token-value"),
        ("ssh://git:token-value@example.test/org/repo.git", "token-value"),
        ("ssh://admin@example.test/org/repo.git", "admin"),
        ("file:///tmp/private-repo", "private-repo"),
        ("ext::sh -c token-value", "token-value"),
        ("https://example.test@evil.test/org/repo.git", "example.test"),
    ],
)
def test_credential_bearing_and_unsafe_repository_urls_fail_without_echoing_secrets(
    unsafe_url: str, secret_marker: str
) -> None:
    with pytest.raises(RepositoryUrlError) as exc_info:
        RepositoryLocation.parse(unsafe_url)

    assert exc_info.value.code == "PROJECT_REPOSITORY_URL_INVALID"
    assert secret_marker not in str(exc_info.value)
    assert unsafe_url not in str(exc_info.value)


def test_repository_policy_rejects_unapproved_host_before_credentials_attach() -> None:
    reference = SecretRef(provider="provider_secret", name="PRIVATE_GIT_TOKEN")

    with pytest.raises(RepositoryUrlError, match="host is not allowed") as exc_info:
        RepositoryLocation.parse(
            "https://lookalike.example.test/org/private.git",
            credential=reference,
            allowed_hosts={"example.test"},
            allowed_schemes={"https"},
        )

    assert "PRIVATE_GIT_TOKEN" not in str(exc_info.value)


def test_repository_policy_rejects_unapproved_transport() -> None:
    with pytest.raises(RepositoryUrlError, match="scheme is not allowed"):
        RepositoryLocation.parse(
            "ssh://git@example.test/org/private.git",
            allowed_hosts={"example.test"},
            allowed_schemes={"https"},
        )


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        (
            "HTTPS://EXAMPLE.TEST:443/org/repo.git",
            "https://example.test/org/repo.git",
        ),
        (
            "git@EXAMPLE.TEST:org/repo.git",
            "ssh://git@example.test/org/repo.git",
        ),
        (
            "ssh://git@EXAMPLE.TEST:22/org/repo.git",
            "ssh://git@example.test/org/repo.git",
        ),
    ],
)
def test_repository_identity_is_canonical_and_contains_no_credential(
    source: str, canonical: str
) -> None:
    location = RepositoryLocation.parse(
        source,
        allowed_hosts={"example.test"},
        allowed_schemes={"https", "ssh"},
    )

    assert location.canonical_url == canonical
    assert location.host == "example.test"
    assert "credential" not in location.to_dict()


def test_source_identity_serializes_only_an_opaque_secret_reference() -> None:
    location = RepositoryLocation.parse(
        "https://example.test/org/private.git",
        credential=SecretRef(provider="provider_secret", name="PRIVATE_GIT_TOKEN"),
        allowed_hosts={"example.test"},
        allowed_schemes={"https"},
    )

    serialized = location.to_dict()
    assert serialized["url"] == "https://example.test/org/private.git"
    assert serialized["credential"] == {
        "provider": "provider_secret",
        "name": "PRIVATE_GIT_TOKEN",
    }
    assert "value" not in str(serialized).lower()

