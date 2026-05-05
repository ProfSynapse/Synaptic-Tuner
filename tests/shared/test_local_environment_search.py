from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from shared.environments.fixture_parser import EnvironmentFixture
from shared.environments.local_runtime import LocalEnvironmentRuntime


@pytest.fixture()
def workspace_tempdir(monkeypatch):
    temp_root = Path(__file__).resolve().parents[2] / "tmp" / f"local_search_test_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SYNTHCHAT_ENV_TMPDIR", str(temp_root))
    try:
        yield
    finally:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)


def test_local_search_falls_back_to_keyword_grep_when_phrase_order_differs(workspace_tempdir):
    runtime = LocalEnvironmentRuntime()
    runtime.setup(
        EnvironmentFixture(
            directories=[],
            files={
                "Projects/Atlas/Incidents/auth-callback-loop.md": (
                    "Incident: OAuth callback loop caused repeated redirects.\n"
                    "Root cause: redirect URI mismatch."
                ),
                "Projects/Phoenix/Incidents/oauth-callback-loop.md": (
                    "Phoenix incident belongs to another workspace."
                ),
            },
        )
    )

    try:
        assert runtime.search(
            "OAuth callback loop incident",
            path="Projects/Atlas/Incidents",
        ) == ["Projects/Atlas/Incidents/auth-callback-loop.md"]
    finally:
        runtime.teardown()


def test_local_search_ignores_common_stopwords_for_keyword_matching(workspace_tempdir):
    runtime = LocalEnvironmentRuntime()
    runtime.setup(
        EnvironmentFixture(
            directories=[],
            files={
                "Projects/Atlas/Launch/checklists/customer-comms.md": (
                    "Customer comms approved for launch."
                )
            },
        )
    )

    try:
        assert runtime.search(
            "customer comms are approved",
            path="Projects/Atlas/Launch/checklists",
        ) == ["Projects/Atlas/Launch/checklists/customer-comms.md"]
    finally:
        runtime.teardown()
