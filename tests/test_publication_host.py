"""Actual public native verification and local publication with CPU fixtures.

Provider and ML observations are scripted. This is not live training evidence.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synaptic-tuner"))
spec = importlib.util.spec_from_file_location(
    "publication_host", ROOT / "publish_artifacts.py"
)
consumer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)

from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tests.execution.providers.docker_provider_v1.conftest import profile  # noqa: F401
from tests.execution.providers.docker_provider_v1.test_local_source import material  # noqa: F401
from tests.training.test_docker_host_composition import (
    host_fixture,  # noqa: F401
    start,
)
from tests.training.test_local_run_artifact_publication import complete
from tuner.execution.coordinator_v1.publication import (
    PublicationCodeV1,
    PublicationErrorV1,
)


class Secrets:
    def resolve(self, ref):
        assert ref.provider == "env" and ref.name == "PUBLICATION_KEY"
        return "fixture-publication-key-" + "a" * 40


def configuration(root, run):
    return {
        "schema_version": "synaptic-local-publication-host/v1",
        "consumer": {
            "path": "/unused/consumer.json",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "run": run.to_dict(),
        "root": str(root),
        "destination_ref": "local/native",
        "authority_ref": "publication-host",
        "key_ref": "publication-key",
        "secret": {"provider": "env", "name": "PUBLICATION_KEY"},
        "maximum_artifact_bytes": 4 << 20,
        "maximum_total_bytes": 8 << 20,
        "seconds": 60,
    }


def test_verified_native_transfer_then_cold_lookup_without_streams(
    host_fixture,  # noqa: F811 -- pytest fixture imported for discovery
    monkeypatch,
):
    compose, kwargs, commands, state, roles, grants, root = host_fixture
    database = root / "workflow.sqlite3"
    with SQLiteDurableStoreV1(database) as records, compose(records) as host:
        run, _values = complete(host, kwargs, state, roles, monkeypatch)
        config = configuration(root / "published", run)
        first = consumer.local_publication(
            config,
            "publish",
            runs=host.api.runs,
            secrets=Secrets(),
            clock=host.composition.authority.clock.now,
        )
        assert first["verified"] and first["publication"]["state"] == "verified"
        assert len(first["publication"]["artifacts"]) == 5
        with pytest.raises(FileExistsError):
            consumer.local_publication(
                config,
                "publish",
                runs=host.api.runs,
                secrets=Secrets(),
                clock=host.composition.authority.clock.now,
            )
    before = {
        p.relative_to(root / "published"): (
            p.stat().st_size,
            p.stat().st_mtime_ns,
            p.read_bytes(),
        )
        for directory in ("destination", "spool")
        for p in (root / "published" / directory).rglob("*")
        if p.is_file()
    }
    calls = len(commands)

    def forbidden(*args, **kwargs):
        pytest.fail("cold lookup attempted provider, native stream or grant effect")

    kwargs["cli_configuration"]["runner"] = forbidden
    monkeypatch.setattr(grants, "bind", forbidden)
    monkeypatch.setattr(RunsAPI, "artifacts", forbidden)
    with (
        SQLiteDurableStoreV1(database, create=False) as records,
        compose(records, cold=True) as host,
    ):
        cold = consumer.local_publication(
            config,
            "lookup",
            runs=host.api.runs,
            secrets=Secrets(),
            clock=host.composition.authority.clock.now,
        )
        assert cold == first
        altered = dict(config, key_ref="different-key")
        with pytest.raises(ValueError, match="original publication binding"):
            consumer.local_publication(
                altered,
                "lookup",
                runs=host.api.runs,
                secrets=Secrets(),
                clock=host.composition.authority.clock.now,
            )
    after = {
        p.relative_to(root / "published"): (
            p.stat().st_size,
            p.stat().st_mtime_ns,
            p.read_bytes(),
        )
        for directory in ("destination", "spool")
        for p in (root / "published" / directory).rglob("*")
        if p.is_file()
    }
    assert before == after and len(commands) == calls
    assert {p.name for p in (root / "published/spool").iterdir()}


def test_missing_cold_publication_refuses_without_setup(tmp_path):
    config = configuration(
        tmp_path / "missing", consumer.TrainingRunRef("run/fixture", "project/fixture")
    )
    with pytest.raises(FileNotFoundError):
        consumer.local_publication(
            config,
            "lookup",
            runs=None,
            secrets=Secrets(),
            clock=lambda: "2026-10-14T00:00:00Z",
        )
    assert not (tmp_path / "missing").exists()


def test_unverified_native_run_cannot_publish(host_fixture):  # noqa: F811
    compose, kwargs, _, _, _, _, root = host_fixture
    with (
        SQLiteDurableStoreV1(root / "workflow.sqlite3") as records,
        compose(records) as host,
    ):
        _, _, _, run = start(host, kwargs)
        config = configuration(root / "unverified", run)
        with pytest.raises(PublicationErrorV1) as error:
            consumer.local_publication(
                config,
                "publish",
                runs=host.api.runs,
                secrets=Secrets(),
                clock=host.composition.authority.clock.now,
            )
        assert error.value.code is PublicationCodeV1.SOURCE_INVALID
        assert not list((root / "unverified/destination").iterdir())
        assert not list((root / "unverified/spool").iterdir())
