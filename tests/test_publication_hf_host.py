"""Actual CLI config and retained HF read composition; scripted SDK only."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synaptic-tuner"))
spec = importlib.util.spec_from_file_location(
    "hf_publication_host", ROOT / "publish_artifacts.py"
)
consumer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)

from synaptic_tuner.api.v1.runs_facade import RunsAPI
from synaptic_tuner.api.v1.secrets import SecretRef
from tests.examples.test_hf_training_qualification import configured  # noqa: F401
from tests.execution.providers.test_hf_jobs_public_family import (
    Clock,
    Secrets,
    _complete,
    _native_values,
    material,  # noqa: F401
    sdk,
)


def test_cli_retained_hf_native_publication_then_source_free_lookup(
    configured,  # noqa: F811 -- imported pytest fixture
    tmp_path,
    monkeypatch,
    capsys,
):
    config, fixture = configured
    root = tmp_path / "training-state"
    config["state_root"] = str(root)
    config["host"].update(
        grant_ref="fixture-grant",
        allowed_requirements=[
            {
                "operation": "training.start",
                "paid_effect": True,
                "maximum_cost_minor_units": 100,
                "currency": "USD",
            }
        ],
    )
    for name in ("AUTHORITY", "READER", "HOST_TOKEN", "JOB_TOKEN"):
        monkeypatch.setenv(name, Secrets().resolve(SecretRef("env", name)))
    monkeypatch.setenv("PUBLICATION_KEY", "fixture-publication-" + "k" * 40)
    adapter = consumer.hf_host
    result = consumer.launch.launch(
        config, adapter=adapter, adapter_options={"sdk": sdk, "api": fixture.provider}
    )
    retained = consumer.launch.load_config(root / "config.json")
    with adapter._preparation(retained, Clock()) as (prep, _, _):
        values = _native_values(prep)
        fixture.preparation = prep
    _complete(fixture, values)
    options = {"sdk": sdk, "api": fixture.provider}
    assert (
        consumer.launch.known(
            retained, "observe", adapter=adapter, adapter_options=options
        )["state"]
        == "succeeded"
    )
    assert consumer.launch.known(
        retained, "verify", adapter=adapter, adapter_options=options
    )["verified"]
    raw = (root / "config.json").read_bytes()
    publication = {
        "schema_version": "synaptic-local-publication-host/v1",
        "consumer": {
            "path": str(root / "config.json"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
        "run": result["start"]["run"],
        "root": str(tmp_path / "publication"),
        "destination_ref": "local/native",
        "authority_ref": "publication-host",
        "key_ref": "publication-key",
        "secret": {"provider": "env", "name": "PUBLICATION_KEY"},
        "maximum_artifact_bytes": 4 << 20,
        "maximum_total_bytes": 8 << 20,
        "seconds": 60,
    }
    path = tmp_path / "publication.json"
    path.write_text(json.dumps(publication))
    for item in (
        config["request_file"],
        config["hf_jobs"]["material_file"],
        config["hf_jobs"]["prepared_file"],
    ):
        Path(item["path"]).unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("publication attempted provider write or submission")

    fixture.provider.run_job = forbidden
    original_open = adapter.open_host

    def exact_read(config, **kwargs):
        assert kwargs["mode"] == "read"
        return original_open(config, **kwargs, sdk=sdk, api=fixture.provider)

    monkeypatch.setattr(adapter, "open_host", exact_read)
    consumer.main(["publish", "--config", str(path)])
    first = json.loads(capsys.readouterr().out)
    assert first["verified"] and len(first["publication"]["artifacts"]) == 5
    monkeypatch.setattr(RunsAPI, "artifacts", forbidden)
    fixture.provider.list_bucket_tree = forbidden
    fixture.provider.download_bucket_files = forbidden
    fixture.provider.inspect_job = forbidden
    consumer.main(["lookup", "--config", str(path)])
    assert json.loads(capsys.readouterr().out) == first


def test_uncommitted_loaded_consumer_rejects_before_host_open(tmp_path, monkeypatch):
    original = {
        "provider": "hf_jobs",
        "run": {"run_id": "fixture", "project_ref": "project"},
    }
    raw = json.dumps(original).encode()
    monkeypatch.setattr(
        consumer, "load", lambda _: {"consumer": {}, "run": original["run"]}
    )
    monkeypatch.setattr(consumer.launch, "read_committed", lambda *a, **k: raw)
    monkeypatch.setattr(
        consumer.launch,
        "load_config",
        lambda _: dict(original, state_root="/uncommitted"),
    )
    # The commitment path field is needed for the existing validation surface.
    monkeypatch.setattr(
        consumer,
        "load",
        lambda _: {"consumer": {"path": "/fixture"}, "run": original["run"]},
    )
    with pytest.raises(ValueError, match="authenticated bytes"):
        consumer.main(["lookup", "--config", "/fixture"])
