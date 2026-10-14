"""Actual killed host and fresh readers; disk SDK/ML/image boundaries are explicit."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import sqlite3
import subprocess
import sys

import pytest

sdk = pytest.importorskip("huggingface_hub")
pytestmark = pytest.mark.skipif(
    sdk.__version__ != "1.33.0" or os.name != "posix",
    reason="pinned HF1.33/POSIX process proof",
)

from tests.execution.providers import test_hf_jobs_public_family as family
from tests.execution.providers import test_hf_jobs_public_lifecycle as lifecycle
from tuner.execution.providers.hf_jobs_provider_v1 import retained_reader
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference import compose_reference_authority
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunListRequest
from huggingface_hub import BucketFile

_SOURCE = Path(retained_reader.__file__).resolve().parents[4]
_TESTS = Path(__file__).resolve().parents[3]
_BOOTSTRAP = """
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from tests.execution.providers.test_hf_jobs_process_readback import _child
_child(sys.argv[3], Path(sys.argv[4]), Path(sys.argv[2]))
"""


def _write(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert len(raw) <= 262144
    with os.fdopen(
        os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
    ) as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def _census(root):
    with sqlite3.connect(
        (root / "workflow.sqlite3").as_uri() + "?mode=ro", uri=True
    ) as db:
        rows = [
            list(db.execute(query))
            for query in (
                "SELECT partition,key,revision,canonical FROM records ORDER BY partition,key",
                "SELECT partition,stream_key,sequence,canonical FROM streams ORDER BY partition,stream_key,sequence",
            )
        ]
    digest = hashlib.sha256()
    for table in rows:
        for row in table:
            for value in row:
                raw = value if type(value) is bytes else json.dumps(value).encode()
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
    return digest.hexdigest()


def _producer(root):
    patch = pytest.MonkeyPatch()
    source = root / "original-inputs"
    source.mkdir(mode=0o700)
    value = family.material.__wrapped__(source)
    store = SQLiteDurableStoreV1(root / "workflow.sqlite3")
    api, composition, ports = lifecycle.host(value, records=store, streams=store)
    run, plan, _ = lifecycle.start(api, value)
    job = root / "provider-execution"
    job.mkdir(mode=0o700)
    # Real bootstrap/source worker, with the previously accepted explicit
    # runtime/model/trainer boundary doubles. Completion is emitted by worker.
    family.test_actual_bootstrap_worker_with_owned_files_and_injected_ml_effects(
        value, job, patch, None
    )
    output = job / "output"
    completion = json.loads((output / "completion.json").read_bytes())
    rows = {}
    for member in completion["artifacts"]:
        raw = (output / member["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == member["sha256"]
        rows[value.preparation.selection.run_prefix + "/" + member["path"]] = {
            "size": len(raw),
            "sha256": member["sha256"],
            "base64": base64.b64encode(raw).decode(),
        }
    raw = (output / "completion.json").read_bytes()
    rows[value.preparation.selection.run_prefix + "/completion.json"] = {
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode(),
    }
    # Native admission needs the remote input member's listing identity, not
    # its bytes. No original dataset bytes survive into the transport fixture.
    identity = value.preparation.input_identity
    rows[
        value.preparation.selection.run_prefix
        + "/input/"
        + identity.content_digest
        + "/private-dataset.jsonl"
    ] = {"size": identity.size_bytes, "sha256": identity.content_digest}
    submits = [item[1] for item in value.provider.calls if item[0] == "submit"]
    assert len(submits) == 1
    request = dict(submits[0])
    request["volumes"] = [item.to_dict() for item in request["volumes"]]
    _write(
        root / "provider.json",
        {"request": request, "rows": rows, "status": "SCHEDULING"},
    )
    _write(
        root / "keys.json",
        {
            "authority": ports.secrets.resolve(SecretRef("env", "AUTHORITY")),
            "reader_base64": base64.b64encode(b"r" * 32).decode(),
            "epoch": ports.clock.now_epoch(),
        },
    )
    _write(root / "run.json", run.to_dict())
    print(
        json.dumps(
            {
                "ready": True,
                "pid": os.getpid(),
                "run": run.to_dict(),
                "census": _census(root),
            }
        ),
        flush=True,
    )
    signal.alarm(60)
    signal.pause()
    raise AssertionError("parent must kill producer")


def _consumer(root, role):
    state = json.loads((root / "provider.json").read_bytes())
    keys = json.loads((root / "keys.json").read_bytes())
    run = TrainingRunRef.from_dict(json.loads((root / "run.json").read_bytes()))
    events = []

    class Secrets:
        def resolve(self, ref):
            return (
                keys["authority"]
                if ref.name == "AUTHORITY"
                else family.Secrets().resolve(ref)
            )

    class Clock:
        def now_epoch(self):
            return keys["epoch"] + 60

        def now(self):
            from datetime import datetime, timezone

            return (
                datetime.fromtimestamp(self.now_epoch(), timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

    class Provider:
        endpoint = "https://huggingface.co"

        def inspect_job(self, *, job_id, namespace, token):
            assert job_id == "fixture-job-1" and namespace == "fixture-owner"
            events.append("inspect")
            request = dict(state["request"])
            if role == "mismatch":
                request["labels"] = {**request["labels"], "synaptic-material": "f" * 64}
            return family._job(request, state["status"])

        def list_bucket_tree(self, bucket_id, *, prefix, recursive, token):
            events.append("list")
            return [
                BucketFile(
                    type="file", path=name, size=row["size"], xetHash=row["sha256"]
                )
                for name, row in state["rows"].items()
                if name.startswith(prefix + "/")
            ]

        def download_bucket_files(
            self, bucket_id, files, *, token, raise_on_missing_files
        ):
            events.append("download")
            assert raise_on_missing_files
            for member, destination in files:
                destination.write_bytes(
                    base64.b64decode(
                        state["rows"][member.path]["base64"], validate=True
                    )
                )

        def forbidden(self, *args, **kwargs):
            raise AssertionError("no provider mutation on cold read")

        run_job = batch_bucket_files = cancel_job = forbidden

    secrets, clock = Secrets(), Clock()
    authority = compose_reference_authority(
        clock=clock, secrets=secrets, authority_secret=SecretRef("env", "AUTHORITY")
    )
    scratch = root / ("scratch-" + role)
    scratch.mkdir(mode=0o700)
    # This is deliberately the first SQLite open in a fresh process.
    with SQLiteDurableStoreV1(root / "workflow.sqlite3", create=False) as store:
        patch = pytest.MonkeyPatch()
        from tuner.execution.providers.hf_jobs_provider_v1.preparation import (
            HFJobsPreparation,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("cold reads cannot prepare or capture")

        patch.setattr(HFJobsPreparation, "__init__", forbidden)
        patch.setattr(retained_reader, "_capture", forbidden)
        view = retained_reader.open_retained_hf_runs(
            run=run,
            records=store,
            streams=store,
            authority=authority,
            secrets=secrets,
            sdk=sdk,
            api=Provider(),
            scratch_root=scratch,
            reader_authority_ref="fixture-reader",
            reader_key_ref="fixture-reader-key",
            reader_key=base64.b64decode(keys["reader_base64"], validate=True),
        )
        before = view.runs.show(run).state.value
        if role == "mismatch":
            from synaptic_tuner.api.v1.runs_facade import RunOperationError

            with pytest.raises(RunOperationError) as failure:
                view.runs.outcome(run)
            assert failure.value.code.value == "provider_read_invalid"
            assert view.runs.show(run).state.value == before == "queued"
            assert not view.runs.show(run).artifacts
        elif role == "pending":
            assert view.runs.outcome(run).state.value == before == "queued"
        elif role == "verified":
            assert before == "succeeded"
            assert len(view.runs.show(run).artifacts) == 5
            assert (
                view.runs.list(RunListRequest(run.project_ref)).outcomes[0].run == run
            )
            assert not events
        else:
            assert view.runs.outcome(run).state.value == "succeeded"
            assert view.runs.verify(run).verified
            for artifact in view.runs.show(run).artifacts:
                raw = b"".join(
                    view.runs.artifacts(
                        RunArtifactRequest(run, artifact.role, artifact.size_bytes)
                    ).iter_bytes()
                )
                assert hashlib.sha256(raw).hexdigest() == artifact.sha256
    return {
        "pid": os.getpid(),
        "before": before,
        "census": _census(root),
        "events": events,
    }


def _child(role, root, source):
    assert Path(retained_reader.__file__).resolve().parents[4] == source
    if _TESTS != source:
        assert Path(retained_reader.__file__).resolve().is_relative_to(Path(sys.prefix))
        assert not Path(retained_reader.__file__).resolve().is_relative_to(_TESTS)
    assert not any(
        os.environ.get(name)
        for name in ("HF_TOKEN", "HF_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    )
    if role == "producer":
        _producer(root)
    else:
        print(json.dumps(_consumer(root, role)), flush=True)


def _command(root, role):
    return [
        sys.executable,
        "-I",
        "-B",
        "-c",
        _BOOTSTRAP,
        str(_TESTS),
        str(_SOURCE),
        role,
        str(root),
    ]


def test_killed_hf_producer_has_source_free_fresh_known_job_readers(tmp_path):
    root = tmp_path / "proof"
    root.mkdir(mode=0o700)
    scratch = root / "tmp"
    scratch.mkdir(mode=0o700)
    environment = {"PATH": os.defpath, "TMPDIR": str(scratch)}
    producer = subprocess.Popen(
        _command(root, "producer"),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(producer.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=60)
            payload = os.read(producer.stdout.fileno(), 4096)
            if not payload:
                producer.wait(timeout=5)
                raise AssertionError(producer.stderr.read(16384).decode())
            ready = json.loads(payload)
        assert ready["pid"] == producer.pid
        producer.kill()
        assert producer.wait(timeout=5) == -signal.SIGKILL
    finally:
        if producer.poll() is None:
            producer.kill()
            producer.wait(timeout=5)
        producer.stdout.close()
        producer.stderr.close()
    shutil.rmtree(root / "original-inputs")
    shutil.rmtree(root / "provider-execution")
    observed = []
    for role in ("pending", "mismatch", "completed", "verified"):
        if role == "completed":
            state = json.loads((root / "provider.json").read_bytes())
            state["status"] = "COMPLETED"
            temporary = root / "completed.json"
            _write(temporary, state)
            temporary.replace(root / "provider.json")
        result = subprocess.run(
            _command(root, role), env=environment, capture_output=True, timeout=60
        )
        assert result.returncode == 0, result.stderr.decode()
        observed.append(json.loads(result.stdout))
    first, rejected, second, third = observed
    assert rejected["census"] == ready["census"]
    assert len({producer.pid, *(item["pid"] for item in observed)}) == 5
    assert first["census"] == ready["census"]
    assert second["census"] != ready["census"] and second["before"] == "queued"
    assert third["census"] == second["census"] and third["events"] == []
