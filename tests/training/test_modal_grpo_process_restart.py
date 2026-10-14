"""Killed GRPO host and fresh readers over real native outputs and disk SDK bytes.

The provider/image/ML boundaries are explicit CPU fixtures. No GPU, live Modal,
provider cancellation, or parameter-quality qualification is claimed.
"""

from __future__ import annotations

import base64
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from tests.training import test_modal_process_restart as process
from tests.training import test_modal_grpo_standalone_runner as joined
from tuner.training import modal_retained_reader as reader
from tuner.training import modal_standalone_runner as runner
from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1

_SOURCE = Path(reader.__file__).resolve().parents[2]
_TEST_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP = """
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from tests.training.test_modal_grpo_process_restart import _child
_child(sys.argv[3], Path(sys.argv[4]), Path(sys.argv[2]))
"""


def _cancel_descriptor_fixture(patch):
    from dataclasses import replace
    from tuner.training import modal_host_composition as composition

    original = composition._modal_packaged_descriptor

    def descriptor(**kwargs):
        value = original(**kwargs)
        return replace(
            value, capabilities=replace(value.capabilities, cancel=True, reconcile=True)
        )

    patch.setattr(composition, "_modal_packaged_descriptor", descriptor)
    patch.setattr(reader, "_modal_packaged_descriptor", descriptor)


def _producer(root):
    patch = pytest.MonkeyPatch()
    uncertain_cancel = (root / "uncertain-cancel").exists()
    captured = []
    if uncertain_cancel:
        _cancel_descriptor_fixture(patch)
        from tuner.training import modal_host_composition as composition

        original_generic = composition.compose_reference_host
        patch.setattr(
            composition,
            "compose_reference_host",
            lambda **kwargs: original_generic(
                **kwargs, allow_current_cancel_authorization=True
            ),
        )
        original_compose = runner.compose_modal_packaged_reference_host

        def compose(**kwargs):
            value = original_compose(**kwargs)
            captured.append(value)
            return value

        patch.setattr(runner, "compose_modal_packaged_reference_host", compose)
    host = root / "host"
    host.mkdir(mode=0o700)
    # The parent owns fixture storage even when this producer is abruptly killed.
    original_temporary = tempfile.TemporaryDirectory

    def owned_temporary(**kwargs):
        kwargs["dir"] = root
        return original_temporary(**kwargs)

    patch.setattr(joined, "TemporaryDirectory", owned_temporary)
    finalizers = []
    plan, context, events, _, cpu, trainer, _, _ = joined._joined(
        host, patch, SimpleNamespace(addfinalizer=finalizers.append), remove_source=True
    )

    def ready(self, binding, call_ref, **kwargs):
        from synaptic_tuner.api.v1.results import TrainingRunRef
        from tuner.execution.providers.modal.contracts import operation_path

        run = TrainingRunRef(
            binding.execution_binding.run_ref, binding.command.preparation.project_ref
        )
        retained = reader.load_retained_modal_run(
            host / "state/synaptic-training" / run.run_id, expected_run=run
        )
        sdk = sys.modules["modal"]
        if uncertain_cancel:

            def lost_cancel(call, *, terminate_containers):
                assert terminate_containers is True and call.object_id == call_ref
                raise RuntimeError("controlled cancellation reply loss")

            patch.setattr(sdk.FunctionCall, "cancel", lost_cancel, raising=False)
            assert (
                captured[0].api.runs.cancel(run, "stop").state.value
                == "reconcile_required"
            )
            workflow = captured[0].composition.stores.workflow_store.get(run)
            assert workflow.bound_cancellation is None
            expected_cancel = workflow.cancel.to_dict()
        else:
            expected_cancel = None
        actual_result = sdk.FunctionCall.from_id(call_ref, client=None).get(timeout=0)
        assert actual_result["status_code"] == "completed"
        facts = binding.provider_facts
        transport = {}
        artifact_hashes = {}
        effect = binding.command.operation.effect.effect_id
        for volume_id, name in retained.inputs["volume_names"]:
            if volume_id not in (facts.control_volume_id, facts.artifact_volume_id):
                continue
            volume = sdk.Volume(name)
            leaves = {}
            for entry in volume.iterdir(operation_path(effect), recursive=True):
                raw = b"".join(volume.read_file(entry.path))
                assert len(raw) <= 4 * 1024 * 1024
                leaves[entry.path] = base64.b64encode(raw).decode()
            transport[volume_id] = leaves
        completion_raw = base64.b64decode(
            transport[facts.control_volume_id][
                operation_path(effect, "evidence") + "/packaged-completion.json"
            ]
        )
        completion = json.loads(completion_raw)
        for member in completion["members"]:
            raw = base64.b64decode(transport[facts.artifact_volume_id][member["path"]])
            assert hashlib.sha256(raw).hexdigest() == member["sha256"]
            artifact_hashes[member["role"]] = member["sha256"]
        assert len(artifact_hashes) == 5 and len(cpu) == len(trainer) == 1
        process._write(
            root / "provider.json",
            {
                "phase": "pending",
                "expected_state": "reconcile_required"
                if uncertain_cancel
                else "queued",
                "expected_cancel": expected_cancel,
                "facts": facts.to_dict(),
                "client_binding": retained.inputs["client_binding"],
                "volume_names": retained.inputs["volume_names"],
                "run": run.to_dict(),
                "call_ref": call_ref,
                "call_result": actual_result,
                "volumes": transport,
                "artifact_hashes": artifact_hashes,
                "maximum_artifact_bytes": 4 * 1024 * 1024,
            },
        )
        print(
            json.dumps(
                {
                    "ready": True,
                    "pid": os.getpid(),
                    "run": run.to_dict(),
                    "effects": events,
                    "census_digest": process._census_digest(root, run.to_dict()),
                }
            ),
            file=sys.__stdout__,
            flush=True,
        )
        signal.alarm(60)
        signal.pause()
        raise AssertionError("producer must be killed")

    patch.setattr(ModalPackagedCoordinatorReaderV1, "_poll_packaged_call", ready)
    with redirect_stdout(sys.stderr):
        runner.run_modal_standalone_job(
            plan=plan,
            context=context,
            modal_profile="fixture",
            modal_environment="main",
        )


def _child(role, root, expected_source):
    assert not any(os.environ.get(name) for name in process._CREDENTIAL_NAMES)
    assert Path(reader.__file__).resolve().parents[2] == expected_source
    assert Path(runner.__file__).resolve().parents[2] == expected_source
    if _TEST_ROOT != expected_source:
        assert Path(reader.__file__).resolve().is_relative_to(Path(sys.prefix))
        assert not Path(reader.__file__).resolve().is_relative_to(_TEST_ROOT)
    if role == "producer":
        _producer(root)
    else:
        if (root / "uncertain-cancel").exists():
            _cancel_descriptor_fixture(pytest.MonkeyPatch())
        print(json.dumps(process._consumer(root, role)), flush=True)


def _command(root, role):
    return [
        sys.executable,
        "-I",
        "-B",
        "-c",
        _BOOTSTRAP,
        str(_TEST_ROOT),
        str(_SOURCE),
        role,
        str(root),
    ]


def _trusted_proof_root(request):
    # Provider VolumeRootBinding validates every ancestor, so pytest's /tmp
    # tree cannot host these simulated provider mounts. The parent owns this
    # trusted private tree and its cleanup even when the producer is SIGKILLed.
    temporary = tempfile.TemporaryDirectory(
        prefix="modal-grpo-process-", dir=Path.home()
    )
    request.addfinalizer(temporary.cleanup)
    return Path(temporary.name)


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL process proof requires POSIX")
def test_grpo_killed_producer_then_three_fresh_public_readers(
    request, *, uncertain_cancel=False
):
    root = _trusted_proof_root(request) / "grpo-process-proof"
    root.mkdir(mode=0o700)
    if uncertain_cancel:
        (root / "uncertain-cancel").touch(mode=0o600)
    scratch = root / "scratch"
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
            assert selector.select(timeout=60), "producer readiness timeout"
            payload = os.read(producer.stdout.fileno(), 4096)
            if not payload:
                producer.wait(timeout=5)
                raise AssertionError(producer.stderr.read(16384).decode())
            ready = json.loads(payload)
        assert ready["pid"] == producer.pid and ready["ready"]
        assert ready["effects"] == ["local", "bootstrap", "cpu", "training"]
        producer.kill()
        assert producer.wait(timeout=5) == -signal.SIGKILL
    finally:
        if producer.poll() is None:
            producer.kill()
            producer.wait(timeout=5)
        producer.stdout.close()
        producer.stderr.close()
    # Retain only actual host persistence and disk remote bytes. No original
    # source/profile/config/publication or model fixture may assist readers.
    run_root = root / "host/state/synaptic-training" / ready["run"]["run_id"]
    for path in (root / "host").iterdir():
        if path.name != "state":
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for path in root.glob("modal-grpo-join-*"):
        shutil.rmtree(path)
    for path in run_root.iterdir():
        if path.name not in {
            "authority-keys.json",
            "readback-inputs.json",
            "workflow.sqlite3",
            "workflow.sqlite3-wal",
            "workflow.sqlite3-shm",
            "workflow.sqlite3-journal",
        }:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    observed = []
    for role in ("pending", "completed", "verified"):
        if role == "completed":
            state = process._snapshot(root)
            state["phase"] = "completed"
            # Authentic but cross-bound completion: no success may be adopted.
            from copy import deepcopy
            from synaptic_tuner.api.v1.results import TrainingRunRef
            from tuner.training.modal_host_authority import HMACAuthenticator

            bad = deepcopy(state)
            control = bad["volumes"][
                reader.ModalPackagedRuntimeFactsV1.from_dict(
                    bad["facts"]
                ).control_volume_id
            ]
            member = next(
                name for name in control if name.endswith("/packaged-completion.json")
            )
            document = json.loads(base64.b64decode(control[member]))
            document["execution_binding_digest"] = "f" * 64
            raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
            retained = reader.load_retained_modal_run(
                run_root, expected_run=TrainingRunRef.from_dict(ready["run"])
            )
            key_ref = "modal-runtime-release-qualification-hmac-v1"
            signer = HMACAuthenticator(
                {key_ref: retained.qualification_key},
                allowed_purposes=frozenset({"modal-packaged-completion/v1"}),
            )
            control[member] = base64.b64encode(raw).decode()
            control[member.removesuffix(".json") + ".mac"] = base64.b64encode(
                signer.sign("modal-packaged-completion/v1", raw, key_ref)
            ).decode()
            bad["call_result"]["completion_sha256"] = hashlib.sha256(raw).hexdigest()
            temporary = root / "bad-provider.json"
            process._write(temporary, bad)
            temporary.replace(root / "provider.json")
            rejected = subprocess.run(
                _command(root, "mismatch"),
                env=environment,
                capture_output=True,
                timeout=60,
            )
            assert rejected.returncode == 0, rejected.stderr.decode()
            negative = json.loads(rejected.stdout)
            assert negative["census_digest"] == ready["census_digest"]
            assert negative["verified_artifacts"] == 0
            temporary = root / "completed-provider.json"
            process._write(temporary, state)
            temporary.replace(root / "provider.json")
        result = subprocess.run(
            _command(root, role), env=environment, capture_output=True, timeout=60
        )
        assert result.returncode == 0, result.stderr.decode()
        observed.append(json.loads(result.stdout))
    pending, completed, verified = observed
    assert len({producer.pid, *(item["pid"] for item in observed)}) == 4
    assert (
        pending["state"] == ("reconcile_required" if uncertain_cancel else "queued")
        and pending["census_digest"] == ready["census_digest"]
    )
    assert (
        completed["before"] == ("reconcile_required" if uncertain_cancel else "queued")
        and completed["state"] == "succeeded"
    )
    assert completed["verified_artifacts"] == verified["verified_artifacts"] == 5
    assert completed["census_digest"] != ready["census_digest"]
    assert (
        verified["census_digest"] == completed["census_digest"]
        and not verified["reads"]
    )
    assert process._census_digest(root, ready["run"]) == verified["census_digest"]
