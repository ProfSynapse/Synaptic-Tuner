"""Real SQLite/public/guardian host controls with a fixed CPU worker only."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import multiprocessing
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from examples.training_qualification import native_host, native_host_lifecycle as lifecycle
from tests.cli import test_native_host as host_fixtures
from tuner.runtime import native_lifecycle_protocol, native_process_guardian
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.training.native_mlx_compilation import NativeMLXProfile, compile_native_sft
from synaptic_tuner.api.v1.native_mlx import NativeLifecycleLaunchV1, NativeLifecyclePolicyV1
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.results import TrainingRunRef, TrainingRunState
from synaptic_tuner.api.v1.runs_facade import RunOutcome, RunLogEntry, RunLogLevel

host_case = host_fixtures.host_case
publication = host_fixtures.publication

@pytest.fixture
def lifecycle_case(host_case, publication):
    config, clock, remote, calls = host_case
    root = Path(config["profile_file"]).parent
    runtime = Path(config["engine_root"]) / "tuner/runtime"
    runtime.mkdir(parents=True, mode=0o700)
    for path in (runtime.parent.parent, runtime.parent): path.chmod(0o700)
    guardian, codec = runtime / "native_process_guardian.py", runtime / "native_lifecycle_protocol.py"
    guardian.write_bytes(Path(native_process_guardian.__file__).read_bytes())
    codec.write_bytes(Path(native_lifecycle_protocol.__file__).read_bytes())
    executable = root / "python"
    executable.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    executable.chmod(0o700)
    launch = NativeLifecycleLaunchV1(NativeLifecyclePolicyV1(32, 65536),
                                    hashlib.sha256(codec.read_bytes()).hexdigest())

    def select(scenario):
        config.update(schema="synaptic-native-host-config/v2", lifecycle={
            "launch": launch.to_dict(), "scenario": scenario,
            "trigger_step": None if scenario == "complete" else 1,
            "poll_interval_ms": 10, "host_timeout_seconds": 16, "recovery_timeout_seconds": 5})
        worker = runtime / "native_mlx_child.py"
        # This substitutes ML with immutable synthetic fixture artifacts only.
        worker.write_text("import json,os,sys,time\nfrom pathlib import Path\n"
            "request=json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes())\n"
            "fd=int(sys.argv[sys.argv.index('--progress-fd')+1])\n"
            "event={'kind':'progress','phase':'training','step':1,'metrics':{'loss':1.0}}\n"
            "os.write(fd,json.dumps(event,sort_keys=True,separators=(',',':')).encode()+b'\\n')\n"
            + ("time.sleep(2)\n"
               f"source=Path({str(publication['output'])!r})\n"
               "for p in source.iterdir(): (Path(request['output'])/p.name).write_bytes(p.read_bytes())\n"
               if scenario == "complete" else "while True: time.sleep(.01)\n"))
        for path in (worker, guardian, codec): path.chmod(0o600)
        capture = strict_json(Path(config["runtime_capture_file"]).read_bytes())
        capture["runtime"]["executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
        capture["trees"][0]["members"] = [{"path": "runtime/" + path.name,
            "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (worker, guardian, codec)]
        Path(config["runtime_capture_file"]).write_bytes(canonical(capture))
        profile = NativeMLXProfile.from_dict(strict_json(Path(config["profile_file"]).read_bytes()))
        profile = replace(profile, runtime_digest=hashlib.sha256(canonical(capture)).hexdigest())
        Path(config["profile_file"]).write_bytes(canonical(asdict(profile)))
        training = TrainingInputV1.from_dict(strict_json(Path(config["training_input_file"]).read_bytes()))
        identity = PreparedTrainingInputIdentity.from_dict(strict_json(Path(config["prepared_identity_file"]).read_bytes()))
        projection = compile_native_sft(training, identity, profile)
        binding_path = publication["output"] / "input_binding.json"
        binding = strict_json(binding_path.read_bytes())
        binding.update(workload_digest=projection.workload.fingerprint, runtime_digest=profile.runtime_digest)
        binding_path.write_bytes(canonical(binding))
        result_path = publication["output"] / "native_result.json"
        result = strict_json(result_path.read_bytes())
        result.update(input_binding_digest=hashlib.sha256(canonical(binding)).hexdigest(),
                      workload_digest=projection.workload.fingerprint, runtime_digest=profile.runtime_digest)
        result_path.write_bytes(canonical(result))
        (publication["output"] / "workload.json").write_bytes(projection.workload.canonical_bytes)
        unsigned = {**native_host.inspect(config)[0]["admission_candidate"],
                    "expires_at": "2026-10-20T12:30:00Z", "authority_ref": "root-review", "key_ref": "root-key"}
        subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(root / "root-private.pem"),
                        "-out", config["admission_signature_file"]], input=canonical(unsigned),
                       check=True, capture_output=True)
        signature = Path(config["admission_signature_file"]).read_bytes()
        Path(config["admission_file"]).write_bytes(canonical({**unsigned, "tag": hashlib.sha256(signature).hexdigest()}))
        return config
    return select, clock, calls


def operate(config, action):
    return native_host.operate(config, action=action,
                               trusted_root_key_sha256=config["root_public_key_sha256"])


def test_config_first_public_completion_and_retained_logs(lifecycle_case):
    select, clock, calls = lifecycle_case
    config = select("complete")
    report = operate(config, "qualify")
    assert report["outcome"]["state"] == "succeeded" and report["trigger"] is None
    assert len(report["entries"]) == 1 and calls == []
    assert report["pending_progress"][0]["sequence"] == 0
    assert report["pending_progress"][0]["before"]["state"] == "running"
    assert report["pending_progress"][0]["after"]["state"] == "running"
    with pytest.raises(FileExistsError): operate(config, "qualify")
    clock["now"] = "2026-10-20T14:00:00Z"
    assert operate(config, "verify")["verified"]
    assert operate(config, "reverify")["verified"]
    assert operate(config, "logs")["entries"] == report["entries"]
    assert (Path(config["state_root"]) / "lifecycle-observation.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="workflow"):
        operate(config, "start")


def test_config_first_public_cancellation_is_confirmed(lifecycle_case):
    select, _, calls = lifecycle_case
    config = select("cancel")
    report = operate(config, "qualify")
    assert report["cancel_ack"]["state"] == "cancel_requested"
    assert report["outcome"]["state"] == "cancelled" and report["trigger"] is not None
    assert operate(config, "observe")["state"] == "cancelled" and calls == []


@pytest.mark.skipif(os.name != "posix", reason="native qualification uses POSIX owner pipes")
def test_actual_caller_exit_then_new_process_cold_reconciliation(lifecycle_case):
    select, clock, calls = lifecycle_case
    config = select("caller-loss")
    child = multiprocessing.get_context("fork").Process(target=operate, args=(config, "qualify"))
    child.start()
    child.join(10)
    try:
        assert child.exitcode == lifecycle.CALLER_LOSS_EXIT
    finally:
        if child.is_alive(): child.kill(); child.join(2)
    trigger = strict_json((Path(config["state_root"]) / "caller-loss-trigger.json").read_bytes())
    assert trigger["trigger"]["observed_outcome"]["state"] == "running"
    clock["now"] = "2026-10-20T14:00:00Z"
    report = operate(config, "recover")
    assert report["outcome"]["state"] == "failed" and calls == []
    assert report["entries"] == trigger["entries"]
    assert operate(config, "observe")["state"] == "failed"
    with pytest.raises(FileExistsError): operate(config, "recover")


@pytest.mark.parametrize("mutation", ["scenario", "trigger", "budget", "poll", "timeout", "unknown",
                                     "report_records", "report_scope"])
def test_host_policy_tamper_rejects_before_effects(lifecycle_case, mutation):
    select, _, calls = lifecycle_case
    config = deepcopy(select("cancel"))
    if mutation == "scenario": config["lifecycle"]["scenario"] = "caller-loss"
    elif mutation == "trigger": config["lifecycle"]["trigger_step"] = 3
    elif mutation == "budget": config["lifecycle"]["launch"]["policy"]["maximum_progress_records"] = 1
    elif mutation == "poll": config["lifecycle"]["poll_interval_ms"] = True
    elif mutation == "timeout": config["lifecycle"]["host_timeout_seconds"] = 1
    elif mutation == "report_records": config["lifecycle"]["launch"]["policy"]["maximum_progress_records"] = 65
    elif mutation == "report_scope": config["project_ref"] = "a" * 129
    else: config["lifecycle"]["extra"] = "unsupported"
    with pytest.raises((ValueError, TypeError)):
        operate(config, "qualify")
    assert not Path(config["state_root"]).exists() and calls == []


@pytest.mark.parametrize("mutation", ["signature", "missing_pin", "wrong_pin", "codec"])
def test_lifecycle_requires_external_signature_and_independent_pin(lifecycle_case, mutation):
    select, _, calls = lifecycle_case
    config = deepcopy(select("complete"))
    pin = config["root_public_key_sha256"]
    if mutation == "signature": Path(config["admission_signature_file"]).write_bytes(b"invalid")
    elif mutation == "missing_pin": pin = None
    elif mutation == "wrong_pin": pin = "f" * 64
    else: config["lifecycle"]["launch"]["protocol_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        native_host.operate(config, action="qualify", trusted_root_key_sha256=pin)
    assert not Path(config["state_root"]).exists() and calls == []


def test_root_lifecycle_projection_cannot_authenticate_different_grant(lifecycle_case):
    select, _, _ = lifecycle_case
    config = select("complete")
    _, selected = native_host.inspect(config, require_admission=True,
                                      trusted_root_key_sha256=config["root_public_key_sha256"])
    planning = selected[0]
    assert planning.lifecycle_allowed()
    authority, grant = planning._lifecycle_authority, planning._lifecycle_grant
    assert not authority.authenticate(replace(grant, attempt="different-attempt"))
    assert not authority.authenticate(replace(grant, tag="f" * 64))
    assert not authority.authenticate(object())
    Path(config["admission_signature_file"]).write_bytes(b"invalid")
    assert not planning.lifecycle_allowed()


def test_completion_during_log_read_is_not_live_progress_or_cancel(lifecycle_case):
    select, _, _ = lifecycle_case
    config = select("cancel")
    run = TrainingRunRef("run", "project")
    outcomes = iter(RunOutcome("synaptic-run-outcome/v1", run, state) for state in (
        TrainingRunState.RUNNING, TrainingRunState.SUCCEEDED))
    message = canonical({"kind": "progress", "phase": "training", "step": 1,
                         "metrics": {"loss": 1.0}}).decode()
    entry = RunLogEntry(0, "2026-10-20T12:00:00Z", RunLogLevel.INFO, "native.progress", message, len(message))
    class Runs:
        def outcome(self, run): return next(outcomes)
        def logs(self, request): return SimpleNamespace(entries=(entry,), truncated=False)
        def cancel(self, *args): pytest.fail("terminal race must not request cancellation")
    writes = []
    report = lifecycle.qualify(config, SimpleNamespace(runs=Runs()),
        SimpleNamespace(accepted=True, run=run), Path(config["state_root"]),
        lambda *args: writes.append(args))
    assert report["pending_progress"] == [] and report["cancel_ack"] is None
    assert report["trigger"] is None and report["outcome"]["state"] == "succeeded"
    assert len(writes) == 1
