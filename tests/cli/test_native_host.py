"""Host trust boundary tests; no ML package import or Mac execution."""

from dataclasses import asdict, replace
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest

from examples.training_qualification import native_host
from examples.training_qualification.bounded_command import BoundedCommandError, run_bounded_command
from tuner.runtime.native_mlx_material import canonical
from tuner.training.native_mlx_planning import NativeRestrictedAdmissionV1

from tests.runtime import test_native_mlx_publication as publication_fixtures
publication = publication_fixtures.publication
from tests.runtime.test_native_mlx_child import inputs
from tests.runtime.test_native_mlx_family import prepare as prepare_family
from tuner.runtime.native_mlx_material import strict_json
from tuner.runtime.native_process import NativeOwnedProcess
from tuner.training.native_mlx_compilation import compile_native_sft


def test_external_root_signature_is_required_and_binds_exact_admission(tmp_path):
    private = tmp_path / "root-private.pem"
    public = tmp_path / "root-public.pem"
    signature = tmp_path / "admission.sig"
    subprocess.run(["openssl", "genpkey", "-algorithm", "EC", "-pkeyopt",
                    "ec_paramgen_curve:P-256", "-out", str(private)], check=True,
                   capture_output=True)
    subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
                   check=True, capture_output=True)
    candidate = NativeRestrictedAdmissionV1(
        *("a" * 64 for _ in range(9)), "run", "attempt-1", 3,
        "2026-10-20T12:30:00Z", "root-review", "root-key", "0" * 64,
    )
    signed = asdict(candidate)
    signed.pop("tag")
    subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(private),
                    "-out", str(signature)], input=canonical(signed), check=True,
                   capture_output=True)
    proof = replace(candidate, tag=hashlib.sha256(signature.read_bytes()).hexdigest())
    trusted = hashlib.sha256(public.read_bytes()).hexdigest()
    authority = native_host.RootSignatureAuthority(public, trusted, signature)
    assert authority.authenticate(proof)
    assert not authority.authenticate(replace(proof, run_id="another-run"))
    with pytest.raises(ValueError, match="reviewed root public key"):
        native_host.RootSignatureAuthority(public, "f" * 64, signature)
    signature.write_bytes(b"replacement")
    assert not authority.authenticate(proof)


def test_source_record_requires_clean_exact_pushed_ref(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(native_host, "ENGINE_ROOT", root)
    commit = "a" * 40
    config = {"engine_root": str(root), "engine_commit": commit,
              "engine_remote": "https://example.invalid/engine.git",
              "engine_ref": "refs/heads/native-review"}
    observations = {
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("rev-parse", "HEAD"): commit,
        ("remote", "get-url", "origin"): config["engine_remote"],
        ("ls-remote", "--exit-code", "origin", config["engine_ref"]):
            f"{commit}\t{config['engine_ref']}",
    }

    def command(argv, **_):
        return observations[tuple(argv[1:])]

    monkeypatch.setattr(native_host, "_command", command)
    record, digest = native_host.source_record(config)
    assert record["commit"] == commit and digest == hashlib.sha256(canonical(record)).hexdigest()
    observations[("status", "--porcelain", "--untracked-files=all")] = "?? stray.py"
    with pytest.raises(ValueError, match="dirty"):
        native_host.source_record(config)
    observations[("status", "--porcelain", "--untracked-files=all")] = ""
    observations[("ls-remote", "--exit-code", "origin", config["engine_ref"])] = "b" * 40
    with pytest.raises(ValueError, match="pushed source"):
        native_host.source_record(config)


def test_snapshot_candidate_inventories_complete_owned_bytes():
    original_umask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-host-snapshot-", dir=Path.home()) as name:
            root = Path(name)
            (root / "config.json").write_bytes(b'{"num_hidden_layers":2}')
            (root / "model.safetensors").write_bytes(b"candidate-bytes")
            candidate = native_host.snapshot_candidate(root=root, repository="configured/model",
                                                       revision="a" * 40)
            assert tuple(item["path"] for item in candidate["files"]) == (
                "config.json", "model.safetensors",
            )
            manifest = root.parent / (root.name + "-manifest.json")
            try:
                manifest.write_bytes(canonical(candidate))
                native_host._snapshot(manifest)
                (root / "model.safetensors").write_bytes(b"drift")
                with pytest.raises(ValueError, match="snapshot"):
                    native_host._snapshot(manifest)
            finally:
                manifest.unlink(missing_ok=True)
    finally:
        os.umask(original_umask)


@pytest.mark.skipif(os.name != "posix", reason="native host requires POSIX process groups")
@pytest.mark.parametrize("stdout_size,stderr_size", [
    (8192, 0), (0, 8192), (2048, 2049),
])
def test_host_command_output_is_bounded_while_draining(stdout_size, stderr_size):
    code = ("import sys; sys.stdout.buffer.write(b'a' * %d); "
            "sys.stderr.buffer.write(b'b' * %d)" % (stdout_size, stderr_size))
    with pytest.raises(BoundedCommandError, match="bounded_command_failed"):
        run_bounded_command([sys.executable, "-I", "-c", code],
                            environment={"PATH": "/usr/bin:/bin"}, maximum=4096, timeout=3)
    status, output = run_bounded_command(
        [sys.executable, "-I", "-c", "import sys; sys.stdout.write('OK')"],
        maximum=4096, timeout=3,
    )
    assert status == 0 and output == b"OK"


@pytest.mark.skipif(os.name != "posix", reason="native host requires POSIX process groups")
def test_host_command_deadline_kills_owned_session(tmp_path):
    pid_file = tmp_path / "owned-pid"
    code = ("import os,time; from pathlib import Path; "
            f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
            "time.sleep(30)")
    started = time.monotonic()
    with pytest.raises(BoundedCommandError, match="bounded_command_failed"):
        run_bounded_command([sys.executable, "-I", "-c", code], maximum=4096, timeout=.2)
    assert time.monotonic() - started < 3
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(os.name != "posix", reason="native host requires POSIX process groups")
def test_host_command_flood_kills_owned_session(tmp_path):
    pid_file = tmp_path / "flood-pid"
    code = ("import os,sys; from pathlib import Path; "
            f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
            "while True:\n sys.stdout.buffer.write(b'x' * 65536)\n sys.stdout.buffer.flush()")
    with pytest.raises(BoundedCommandError, match="bounded_command_failed"):
        run_bounded_command([sys.executable, "-I", "-c", code], maximum=4096, timeout=3)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(os.name != "posix", reason="native host requires POSIX process groups")
@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_host_command_interruption_reaps_owned_child(monkeypatch, interruption):
    from examples.training_qualification import bounded_command

    children = []
    real_popen = bounded_command.subprocess.Popen
    selector_class = bounded_command.selectors.DefaultSelector

    def popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    class InterruptingSelector(selector_class):
        def select(self, timeout=None):
            raise interruption()

    monkeypatch.setattr(bounded_command.subprocess, "Popen", popen)
    monkeypatch.setattr(bounded_command.selectors, "DefaultSelector", InterruptingSelector)
    try:
        with pytest.raises(interruption):
            run_bounded_command(
                [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                maximum=4096, timeout=3,
            )
        assert len(children) == 1 and children[0].returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(children[0].pid, 0)
    finally:
        # A failed regression must not leave the test's child behind.
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)


@pytest.fixture
def host_case(publication, monkeypatch):
    """Real host composition/store with only Mac metadata and ML child substituted."""
    from tuner.training import native_mlx_family as family_module
    state = prepare_family(publication)
    root = state.root
    training, identity, profile = inputs()
    rows = [
        {"schema_version": "syntunia-sft-row/v1", "format": "raw_text",
         "row_id": f"row-{index}", "text": "example", "split": "train" if index < 3 else "validation"}
        for index in range(5)
    ]
    dataset = b"".join(canonical(row) + b"\n" for row in rows)
    identity = replace(identity, content_digest=hashlib.sha256(dataset).hexdigest(),
                       size_bytes=len(dataset))
    source = {"schema": "synaptic-native-pushed-source/v1", "url": "https://example.invalid/engine.git",
              "ref": "refs/heads/native-review", "commit": "a" * 40}
    source_digest = hashlib.sha256(canonical(source)).hexdigest()
    hardware = {"schema": "synaptic-native-hardware-observation/v1", "machine": "arm64",
                "macos": "15.0", "kernel": "fixture", "model": "fixture-M4", "cpu": "Apple M4",
                "memory_bytes": 25769803776}
    hardware_digest = hashlib.sha256(canonical(hardware)).hexdigest()
    profile = replace(profile, source_digest=source_digest,
                      runtime_digest=hashlib.sha256(state.kwargs["runtime_capture"]).hexdigest())
    projection = compile_native_sft(training, identity, profile)
    binding_path = publication["output"] / "input_binding.json"
    binding = strict_json(binding_path.read_bytes())
    binding.update(source=identity.to_dict(), workload_digest=projection.workload.fingerprint,
                   runtime_digest=profile.runtime_digest)
    binding_path.write_bytes(canonical(binding))
    result_path = publication["output"] / "native_result.json"
    result = strict_json(result_path.read_bytes())
    result.update(input_binding_digest=hashlib.sha256(binding_path.read_bytes()).hexdigest(),
                  workload_digest=projection.workload.fingerprint,
                  runtime_digest=profile.runtime_digest)
    result_path.write_bytes(canonical(result))
    (publication["output"] / "workload.json").write_bytes(projection.workload.canonical_bytes)
    proposal_path = Path(native_host.ENGINE_ROOT) / "Trainers/mlx_sft_mac/runtime-candidate/sft-qualification-proposal.json"
    proposal = json.loads(proposal_path.read_text())
    proposal["profile_ref"] = profile.profile_ref
    proposal["model"].update(repository=profile.model_ref, revision=profile.model_revision,
                             tokenizer_repository=profile.model_ref,
                             tokenizer_revision=profile.tokenizer_revision)
    proposal["runtime"].update(dependency_lock_sha256=hashlib.sha256(b"fixture-hash-lock").hexdigest())
    proposal["bounds"].update(timeout_seconds_per_attempt=5, termination_grace_seconds=1,
                              maximum_output_bytes_per_stream=4096,
                              maximum_artifact_file_bytes=1024 * 1024,
                              maximum_artifact_set_bytes=4 * 1024 * 1024)
    files = {
        "proposal.json": proposal,
        "profile.json": asdict(profile),
        "training.json": training.to_dict(),
        "identity.json": identity.to_dict(),
        "model-manifest.json": asdict(publication["model"]),
        "tokenizer-manifest.json": asdict(publication["tokenizer"]),
    }
    for filename, value in files.items():
        (root / filename).write_bytes(canonical(value))
    public_key, private_key = root / "root-public.pem", root / "root-private.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "EC", "-pkeyopt",
                    "ec_paramgen_curve:P-256", "-out", str(private_key)], check=True,
                   capture_output=True)
    subprocess.run(["openssl", "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)],
                   check=True, capture_output=True)
    config = {
        "schema": "synaptic-native-host-config/v1", "proposal_file": str(root / "proposal.json"),
        "engine_root": str(root / "engine"), "engine_remote": source["url"],
        "engine_ref": source["ref"], "engine_commit": source["commit"],
        "runtime_capture_file": str(root / "capture.json"),
        "dependency_lock_file": str(root / "dependency.lock"),
        "profile_file": str(root / "profile.json"), "training_input_file": str(root / "training.json"),
        "prepared_identity_file": str(root / "identity.json"),
        "prepared_publication": str(root / "prepared-publication"),
        "model_snapshot_file": str(root / "model-manifest.json"),
        "tokenizer_snapshot_file": str(root / "tokenizer-manifest.json"),
        "admission_file": str(root / "admission.json"),
        "admission_signature_file": str(root / "admission.sig"),
        "root_public_key_file": str(public_key),
        "root_public_key_sha256": hashlib.sha256(public_key.read_bytes()).hexdigest(),
        "limits": {"timeout_seconds": 5, "termination_grace_seconds": 1,
                   "maximum_output_bytes": 4096, "maximum_file_bytes": 1024 * 1024},
        "scope": {"account_ref": "owner", "namespace_ref": "machine"},
        "run_id": "run", "project_ref": "project", "request_id": "request",
        "attempt": "attempt-1", "attempt_kind": "raw_full_sequence",
        "state_root": str(root / "attempt-state"),
        "host": {
            "authority_secret": {"provider": "env", "name": "NATIVE_TEST_AUTHORITY"},
            "evidence_secret": {"provider": "env", "name": "NATIVE_TEST_EVIDENCE"},
            "material_secret": {"provider": "env", "name": "NATIVE_TEST_MATERIAL"},
            "grant_ref": "native-test-grant", "reader_authority_ref": "reader-test",
            "reader_key_ref": "reader-key", "material_authority_ref": "material-test",
            "material_key_ref": "material-key",
        },
        "tokenizer_members": list(publication["tokenizer_members"]),
        "artifact_budget": {"per_file": 1024 * 1024, "aggregate": 4 * 1024 * 1024},
        "use_fast_tokenizer": profile.use_fast_tokenizer,
    }
    for key in ("NATIVE_TEST_AUTHORITY", "NATIVE_TEST_EVIDENCE", "NATIVE_TEST_MATERIAL"):
        monkeypatch.setenv(key, "test-secret-" + key + "-" + "x" * 32)
    clock = {"now": "2026-10-20T12:00:00Z"}

    class TestClock:
        def now(self):
            return clock["now"]

        def now_epoch(self):
            from datetime import datetime
            return int(datetime.fromisoformat(clock["now"].replace("Z", "+00:00")).timestamp())

    monkeypatch.setattr(native_host, "Clock", TestClock)
    remote = {"advanced": False}
    def source_record(selected, *, check_remote=True):
        if check_remote and remote["advanced"]:
            raise ValueError("native pushed source ref differs")
        return source, source_digest
    monkeypatch.setattr(native_host, "source_record", source_record)
    monkeypatch.setattr(native_host, "hardware_record", lambda: (hardware, hardware_digest))
    monkeypatch.setattr(native_host, "verify_native_material", lambda *a, **k: None)
    monkeypatch.setattr(family_module, "verify_native_material", lambda *a, **k: None)
    monkeypatch.setattr(native_host, "read_retained_native_source", lambda prepared, selected: dataset)
    admission_fields = native_host.inspect(config)[0]["admission_candidate"]
    unsigned = {**admission_fields, "expires_at": "2026-10-20T12:30:00Z",
                "authority_ref": "root-review", "key_ref": "root-key"}
    signature = root / "admission.sig"
    subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(private_key),
                    "-out", str(signature)], input=canonical(unsigned), check=True,
                   capture_output=True)
    admission = {**unsigned, "tag": hashlib.sha256(signature.read_bytes()).hexdigest()}
    (root / "admission.json").write_bytes(canonical(admission))
    calls = []
    def child(self, invocation):
        calls.append(invocation.digest)
        self._retain(invocation, "claim", {"automatic_replay": False})
        output = Path(invocation.private_root) / "output"
        for item in publication["output"].iterdir():
            (output / item.name).write_bytes(item.read_bytes())
        outcome = {"schema": "synaptic-native-process-outcome/v1", "reason": "worker_exited",
                   "exit_code": 0, "termination_confirmed": True,
                   "stdout_bytes": 0, "stderr_bytes": 0}
        self._retain(invocation, "outcome", outcome)
        return outcome
    monkeypatch.setattr(NativeOwnedProcess, "run_once", child)
    return config, clock, remote, calls


@pytest.mark.parametrize("mutation", [
    "learning_rate", "rank", "targets", "batch", "tokenizer", "state_root",
    "project", "request", "artifact_budget",
])
def test_root_admission_rejects_changed_execution_selection(host_case, mutation):
    config, _, _, calls = host_case
    changed = deepcopy(config)
    def edit(path, update):
        document = strict_json(Path(path).read_bytes())
        update(document)
        Path(path).write_bytes(canonical(document))
    if mutation == "learning_rate":
        edit(config["training_input_file"], lambda d: d["hyperparameters"].update(learning_rate=.0002))
        edit(config["proposal_file"], lambda d: d["optimizer"].update(learning_rate=.0002))
    elif mutation == "rank":
        edit(config["training_input_file"], lambda d: d["hyperparameters"].update(
            lora_rank=8, lora_alpha=64))
        edit(config["proposal_file"], lambda d: d["lora"].update(rank=8))
    elif mutation == "targets":
        edit(config["training_input_file"], lambda d: d["hyperparameters"].update(
            lora_target_modules=["self_attn.q_proj"]))
        edit(config["proposal_file"], lambda d: d["lora"].update(targets=["self_attn.q_proj"]))
    elif mutation == "batch":
        edit(config["training_input_file"], lambda d: d["hyperparameters"].update(
            batch_size=2, max_seq_length=64))
        edit(config["proposal_file"], lambda d: d["training"].update(batch_size=2, max_seq_length=64))
    elif mutation == "tokenizer":
        changed["use_fast_tokenizer"] = False
        edit(config["profile_file"], lambda d: d.update(use_fast_tokenizer=False))
    elif mutation == "state_root":
        changed["state_root"] = str(Path(config["state_root"]).with_name("fresh-state"))
    elif mutation == "project":
        changed["project_ref"] = "different-project"
    elif mutation == "request":
        changed["request_id"] = "different-request"
    elif mutation == "artifact_budget":
        changed["artifact_budget"]["aggregate"] = 8 * 1024 * 1024
        edit(config["proposal_file"], lambda d: d["bounds"].update(
            maximum_artifact_set_bytes=8 * 1024 * 1024))
    with pytest.raises(ValueError, match="restricted admission selection differs"):
        native_host.inspect(changed, require_admission=True,
                            trusted_root_key_sha256=config["root_public_key_sha256"])
    assert not Path(changed["state_root"]).exists() and calls == []


@pytest.mark.skipif(os.name != "posix", reason="cold native host read uses POSIX fork")
def test_cold_public_readback_survives_expiry_and_remote_ref_advance(host_case):
    config, clock, remote, calls = host_case
    trusted = config["root_public_key_sha256"]
    start = native_host.operate(config, action="start", trusted_root_key_sha256=trusted)
    assert start["accepted"] and len(calls) == 1
    clock["now"] = "2026-10-20T14:00:00Z"
    remote["advanced"] = True
    def cold_reader(connection):
        try:
            shown = native_host.operate(config, action="show", trusted_root_key_sha256=trusted)
            observed = native_host.operate(config, action="observe", trusted_root_key_sha256=trusted)
            verified = native_host.operate(config, action="verify", trusted_root_key_sha256=trusted)
            reverified = native_host.operate(config, action="reverify", trusted_root_key_sha256=trusted)
            downloaded = native_host.operate(
                config, action="download", trusted_root_key_sha256=trusted,
                destination=Path(config["state_root"]) / "downloaded",
            )
            connection.send((shown, observed, verified, reverified, downloaded, len(calls)))
        finally:
            connection.close()
    parent, child = multiprocessing.get_context("fork").Pipe(duplex=False)
    reader = multiprocessing.get_context("fork").Process(target=cold_reader, args=(child,))
    reader.start()
    child.close()
    try:
        assert parent.poll(10), "cold native reader did not finish"
        shown, observed, verified, reverified, downloaded, child_calls = parent.recv()
        reader.join(2)
        assert reader.exitcode == 0 and child_calls == 1
    finally:
        if reader.is_alive():
            reader.terminate()
            reader.join(2)
        parent.close()
    assert shown["run"]["run_id"] == "run"
    assert observed["state"] == "succeeded"
    assert verified["verified"] is True
    assert reverified["verified"] is True
    assert len(downloaded["paths"]) == 5 and len(calls) == 1
    with pytest.raises(ValueError, match="pushed source"):
        native_host.operate(config, action="start", trusted_root_key_sha256=trusted)
    remote["advanced"] = False
    with pytest.raises(ValueError, match="expired"):
        native_host.operate(config, action="start", trusted_root_key_sha256=trusted)
    assert len(calls) == 1
