"""Config-first, generation-only native host; never submits training.

An external signature admits one exact bounded attempt. Original authority can
read completed evidence after start expiry; occupied states never authorize retry.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import io
import os
from pathlib import Path
import re
import stat
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bounded_command import run_bounded_command
from native_host import source_record, hardware_record, _snapshot
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from tuner.runtime.native_files import NativeFile, NativeTree
from tuner.runtime.native_mlx_artifacts import finite_adapter_tensors
from tuner.runtime.native_mlx_evaluation import parse_workload, read_cases, validate_record, validate_result
from tuner.runtime.native_mlx_input import NativeModelSnapshot
from tuner.runtime.native_mlx_material import canonical, strict_json, verify_native_material
from tuner.runtime.native_process import NativeInvocation, NativeOwnedProcess, NativeProcessLimits


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, maximum):
    p = Path(path)
    if not p.is_absolute():
        raise ValueError("native evaluation requires absolute paths")
    with NativeTree(p.parent) as tree:
        return tree.read(p.name, maximum=maximum)


def document(path, maximum=2 << 20):
    raw = read(path, maximum)
    value = strict_json(raw, maximum=maximum)
    if canonical(value) != raw:
        raise ValueError("native evaluation document must be canonical")
    return value


def mkdir(path):
    with NativeTree(path.parent, private=True) as parent:
        os.mkdir(path.name, mode=0o700, dir_fd=parent.fd)
        parent.check()


def direct_names(tree, maximum=256):
    tree.check()
    names = []
    with os.scandir(tree.fd) as entries:
        for entry in entries:
            if len(names) == maximum:
                raise ValueError("native evaluation member bound exceeded")
            info = os.stat(entry.name, dir_fd=tree.fd, follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) not in (0o400, 0o600)):
                raise ValueError("native evaluation member custody differs")
            names.append(entry.name)
    return tuple(sorted(names))


def adapter_files(raw, *, model):
    """Inspect only bounded plain native archive bytes; never extractall."""
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            if (len(result) >= 35 or not member.isfile() or member.name in result
                    or not (member.name in {"adapters.safetensors", "adapter_config.json", "native_model.json"}
                            or re.fullmatch(r"[0-9]{7}_adapters\.safetensors", member.name))
                    or not 0 < member.size <= 64 << 20 or member.mode != 0o600
                    or member.uid != 0 or member.gid != 0 or member.mtime != 0):
                raise ValueError("native evaluation adapter archive differs")
            value = archive.extractfile(member).read(member.size + 1)
            if len(value) != member.size:
                raise ValueError("native evaluation adapter member differs")
            result[member.name] = value
    if not {"adapters.safetensors", "adapter_config.json", "native_model.json"} <= set(result):
        raise ValueError("native evaluation adapter inventory differs")
    metadata = strict_json(result["native_model.json"], maximum=8192)
    if (type(metadata) is not dict or set(metadata) != {
            "schema", "format", "repository", "revision", "runtime_digest", "snapshot_digest", "workload_digest"}
            or metadata["schema"] != "synaptic-native-mlx-model/v1"
            or metadata["format"] != "mlx-lm-lora-safetensors"
            or canonical(metadata) != result["native_model.json"]
            or any(type(metadata[name]) is not str or not re.fullmatch(r"[a-f0-9]{64}", metadata[name])
                   for name in ("runtime_digest", "workload_digest"))
            or (metadata["repository"], metadata["revision"], metadata["snapshot_digest"]) != (
                model.repository, model.revision, model.digest)):
        raise ValueError("native evaluation adapter model binding differs")
    config = strict_json(result["adapter_config.json"], maximum=8192)
    if (type(config) is not dict or canonical(config) != result["adapter_config.json"]
            or set(config) != {"fine_tune_type", "num_layers", "lora_parameters"}
            or config["fine_tune_type"] != "lora" or type(config["lora_parameters"]) is not dict
            or set(config["lora_parameters"]) != {"rank", "scale", "dropout", "keys"}):
        raise ValueError("native evaluation adapter configuration differs")
    params = config["lora_parameters"]
    declared = {"artifact_sha256": sha(raw), "adapter_sha256": sha(result["adapters.safetensors"]),
        "lora": {"layers": config["num_layers"], "rank": params["rank"], "scale": params["scale"],
                 "dropout": params["dropout"], "targets": params["keys"]}}
    for name, value in result.items():
        if name.endswith(".safetensors"):
            finite_adapter_tensors(value)
    return result, declared


def plan(config, *, retained=False):
    expected = {"schema", "engine_root", "engine_commit", "engine_ref", "engine_remote",
        "runtime_capture_file", "dependency_lock_file", "model_snapshot_file", "tokenizer_snapshot_file",
        "cases_file", "adapter_file", "profile", "generation", "state_root", "attempt", "limits",
        "admission_file", "admission_signature_file", "root_public_key_file", "root_public_key_sha256"}
    if type(config) is not dict or set(config) != expected or config["schema"] != "synaptic-native-evaluation-host/v1":
        raise ValueError("native evaluation host configuration differs")
    source, source_digest = source_record(config, check_remote=not retained)
    hardware, _ = hardware_record()
    capture_raw = read(config["runtime_capture_file"], 16 << 20)
    lock = read(config["dependency_lock_file"], 1 << 20)
    verify_native_material(capture_raw, expected_digest=sha(capture_raw), dependency_lock=lock)
    capture = strict_json(capture_raw)
    state = Path(config["state_root"])
    if not state.is_absolute() or str(state) != os.path.normpath(str(state)):
        raise ValueError("native evaluation state path differs")
    with NativeTree(state.parent, private=True) as parent:
        if not retained:
            try:
                os.stat(state.name, dir_fd=parent.fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("native evaluation state already claimed")
    if type(config["attempt"]) is not str or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", config["attempt"]):
        raise ValueError("native evaluation attempt differs")
    model = _snapshot(config["model_snapshot_file"])
    tokenizer = _snapshot(config["tokenizer_snapshot_file"])
    cases_raw = read(config["cases_file"], 2 << 20)
    case_doc = strict_json(cases_raw, maximum=2 << 20)
    files, adapter, adapter_snapshot = {}, None, None
    if config["adapter_file"] is not None:
        selected = config["adapter_file"]
        if type(selected) is not dict or set(selected) != {"path", "sha256"}:
            raise ValueError("native evaluation adapter source differs")
        blob = read(selected["path"], 64 << 20)
        if sha(blob) != selected["sha256"]:
            raise ValueError("native evaluation adapter artifact hash differs")
        files, adapter = adapter_files(blob, model=model)
        adapter_snapshot = NativeModelSnapshot(model.repository, model.revision, str(state / "adapter"),
            tuple(NativeFile(name, len(value), sha(value)) for name, value in sorted(files.items())))
    profile = config["profile"]
    if type(profile) is not dict or set(profile) != {"profile_ref", "model_ref", "model_revision", "tokenizer_revision",
            "quantization_bits", "maximum_cases",
            "maximum_context_tokens", "maximum_response_bytes", "maximum_evidence_bytes"}:
        raise ValueError("native evaluation profile controls differ")
    if (profile["model_ref"], profile["model_revision"], profile["tokenizer_revision"]) != (
            model.repository, model.revision, tokenizer.revision):
        raise ValueError("native evaluation named profile model compatibility differs")
    workload = {"schema_version": "synaptic-native-mlx-evaluation-workload/v1", "method": "evaluation",
        "profile": {**profile, "runtime_digest": sha(capture_raw), "source_digest": source_digest},
        "generation": config["generation"], "adapter": adapter,
        "cases_sha256": sha(cases_raw), "case_count": len(case_doc["cases"])}
    parse_workload(canonical(workload))
    read_cases(cases_raw, workload)
    limits = NativeProcessLimits(**config["limits"])
    if limits.maximum_file_bytes < profile["maximum_evidence_bytes"]:
        raise ValueError("native evaluation process/evidence bounds differ")
    request = {"schema": "synaptic-native-evaluation-child-request/v1", "workload": workload,
        "runtime_capture": str(state / "capture.json"), "dependency_lock": str(state / "dependency.lock"),
        "cases": str(state / "cases.json"), "model_snapshot": asdict(model),
        "tokenizer_snapshot": asdict(tokenizer), "adapter_snapshot": asdict(adapter_snapshot) if adapter_snapshot else None,
        "output": str(state / "output")}
    selection = {"schema": "synaptic-native-evaluation-selection/v1", "config": config,
        "source": source, "hardware": hardware, "request": request, "dependency_lock_sha256": sha(lock)}
    binding = sha(canonical(selection))
    inventory = {str(Path(t["root"]) / m["path"]): m["sha256"] for t in capture["trees"] for m in t["members"]}
    guardian = str(ROOT / "tuner/runtime/native_process_guardian.py")
    worker = str(ROOT / "tuner/runtime/native_mlx_child.py")
    invocation = NativeInvocation(config["attempt"], binding, capture["executable"],
        capture["runtime"]["executable_sha256"], guardian, inventory[guardian], worker, inventory[worker],
        str(state / "request.json"), sha(canonical(request)), str(state), limits)
    return selection, invocation, capture_raw, lock, cases_raw, files


def authenticate(config, selection, *, starting, trusted_root_key_sha256):
    return verify_admission(selection, admission_file=config["admission_file"],
        signature_file=config["admission_signature_file"], key_file=config["root_public_key_file"],
        trusted_root_key_sha256=trusted_root_key_sha256, starting=starting)


def verify_admission(selection, *, admission_file, signature_file, key_file, trusted_root_key_sha256, starting):
    config = selection["config"]
    key = read(key_file, 16384)
    if (type(trusted_root_key_sha256) is not str or not re.fullmatch(r"[a-f0-9]{64}", trusted_root_key_sha256)
            or sha(key) != config["root_public_key_sha256"] or sha(key) != trusted_root_key_sha256):
        raise ValueError("native evaluation public key pin differs")
    admission = document(admission_file, 16384)
    if (type(admission) is not dict or set(admission) != {"schema", "selection_digest", "attempt", "expires_at"}
            or admission["schema"] != "synaptic-native-evaluation-admission/v1"
            or admission["selection_digest"] != sha(canonical(selection)) or admission["attempt"] != config["attempt"]):
        raise ValueError("native evaluation admission differs")
    expiry = datetime.strptime(admission["expires_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if starting and not 0 < (expiry - datetime.now(timezone.utc)).total_seconds() <= 3600:
        raise ValueError("native evaluation start window differs")
    signature = read(signature_file, 16384)
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}
    key_status, modulus = run_bounded_command(["openssl", "rsa", "-pubin", "-in", str(key_file),
        "-noout", "-modulus"], environment=environment, maximum=4096, timeout=10)
    matched = re.fullmatch(rb"Modulus=([A-F0-9]+)\n?", modulus)
    if (key_status != 0 or matched is None or len(matched[1]) not in (512, 768, 1024)
            or len(signature) * 2 != len(matched[1])):
        raise ValueError("native evaluation RSA signature extent differs")
    status, out = run_bounded_command(["openssl", "dgst", "-sha256", "-verify", str(key_file),
        "-signature", str(signature_file)], input_bytes=canonical(admission),
        environment=environment, maximum=4096, timeout=10)
    if (status != 0 or out.strip() != b"Verified OK" or read(key_file, 16384) != key
            or read(signature_file, 16384) != signature):
        raise ValueError("native evaluation signature refused")
    return admission


def readback(config, selection, invocation):
    state = Path(config["state_root"])
    with SQLiteDurableStoreV1(state / "workflow.sqlite3", create=False) as records:
        observed = NativeOwnedProcess(records).observe(invocation)
    if (observed.get("state") != "terminated" or observed["outcome"].get("exit_code") != 0
            or observed["outcome"].get("termination_confirmed") is not True):
        raise ValueError("native evaluation is incomplete or uncertain")
    workload = selection["request"]["workload"]
    with NativeTree(state, private=True) as tree, NativeTree(state / "output", private=True) as output:
        if tree.read("selection.json", maximum=4 << 20) != canonical(selection):
            raise ValueError("native evaluation retained selection differs")
        result = strict_json(output.read("result.json", maximum=1 << 20))
        validate_result(result, selection["request"])
        cases = read_cases(tree.read("cases.json", maximum=2 << 20), workload)
        names, total = [], 0
        for index, (member, case) in enumerate(zip(result["case_files"], cases)):
            name = f"case-{index:04d}.json"
            if type(member) is not dict or set(member) != {"name", "sha256", "size"} or member["name"] != name:
                raise ValueError("native evaluation case inventory differs")
            raw = output.read(name, maximum=workload["profile"]["maximum_evidence_bytes"])
            if len(raw) != member["size"] or sha(raw) != member["sha256"]:
                raise ValueError("native evaluation case bytes differ")
            row = strict_json(raw)
            validate_record(row, case, workload)
            names.append(name)
            total += len(raw)
        if (direct_names(output) != tuple(sorted(["result.json", *names])) or result["case_bytes"] != total
                or total > workload["profile"]["maximum_evidence_bytes"]):
            raise ValueError("native evaluation final inventory differs")
    return {"schema": "synaptic-native-evaluation-readback/v1", "status": "COMPLETE",
        "selection_digest": sha(canonical(selection)), "result_sha256": sha(canonical(result)),
        "case_count": len(cases), "outcome": observed["outcome"], "training_submitted": False}


def export(config, selection, invocation, destination):
    """Copy complete private evidence; no model loads or new execution authority."""
    receipt = readback(config, selection, invocation)
    state = Path(config["state_root"])
    with NativeTree(state, private=True) as root, NativeTree(state / "output", private=True) as output:
        result_raw = output.read("result.json", maximum=1 << 20)
        if sha(result_raw) != receipt["result_sha256"]:
            raise ValueError("native evaluation export result changed")
        result = strict_json(result_raw)
        values = {"selection.json": canonical(selection), "host-readback.json": canonical(receipt),
            "cases.json": root.read("cases.json", maximum=2 << 20), "result.json": result_raw,
            "admission.json": read(config["admission_file"], 16384),
            "admission.sig": read(config["admission_signature_file"], 16384),
            "root-public-key.pem": read(config["root_public_key_file"], 16384)}
        total = 0
        for item in result["case_files"]:
            raw = output.read(item["name"], maximum=selection["request"]["workload"]["profile"]["maximum_evidence_bytes"])
            if sha(raw) != item["sha256"] or len(raw) != item["size"]:
                raise ValueError("native evaluation export case changed")
            values[item["name"]] = raw
            total += len(raw)
        if total > selection["request"]["workload"]["profile"]["maximum_evidence_bytes"]:
            raise ValueError("native evaluation export bytes exceeded")
    destination = Path(destination)
    mkdir(destination)
    manifest = {"schema": "synaptic-native-evaluation-export/v1", "selection_digest": sha(canonical(selection)),
        "members": [{"name": name, "sha256": sha(raw), "size": len(raw)} for name, raw in sorted(values.items())],
        "execution_witness": "operator-retained owned-process receipt; not portable provider authentication"}
    with NativeTree(destination, private=True) as tree:
        for name, raw in sorted(values.items()):
            tree.write_exclusive(name, raw, maximum=32 << 20)
        tree.write_exclusive("manifest.json", canonical(manifest))
    return {"schema": "synaptic-native-evaluation-export-result/v1", "status": "EXPORTED",
        "manifest_sha256": sha(canonical(manifest)), "case_count": result["case_count"],
        "destination": str(destination), "model_executed": False}


def run(action, config_path, *, expires_at=None, trusted_root_key_sha256=None, destination=None):
    if action not in {"inspect", "admission-candidate", "execute", "readback", "export"}:
        raise ValueError("native evaluation action differs")
    config = document(config_path)
    selection, invocation, capture, lock, cases, files = plan(config, retained=action in {"readback", "export"})
    if action in {"inspect", "admission-candidate"}:
        if action == "admission-candidate":
            expiry = datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if not 0 < (expiry - datetime.now(timezone.utc)).total_seconds() <= 3600:
                raise ValueError("native evaluation candidate window differs")
            return {"schema": "synaptic-native-evaluation-admission/v1", "selection_digest": sha(canonical(selection)),
                "attempt": config["attempt"], "expires_at": expires_at}
        return {"schema": "synaptic-native-evaluation-inspection/v1", "status": "REVIEW_REQUIRED",
            "selection_digest": sha(canonical(selection)), "source": selection["source"],
            "workload": selection["request"]["workload"], "limits": config["limits"]}
    authenticate(config, selection, starting=action == "execute", trusted_root_key_sha256=trusted_root_key_sha256)
    if action == "readback":
        return readback(config, selection, invocation)
    if action == "export":
        return export(config, selection, invocation, destination)
    state = Path(config["state_root"])
    mkdir(state)
    mkdir(state / "output")
    if files:
        mkdir(state / "adapter")
        with NativeTree(state / "adapter", private=True) as tree:
            for name, raw in sorted(files.items()):
                tree.write_exclusive(name, raw, maximum=64 << 20)
    with NativeTree(state, private=True) as tree:
        for name, raw in (("capture.json", capture), ("dependency.lock", lock), ("cases.json", cases),
                ("request.json", canonical(selection["request"])), ("selection.json", canonical(selection))):
            tree.write_exclusive(name, raw, maximum=16 << 20)
    authenticate(config, selection, starting=True, trusted_root_key_sha256=trusted_root_key_sha256)
    with SQLiteDurableStoreV1(state / "workflow.sqlite3", create=True) as records:
        NativeOwnedProcess(records).run_once(invocation)
    return readback(config, selection, invocation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "admission-candidate", "execute", "readback", "export"))
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--expires-at")
    parser.add_argument("--trusted-root-key-sha256")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.action, args.configuration, expires_at=args.expires_at,
            trusted_root_key_sha256=args.trusted_root_key_sha256, destination=args.destination)
        raw = canonical(result)
        if len(raw) > 8192:
            raise ValueError("native evaluation summary exceeded")
    except Exception:
        print('{"status":"REFUSED_OR_UNCERTAIN","automatic_retry":false}')
        return 2
    sys.stdout.buffer.write(raw + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
