"""Provider-free release chain: all image facts below are test fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from tuner.cloud import derived_training_image as d
from tuner.cloud.hf_training_image_lock import CommandResult
from tuner.runtime.packaged_worker_closure import load_packaged_worker_closure, stable_read


def write(path, value):
    path.write_bytes(d._canonical_bytes(value))
    return path


@pytest.fixture
def chain(tmp_path):
    from zipfile import ZipFile
    wheel_path = tmp_path / "synaptic_tuner-1.0.0-py3-none-any.whl"
    with ZipFile(wheel_path, "w") as archive:
        archive.writestr("Trainers/sft/train_sft.py", "raise AssertionError('must never execute')\n")
    wheel_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    capability = {"compatibility": {"methods": ["sft"], "models": [{"ref": "org/model", "revision": "a" * 40}], "dataset_formats": ["syntunia-sft-row/v2"]},
                  "contracts": {"workload_schema": "synaptic-sft-workload/v1", "prepared_input_schema": "synaptic-prepared-training-input/v1", "artifact_contract_schema": "synaptic-sft-artifacts/v1"}}
    packaged = {
        "wheel": {"filename": "synaptic_tuner-1.0.0-py3-none-any.whl", "distribution": "synaptic-tuner", "version": "1.0.0", "sha256": wheel_digest},
        "bootstrap": [{"filename": "bootstrap-1.0.0-py3-none-any.whl", "distribution": "bootstrap", "version": "1.0.0", "sha256": "c" * 64}],
        "python": {"implementation": "cpython", "version": "3.12.3", "executable": "/opt/python/bin/python3.12", "executable_digest": "d" * 64, "purelib": "/opt/python/lib/python3.12/site-packages", "platlib": "/opt/python/lib/python3.12/site-packages"},
        "capabilities": capability,
    }
    profile_path = write(tmp_path / "profile.json", {"schema_version": d.PROFILE_SCHEMA, "name": "test-runtime", "platform": d.PLATFORM, "base_image": "docker.io/unsloth/unsloth@sha256:" + "1" * 64, "python_executable": packaged["python"]["executable"], "packages": [], "packaged_runtime": packaged})
    profile = d.load_profile(profile_path)
    closure = load_packaged_worker_closure().digest
    inventory = [{"name": "bootstrap", "version": "1.0.0"}, {"name": "synaptic-tuner", "version": "1.0.0"}]
    measured = {
        "schema_version": "synaptic-packaged-runtime-inspector/v1",
        "package": {"name": "synaptic-tuner", "version": "1.0.0", "digest": wheel_digest, "source_provenance_digest": "e" * 64},
        "python": {k: packaged["python"][k] for k in ("implementation", "version", "executable", "executable_digest")},
        "installed_distributions": {"inventory": inventory, "count": 2, "digest": hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()},
        "platform": {"system": "linux", "machine": "x86_64", "cuda_version": None, "runtime_facts": {"python_cache_tag": "cpython-312"}},
        "worker": {"entrypoint": d.PACKAGED_TRAINING_WORKER_ENTRYPOINT, "closure_digest": closure},
        "closure": {"verified_digest": closure}, "contracts": capability["contracts"], "capabilities": capability,
        "build_inputs_digest": hashlib.sha256(d._canonical_bytes(packaged)).hexdigest(),
        "provenance": {"bootstrap": "f" * 64, "synaptic-tuner": "e" * 64},
    }
    image_ref = "registry.example/synaptic/runtime@sha256:" + "2" * 64
    config_digest = "sha256:" + "3" * 64
    metadata = {"containerimage.digest": "sha256:" + "2" * 64, "containerimage.config.digest": config_digest,
                "buildx.build.provenance": {"invocation": {"parameters": {"args": {"build-arg:SYNAPTIC_RUNTIME_INPUTS_SHA256": measured["build_inputs_digest"]}}}}}
    receipt = {"schema_version": d.BUILD_RECEIPT_SCHEMA, "status": "BUILT_NOT_ATTESTED", "profile_sha256": profile.canonical_sha256, "base_image": profile.base_image,
               "dockerfile_sha256": d._sha256(d.render_dockerfile(profile).encode()), "tag": "registry.example/synaptic/runtime:test",
               "final_oci_reference": image_ref, "final_oci_digest": "sha256:" + "2" * 64, "image_config_digest": config_digest,
               "repo_digests": [image_ref], "build_metadata": metadata, "build_metadata_sha256": d._sha256(d._canonical_bytes(metadata))}
    build_path = write(tmp_path / "build.json", receipt)
    runtime = {"schema_version": "synaptic-derived-training-image-runtime/v1", "packages": {}, "final_runtime": measured}
    candidate = {**{k: v for k, v in receipt.items() if k not in {"schema_version", "status", "tag"}}, "schema_version": d.CANDIDATE_SCHEMA, "status": "CANDIDATE_ONLY", "build_receipt_sha256": d._sha256(build_path.read_bytes()), "runtime": runtime, "runtime_bytes_sha256": d._sha256(d._runtime_payload_bytes(runtime))}
    candidate_path = write(tmp_path / "candidate.json", candidate)
    evidence = {"profile_path": profile_path, "build_receipt_path": build_path, "candidate_path": candidate_path, "image_verification_path": tmp_path / "image.json"}
    d.verify_candidate(profile_path=profile_path, candidate_path=candidate_path, output=evidence["image_verification_path"])
    # Dummy executable files authenticate command construction; runner never
    # invokes a process, Docker, an image or a provider.
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"test executable identity")
    config = tmp_path / "docker-config"
    config.mkdir()
    calls = []
    def runner(spec):
        calls.append(spec)
        if "inspect" in spec.argv:
            labels = {"ai.synapticlabs.derived-training-profile": profile.canonical_sha256, "ai.synapticlabs.derived-training-base": profile.base_image}
            return CommandResult(stdout=(json.dumps(config_digest) + "\n" + json.dumps([image_ref]) + "\n" + json.dumps(labels)).encode())
        assert spec.argv[spec.argv.index("--entrypoint") + 1] == packaged["python"]["executable"]
        assert "-I" in spec.argv and "-S" in spec.argv
        assert "/usr/bin/env" not in spec.argv
        return CommandResult(stdout=d._canonical_bytes(measured))
    capture_path = tmp_path / "capture.json"
    d.capture_final_runtime(**evidence, docker=docker, docker_config=config, output=capture_path, runner=runner)
    compatibility_path = write(tmp_path / "compatibility.json", {"release_ref": "runtime:test", "compatibility": capability["compatibility"]})
    return {"evidence": evidence, "capture_path": capture_path, "compatibility_path": compatibility_path,
            "root": tmp_path, "profile": profile, "measured": measured, "calls": calls}


def build(chain, name="release.json"):
    return d.build_runtime_release(**chain["evidence"], capture_path=chain["capture_path"], compatibility_path=chain["compatibility_path"], output=chain["root"] / name)


def local_inputs(chain):
    release = build(chain)
    config = write(chain["root"] / "local-config.json", {"schema_version": d.LOCAL_QUALIFICATION_CONFIG_SCHEMA,
        "protocol": "synaptic-installed-child-cpu/v1", "runtime_release_digest": release.manifest_digest})
    return dict(**chain["evidence"], capture_path=chain["capture_path"], release_path=chain["root"] / "release.json", qualification_config_path=config)


def test_local_cpu_plan_has_fixed_offline_policy_without_docker(chain):
    inputs = local_inputs(chain)
    output = chain["root"] / "local.json"
    plan = d.qualify_local_runtime(**inputs, docker=chain["root"] / "absent", docker_config=chain["root"] / "absent-config", output=output)
    assert plan["status"] == "PLAN_ONLY" and not output.exists()
    argv = plan["argv"]
    for flag in ("--pull=never", "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--runtime=runc", "--env=NVIDIA_VISIBLE_DEVICES=void", "--env=CUDA_VISIBLE_DEVICES=", "-I", "-S"):
        assert flag in argv
    assert not any(item in argv for item in ("--gpus", "--mount", "-v", "--privileged", "--env"))
    assert "os.environ.clear()" in argv[-1]
    assert [item for item in argv if item.startswith("--env=")] == ["--env=NVIDIA_VISIBLE_DEVICES=void", "--env=CUDA_VISIBLE_DEVICES="]


@pytest.mark.parametrize("mutation", [None, "child", "image", "stderr", "runtime_unavailable"])
def test_local_cpu_capture_revalidates_and_verifies_exclusive_evidence(chain, mutation):
    inputs = local_inputs(chain)
    profile, release, image, expected = d._local_qualification_inputs(**inputs)
    count = 0
    def runner(spec):
        nonlocal count
        count += 1
        if "inspect" in spec.argv:
            identity = image["image_config_digest"] if mutation != "image" or count == 1 else "sha256:" + "0" * 64
            labels = {"ai.synapticlabs.derived-training-profile": profile.canonical_sha256, "ai.synapticlabs.derived-training-base": profile.base_image}
            return CommandResult(stdout=(json.dumps(identity) + "\n" + json.dumps([image["final_oci_reference"]]) + "\n" + json.dumps(labels)).encode())
        if mutation == "runtime_unavailable":
            raise d.DerivedTrainingImageError("COMMAND_FAILED")
        child = dict(expected["child"])
        if mutation == "child": child["training_executed"] = True
        return CommandResult(stdout=d._canonical_bytes(child), stderr=b"private" if mutation == "stderr" else b"")
    output = chain["root"] / "local.json"
    kwargs = dict(**inputs, docker=chain["root"] / "docker.exe", docker_config=chain["root"] / "docker-config", output=output, execute=True, runner=runner)
    if mutation:
        with pytest.raises(d.DerivedTrainingImageError): d.qualify_local_runtime(**kwargs)
        assert not output.exists()
        if mutation == "runtime_unavailable": assert count == 2  # No runtime fallback or retry.
        return
    assert d.qualify_local_runtime(**kwargs) == expected
    import jsonschema
    schema = Path(__file__).resolve().parents[2] / "schemas/synaptic-packaged-local-qualification-evidence-v1.schema.json"
    jsonschema.validate(expected, json.loads(schema.read_bytes()))
    result = d.verify_local_runtime(**inputs, evidence_path=output, output=chain["root"] / "local-verified.json")
    jsonschema.validate(result, json.loads(schema.read_bytes()))
    with pytest.raises(d.DerivedTrainingImageError, match="OUTPUT_INVALID"): d.qualify_local_runtime(**kwargs)
    changed = dict(expected, training_executed=True)
    write(output, changed)
    with pytest.raises(d.DerivedTrainingImageError, match="LOCAL_QUALIFICATION_EVIDENCE_INVALID"):
        d.verify_local_runtime(**inputs, evidence_path=output, output=chain["root"] / "bad-local.json")


@pytest.mark.parametrize("field", ["runtime_release_digest", "protocol", "extra"])
def test_local_cpu_config_rejects_substitution(chain, field):
    inputs = local_inputs(chain)
    config = json.loads(inputs["qualification_config_path"].read_bytes())
    config[field] = "untrusted"
    write(inputs["qualification_config_path"], config)
    with pytest.raises(d.DerivedTrainingImageError): d._local_qualification_inputs(**inputs)


def test_complete_non_circular_chain_and_exclusive_outputs(chain):
    release = build(chain)
    report_path = chain["root"] / "release-verification.json"
    d.verify_runtime_release(**chain["evidence"], release_path=chain["root"] / "release.json", capture_path=chain["capture_path"], output=report_path)
    result = d.promote_runtime_release(**chain["evidence"], release_path=chain["root"] / "release.json", verification_path=report_path, final_runtime_capture_path=chain["capture_path"], output=chain["root"] / "promotion.json")
    assert result["runtime_release_digest"] == release.manifest_digest
    assert result["status"] == "PROMOTED_LOCAL_ONLY"
    assert len(chain["calls"]) == 3
    before = (chain["root"] / "release.json").read_bytes()
    with pytest.raises(d.DerivedTrainingImageError, match="OUTPUT_INVALID"): build(chain)
    assert (chain["root"] / "release.json").read_bytes() == before


@pytest.mark.parametrize("field", ["image-reference", "image-digest", "python", "package", "closure", "capabilities", "inventory", "provenance", "platform"])
def test_capture_substitution_rejected_at_build_verify_and_promote(chain, field):
    build(chain)
    report = chain["root"] / "verified.json"
    d.verify_runtime_release(**chain["evidence"], release_path=chain["root"] / "release.json", capture_path=chain["capture_path"], output=report)
    capture = json.loads(chain["capture_path"].read_bytes())
    if field == "image-reference": capture["image"]["reference"] = "registry.example/other/runtime@sha256:" + "2" * 64
    elif field == "image-digest": capture["image"]["digest"] = "0" * 64
    elif field == "python": capture["measured"][field]["executable"] = "/usr/bin/python"
    elif field == "package": capture["measured"][field]["digest"] = "0" * 64
    elif field == "closure": capture["measured"][field]["verified_digest"] = "0" * 64
    elif field == "capabilities": capture["measured"][field]["compatibility"]["methods"] = ["sft", "kto"]
    elif field == "inventory": capture["measured"]["installed_distributions"]["count"] = 3
    elif field == "provenance": capture["measured"][field]["synaptic-tuner"] = hashlib.sha256(b"").hexdigest()
    else: capture["measured"][field]["machine"] = "aarch64"
    write(chain["capture_path"], capture)
    for action in (
        lambda: build(chain, "bad-release.json"),
        lambda: d.verify_runtime_release(**chain["evidence"], release_path=chain["root"] / "release.json", capture_path=chain["capture_path"], output=chain["root"] / "bad-verification.json"),
        lambda: d.promote_runtime_release(**chain["evidence"], release_path=chain["root"] / "release.json", verification_path=report, final_runtime_capture_path=chain["capture_path"], output=chain["root"] / "bad-promotion.json"),
    ):
        with pytest.raises(d.DerivedTrainingImageError): action()


def test_capability_overclaim_rejected(chain):
    compatibility = json.loads(chain["compatibility_path"].read_bytes())
    compatibility["compatibility"]["models"].append({"ref": "org/other", "revision": "b" * 40})
    write(chain["compatibility_path"], compatibility)
    with pytest.raises(d.DerivedTrainingImageError, match="CAPABILITY_MISMATCH"): build(chain)


def test_buildkit_artifact_binding_required(chain):
    path = chain["evidence"]["build_receipt_path"]
    receipt = json.loads(path.read_bytes())
    receipt["build_metadata"].pop("buildx.build.provenance")
    receipt["build_metadata_sha256"] = d._sha256(d._canonical_bytes(receipt["build_metadata"]))
    write(path, receipt)
    with pytest.raises(d.DerivedTrainingImageError, match="BUILD_METADATA_INVALID"): build(chain)


def test_wheel_artifact_bytes_must_match_reviewed_hash(chain):
    profile = chain["profile"]
    (chain["root"] / profile.packaged_runtime["wheel"]["filename"]).write_bytes(b"substitution")
    context = chain["root"] / "context"
    context.mkdir()
    with pytest.raises(d.DerivedTrainingImageError, match="WHEEL_ARTIFACT_INVALID"):
        d._stage_packaged_inputs(profile, context)


def test_stable_read_bounded_hardlink_and_mutation(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    source.write_bytes(b"12345678")
    with pytest.raises(OSError): stable_read(source, 4)
    alias = tmp_path / "alias.json"
    os.link(source, alias)
    with pytest.raises(OSError): stable_read(source)
    # New inode for the mutation test; leave the hardlinked fixture intact.
    source = tmp_path / "mutable.json"
    source.write_bytes(b"12345678")
    read = os.read
    def mutate(fd, size):
        result = read(fd, size)
        source.write_bytes(b"87654321")
        return result
    monkeypatch.setattr(os, "read", mutate)
    with pytest.raises(OSError): stable_read(source)


def test_inspector_contains_exact_interpreter_and_installed_closure_entrypoint(chain):
    script = d._final_runtime_inspector(chain["profile"])
    compile(script, "<inspector>", "exec")
    assert "sysconfig.get_paths()" in script
    assert "inspect_installed_runtime(expected)" in script
    assert "os.path.realpath(sys.executable)" in script
    assert "source.read(67108865)" in script


def test_packaged_diagnostic_report_retains_exact_measurements(chain):
    image = json.loads(chain["evidence"]["image_verification_path"].read_bytes())
    validated = d.validate_verification_report(profile_path=chain["evidence"]["profile_path"], verification_report_path=chain["evidence"]["image_verification_path"], image=image["final_oci_reference"])
    assert validated["final_runtime"] == chain["measured"]


def test_exact_wheels_and_bootstrap_are_staged_without_install(chain):
    from dataclasses import replace
    packaged = copy.deepcopy(chain["profile"].packaged_runtime)
    for item in [packaged["wheel"], *packaged["bootstrap"]]:
        raw = ("reviewed fixture " + item["distribution"]).encode()
        (chain["root"] / item["filename"]).write_bytes(raw)
        item["sha256"] = hashlib.sha256(raw).hexdigest()
    profile = replace(chain["profile"], packaged_runtime=packaged)
    context = chain["root"] / "staged"
    context.mkdir()
    d._stage_packaged_inputs(profile, context)
    assert json.loads((context / "runtime" / "build-inputs.json").read_bytes()) == packaged
    requirements = (context / "runtime" / "requirements.txt").read_text()
    for item in [packaged["wheel"], *packaged["bootstrap"]]:
        assert "--hash=sha256:" + item["sha256"] in requirements
    dockerfile = d.render_dockerfile(profile)
    assert "--require-hashes" in dockerfile and "--no-index" in dockerfile
    assert dockerfile.count("-I -m pip check 2>&1") == 2
    assert dockerfile.index("before=") < dockerfile.index("-I -m pip install") < dockerfile.index("after=")
    assert 'test "$before" = "$after"' in dockerfile


def test_interpreter_substitution_stops_before_package_import(chain):
    import subprocess
    import sys
    # The current host interpreter cannot satisfy this deliberately different
    # reviewed Linux identity. No image or installation is involved.
    result = subprocess.run([sys.executable, "-I", "-S", "-c", d._final_runtime_inspector(chain["profile"])], capture_output=True, timeout=10)
    assert result.returncode != 0 and b"INTERPRETER_INVALID" in result.stderr
    assert not result.stdout
