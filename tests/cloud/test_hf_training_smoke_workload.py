from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tuner.cloud.hf_training_smoke_workload import (
    DATASET, DATASET_GIT_BLOB, DATASET_SHA256, RECIPE_PATH,
    TrainingSmokeWorkloadError, build_workload,
    validate_remote_argv, validate_runtime_lock,
)
from tuner.cloud.hf_training_smoke_contract import (
    RUNTIME_PYTHON_IMPLEMENTATION,
    RUNTIME_PYTHON_VERSION,
    document_sha256,
)
from tuner.cloud.hf_training_smoke_remote_entry import (
    RemoteTrainingSmokeError, _reject_remote_credentials, run as run_remote,
)


REPO = Path(__file__).resolve().parents[2]


def test_dataset_identity_is_bound_to_committed_blob_bytes() -> None:
    attributes = (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert f"{DATASET} text eol=lf" in attributes
    checked_out = (REPO / DATASET).read_bytes()
    assert b"\r" not in checked_out
    assert hashlib.sha256(checked_out).hexdigest() == DATASET_SHA256
    content = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "cat-file", "blob", f"HEAD:{DATASET}"],
        cwd=REPO,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    blob = subprocess.run(
        ["git", "hash-object", "--stdin"],
        cwd=REPO,
        check=True,
        input=content,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode("ascii").strip()
    assert hashlib.sha256(content).hexdigest() == DATASET_SHA256
    assert blob == DATASET_GIT_BLOB


def _accepted_lock(path: Path, requested_kind: str = "manifest") -> Path:
    child_digest = "sha256:" + "2" * 64
    index_digest = "sha256:" + "1" * 64 if requested_kind == "index" else None
    index_media_type = (
        "application/vnd.docker.distribution.manifest.list.v2+json"
        if requested_kind == "index"
        else None
    )
    body = {
        "anonymous_loading": {"token": False, "trust_remote_code": False, "use_safetensors": True},
        "created_at": "2026-08-20T12:00:00Z",
        "image": {
            "registry_repository": "docker.io/unsloth/unsloth",
            "provider_repository": "unsloth/unsloth",
            "requested_digest": index_digest or child_digest,
            "requested_media_type": index_media_type or "application/vnd.docker.distribution.manifest.v2+json",
            "requested_kind": requested_kind,
            "index_digest": index_digest,
            "index_media_type": index_media_type,
            "child_digest": child_digest,
            "child_media_type": "application/vnd.docker.distribution.manifest.v2+json",
            "config_digest": "sha256:" + "3" * 64,
            "config_media_type": "application/vnd.docker.container.image.v1+json",
            "config_size": 123,
            "platform": "linux/amd64",
            "layers": [{
                "media_type": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                "digest": "sha256:" + "4" * 64,
                "size": 456,
            }],
            "provider_reference": f"unsloth/unsloth@{child_digest}",
        },
        "runtime": {
            "python_implementation": RUNTIME_PYTHON_IMPLEMENTATION,
            "packages": {"torch": "2.9.0"}, "python": RUNTIME_PYTHON_VERSION,
            "signatures": {"FastLanguageModel.from_pretrained": "revision,token,trust_remote_code,use_safetensors"},
        },
        "schema_version": "synaptic-hf-training-runtime-lock/v1",
    }
    payload = {**body, "lock_id": document_sha256(body)}
    path.write_bytes((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"))
    return path


def test_checked_in_runtime_lock_is_exact_promoted_contract() -> None:
    path = REPO / "Trainers/cloud/runtime-locks/hf_training_smoke_unsloth_2026_1_2.json"
    lock, content = validate_runtime_lock(path)
    assert lock["lock_id"] == "191f5df364a3257da02986921bb095b4471f3422cd78a715670bd90c98738879"
    assert lock["runtime"]["python_implementation"] == RUNTIME_PYTHON_IMPLEMENTATION
    assert lock["runtime"]["python"] == RUNTIME_PYTHON_VERSION
    assert hashlib.sha256(content).hexdigest() == (
        "47b62bb0f1bff1d79b8b3ac2ca314fad268080da60419d0214c997e91f655060"
    )

    arguments = {"source_lock_sha256": "6" * 64, "artifact_slot": "7" * 64}
    first = build_workload(REPO, **arguments)
    second = build_workload(REPO, **arguments)
    assert first == second
    assert first.runtime_lock_sha256 == hashlib.sha256(content).hexdigest()
    assert first.image == lock["image"]["provider_reference"]


@pytest.mark.parametrize("requested_kind", ["manifest", "index"])
def test_builds_constant_ordered_argv_from_reviewed_lock(
    tmp_path: Path, requested_kind: str
) -> None:
    lock = _accepted_lock(tmp_path / "lock.json", requested_kind)
    workload = build_workload(REPO, source_lock_sha256="6" * 64, artifact_slot="7" * 64, runtime_lock_path=lock)
    assert workload.image == "unsloth/unsloth@sha256:" + "2" * 64
    assert workload.argv[::2] == (
        "--recipe", "--recipe-sha256", "--runtime-lock", "--runtime-lock-sha256",
        "--source-lock", "--source-lock-sha256", "--artifact-root", "--artifact-slot",
        "--project-root", "--engine-root",
    )
    assert hashlib.sha256((REPO / RECIPE_PATH).read_bytes()).hexdigest() == workload.recipe_sha256
    assert workload.workload_sha256 != workload.runtime_lock_sha256


def test_workload_rejects_provider_reference_drift_without_fallback(tmp_path: Path) -> None:
    lock = _accepted_lock(tmp_path / "lock.json")
    payload = json.loads(lock.read_text(encoding="ascii"))
    payload["image"]["provider_reference"] = "unsloth/unsloth@sha256:" + "9" * 64
    body = {key: value for key, value in payload.items() if key != "lock_id"}
    payload["lock_id"] = document_sha256(body)
    lock.write_bytes((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"))
    with pytest.raises(TrainingSmokeWorkloadError, match="reviewed canonical contract"):
        build_workload(
            REPO,
            source_lock_sha256="6" * 64,
            artifact_slot="7" * 64,
            runtime_lock_path=lock,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda argv: argv[2:] + argv[:2],
        lambda argv: ["--recipe=x", *argv[1:]],
        lambda argv: [*argv, "--recipe", "x"],
    ],
)
def test_rejects_reordering_equals_and_duplicate_options(mutation) -> None:
    good = []
    for index, option in enumerate((
        "--recipe", "--recipe-sha256", "--runtime-lock", "--runtime-lock-sha256",
        "--source-lock", "--source-lock-sha256", "--artifact-root", "--artifact-slot",
        "--project-root", "--engine-root",
    )):
        good.extend((option, str(index)))
    with pytest.raises(TrainingSmokeWorkloadError):
        validate_remote_argv(mutation(good))


def test_rejects_tampered_recipe_copy(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    destination = repository / RECIPE_PATH
    destination.parent.mkdir(parents=True)
    destination.write_bytes((REPO / RECIPE_PATH).read_bytes() + b"\n")
    with pytest.raises(TrainingSmokeWorkloadError, match="recipe bytes"):
        build_workload(repository, source_lock_sha256="6" * 64, artifact_slot="7" * 64, runtime_lock_path=_accepted_lock(tmp_path / "lock.json"))


def test_remote_entry_rejects_credentials_without_echoing_value(monkeypatch) -> None:
    secret = "do-not-echo-this-secret"
    monkeypatch.setenv("HF_TOKEN", secret)
    with pytest.raises(RemoteTrainingSmokeError) as caught:
        _reject_remote_credentials()
    assert secret not in str(caught.value)


def test_remote_entry_rejects_noncanonical_argv_before_filesystem_access() -> None:
    with pytest.raises(RemoteTrainingSmokeError, match="arguments are invalid"):
        run_remote(["--recipe=hostile"])


def test_provider_command_and_remote_argv_hashes_are_deterministic(tmp_path: Path) -> None:
    lock = _accepted_lock(tmp_path / "lock.json")
    from tuner.cloud.hf_volume_transport import HFVerifiedVolumeSpec
    spec = HFVerifiedVolumeSpec(
        source="owner/source", capsule_path="capsule", capsule_manifest_sha256="1" * 64,
        source_lock_path="source-lock.json", source_lock_sha256="6" * 64,
        checkout_policy_path="checkout-policy.json", checkout_policy_sha256="2" * 64,
        local_root=tmp_path, path="prefix",
    )
    workload = build_workload(
        REPO, source_lock_sha256="6" * 64, artifact_slot="7" * 64,
        runtime_lock_path=lock, source_volume_spec=spec,
        expected_project_root="/workspace/source/project",
        expected_engine_root="/workspace/source/engine",
        expected_project_commit="3" * 40, expected_engine_commit="4" * 40,
        expected_mode="dual_clone",
    )
    assert workload.provider_command
    assert workload.remote_argv == workload.argv
    assert workload.remote_argv_sha256 == hashlib.sha256(
        (json.dumps(list(workload.remote_argv), sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    ).hexdigest()
    assert workload.provider_command_sha256 == hashlib.sha256(
        (json.dumps(list(workload.provider_command), sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    ).hexdigest()
