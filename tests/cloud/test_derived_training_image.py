from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from tuner.cloud import derived_training_image as derived_image
from tuner.cloud.derived_training_image import (
    CANDIDATE_SCHEMA,
    DerivedTrainingImageError,
    VERIFICATION_SCHEMA,
    build_command,
    build_derived_image,
    capture_candidate,
    load_profile,
    plan_profile,
    render_dockerfile,
    validate_verification_report,
    verify_candidate,
    verify_effectful_launch,
)
from tuner.cloud.hf_training_image_lock import CommandResult, CommandSpec


BASE = "docker.io/unsloth/unsloth@sha256:" + "1" * 64
FINAL = "registry.example/syntunia/trainer@sha256:" + "2" * 64


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _profile(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "synaptic-derived-training-image-profile/v1",
                "name": "generic-trainer",
                "platform": "linux/amd64",
                "base_image": BASE,
                "python_executable": "/opt/conda/bin/python3",
                "packages": [
                    {
                        "distribution": "transformers",
                        "source": "index",
                        "version": "5.5.0",
                    },
                    {
                        "distribution": "unsloth",
                        "source": "vcs",
                        "url": "https://github.com/unslothai/unsloth.git",
                        "commit": "a" * 40,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _candidate(path: Path, profile_path: Path) -> Path:
    profile = load_profile(profile_path)
    build_metadata = {
        "containerimage.digest": "sha256:" + "2" * 64,
        "containerimage.config.digest": "sha256:" + "3" * 64,
    }
    runtime = {
        "schema_version": "synaptic-derived-training-image-runtime/v1",
        "packages": {
            "transformers": {"version": "5.5.0", "direct_url": None},
            "unsloth": {
                "version": "2026.9.1",
                "direct_url": {
                    "url": "https://github.com/unslothai/unsloth.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "a" * 40,
                        "requested_revision": "a" * 40,
                    },
                },
            },
        },
    }
    value = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "CANDIDATE_ONLY",
        "profile_sha256": profile.canonical_sha256,
        "base_image": BASE,
        "final_oci_reference": FINAL,
        "final_oci_digest": "sha256:" + "2" * 64,
        "image_config_digest": "sha256:" + "3" * 64,
        "repo_digests": [FINAL],
        "build_receipt_sha256": "sha256:" + "4" * 64,
        "dockerfile_sha256": _digest(render_dockerfile(profile).encode("utf-8")),
        "build_metadata": build_metadata,
        "build_metadata_sha256": _digest(
            (json.dumps(build_metadata, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ),
        "runtime": runtime,
        "runtime_bytes_sha256": _digest(
            (json.dumps(runtime, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ),
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_plan_is_deterministic_and_uses_only_pinned_sources(tmp_path: Path) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    profile = load_profile(profile_path)
    first = plan_profile(profile_path)
    second = plan_profile(profile_path)

    assert first == second
    assert first["status"] == "PLAN_ONLY"
    dockerfile = render_dockerfile(profile)
    assert f"FROM {BASE}" in dockerfile
    assert "transformers==5.5.0" in dockerfile
    assert "git+https://github.com/unslothai/unsloth.git@" + "a" * 40 in dockerfile
    assert ":latest" not in dockerfile


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_image", "docker.io/unsloth/unsloth:latest"),
        ("commit", "main"),
    ],
)
def test_profile_rejects_floating_inputs(
    tmp_path: Path, field: str, value: str
) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if field == "base_image":
        document[field] = value
    else:
        document["packages"][1][field] = value
    profile_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(DerivedTrainingImageError):
        load_profile(profile_path)


def test_build_command_is_explicit_buildx_load_without_push(tmp_path: Path) -> None:
    spec = build_command(
        docker=tmp_path / "docker",
        docker_config=tmp_path / "config",
        tag="registry.example/syntunia/trainer:candidate",
        dockerfile=tmp_path / "Dockerfile",
        metadata_file=tmp_path / "metadata.json",
    )

    assert spec.argv[3:6] == ("buildx", "build", "--pull")
    assert "--load" in spec.argv
    assert "--push" not in spec.argv
    assert "--metadata-file" in spec.argv


def test_windows_build_environment_does_not_inherit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker = tmp_path / "docker-install" / "docker.exe"
    monkeypatch.setattr(derived_image, "_WINDOWS", True)
    monkeypatch.setattr(derived_image, "_PATH_SEPARATOR", ";")
    monkeypatch.setenv("PATH", "UNTRUSTED-AMBIENT-PATH")

    spec = build_command(
        docker=docker,
        docker_config=tmp_path / "config",
        tag="registry.example/syntunia/trainer:candidate",
        dockerfile=tmp_path / "Dockerfile",
        metadata_file=tmp_path / "metadata.json",
    )

    assert spec.env == {
        "BUILDX_METADATA_PROVENANCE": "max",
        "DOCKER_CONTENT_TRUST": "1",
        "PATH": derived_image.os.defpath,
    }
    assert "UNTRUSTED-AMBIENT-PATH" not in spec.env["PATH"]


def test_non_windows_build_environment_remains_closed_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(derived_image, "_WINDOWS", False)
    monkeypatch.setenv("PATH", "UNTRUSTED-AMBIENT-PATH")

    spec = build_command(
        docker=tmp_path / "docker",
        docker_config=tmp_path / "config",
        tag="registry.example/syntunia/trainer:candidate",
        dockerfile=tmp_path / "Dockerfile",
        metadata_file=tmp_path / "metadata.json",
    )

    assert spec.env["PATH"] == derived_image.os.defpath
    assert "UNTRUSTED-AMBIENT-PATH" not in spec.env["PATH"]


def test_windows_build_environment_rejects_path_separator_in_docker_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(derived_image, "_WINDOWS", True)
    monkeypatch.setattr(derived_image, "_PATH_SEPARATOR", ";")

    with pytest.raises(DerivedTrainingImageError, match="DOCKER_INVALID"):
        build_command(
            docker=tmp_path / "bad;directory" / "docker.exe",
            docker_config=tmp_path / "config",
            tag="registry.example/syntunia/trainer:candidate",
            dockerfile=tmp_path / "Dockerfile",
            metadata_file=tmp_path / "metadata.json",
        )


def test_build_and_capture_bind_buildkit_digest_and_runtime_provenance(
    tmp_path: Path,
) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    profile = load_profile(profile_path)
    resources = tmp_path / "docker-install" / "resources"
    docker = resources / "bin" / "docker.exe"
    docker.parent.mkdir(parents=True)
    docker.write_bytes(b"docker")
    buildx = resources / "cli-plugins" / "docker-buildx.exe"
    buildx.parent.mkdir()
    buildx.write_bytes(b"buildx")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    final_digest = "sha256:" + "2" * 64
    config_digest = "sha256:" + "3" * 64

    calls: list[tuple[str, ...]] = []
    private_build_configs: list[Path] = []

    def runner(spec: CommandSpec) -> CommandResult:
        calls.append(spec.argv)
        if "buildx" in spec.argv:
            command_config = Path(spec.argv[spec.argv.index("--config") + 1])
            private_build_configs.append(command_config)
            assert command_config != docker_config
            assert json.loads(
                (command_config / "config.json").read_text(encoding="ascii")
            ) == {"cliPluginsExtraDirs": [str(buildx.parent.resolve())]}
            assert spec.env["PATH"] == derived_image.os.defpath
            metadata = Path(spec.argv[spec.argv.index("--metadata-file") + 1])
            metadata.write_text(
                json.dumps(
                    {
                        "containerimage.digest": final_digest,
                        "containerimage.config.digest": config_digest,
                    }
                ),
                encoding="utf-8",
            )
            return CommandResult()
        if "inspect" in spec.argv:
            return CommandResult(
                stdout=(
                    json.dumps(config_digest)
                    + "\n[]\n"
                    + json.dumps(
                        {
                            "ai.synapticlabs.derived-training-profile": profile.canonical_sha256,
                            "ai.synapticlabs.derived-training-base": BASE,
                        }
                    )
                ).encode()
            )
        assert "run" in spec.argv
        entrypoint = spec.argv.index("--entrypoint")
        assert spec.argv[entrypoint + 2] == config_digest
        assert spec.argv[entrypoint + 3 : entrypoint + 6] == ("-I", "-S", "-c")
        return CommandResult(
            stdout=(json.dumps(
                {
                    "schema_version": "synaptic-derived-training-image-runtime/v1",
                    "packages": {
                        "transformers": {"version": "5.5.0", "direct_url": None},
                        "unsloth": {
                            "version": "2026.9.1",
                            "direct_url": {
                                "url": "https://github.com/unslothai/unsloth.git",
                                "vcs_info": {
                                    "vcs": "git",
                                    "commit_id": "a" * 40,
                                    "requested_revision": "a" * 40,
                                },
                            },
                        },
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n").encode()
        )

    receipt_path = tmp_path / "receipt.json"
    receipt = build_derived_image(
        profile_path=profile_path,
        docker=docker,
        docker_config=docker_config,
        tag="registry.example/syntunia/trainer:candidate",
        output=receipt_path,
        runner=runner,
    )
    assert receipt["final_oci_reference"] == (
        "registry.example/syntunia/trainer@" + final_digest
    )
    assert receipt["image_config_digest"] == config_digest
    assert private_build_configs and not private_build_configs[0].exists()
    assert not any(docker_config.iterdir())

    candidate = capture_candidate(
        profile_path=profile_path,
        build_receipt_path=receipt_path,
        docker=docker,
        docker_config=docker_config,
        output=tmp_path / "candidate.json",
        runner=runner,
    )
    assert candidate["final_oci_digest"] == final_digest
    assert candidate["runtime"]["packages"]["unsloth"]["direct_url"]["vcs_info"][
        "commit_id"
    ] == "a" * 40
    capture_inspects = [
        call[-1] for call in calls if "inspect" in call
    ]
    assert capture_inspects[-2:] == [
        "registry.example/syntunia/trainer:candidate",
        config_digest,
    ]


def test_windows_build_rejects_buildx_replacement_during_command(
    tmp_path: Path,
) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    resources = tmp_path / "docker-install" / "resources"
    docker = resources / "bin" / "docker.exe"
    docker.parent.mkdir(parents=True)
    docker.write_bytes(b"docker")
    buildx = resources / "cli-plugins" / "docker-buildx.exe"
    buildx.parent.mkdir()
    buildx.write_bytes(b"buildx")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()

    def runner(_spec: CommandSpec) -> CommandResult:
        buildx.write_bytes(b"replaced-buildx")
        return CommandResult()

    with pytest.raises(
        DerivedTrainingImageError, match="DOCKER_AUTHORITY_CHANGED"
    ):
        build_derived_image(
            profile_path=profile_path,
            docker=docker,
            docker_config=docker_config,
            tag="registry.example/syntunia/trainer:candidate",
            output=tmp_path / "receipt.json",
            runner=runner,
        )
    assert not any(docker_config.iterdir())


def test_windows_buildx_allows_transient_directory_timestamp_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(derived_image, "_WINDOWS", True)
    resources = tmp_path / "docker-install" / "resources"
    docker = resources / "bin" / "docker.exe"
    docker.parent.mkdir(parents=True)
    docker.write_bytes(b"docker")
    buildx = resources / "cli-plugins" / "docker-buildx.exe"
    buildx.parent.mkdir()
    buildx.write_bytes(b"buildx")
    config_directory = tmp_path / "buildx-config"
    config_directory.mkdir()
    authority = derived_image._create_buildx_authority(
        docker=docker, config_directory=config_directory,
    )
    assert authority is not None
    before = config_directory.stat().st_mtime_ns
    os.utime(config_directory, ns=(before + 1_000_000_000, before + 1_000_000_000))
    (config_directory / "buildx").mkdir()
    (config_directory / ".token_seed").write_bytes(b"buildx-owned-seed")
    (config_directory / ".token_seed.lock").write_bytes(b"")
    derived_image._assert_buildx_authority(authority)
    (config_directory / "unexpected").write_bytes(b"unexpected")
    with pytest.raises(DerivedTrainingImageError, match="DOCKER_AUTHORITY_CHANGED"):
        derived_image._assert_buildx_authority(authority)


def test_verify_emits_diagnostic_report_with_vcs_commit_provenance(tmp_path: Path) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    candidate_path = _candidate(tmp_path / "candidate.json", profile_path)
    output = tmp_path / "verification.json"

    verification = verify_candidate(
        profile_path=profile_path, candidate_path=candidate_path, output=output
    )

    assert verification["schema_version"] == VERIFICATION_SCHEMA
    assert verification["status"] == "DIAGNOSTIC_PASS"
    assert verification["final_oci_reference"] == FINAL
    assert (
        verification["packages"]["unsloth"]["direct_url"]["vcs_info"]["commit_id"]
        == "a" * 40
    )
    assert validate_verification_report(
        profile_path=profile_path, verification_report_path=output, image=FINAL
    ) == verification
    assert validate_verification_report(
        profile_path=profile_path,
        verification_report_path=output,
        image="sha256:" + "3" * 64,
    ) == verification


def test_verify_rejects_wrong_commit_and_launch_image(tmp_path: Path) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    candidate_path = _candidate(tmp_path / "candidate.json", profile_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["runtime"]["packages"]["unsloth"]["direct_url"]["vcs_info"][
        "commit_id"
    ] = "b" * 40
    candidate["runtime_bytes_sha256"] = _digest(
        (
            json.dumps(candidate["runtime"], sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    )
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(DerivedTrainingImageError, match="PACKAGE_MISMATCH"):
        verify_candidate(profile_path=profile_path, candidate_path=candidate_path)

    candidate_path = _candidate(tmp_path / "candidate-2.json", profile_path)
    verification_path = tmp_path / "verification.json"
    verify_candidate(
        profile_path=profile_path,
        candidate_path=candidate_path,
        output=verification_path,
    )
    with pytest.raises(DerivedTrainingImageError, match="IMAGE_QUALIFICATION_INVALID"):
        validate_verification_report(
            profile_path=profile_path,
            verification_report_path=verification_path,
            image="registry.example/syntunia/other@sha256:" + "2" * 64,
        )


def test_capture_rejects_arbitrary_dockerfile_and_metadata_hashes(
    tmp_path: Path,
) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    profile = load_profile(profile_path)
    config_digest = "sha256:" + "3" * 64
    metadata = {
        "containerimage.digest": "sha256:" + "2" * 64,
        "containerimage.config.digest": config_digest,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "synaptic-derived-training-image-build-receipt/v1",
                "status": "BUILT_NOT_ATTESTED",
                "profile_sha256": profile.canonical_sha256,
                "base_image": BASE,
                "dockerfile_sha256": "sha256:" + "8" * 64,
                "tag": "registry.example/syntunia/trainer:candidate",
                "final_oci_reference": FINAL,
                "final_oci_digest": "sha256:" + "2" * 64,
                "image_config_digest": config_digest,
                "repo_digests": [FINAL],
                "build_metadata": metadata,
                "build_metadata_sha256": "sha256:" + "7" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DerivedTrainingImageError, match="BUILD_RECEIPT_INVALID"):
        capture_candidate(
            profile_path=profile_path,
            build_receipt_path=receipt_path,
            docker=tmp_path / "unused-docker",
            docker_config=tmp_path / "unused-config",
            output=tmp_path / "candidate.json",
            runner=lambda _spec: pytest.fail("Docker must not be called"),
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["dockerfile_sha256"] = _digest(
        render_dockerfile(profile).encode("utf-8")
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(DerivedTrainingImageError, match="BUILD_RECEIPT_INVALID"):
        capture_candidate(
            profile_path=profile_path,
            build_receipt_path=receipt_path,
            docker=tmp_path / "unused-docker",
            docker_config=tmp_path / "unused-config",
            output=tmp_path / "candidate-2.json",
            runner=lambda _spec: pytest.fail("Docker must not be called"),
        )


def test_candidate_and_report_recompute_reviewed_dockerfile_hash(
    tmp_path: Path,
) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    candidate_path = _candidate(tmp_path / "candidate.json", profile_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["dockerfile_sha256"] = "sha256:" + "8" * 64
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(DerivedTrainingImageError, match="CANDIDATE_INVALID"):
        verify_candidate(profile_path=profile_path, candidate_path=candidate_path)

    candidate_path = _candidate(tmp_path / "candidate-2.json", profile_path)
    report_path = tmp_path / "verification.json"
    verify_candidate(
        profile_path=profile_path,
        candidate_path=candidate_path,
        output=report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dockerfile_sha256"] = "sha256:" + "8" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(
        DerivedTrainingImageError, match="IMAGE_QUALIFICATION_INVALID"
    ):
        validate_verification_report(
            profile_path=profile_path,
            verification_report_path=report_path,
            image="sha256:" + "3" * 64,
        )


def _inspect_bytes(
    *, image_id: str, profile_sha256: str, repo_digests: list[str] | None = None
) -> bytes:
    return (
        json.dumps(image_id)
        + "\n"
        + json.dumps(repo_digests or [])
        + "\n"
        + json.dumps(
            {
                "ai.synapticlabs.derived-training-profile": profile_sha256,
                "ai.synapticlabs.derived-training-base": BASE,
            }
        )
    ).encode()


def _runtime_bytes(packages: dict[str, object]) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": "synaptic-derived-training-image-runtime/v1",
                "packages": packages,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def test_handwritten_diagnostic_report_cannot_bypass_live_docker(
    tmp_path: Path,
) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    profile = load_profile(profile_path)
    config_digest = "sha256:" + "3" * 64
    report_path = tmp_path / "handwritten-verification.json"
    report = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "DIAGNOSTIC_PASS",
        "profile_sha256": profile.canonical_sha256,
        "candidate_sha256": "sha256:" + "4" * 64,
        "build_receipt_sha256": "sha256:" + "5" * 64,
        "dockerfile_sha256": _digest(render_dockerfile(profile).encode("utf-8")),
        "build_metadata": {
            "containerimage.digest": "sha256:" + "2" * 64,
            "containerimage.config.digest": config_digest,
        },
        "build_metadata_sha256": "",
        "base_image": BASE,
        "final_oci_reference": FINAL,
        "final_oci_digest": "sha256:" + "2" * 64,
        "image_config_digest": config_digest,
        "packages": {
            "transformers": {"version": "5.5.0", "direct_url": None},
            "unsloth": {
                "version": "2026.9.1",
                "direct_url": {
                    "url": "https://github.com/unslothai/unsloth.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "a" * 40,
                        "requested_revision": "a" * 40,
                    },
                },
            },
        },
    }
    report["build_metadata_sha256"] = _digest(
        (
            json.dumps(
                report["build_metadata"], sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode()
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_verification_report(
        profile_path=profile_path,
        verification_report_path=report_path,
        image=config_digest,
    )["status"] == "DIAGNOSTIC_PASS"

    docker = tmp_path / "docker"
    docker.write_bytes(b"docker")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    calls = 0

    def runner(_spec: CommandSpec) -> CommandResult:
        nonlocal calls
        calls += 1
        return CommandResult(
            stdout=_inspect_bytes(
                image_id="sha256:" + "9" * 64,
                profile_sha256=profile.canonical_sha256,
            )
        )

    with pytest.raises(DerivedTrainingImageError, match="LIVE_IMAGE_MISMATCH"):
        verify_effectful_launch(
            profile_path=profile_path,
            verification_report_path=report_path,
            image=config_digest,
            docker=docker,
            docker_config=docker_config,
            runner=runner,
        )
    assert calls == 1


def test_capture_fails_if_exact_image_identity_changes_after_probe(
    tmp_path: Path,
) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    profile = load_profile(profile_path)
    docker = tmp_path / "docker"
    docker.write_bytes(b"docker")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    config_digest = "sha256:" + "3" * 64
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "synaptic-derived-training-image-build-receipt/v1",
                "status": "BUILT_NOT_ATTESTED",
                "profile_sha256": profile.canonical_sha256,
                "base_image": BASE,
                "dockerfile_sha256": _digest(
                    render_dockerfile(profile).encode("utf-8")
                ),
                "tag": "registry.example/syntunia/trainer:candidate",
                "final_oci_reference": FINAL,
                "final_oci_digest": "sha256:" + "2" * 64,
                "image_config_digest": config_digest,
                "repo_digests": [FINAL],
                "build_metadata": {
                    "containerimage.digest": "sha256:" + "2" * 64,
                    "containerimage.config.digest": config_digest,
                },
                "build_metadata_sha256": _digest(
                    (
                        json.dumps(
                            {
                                "containerimage.digest": "sha256:" + "2" * 64,
                                "containerimage.config.digest": config_digest,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                ),
            }
        ),
        encoding="utf-8",
    )
    runtime = _runtime_bytes(
        {
            "transformers": {"version": "5.5.0", "direct_url": None},
            "unsloth": {
                "version": "2026.9.1",
                "direct_url": {
                    "url": "https://github.com/unslothai/unsloth.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": "a" * 40,
                        "requested_revision": "a" * 40,
                    },
                },
            },
        }
    )
    inspect_count = 0

    def runner(spec: CommandSpec) -> CommandResult:
        nonlocal inspect_count
        if "run" in spec.argv:
            image_index = spec.argv.index("--entrypoint") + 2
            assert spec.argv[image_index] == config_digest
            return CommandResult(stdout=runtime)
        inspect_count += 1
        image_id = config_digest if inspect_count == 1 else "sha256:" + "9" * 64
        return CommandResult(
            stdout=_inspect_bytes(
                image_id=image_id,
                profile_sha256=profile.canonical_sha256,
                repo_digests=[FINAL],
            )
        )

    with pytest.raises(DerivedTrainingImageError, match="DOCKER_EVIDENCE_INVALID"):
        capture_candidate(
            profile_path=profile_path,
            build_receipt_path=receipt_path,
            docker=docker,
            docker_config=docker_config,
            output=tmp_path / "candidate.json",
            runner=runner,
        )


def test_live_verification_rejects_docker_authority_change(tmp_path: Path) -> None:
    profile_path = _profile(tmp_path / "profile.yaml")
    candidate_path = _candidate(tmp_path / "candidate.json", profile_path)
    report_path = tmp_path / "verification.json"
    verify_candidate(
        profile_path=profile_path,
        candidate_path=candidate_path,
        output=report_path,
    )
    profile = load_profile(profile_path)
    docker = tmp_path / "docker"
    docker.write_bytes(b"docker")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()

    def runner(_spec: CommandSpec) -> CommandResult:
        (docker_config / "changed").write_text("changed", encoding="utf-8")
        return CommandResult(
            stdout=_inspect_bytes(
                image_id="sha256:" + "3" * 64,
                profile_sha256=profile.canonical_sha256,
                repo_digests=[FINAL],
            )
        )

    with pytest.raises(DerivedTrainingImageError, match="DOCKER_AUTHORITY_CHANGED"):
        verify_effectful_launch(
            profile_path=profile_path,
            verification_report_path=report_path,
            image=FINAL,
            docker=docker,
            docker_config=docker_config,
            runner=runner,
        )
