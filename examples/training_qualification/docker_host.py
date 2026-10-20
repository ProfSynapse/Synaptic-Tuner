"""Host-local Docker composition using existing typed material and ports."""

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time

from launch import read_committed, _write_bytes, _json, SFT_ROLES
from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.providers import ProviderRef, ProviderDescriptor
from synaptic_tuner.api.v1.planning import TrainingPlanBasisV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.project.execution_source import ExecutionSourceV1
from tuner.training import default_recipe_registry
from tuner.training.contracts import (
    TrainingPlan,
    CanonicalDocument,
    RuntimeSpec,
    ResourceSpec,
    ArtifactPolicy,
    AcceleratorDeviceRequestV1,
)
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
from tuner.training.docker_host_composition import compose_docker_reference_host
from tuner.execution.providers.docker_provider_v1.preparation import (
    DockerTrainingPreparationBridgeV1,
)
from tuner.execution.providers.docker_provider_v1.local_source import _Root
from tuner.execution.providers.modal.mounted_io import hash_regular
from tuner.execution.providers.docker_provider_v1.local_artifacts import (
    DockerLocalArtifactsV1,
    _DIR,
)
from tuner.execution.providers.docker_provider_v1.model import (
    DockerProfileV1,
    DockerImageV1,
    DockerRuntimeV1,
    DockerWorkloadV1,
    DockerRootsV1,
    DockerArtifactContractV1,
)
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.execution.foundation_v2.executors import (
    ExecutorDescriptorV1,
    AdapterDescriptorV1,
)
from tuner.runtime.dispatch import (
    WorkerControlLocationV1,
    CanonicalWorkloadFileLocationV1,
    build_source_worker_invocation,
    materialize_worker_bundle,
)
from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest
from tuner.runtime.packaged_worker_closure import stable_read
from tuner.dataset_prep import LocalPreparedTrainingInputSource
from tuner.cloud.derived_training_image import verify_effectful_launch
from tuner.cloud.hf_training_image_lock import subprocess_runner


def _profile(raw):
    value = _json(raw)
    runtime = value["runtime"]
    devices = runtime["accelerator_devices"]
    profile = DockerProfileV1.build(
        provider=ProviderRef.from_dict(value["provider"]),
        descriptor=ProviderDescriptor.from_dict(value["descriptor"]),
        scope=ExecutionScopeV1(**value["scope"]),
        executor_descriptor=ExecutorDescriptorV1(**value["executor_descriptor"]),
        adapter_descriptor=AdapterDescriptorV1(**value["adapter_descriptor"]),
        image=DockerImageV1(**value["image"]),
        runtime=DockerRuntimeV1(
            **{
                **runtime,
                "accelerator_devices": AcceleratorDeviceRequestV1(
                    devices["kind"],
                    tuple(devices["device_indices"]),
                    tuple(devices["capabilities"]),
                ),
            }
        ),
        workload=DockerWorkloadV1(
            tuple(value["workload"]["arguments"]),
            tuple(value["workload"]["environment_keys"]),
            value["workload"]["workload_digest"],
        ),
        roots=DockerRootsV1(**value["roots"]),
        artifacts=DockerArtifactContractV1(
            tuple(value["artifacts"]["roles"]),
            value["artifacts"]["maximum_artifact_bytes"],
            value["artifacts"]["maximum_total_bytes"],
        ),
        resource_digest=value["resource_digest"],
        quote_digest=value["quote_digest"],
        secret_requirements_digest=value["secret_requirements_digest"],
    )
    if (
        profile.to_dict() != value
        or set(profile.artifacts.roles) != SFT_ROLES
        or profile.provider.provider_id != "docker"
    ):
        raise ValueError("exact native SFT Docker profile required")
    return profile


def model_inventory(artifact_root, snapshot):
    """Hash the complete private offline snapshot through bounded regular reads."""
    artifact_root = Path(artifact_root)
    snapshot = Path(snapshot)
    with ExitStack() as stack:
        guard = _Root(artifact_root)
        stack.callback(guard.stack.close)
        guard.check()
        if not snapshot.is_relative_to(artifact_root) or ".." in snapshot.parts:
            raise ValueError("docker_model_snapshot_outside_artifact_root")
        rows = []
        total = 0
        parent = guard.fd
        chain = []
        for part in snapshot.relative_to(artifact_root).parts:
            child = os.open(part, _DIR, dir_fd=parent)
            stack.callback(os.close, child)
            chain.append((parent, part, child))
            parent = child
        pending = [(snapshot, parent)]
        seen = 0
        directories = 0
        while pending:
            directory, directory_fd = pending.pop()
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    seen += 1
                    if seen > 20_000:
                        raise ValueError("docker_model_inventory_bound")
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode) and not entry.is_symlink():
                        directories += 1
                        if directories > 256:
                            raise ValueError("docker_model_inventory_bound")
                        child = os.open(entry.name, _DIR, dir_fd=directory_fd)
                        stack.callback(os.close, child)
                        chain.append((directory_fd, entry.name, child))
                        pending.append((directory / entry.name, child))
                    elif stat.S_ISREG(info.st_mode) and not entry.is_symlink():
                        if len(rows) >= 20_000 or total + info.st_size > 1 << 40:
                            raise ValueError("docker_model_inventory_bound")
                        path = directory / entry.name
                        size, digest = hash_regular(artifact_root, path, info.st_size)
                        if size != info.st_size:
                            raise ValueError("docker_model_member_changed")
                        rows.append({"path": path.relative_to(snapshot).as_posix(),
                                     "size_bytes": size, "sha256": digest})
                        total += size
                    else:
                        raise ValueError("docker_model_member_invalid")
        DockerLocalArtifactsV1._recheck(chain)
        guard.check()
        if not rows:
            raise ValueError("docker_model_snapshot_empty")
        return sorted(rows, key=lambda item: item["path"])


def verify_model_receipt(raw, *, selected, model, configuration_sha256,
                         runtime_declaration_sha256):
    """Bind fresh prepared bytes to the later public consumer's exact inputs."""
    from Trainers.sft.runtime_v1 import RuntimeRoots, _require_local_model_snapshot

    receipt = _json(raw)
    artifact_root = Path(selected["artifact_host_root"])
    cache = artifact_root / "run/cache"
    expected = (cache / "model" / ("models--" + model["ref"].replace("/", "--"))
                / "snapshots" / model["revision"])
    if (set(receipt) != {"schema_version", "model", "configuration_sha256",
                         "runtime_declaration_sha256", "artifact_host_root",
                         "snapshot", "files"}
            or receipt["schema_version"] != "synaptic-docker-model-preparation/v1"
            or receipt["model"] != model
            or receipt["configuration_sha256"] != configuration_sha256
            or receipt["runtime_declaration_sha256"] != runtime_declaration_sha256
            or receipt["artifact_host_root"] != str(artifact_root)
            or receipt["snapshot"] != str(expected)
            or not isinstance(receipt["files"], list)):
        raise ValueError("docker_model_receipt_mismatch")
    roots = RuntimeRoots(
        engine=Path(selected["source_host_root"]) / "engine",
        project=Path(selected["source_host_root"]) / "project",
        artifacts=artifact_root / "run/artifacts", state=artifact_root / "run/state",
        tracking=artifact_root / "run/tracking", cache=cache,
        tmp=artifact_root / "run/tmp",
    )
    _require_local_model_snapshot(model, roots, {
        "SYNAPTIC_CACHE_ROOT": str(cache), "SYNAPTIC_MODEL_SNAPSHOT": str(expected),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    if model_inventory(artifact_root, expected) != receipt["files"]:
        raise ValueError("docker_model_receipt_bytes_changed")
    return receipt


def _components(config):
    if config["provider"] != "docker":
        raise ValueError("Docker provider required")
    selected = config["docker"]
    fields = {
        "profile_file",
        "material_file",
        "image_profile_file",
        "image_verification_file",
        "executable",
        "config_directory",
        "endpoint",
        "daemon_id",
        "source_host_root",
        "source_container_root",
        "artifact_host_root",
        "artifact_container_root",
        "control_container_root",
        "prepared_publication",
        "role_paths",
        "command_seconds",
        "read_seconds",
    }
    if set(selected) - {"allow_current_cancel_authorization", "model_preparation_file"} != fields:
        raise ValueError("Docker configuration fields invalid")
    profile = _profile(read_committed(selected["profile_file"]))
    recipes = default_recipe_registry()
    material = CoordinatorResolvedMaterial.parse(
        read_committed(selected["material_file"]), recipes
    )
    document = json.loads(material.canonical_bytes)
    if document["workload"]["schema_version"] != "synaptic-sft-workload/v1":
        raise ValueError("only compiled SFT supported")
    rich = TrainingPlan(
        ExecutionSourceV1.from_dict(document["execution_source"]),
        CanonicalDocument.from_mapping(document["execution_context"]),
        CanonicalDocument.from_mapping(document["resolved_config"]),
        CanonicalDocument.from_mapping(document["workload"]),
        RuntimeSpec(**document["runtime"]),
        ResourceSpec(**document["resources"]),
        ArtifactPolicy(
            tuple(document["artifact_policy"]["required_kinds"]),
            document["artifact_policy"]["retain_checkpoints"],
        ),
    )
    if "model_preparation_file" in selected:
        verify_model_receipt(
            read_committed(selected["model_preparation_file"]),
            selected=selected, model=rich.resolved_config.to_dict()["model"],
            configuration_sha256=hashlib.sha256(
                rich.resolved_config.canonical_json.encode()).hexdigest(),
            runtime_declaration_sha256=hashlib.sha256(
                rich.execution_context.canonical_json.encode()).hexdigest(),
        )
    checked, resolved = DockerTrainingPreparationBridgeV1(profile).coordinator_material(
        rich, material, recipes
    )
    if (
        checked.canonical_bytes != material.canonical_bytes
        or read_committed(config["request_file"]) != material.request_bytes
    ):
        raise ValueError("request/material binding mismatch")
    return profile, rich, material, recipes, TrainingPlanBasisV1.from_resolved(resolved)


def preview(config):
    """No credentials, daemon, input lease, download or operational preflight."""
    profile, rich, material, _, basis = _components(config)
    return {
        "basis": basis.to_dict(),
        "provider": profile.provider.to_dict(),
        "run": TrainingRunRef(material.run_id, basis.project_ref).to_dict(),
        "material_digest": hashlib.sha256(material.canonical_bytes).hexdigest(),
        "profile_digest": profile.profile_digest,
        "public_preflight": "deferred_until_host_composition",
    }


def retain_inputs(config, root):
    # Preserve the existing canonical documents, never serialize a new rich type.
    for key, name in (
        ("material_file", "docker-material.json"),
        ("profile_file", "docker-profile.json"),
    ):
        raw = read_committed(config["docker"][key])
        target = root / name
        _write_bytes(target, raw)
        config["docker"][key] = {
            "path": str(target),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    raw = read_committed(config["request_file"])
    target = root / "canonical-request.json"
    _write_bytes(target, raw)
    config["request_file"] = {
        "path": str(target),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    return config


def selected_runner(endpoint, runner=subprocess_runner):
    """Select the same Unix daemon for image qualification and actual session."""
    if (
        type(endpoint) is not str
        or not endpoint.startswith("unix:///")
        or "\x00" in endpoint
    ):
        raise ValueError("explicit local Unix endpoint required")

    def run(spec):
        # Preserve the authoritative executable/config prefix; override context.
        if len(spec.argv) < 3 or spec.argv[1] != "--config":
            raise ValueError("unexpected Docker command shape")
        if "--host" in spec.argv or "-H" in spec.argv:
            raise ValueError("ambiguous Docker connection")
        return runner(
            replace(spec, argv=(*spec.argv[:3], "--host", endpoint, *spec.argv[3:]))
        )

    return run


def _require_runtime_root_platform():
    if (
        os.name != "posix"
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or any(
            function not in os.supports_dir_fd
            for function in (os.open, os.mkdir, os.stat)
        )
        or os.stat not in os.supports_follow_symlinks
    ):
        raise ValueError("docker_runtime_roots_platform_unavailable")


def _prepare_writable_roots(execution_source, artifact_root, artifact_mount):
    """Complete the locked writable layout while preserving a verified cache."""
    _require_runtime_root_platform()
    names = ("artifacts", "state", "tracking", "cache", "tmp")
    locked = execution_source.roots
    mount = PurePosixPath(artifact_mount)
    if (
        set(locked) != {*names, "engine", "project"}
        or not mount.is_absolute()
        or ".." in mount.parts
    ):
        raise ValueError("docker_runtime_roots_invalid")
    relatives = []
    for name in names:
        path = PurePosixPath(locked[name])
        if (
            not path.is_absolute()
            or ".." in path.parts
            or not path.is_relative_to(mount)
            or path == mount
        ):
            raise ValueError("docker_runtime_roots_invalid")
        relatives.append(path.relative_to(mount))
    if len(set(relatives)) != len(relatives) or any(
        left in right.parents
        for left in relatives
        for right in relatives
        if left != right
    ):
        raise ValueError("docker_runtime_roots_invalid")

    try:
        with ExitStack() as held:
            guard = _Root(artifact_root)
            held.callback(guard.stack.close)
            chains = []
            for relative in relatives:
                guard.check()
                parent = guard.fd
                chain = []
                for part in relative.parts:
                    try:
                        os.mkdir(part, 0o700, dir_fd=parent)
                        os.fsync(parent)
                    except FileExistsError:
                        pass
                    child = os.open(part, _DIR, dir_fd=parent)
                    held.callback(os.close, child)
                    info = os.fstat(child)
                    if (
                        not stat.S_ISDIR(info.st_mode)
                        or info.st_uid != os.getuid()
                        or stat.S_IMODE(info.st_mode) != 0o700
                    ):
                        raise ValueError("docker_runtime_roots_invalid")
                    chain.append((parent, part, child))
                    parent = child
                chains.append(chain)
            guard.check()
            for chain in chains:
                DockerLocalArtifactsV1._recheck(chain)
    except (OSError, ValueError) as error:
        raise ValueError("docker_runtime_roots_invalid") from error


@contextmanager
def open_host(
    config,
    *,
    state_root,
    mode,
    ports=None,
    runner=None,
    launch_authority=None,
    source_inputs=None,
):
    """Cold Docker still needs original qualification assets and local mount roots."""
    if mode not in {"start", "read"}:
        raise ValueError("host mode invalid")
    if mode == "start":
        _require_runtime_root_platform()
    selected = config["docker"]
    profile, rich, material, recipes, basis = _components(config)
    run = TrainingRunRef(material.run_id, basis.project_ref)
    if config["run"] != run.to_dict():
        raise ValueError("configured run differs from material")
    for name in ("command_seconds", "read_seconds"):
        if type(selected[name]) is not int or not 1 <= selected[name] <= 300:
            raise ValueError("command/read bound invalid")
    if type(selected.get("allow_current_cancel_authorization", False)) is not bool:
        raise ValueError("exact cancellation option required")
    if set(selected["role_paths"]) != SFT_ROLES:
        raise ValueError("native five-role path mapping required")
    if ports is None:
        from launch import host_ports

        with host_ports(config, state_root, create=mode == "start") as actual:
            with open_host(
                config,
                state_root=state_root,
                mode=mode,
                ports=actual,
                runner=runner,
                launch_authority=launch_authority,
                source_inputs=source_inputs,
            ) as host:
                yield host
        return
    with ExitStack() as stack:
        cli_runner = runner or subprocess_runner
        if launch_authority is None:
            # The original bounded assets must survive cold composition. This
            # is current identity verification, never image building/provisioning.
            read_committed(selected["image_profile_file"])
            read_committed(selected["image_verification_file"])
            launch_authority = verify_effectful_launch(
                profile_path=Path(selected["image_profile_file"]["path"]),
                verification_report_path=Path(
                    selected["image_verification_file"]["path"]
                ),
                image=f"{profile.image.image_ref}@{profile.image.image_digest}",
                docker=Path(selected["executable"]),
                docker_config=Path(selected["config_directory"]),
                runner=selected_runner(selected["endpoint"], cli_runner),
            )
        source_root, artifact_root = (
            Path(selected["source_host_root"]),
            Path(selected["artifact_host_root"]),
        )
        if mode == "start":
            for root in (source_root, artifact_root):
                if not root.exists():
                    root.mkdir(mode=0o700)
            _prepare_writable_roots(
                rich.execution_source,
                artifact_root,
                selected["artifact_container_root"],
            )
        bundle = materialize_worker_bundle(
            build_source_worker_invocation(
                rich,
                WorkerControlLocationV1(
                    PurePosixPath(selected["control_container_root"])
                ),
                CanonicalWorkloadFileLocationV1(
                    PurePosixPath(selected["control_container_root"])
                ),
            )
        )
        if mode == "start" and source_inputs is None:
            identity = PreparedTrainingInputIdentity.from_dict(
                rich.resolved_config.to_dict()["dataset"]
            )
            if (
                identity.format != "syntunia-sft-row/v1"
                or identity.size_bytes > 2 << 20
            ):
                raise ValueError(
                    "first consumer supports v1 prepared input through2MiB"
                )
            lease = LocalPreparedTrainingInputSource(
                Path(selected["prepared_publication"]), identity
            ).open_lease()
            stack.callback(lease.close)
            closure = parse_offline_sft_worker_manifest(
                bundle.closure_manifest_bytes,
                source_ref=bundle.closure_manifest_source,
                manifest_path=Path(bundle.closure_manifest_runtime_path),
            )
            import tuner

            engine = Path(tuner.__file__).resolve().parent.parent
            members = {
                item.path.as_posix(): stable_read(engine / item.path, item.size_bytes)
                for item in closure.closure.members
            }
            source_inputs = {
                "bundle": bundle,
                "closure_members": members,
                "prepared_lease": lease,
                "source_root": source_root,
                "source_container_root": PurePosixPath(
                    selected["source_container_root"]
                ),
                "artifact_root": artifact_root,
                "artifact_container_root": PurePosixPath(
                    selected["artifact_container_root"]
                ),
                "deadline": time.monotonic() + selected["command_seconds"],
            }
        cli = {
            "authority": launch_authority,
            "endpoint": selected["endpoint"],
            "daemon_id": selected["daemon_id"],
            "image_refs": {profile.image.image_ref: profile.image.image_digest},
            "source_roots": {
                profile.roots.source_ref: (
                    source_root,
                    selected["source_container_root"],
                )
            },
            "artifact_roots": {
                profile.roots.artifact_ref: (
                    artifact_root,
                    selected["artifact_container_root"],
                )
            },
            "environment": dict(bundle.dispatch.environment),
            "deadline": time.monotonic() + selected["command_seconds"],
            "runner": cli_runner,
        }
        host = stack.enter_context(
            compose_docker_reference_host(
                profile=profile,
                rich_plan=rich,
                material=material,
                recipes=recipes,
                ports=ports,
                authority_secret=SecretRef.from_dict(
                    config["host"]["authority_secret"]
                ),
                evidence_secret=SecretRef.from_dict(config["host"]["evidence_secret"]),
                cli_configuration=cli,
                source_inputs=source_inputs if mode == "start" else None,
                artifact_root=artifact_root,
                role_paths=selected["role_paths"],
                read_seconds=selected["read_seconds"],
                allow_current_cancel_authorization=selected.get(
                    "allow_current_cancel_authorization", False
                ),
            )
        )
        yield host.api.training, host.api.runs, profile.provider, run
