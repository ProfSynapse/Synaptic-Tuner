"""Host-owned Docker preparation; no daemon, GPU or model effects in prepare."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import secrets
import sys


def filesystem_probe(configuration):
    """Fixed CPU diagnostic through admitted native writer; no engine authority."""
    import os
    import types
    import stat

    value = json.loads(Path(configuration).read_bytes())
    if set(value) != {"runtime_file", "runtime_sha256", "output_root"}:
        raise ValueError("filesystem_probe_configuration_invalid")
    root = Path(value["output_root"])
    info = root.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or root.is_symlink()
        or stat.S_IMODE(info.st_mode) != 0o700
        or any(root.iterdir())
    ):
        raise ValueError("filesystem_probe_output_invalid")
    fd = os.open(value["runtime_file"], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or not 0 < info.st_size <= 2 << 20
        ):
            raise ValueError("filesystem_probe_runtime_invalid")
        raw = bytearray()
        while len(raw) <= 2 << 20:
            chunk = os.read(fd, 1048576)
            if not chunk:
                break
            raw.extend(chunk)
        if (
            len(raw) != info.st_size
            or hashlib.sha256(raw).hexdigest() != value["runtime_sha256"]
        ):
            raise ValueError("filesystem_probe_runtime_mismatch")
        name = "_synaptic_native_filesystem_probe"
        module = types.ModuleType(name)
        module.__file__ = value["runtime_file"]
        original = sys.modules.get(name)
        sys.modules[name] = module
        try:
            exec(compile(bytes(raw), module.__file__, "exec"), module.__dict__)
        finally:
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        native = module.__dict__
        with open("/proc/self/status", "rb") as status:
            status_raw = status.read(65537)
        if len(status_raw) > 65536:
            raise ValueError("filesystem_probe_status_invalid")
        masks = [
            line.split()[1]
            for line in status_raw.splitlines()
            if line.startswith(b"Umask:")
        ]
        if (
            len(masks) != 1
            or len(masks[0]) != 4
            or any(c not in b"01234567" for c in masks[0])
        ):
            raise ValueError("filesystem_probe_umask_invalid")
        before = int(masks[0], 8)
        artifact = native["_write_artifact"](
            root, "training_metrics", "native-write-probe.json", b'{"fixture":true}\n'
        )
        written = (root / artifact["path"]).lstat()
        return {
            "scope": "cpu_native_filesystem_only",
            "umask": oct(before),
            "uid": written.st_uid,
            "gid": written.st_gid,
            "mode": oct(stat.S_IMODE(written.st_mode)),
            "sha256": artifact["sha256"],
            "size_bytes": artifact["size"],
            "training_executed": False,
            "gpu_qualified": False,
        }
    finally:
        os.close(fd)


# This fixed diagnostic needs stdlib only in an unmodified base image. Normal
# material/model preparation below still imports the actual installed engine.
if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "probe-filesystem":
    try:
        print(json.dumps(filesystem_probe(sys.argv[2]), allow_nan=False))
    except Exception:
        print(json.dumps({"diagnostic_code": "docker_filesystem_probe_failed"}))
        raise SystemExit(2)
    raise SystemExit(0)


sys.path.insert(0, str(Path(__file__).resolve().parent))
import docker_host
import launch
import prepare_hf as common

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.dataset_prep.publication import snapshot_prepared_dataset_v1
from tuner.execution.evidence import SOURCE_EVIDENCE_PURPOSE
from tuner.execution.providers.docker_provider_v1.preparation import (
    DockerTrainingPreparationBridgeV1,
    _NO_SECRET_REQUIREMENTS_DIGEST,
    _ZERO_QUOTE_DIGEST,
    _resource_digest,
)
from tuner.execution.providers.modal.model_snapshot import (
    MODEL_SNAPSHOT_PREPARATION_STAGES,
    ModelSnapshotPreparationError,
    _bind_hub_api,
    prepare_model_snapshot,
)
from tuner.project.context import ProjectContext
from tuner.project.git_verification import (
    GitCliLocalSourceInspector,
    GitLsRemotePushedCommitVerifier,
)
from tuner.project.manifest import load_project_manifest
from tuner.runtime.dispatch import (
    WorkerControlLocationV1,
    CanonicalWorkloadFileLocationV1,
    build_source_worker_invocation,
    materialize_worker_bundle,
)
from tuner.runtime.releases import parse_packaged_runtime_release
from tuner.training import TrainingService, default_recipe_registry
from tuner.training.contracts import (
    ArtifactPolicy,
    CanonicalDocument,
    ResolvedTrainingComponents,
    RuntimeSpec,
    ResourceSpec,
)
from tuner.training.coordinator_material import (
    CoordinatorResolvedMaterial,
    derive_coordinator_material,
)

ROLES = tuple(
    sorted(
        (
            "final_model",
            "tokenizer",
            "training_lineage",
            "training_metrics",
            "workload_record",
        )
    )
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def declaration(config):
    """Pure declaration of exact selected release pins and Docker mount layout."""
    release = parse_packaged_runtime_release(
        common._document(config["runtime_release"]).to_dict()
    )
    model = common._document(config["resolved_config"]).to_dict()["model"]
    if (
        "sft" not in release.compatible_methods
        or (model["ref"], model["revision"]) not in release.compatible_models
        or "syntunia-sft-row/v1" not in release.compatible_dataset_formats
        or model["tokenizer_revision"] != model["revision"]
    ):
        raise ValueError("docker_release_selection_invalid")
    selected = config["consumer"]["docker"]
    if (
        selected["source_container_root"],
        selected["artifact_container_root"],
        selected["control_container_root"],
    ) != ("/source", "/artifacts", "/source/control"):
        raise ValueError("docker_runtime_layout_invalid")
    roots = {"engine": "/source/engine", "project": "/source/project"}
    roots.update(
        {
            name: "/artifacts/run/" + name
            for name in ("state", "tracking", "cache", "tmp", "artifacts")
        }
    )
    environment = common._environment(config)
    roots["project"] = environment["SYNAPTIC_PROJECT_ROOT"]
    if environment.get("PYTHONPATH") != roots["engine"]:
        raise ValueError("docker_runtime_environment_invalid")
    return CanonicalDocument.from_mapping(
        {
            "schema_version": "docker-runtime-declaration/v1",
            "release_digest": sha(release.canonical_bytes()),
            "image": release.image_ref,
            "inventory_sha256": release.installed_distributions_digest,
            "interpreter": {
                "implementation": release.python_implementation,
                "version": release.python_version,
                "executable": release.python_executable,
                "executable_digest": release.python_executable_digest,
            },
            "roots": roots,
            "writable_capability_root": "/artifacts",
            "environment": environment,
            "secret_requirements_digest": _NO_SECRET_REQUIREMENTS_DIGEST,
            "provider_runtime_requirements_digest": sha(release.canonical_bytes()),
        }
    )


def prepare(config, *, pushed_sources, authenticator, clock):
    """Real source admission and compiler; no provider or model construction."""
    root = Path(config["output_root"])
    if not root.is_absolute() or root.exists() or root.is_symlink():
        raise ValueError("docker_output_root_used_or_invalid")
    runtime = declaration(config)
    verified, raw = snapshot_prepared_dataset_v1(Path(config["prepared_publication"]))
    if len(raw) > 2 << 20:
        raise ValueError("docker_prepared_input_too_large")
    identity = PreparedTrainingInputIdentity(
        "prepared://sha256/" + verified.semantic_identity.dataset_digest,
        verified.semantic_identity.dataset_digest,
        sha(raw),
        len(raw),
        "syntunia-sft-row/v1",
    )
    normalized = common._document(config["resolved_config"])
    if normalized.to_dict().get("dataset") != identity.to_dict():
        raise ValueError("docker_prepared_configuration_mismatch")
    context = ProjectContext.host(
        project_root=Path(config["project"]["root"]),
        engine_root=Path(config["project"]["engine_root"]),
        manifest_path=Path(config["project"]["manifest"]),
        invocation_cwd=Path(config["project"]["root"]),
    )
    manifest = load_project_manifest(context.manifest_path)
    source, lock, inspected = common.authenticated_source(
        config,
        context,
        normalized,
        identity,
        runtime,
        runtime.to_dict(),
        manifest,
        pushed_sources=pushed_sources,
        authenticator=authenticator,
        clock=clock,
    )
    d = runtime.to_dict()
    components = ResolvedTrainingComponents(
        source,
        runtime,
        normalized,
        RuntimeSpec(d["image"], d["inventory_sha256"], source.python_version),
        ResourceSpec(**config["resources"]),
        ArtifactPolicy(ROLES, False),
    )

    class Resolver:
        def resolve(self, request, *, context):
            return components

    recipes = default_recipe_registry()
    service = TrainingService(context=context, resolver=Resolver(), recipes=recipes)
    request = service.load(normalized)
    resolved = service.resolve(request)
    rich = service.plan(resolved)
    material = derive_coordinator_material(
        resolved,
        recipes,
        request_id=config["request_id"],
        project_ref=manifest.project_id,
        run_id=source.run_id,
    )
    material = CoordinatorResolvedMaterial.parse(material.canonical_bytes, recipes)
    bundle = materialize_worker_bundle(
        build_source_worker_invocation(
            rich,
            WorkerControlLocationV1(PurePosixPath("/source/control")),
            CanonicalWorkloadFileLocationV1(PurePosixPath("/source/control")),
        )
    )
    p = json.loads(json.dumps(config["profile"], allow_nan=False))
    common._exact(
        p,
        {
            "provider",
            "descriptor",
            "scope",
            "executor_descriptor",
            "adapter_descriptor",
            "runtime",
            "roots",
            "artifacts",
        },
    )
    common._exact(p["artifacts"], {"maximum_artifact_bytes", "maximum_total_bytes"})
    p["schema_version"] = "synaptic-docker-profile/v1"
    image_ref, image_digest = rich.runtime.image.rsplit("@", 1)
    p.update(
        image={
            "image_ref": image_ref,
            "image_digest": image_digest,
            "presence_policy": "present_only",
        },
        workload={
            "arguments": list(bundle.dispatch.argv),
            "environment_keys": sorted(dict(bundle.dispatch.environment)),
            "workload_digest": bundle.workload_sha256,
        },
        resource_digest=_resource_digest(rich),
        quote_digest=_ZERO_QUOTE_DIGEST,
        secret_requirements_digest=_NO_SECRET_REQUIREMENTS_DIGEST,
    )
    p["artifacts"]["roles"] = list(ROLES)
    profile = docker_host._profile(
        CanonicalDocument.from_mapping(p).canonical_json.encode()
    )
    DockerTrainingPreparationBridgeV1(profile).coordinator_material(
        rich, material, recipes
    )
    if not common._same_checkout(
        inspected, GitCliLocalSourceInspector().inspect(context=context)
    ):
        raise ValueError("docker_sources_changed")
    consumer = json.loads(json.dumps(config["consumer"], allow_nan=False))
    root.mkdir(mode=0o700)
    launch._private(root)
    for name, data in (
        ("request.json", material.request_bytes),
        ("material.json", material.canonical_bytes),
        (
            "docker-profile.json",
            CanonicalDocument.from_mapping(profile.to_dict()).canonical_json.encode(),
        ),
        ("docker-runtime-declaration.json", runtime.canonical_json.encode()),
        (
            "source-lock.json",
            CanonicalDocument.from_mapping(lock.to_dict()).canonical_json.encode(),
        ),
    ):
        launch._write_bytes(root / name, data)

    def spec(name):
        data = common.stable_read(root / name, 2 << 20)
        return {"path": str(root / name), "sha256": sha(data), "size_bytes": len(data)}

    consumer["request_file"] = spec("request.json")
    consumer["docker"]["material_file"] = spec("material.json")
    consumer["docker"]["profile_file"] = spec("docker-profile.json")
    consumer["docker"]["prepared_publication"] = config["prepared_publication"]
    launch._write(root / "consumer-candidate.json", consumer)
    consumer = launch.load_config(root / "consumer-candidate.json")
    preview = docker_host.preview(consumer)
    launch._write(root / "preview.json", preview)
    launch._write(root / "consumer.json", consumer)
    return {
        "consumer_config": str(root / "consumer.json"),
        "preview": preview,
        "public_preflight": "deferred_until_host_composition",
    }


def model_preflight():
    """Check the existing SDK call contract before any one-use model claim."""
    try:
        from huggingface_hub import HfApi, snapshot_download
        _bind_hub_api(HfApi, snapshot_download)
    except Exception:
        raise ModelSnapshotPreparationError("SDK_ADMISSION") from None
    return {
        "model_sdk_contract": "compatible",
        "model_download": False,
        "provider_qualified": False,
    }


def prepare_model(
    config, *, model_preparer=prepare_model_snapshot, sdk_preflight=model_preflight,
):
    """Explicit pinned public download; never writes the sealed source mount."""
    from contextlib import ExitStack
    from tuner.execution.providers.docker_provider_v1.local_source import _Root
    from Trainers.sft.runtime_v1 import RuntimeRoots, _require_local_model_snapshot

    sdk_preflight()
    output = Path(config["output_root"])
    launch._private(output)
    consumer = launch.load_config(output / "consumer.json")
    _, rich, _, _, _ = docker_host._components(consumer)
    if rich.execution_context != declaration(config):
        raise ValueError("docker_model_configuration_changed")
    model = rich.resolved_config.to_dict()["model"]
    selected = consumer["docker"]
    source, artifact = (
        Path(selected["source_host_root"]),
        Path(selected["artifact_host_root"]),
    )
    for root in (source, artifact):
        if not root.exists():
            launch._private(root.parent)
            root.mkdir(mode=0o700)
        launch._private(root)
    with ExitStack() as held:
        guards = []
        for root in (source, artifact):
            guard = _Root(root)
            held.callback(guard.stack.close)
            guards.append(guard)
            guard.check()
        if any(source.iterdir()):
            raise ValueError("docker_source_not_empty")
        mapped = {}
        for name, raw in rich.execution_source.roots.items():
            path = PurePosixPath(raw)
            if path.is_relative_to("/source"):
                mapped[name] = source.joinpath(*path.relative_to("/source").parts)
            elif path.is_relative_to("/artifacts"):
                mapped[name] = artifact.joinpath(*path.relative_to("/artifacts").parts)
            else:
                raise ValueError("docker_model_root_unmapped")
        # All writable roots are explicitly private, before the prepared-input
        # slot is claimed. The exact slot itself is never created or adopted.
        run = artifact / "run"
        if not run.exists():
            run.mkdir(mode=0o700)
        launch._private(run)
        for name in ("state", "tracking", "cache", "tmp", "artifacts"):
            path = mapped[name]
            if not path.exists():
                path.mkdir(mode=0o700)
            launch._private(path)
        persistent, scratch = output / "model-persistent", output / "model-scratch"
        launch._write(
            output / "model-preparation-attempt.json",
            {
                "model": model,
                "destination": str(mapped["cache"]),
                "source_mutation": False,
            },
        )
        for path in (persistent, scratch):
            path.mkdir(mode=0o700)
        snapshot = model_preparer(
            model_ref=model["ref"],
            revision=model["revision"],
            token=False,
            persistent_root=persistent,
            destination_root=mapped["cache"],
            scratch_root=scratch,
        )
        expected = (
            mapped["cache"]
            / "model"
            / ("models--" + model["ref"].replace("/", "--"))
            / "snapshots"
            / model["revision"]
        )
        if snapshot != expected:
            raise ValueError("docker_model_snapshot_mismatch")
        environment = {
            "SYNAPTIC_CACHE_ROOT": str(mapped["cache"]),
            "SYNAPTIC_MODEL_SNAPSHOT": str(expected),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        checked = _require_local_model_snapshot(
            model, RuntimeRoots(**mapped), environment
        )
        for guard in guards:
            guard.check()
        for root in (source, artifact):
            launch._private(root)
        if any(source.iterdir()):
            raise ValueError("docker_source_changed")
        result = {
            "snapshot": str(checked),
            "model_ref": model["ref"],
            "revision": model["revision"],
            "source_mutation": False,
            "provider_qualified": False,
        }
        launch._write(output / "model-preparation.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "model-preflight", "prepare-model"))
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args(argv)
    try:
        config = common._document(args.configuration).to_dict()
        if args.action == "plan":
            result = {
                "runtime": declaration(config).to_dict(),
                "operational_preflight": False,
                "model_download": False,
                "provider_qualified": False,
            }
        elif args.action == "model-preflight":
            result = model_preflight()
        elif args.action == "prepare-model":
            result = prepare_model(config)
        else:
            a = config["source_authority"]
            secret = SecretRef.from_dict(a["secret"])
            host = config["consumer"]["host"]
            if secret in (
                SecretRef.from_dict(host["authority_secret"]),
                SecretRef.from_dict(host["evidence_secret"]),
            ):
                raise ValueError("docker_source_key_not_separate")
            key = launch.NamedSecrets().resolve(secret)
            if type(key) is not str or len(key.encode()) < 32:
                raise ValueError("docker_source_key_unavailable")
            auth_module = common._example_module("docker_source_auth", "authority.py")
            auth = auth_module.HMACAuthenticator(
                {a["key_ref"]: key.encode()},
                allowed_purposes=frozenset({SOURCE_EVIDENCE_PURPOSE}),
            )
            context = ProjectContext.host(
                project_root=Path(config["project"]["root"]),
                engine_root=Path(config["project"]["engine_root"]),
            )
            lock = GitCliLocalSourceInspector().inspect(context=context)
            refs = frozenset(
                (s.location.canonical_url, "refs/heads/" + s.branch)
                for s in (lock.project_source, lock.engine_source)
            )
            remote = common._example_module(
                "docker_source_remote", "remote.py"
            ).ModalChatGitRemote(refs)
            clock = auth_module.UTCClock()
            verifier = GitLsRemotePushedCommitVerifier(
                remote,
                auth,
                clock=clock.now,
                audience_ref=a["audience_ref"],
                issuer_ref=a["issuer_ref"],
                key_ref=a["key_ref"],
                challenge_factory=lambda: secrets.token_hex(24),
                evidence_ref_factory=lambda: secrets.token_hex(24),
            )
            result = prepare(
                config, pushed_sources=verifier, authenticator=auth, clock=clock
            )
    except Exception as error:
        diagnostic = {"diagnostic_code": "docker_material_preparation_failed"}
        if (
            type(error) is ModelSnapshotPreparationError
            and type(error.stage) is str
            and error.stage in MODEL_SNAPSHOT_PREPARATION_STAGES
        ):
            diagnostic["model_preparation_stage"] = error.stage
        print(json.dumps(diagnostic))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
