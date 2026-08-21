from __future__ import annotations

import ast
import hashlib
import gzip
import inspect
import io
import json
import sys
import tarfile
import textwrap
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tuner.cloud.hf_training_image_lock import (
    CANDIDATE_SCHEMA, DIAGNOSTIC_OVERALL_TIMEOUT_SECONDS,
    CommandResult, CommandSpec, MetadataDiagnosticStageError,
    PythonRuntimeIdentityDiagnosticError, RuntimeSubstageDiagnosticError,
    TrainingImageLockError,
    build_inspect_command, build_pull_command, build_runtime_command, build_save_command,
    canonical_runtime_lock_from_candidate,
    capture_candidate, diagnose_runtime_metadata, diagnose_runtime_metadata_attributed,
    diagnose_runtime_substage_attributed, observe_python_runtime_identity,
    subprocess_runner, validate_oci_documents,
)
from tuner.cloud.hf_training_docker_archive import MAX_ARCHIVE_BYTES
from tuner.cloud.hf_training_image_operation_lock import ImageOperationLockError
from tuner.cloud.hf_training_oci_registry import CHILD_MEDIA_TYPE, REGISTRY_REPOSITORY, RegistryDocuments
from tuner.cloud.hf_training_smoke_contract import (
    RUNTIME_PYTHON_IMPLEMENTATION,
    RUNTIME_PYTHON_VERSION,
    document_sha256,
    validate_runtime_lock,
)


def _raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _runtime_value() -> dict[str, object]:
    packages = {
        name: "1.0.0" for name in (
            "accelerate", "datasets", "huggingface-hub", "numpy", "peft",
            "safetensors", "torch", "transformers", "trl", "unsloth",
            "unsloth-zoo",
        )
    }
    return {
        "python_implementation": RUNTIME_PYTHON_IMPLEMENTATION,
        "python": RUNTIME_PYTHON_VERSION,
        "packages": packages,
        "signatures": {
            "TrainerCallback.on_optimizer_step": "(self, args, state, control, **kwargs)",
            "safetensors.safe_open": "(filename, framework, device='cpu')",
            "torch.load": "(f, map_location=None, **kwargs)",
            "unsloth.import": "GPU_RUNTIME_REQUIRED",
        },
    }


def _runtime_raw() -> bytes:
    return _raw(_runtime_value())


def _config_raw() -> bytes:
    return _raw({"architecture": "amd64", "os": "linux", "rootfs": {"type": "layers", "diff_ids": [_digest(b"layer")]}})


def _authority_result(spec: CommandSpec) -> bytes | None:
    if "version" in spec.argv:
        return _raw({"ClientVersion": "27.0.0", "ServerVersion": "27.0.0"})
    if "info" in spec.argv:
        return _raw({
            "ID": "daemon-id", "ServerVersion": "27.0.0", "OSType": "linux",
            "Architecture": "x86_64", "Name": "docker-desktop",
            "DockerRootDir": "/var/lib/docker", "Driver": "overlay2",
            "SecurityOptions": ["name=seccomp"],
        })
    if "context" in spec.argv and "inspect" in spec.argv:
        return _raw({"Name": "default", "DockerEndpoint": "npipe:////./pipe/docker_engine", "SkipTLSVerify": False})
    return None


def _documents(kind: str = "manifest") -> RegistryDocuments:
    config = _config_raw()
    child = _raw({
        "schemaVersion": 2, "mediaType": CHILD_MEDIA_TYPE,
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": _digest(config), "size": len(config)},
        "layers": [{"mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip", "digest": "sha256:" + "4" * 64, "size": 200}],
    })
    if kind == "index":
        requested = _raw({
            "schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
            "manifests": [{"mediaType": CHILD_MEDIA_TYPE, "digest": _digest(child), "size": len(child), "platform": {"os": "linux", "architecture": "amd64"}}],
        })
        requested_media = "application/vnd.docker.distribution.manifest.list.v2+json"
        child_raw: bytes | None = child
    else:
        requested, requested_media, child_raw = child, CHILD_MEDIA_TYPE, None
    return RegistryDocuments(
        requested, child_raw, _digest(requested), requested_media, kind,
        _digest(child), CHILD_MEDIA_TYPE, _digest(config),
        "application/vnd.docker.container.image.v1+json", len(config),
    )


def _oci_documents() -> tuple[RegistryDocuments, bytes]:
    config = _config_raw()
    layer_blob = gzip.compress(b"layer", mtime=0)
    child = _raw({
        "schemaVersion": 2, "mediaType": CHILD_MEDIA_TYPE,
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "digest": _digest(config), "size": len(config),
        },
        "layers": [{
            "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
            "digest": _digest(layer_blob), "size": len(layer_blob),
        }],
    })
    documents = RegistryDocuments(
        child, None, _digest(child), CHILD_MEDIA_TYPE, "manifest",
        _digest(child), CHILD_MEDIA_TYPE, _digest(config),
        "application/vnd.docker.container.image.v1+json", len(config),
    )
    return documents, layer_blob


def _write_oci_compat_archive(
    path: Path, documents: RegistryDocuments, layer_blob: bytes, *, annotated: bool,
) -> None:
    config = _config_raw()
    config_path = "blobs/sha256/" + documents.config_digest[7:]
    child_path = "blobs/sha256/" + documents.child_digest[7:]
    layer_digest = json.loads(documents.requested_raw)["layers"][0]["digest"]
    layer_path = "blobs/sha256/" + layer_digest[7:]
    descriptor: dict[str, object] = {
        "mediaType": CHILD_MEDIA_TYPE, "digest": documents.child_digest,
        "size": len(documents.requested_raw),
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    if annotated:
        descriptor["annotations"] = {
            "containerd.io/distribution.source.docker.io": "unsloth/unsloth",
        }
    index = _raw({
        "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [descriptor],
    })
    compatibility = _raw([{
        "Config": config_path, "RepoTags": None, "Layers": [layer_path],
    }])
    members = [
        ("blobs", b"", tarfile.DIRTYPE),
        ("blobs/sha256", b"", tarfile.DIRTYPE),
        ("oci-layout", _raw({"imageLayoutVersion": "1.0.0"}), tarfile.REGTYPE),
        ("index.json", index, tarfile.REGTYPE),
        (child_path, documents.requested_raw, tarfile.REGTYPE),
        (config_path, config, tarfile.REGTYPE),
        (layer_path, layer_blob, tarfile.REGTYPE),
        ("manifest.json", compatibility, tarfile.REGTYPE),
    ]
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = 0 if kind == tarfile.DIRTYPE else len(raw)
            archive.addfile(info, None if kind == tarfile.DIRTYPE else io.BytesIO(raw))


def _documents_with_sizes(config_size: object, layer_sizes: list[object]) -> RegistryDocuments:
    base = _documents()
    child = json.loads(base.requested_raw)
    child["config"]["size"] = config_size
    child["layers"] = [
        {
            "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
            "digest": "sha256:" + f"{index + 1:064x}",
            "size": size,
        }
        for index, size in enumerate(layer_sizes)
    ]
    raw = _raw(child)
    return replace(
        base, requested_raw=raw, requested_digest=_digest(raw),
        child_digest=_digest(raw), config_size=config_size,
    )


def _write_archive(path: Path, documents: RegistryDocuments) -> None:
    config = _config_raw()
    config_name = documents.config_digest[7:] + ".json"
    layers = ["layer/layer.tar"]
    manifest = _raw([{"Config": config_name, "RepoTags": None, "Layers": layers}])
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw in (("manifest.json", manifest), (config_name, config), (layers[0], b"layer")):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))


@pytest.mark.parametrize("kind", ["manifest", "index"])
def test_oci_chain_binds_t_r1_identity_config_and_ordered_layers(kind: str) -> None:
    documents = _documents(kind)
    identity = validate_oci_documents(documents)
    assert identity["requested_kind"] == kind
    assert identity["registry_repository"] == REGISTRY_REPOSITORY
    assert identity["provider_reference"] == f"unsloth/unsloth@{documents.child_digest}"
    assert identity["layers"] == [{"media_type": "application/vnd.docker.image.rootfs.diff.tar.gzip", "digest": "sha256:" + "4" * 64, "size": 200}]
    assert identity["index_digest"] == (None if kind == "manifest" else documents.requested_digest)


@pytest.mark.parametrize("tamper", ["requested", "child", "layer"])
def test_oci_chain_fails_closed_on_authenticated_evidence_drift(tamper: str) -> None:
    documents = _documents("index")
    if tamper == "requested":
        documents = replace(documents, requested_raw=documents.requested_raw + b" ")
    elif tamper == "child":
        documents = replace(documents, child_raw=(documents.child_raw or b"") + b" ")
    else:
        child = json.loads(documents.child_raw)
        child["layers"][0]["size"] = 0
        child_raw = _raw(child)
        index = json.loads(documents.requested_raw)
        index["manifests"][0]["digest"], index["manifests"][0]["size"] = _digest(child_raw), len(child_raw)
        requested = _raw(index)
        documents = replace(documents, requested_raw=requested, requested_digest=_digest(requested), child_raw=child_raw, child_digest=_digest(child_raw))
    with pytest.raises(TrainingImageLockError, match="EVIDENCE_INVALID"):
        validate_oci_documents(documents)


def test_exact_child_pull_and_runtime_commands_have_hardened_contract(tmp_path: Path) -> None:
    child = "sha256:" + "a" * 64
    reference = f"unsloth/unsloth@{child}"
    pull = build_pull_command(docker=tmp_path / "docker", config_dir=tmp_path / "empty", provider_reference=reference)
    assert pull.argv[-1] == reference and "--quiet" in pull.argv and ("--platform", "linux/amd64") == pull.argv[-3:-1]
    command = build_runtime_command(docker=tmp_path / "docker", config_dir=tmp_path / "empty", repository="unsloth/unsloth", child_digest=child)
    argv = command.argv
    assert reference in argv and "--pull=never" in argv and ("--network", "none") == argv[argv.index("--network"):argv.index("--network") + 2]
    assert "--read-only" in argv and "no-new-privileges" in argv
    assert "-v" not in argv and "--volume" not in argv and not any("docker.sock" in part for part in argv)
    reference_index = argv.index(reference)
    assert argv[reference_index - 2:reference_index] == ("--entrypoint", "python")
    assert argv[reference_index + 1:reference_index + 3] == ("-I", "-c")
    assert len(argv[reference_index + 1:]) == 3
    assert argv.count("python") == 1
    assert argv[argv.index("--workdir"):argv.index("--workdir") + 2] == ("--workdir", "/tmp")
    assert argv[argv.index("--tmpfs") + 1] == "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777,uid=65534,gid=65534"
    assert not any(part in argv for part in ("--gpus", "--mount", "--volume", "-v", "sh", "bash"))
    for fixed_environment in (
        "HOME=/tmp/home", "HF_HOME=/tmp/hf", "XDG_CACHE_HOME=/tmp/xdg",
        "TORCH_HOME=/tmp/torch", "HF_HUB_DISABLE_IMPLICIT_TOKEN=1",
        "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1",
        "PYTHONNOUSERSITE=1",
    ):
        assert fixed_environment in argv
    inspector = argv[-1]
    compile(inspector, "<hf-training-runtime-inspector>", "exec")
    assert inspector.startswith("import sys\nsys.dont_write_bytecode=True\n")
    assert inspector.index("sys.dont_write_bytecode=True") < inspector.index("import torch")
    assert f"platform.python_implementation()!={RUNTIME_PYTHON_IMPLEMENTATION!r}" in inspector
    assert f"platform.python_version()!={RUNTIME_PYTHON_VERSION!r}" in inspector
    assert inspector.index("PYTHON_RUNTIME_INVALID") < inspector.index("metadata.version(name)")
    assert "from unsloth import" not in inspector and "import unsloth" not in inspector
    assert "find_spec('unsloth')" in inspector and "GPU_RUNTIME_REQUIRED" in inspector
    assert "getusersitepackages" in inspector and "resolve(strict=True)" in inspector
    assert "S_ISREG" in inspector and "is_symlink" in inspector
    inspect = build_inspect_command(docker=tmp_path / "docker", config_dir=tmp_path / "empty", provider_reference=reference)
    assert inspect.argv[-1] == reference and inspect.argv[5:7] == ("image", "inspect")
    assert "RepoDigests" in inspect.argv[-2]
    save = build_save_command(
        docker=tmp_path / "docker", config_dir=tmp_path / "empty",
        provider_reference=reference,
    )
    assert save.argv[-1] == reference and save.argv[5:7] == ("image", "save")
    assert save.argv[-3:-1] == ("--platform", "linux/amd64")
    for invalid_reference in ("unsloth/unsloth", "unsloth/unsloth:latest", f"docker.io/{reference}"):
        with pytest.raises(TrainingImageLockError, match="IMAGE_INVALID"):
            build_save_command(
                docker=tmp_path / "docker", config_dir=tmp_path / "empty",
                provider_reference=invalid_reference,
            )


def test_runtime_substage_probe_has_exact_closed_order_and_envelopes(tmp_path: Path) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    docker = tmp_path / "docker.exe"
    config = tmp_path / "docker-config"
    digest = "sha256:" + "a" * 64
    spec = image_lock.build_runtime_command(
        docker=docker, config_dir=config, repository="unsloth/unsloth",
        child_digest=digest, runtime_substage_attribution=True,
    )
    inspector = spec.argv[-1]
    substages = [
        "python_bootstrap", "python_runtime", "workspace_setup",
        "distribution_metadata", "torch_import", "safetensors_import",
        "transformers_import", "signature_introspection", "unsloth_spec",
        "unsloth_origin", "unsloth_package_root", "site_roots", "site_membership",
        "user_site_isolation", "origin_chain", "result_serialization",
    ]
    positions = [inspector.index(f"stage='{stage}'") for stage in substages]
    assert positions == sorted(positions)
    compile(inspector, "<runtime-substage-inspector>", "exec")
    assert "synaptic-hf-training-runtime-substage/v1" in inspector
    assert "failure_lines[stage]" in inspector
    assert "except Exception:" in inspector and "except BaseException:" not in inspector
    assert inspector.index("try:") < inspector.index("import json")
    assert "'runtime':{'python_implementation':platform.python_implementation(),'python':platform.python_version()" in inspector
    assert spec.maximum_output_bytes == 65536
    assert spec.timeout_seconds == image_lock.RUNTIME_TIMEOUT_SECONDS
    assert "--pull=never" in spec.argv and "--network" in spec.argv


def test_runtime_validator_is_contiguous_before_substage_parser() -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    validator = inspect.getsource(image_lock._validate_runtime_evidence)
    assert "python_implementation = runtime" in validator
    assert "python = runtime" in validator
    assert "packages = runtime" in validator
    assert "signatures = runtime" in validator
    assert "_parse_runtime_substage_evidence" not in validator
    source = Path(image_lock.__file__).read_text(encoding="utf-8")
    assert source.index("def _validate_runtime_evidence") < source.index(
        "def _parse_runtime_substage_evidence",
    )


@pytest.mark.parametrize("runtime_substage", [
    "python_bootstrap", "python_runtime", "workspace_setup", "distribution_metadata",
    "torch_import", "safetensors_import", "transformers_import",
    "signature_introspection", "unsloth_spec", "unsloth_origin",
    "unsloth_package_root", "site_roots", "site_membership", "user_site_isolation",
    "origin_chain", "result_serialization",
])
def test_generated_substage_inspector_executes_literal_failure_reporter(
    runtime_substage: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    inspector = image_lock._runtime_substage_inspector()
    needle = "try:\n stage='python_bootstrap'\n"
    injected = inspector.replace(
        needle,
        f"try:\n stage={runtime_substage!r}\n raise RuntimeError('private-detail')\n",
        1,
    )
    assert injected != inspector
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(injected, "<injected-runtime-substage>", "exec"), {})
    assert output.getvalue().encode("ascii") == image_lock._RUNTIME_SUBSTAGE_FAILURE_BYTES[
        runtime_substage
    ]


def test_generated_substage_inspector_does_not_catch_baseexception() -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    inspector = image_lock._runtime_substage_inspector()
    injected = inspector.replace(
        "try:\n stage='python_bootstrap'\n",
        "try:\n stage='python_bootstrap'\n raise KeyboardInterrupt('cancel')\n",
        1,
    )
    output = io.StringIO()
    with redirect_stdout(output), pytest.raises(KeyboardInterrupt):
        exec(compile(injected, "<cancelled-runtime-substage>", "exec"), {})
    assert output.getvalue() == ""


@pytest.mark.parametrize("raw,canonical", [
    ("cpython", "CPython"), ("pypy", "PyPy"), ("graalpy", "GraalPy"),
    ("jython", "Jython"), ("ironpython", "IronPython"),
])
def test_generated_python_identity_inspector_maps_closed_implementations(
    tmp_path: Path, raw: str, canonical: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    spec = image_lock.build_python_runtime_identity_command(
        docker=tmp_path / "docker", config_dir=tmp_path / "config",
        repository="unsloth/unsloth", child_digest="sha256:" + "a" * 64,
    )
    inspector = spec.argv[-1]
    assert inspector.startswith("import sys\n")
    assert "import " not in inspector[len("import sys\n"):]
    assert spec.maximum_output_bytes == 256
    injected = inspector.replace("raw=sys.implementation.name", f"raw={raw!r}").replace(
        "version_info=sys.version_info",
        "version_info=type('V',(),{'major':3,'minor':12,'micro':7,'releaselevel':'final','serial':0})()",
    )
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(injected, "<python-identity>", "exec"), {})
    observed = output.getvalue().encode("ascii")
    assert image_lock._parse_python_runtime_identity(observed) == {
        "implementation": canonical,
        "schema_version": "synaptic-hf-training-python-runtime-identity/v1",
        "status": "OBSERVED",
        "version": "3.12.7",
    }


@pytest.mark.parametrize("replacement", [
    "raw='unknown'",
    "version_info=type('V',(),{'major':True,'minor':12,'micro':7,'releaselevel':'final','serial':0})()",
    "version_info=type('V',(),{'major':0,'minor':12,'micro':7,'releaselevel':'final','serial':0})()",
    "version_info=type('V',(),{'major':1000,'minor':12,'micro':7,'releaselevel':'final','serial':0})()",
    "version_info=type('V',(),{'major':3,'minor':-1,'micro':7,'releaselevel':'final','serial':0})()",
    "version_info=type('V',(),{'major':3,'minor':1000,'micro':7,'releaselevel':'final','serial':0})()",
    "version_info=type('V',(),{'major':3,'minor':12,'micro':-1,'releaselevel':'final','serial':0})()",
    "version_info=type('V',(),{'major':3,'minor':12,'micro':1000,'releaselevel':'final','serial':0})()",
    "version_info=type('V',(),{'major':3,'minor':12,'micro':7,'releaselevel':'candidate','serial':0})()",
    "version_info=type('V',(),{'major':3,'minor':12,'micro':7,'releaselevel':'final','serial':True})()",
    "version_info=type('V',(),{'major':3,'minor':12,'micro':7,'releaselevel':'final','serial':1})()",
])
def test_generated_python_identity_inspector_emits_fixed_failure(
    tmp_path: Path, replacement: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    inspector = image_lock.build_python_runtime_identity_command(
        docker=tmp_path / "docker", config_dir=tmp_path / "config",
        repository="unsloth/unsloth", child_digest="sha256:" + "a" * 64,
    ).argv[-1]
    if replacement.startswith("raw="):
        injected = inspector.replace("raw=sys.implementation.name", replacement)
    else:
        injected = inspector.replace("version_info=sys.version_info", replacement)
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(injected, "<python-identity-failure>", "exec"), {})
    assert output.getvalue().encode("ascii") == image_lock._PYTHON_IDENTITY_FAILED_BYTES


def test_generated_python_identity_inspector_preserves_baseexception(tmp_path: Path) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    inspector = image_lock.build_python_runtime_identity_command(
        docker=tmp_path / "docker", config_dir=tmp_path / "config",
        repository="unsloth/unsloth", child_digest="sha256:" + "a" * 64,
    ).argv[-1].replace("try:\n", "try:\n raise KeyboardInterrupt('cancel')\n", 1)
    output = io.StringIO()
    with redirect_stdout(output), pytest.raises(KeyboardInterrupt):
        exec(compile(inspector, "<python-identity-cancel>", "exec"), {})
    assert output.getvalue() == ""


@pytest.mark.parametrize("tamper", [
    "unknown", "leading_zero", "too_large", "prerelease", "whitespace",
    "reordered", "trailing", "failed_whitespace",
])
def test_python_runtime_identity_parser_rejects_hostile_evidence(tamper: str) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    value: dict[str, object] = {
        "implementation": "CPython",
        "schema_version": "synaptic-hf-training-python-runtime-identity/v1",
        "status": "OBSERVED",
        "version": "3.12.7",
    }
    if tamper == "unknown":
        value["implementation"] = "Unknown"
    elif tamper == "leading_zero":
        value["version"] = "03.12.7"
    elif tamper == "too_large":
        value["version"] = "1000.12.7"
    elif tamper == "prerelease":
        value["version"] = "3.12.7rc1"
    elif tamper == "whitespace":
        raw = b" " + _raw(value)
    elif tamper == "reordered":
        raw = json.dumps(
            {
                "status": "OBSERVED", "version": "3.12.7",
                "implementation": "CPython",
                "schema_version": "synaptic-hf-training-python-runtime-identity/v1",
            },
            ensure_ascii=True, separators=(",", ":"),
        ).encode("ascii")
    elif tamper == "trailing":
        raw = _raw(value) + b"\n"
    else:
        raw = b" " + image_lock._PYTHON_IDENTITY_FAILED_BYTES
    if tamper in {"unknown", "leading_zero", "too_large", "prerelease"}:
        raw = _raw(value)
    with pytest.raises(TrainingImageLockError):
        image_lock._parse_python_runtime_identity(raw)
    assert image_lock._parse_python_runtime_identity(
        image_lock._PYTHON_IDENTITY_FAILED_BYTES,
    ) is None


def test_runtime_probe_closes_unsloth_spec_and_origin_without_importing_it(
    tmp_path: Path,
) -> None:
    child = "sha256:" + "a" * 64
    command = build_runtime_command(
        docker=tmp_path / "docker", config_dir=tmp_path / "empty",
        repository="unsloth/unsloth", child_digest=child,
    )
    probe = command.argv[-1]
    for required in (
        "spec is None", "spec.loader is None", "spec.origin", "origin.is_absolute()",
        "origin.is_symlink()", "origin.resolve(strict=True)", "origin.lstat()",
        "spec.submodule_search_locations", "package_root != origin.parent",
        "sys.base_prefix", "sys.prefix", "site.getsitepackages()",
        "site.getusersitepackages()", "selected is None", "selected.is_symlink()",
    ):
        assert required in probe
    assert "metadata.version(name)" in probe
    assert "'unsloth'" in probe and "'unsloth-zoo'" in probe
    assert "importlib.import_module('unsloth')" not in probe
    assert "from unsloth" not in probe and "import unsloth" not in probe


def _execute_runtime_probe(
    tmp_path: Path, monkeypatch, capsys, *, attack: str | None = None,
    substage: bool = False, return_raw: bool = False,
    python_implementation: str = RUNTIME_PYTHON_IMPLEMENTATION,
    python_version: str = RUNTIME_PYTHON_VERSION,
) -> dict[str, object] | bytes:
    import importlib.metadata
    import importlib.util
    import platform
    import site

    base = (tmp_path / "interpreter").resolve()
    site_root = base / "lib" / "python3.12" / "site-packages"
    package_root = site_root / "unsloth"
    package_root.mkdir(parents=True)
    origin = package_root / "__init__.py"
    origin.write_text("# synthetic package origin\n", encoding="ascii")
    user_site = (tmp_path / "user-site").resolve()
    user_site.mkdir()
    loader: object | None = object()
    locations: list[str] = [str(package_root)]
    spec_origin: str | None = str(origin)
    if attack == "loader_missing":
        loader = None
    elif attack == "origin_relative":
        spec_origin = "unsloth/__init__.py"
    elif attack == "origin_outside_site":
        outside = (tmp_path / "outside.py").resolve()
        outside.write_text("# outside\n", encoding="ascii")
        spec_origin, locations = str(outside), [str(outside.parent)]
    elif attack == "origin_user_site":
        user_package = user_site / "unsloth"
        user_package.mkdir()
        user_origin = user_package / "__init__.py"
        user_origin.write_text("# user\n", encoding="ascii")
        spec_origin, locations = str(user_origin), [str(user_package)]
        site_root = user_site
    elif attack == "locations_missing":
        locations = []
    elif attack == "locations_duplicate":
        locations = [str(package_root), str(package_root)]
    elif attack == "package_root_mismatch":
        locations = [str(site_root)]

    spec = None if attack == "spec_missing" else SimpleNamespace(
        loader=loader, origin=spec_origin, submodule_search_locations=locations,
    )
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.setattr(sys, "prefix", str(base))
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_root)])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec if name == "unsloth" else None)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "1.0.0")
    monkeypatch.setattr(platform, "python_implementation", lambda: python_implementation)
    monkeypatch.setattr(platform, "python_version", lambda: python_version)
    monkeypatch.setattr("os.makedirs", lambda *_args, **_kwargs: None)

    torch = ModuleType("torch")
    torch.load = lambda f, **kwargs: None
    safetensors = ModuleType("safetensors")
    safetensors.safe_open = lambda filename, framework, device="cpu": None
    transformers = ModuleType("transformers")
    class TrainerCallback:
        def on_optimizer_step(self, args, state, control, **kwargs):
            return None
    transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    command = build_runtime_command(
        docker=tmp_path / "docker", config_dir=tmp_path / "empty",
        repository="unsloth/unsloth", child_digest="sha256:" + "a" * 64,
        runtime_substage_attribution=substage,
    )
    exec(compile(command.argv[-1], "<runtime-probe>", "exec"), {})
    output = capsys.readouterr().out
    return output.encode("ascii") if return_raw else json.loads(output)


def test_runtime_probe_accepts_immutable_interpreter_site_origin(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    runtime = _execute_runtime_probe(tmp_path, monkeypatch, capsys)
    assert runtime["signatures"]["unsloth.import"] == "GPU_RUNTIME_REQUIRED"
    assert runtime["python_implementation"] == RUNTIME_PYTHON_IMPLEMENTATION
    assert runtime["python"] == RUNTIME_PYTHON_VERSION
    assert set(runtime) == {"python_implementation", "python", "packages", "signatures"}
    assert set(runtime["packages"]) == set(_runtime_value()["packages"])
    assert set(runtime["signatures"]) == set(_runtime_value()["signatures"])


def test_generated_runtime_substage_pass_is_exact_canonical_bytes(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    raw = _execute_runtime_probe(
        tmp_path, monkeypatch, capsys, substage=True, return_raw=True,
    )
    assert isinstance(raw, bytes)
    runtime, failed = image_lock._parse_runtime_substage_evidence(raw)
    assert failed is None and runtime is not None
    assert runtime["python_implementation"] == RUNTIME_PYTHON_IMPLEMENTATION
    assert runtime["python"] == RUNTIME_PYTHON_VERSION
    assert runtime["signatures"]["unsloth.import"] == "GPU_RUNTIME_REQUIRED"
    parsed = json.loads(raw)
    assert raw == json.dumps(
        parsed, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")


@pytest.mark.parametrize(
    ("python_implementation", "python_version"),
    [("PyPy", RUNTIME_PYTHON_VERSION), (RUNTIME_PYTHON_IMPLEMENTATION, "3.11.13")],
)
def test_runtime_probe_rejects_child_interpreter_identity_drift(
    tmp_path: Path, monkeypatch, capsys,
    python_implementation: str, python_version: str,
) -> None:
    with pytest.raises(RuntimeError, match="PYTHON_RUNTIME_INVALID"):
        _execute_runtime_probe(
            tmp_path, monkeypatch, capsys,
            python_implementation=python_implementation,
            python_version=python_version,
        )


@pytest.mark.parametrize(
    ("python_implementation", "python_version"),
    [("PyPy", RUNTIME_PYTHON_VERSION), (RUNTIME_PYTHON_IMPLEMENTATION, "3.11.13")],
)
def test_runtime_substage_probe_attributes_child_interpreter_identity_drift(
    tmp_path: Path, monkeypatch, capsys,
    python_implementation: str, python_version: str,
) -> None:
    raw = _execute_runtime_probe(
        tmp_path, monkeypatch, capsys, substage=True, return_raw=True,
        python_implementation=python_implementation,
        python_version=python_version,
    )
    assert raw == (
        b'{"runtime_substage":"python_runtime",'
        b'"schema_version":"synaptic-hf-training-runtime-substage/v1",'
        b'"status":"FAILED"}'
    )


@pytest.mark.parametrize(
    "attack",
    [
        "spec_missing", "loader_missing", "origin_relative", "origin_outside_site",
        "origin_user_site", "locations_missing", "locations_duplicate",
        "package_root_mismatch",
    ],
)
def test_runtime_probe_rejects_hostile_unsloth_spec_or_origin(
    tmp_path: Path, monkeypatch, capsys, attack: str,
) -> None:
    with pytest.raises(RuntimeError, match="UNSLOTH_(SPEC|ORIGIN|SITE)_INVALID"):
        _execute_runtime_probe(tmp_path, monkeypatch, capsys, attack=attack)


def test_runtime_evidence_accepts_only_exact_closed_schema() -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    image_lock._validate_runtime_evidence(_runtime_value())


def test_candidate_runtime_validates_unchanged_against_canonical_runtime_lock_schema() -> None:
    runtime = _runtime_value()
    before = _raw(runtime)
    assert set(runtime) == {
        "python_implementation", "python", "packages", "signatures",
    }
    assert runtime["python_implementation"] == RUNTIME_PYTHON_IMPLEMENTATION
    assert runtime["python"] == RUNTIME_PYTHON_VERSION
    lock = {
        "schema_version": "synaptic-hf-training-runtime-lock/v1",
        "lock_id": "0" * 64,
        "created_at": "2026-08-21T12:00:00Z",
        "image": validate_oci_documents(_documents()),
        "runtime": runtime,
        "anonymous_loading": {
            "token": False, "trust_remote_code": False, "use_safetensors": True,
        },
    }
    lock["lock_id"] = document_sha256({
        key: value for key, value in lock.items() if key != "lock_id"
    })
    validated = validate_runtime_lock(lock)
    assert validated["runtime"] == runtime
    assert _raw(runtime) == before


@pytest.mark.parametrize(
    "tamper",
    [
        "root_missing", "root_extra", "implementation_missing", "implementation_bool",
        "implementation_control", "implementation_drift", "python_bool", "python_control",
        "python_drift",
        "package_missing", "package_extra", "package_bool", "package_long",
        "signature_missing", "signature_extra", "signature_bool", "signature_control",
        "unsloth_missing", "unsloth_bool", "unsloth_wrong_sentinel", "old_root_unsloth",
    ],
)
def test_runtime_evidence_rejects_hostile_shapes(tamper: str) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    runtime = _runtime_value()
    if tamper == "root_missing":
        runtime.pop("python")
    elif tamper == "root_extra":
        runtime["extra"] = False
    elif tamper == "implementation_missing":
        runtime.pop("python_implementation")
    elif tamper == "implementation_bool":
        runtime["python_implementation"] = True
    elif tamper == "implementation_control":
        runtime["python_implementation"] = RUNTIME_PYTHON_IMPLEMENTATION + "\n"
    elif tamper == "implementation_drift":
        runtime["python_implementation"] = "PyPy"
    elif tamper == "python_bool":
        runtime["python"] = True
    elif tamper == "python_control":
        runtime["python"] = RUNTIME_PYTHON_VERSION + "\n"
    elif tamper == "python_drift":
        runtime["python"] = "3.11.13"
    elif tamper == "package_missing":
        runtime["packages"].pop("unsloth")
    elif tamper == "package_extra":
        runtime["packages"]["unknown"] = "1.0"
    elif tamper == "package_bool":
        runtime["packages"]["torch"] = True
    elif tamper == "package_long":
        runtime["packages"]["torch"] = "x" * 257
    elif tamper == "signature_missing":
        runtime["signatures"].pop("torch.load")
    elif tamper == "signature_extra":
        runtime["signatures"]["FastLanguageModel.from_pretrained"] = "()"
    elif tamper == "signature_bool":
        runtime["signatures"]["torch.load"] = False
    elif tamper == "signature_control":
        runtime["signatures"]["torch.load"] = "(f)\x00"
    elif tamper == "unsloth_missing":
        runtime["signatures"].pop("unsloth.import")
    elif tamper == "unsloth_bool":
        runtime["signatures"]["unsloth.import"] = False
    elif tamper == "unsloth_wrong_sentinel":
        runtime["signatures"]["unsloth.import"] = "IMPORTED"
    else:
        runtime["unsloth"] = {"import": "GPU_RUNTIME_REQUIRED"}
    with pytest.raises(TrainingImageLockError, match="INSPECTOR_INVALID"):
        image_lock._validate_runtime_evidence(runtime)


@pytest.mark.parametrize(
    "diagnostic", [diagnose_runtime_metadata, diagnose_runtime_metadata_attributed],
)
def test_metadata_diagnostic_runs_exact_closed_thirteen_step_flow(
    tmp_path: Path, monkeypatch, diagnostic,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    image = f"{REGISTRY_REPOSITORY}@{documents.requested_digest}"
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"reviewed-docker")
    config = tmp_path / "docker-config"
    config.mkdir()
    events: list[str] = []
    commands: list[CommandSpec] = []
    registry_calls = 0

    monkeypatch.setattr(image_lock.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(image_lock.platform, "python_version", lambda: "3.12.7")

    @contextmanager
    def recording_lock(repository: str, digest: str):
        assert (repository, digest) == (REGISTRY_REPOSITORY, documents.child_digest)
        events.append("lock.enter")
        yield "operation-key"
        events.append("lock.exit")

    monkeypatch.setattr(image_lock, "image_operation_lock", recording_lock)

    def registry_fetcher(reference: str) -> RegistryDocuments:
        nonlocal registry_calls
        registry_calls += 1
        assert reference == image
        events.append("registry")
        return documents

    inspected = _raw({
        "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
        "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    })

    def runner(spec: CommandSpec) -> CommandResult:
        commands.append(spec)
        authority = _authority_result(spec)
        if authority is not None:
            stage = "context" if "context" in spec.argv else ("version" if "version" in spec.argv else "info")
            events.append(stage)
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            events.append("inspect")
            return CommandResult(inspected)
        if "run" in spec.argv:
            events.append("runtime")
            return CommandResult(_runtime_raw())
        raise AssertionError(spec.argv)

    result = diagnostic(
        image=image, docker=docker, docker_config=config, runner=runner,
        registry_fetcher=registry_fetcher,
    )

    assert result == {
        "schema_version": "synaptic-hf-training-image-metadata-diagnostic/v1",
        "status": "PASS",
    }
    assert DIAGNOSTIC_OVERALL_TIMEOUT_SECONDS == 900
    assert events == [
        "registry", "lock.enter", "version", "info", "context", "inspect",
        "runtime", "inspect", "version", "info", "context", "registry", "lock.exit",
    ]
    assert registry_calls == 2 and len(commands) == 9
    assert sum("run" in spec.argv for spec in commands) == 1
    assert sum("image" in spec.argv and "inspect" in spec.argv for spec in commands) == 2
    assert not any("pull" in spec.argv or "save" in spec.argv for spec in commands)
    assert all(spec.timeout_seconds <= 300 for spec in commands)
    assert list(tmp_path.rglob("*.candidate.json")) == []
    assert list(tmp_path.rglob("*.tar")) == []


def test_metadata_diagnostic_rejects_host_before_external_actions(
    tmp_path: Path, monkeypatch,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    monkeypatch.setattr(image_lock.platform, "python_implementation", lambda: "PyPy")
    monkeypatch.setattr(image_lock.platform, "python_version", lambda: "3.12.7")
    with pytest.raises(TrainingImageLockError, match="INSPECTOR_INVALID"):
        diagnose_runtime_metadata(
            image="docker.io/unsloth/unsloth@sha256:" + "a" * 64,
            docker=tmp_path / "missing-docker", docker_config=tmp_path / "missing-config",
            runner=lambda spec: (_ for _ in ()).throw(AssertionError(spec.argv)),
            registry_fetcher=lambda reference: (_ for _ in ()).throw(AssertionError(reference)),
        )


def test_metadata_diagnostic_executes_runtime_at_most_once_on_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"reviewed-docker")
    config = tmp_path / "docker-config"
    config.mkdir()
    monkeypatch.setattr(image_lock.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(image_lock.platform, "python_version", lambda: "3.12.7")

    @contextmanager
    def operation_lock(repository: str, digest: str):
        yield "operation-key"

    monkeypatch.setattr(image_lock, "image_operation_lock", operation_lock)
    runtime_calls = 0

    def runner(spec: CommandSpec) -> CommandResult:
        nonlocal runtime_calls
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(_raw({
                "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
                "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
                "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
            }))
        if "run" in spec.argv:
            runtime_calls += 1
            raise TrainingImageLockError("COMMAND_FAILED")
        raise AssertionError(spec.argv)

    with pytest.raises(TrainingImageLockError, match="COMMAND_FAILED"):
        diagnose_runtime_metadata(
            image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}",
            docker=docker, docker_config=config, runner=runner,
            registry_fetcher=lambda reference: documents,
        )
    assert runtime_calls == 1


@pytest.mark.parametrize(
    ("failed_stage", "category", "reason_code", "hint", "command_failure"),
    [
        ("preflight", "identity", "INSPECTOR_INVALID", None, None),
        ("registry_initial", "document", "EVIDENCE_INVALID", None, None),
        ("operation_lock", "cleanup", "OPERATION_LOCK_CLEANUP_FAILED", None, None),
        ("docker_authority_initial", "nonzero", "COMMAND_FAILED", "nonzero", 1),
        ("cache_identity_initial", "identity", "CACHE_IDENTITY_INVALID", None, 4),
        ("runtime_metadata", "runtime", "COMMAND_FAILED", None, 5),
        ("cache_identity_final", "timeout", "OPERATION_TIMEOUT", None, 6),
        ("docker_authority_final", "identity", "CACHE_IDENTITY_INVALID", None, 7),
        ("registry_final", "document", "EVIDENCE_INVALID", None, None),
        ("final_integrity", "cleanup", "COMMAND_FAILED", "cleanup", None),
    ],
)
def test_attributed_metadata_diagnostic_reports_exact_first_failure(
    tmp_path: Path, monkeypatch, failed_stage: str, category: str,
    reason_code: str, hint: str | None, command_failure: int | None,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    image = f"{REGISTRY_REPOSITORY}@{documents.requested_digest}"
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"reviewed-docker")
    config = tmp_path / "docker-config"
    config.mkdir()
    monkeypatch.setattr(image_lock.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(image_lock.platform, "python_version", lambda: "3.12.7")
    if failed_stage == "preflight":
        monkeypatch.setattr(image_lock.platform, "python_implementation", lambda: "PyPy")

    registry_calls = 0

    def registry_fetcher(reference: str) -> RegistryDocuments:
        nonlocal registry_calls
        registry_calls += 1
        if (
            (failed_stage == "registry_initial" and registry_calls == 1)
            or (failed_stage == "registry_final" and registry_calls == 2)
        ):
            raise TrainingImageLockError(reason_code, diagnostic_category=hint)
        return documents

    @contextmanager
    def operation_lock(repository: str, digest: str):
        yield "operation-key"
        if failed_stage == "operation_lock":
            raise ImageOperationLockError(reason_code)

    monkeypatch.setattr(image_lock, "image_operation_lock", operation_lock)
    inspected = _raw({
        "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
        "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    })
    commands = 0

    def runner(spec: CommandSpec) -> CommandResult:
        nonlocal commands
        commands += 1
        if command_failure == commands:
            raise TrainingImageLockError(reason_code, diagnostic_category=hint)
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(inspected)
        if "run" in spec.argv:
            return CommandResult(_runtime_raw())
        raise AssertionError(spec.argv)

    if failed_stage == "final_integrity":
        original_assert_identity = image_lock._assert_docker_identity
        identity_checks = 0

        def final_identity_check(executable) -> None:
            nonlocal identity_checks
            identity_checks += 1
            if identity_checks == 19:
                raise TrainingImageLockError(reason_code, diagnostic_category=hint)
            original_assert_identity(executable)

        monkeypatch.setattr(image_lock, "_assert_docker_identity", final_identity_check)

    with pytest.raises(MetadataDiagnosticStageError) as caught:
        diagnose_runtime_metadata_attributed(
            image=image, docker=docker, docker_config=config, runner=runner,
            registry_fetcher=registry_fetcher,
        )
    assert str(caught.value) == "DIAGNOSTIC_STAGE_REJECTED"
    assert caught.value.failed_stage == failed_stage
    assert caught.value.category == category
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert vars(caught.value) == {}
    with pytest.raises(AttributeError):
        caught.value.failed_stage = "preflight"


def test_metadata_diagnostic_public_apis_share_one_private_state_machine() -> None:
    default_source = inspect.getsource(diagnose_runtime_metadata)
    attributed_source = inspect.getsource(diagnose_runtime_metadata_attributed)
    assert default_source.count("_diagnose_runtime_metadata(") == 1
    assert attributed_source.count("_diagnose_runtime_metadata(") == 1
    assert inspect.signature(diagnose_runtime_metadata) == inspect.signature(
        diagnose_runtime_metadata_attributed,
    )
    assert inspect.signature(diagnose_runtime_metadata) == inspect.signature(
        diagnose_runtime_substage_attributed,
    )
    assert inspect.signature(diagnose_runtime_metadata) == inspect.signature(
        observe_python_runtime_identity,
    )
    assert inspect.getsource(diagnose_runtime_substage_attributed).count(
        "_diagnose_runtime_metadata(",
    ) == 1
    assert inspect.getsource(observe_python_runtime_identity).count(
        "_diagnose_runtime_metadata(",
    ) == 1


@pytest.mark.parametrize(
    ("hint", "expected"),
    [("nonzero", "nonzero"), ("cleanup", "cleanup"), ("hostile", None)],
)
def test_training_image_lock_error_has_closed_diagnostic_hint(
    hint: str, expected: str | None,
) -> None:
    error = TrainingImageLockError("COMMAND_FAILED", diagnostic_category=hint)
    assert str(error) == "COMMAND_FAILED"
    assert error.reason_code == "COMMAND_FAILED"
    assert error.diagnostic_category == expected


@pytest.mark.parametrize(
    ("stage", "reason", "hint", "expected"),
    [
        ("runtime_metadata", "OPERATION_TIMEOUT", None, "timeout"),
        ("operation_lock", "OPERATION_LOCK_TIMEOUT", None, "timeout"),
        ("operation_lock", "OPERATION_LOCK_CLEANUP_FAILED", None, "cleanup"),
        ("runtime_metadata", "COMMAND_FAILED", None, "runtime"),
        ("registry_final", "CACHE_IDENTITY_INVALID", None, "document"),
        ("cache_identity_final", "CACHE_IDENTITY_INVALID", None, "identity"),
        ("registry_initial", "OPERATION_TIMEOUT", "nonzero", "nonzero"),
        ("preflight", "IMAGE_INVALID", "cleanup", "cleanup"),
    ],
)
def test_metadata_diagnostic_category_mapping_precedence(
    stage: str, reason: str, hint: str | None, expected: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    error = TrainingImageLockError(reason, diagnostic_category=hint)
    assert image_lock._metadata_diagnostic_category(stage, error) == expected


@pytest.mark.parametrize("runtime_substage", [
    "child_unreported", "python_bootstrap", "python_runtime", "workspace_setup",
    "distribution_metadata", "torch_import", "safetensors_import",
    "transformers_import", "signature_introspection", "unsloth_spec",
    "unsloth_origin", "unsloth_package_root", "site_roots", "site_membership",
    "user_site_isolation", "origin_chain", "result_serialization", "invalid_output",
])
def test_runtime_substage_attribution_defers_until_all_postguards(
    tmp_path: Path, monkeypatch, runtime_substage: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    image = f"{REGISTRY_REPOSITORY}@{documents.requested_digest}"
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"reviewed-docker")
    config = tmp_path / "docker-config"
    config.mkdir()
    monkeypatch.setattr(image_lock.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(image_lock.platform, "python_version", lambda: "3.12.7")
    events: list[str] = []

    @contextmanager
    def operation_lock(repository: str, digest: str):
        events.append("lock.enter")
        yield "operation-key"
        events.append("lock.exit")

    monkeypatch.setattr(image_lock, "image_operation_lock", operation_lock)
    registry_calls = 0

    def registry_fetcher(reference: str) -> RegistryDocuments:
        nonlocal registry_calls
        registry_calls += 1
        events.append("registry")
        return documents

    inspected = _raw({
        "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
        "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    })
    commands: list[CommandSpec] = []

    def runner(spec: CommandSpec) -> CommandResult:
        commands.append(spec)
        authority = _authority_result(spec)
        if authority is not None:
            events.append("authority")
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            events.append("inspect")
            return CommandResult(inspected)
        if "run" in spec.argv:
            events.append("runtime")
            if runtime_substage == "child_unreported":
                raise TrainingImageLockError(
                    "COMMAND_FAILED", diagnostic_category="nonzero",
                )
            if runtime_substage == "invalid_output":
                return CommandResult(b"{}")
            return CommandResult(_raw({
                "runtime_substage": runtime_substage,
                "schema_version": "synaptic-hf-training-runtime-substage/v1",
                "status": "FAILED",
            }))
        raise AssertionError(spec.argv)

    with pytest.raises(RuntimeSubstageDiagnosticError) as caught:
        diagnose_runtime_substage_attributed(
            image=image, docker=docker, docker_config=config, runner=runner,
            registry_fetcher=registry_fetcher,
        )
    assert str(caught.value) == "RUNTIME_SUBSTAGE_REJECTED"
    expected_substage = (
        "child_unreported" if runtime_substage == "invalid_output" else runtime_substage
    )
    assert caught.value.runtime_substage == expected_substage
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert vars(caught.value) == {}
    with pytest.raises(AttributeError):
        caught.value.runtime_substage = "child_unreported"
    assert registry_calls == 2 and len(commands) == 9
    assert sum("run" in spec.argv for spec in commands) == 1
    assert sum("image" in spec.argv and "inspect" in spec.argv for spec in commands) == 2
    assert events[-1] == "lock.exit"
    assert events.index("runtime") < max(
        index for index, event in enumerate(events) if event in {"inspect", "authority", "registry"}
    )
    assert not any("pull" in spec.argv or "save" in spec.argv for spec in commands)


def test_runtime_substage_probe_pass_uses_standard_diagnostic_success(
    tmp_path: Path, monkeypatch,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"reviewed-docker")
    config = tmp_path / "docker-config"
    config.mkdir()
    monkeypatch.setattr(image_lock.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(image_lock.platform, "python_version", lambda: "3.12.7")

    @contextmanager
    def operation_lock(repository: str, digest: str):
        yield "operation-key"

    monkeypatch.setattr(image_lock, "image_operation_lock", operation_lock)
    inspected = _raw({
        "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
        "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    })

    def runner(spec: CommandSpec) -> CommandResult:
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(inspected)
        if "run" in spec.argv:
            return CommandResult(_raw({
                "runtime": _runtime_value(),
                "schema_version": "synaptic-hf-training-runtime-substage/v1",
                "status": "PASS",
            }))
        raise AssertionError(spec.argv)

    assert diagnose_runtime_substage_attributed(
        image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}",
        docker=docker, docker_config=config, runner=runner,
        registry_fetcher=lambda reference: documents,
    ) == {
        "schema_version": "synaptic-hf-training-image-metadata-diagnostic/v1",
        "status": "PASS",
    }


@pytest.mark.parametrize(
    "raw", [b"", b"{}", b"not-json", b"x" * 65537],
    ids=["empty", "empty_object", "malformed", "oversize"],
)
def test_runtime_substage_invalid_child_output_is_closed(raw: bytes) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    with pytest.raises(TrainingImageLockError):
        image_lock._parse_runtime_substage_evidence(raw)


@pytest.mark.parametrize("variant", ["whitespace", "reordered", "escaped", "trailing"])
def test_runtime_substage_failure_requires_exact_canonical_bytes(variant: str) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    canonical = image_lock._RUNTIME_SUBSTAGE_FAILURE_BYTES["torch_import"]
    if variant == "whitespace":
        hostile = b" " + canonical
    elif variant == "reordered":
        hostile = (
            b'{"status":"FAILED","runtime_substage":"torch_import",'
            b'"schema_version":"synaptic-hf-training-runtime-substage/v1"}'
        )
    elif variant == "escaped":
        hostile = canonical.replace(b"torch_import", b"torch\\u005fimport")
    else:
        hostile = canonical + b"\n"
    with pytest.raises(TrainingImageLockError, match="INSPECTOR_INVALID"):
        image_lock._parse_runtime_substage_evidence(hostile)


@pytest.mark.parametrize("tamper", ["missing", "extra", "schema", "status", "runtime"])
def test_runtime_substage_pass_rejects_hostile_fields(tamper: str) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    value: dict[str, object] = {
        "runtime": _runtime_value(),
        "schema_version": "synaptic-hf-training-runtime-substage/v1",
        "status": "PASS",
    }
    if tamper == "missing":
        value.pop("runtime")
    elif tamper == "extra":
        value["detail"] = "private"
    elif tamper == "schema":
        value["schema_version"] = "v2"
    elif tamper == "status":
        value["status"] = "FAILED"
    else:
        value["runtime"] = {"python": "3.12.7"}
    with pytest.raises(TrainingImageLockError):
        image_lock._parse_runtime_substage_evidence(_raw(value))


@pytest.mark.parametrize(
    "variant",
    ["leading", "interior", "top_reordered", "nested_reordered", "escaped", "unicode", "trailing"],
)
def test_runtime_substage_pass_requires_exact_canonical_bytes(variant: str) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    value = {
        "runtime": _runtime_value(),
        "schema_version": "synaptic-hf-training-runtime-substage/v1",
        "status": "PASS",
    }
    canonical = _raw(value)
    if variant == "leading":
        hostile = b" " + canonical
    elif variant == "interior":
        hostile = canonical.replace(b'":', b'": ', 1)
    elif variant == "top_reordered":
        hostile = json.dumps(
            {
                "status": "PASS",
                "schema_version": "synaptic-hf-training-runtime-substage/v1",
                "runtime": _runtime_value(),
            },
            ensure_ascii=True, separators=(",", ":"),
        ).encode("ascii")
    elif variant == "nested_reordered":
        runtime = _runtime_value()
        hostile = (
            b'{"runtime":'
            + json.dumps(
                {
                    "signatures": runtime["signatures"],
                    "python": runtime["python"],
                    "packages": runtime["packages"],
                    "python_implementation": runtime["python_implementation"],
                },
                ensure_ascii=True, sort_keys=False, separators=(",", ":"),
            ).encode("ascii")
            + b',"schema_version":"synaptic-hf-training-runtime-substage/v1","status":"PASS"}'
        )
    elif variant == "escaped":
        hostile = canonical.replace(b"/v1", b"\\/v1", 1)
    elif variant == "unicode":
        hostile = canonical.replace(b'"PASS"', b'"\\u0050ASS"', 1)
    else:
        hostile = canonical + b"\n"
    assert hostile != canonical
    with pytest.raises(TrainingImageLockError, match="INSPECTOR_INVALID"):
        image_lock._parse_runtime_substage_evidence(hostile)


def test_runtime_substage_child_failure_is_superseded_by_postguard_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"reviewed-docker")
    config = tmp_path / "docker-config"
    config.mkdir()
    monkeypatch.setattr(image_lock.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(image_lock.platform, "python_version", lambda: "3.12.7")

    @contextmanager
    def operation_lock(repository: str, digest: str):
        yield "operation-key"

    monkeypatch.setattr(image_lock, "image_operation_lock", operation_lock)
    inspected = _raw({
        "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
        "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    })
    inspect_checks = 0
    original_inspect_identity = image_lock._inspect_identity

    def inspect_identity(raw: bytes, *, identity):
        nonlocal inspect_checks
        inspect_checks += 1
        if inspect_checks == 2:
            raise TrainingImageLockError("OPERATION_TIMEOUT")
        return original_inspect_identity(raw, identity=identity)

    monkeypatch.setattr(image_lock, "_inspect_identity", inspect_identity)

    def runner(spec: CommandSpec) -> CommandResult:
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(inspected)
        if "run" in spec.argv:
            return CommandResult(_raw({
                "runtime_substage": "torch_import",
                "schema_version": "synaptic-hf-training-runtime-substage/v1",
                "status": "FAILED",
            }))
        raise AssertionError(spec.argv)

    with pytest.raises(MetadataDiagnosticStageError) as caught:
        diagnose_runtime_substage_attributed(
            image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}",
            docker=docker, docker_config=config, runner=runner,
            registry_fetcher=lambda reference: documents,
        )
    assert caught.value.failed_stage == "cache_identity_final"
    assert caught.value.category == "timeout"
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize("mode", ["observed", "fixed_failure", "nonzero", "invalid"])
def test_python_runtime_identity_uses_one_command_and_all_postguards(
    tmp_path: Path, monkeypatch, mode: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"reviewed-docker")
    config = tmp_path / "docker-config"
    config.mkdir()
    monkeypatch.setattr(image_lock.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(image_lock.platform, "python_version", lambda: "3.12.7")
    events: list[str] = []

    @contextmanager
    def operation_lock(repository: str, digest: str):
        events.append("lock.enter")
        yield "operation-key"
        events.append("lock.exit")

    monkeypatch.setattr(image_lock, "image_operation_lock", operation_lock)
    registry_calls = 0

    def registry_fetcher(reference: str) -> RegistryDocuments:
        nonlocal registry_calls
        registry_calls += 1
        events.append("registry")
        return documents

    inspected = _raw({
        "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
        "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    })
    commands: list[CommandSpec] = []
    observation = _raw({
        "implementation": "CPython",
        "schema_version": "synaptic-hf-training-python-runtime-identity/v1",
        "status": "OBSERVED",
        "version": "3.11.9",
    })

    def runner(spec: CommandSpec) -> CommandResult:
        commands.append(spec)
        authority = _authority_result(spec)
        if authority is not None:
            events.append("authority")
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            events.append("inspect")
            return CommandResult(inspected)
        if "run" in spec.argv:
            events.append("identity")
            assert "import torch" not in spec.argv[-1] and "import sys" in spec.argv[-1]
            if mode == "nonzero":
                raise TrainingImageLockError("COMMAND_FAILED", diagnostic_category="nonzero")
            if mode == "fixed_failure":
                return CommandResult(image_lock._PYTHON_IDENTITY_FAILED_BYTES)
            if mode == "invalid":
                return CommandResult(b"{}")
            return CommandResult(observation)
        raise AssertionError(spec.argv)

    kwargs = {
        "image": f"{REGISTRY_REPOSITORY}@{documents.requested_digest}",
        "docker": docker, "docker_config": config, "runner": runner,
        "registry_fetcher": registry_fetcher,
    }
    if mode == "observed":
        assert observe_python_runtime_identity(**kwargs) == json.loads(observation)
    else:
        with pytest.raises(PythonRuntimeIdentityDiagnosticError) as caught:
            observe_python_runtime_identity(**kwargs)
        assert str(caught.value) == "PYTHON_RUNTIME_IDENTITY_REJECTED"
        assert caught.value.__context__ is None and caught.value.__cause__ is None
        assert vars(caught.value) == {}
    assert registry_calls == 2 and len(commands) == 9
    assert sum("run" in spec.argv for spec in commands) == 1
    assert sum("image" in spec.argv and "inspect" in spec.argv for spec in commands) == 2
    assert events[-1] == "lock.exit"
    assert not any("pull" in spec.argv or "save" in spec.argv for spec in commands)
    assert list(config.iterdir()) == []
    assert list(tmp_path.rglob("*.candidate.json")) == []
    assert list(tmp_path.rglob("*.tar")) == []


@pytest.mark.parametrize("kind", ["manifest", "index"])
@pytest.mark.parametrize("id_mode", ["config", "manifest"])
def test_capture_is_candidate_only_pulls_exact_child_and_writes_once(
    tmp_path: Path, monkeypatch, kind: str, id_mode: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    fake_repository = tmp_path / "repository"
    fake_repository.mkdir()
    monkeypatch.setattr(image_lock, "_repository_root", lambda: fake_repository)
    documents = _documents(kind)
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"synthetic")
    output = tmp_path / "external" / "image-lock.candidate.json"
    output.parent.mkdir()
    runtime = _runtime_raw()
    seen: list[CommandSpec] = []
    archive_seen = []
    lock_seen: list[tuple[str, str]] = []

    @contextmanager
    def recording_lock(repository: str, digest: str):
        lock_seen.append((repository, digest))
        yield "operation-key"

    monkeypatch.setattr(image_lock, "image_operation_lock", recording_lock)

    def runner(spec: CommandSpec) -> CommandResult:
        seen.append(spec)
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "inspect" in spec.argv:
            inspected = {
                "Id": documents.config_digest if id_mode == "config" else documents.child_digest,
                "Os": "linux", "Architecture": "amd64",
                "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
                "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
            }
            return CommandResult(_raw(inspected))
        return CommandResult(runtime if "run" in spec.argv else b"")

    def archive_runner(spec, destination: Path) -> None:
        archive_seen.append(spec)
        _write_archive(destination, documents)

    candidate = capture_candidate(
        image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}", docker=docker,
        docker_config=docker_config, output=output, runner=runner,
        registry_fetcher=lambda _reference: documents,
        archive_runner=archive_runner,
    )
    assert candidate["schema_version"] == CANDIDATE_SCHEMA and candidate["review_status"] == "CANDIDATE_ONLY"
    assert json.loads(output.read_text())["image"]["requested_kind"] == kind
    captured_runtime = candidate["runtime"]
    canonical_lock = {
        "schema_version": "synaptic-hf-training-runtime-lock/v1",
        "lock_id": "0" * 64,
        "created_at": "2026-08-21T12:00:00Z",
        "image": candidate["image"],
        "runtime": captured_runtime,
        "anonymous_loading": candidate["anonymous_loading"],
    }
    canonical_lock["lock_id"] = document_sha256({
        key: value for key, value in canonical_lock.items() if key != "lock_id"
    })
    validated_lock = validate_runtime_lock(canonical_lock)
    assert validated_lock["runtime"] == captured_runtime
    assert candidate["runtime"] is captured_runtime
    assert len(seen) == 9
    assert sum("pull" in spec.argv for spec in seen) == 0
    assert sum("image" in spec.argv and "inspect" in spec.argv for spec in seen) == 2
    assert sum("run" in spec.argv for spec in seen) == 1
    assert len(archive_seen) == 1 and archive_seen[0].argv[-1] == f"unsloth/unsloth@{documents.child_digest}"
    assert archive_seen[0].argv[-3:-1] == ("--platform", "linux/amd64")
    assert lock_seen == [(REGISTRY_REPOSITORY, documents.child_digest)]
    assert archive_seen[0].timeout_seconds == 900
    assert not output.with_name(output.name + ".docker-save.tar.tmp").exists()
    assert candidate["capture"]["ordered_layer_diff_ids"] == [_digest(b"layer")]
    assert candidate["capture"]["archive_format"] == "LEGACY_DOCKER"
    assert candidate["capture"]["compatibility_manifest_sha256"] is None
    assert candidate["capture"]["index_source_annotation_sha256"] is None
    assert candidate["capture"]["local_store_identity"] == {
        "mode": "CONFIG_ID" if id_mode == "config" else "MANIFEST_TARGET_ID",
        "image_id": documents.config_digest if id_mode == "config" else documents.child_digest,
        "repo_digests": [f"unsloth/unsloth@{documents.child_digest}"],
        "platform": "linux/amd64", "rootfs_type": "layers",
        "ordered_layer_diff_ids": [_digest(b"layer")],
    }
    assert candidate["capture"]["docker_authority"]["executable"]["sha256"] == _digest(b"synthetic")
    assert image_lock.CAPTURE_OVERALL_TIMEOUT_SECONDS == 2700
    assert max(spec.timeout_seconds for spec in seen if "run" in spec.argv) == 300
    assert all(spec.timeout_seconds <= 120 for spec in seen if "run" not in spec.argv)
    with pytest.raises(TrainingImageLockError, match="OUTPUT_INVALID"):
        capture_candidate(
            image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}", docker=docker,
            docker_config=docker_config, output=output, runner=runner,
            registry_fetcher=lambda _reference: documents,
            archive_runner=archive_runner,
        )


@pytest.mark.parametrize("drift", ["id", "os", "architecture", "rootfs", "count"])
def test_capture_rejects_local_inspect_identity_before_save(
    tmp_path: Path, monkeypatch, drift: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    fake_repository = tmp_path / "repository"
    fake_repository.mkdir()
    monkeypatch.setattr(image_lock, "_repository_root", lambda: fake_repository)
    documents = _documents()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"synthetic")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    output_dir = tmp_path / "external"
    output_dir.mkdir()
    output = output_dir / "lock.candidate.json"
    inspected = {
        "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
        "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    }
    if drift == "id":
        inspected["Id"] = "sha256:" + "0" * 64
    elif drift == "os":
        inspected["Os"] = "windows"
    elif drift == "architecture":
        inspected["Architecture"] = "arm64"
    elif drift == "rootfs":
        inspected["RootFS"] = {"Type": "hostile", "Layers": [_digest(b"layer")]}
    else:
        inspected["RootFS"] = {"Type": "layers", "Layers": []}

    def runner(spec: CommandSpec) -> CommandResult:
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        return CommandResult(_raw(inspected) if "inspect" in spec.argv else b"")

    archive_calls = 0

    def archive_runner(_spec, _destination: Path) -> None:
        nonlocal archive_calls
        archive_calls += 1

    with pytest.raises(TrainingImageLockError, match="EVIDENCE_INVALID"):
        capture_candidate(
            image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}",
            docker=docker, docker_config=docker_config, output=output,
            runner=runner, registry_fetcher=lambda _reference: documents,
            archive_runner=archive_runner,
        )
    assert archive_calls == 0 and not output.exists()


@pytest.mark.parametrize("annotated", [False, True])
def test_zero_pull_capture_persists_oci_compat_archive_evidence(
    tmp_path: Path, monkeypatch, annotated: bool,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(image_lock, "_repository_root", lambda: repository)
    documents, layer_blob = _oci_documents()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"synthetic")
    config = tmp_path / "docker-config"
    config.mkdir()
    output = tmp_path / "external" / "oci.candidate.json"
    output.parent.mkdir()
    seen: list[CommandSpec] = []

    def runner(spec: CommandSpec) -> CommandResult:
        seen.append(spec)
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(_raw({
                "Id": documents.child_digest,
                "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
                "Os": "linux", "Architecture": "amd64",
                "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
            }))
        if "run" in spec.argv:
            return CommandResult(_runtime_raw())
        raise AssertionError(spec.argv)

    candidate = capture_candidate(
        image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}",
        docker=docker, docker_config=config, output=output, runner=runner,
        registry_fetcher=lambda _reference: documents,
        archive_runner=lambda _spec, destination: _write_oci_compat_archive(
            destination, documents, layer_blob, annotated=annotated,
        ),
    )
    assert not any("pull" in spec.argv for spec in seen)
    assert candidate["capture"]["archive_format"] == "OCI_LAYOUT_COMPAT"
    compatibility_hash = candidate["capture"]["compatibility_manifest_sha256"]
    assert isinstance(compatibility_hash, str) and len(compatibility_hash) == 71
    assert candidate["capture"]["index_source_annotation_sha256"] == (
        _digest(b"unsloth/unsloth") if annotated else None
    )
    assert candidate["capture"]["local_store_identity"]["mode"] == "MANIFEST_TARGET_ID"
    assert not output.with_name(output.name + ".docker-save.tar.tmp").exists()


@pytest.mark.parametrize(
    ("image_id_field", "expected_mode"),
    [("config_digest", "CONFIG_ID"), ("child_digest", "MANIFEST_TARGET_ID")],
)
def test_local_store_identity_accepts_only_two_closed_modes(
    image_id_field: str, expected_mode: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    identity = validate_oci_documents(documents)
    observed = image_lock._inspect_identity(_raw({
        "Id": getattr(documents, image_id_field),
        "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
        "Os": "linux", "Architecture": "amd64",
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    }), identity=identity)
    assert observed.mode == expected_mode
    assert observed.image_id == getattr(documents, image_id_field)
    assert observed.repo_digests == (f"unsloth/unsloth@{documents.child_digest}",)


@pytest.mark.parametrize(
    "case",
    [
        "null", "string", "empty", "malformed", "duplicate", "additional",
        "aliased", "wrong",
    ],
)
def test_local_store_identity_rejects_noncanonical_repo_digests(case: str) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    identity = validate_oci_documents(documents)
    expected = f"unsloth/unsloth@{documents.child_digest}"
    repo_digests: object = {
        "null": None,
        "string": expected,
        "empty": [],
        "malformed": ["malformed"],
        "duplicate": [expected, expected],
        "additional": [expected, f"unsloth/other@{documents.child_digest}"],
        "aliased": [f"docker.io/unsloth/unsloth@{documents.child_digest}"],
        "wrong": ["unsloth/unsloth@sha256:" + "0" * 64],
    }[case]
    with pytest.raises(TrainingImageLockError, match="EVIDENCE_INVALID"):
        image_lock._inspect_identity(_raw({
            "Id": documents.config_digest, "RepoDigests": repo_digests,
            "Os": "linux", "Architecture": "amd64",
            "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
        }), identity=identity)


def test_local_store_identity_rejects_missing_repo_digests_and_third_id() -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    identity = validate_oci_documents(documents)
    missing = {
        "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    }
    third = dict(missing, Id="sha256:" + "0" * 64, RepoDigests=[f"unsloth/unsloth@{documents.child_digest}"])
    for hostile in (missing, third):
        with pytest.raises(TrainingImageLockError, match="EVIDENCE_INVALID"):
            image_lock._inspect_identity(_raw(hostile), identity=identity)


def test_candidate_tool_has_no_canonical_lock_write_capability() -> None:
    with pytest.raises(TrainingImageLockError, match="PROMOTION_FORBIDDEN"):
        canonical_runtime_lock_from_candidate({})


def test_capture_rechecks_anonymous_docker_config_before_inspection(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    fake_repository = tmp_path / "repository"
    fake_repository.mkdir()
    monkeypatch.setattr(image_lock, "_repository_root", lambda: fake_repository)
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"synthetic")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    output = tmp_path / "external" / "lock.candidate.json"
    output.parent.mkdir()
    calls = 0

    def runner(_spec: CommandSpec) -> CommandResult:
        nonlocal calls
        calls += 1
        authority = _authority_result(_spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in _spec.argv and "inspect" in _spec.argv:
            (docker_config / "hostile-config.json").write_text("{}")
        return CommandResult()

    documents = _documents()
    with pytest.raises(TrainingImageLockError, match="DOCKER_CONFIG_INVALID"):
        capture_candidate(
            image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}", docker=docker,
            docker_config=docker_config, output=output, runner=runner,
            registry_fetcher=lambda _reference: documents,
        )
    assert calls == 4 and not output.exists()


def test_candidate_output_inside_repository_is_rejected(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    fake_repository = tmp_path / "repository"
    fake_repository.mkdir()
    monkeypatch.setattr(image_lock, "_repository_root", lambda: fake_repository)
    with pytest.raises(TrainingImageLockError, match="OUTPUT_INVALID"):
        image_lock._fresh_output(fake_repository / "candidate.candidate.json")


def test_subprocess_runner_sanitizes_output_bound_and_timeout(monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    class FakeProcess:
        stdout = io.BytesIO(b"secret-output")
        stderr = io.BytesIO()
        waits = 0
        def wait(self, timeout=None):
            self.waits += 1
            return 0
        def kill(self):
            pass

    popen_kwargs = {}
    def popen(*_args, **kwargs):
        popen_kwargs.update(kwargs)
        return FakeProcess()
    monkeypatch.setattr(image_lock.subprocess, "Popen", popen)
    with pytest.raises(TrainingImageLockError) as caught:
        subprocess_runner(CommandSpec(("docker",), {}, 1, 4))
    assert str(caught.value) == "COMMAND_FAILED" and "secret" not in str(caught.value)
    assert popen_kwargs.get("creationflags") == image_lock.subprocess.CREATE_NEW_PROCESS_GROUP

    process = FakeProcess()
    def timeout(timeout=None):
        process.waits += 1
        if process.waits == 1:
            raise image_lock.subprocess.TimeoutExpired("docker", timeout)
        return -9
    process.wait = timeout
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(TrainingImageLockError, match="OPERATION_TIMEOUT"):
        subprocess_runner(CommandSpec(("docker",), {}, 1, 4))

    class BrokenStream:
        def read(self, _size):
            raise OSError("reader-secret")

    broken = FakeProcess()
    broken.stdout = BrokenStream()
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: broken)
    with pytest.raises(TrainingImageLockError, match="COMMAND_FAILED") as caught:
        subprocess_runner(CommandSpec(("docker",), {}, 1, 4))
    assert "reader-secret" not in str(caught.value)

    nonzero = FakeProcess()
    nonzero.stdout = io.BytesIO()
    nonzero.wait = lambda timeout=None: 17
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: nonzero)
    with pytest.raises(TrainingImageLockError, match="COMMAND_FAILED") as caught:
        subprocess_runner(CommandSpec(("docker",), {}, 1, 4))
    assert caught.value.diagnostic_category == "nonzero"


def test_subprocess_runner_has_one_outer_ownership_handler_and_preserves_preownership_cancel(
    monkeypatch, capsys,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    function = ast.parse(textwrap.dedent(inspect.getsource(subprocess_runner))).body[0]
    outer = function.body[-1]
    assert isinstance(outer, ast.Try)
    assert any(
        isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
        for handler in outer.handlers
    )
    protected = ast.unparse(ast.Module(body=outer.body, type_ignores=[]))
    assert protected.index("process = subprocess.Popen") < protected.index(
        "pipes = [process.stdout, process.stderr]",
    )

    cancellation = KeyboardInterrupt("private-preownership-detail")
    monkeypatch.setattr(
        image_lock.subprocess, "Popen",
        lambda *a, **k: (_ for _ in ()).throw(cancellation),
    )
    monkeypatch.setattr(
        image_lock, "_terminate_process_tree",
        lambda process: (_ for _ in ()).throw(AssertionError("no process is owned")),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        subprocess_runner(CommandSpec(("docker",), {}, 21, 1024))
    assert caught.value is cancellation
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_subprocess_runner_cancellation_cleans_tree_and_readers_before_lock_release(
    monkeypatch, capsys,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    events: list[object] = []
    lock_held = False
    cancellation = KeyboardInterrupt("cancelled")

    class TrackingLock:
        def __enter__(self):
            nonlocal lock_held
            lock_held = True
            events.append("lock.enter")

        def __exit__(self, *_args):
            nonlocal lock_held
            events.append("lock.exit")
            lock_held = False

    class FakeProcess:
        waits = 0

        class Pipe(io.BytesIO):
            def __init__(self, label, value):
                super().__init__(value)
                self.label = label

            def close(self):
                events.append((f"{self.label}.close", lock_held))
                super().close()

        stdout = Pipe("stdout", b"private-child-stdout")
        stderr = Pipe("stderr", b"private-child-stderr")

        def wait(self, timeout=None):
            self.waits += 1
            events.append(("wait", timeout, lock_held))
            if self.waits == 1:
                raise cancellation
            raise image_lock.subprocess.TimeoutExpired("docker", timeout)

        def poll(self):
            events.append(("poll", lock_held))
            return -9

    class Reader:
        instances: list["Reader"] = []

        def __init__(self, *args, **kwargs):
            self.alive = True
            self.instances.append(self)

        def start(self):
            events.append(("reader.start", lock_held))

        def join(self, timeout=None):
            assert process.stdout.closed and process.stderr.closed
            events.append(("reader.join", timeout, lock_held))
            self.alive = False

        def is_alive(self):
            return self.alive

    process = FakeProcess()
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(image_lock.threading, "Thread", Reader)

    def terminate(observed_process):
        assert observed_process is process
        events.append(("terminate-tree", lock_held))

    monkeypatch.setattr(image_lock, "_terminate_process_tree", terminate)
    with pytest.raises(KeyboardInterrupt) as caught:
        with TrackingLock():
            subprocess_runner(CommandSpec(("docker",), {}, 27, 1024))

    assert caught.value is cancellation
    assert events == [
        "lock.enter",
        ("reader.start", True), ("reader.start", True),
        ("wait", 27, True),
        ("terminate-tree", True),
        ("wait", 10, True),
        ("terminate-tree", True),
        ("poll", True),
        ("stdout.close", True), ("stderr.close", True),
        ("reader.join", 5, True), ("reader.join", 5, True),
        "lock.exit",
    ]
    assert all(not reader.is_alive() for reader in Reader.instances)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_subprocess_runner_cleans_direct_pipes_on_immediate_post_popen_cancellation(
    monkeypatch, capsys,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    events: list[object] = []
    lock_held = False
    cancellation = KeyboardInterrupt("private-post-popen-detail")

    class Lock:
        def __enter__(self):
            nonlocal lock_held
            lock_held = True
            events.append("lock.enter")

        def __exit__(self, *_args):
            nonlocal lock_held
            events.append("lock.exit")
            lock_held = False

    class Pipe(io.BytesIO):
        def __init__(self, label, value):
            super().__init__(value)
            self.label = label

        def close(self):
            events.append((f"{self.label}.close", lock_held))
            super().close()

    class Process:
        _stdout = Pipe("stdout", b"private-stdout")
        stderr = Pipe("stderr", b"private-stderr")
        stdout_reads = 0

        @property
        def stdout(self):
            self.stdout_reads += 1
            if self.stdout_reads == 1:
                raise cancellation
            return self._stdout

        def wait(self, timeout=None):
            events.append(("reap", timeout, lock_held))
            raise image_lock.subprocess.TimeoutExpired("docker", timeout)

        def poll(self):
            events.append(("poll", lock_held))
            return -9

    process = Process()
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(
        image_lock, "_terminate_process_tree",
        lambda observed: events.append(("terminate-tree", observed is process, lock_held)),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        with Lock():
            subprocess_runner(CommandSpec(("docker",), {}, 21, 1024))

    assert caught.value is cancellation
    assert events == [
        "lock.enter",
        ("terminate-tree", True, True), ("reap", 10, True),
        ("terminate-tree", True, True), ("poll", True),
        ("stdout.close", True), ("stderr.close", True),
        "lock.exit",
    ]
    assert process._stdout.closed and process.stderr.closed
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("cancelled_start", [1, 2])
def test_subprocess_runner_cleans_owned_process_when_reader_start_is_cancelled(
    monkeypatch, capsys, cancelled_start: int,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    events: list[object] = []
    lock_held = False
    cancellation = KeyboardInterrupt("private-start-detail")

    class Lock:
        def __enter__(self):
            nonlocal lock_held
            lock_held = True
            events.append("lock.enter")

        def __exit__(self, *_args):
            nonlocal lock_held
            events.append("lock.exit")
            lock_held = False

    class Pipe(io.BytesIO):
        def close(self):
            events.append(("pipe.close", lock_held))
            super().close()

    class Process:
        stdout = Pipe(b"private-stdout")
        stderr = Pipe(b"private-stderr")

        def wait(self, timeout=None):
            events.append(("reap", timeout, lock_held))
            raise image_lock.subprocess.TimeoutExpired("docker", timeout)

        def poll(self):
            events.append(("poll", lock_held))
            return -9

    class Reader:
        starts = 0

        def __init__(self, *args, **kwargs):
            self.alive = False

        def start(self):
            type(self).starts += 1
            events.append(("reader.start", type(self).starts, lock_held))
            if type(self).starts == cancelled_start:
                raise cancellation
            self.alive = True

        def join(self, timeout=None):
            assert process.stdout.closed and process.stderr.closed
            events.append(("reader.join", timeout, lock_held))
            self.alive = False

        def is_alive(self):
            return self.alive

    process = Process()
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(image_lock.threading, "Thread", Reader)
    monkeypatch.setattr(
        image_lock, "_terminate_process_tree",
        lambda observed: events.append(("terminate-tree", observed is process, lock_held)),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        with Lock():
            subprocess_runner(CommandSpec(("docker",), {}, 21, 1024))

    assert caught.value is cancellation
    assert events[-1] == "lock.exit"
    assert ("terminate-tree", True, True) in events
    assert ("reap", 10, True) in events
    assert ("poll", True) in events
    assert events.count(("pipe.close", True)) == 2
    assert sum(event == ("reader.join", 5, True) for event in events) == cancelled_start - 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_subprocess_runner_joins_reader_started_before_start_raises(
    monkeypatch, capsys,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    events: list[object] = []
    lock_held = False
    cancellation = KeyboardInterrupt("private-start-return-detail")

    class Lock:
        def __enter__(self):
            nonlocal lock_held
            lock_held = True

        def __exit__(self, *_args):
            nonlocal lock_held
            events.append("lock.exit")
            lock_held = False

    class Pipe(io.BytesIO):
        def close(self):
            events.append(("pipe.close", lock_held))
            super().close()

    class Process:
        stdout = Pipe(b"private-stdout")
        stderr = Pipe(b"private-stderr")

        def wait(self, timeout=None):
            events.append(("reap", timeout, lock_held))
            return -9

    class Reader:
        def __init__(self, *args, **kwargs):
            self.alive = False
            self.ident = None

        def start(self):
            self.alive = True
            self.ident = 73
            events.append(("reader.live", lock_held))
            raise cancellation

        def join(self, timeout=None):
            assert process.stdout.closed and process.stderr.closed
            events.append(("reader.join", timeout, lock_held))
            self.alive = False

        def is_alive(self):
            return self.alive

    process = Process()
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(image_lock.threading, "Thread", Reader)
    monkeypatch.setattr(
        image_lock, "_terminate_process_tree",
        lambda observed: events.append(("terminate-tree", observed is process, lock_held)),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        with Lock():
            subprocess_runner(CommandSpec(("docker",), {}, 21, 1024))

    assert caught.value is cancellation
    assert events == [
        ("reader.live", True), ("terminate-tree", True, True),
        ("reap", 10, True), ("pipe.close", True), ("pipe.close", True),
        ("reader.join", 5, True), "lock.exit",
    ]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_subprocess_runner_reader_construction_failure_is_sanitized_and_cleaned(
    monkeypatch, capsys,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    events: list[object] = []
    lock_held = False

    class Lock:
        def __enter__(self):
            nonlocal lock_held
            lock_held = True

        def __exit__(self, *_args):
            nonlocal lock_held
            events.append("lock.exit")
            lock_held = False

    class Pipe(io.BytesIO):
        def close(self):
            events.append(("pipe.close", lock_held))
            super().close()

    class Process:
        stdout = Pipe(b"private-stdout")
        stderr = Pipe(b"private-stderr")

        def wait(self, timeout=None):
            events.append(("reap", timeout, lock_held))
            return -9

    process = Process()
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(
        image_lock.threading, "Thread",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("private-constructor-detail")),
    )
    monkeypatch.setattr(
        image_lock, "_terminate_process_tree",
        lambda observed: events.append(("terminate-tree", observed is process, lock_held)),
    )
    with pytest.raises(TrainingImageLockError, match="COMMAND_FAILED") as caught:
        with Lock():
            subprocess_runner(CommandSpec(("docker",), {}, 21, 1024))

    assert "private" not in str(caught.value)
    assert caught.value.diagnostic_category == "cleanup"
    assert events == [
        ("terminate-tree", True, True), ("reap", 10, True),
        ("pipe.close", True), ("pipe.close", True), "lock.exit",
    ]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_subprocess_runner_fails_closed_when_owned_process_cannot_be_proven_gone(
    monkeypatch, capsys,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    cancellation = KeyboardInterrupt("private-cancellation-detail")

    class Process:
        stdout = io.BytesIO(b"private-stdout")
        stderr = io.BytesIO(b"private-stderr")
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise cancellation
            raise image_lock.subprocess.TimeoutExpired("docker", timeout)

        def poll(self):
            return None

    process = Process()
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(image_lock, "_terminate_process_tree", lambda observed: None)
    with pytest.raises(TrainingImageLockError, match="COMMAND_FAILED") as caught:
        subprocess_runner(CommandSpec(("docker",), {}, 21, 1024))
    assert caught.value.__cause__ is None
    assert caught.value.diagnostic_category == "cleanup"
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_subprocess_runner_preserves_cancellation_during_owned_pipe_close(
    monkeypatch, capsys,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    cancellation = KeyboardInterrupt("private-pipe-detail")

    class InterruptingPipe(io.BytesIO):
        closes = 0

        def close(self):
            type(self).closes += 1
            if type(self).closes == 1:
                raise cancellation
            super().close()

    class Process:
        stdout = InterruptingPipe(b"")
        stderr = io.BytesIO(b"")

        def wait(self, timeout=None):
            return 0

    process = Process()
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(image_lock, "_terminate_process_tree", lambda observed: None)
    with pytest.raises(KeyboardInterrupt) as caught:
        subprocess_runner(CommandSpec(("docker",), {}, 21, 1024))
    assert caught.value is cancellation
    assert process.stdout.closed and process.stderr.closed
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_candidate_writer_handles_short_writes_and_cleans_fsync_failure(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    destination = tmp_path / "candidate.candidate.json"
    original_write = image_lock.os.write

    def short_write(descriptor: int, payload) -> int:
        return original_write(descriptor, bytes(payload[: max(1, len(payload) // 2)]))

    monkeypatch.setattr(image_lock.os, "write", short_write)
    image_lock._write_candidate(destination, b'{"closed":true}\n')
    assert destination.read_bytes() == b'{"closed":true}\n'

    failed = tmp_path / "failed.candidate.json"
    monkeypatch.setattr(image_lock.os, "fsync", lambda _descriptor: (_ for _ in ()).throw(OSError("fsync-secret")))
    with pytest.raises(TrainingImageLockError, match="OUTPUT_INVALID"):
        image_lock._write_candidate(failed, b"payload")
    assert not failed.exists()


def test_runner_rejects_incomplete_readers_and_terminates_owned_tree(monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    class FakeProcess:
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        killed = 0
        def wait(self, timeout=None):
            return 0
        def kill(self):
            self.killed += 1

    class StuckThread:
        def __init__(self, *args, **kwargs):
            self.started = False
        def start(self):
            self.started = True
        def join(self, timeout=None):
            return None
        def is_alive(self):
            return True

    process = FakeProcess()
    monkeypatch.setattr(image_lock.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(image_lock.threading, "Thread", StuckThread)
    with pytest.raises(TrainingImageLockError, match="COMMAND_FAILED"):
        subprocess_runner(CommandSpec(("docker",), {}, 1, 4))
    assert process.killed >= 1


def test_windows_tree_termination_requests_descendants(monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    class FakeProcess:
        pid = 1234
        def kill(self):
            raise AssertionError("taskkill tree path must be used")

    seen = []
    monkeypatch.setattr(
        image_lock.subprocess, "run",
        lambda argv, **kwargs: seen.append(tuple(argv)) or SimpleNamespace(returncode=0),
    )
    image_lock._terminate_process_tree(FakeProcess())
    assert seen == [("taskkill.exe", "/PID", "1234", "/T", "/F")]


def test_windows_tree_termination_falls_back_when_taskkill_fails(monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    class FakeProcess:
        pid = 1234
        killed = 0
        def kill(self):
            self.killed += 1

    process = FakeProcess()
    monkeypatch.setattr(
        image_lock.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1),
    )
    image_lock._terminate_process_tree(process)
    assert process.killed == 1


@pytest.mark.parametrize("attack", ["executable", "daemon", "reference"])
def test_capture_rejects_docker_authority_or_reference_drift(
    tmp_path: Path, monkeypatch, attack: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    fake_repository = tmp_path / "repository"
    fake_repository.mkdir()
    monkeypatch.setattr(image_lock, "_repository_root", lambda: fake_repository)
    documents = _documents()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"synthetic")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    output = tmp_path / "external" / "lock.candidate.json"
    output.parent.mkdir()
    info_calls = 0
    inspect_calls = 0

    def runner(spec: CommandSpec) -> CommandResult:
        nonlocal info_calls, inspect_calls
        authority = _authority_result(spec)
        if authority is not None:
            if "version" in spec.argv and attack == "executable":
                docker.write_bytes(b"substitute")
            if "info" in spec.argv:
                info_calls += 1
                if attack == "daemon" and info_calls > 1:
                    value = json.loads(authority)
                    value["ID"] = "different-daemon"
                    authority = _raw(value)
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            inspect_calls += 1
            config_id = documents.config_digest
            if attack == "reference" and inspect_calls > 1:
                config_id = "sha256:" + "0" * 64
            return CommandResult(_raw({
                "Id": config_id, "Os": "linux", "Architecture": "amd64",
                "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
                "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
            }))
        if "run" in spec.argv:
            return CommandResult(_runtime_raw())
        return CommandResult()

    with pytest.raises(TrainingImageLockError, match="IMAGE_INVALID|EVIDENCE_INVALID|CACHE_IDENTITY_INVALID"):
        capture_candidate(
            image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}", docker=docker,
            docker_config=docker_config, output=output, runner=runner,
            registry_fetcher=lambda _reference: documents,
            archive_runner=lambda _spec, destination: _write_archive(destination, documents),
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "drift", ["mode_and_id", "third_id", "repo_digests", "platform", "rootfs_type", "diff_ids"],
)
def test_capture_rejects_initial_final_local_store_identity_drift(
    tmp_path: Path, monkeypatch, drift: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    fake_repository = tmp_path / "repository"
    fake_repository.mkdir()
    monkeypatch.setattr(image_lock, "_repository_root", lambda: fake_repository)
    documents = _documents()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"synthetic")
    config = tmp_path / "docker-config"
    config.mkdir()
    output = tmp_path / "external" / "lock.candidate.json"
    output.parent.mkdir()
    inspect_calls = 0

    def runner(spec: CommandSpec) -> CommandResult:
        nonlocal inspect_calls
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            inspect_calls += 1
            inspected = {
                "Id": documents.config_digest,
                "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
                "Os": "linux", "Architecture": "amd64",
                "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
            }
            if inspect_calls == 2:
                if drift == "mode_and_id":
                    inspected["Id"] = documents.child_digest
                elif drift == "third_id":
                    inspected["Id"] = "sha256:" + "0" * 64
                elif drift == "repo_digests":
                    inspected["RepoDigests"] = [f"docker.io/unsloth/unsloth@{documents.child_digest}"]
                elif drift == "platform":
                    inspected["Architecture"] = "arm64"
                elif drift == "rootfs_type":
                    inspected["RootFS"] = {"Type": "hostile", "Layers": [_digest(b"layer")]}
                else:
                    inspected["RootFS"] = {"Type": "layers", "Layers": [_digest(b"different")]}
            return CommandResult(_raw(inspected))
        if "run" in spec.argv:
            return CommandResult(_runtime_raw())
        raise AssertionError(spec.argv)

    with pytest.raises(TrainingImageLockError, match="EVIDENCE_INVALID|CACHE_IDENTITY_INVALID"):
        capture_candidate(
            image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}",
            docker=docker, docker_config=config, output=output, runner=runner,
            registry_fetcher=lambda _reference: documents,
            archive_runner=lambda _spec, destination: _write_archive(destination, documents),
        )
    assert inspect_calls == 2
    assert not output.exists()
    assert not output.with_name(output.name + ".docker-save.tar.tmp").exists()


def _warm_inputs(tmp_path: Path, monkeypatch):
    from tuner.cloud import hf_training_image_lock as image_lock

    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(image_lock, "_repository_root", lambda: repository)
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"synthetic")
    config = tmp_path / "docker-config"
    config.mkdir()
    documents = _documents()
    image = f"{REGISTRY_REPOSITORY}@{documents.requested_digest}"
    return docker, config, documents, image


@pytest.mark.parametrize("id_mode", ["config", "manifest"])
def test_warm_runs_one_exact_pull_with_frozen_deadlines_and_no_artifacts(
    tmp_path: Path, monkeypatch, id_mode: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    docker, config, documents, image = _warm_inputs(tmp_path, monkeypatch)
    seen: list[CommandSpec] = []

    def runner(spec: CommandSpec) -> CommandResult:
        seen.append(spec)
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(_raw({
                "Id": documents.config_digest if id_mode == "config" else documents.child_digest,
                "Os": "linux", "Architecture": "amd64",
                "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
                "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
            }))
        return CommandResult()

    result = image_lock.warm_image_cache(
        image=image, docker=docker, docker_config=config, runner=runner,
        registry_fetcher=lambda _reference: documents,
    )
    pulls = [spec for spec in seen if "pull" in spec.argv]
    assert result["status"] == "CACHE_WARMED"
    assert len(pulls) == 1 and pulls[0].timeout_seconds == 3600
    assert pulls[0].argv[-1] == f"unsloth/unsloth@{documents.child_digest}"
    assert sum("image" in spec.argv and "inspect" in spec.argv for spec in seen) == 1
    assert not any("run" in spec.argv for spec in seen)
    assert list(tmp_path.rglob("*.candidate.json")) == []
    assert list(tmp_path.rglob("*.receipt*")) == []


def test_warm_failure_never_retries_pull(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    docker, config, documents, image = _warm_inputs(tmp_path, monkeypatch)
    pulls = 0

    def runner(spec: CommandSpec) -> CommandResult:
        nonlocal pulls
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "pull" in spec.argv:
            pulls += 1
            raise TrainingImageLockError("COMMAND_FAILED")
        raise AssertionError(spec.argv)

    with pytest.raises(TrainingImageLockError, match="COMMAND_FAILED"):
        image_lock.warm_image_cache(
            image=image, docker=docker, docker_config=config, runner=runner,
            registry_fetcher=lambda _reference: documents,
        )
    assert pulls == 1


@pytest.mark.parametrize("drift", ["id", "os", "architecture", "rootfs", "count"])
def test_warm_rejects_post_pull_cache_identity_drift(
    tmp_path: Path, monkeypatch, drift: str,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    docker, config, documents, image = _warm_inputs(tmp_path, monkeypatch)
    inspected = {
        "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
        "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
        "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
    }
    if drift == "id":
        inspected["Id"] = "sha256:" + "0" * 64
    elif drift == "os":
        inspected["Os"] = "windows"
    elif drift == "architecture":
        inspected["Architecture"] = "arm64"
    elif drift == "rootfs":
        inspected["RootFS"] = {"Type": "hostile", "Layers": [_digest(b"layer")]}
    else:
        inspected["RootFS"] = {"Type": "layers", "Layers": []}
    pulls = 0

    def runner(spec: CommandSpec) -> CommandResult:
        nonlocal pulls
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "pull" in spec.argv:
            pulls += 1
            return CommandResult()
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(_raw(inspected))
        raise AssertionError(spec.argv)

    with pytest.raises(TrainingImageLockError, match="EVIDENCE_INVALID"):
        image_lock.warm_image_cache(
            image=image, docker=docker, docker_config=config, runner=runner,
            registry_fetcher=lambda _reference: documents,
        )
    assert pulls == 1


def test_warm_keys_shared_lock_from_registry_repository_and_child(
    tmp_path: Path, monkeypatch,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    docker, config, documents, image = _warm_inputs(tmp_path, monkeypatch)
    observed: list[tuple[str, str]] = []

    @contextmanager
    def recording_lock(repository: str, digest: str):
        observed.append((repository, digest))
        yield "operation-key"

    def runner(spec: CommandSpec) -> CommandResult:
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(_raw({
                "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
                "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
                "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
            }))
        return CommandResult()

    monkeypatch.setattr(image_lock, "image_operation_lock", recording_lock)
    image_lock.warm_image_cache(
        image=image, docker=docker, docker_config=config, runner=runner,
        registry_fetcher=lambda _reference: documents,
    )
    assert observed == [(REGISTRY_REPOSITORY, documents.child_digest)]


def test_warm_maps_shared_lock_contention_closed(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    docker, config, documents, image = _warm_inputs(tmp_path, monkeypatch)

    @contextmanager
    def busy(*_args, **_kwargs):
        raise ImageOperationLockError("OPERATION_LOCK_TIMEOUT")
        yield

    monkeypatch.setattr(image_lock, "image_operation_lock", busy)
    with pytest.raises(TrainingImageLockError, match="OPERATION_LOCK_TIMEOUT"):
        image_lock.warm_image_cache(
            image=image, docker=docker, docker_config=config,
            runner=lambda _spec: CommandResult(),
            registry_fetcher=lambda _reference: documents,
        )


def test_capture_never_builds_or_executes_pull(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    original = image_lock.build_pull_command
    monkeypatch.setattr(
        image_lock, "build_pull_command",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("capture pull forbidden")),
    )
    try:
        test_capture_is_candidate_only_pulls_exact_child_and_writes_once(
            tmp_path, monkeypatch, "manifest", "config",
        )
    finally:
        monkeypatch.setattr(image_lock, "build_pull_command", original)


def test_registry_aggregate_deadline_fails_before_docker(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    docker, config, documents, image = _warm_inputs(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(image_lock.time, "monotonic", lambda: clock[0])
    def slow_fetch(_reference):
        clock[0] = 121.0
        return documents
    with pytest.raises(TrainingImageLockError, match="OPERATION_TIMEOUT"):
        image_lock.warm_image_cache(
            image=image, docker=docker, docker_config=config,
            runner=lambda _spec: (_ for _ in ()).throw(AssertionError("docker forbidden")),
            registry_fetcher=slow_fetch,
        )


def test_registry_budget_counts_only_active_time_and_clamps_requests(monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents()
    clock = [0.0]
    monkeypatch.setattr(image_lock.time, "monotonic", lambda: clock[0])
    budget = image_lock._RegistryActiveBudget()

    def fetch(_reference):
        clock[0] += 1.0
        return documents

    budget.fetch("ignored", fetch)
    assert budget.active_seconds == 1.0
    clock[0] += 1000.0
    assert budget.remaining_seconds() == 119.0
    budget.fetch("ignored", fetch)
    assert budget.active_seconds == 2.0
    budget.active_seconds = 100.0
    assert budget.request_timeout_seconds() == 20.0
    budget.active_seconds = 119.5
    assert budget.request_timeout_seconds() == 0.5


@pytest.mark.parametrize(
    "documents",
    [
        _documents_with_sizes(1, [MAX_ARCHIVE_BYTES]),
        _documents_with_sizes(1, [1099511627776] * 256),
        _documents_with_sizes(True, [1]),
        _documents_with_sizes(1, [True]),
    ],
)
def test_oversize_or_boolean_oci_sizes_fail_before_lock_or_docker(tmp_path: Path, monkeypatch, documents: RegistryDocuments) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"synthetic")
    config = tmp_path / "docker-config"
    config.mkdir()
    lock_calls = 0
    docker_calls = 0

    @contextmanager
    def forbidden_lock(*_args, **_kwargs):
        nonlocal lock_calls
        lock_calls += 1
        yield

    def forbidden_runner(_spec):
        nonlocal docker_calls
        docker_calls += 1
        return CommandResult()

    monkeypatch.setattr(image_lock, "image_operation_lock", forbidden_lock)
    with pytest.raises(TrainingImageLockError, match="EVIDENCE_INVALID"):
        image_lock.warm_image_cache(
            image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}",
            docker=docker, docker_config=config, runner=forbidden_runner,
            registry_fetcher=lambda _reference: documents,
        )
    assert lock_calls == 0 and docker_calls == 0


def test_aggregate_size_accepts_exact_64_gib_limit() -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    documents = _documents_with_sizes(1, [MAX_ARCHIVE_BYTES - 1])
    identity = image_lock.validate_oci_documents(documents)
    assert identity["config_size"] + sum(layer["size"] for layer in identity["layers"]) == MAX_ARCHIVE_BYTES


def test_frozen_overall_deadlines_clamp_phases_and_fail_closed(monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    assert image_lock.WARM_OVERALL_TIMEOUT_SECONDS == 3900
    assert image_lock.CAPTURE_OVERALL_TIMEOUT_SECONDS == 2700
    readings = iter((0.0, 2699.2, 2700.0))
    monkeypatch.setattr(image_lock.time, "monotonic", lambda: next(readings, 2700.0))
    deadline = image_lock._OperationDeadline.start(2700)
    assert deadline.remaining_seconds(900) == 1
    with pytest.raises(TrainingImageLockError, match="OPERATION_TIMEOUT"):
        deadline.check()


def test_capture_rejects_final_registry_identity_drift(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    fake_repository = tmp_path / "repository"
    fake_repository.mkdir()
    monkeypatch.setattr(image_lock, "_repository_root", lambda: fake_repository)
    documents = _documents()
    docker = tmp_path / "docker.exe"
    docker.write_bytes(b"synthetic")
    config = tmp_path / "docker-config"
    config.mkdir()
    output = tmp_path / "external" / "lock.candidate.json"
    output.parent.mkdir()
    fetches = 0

    def registry_fetcher(_reference: str):
        nonlocal fetches
        fetches += 1
        if fetches == 1:
            return documents
        return replace(documents, requested_raw=documents.requested_raw + b" ")

    def runner(spec: CommandSpec) -> CommandResult:
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(_raw({
                "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
                "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
                "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
            }))
        if "run" in spec.argv:
            return CommandResult(_runtime_raw())
        raise AssertionError(spec.argv)

    with pytest.raises(TrainingImageLockError, match="EVIDENCE_INVALID|CACHE_IDENTITY_INVALID"):
        capture_candidate(
            image=f"{REGISTRY_REPOSITORY}@{documents.requested_digest}",
            docker=docker, docker_config=config, output=output, runner=runner,
            registry_fetcher=registry_fetcher,
            archive_runner=lambda _spec, destination: _write_archive(destination, documents),
        )
    assert fetches == 2 and not output.exists()


def test_capture_lock_contention_creates_no_candidate_or_archive(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock
    from tuner.cloud.hf_training_image_operation_lock import ImageOperationLockError

    docker, config, documents, image = _warm_inputs(tmp_path, monkeypatch)
    output = tmp_path / "external" / "lock.candidate.json"
    output.parent.mkdir()

    @contextmanager
    def busy(*_args, **_kwargs):
        raise ImageOperationLockError("OPERATION_LOCK_TIMEOUT")
        yield

    monkeypatch.setattr(image_lock, "image_operation_lock", busy)
    with pytest.raises(TrainingImageLockError, match="OPERATION_LOCK_TIMEOUT"):
        capture_candidate(
            image=image, docker=docker, docker_config=config, output=output,
            runner=lambda _spec: (_ for _ in ()).throw(AssertionError("docker forbidden")),
            registry_fetcher=lambda _reference: documents,
        )
    assert not output.exists()
    assert not output.with_name(output.name + ".docker-save.tar.tmp").exists()


def test_capture_preserves_archive_race_file_it_did_not_own(
    tmp_path: Path, monkeypatch,
) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock
    from tuner.cloud.hf_training_docker_archive import DockerArchiveError

    docker, config, documents, image = _warm_inputs(tmp_path, monkeypatch)
    output = tmp_path / "external" / "lock.candidate.json"
    output.parent.mkdir()
    archive = output.with_name(output.name + ".docker-save.tar.tmp")

    def runner(spec: CommandSpec) -> CommandResult:
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            return CommandResult(_raw({
                "Id": documents.config_digest, "Os": "linux", "Architecture": "amd64",
                "RepoDigests": [f"unsloth/unsloth@{documents.child_digest}"],
                "RootFS": {"Type": "layers", "Layers": [_digest(b"layer")]},
            }))
        raise AssertionError(spec.argv)

    def racing_archive_runner(_spec, destination: Path) -> None:
        destination.write_bytes(b"foreign-race-winner")
        raise DockerArchiveError("ARCHIVE_OUTPUT_INVALID")

    with pytest.raises(TrainingImageLockError, match="OUTPUT_INVALID"):
        capture_candidate(
            image=image, docker=docker, docker_config=config, output=output,
            runner=runner, registry_fetcher=lambda _reference: documents,
            archive_runner=racing_archive_runner,
        )
    assert archive.read_bytes() == b"foreign-race-winner"
    assert not output.exists()


def test_capture_missing_cache_fails_closed_without_pull_or_save(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_image_lock as image_lock

    docker, config, documents, image = _warm_inputs(tmp_path, monkeypatch)
    output = tmp_path / "external" / "lock.candidate.json"
    output.parent.mkdir()
    seen: list[CommandSpec] = []
    archive_calls = 0

    def runner(spec: CommandSpec) -> CommandResult:
        seen.append(spec)
        authority = _authority_result(spec)
        if authority is not None:
            return CommandResult(authority)
        if "image" in spec.argv and "inspect" in spec.argv:
            raise TrainingImageLockError("COMMAND_FAILED")
        raise AssertionError(spec.argv)

    def archive_runner(_spec, _destination):
        nonlocal archive_calls
        archive_calls += 1

    with pytest.raises(TrainingImageLockError, match="CACHE_IDENTITY_INVALID"):
        capture_candidate(
            image=image, docker=docker, docker_config=config, output=output,
            runner=runner, registry_fetcher=lambda _reference: documents,
            archive_runner=archive_runner,
        )
    assert not any("pull" in spec.argv for spec in seen)
    assert archive_calls == 0 and not output.exists()
