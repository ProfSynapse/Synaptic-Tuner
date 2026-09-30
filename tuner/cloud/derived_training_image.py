"""Config-first qualification for immutable derived training images.

This module intentionally does not mutate the protected historical HF or Modal
runtime locks.  It plans a derived OCI image, captures review-only evidence, and
produces an explicit local diagnostic report after offline verification.  A
report is not launch authority: effectful callers must revalidate the exact
image through an identity-stable Docker executable/config and the same selected
local Docker connection used by the subsequent run.
This is an honest-local-operator consistency boundary, not cryptographic
provenance against a hostile image, Docker daemon, or local administrator.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import yaml
from yaml.resolver import BaseResolver

from tuner.cloud.hf_training_image_lock import (
    CommandResult,
    CommandSpec,
    TrainingImageLockError,
    parse_image_reference,
    subprocess_runner,
)
from tuner.runtime.packaged_training_worker import PACKAGED_TRAINING_WORKER_ENTRYPOINT
from tuner.runtime.packaged_worker_closure import load_packaged_worker_closure, stable_read
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1


PROFILE_SCHEMA = "synaptic-derived-training-image-profile/v1"
BUILD_RECEIPT_SCHEMA = "synaptic-derived-training-image-build-receipt/v1"
CANDIDATE_SCHEMA = "synaptic-derived-training-image-candidate/v1"
VERIFICATION_SCHEMA = "synaptic-derived-training-image-verification/v1"
RELEASE_VERIFICATION_SCHEMA = "synaptic-packaged-runtime-release-verification/v1"
RELEASE_PROMOTION_SCHEMA = "synaptic-packaged-runtime-release-promotion/v1"
FINAL_RUNTIME_CAPTURE_SCHEMA = "synaptic-packaged-runtime-final-capture/v1"
PLATFORM = "linux/amd64"
MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_COMMAND_OUTPUT_BYTES = 1024 * 1024
BUILD_TIMEOUT_SECONDS = 3600
CAPTURE_TIMEOUT_SECONDS = 300

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_VERSION = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.!+_-]{0,126}[A-Za-z0-9])?$")
_TAG = re.compile(
    r"^(?=.{1,255}$)[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
    r":[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$"
)
_REPOSITORY = re.compile(
    r"^(?=.{1,255}$)[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$"
)


class DerivedTrainingImageError(RuntimeError):
    """Closed error for provider-free planning and local Docker maintenance."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _unique_object_pairs(pairs: list[tuple[object, object]]) -> dict[object, object]:
    value: dict[object, object] = {}
    for key, item in pairs:
        try:
            if key in value:
                raise ValueError("duplicate object key")
            value[key] = item
        except TypeError as exc:
            raise ValueError("invalid object key") from exc
    return value


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON number")


def _load_json(raw: bytes) -> object:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_object_pairs,
        parse_constant=_reject_json_constant,
    )


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    return _unique_object_pairs(loader.construct_pairs(node, deep=deep))


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class PackageExpectation:
    distribution: str
    source: str
    version: str | None = None
    url: str | None = None
    commit: str | None = None

    def requirement(self) -> str:
        if self.source == "index":
            assert self.version is not None
            return f"{self.distribution}=={self.version}"
        assert self.url is not None and self.commit is not None
        return f"{self.distribution} @ git+{self.url}@{self.commit}"

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "distribution": self.distribution,
            "source": self.source,
        }
        if self.version is not None:
            value["version"] = self.version
        if self.url is not None:
            value["url"] = self.url
        if self.commit is not None:
            value["commit"] = self.commit
        return value


@dataclass(frozen=True)
class DerivedImageProfile:
    name: str
    base_image: str
    python_executable: str
    packages: tuple[PackageExpectation, ...]
    canonical_sha256: str
    packaged_runtime: dict[str, object] | None = None
    artifact_root: Path | None = None


@dataclass(frozen=True)
class _DockerAuthority:
    executable: Path
    executable_sha256: str
    executable_identity: tuple[int, int, int, int]
    config_directory: Path
    config_identity: tuple[int, int, int]


@dataclass(frozen=True)
class _BuildxAuthority:
    plugin_directory: Path
    plugin_directory_identity: tuple[int, int, int]
    plugin_executable: Path
    plugin_executable_sha256: str
    plugin_executable_identity: tuple[int, int, int, int]
    config_directory: Path
    config_directory_identity: tuple[int, int]
    config_file: Path
    config_file_sha256: str
    config_file_identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class DockerLaunchAuthority:
    """Identity-bound Docker command surface returned by live verification."""

    evidence: Mapping[str, object]
    _authority: _DockerAuthority

    @property
    def environment(self) -> dict[str, str]:
        return _docker_env(self._authority.executable)

    def assert_current(self) -> None:
        _assert_docker_authority(self._authority)

    def command(self, arguments: list[str] | tuple[str, ...]) -> list[str]:
        self.assert_current()
        return [
            str(self._authority.executable),
            "--config",
            str(self._authority.config_directory),
            *arguments,
        ]


Runner = Callable[[CommandSpec], CommandResult]


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DerivedTrainingImageError("DOCUMENT_INVALID") from exc


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _read_object(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        raw = stable_read(path, MAX_DOCUMENT_BYTES)
    except OSError as exc:
        raise DerivedTrainingImageError("DOCUMENT_INVALID") from exc
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        raise DerivedTrainingImageError("DOCUMENT_INVALID")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
        else:
            value = _load_json(raw)
    except (UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise DerivedTrainingImageError("DOCUMENT_INVALID") from exc
    if not isinstance(value, dict):
        raise DerivedTrainingImageError("DOCUMENT_INVALID")
    return value, raw


def _closed(
    value: Mapping[str, object], required: set[str], optional: set[str] | None = None
) -> None:
    if not required <= set(value) or not set(value) <= required | (optional or set()):
        raise DerivedTrainingImageError("DOCUMENT_INVALID")


def _validate_https_git_url(value: object) -> str:
    if not isinstance(value, str) or len(value) > 512:
        raise DerivedTrainingImageError("PROFILE_INVALID")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(".git")
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise DerivedTrainingImageError("PROFILE_INVALID")
    return value


def load_profile(path: Path) -> DerivedImageProfile:
    value, _raw = _read_object(path)
    _closed(
        value,
        {"schema_version", "name", "platform", "base_image", "python_executable", "packages"},
        {"packaged_runtime"},
    )
    if value["schema_version"] != PROFILE_SCHEMA or value["platform"] != PLATFORM:
        raise DerivedTrainingImageError("PROFILE_INVALID")
    name = value["name"]
    base_image = value["base_image"]
    python_executable = value["python_executable"]
    packages = value["packages"]
    if (
        not isinstance(name, str)
        or _NAME.fullmatch(name) is None
        or not isinstance(base_image, str)
        or not isinstance(python_executable, str)
        or not python_executable.startswith("/")
        or len(python_executable) > 256
        or not isinstance(packages, list)
        or not (0 if "packaged_runtime" in value else 1) <= len(packages) <= 64
    ):
        raise DerivedTrainingImageError("PROFILE_INVALID")
    try:
        parse_image_reference(base_image)
    except TrainingImageLockError as exc:
        raise DerivedTrainingImageError("BASE_IMAGE_INVALID") from exc

    expectations: list[PackageExpectation] = []
    seen: set[str] = set()
    for raw_package in packages:
        if not isinstance(raw_package, dict):
            raise DerivedTrainingImageError("PROFILE_INVALID")
        source = raw_package.get("source")
        if source == "index":
            _closed(raw_package, {"distribution", "source", "version"})
        elif source == "vcs":
            _closed(raw_package, {"distribution", "source", "url", "commit"})
        else:
            raise DerivedTrainingImageError("PROFILE_INVALID")
        distribution = raw_package.get("distribution")
        if not isinstance(distribution, str):
            raise DerivedTrainingImageError("PROFILE_INVALID")
        normalized = re.sub(r"[-_.]+", "-", distribution.lower())
        if _NAME.fullmatch(normalized) is None or normalized in seen:
            raise DerivedTrainingImageError("PROFILE_INVALID")
        seen.add(normalized)
        if source == "index":
            version = raw_package.get("version")
            if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
                raise DerivedTrainingImageError("PROFILE_INVALID")
            expectations.append(PackageExpectation(normalized, "index", version=version))
        else:
            commit = raw_package.get("commit")
            if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
                raise DerivedTrainingImageError("PROFILE_INVALID")
            expectations.append(
                PackageExpectation(
                    normalized,
                    "vcs",
                    url=_validate_https_git_url(raw_package.get("url")),
                    commit=commit,
                )
            )

    normalized_profile = {
        "schema_version": PROFILE_SCHEMA,
        "name": name,
        "platform": PLATFORM,
        "base_image": base_image,
        "python_executable": python_executable,
        "packages": [item.to_dict() for item in expectations],
    }
    packaged = value.get("packaged_runtime")
    if packaged is not None:
        _validate_packaged_profile(packaged, python_executable)
        normalized_profile["packaged_runtime"] = packaged
    return DerivedImageProfile(
        name=name,
        base_image=base_image,
        python_executable=python_executable,
        packages=tuple(expectations),
        canonical_sha256=_sha256(_canonical_bytes(normalized_profile)),
        packaged_runtime=packaged,
        artifact_root=path.absolute().parent,
    )


def render_dockerfile(profile: DerivedImageProfile) -> str:
    requirements = " \\\n    ".join(shlex.quote(item.requirement()) for item in profile.packages)
    result = (
        f"FROM {profile.base_image}\n"
        f"LABEL ai.synapticlabs.derived-training-profile={json.dumps(profile.canonical_sha256)} \\\n"
        f"      ai.synapticlabs.derived-training-base={json.dumps(profile.base_image)}\n"
    )
    if requirements:
        result += f"RUN {shlex.quote(profile.python_executable)} -m pip install --no-cache-dir --upgrade \\\n    {requirements}\n"
    if profile.packaged_runtime is not None:
        digest = hashlib.sha256(_canonical_bytes(profile.packaged_runtime)).hexdigest()
        result += ("ARG SYNAPTIC_RUNTIME_INPUTS_SHA256\n"
                   f"RUN test \"$SYNAPTIC_RUNTIME_INPUTS_SHA256\" = '{digest}'\n"
                   "COPY runtime/ /opt/synaptic-runtime/\n"
                   f"RUN echo '{digest}  /opt/synaptic-runtime/build-inputs.json' | sha256sum -c -\n"
                   "RUN set -eu; \\\n"
                   f"    before=\"$({shlex.quote(profile.python_executable)} -I -m pip check 2>&1)\" || test \"$?\" -eq 1; \\\n"
                   f"    {shlex.quote(profile.python_executable)} -I -m pip install --no-index --no-deps --require-hashes --no-cache-dir -r /opt/synaptic-runtime/requirements.txt; \\\n"
                   f"    after=\"$({shlex.quote(profile.python_executable)} -I -m pip check 2>&1)\" || test \"$?\" -eq 1; \\\n"
                   "    test \"$before\" = \"$after\"\n")
    return result


def _validate_packaged_profile(value: object, executable: str) -> None:
    if not isinstance(value, dict):
        raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
    _closed(value, {"wheel", "bootstrap", "python", "capabilities"})
    wheels = [value["wheel"], *value["bootstrap"]] if isinstance(value["bootstrap"], list) else []
    if not 2 <= len(wheels) <= 128:
        raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
    names, filenames = set(), set()
    for wheel in wheels:
        if not isinstance(wheel, dict): raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
        _closed(wheel, {"filename", "distribution", "version", "sha256"})
        name, filename = wheel["distribution"], wheel["filename"]
        if (not isinstance(name, str) or _NAME.fullmatch(name) is None or name in names
                or re.sub(r"[-_.]+", "-", name) != name
                or not isinstance(filename, str) or re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename) is None or filename in filenames
                or not isinstance(wheel["version"], str) or _VERSION.fullmatch(wheel["version"]) is None
                or not isinstance(wheel["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"]) is None):
            raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
        names.add(name); filenames.add(filename)
    if value["wheel"]["distribution"] != "synaptic-tuner": raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
    python = value["python"]
    if not isinstance(python, dict): raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
    _closed(python, {"implementation", "version", "executable", "executable_digest", "purelib", "platlib"})
    if python["executable"] != executable or python["implementation"] != "cpython" or re.fullmatch(r"[0-9a-f]{64}", str(python["executable_digest"])) is None:
        raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
    for key in ("executable", "purelib", "platlib"):
        if not isinstance(python[key], str) or re.fullmatch(r"/[A-Za-z0-9_./+-]+", python[key]) is None or ".." in python[key].split("/"):
            raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
    capability = value["capabilities"]
    if not isinstance(capability, dict): raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
    _closed(capability, {"compatibility", "contracts"})
    # Shared release parser validates capability semantics at reconstruction;
    # here enforce closed transport and bounded canonical bytes.
    if not isinstance(capability["compatibility"], dict) or not isinstance(capability["contracts"], dict): raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")
    _closed(capability["compatibility"], {"methods", "models", "dataset_formats"})
    _closed(capability["contracts"], {"workload_schema", "prepared_input_schema", "artifact_contract_schema"})
    if len(_canonical_bytes(value)) > MAX_DOCUMENT_BYTES: raise DerivedTrainingImageError("PACKAGED_PROFILE_INVALID")


def _stage_packaged_inputs(profile: DerivedImageProfile, context: Path) -> None:
    if profile.packaged_runtime is None: return
    value = profile.packaged_runtime
    target = context / "runtime"
    target.mkdir()
    requirements = []
    for wheel in [value["wheel"], *value["bootstrap"]]:
        raw = stable_read(profile.artifact_root / wheel["filename"], 256 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != wheel["sha256"]: raise DerivedTrainingImageError("WHEEL_ARTIFACT_INVALID")
        _write_exclusive_bytes(target / wheel["filename"], raw)
        requirements.append(f"/opt/synaptic-runtime/{wheel['filename']} --hash=sha256:{wheel['sha256']}")
    _write_exclusive_bytes(target / "requirements.txt", ("\n".join(requirements) + "\n").encode())
    _write_exclusive(target / "build-inputs.json", value)


def plan_profile(profile_path: Path) -> dict[str, object]:
    profile = load_profile(profile_path)
    dockerfile = render_dockerfile(profile)
    return {
        "schema_version": "synaptic-derived-training-image-plan/v1",
        "status": "PLAN_ONLY",
        "profile": profile.name,
        "profile_sha256": profile.canonical_sha256,
        "base_image": profile.base_image,
        "platform": PLATFORM,
        "packages": [item.to_dict() for item in profile.packages],
        "dockerfile": dockerfile,
        "dockerfile_sha256": _sha256(dockerfile.encode("utf-8")),
    }


def _validate_tag(tag: str) -> tuple[str, str]:
    if "@" in tag or _TAG.fullmatch(tag) is None:
        raise DerivedTrainingImageError("OUTPUT_TAG_INVALID")
    repository, separator, _tag = tag.rpartition(":")
    if not separator or not repository:
        raise DerivedTrainingImageError("OUTPUT_TAG_INVALID")
    return tag, repository


def _validate_oci_reference(reference: object) -> tuple[str, str]:
    if not isinstance(reference, str) or reference.count("@") != 1:
        raise DerivedTrainingImageError("OCI_REFERENCE_INVALID")
    repository, digest = reference.split("@", 1)
    if _REPOSITORY.fullmatch(repository) is None or _DIGEST.fullmatch(digest) is None:
        raise DerivedTrainingImageError("OCI_REFERENCE_INVALID")
    return repository, digest


_WINDOWS = os.name == "nt"
_PATH_SEPARATOR = os.pathsep


def _docker_search_path(docker: Path) -> str:
    """Return a closed search path without inheriting ambient executables."""

    if _WINDOWS and _PATH_SEPARATOR in str(docker.parent):
        raise DerivedTrainingImageError("DOCKER_INVALID")
    return os.defpath


def _docker_env(docker: Path, *, buildx: bool = False) -> dict[str, str]:
    environment = {
        "DOCKER_CONTENT_TRUST": "1",
        "PATH": _docker_search_path(docker),
    }
    if buildx:
        environment["BUILDX_METADATA_PROVENANCE"] = "max"
    return environment


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise DerivedTrainingImageError("DOCKER_INVALID") from exc
    return "sha256:" + digest.hexdigest()


def _validate_docker_inputs(docker: Path, docker_config: Path) -> _DockerAuthority:
    try:
        if (
            not docker.is_absolute()
            or docker.is_symlink()
            or not docker.is_file()
            or not docker_config.is_absolute()
            or docker_config.is_symlink()
            or not docker_config.is_dir()
            or any(docker_config.iterdir())
        ):
            raise OSError("invalid Docker inputs")
        executable = docker.resolve(strict=True)
        config_directory = docker_config.resolve(strict=True)
        executable_stat = executable.stat()
        config_stat = config_directory.stat()
    except OSError as exc:
        raise DerivedTrainingImageError("DOCKER_INVALID") from exc
    return _DockerAuthority(
        executable=executable,
        executable_sha256=_file_sha256(executable),
        executable_identity=(
            executable_stat.st_dev,
            executable_stat.st_ino,
            executable_stat.st_size,
            executable_stat.st_mtime_ns,
        ),
        config_directory=config_directory,
        config_identity=(
            config_stat.st_dev,
            config_stat.st_ino,
            config_stat.st_mtime_ns,
        ),
    )


def _assert_docker_authority(authority: _DockerAuthority) -> None:
    try:
        if (
            authority.executable.is_symlink()
            or not authority.executable.is_file()
            or authority.config_directory.is_symlink()
            or not authority.config_directory.is_dir()
            or any(authority.config_directory.iterdir())
        ):
            raise OSError("Docker authority changed")
        executable_stat = authority.executable.stat()
        config_stat = authority.config_directory.stat()
    except OSError as exc:
        raise DerivedTrainingImageError("DOCKER_AUTHORITY_CHANGED") from exc
    if (
        (
            executable_stat.st_dev,
            executable_stat.st_ino,
            executable_stat.st_size,
            executable_stat.st_mtime_ns,
        )
        != authority.executable_identity
        or (
            config_stat.st_dev,
            config_stat.st_ino,
            config_stat.st_mtime_ns,
        )
        != authority.config_identity
        or _file_sha256(authority.executable) != authority.executable_sha256
    ):
        raise DerivedTrainingImageError("DOCKER_AUTHORITY_CHANGED")


def _windows_buildx_plugin(docker: Path) -> tuple[Path, Path] | None:
    if not _WINDOWS:
        return None
    plugin_directory = docker.parent.parent / "cli-plugins"
    plugin_executable = plugin_directory / "docker-buildx.exe"
    if (
        _PATH_SEPARATOR in str(plugin_directory)
        or not plugin_directory.is_absolute()
    ):
        raise DerivedTrainingImageError("DOCKER_INVALID")
    return plugin_directory, plugin_executable


def _create_buildx_authority(
    *, docker: Path, config_directory: Path
) -> _BuildxAuthority | None:
    plugin = _windows_buildx_plugin(docker)
    if plugin is None:
        return None
    plugin_directory, plugin_executable = plugin
    config_file = config_directory / "config.json"
    try:
        if (
            plugin_directory.is_symlink()
            or not plugin_directory.is_dir()
            or plugin_executable.is_symlink()
            or not plugin_executable.is_file()
            or config_directory.is_symlink()
            or not config_directory.is_dir()
            or any(config_directory.iterdir())
        ):
            raise OSError("invalid Buildx authority")
        plugin_directory = plugin_directory.resolve(strict=True)
        plugin_executable = plugin_executable.resolve(strict=True)
        if plugin_executable.parent != plugin_directory:
            raise OSError("invalid Buildx layout")
        payload = {
            "cliPluginsExtraDirs": [str(plugin_directory)],
        }
        _write_exclusive(config_file, payload)
        plugin_directory_stat = plugin_directory.stat()
        plugin_executable_stat = plugin_executable.stat()
        config_directory_stat = config_directory.stat()
        config_file_stat = config_file.stat()
    except (OSError, DerivedTrainingImageError) as exc:
        raise DerivedTrainingImageError("DOCKER_INVALID") from exc
    return _BuildxAuthority(
        plugin_directory=plugin_directory,
        plugin_directory_identity=(
            plugin_directory_stat.st_dev,
            plugin_directory_stat.st_ino,
            plugin_directory_stat.st_mtime_ns,
        ),
        plugin_executable=plugin_executable,
        plugin_executable_sha256=_file_sha256(plugin_executable),
        plugin_executable_identity=(
            plugin_executable_stat.st_dev,
            plugin_executable_stat.st_ino,
            plugin_executable_stat.st_size,
            plugin_executable_stat.st_mtime_ns,
        ),
        config_directory=config_directory,
        config_directory_identity=(
            config_directory_stat.st_dev,
            config_directory_stat.st_ino,
        ),
        config_file=config_file,
        config_file_sha256=_file_sha256(config_file),
        config_file_identity=(
            config_file_stat.st_dev,
            config_file_stat.st_ino,
            config_file_stat.st_size,
            config_file_stat.st_mtime_ns,
        ),
    )


def _assert_buildx_authority(authority: _BuildxAuthority | None) -> None:
    if authority is None:
        return
    try:
        if (
            authority.plugin_directory.is_symlink()
            or not authority.plugin_directory.is_dir()
            or authority.plugin_executable.is_symlink()
            or not authority.plugin_executable.is_file()
            or authority.plugin_executable.parent != authority.plugin_directory
        ):
            raise OSError("Buildx authority changed")
        if authority.config_directory.is_symlink() or not authority.config_directory.is_dir():
            raise OSError("Buildx authority changed")
        if authority.config_file.is_symlink() or not authority.config_file.is_file():
            raise OSError("Buildx authority changed")
        state_directory = authority.config_directory / "buildx"
        token_seed = authority.config_directory / ".token_seed"
        token_seed_lock = authority.config_directory / ".token_seed.lock"
        members = set(authority.config_directory.iterdir())
        if (authority.config_file not in members
                or not members <= {authority.config_file, state_directory, token_seed, token_seed_lock}
                or (token_seed in members) != (token_seed_lock in members)):
            raise OSError("Buildx authority changed")
        if state_directory in members and (
            state_directory.is_symlink() or not state_directory.is_dir()
        ):
            raise OSError("Buildx authority changed")
        for path in (token_seed, token_seed_lock):
            if path in members and (
                path.is_symlink() or not path.is_file() or path.stat().st_size > 4096
            ):
                raise OSError("Buildx authority changed")
        plugin_directory_stat = authority.plugin_directory.stat()
        plugin_executable_stat = authority.plugin_executable.stat()
        config_directory_stat = authority.config_directory.stat()
        config_file_stat = authority.config_file.stat()
        if (
            (
                plugin_directory_stat.st_dev,
                plugin_directory_stat.st_ino,
                plugin_directory_stat.st_mtime_ns,
            )
            != authority.plugin_directory_identity
            or (
                plugin_executable_stat.st_dev,
                plugin_executable_stat.st_ino,
                plugin_executable_stat.st_size,
                plugin_executable_stat.st_mtime_ns,
            )
            != authority.plugin_executable_identity
            # Buildx may create and remove temporary entries here. The sole
            # surviving config file, its identity, and its bytes are checked.
            or (
                config_directory_stat.st_dev,
                config_directory_stat.st_ino,
            )
            != authority.config_directory_identity
            or (
                config_file_stat.st_dev,
                config_file_stat.st_ino,
                config_file_stat.st_size,
                config_file_stat.st_mtime_ns,
            )
            != authority.config_file_identity
            or _file_sha256(authority.plugin_executable)
            != authority.plugin_executable_sha256
            or _file_sha256(authority.config_file) != authority.config_file_sha256
        ):
            raise OSError("Buildx authority changed")
    except (OSError, DerivedTrainingImageError) as exc:
        raise DerivedTrainingImageError("DOCKER_AUTHORITY_CHANGED") from exc


def _run_authorized(
    authority: _DockerAuthority, runner: Runner, command: CommandSpec
) -> CommandResult:
    _assert_docker_authority(authority)
    result = runner(command)
    _assert_docker_authority(authority)
    return result


def _run_buildx_authorized(
    authority: _DockerAuthority,
    buildx_authority: _BuildxAuthority | None,
    runner: Runner,
    command: CommandSpec,
) -> CommandResult:
    _assert_docker_authority(authority)
    _assert_buildx_authority(buildx_authority)
    result = runner(command)
    _assert_buildx_authority(buildx_authority)
    _assert_docker_authority(authority)
    return result


def build_command(
    *, docker: Path, docker_config: Path, tag: str, dockerfile: Path, metadata_file: Path,
    runtime_inputs_digest: str | None = None,
) -> CommandSpec:
    _validate_tag(tag)
    return CommandSpec(
        argv=(
            str(docker), "--config", str(docker_config), "buildx", "build",
            "--pull", "--no-cache", "--platform", PLATFORM, "--load",
            "--provenance=mode=max", "--tag", tag, "--metadata-file",
            str(metadata_file), "--file", str(dockerfile),
            *(("--build-arg", "SYNAPTIC_RUNTIME_INPUTS_SHA256=" + runtime_inputs_digest) if runtime_inputs_digest is not None else ()),
            str(dockerfile.parent),
        ),
        env=_docker_env(docker, buildx=True),
        timeout_seconds=BUILD_TIMEOUT_SECONDS,
        maximum_output_bytes=MAX_COMMAND_OUTPUT_BYTES,
    )


def _docker_argv(docker: Path, docker_config: Path | None) -> tuple[str, ...]:
    if docker_config is None:
        return (str(docker),)
    return (str(docker), "--config", str(docker_config))


def _inspect_command(
    *, docker: Path, docker_config: Path | None, image: str
) -> CommandSpec:
    return CommandSpec(
        argv=_docker_argv(docker, docker_config) + (
            "image", "inspect",
            "--format",
            '{{json .Id}}\n{{json .RepoDigests}}\n{{json .Config.Labels}}',
            image,
        ),
        env=_docker_env(docker),
        timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
        maximum_output_bytes=MAX_COMMAND_OUTPUT_BYTES,
    )


def _parse_inspect(raw: bytes) -> tuple[str, tuple[str, ...], dict[str, str]]:
    try:
        lines = raw.decode("utf-8").splitlines()
        if len(lines) != 3:
            raise ValueError("line count")
        image_id, repo_digests, labels = (
            _load_json(line.encode("utf-8")) for line in lines
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID") from exc
    if (
        not isinstance(image_id, str)
        or _DIGEST.fullmatch(image_id) is None
        or not isinstance(repo_digests, list)
        or any(not isinstance(item, str) or "@sha256:" not in item for item in repo_digests)
        or not isinstance(labels, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in labels.items())
    ):
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID")
    return image_id, tuple(repo_digests), labels


def _runtime_inspector(expectations: tuple[PackageExpectation, ...]) -> str:
    names = [item.distribution for item in expectations]
    return ";".join(
        (
            "import sys,sysconfig,json",
            "p=sysconfig.get_paths()",
            "sys.path.extend(dict.fromkeys(x for x in (p.get('purelib'),p.get('platlib')) if x))",
            "import importlib.metadata as m",
            f"names={names!r}",
            "out={}",
            "[(lambda d,n:out.__setitem__(n,{'version':d.version,'direct_url':json.loads(d.read_text('direct_url.json')) if (d.read_text('direct_url.json') or '').strip() else None}))(m.distribution(n),n) for n in names]",
            "print(json.dumps({'schema_version':'synaptic-derived-training-image-runtime/v1','packages':out},sort_keys=True,separators=(',',':')))"
        )
    )


def _runtime_command(
    *, docker: Path, docker_config: Path | None, image: str, profile: DerivedImageProfile
) -> CommandSpec:
    return CommandSpec(
        argv=_docker_argv(docker, docker_config) + (
            "run", "--rm",
            "--network", "none", "--entrypoint", profile.python_executable,
            image, "-I", "-S", "-c", _runtime_inspector(profile.packages),
        ),
        env=_docker_env(docker),
        timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
        maximum_output_bytes=MAX_COMMAND_OUTPUT_BYTES,
    )


def _write_exclusive(path: Path, value: object) -> None:
    _write_exclusive_bytes(path, _canonical_bytes(value))


def _write_exclusive_bytes(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if stable_read(path, len(payload)) != payload:
            raise OSError("readback mismatch")
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        # Preserve partial evidence for diagnosis; never unlink a substituted
        # pathname after a failed write/readback.
        raise DerivedTrainingImageError("OUTPUT_INVALID") from exc


def build_derived_image(
    *, profile_path: Path, docker: Path, docker_config: Path, tag: str,
    output: Path, runner: Runner = subprocess_runner,
) -> dict[str, object]:
    profile = load_profile(profile_path)
    tag, repository = _validate_tag(tag)
    if output.exists() or output.is_symlink() or output.suffix.lower() != ".json":
        raise DerivedTrainingImageError("OUTPUT_INVALID")
    authority = _validate_docker_inputs(docker, docker_config)
    dockerfile_text = render_dockerfile(profile)
    with tempfile.TemporaryDirectory(prefix="syntunia-derived-image-") as raw_tmp:
        temporary = Path(raw_tmp)
        dockerfile = temporary / "Dockerfile"
        metadata_path = temporary / "build-metadata.json"
        buildx_config = temporary / "docker-config"
        buildx_config.mkdir()
        buildx_authority = _create_buildx_authority(
            docker=authority.executable,
            config_directory=buildx_config,
        )
        command_config = (
            buildx_authority.config_directory
            if buildx_authority is not None
            else authority.config_directory
        )
        dockerfile.write_text(dockerfile_text, encoding="utf-8", newline="\n")
        _stage_packaged_inputs(profile, temporary)
        _run_buildx_authorized(authority, buildx_authority, runner, build_command(
            docker=authority.executable,
            docker_config=command_config,
            tag=tag,
            dockerfile=dockerfile, metadata_file=metadata_path,
            runtime_inputs_digest=(hashlib.sha256(_canonical_bytes(profile.packaged_runtime)).hexdigest() if profile.packaged_runtime is not None else None),
        ))
        try:
            metadata_raw = stable_read(metadata_path, MAX_DOCUMENT_BYTES)
            if not metadata_raw or len(metadata_raw) > MAX_DOCUMENT_BYTES:
                raise ValueError("metadata size")
            metadata = _load_json(metadata_raw)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID") from exc
        if not isinstance(metadata, dict):
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID")
        _validate_packaged_build_metadata(metadata, profile)
        try:
            metadata_bytes = _canonical_bytes(metadata)
        except DerivedTrainingImageError as exc:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID") from exc
        if len(metadata_bytes) > MAX_DOCUMENT_BYTES:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID")
        final_digest = metadata.get("containerimage.digest")
        config_digest = metadata.get("containerimage.config.digest")
        if not isinstance(final_digest, str) or _DIGEST.fullmatch(final_digest) is None:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID:FINAL_DIGEST")
        config_digest_omitted = "containerimage.config.digest" not in metadata
        if config_digest_omitted and profile.packaged_runtime is None:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID:CONFIG_DIGEST")
        if not config_digest_omitted and (
            not isinstance(config_digest, str) or _DIGEST.fullmatch(config_digest) is None
        ):
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID:CONFIG_DIGEST")
        inspect = _run_authorized(
            authority,
            runner,
            _inspect_command(
                docker=authority.executable,
                docker_config=authority.config_directory,
                image=tag,
            ),
        )
        image_id, repo_digests, labels = _parse_inspect(inspect.stdout)
        final_reference = f"{repository}@{final_digest}"
        if config_digest_omitted:
            # Docker's local multi-platform export may omit the config digest.
            # In that case, use the inspected image ID only when the same
            # inspection binds this tag to Buildx's exact final OCI digest.
            if final_reference not in repo_digests:
                raise DerivedTrainingImageError("BUILD_METADATA_INVALID:REPO_DIGEST")
            config_digest = image_id
        if image_id != config_digest:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID:IMAGE_CONFIG_DIGEST")
        if labels.get("ai.synapticlabs.derived-training-profile") != profile.canonical_sha256:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID:PROFILE_LABEL")
        if labels.get("ai.synapticlabs.derived-training-base") != profile.base_image:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID:BASE_LABEL")
        receipt = {
            "schema_version": BUILD_RECEIPT_SCHEMA,
            "status": "BUILT_NOT_ATTESTED",
            "profile_sha256": profile.canonical_sha256,
            "base_image": profile.base_image,
            "dockerfile_sha256": _sha256(dockerfile_text.encode("utf-8")),
            "tag": tag,
            "final_oci_reference": final_reference,
            "final_oci_digest": final_digest,
            "image_config_digest": config_digest,
            "repo_digests": list(repo_digests),
            "build_metadata": metadata,
            "build_metadata_sha256": _sha256(metadata_bytes),
        }
    _write_exclusive(output, receipt)
    return receipt


def _parse_runtime(raw: bytes) -> dict[str, object]:
    try:
        value = _load_json(raw)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise DerivedTrainingImageError("RUNTIME_EVIDENCE_INVALID") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "packages"}
        or value.get("schema_version") != "synaptic-derived-training-image-runtime/v1"
        or not isinstance(value.get("packages"), dict)
    ):
        raise DerivedTrainingImageError("RUNTIME_EVIDENCE_INVALID")
    return value


def _runtime_payload_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DerivedTrainingImageError("RUNTIME_EVIDENCE_INVALID") from exc


def capture_candidate(
    *, profile_path: Path, build_receipt_path: Path, docker: Path,
    docker_config: Path, output: Path, runner: Runner = subprocess_runner,
) -> dict[str, object]:
    profile = load_profile(profile_path)
    receipt, receipt_raw = _read_object(build_receipt_path)
    _validate_build_receipt(receipt, profile)
    if output.exists() or output.is_symlink() or output.suffix.lower() != ".json":
        raise DerivedTrainingImageError("OUTPUT_INVALID")
    authority = _validate_docker_inputs(docker, docker_config)
    tag = str(receipt["tag"])
    inspect = _run_authorized(
        authority,
        runner,
        _inspect_command(
            docker=authority.executable,
            docker_config=authority.config_directory,
            image=tag,
        ),
    )
    image_id, repo_digests, labels = _parse_inspect(inspect.stdout)
    if (
        image_id != receipt["image_config_digest"]
        or labels.get("ai.synapticlabs.derived-training-profile") != profile.canonical_sha256
        or labels.get("ai.synapticlabs.derived-training-base") != profile.base_image
    ):
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID")
    if repo_digests and receipt["final_oci_reference"] not in repo_digests:
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID")
    # The configured tag is mutable. From this point onward use only the exact
    # content-addressed image ID established by the first inspect.
    runtime_result = _run_authorized(
        authority,
        runner,
        _runtime_command(
            docker=authority.executable,
            docker_config=authority.config_directory,
            image=image_id,
            profile=profile,
        )
    )
    runtime = _parse_runtime(runtime_result.stdout)
    if profile.packaged_runtime is not None:
        measured = _run_authorized(authority, runner, _final_runtime_command(
            docker=authority.executable, docker_config=authority.config_directory,
            image=image_id, profile=profile))
        runtime["final_runtime"] = _load_json(measured.stdout)
    final_inspect = _run_authorized(
        authority,
        runner,
        _inspect_command(
            docker=authority.executable,
            docker_config=authority.config_directory,
            image=image_id,
        )
    )
    final_image_id, final_repo_digests, final_labels = _parse_inspect(
        final_inspect.stdout
    )
    if (
        final_image_id != image_id
        or final_labels.get("ai.synapticlabs.derived-training-profile")
        != profile.canonical_sha256
        or final_labels.get("ai.synapticlabs.derived-training-base")
        != profile.base_image
    ):
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID")
    if final_repo_digests and receipt["final_oci_reference"] not in final_repo_digests:
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID")
    candidate = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "CANDIDATE_ONLY",
        "profile_sha256": profile.canonical_sha256,
        "base_image": profile.base_image,
        "final_oci_reference": receipt["final_oci_reference"],
        "final_oci_digest": receipt["final_oci_digest"],
        "image_config_digest": image_id,
        "repo_digests": list(final_repo_digests),
        "build_receipt_sha256": _sha256(receipt_raw),
        "dockerfile_sha256": receipt["dockerfile_sha256"],
        "build_metadata": receipt["build_metadata"],
        "build_metadata_sha256": receipt["build_metadata_sha256"],
        "runtime": runtime,
        "runtime_bytes_sha256": _sha256(_runtime_payload_bytes(runtime)),
    }
    _write_exclusive(output, candidate)
    return candidate


def _validate_build_receipt(value: Mapping[str, object], profile: DerivedImageProfile) -> None:
    _closed(value, {
        "schema_version", "status", "profile_sha256", "base_image",
        "dockerfile_sha256", "tag", "final_oci_reference", "final_oci_digest",
        "image_config_digest", "repo_digests", "build_metadata",
        "build_metadata_sha256",
    })
    if (
        value.get("schema_version") != BUILD_RECEIPT_SCHEMA
        or value.get("status") != "BUILT_NOT_ATTESTED"
        or value.get("profile_sha256") != profile.canonical_sha256
        or value.get("base_image") != profile.base_image
    ):
        raise DerivedTrainingImageError("BUILD_RECEIPT_INVALID")
    tag = value.get("tag")
    final_reference = value.get("final_oci_reference")
    final_digest = value.get("final_oci_digest")
    try:
        repository = _validate_tag(tag)[1] if isinstance(tag, str) else ""
        final_repository, parsed_final_digest = _validate_oci_reference(final_reference)
    except DerivedTrainingImageError as exc:
        raise DerivedTrainingImageError("BUILD_RECEIPT_INVALID") from exc
    repo_digests = value.get("repo_digests")
    build_metadata = value.get("build_metadata")
    _validate_packaged_build_metadata(build_metadata, profile)
    expected_dockerfile_sha256 = _sha256(
        render_dockerfile(profile).encode("utf-8")
    )
    try:
        build_metadata_bytes = _canonical_bytes(build_metadata)
    except DerivedTrainingImageError as exc:
        raise DerivedTrainingImageError("BUILD_RECEIPT_INVALID") from exc
    if (
        not isinstance(final_digest, str) or _DIGEST.fullmatch(final_digest) is None
        or final_repository != repository
        or parsed_final_digest != final_digest
        or not isinstance(value.get("image_config_digest"), str)
        or _DIGEST.fullmatch(str(value["image_config_digest"])) is None
        or value.get("dockerfile_sha256") != expected_dockerfile_sha256
        or not isinstance(build_metadata, dict)
        or len(build_metadata_bytes) > MAX_DOCUMENT_BYTES
        or _sha256(build_metadata_bytes) != value.get("build_metadata_sha256")
        or build_metadata.get("containerimage.digest") != final_digest
        or not _metadata_config_digest_matches(
            build_metadata, value.get("image_config_digest"),
            final_reference, repo_digests, profile,
        )
        or not isinstance(repo_digests, list)
        or any(not isinstance(item, str) for item in repo_digests)
    ):
        raise DerivedTrainingImageError("BUILD_RECEIPT_INVALID")
    try:
        for item in repo_digests:
            _validate_oci_reference(item)
    except DerivedTrainingImageError as exc:
        raise DerivedTrainingImageError("BUILD_RECEIPT_INVALID") from exc
    if repo_digests and final_reference not in repo_digests:
        raise DerivedTrainingImageError("BUILD_RECEIPT_INVALID")


def _validate_packaged_build_metadata(metadata: object, profile: DerivedImageProfile) -> None:
    if profile.packaged_runtime is None: return
    if not isinstance(metadata, dict):
        raise DerivedTrainingImageError("BUILD_METADATA_INVALID")
    # Buildx's local Docker exporter can omit provenance from the metadata file.
    # The exact build inputs remain bound by the rendered Dockerfile's profile
    # label and digest check, then rechecked inside the final runtime image.
    # If Buildx does provide provenance, never accept a contradictory claim.
    if "buildx.build.provenance" not in metadata:
        return
    try:
        digest = hashlib.sha256(_canonical_bytes(profile.packaged_runtime)).hexdigest()
        provenance = metadata["buildx.build.provenance"]
        if "buildDefinition" in provenance and "invocation" in provenance:
            raise ValueError("ambiguous provenance")
        if "buildDefinition" in provenance:
            args = provenance["buildDefinition"]["externalParameters"]["request"]["args"]
        else:
            args = provenance["invocation"]["parameters"]["args"]
        if args["build-arg:SYNAPTIC_RUNTIME_INPUTS_SHA256"] != digest: raise ValueError
    except (KeyError, TypeError, ValueError):
        raise DerivedTrainingImageError("BUILD_METADATA_INVALID:PROVENANCE") from None


def _metadata_config_digest_matches(
    metadata: Mapping[str, object], config_digest: object,
    final_reference: str, repo_digests: object, profile: DerivedImageProfile,
) -> bool:
    if "containerimage.config.digest" in metadata:
        return metadata["containerimage.config.digest"] == config_digest
    return (
        profile.packaged_runtime is not None
        and isinstance(repo_digests, list)
        and final_reference in repo_digests
    )


def _validate_runtime_against_profile(
    runtime: Mapping[str, object], profile: DerivedImageProfile
) -> None:
    if (
        runtime.get("schema_version") != "synaptic-derived-training-image-runtime/v1"
        or not isinstance(runtime.get("packages"), dict)
        or set(runtime["packages"]) != {item.distribution for item in profile.packages}
    ):
        raise DerivedTrainingImageError("RUNTIME_EVIDENCE_INVALID")
    packages = runtime["packages"]
    assert isinstance(packages, dict)
    for expected in profile.packages:
        actual = packages.get(expected.distribution)
        if not isinstance(actual, dict) or set(actual) != {"version", "direct_url"}:
            raise DerivedTrainingImageError("RUNTIME_EVIDENCE_INVALID")
        version = actual.get("version")
        direct_url = actual.get("direct_url")
        if not isinstance(version, str) or not version or len(version) > 256:
            raise DerivedTrainingImageError("RUNTIME_EVIDENCE_INVALID")
        if expected.source == "index":
            if version != expected.version or direct_url is not None:
                raise DerivedTrainingImageError("PACKAGE_MISMATCH")
            continue
        if not isinstance(direct_url, dict) or set(direct_url) != {"url", "vcs_info"}:
            raise DerivedTrainingImageError("PACKAGE_MISMATCH")

        vcs = direct_url.get("vcs_info")
        if (
            direct_url.get("url") != expected.url
            or not isinstance(vcs, dict)
            or set(vcs) - {"vcs", "commit_id", "requested_revision"}
            or vcs.get("vcs") != "git"
            or vcs.get("commit_id") != expected.commit
            or vcs.get("requested_revision") != expected.commit
        ):
            raise DerivedTrainingImageError("PACKAGE_MISMATCH")

    if profile.packaged_runtime is not None:
        measured = runtime.get("final_runtime")
        _parse_final_runtime(measured, image_verification={
            "packaged_runtime": profile.packaged_runtime,
            "packages": packages, "final_runtime": measured,
        })
    elif "final_runtime" in runtime:
        raise DerivedTrainingImageError("RUNTIME_EVIDENCE_INVALID")


def verify_candidate(
    *, profile_path: Path, candidate_path: Path, output: Path | None = None
) -> dict[str, object]:
    profile = load_profile(profile_path)
    candidate, candidate_raw = _read_object(candidate_path)
    _closed(candidate, {
        "schema_version", "status", "profile_sha256", "base_image",
        "final_oci_reference", "final_oci_digest", "image_config_digest",
        "repo_digests", "build_receipt_sha256", "dockerfile_sha256",
        "build_metadata", "build_metadata_sha256", "runtime",
        "runtime_bytes_sha256",
    })
    final_digest = candidate.get("final_oci_digest")
    final_reference = candidate.get("final_oci_reference")
    try:
        _repository, parsed_final_digest = _validate_oci_reference(final_reference)
    except DerivedTrainingImageError as exc:
        raise DerivedTrainingImageError("CANDIDATE_INVALID") from exc
    repo_digests = candidate.get("repo_digests")
    build_metadata = candidate.get("build_metadata")
    _validate_packaged_build_metadata(build_metadata, profile)
    expected_dockerfile_sha256 = _sha256(
        render_dockerfile(profile).encode("utf-8")
    )
    try:
        build_metadata_bytes = _canonical_bytes(build_metadata)
    except DerivedTrainingImageError as exc:
        raise DerivedTrainingImageError("CANDIDATE_INVALID") from exc
    if (
        candidate.get("schema_version") != CANDIDATE_SCHEMA
        or candidate.get("status") != "CANDIDATE_ONLY"
        or candidate.get("profile_sha256") != profile.canonical_sha256
        or candidate.get("base_image") != profile.base_image
        or not isinstance(final_digest, str)
        or _DIGEST.fullmatch(final_digest) is None
        or parsed_final_digest != final_digest
        or not isinstance(candidate.get("image_config_digest"), str)
        or _DIGEST.fullmatch(str(candidate["image_config_digest"])) is None
        or not isinstance(candidate.get("runtime"), dict)
        or candidate.get("dockerfile_sha256") != expected_dockerfile_sha256
        or not isinstance(build_metadata, dict)
        or len(build_metadata_bytes) > MAX_DOCUMENT_BYTES
        or _sha256(build_metadata_bytes) != candidate.get("build_metadata_sha256")
        or build_metadata.get("containerimage.digest") != final_digest
        or not _metadata_config_digest_matches(
            build_metadata, candidate.get("image_config_digest"),
            final_reference, repo_digests, profile,
        )
        or not isinstance(repo_digests, list)
        or any(not isinstance(item, str) for item in repo_digests)
        or any(
            not isinstance(candidate.get(key), str) or _DIGEST.fullmatch(str(candidate[key])) is None
            for key in (
                "build_receipt_sha256", "runtime_bytes_sha256",
            )
        )
    ):
        raise DerivedTrainingImageError("CANDIDATE_INVALID")
    try:
        for item in repo_digests:
            _validate_oci_reference(item)
    except DerivedTrainingImageError as exc:
        raise DerivedTrainingImageError("CANDIDATE_INVALID") from exc
    if repo_digests and final_reference not in repo_digests:
        raise DerivedTrainingImageError("CANDIDATE_INVALID")
    runtime = candidate["runtime"]
    assert isinstance(runtime, dict)
    if _sha256(_runtime_payload_bytes(runtime)) != candidate["runtime_bytes_sha256"]:
        raise DerivedTrainingImageError("CANDIDATE_INVALID")
    _validate_runtime_against_profile(runtime, profile)
    verification = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "DIAGNOSTIC_PASS",
        "profile_sha256": profile.canonical_sha256,
        "candidate_sha256": _sha256(candidate_raw),
        "build_receipt_sha256": candidate["build_receipt_sha256"],
        "dockerfile_sha256": candidate["dockerfile_sha256"],
        "build_metadata": candidate["build_metadata"],
        "build_metadata_sha256": candidate["build_metadata_sha256"],
        "base_image": profile.base_image,
        "final_oci_reference": final_reference,
        "final_oci_digest": final_digest,
        "image_config_digest": candidate["image_config_digest"],
        "packages": runtime["packages"],
    }
    if profile.packaged_runtime is not None:
        verification["final_runtime"] = runtime["final_runtime"]
        verification["repo_digests"] = candidate["repo_digests"]
    if output is not None:
        if output.exists() or output.is_symlink() or output.suffix.lower() != ".json":
            raise DerivedTrainingImageError("OUTPUT_INVALID")
        _write_exclusive(output, verification)
    return verification


def validate_verification_report(
    *, profile_path: Path, verification_report_path: Path, image: str
) -> dict[str, object]:
    profile = load_profile(profile_path)
    verification, _raw = _read_object(verification_report_path)
    _closed(verification, {
        "schema_version", "status", "profile_sha256", "candidate_sha256",
        "build_receipt_sha256", "dockerfile_sha256", "build_metadata",
        "build_metadata_sha256",
        "base_image", "final_oci_reference", "final_oci_digest",
        "image_config_digest", "packages",
    }, {"final_runtime", "repo_digests"} if profile.packaged_runtime is not None else set())
    final_reference = verification.get("final_oci_reference")
    final_digest = verification.get("final_oci_digest")
    image_config_digest = verification.get("image_config_digest")
    build_metadata = verification.get("build_metadata")
    expected_dockerfile_sha256 = _sha256(
        render_dockerfile(profile).encode("utf-8")
    )
    try:
        build_metadata_bytes = _canonical_bytes(build_metadata)
    except DerivedTrainingImageError as exc:
        raise DerivedTrainingImageError("IMAGE_QUALIFICATION_INVALID") from exc
    try:
        _repository, parsed_final_digest = _validate_oci_reference(final_reference)
    except DerivedTrainingImageError as exc:
        raise DerivedTrainingImageError("IMAGE_QUALIFICATION_INVALID") from exc
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA
        or verification.get("status") != "DIAGNOSTIC_PASS"
        or verification.get("profile_sha256") != profile.canonical_sha256
        or verification.get("base_image") != profile.base_image
        or not isinstance(final_digest, str)
        or _DIGEST.fullmatch(final_digest) is None
        or parsed_final_digest != final_digest
        or not isinstance(image_config_digest, str)
        or _DIGEST.fullmatch(image_config_digest) is None
        or verification.get("dockerfile_sha256") != expected_dockerfile_sha256
        or not isinstance(build_metadata, dict)
        or len(build_metadata_bytes) > MAX_DOCUMENT_BYTES
        or _sha256(build_metadata_bytes)
        != verification.get("build_metadata_sha256")
        or build_metadata.get("containerimage.digest") != final_digest
        or not _metadata_config_digest_matches(
            build_metadata, image_config_digest,
            final_reference, verification.get("repo_digests"), profile,
        )
        or image not in {final_reference, image_config_digest}
        or not isinstance(verification.get("candidate_sha256"), str)
        or _DIGEST.fullmatch(str(verification["candidate_sha256"])) is None
        or any(
            not isinstance(verification.get(key), str)
            or _DIGEST.fullmatch(str(verification[key])) is None
            for key in (
                "build_receipt_sha256",
            )
        )
        or not isinstance(verification.get("packages"), dict)
    ):
        raise DerivedTrainingImageError("IMAGE_QUALIFICATION_INVALID")
    runtime = {
        "schema_version": "synaptic-derived-training-image-runtime/v1",
        "packages": verification["packages"],
    }
    if profile.packaged_runtime is not None:
        _validate_packaged_build_metadata(build_metadata, profile)
        runtime["final_runtime"] = verification.get("final_runtime")
    _validate_runtime_against_profile(runtime, profile)
    return verification


def verify_effectful_launch(
    *,
    profile_path: Path,
    verification_report_path: Path,
    image: str,
    docker: Path,
    docker_config: Path,
    runner: Runner = subprocess_runner,
) -> DockerLaunchAuthority:
    """Freshly bind an immutable launch image to live Docker evidence.

    The checked-in/offline report is intentionally only diagnostic.  This
    function is the effectful local consistency boundary: it inspects the configured
    immutable reference, probes the exact returned image-config digest, and
    re-inspects that digest before returning.  Nothing is written or promoted.
    """

    profile = load_profile(profile_path)
    verification = validate_verification_report(
        profile_path=profile_path,
        verification_report_path=verification_report_path,
        image=image,
    )
    authority = _validate_docker_inputs(docker, docker_config)
    expected_config = str(verification["image_config_digest"])
    expected_final = str(verification["final_oci_reference"])

    inspected = _run_authorized(
        authority,
        runner,
        _inspect_command(
            docker=authority.executable,
            docker_config=authority.config_directory,
            image=image,
        ),
    )
    image_id, repo_digests, labels = _parse_inspect(inspected.stdout)
    if (
        image_id != expected_config
        or labels.get("ai.synapticlabs.derived-training-profile")
        != profile.canonical_sha256
        or labels.get("ai.synapticlabs.derived-training-base")
        != profile.base_image
        or (image == expected_final and expected_final not in repo_digests)
    ):
        raise DerivedTrainingImageError("LIVE_IMAGE_MISMATCH")

    runtime_result = _run_authorized(
        authority,
        runner,
        _runtime_command(
            docker=authority.executable,
            docker_config=authority.config_directory,
            image=image_id,
            profile=profile,
        ),
    )
    runtime = _parse_runtime(runtime_result.stdout)
    if profile.packaged_runtime is not None:
        measured = _run_authorized(authority, runner, _final_runtime_command(
            docker=authority.executable, docker_config=authority.config_directory,
            image=image_id, profile=profile))
        runtime["final_runtime"] = _load_json(measured.stdout)
        if runtime["final_runtime"] != verification.get("final_runtime"):
            raise DerivedTrainingImageError("LIVE_IMAGE_MISMATCH")
    _validate_runtime_against_profile(runtime, profile)
    if runtime.get("packages") != verification.get("packages"):
        raise DerivedTrainingImageError("LIVE_IMAGE_MISMATCH")

    final_inspect = _run_authorized(
        authority,
        runner,
        _inspect_command(
            docker=authority.executable,
            docker_config=authority.config_directory,
            image=image_id,
        ),
    )
    final_id, final_repo_digests, final_labels = _parse_inspect(final_inspect.stdout)
    if (
        final_id != expected_config
        or final_labels.get("ai.synapticlabs.derived-training-profile")
        != profile.canonical_sha256
        or final_labels.get("ai.synapticlabs.derived-training-base")
        != profile.base_image
        or (image == expected_final and expected_final not in final_repo_digests)
    ):
        raise DerivedTrainingImageError("LIVE_IMAGE_MISMATCH")
    evidence = {
        "schema_version": "synaptic-derived-training-image-live-verification/v1",
        "status": "LIVE_DOCKER_VERIFIED",
        "profile_sha256": profile.canonical_sha256,
        "image": image,
        "image_config_digest": image_id,
        "runtime_bytes_sha256": _sha256(runtime_result.stdout),
    }
    _assert_docker_authority(authority)
    return DockerLaunchAuthority(evidence=evidence, _authority=authority)


def _final_runtime_inspector(profile: DerivedImageProfile) -> str:
    """Fixed stdlib bootstrap; no site hooks, ambient interpreter or search path."""
    if profile.packaged_runtime is None:
        raise DerivedTrainingImageError("PACKAGED_PROFILE_REQUIRED")
    expected = profile.packaged_runtime
    return f"""import hashlib,json,os,platform,sys,sysconfig
expected = {expected!r}
python = expected['python']
if sys.executable != python['executable'] or platform.python_implementation().lower() != python['implementation'] or platform.python_version() != python['version']:
    raise RuntimeError('INTERPRETER_INVALID')
with open(sys.executable, 'rb') as source:
    raw = source.read(67108865)
if len(raw) > 67108864 or hashlib.sha256(raw).hexdigest() != python['executable_digest']:
    raise RuntimeError('INTERPRETER_INVALID')
venv_root = os.path.dirname(os.path.dirname(sys.executable))
if os.path.isfile(os.path.join(venv_root, 'pyvenv.cfg')):
    paths = sysconfig.get_paths(scheme='venv', vars={{'base': venv_root, 'platbase': venv_root}})
else:
    paths = sysconfig.get_paths()
if any(paths[key] != python[key] for key in ('purelib', 'platlib')):
    raise RuntimeError('PACKAGE_PATH_INVALID')
sys.path.extend(dict.fromkeys(python[key] for key in ('purelib', 'platlib')))
from tuner.runtime.packaged_training_worker import inspect_installed_runtime
print(json.dumps(inspect_installed_runtime(expected),sort_keys=True,separators=(',',':')))
"""


def _final_runtime_command(*, docker: Path, docker_config: Path, image: str, profile: DerivedImageProfile) -> CommandSpec:
    return CommandSpec(
        argv=_docker_argv(docker, docker_config) + (
            "run", "--rm", "--network", "none", "--entrypoint", profile.python_executable,
            image, "-I", "-S", "-c", _final_runtime_inspector(profile),
        ), env=_docker_env(docker), timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
        maximum_output_bytes=MAX_COMMAND_OUTPUT_BYTES,
    )


def _load_image_evidence(*, profile_path: Path, build_receipt_path: Path, candidate_path: Path, image_verification_path: Path) -> tuple[dict[str, object], bytes]:
    """Re-open every immutable image artifact; never trust a detached summary."""
    profile = load_profile(profile_path)
    build, build_raw = _read_object(build_receipt_path)
    _validate_build_receipt(build, profile)
    candidate, candidate_raw = _read_object(candidate_path)
    expected = verify_candidate(profile_path=profile_path, candidate_path=candidate_path)
    if candidate.get("build_receipt_sha256") != _sha256(build_raw):
        raise DerivedTrainingImageError("IMAGE_EVIDENCE_INVALID")
    if (expected["build_receipt_sha256"] != _sha256(build_raw)
            or expected["candidate_sha256"] != _sha256(candidate_raw)
            or expected["profile_sha256"] != profile.canonical_sha256
            or any(expected[key] != build[key] for key in ("final_oci_reference", "final_oci_digest", "image_config_digest", "build_metadata_sha256", "dockerfile_sha256"))):
        raise DerivedTrainingImageError("IMAGE_EVIDENCE_INVALID")
    verification, verification_raw = _read_object(image_verification_path)
    if verification_raw != _canonical_bytes(expected):
        raise DerivedTrainingImageError("IMAGE_EVIDENCE_INVALID")
    if profile.packaged_runtime is None:
        raise DerivedTrainingImageError("PACKAGED_PROFILE_REQUIRED")
    return {**verification, "packaged_runtime": profile.packaged_runtime}, verification_raw


def _parse_final_runtime(value: object, *, image_verification: Mapping[str, object]) -> dict[str, object]:
    """Use only facts cross-bound to the revalidated image candidate and profile."""
    try:
        if not isinstance(value, dict): raise ValueError
        _closed(value, {"schema_version", "package", "python", "installed_distributions", "platform", "worker", "closure", "contracts", "capabilities", "build_inputs_digest", "provenance"})
        expected = image_verification["packaged_runtime"]
        if value != image_verification["final_runtime"]: raise ValueError
        if value["schema_version"] != "synaptic-packaged-runtime-inspector/v1": raise ValueError
        if value["build_inputs_digest"] != hashlib.sha256(_canonical_bytes(expected)).hexdigest(): raise ValueError
        if value["capabilities"] != expected["capabilities"] or value["contracts"] != expected["capabilities"]["contracts"]: raise ValueError
        for key in ("package", "python", "installed_distributions", "platform", "worker", "closure", "provenance"):
            if not isinstance(value[key], dict): raise ValueError
        _closed(value["package"], {"name", "version", "digest", "source_provenance_digest"})
        _closed(value["python"], {"implementation", "version", "executable", "executable_digest"})
        _closed(value["installed_distributions"], {"inventory", "digest", "count"})
        _closed(value["platform"], {"system", "machine", "cuda_version", "runtime_facts"})
        _closed(value["worker"], {"entrypoint", "closure_digest"})
        _closed(value["closure"], {"verified_digest"})
        if value["python"] != {key: expected["python"][key] for key in value["python"]}: raise ValueError
        wheel = expected["wheel"]
        if value["package"] != {"name": "synaptic-tuner", "version": wheel["version"], "digest": wheel["sha256"], "source_provenance_digest": value["provenance"].get("synaptic-tuner")}: raise ValueError
        wheels = [wheel, *expected["bootstrap"]]
        if set(value["provenance"]) != {item["distribution"] for item in wheels}: raise ValueError
        if any(not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None or item == hashlib.sha256(b"").hexdigest() for item in value["provenance"].values()): raise ValueError
        inventory = value["installed_distributions"]["inventory"]
        if not isinstance(inventory, list) or not 1 <= len(inventory) <= 4096 or type(value["installed_distributions"]["count"]) is not int or value["installed_distributions"]["count"] != len(inventory): raise ValueError
        installed = {}
        for item in inventory:
            if not isinstance(item, dict) or set(item) != {"name", "version"}: raise ValueError
            name, version = item["name"], item["version"]
            if not isinstance(name, str) or _NAME.fullmatch(name) is None or re.sub(r"[-_.]+", "-", name) != name or name in installed or not isinstance(version, str) or _VERSION.fullmatch(version) is None: raise ValueError
            installed[name] = version
        if list(installed) != sorted(installed): raise ValueError
        if hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest() != value["installed_distributions"]["digest"]: raise ValueError
        for item in wheels:
            if installed.get(item["distribution"]) != item["version"]: raise ValueError
        for name, item in image_verification["packages"].items():
            if installed.get(name) != item["version"]: raise ValueError
        if value["platform"]["system"] != "linux" or value["platform"]["machine"] != "x86_64": raise ValueError
        closure = load_packaged_worker_closure()
        if value["worker"] != {"entrypoint": PACKAGED_TRAINING_WORKER_ENTRYPOINT, "closure_digest": closure.digest} or value["closure"] != {"verified_digest": closure.digest}: raise ValueError
        return value
    except (KeyError, TypeError, ValueError):
        raise DerivedTrainingImageError("FINAL_RUNTIME_INVALID") from None


def _bound_capture(path: Path, *, image: dict, image_raw: bytes) -> tuple[dict, bytes]:
    capture, raw = _load_final_capture(path)
    if (capture["image_verification_sha256"] != _sha256(image_raw)
            or capture["image"] != {"reference": image["final_oci_reference"], "digest": image["final_oci_digest"][7:]}):
        raise DerivedTrainingImageError("FINAL_RUNTIME_CAPTURE_INVALID")
    _parse_final_runtime(capture["measured"], image_verification=image)
    return capture, raw


def _release_facts(capture: dict, compatibility: dict, release_ref: str) -> dict:
    measured = capture["measured"]
    # Exact capability derivation is deliberately stricter than a subset.
    if compatibility != measured["capabilities"]["compatibility"]:
        raise DerivedTrainingImageError("CAPABILITY_MISMATCH")
    return {"release_ref": release_ref, "image": capture["image"], "compatibility": compatibility,
            **{key: measured[key] for key in ("package", "worker", "python", "installed_distributions", "platform", "contracts") }}


def capture_final_runtime(*, profile_path: Path, build_receipt_path: Path, candidate_path: Path, image_verification_path: Path, docker: Path, docker_config: Path, output: Path, runner: Runner = subprocess_runner) -> dict[str, object]:
    image, image_raw = _load_image_evidence(profile_path=profile_path, build_receipt_path=build_receipt_path, candidate_path=candidate_path, image_verification_path=image_verification_path)
    if output.exists() or output.is_symlink() or output.suffix.lower() != ".json": raise DerivedTrainingImageError("OUTPUT_INVALID")
    authority = _validate_docker_inputs(docker, docker_config)
    profile = load_profile(profile_path)
    inspected = _run_authorized(authority, runner, _inspect_command(docker=authority.executable, docker_config=authority.config_directory, image=str(image["final_oci_reference"])))
    identity, references, labels = _parse_inspect(inspected.stdout)
    if identity != image["image_config_digest"] or labels.get("ai.synapticlabs.derived-training-profile") != profile.canonical_sha256 or labels.get("ai.synapticlabs.derived-training-base") != profile.base_image or (references and image["final_oci_reference"] not in references):
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID")
    result = _run_authorized(authority, runner, _final_runtime_command(docker=authority.executable, docker_config=authority.config_directory, image=identity, profile=profile))
    after = _run_authorized(authority, runner, _inspect_command(docker=authority.executable, docker_config=authority.config_directory, image=identity))
    if _parse_inspect(after.stdout) != (identity, references, labels):
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID")
    measured = _parse_final_runtime(_load_json(result.stdout), image_verification=image)
    capture = {"schema_version": FINAL_RUNTIME_CAPTURE_SCHEMA, "status": "FINAL_RUNTIME_CAPTURED", "image_verification_sha256": _sha256(image_raw), "image": {"reference": image["final_oci_reference"], "digest": str(image["final_oci_digest"])[7:]}, "measured": measured}
    _write_exclusive(output, capture)
    return capture


def _load_final_capture(path: Path) -> tuple[dict[str, object], bytes]:
    value, raw = _read_object(path)
    try:
        _closed(value, {"schema_version", "status", "image_verification_sha256", "image", "measured"})
        if value["schema_version"] != FINAL_RUNTIME_CAPTURE_SCHEMA or value["status"] != "FINAL_RUNTIME_CAPTURED" or not isinstance(value["image"], dict) or set(value["image"]) != {"reference", "digest"} or not isinstance(value["measured"], dict): raise ValueError
        if not isinstance(value["image_verification_sha256"], str) or _DIGEST.fullmatch(value["image_verification_sha256"]) is None: raise ValueError
        return value, raw
    except (KeyError, TypeError, ValueError): raise DerivedTrainingImageError("FINAL_RUNTIME_CAPTURE_INVALID") from None


def _release_from_input(value: object) -> PackagedTrainingRuntimeReleaseV1:
    """Build the audited release contract from an unsigned manifest-shaped input.

    This deliberately delegates all semantic validation and digest construction
    to ``PackagedTrainingRuntimeReleaseV1``.  The input is only a convenient
    transport for the contract's public fields; it is not another release
    schema.
    """

    if not isinstance(value, dict):
        raise DerivedTrainingImageError("RELEASE_INPUT_INVALID")
    required = {
        "release_ref", "package", "worker", "image", "python",
        "installed_distributions", "platform", "compatibility", "contracts",
    }
    try:
        _closed(value, required)
        package = value["package"]
        worker = value["worker"]
        image = value["image"]
        python = value["python"]
        installed = value["installed_distributions"]
        platform = value["platform"]
        compatibility = value["compatibility"]
        contracts = value["contracts"]
        if not all(isinstance(item, dict) for item in (
            package, worker, image, python, installed, platform, compatibility, contracts,
        )):
            raise TypeError
        _closed(package, {"name", "version", "digest", "source_provenance_digest"})
        _closed(worker, {"entrypoint", "closure_digest"})
        _closed(image, {"reference", "digest"})
        _closed(python, {"implementation", "version", "executable", "executable_digest"})
        _closed(installed, {"digest", "count"}, {"inventory"})
        _closed(platform, {"system", "machine", "cuda_version", "runtime_facts"})
        _closed(compatibility, {"methods", "models", "dataset_formats"})
        _closed(contracts, {"workload_schema", "prepared_input_schema", "artifact_contract_schema"})
        release = PackagedTrainingRuntimeReleaseV1.build(
            release_ref=value["release_ref"],
            package_name=package["name"], package_version=package["version"],
            package_digest=package["digest"], source_provenance_digest=package["source_provenance_digest"],
            worker_entrypoint=worker["entrypoint"], worker_closure_digest=worker["closure_digest"],
            image_ref=image["reference"], image_digest=image["digest"],
            python_implementation=python["implementation"], python_version=python["version"],
            python_executable=python["executable"], python_executable_digest=python["executable_digest"],
            installed_distributions_digest=installed["digest"], installed_distribution_count=installed["count"],
            platform_system=platform["system"], platform_machine=platform["machine"],
            cuda_version=platform["cuda_version"], runtime_facts=platform["runtime_facts"],
            compatible_methods=compatibility["methods"], compatible_models=compatibility["models"],
            compatible_dataset_formats=compatibility["dataset_formats"],
            workload_schema=contracts["workload_schema"], prepared_input_schema=contracts["prepared_input_schema"],
            artifact_contract_schema=contracts["artifact_contract_schema"],
        )
        if release.worker_entrypoint != PACKAGED_TRAINING_WORKER_ENTRYPOINT:
            raise ValueError("worker entrypoint is unsupported")
        return release
    except (KeyError, TypeError, ValueError) as exc:
        raise DerivedTrainingImageError("RELEASE_INPUT_INVALID") from exc


def build_runtime_release(*, capture_path: Path, compatibility_path: Path, output: Path, profile_path: Path, build_receipt_path: Path, candidate_path: Path, image_verification_path: Path) -> PackagedTrainingRuntimeReleaseV1:
    image, image_raw = _load_image_evidence(profile_path=profile_path, build_receipt_path=build_receipt_path, candidate_path=candidate_path, image_verification_path=image_verification_path)
    capture, _raw = _bound_capture(capture_path, image=image, image_raw=image_raw)
    compatibility, _ = _read_object(compatibility_path)
    measured = capture["measured"]
    assert isinstance(measured, dict)
    try:
        _closed(compatibility, {"release_ref", "compatibility"})
        facts = _release_facts(capture, compatibility["compatibility"], compatibility["release_ref"])
    except (KeyError, TypeError, ValueError): raise DerivedTrainingImageError("RELEASE_INPUT_INVALID") from None
    release = _release_from_input(facts)
    if output.suffix.lower() != ".json":
        raise DerivedTrainingImageError("OUTPUT_INVALID")
    _write_exclusive_bytes(output, release.canonical_bytes())
    return release


def _load_runtime_release(path: Path) -> tuple[PackagedTrainingRuntimeReleaseV1, bytes]:
    try:
        raw = _read_regular_release(path)
        release = PackagedTrainingRuntimeReleaseV1.from_json(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise DerivedTrainingImageError("RELEASE_INVALID") from exc
    if release.canonical_bytes() != raw:
        raise DerivedTrainingImageError("RELEASE_INVALID")
    return release, raw


def _read_regular_release(path: Path) -> bytes:
    raw = stable_read(path, MAX_DOCUMENT_BYTES)
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        raise OSError("release exceeds byte limit")
    return raw


LOCAL_QUALIFICATION_CONFIG_SCHEMA = "synaptic-packaged-local-qualification-config/v1"
LOCAL_QUALIFICATION_EVIDENCE_SCHEMA = "synaptic-packaged-local-qualification-evidence/v1"


def _local_qualification_inputs(*, qualification_config_path, release_path, capture_path,
        profile_path, build_receipt_path, candidate_path, image_verification_path):
    from tuner.runtime.packaged_training_worker import LOCAL_CPU_PROTOCOL, local_cpu_result
    from zipfile import ZipFile
    from io import BytesIO
    config, config_raw = _read_object(qualification_config_path)
    _closed(config, {"schema_version", "protocol", "runtime_release_digest"})
    release, release_raw = _load_runtime_release(release_path)
    if config != {"schema_version": LOCAL_QUALIFICATION_CONFIG_SCHEMA, "protocol": LOCAL_CPU_PROTOCOL,
                  "runtime_release_digest": release.manifest_digest}:
        raise DerivedTrainingImageError("LOCAL_QUALIFICATION_CONFIG_INVALID")
    image, image_raw = _load_image_evidence(profile_path=profile_path, build_receipt_path=build_receipt_path,
        candidate_path=candidate_path, image_verification_path=image_verification_path)
    capture, capture_raw = _bound_capture(capture_path, image=image, image_raw=image_raw)
    if release.to_dict() != _release_from_input(_release_facts(capture, release.to_dict()["compatibility"], release.release_ref)).to_dict():
        raise DerivedTrainingImageError("RELEASE_CAPTURE_MISMATCH")
    profile = load_profile(profile_path)
    wheel = stable_read(profile.artifact_root / profile.packaged_runtime["wheel"]["filename"], 256 * 1024 * 1024)
    if hashlib.sha256(wheel).hexdigest() != release.package_digest:
        raise DerivedTrainingImageError("WHEEL_ARTIFACT_INVALID")
    with ZipFile(BytesIO(wheel)) as archive:
        members = archive.infolist()
        target = [item for item in members if item.filename == "Trainers/sft/train_sft.py"]
        if len(target) != 1 or target[0].file_size > 4 * 1024 * 1024: raise DerivedTrainingImageError("WHEEL_ARTIFACT_INVALID")
        trainer_digest = hashlib.sha256(archive.read(target[0])).hexdigest()
    expected = {"schema_version": LOCAL_QUALIFICATION_EVIDENCE_SCHEMA, "status": "LOCAL_CPU_DIAGNOSTIC_PASS",
        "scope": "installed_child_cpu_only", "qualification_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "runtime_release_sha256": hashlib.sha256(release_raw).hexdigest(), "runtime_release_digest": release.manifest_digest,
        "worker_closure_digest": release.worker_closure_digest, "package_digest": release.package_digest,
        "python_executable_digest": release.python_executable_digest,
        "installed_distributions_digest": release.installed_distributions_digest,
        "final_runtime_capture_sha256": hashlib.sha256(capture_raw).hexdigest(), "image_verification_sha256": hashlib.sha256(image_raw).hexdigest(),
        "profile_sha256": profile.canonical_sha256.removeprefix("sha256:"), "image_config_digest": image["image_config_digest"],
        "image_reference": image["final_oci_reference"], "child": local_cpu_result(release, trainer_digest),
        "training_executed": False, "gpu_qualified": False, "provider_qualified": False}
    return profile, release, image, expected


def local_qualification_command(*, docker, docker_config, profile, release, image):
    # Fixed harness only. No caller executable, code, environment, or mount selection.
    bootstrap = ("import os,sys;os.environ.clear();"
        f"sys.path.extend({list(dict.fromkeys(profile.packaged_runtime['python'][key] for key in ('purelib', 'platlib')))!r});"
        "from tuner.runtime.packaged_training_worker import local_cpu_main;"
        f"raise SystemExit(local_cpu_main({release.to_dict()!r}))")
    return CommandSpec(argv=_docker_argv(docker, docker_config) + (
        "run", "--rm", "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
        "--runtime=runc", "--env=NVIDIA_VISIBLE_DEVICES=void", "--env=CUDA_VISIBLE_DEVICES=",
        "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=512m", "--cpus=1",
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777", "--workdir=/tmp",
        "--entrypoint", release.python_executable, image, "-I", "-S", "-c", bootstrap),
        env=_docker_env(docker), timeout_seconds=120, maximum_output_bytes=16384)


def qualify_local_runtime(*, docker: Path, docker_config: Path, output: Path, execute=False,
                          runner: Runner = subprocess_runner, **inputs):
    profile, release, image, expected = _local_qualification_inputs(**inputs)
    if output.exists() or output.is_symlink() or output.suffix.lower() != ".json":
        raise DerivedTrainingImageError("OUTPUT_INVALID")
    command = local_qualification_command(docker=docker, docker_config=docker_config, profile=profile,
                                         release=release, image=image["image_config_digest"])
    if not execute:
        return {"schema_version": LOCAL_QUALIFICATION_EVIDENCE_SCHEMA, "status": "PLAN_ONLY",
                "scope": "installed_child_cpu_only", "argv": list(command.argv),
                "training_executed": False, "gpu_qualified": False, "provider_qualified": False}
    authority = _validate_docker_inputs(docker, docker_config)
    before = _parse_inspect(_run_authorized(authority, runner, _inspect_command(
        docker=authority.executable, docker_config=authority.config_directory, image=image["image_config_digest"])).stdout)
    identity, references, labels = before
    if (identity != image["image_config_digest"] or labels.get("ai.synapticlabs.derived-training-profile") != profile.canonical_sha256
            or labels.get("ai.synapticlabs.derived-training-base") != profile.base_image
            or (references and image["final_oci_reference"] not in references)):
        raise DerivedTrainingImageError("DOCKER_EVIDENCE_INVALID")
    result = _run_authorized(authority, runner, command)
    after = _parse_inspect(_run_authorized(authority, runner, _inspect_command(
        docker=authority.executable, docker_config=authority.config_directory, image=identity)).stdout)
    if before != after or result.stderr or result.stdout != _canonical_bytes(expected["child"]):
        raise DerivedTrainingImageError("LOCAL_QUALIFICATION_REJECTED")
    _write_exclusive(output, expected)
    return expected


def verify_local_runtime(*, evidence_path: Path, output: Path, **inputs):
    _, _, _, expected = _local_qualification_inputs(**inputs)
    evidence, raw = _read_object(evidence_path)
    if evidence != expected or raw != _canonical_bytes(expected):
        raise DerivedTrainingImageError("LOCAL_QUALIFICATION_EVIDENCE_INVALID")
    report = {"schema_version": LOCAL_QUALIFICATION_EVIDENCE_SCHEMA, "status": "LOCAL_CPU_DIAGNOSTIC_VERIFIED",
              "evidence_sha256": hashlib.sha256(raw).hexdigest(), "runtime_release_digest": expected["runtime_release_digest"],
              "training_executed": False, "gpu_qualified": False, "provider_qualified": False}
    if output.suffix.lower() != ".json": raise DerivedTrainingImageError("OUTPUT_INVALID")
    _write_exclusive(output, report)
    return report


def verify_runtime_release(*, release_path: Path, capture_path: Path, profile_path: Path, build_receipt_path: Path, candidate_path: Path, image_verification_path: Path, output: Path) -> dict[str, object]:
    """Verify an already-built release without Docker, registry, or provider I/O."""

    release, raw = _load_runtime_release(release_path)
    image, image_raw = _load_image_evidence(profile_path=profile_path, build_receipt_path=build_receipt_path, candidate_path=candidate_path, image_verification_path=image_verification_path)
    capture, capture_raw = _bound_capture(capture_path, image=image, image_raw=image_raw)
    measured = capture["measured"]
    assert isinstance(measured, dict)
    facts = _release_facts(capture, release.to_dict()["compatibility"], release.release_ref)
    if release.to_dict() != _release_from_input(facts).to_dict(): raise DerivedTrainingImageError("RELEASE_CAPTURE_MISMATCH")
    closure = load_packaged_worker_closure()
    if (
        release.worker_entrypoint != PACKAGED_TRAINING_WORKER_ENTRYPOINT
        or release.worker_closure_digest != closure.digest
    ):
        raise DerivedTrainingImageError("WORKER_CLOSURE_MISMATCH")
    report = {
        "schema_version": RELEASE_VERIFICATION_SCHEMA,
        "status": "RELEASE_VERIFIED",
        "runtime_release_digest": release.manifest_digest,
        "runtime_release_sha256": hashlib.sha256(raw).hexdigest(),
        "worker_closure_digest": closure.digest,
        "final_runtime_capture_sha256": _sha256(capture_raw),
        "image_verification_sha256": _sha256(image_raw),
    }
    if output.suffix.lower() != ".json":
        raise DerivedTrainingImageError("OUTPUT_INVALID")
    _write_exclusive(output, report)
    return report


def promote_runtime_release(
    *, release_path: Path, verification_path: Path, final_runtime_capture_path: Path,
    profile_path: Path, build_receipt_path: Path, candidate_path: Path,
    image_verification_path: Path, output: Path,
) -> dict[str, object]:
    """Record a local, immutable promotion after offline release verification.

    Promotion never contacts a registry or provider and intentionally retains no
    provider-native facts.  It is an exclusive local evidence transition, not
    image publication or deployment authority.
    """

    release, release_raw = _load_runtime_release(release_path)
    image, image_raw = _load_image_evidence(profile_path=profile_path, build_receipt_path=build_receipt_path, candidate_path=candidate_path, image_verification_path=image_verification_path)
    capture, capture_raw = _bound_capture(final_runtime_capture_path, image=image, image_raw=image_raw)
    measured = capture["measured"]
    assert isinstance(measured, dict)
    facts = _release_facts(capture, release.to_dict()["compatibility"], release.release_ref)
    if release.to_dict() != _release_from_input(facts).to_dict():
        raise DerivedTrainingImageError("RELEASE_CAPTURE_MISMATCH")
    verification, verification_raw = _read_object(verification_path)
    try:
        _closed(verification, {
            "schema_version", "status", "runtime_release_digest",
            "runtime_release_sha256", "worker_closure_digest", "final_runtime_capture_sha256", "image_verification_sha256",
        })
        closure = load_packaged_worker_closure()
        if (
            verification["schema_version"] != RELEASE_VERIFICATION_SCHEMA
            or verification["status"] != "RELEASE_VERIFIED"
            or verification["runtime_release_digest"] != release.manifest_digest
            or verification["runtime_release_sha256"] != hashlib.sha256(release_raw).hexdigest()
            or verification["worker_closure_digest"] != release.worker_closure_digest
            or release.worker_closure_digest != closure.digest
            or verification["final_runtime_capture_sha256"] != _sha256(capture_raw)
            or verification["image_verification_sha256"] != _sha256(image_raw)
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise DerivedTrainingImageError("RELEASE_VERIFICATION_INVALID") from exc
    promotion = {
        "schema_version": RELEASE_PROMOTION_SCHEMA,
        "status": "PROMOTED_LOCAL_ONLY",
        "runtime_release_digest": release.manifest_digest,
        "runtime_release_sha256": hashlib.sha256(release_raw).hexdigest(),
        "verification_sha256": hashlib.sha256(verification_raw).hexdigest(),
        "final_runtime_capture_sha256": _sha256(capture_raw),
        "image_verification_sha256": _sha256(image_raw),
    }
    if output.suffix.lower() != ".json":
        raise DerivedTrainingImageError("OUTPUT_INVALID")
    _write_exclusive(output, promotion)
    return promotion


__all__ = [
    "BUILD_RECEIPT_SCHEMA", "CANDIDATE_SCHEMA", "FINAL_RUNTIME_CAPTURE_SCHEMA", "RELEASE_PROMOTION_SCHEMA", "RELEASE_VERIFICATION_SCHEMA",
    "DerivedImageProfile", "DerivedTrainingImageError", "DockerLaunchAuthority", "PackageExpectation",
    "PROFILE_SCHEMA", "VERIFICATION_SCHEMA", "build_command", "build_derived_image", "capture_candidate",
    "build_runtime_release", "capture_final_runtime", "load_profile", "plan_profile", "promote_runtime_release", "render_dockerfile", "verify_candidate", "verify_runtime_release",
    "validate_verification_report", "verify_effectful_launch",
]
