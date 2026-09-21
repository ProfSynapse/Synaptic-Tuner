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


PROFILE_SCHEMA = "synaptic-derived-training-image-profile/v1"
BUILD_RECEIPT_SCHEMA = "synaptic-derived-training-image-build-receipt/v1"
CANDIDATE_SCHEMA = "synaptic-derived-training-image-candidate/v1"
VERIFICATION_SCHEMA = "synaptic-derived-training-image-verification/v1"
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
    config_directory_identity: tuple[int, int, int]
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
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        raw = path.read_bytes()
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
        or not 1 <= len(packages) <= 64
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
    return DerivedImageProfile(
        name=name,
        base_image=base_image,
        python_executable=python_executable,
        packages=tuple(expectations),
        canonical_sha256=_sha256(_canonical_bytes(normalized_profile)),
    )


def render_dockerfile(profile: DerivedImageProfile) -> str:
    requirements = " \\\n    ".join(shlex.quote(item.requirement()) for item in profile.packages)
    return (
        f"FROM {profile.base_image}\n"
        f"LABEL ai.synapticlabs.derived-training-profile={json.dumps(profile.canonical_sha256)} \\\n"
        f"      ai.synapticlabs.derived-training-base={json.dumps(profile.base_image)}\n"
        f"RUN {shlex.quote(profile.python_executable)} -m pip install --no-cache-dir --upgrade \\\n"
        f"    {requirements}\n"
    )


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
            config_directory_stat.st_mtime_ns,
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
            or authority.config_directory.is_symlink()
            or not authority.config_directory.is_dir()
            or authority.config_file.is_symlink()
            or not authority.config_file.is_file()
            or tuple(authority.config_directory.iterdir()) != (authority.config_file,)
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
            or (
                config_directory_stat.st_dev,
                config_directory_stat.st_ino,
                config_directory_stat.st_mtime_ns,
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
    *, docker: Path, docker_config: Path, tag: str, dockerfile: Path, metadata_file: Path
) -> CommandSpec:
    _validate_tag(tag)
    return CommandSpec(
        argv=(
            str(docker), "--config", str(docker_config), "buildx", "build",
            "--pull", "--no-cache", "--platform", PLATFORM, "--load",
            "--provenance=mode=max", "--tag", tag, "--metadata-file",
            str(metadata_file), "--file", str(dockerfile), str(dockerfile.parent),
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
    payload = _canonical_bytes(value)
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
        if path.read_bytes() != payload:
            raise OSError("readback mismatch")
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
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
        _run_buildx_authorized(authority, buildx_authority, runner, build_command(
            docker=authority.executable,
            docker_config=command_config,
            tag=tag,
            dockerfile=dockerfile, metadata_file=metadata_path,
        ))
        try:
            metadata_raw = metadata_path.read_bytes()
            if not metadata_raw or len(metadata_raw) > MAX_DOCUMENT_BYTES:
                raise ValueError("metadata size")
            metadata = _load_json(metadata_raw)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID") from exc
        if not isinstance(metadata, dict):
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID")
        try:
            metadata_bytes = _canonical_bytes(metadata)
        except DerivedTrainingImageError as exc:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID") from exc
        if len(metadata_bytes) > MAX_DOCUMENT_BYTES:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID")
        final_digest = metadata.get("containerimage.digest")
        config_digest = metadata.get("containerimage.config.digest")
        if (
            not isinstance(final_digest, str) or _DIGEST.fullmatch(final_digest) is None
            or not isinstance(config_digest, str) or _DIGEST.fullmatch(config_digest) is None
        ):
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID")
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
        if image_id != config_digest:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID")
        if labels.get("ai.synapticlabs.derived-training-profile") != profile.canonical_sha256:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID")
        if labels.get("ai.synapticlabs.derived-training-base") != profile.base_image:
            raise DerivedTrainingImageError("BUILD_METADATA_INVALID")
        receipt = {
            "schema_version": BUILD_RECEIPT_SCHEMA,
            "status": "BUILT_NOT_ATTESTED",
            "profile_sha256": profile.canonical_sha256,
            "base_image": profile.base_image,
            "dockerfile_sha256": _sha256(dockerfile_text.encode("utf-8")),
            "tag": tag,
            "final_oci_reference": f"{repository}@{final_digest}",
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
        "runtime_bytes_sha256": _sha256(runtime_result.stdout),
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
        or build_metadata.get("containerimage.config.digest")
        != value.get("image_config_digest")
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
        or build_metadata.get("containerimage.config.digest")
        != candidate.get("image_config_digest")
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
    })
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
        or build_metadata.get("containerimage.config.digest")
        != image_config_digest
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


__all__ = [
    "BUILD_RECEIPT_SCHEMA", "CANDIDATE_SCHEMA",
    "DerivedImageProfile", "DerivedTrainingImageError", "DockerLaunchAuthority", "PackageExpectation",
    "PROFILE_SCHEMA", "VERIFICATION_SCHEMA", "build_command", "build_derived_image", "capture_candidate",
    "load_profile", "plan_profile", "render_dockerfile", "verify_candidate",
    "validate_verification_report", "verify_effectful_launch",
]
