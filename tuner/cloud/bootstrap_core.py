"""Stdlib-only, provider-neutral reconstruction of a Synaptic source lock.

This module is deliberately independent of the rest of ``tuner`` so the exact
committed bytes can be placed in a verified bootstrap capsule.  Project types
are adapted to primitive mappings by :mod:`tuner.cloud.checkout`.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import base64
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit
from urllib.parse import quote, quote_plus


SOURCE_LOCK_SCHEMA = "synaptic-source-lock/v1"
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_SCP_RE = re.compile(
    r"^(?:(?P<user>[A-Za-z0-9._-]+)@)?"
    r"(?P<host>[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?):"
    r"(?P<path>[A-Za-z0-9._~/-]+)$"
)
_SAFE_REPO_PATH_RE = re.compile(r"^/[A-Za-z0-9._~/-]+$")
_GITMODULE_KEY = re.compile(r"^submodule\.(?P<name>.+)\.path$")
_SECRET_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")


class BootstrapError(RuntimeError):
    """Fail-closed bootstrap error with secret-safe messages."""


CommandRunner = Callable[..., str]
CloneUrlResolver = Callable[[Mapping[str, object]], str]
CredentialResolver = Callable[[Mapping[str, object]], str]

_SOURCE_LOCK_KEYS = {
    "schema_version", "run_id", "created_at", "mode", "sources", "project",
    "configuration", "plugins", "inputs", "runtime", "outputs",
}
_SOURCE_KEYS = {
    "url", "commit", "branch", "dirty", "pushed", "submodule_path",
    "gitlink_commit", "credential",
}
_LOCATION_KEYS = {"url", "scheme", "host", "path", "credential"}
_POLICY_KEYS = {
    "allowed_hosts", "allowed_schemes", "nested_submodules",
    "max_submodule_depth", "ssh",
}
_SSH_KEYS = {"executable", "agent_socket", "known_hosts"}
_CREDENTIAL_KEYS = {"provider", "name"}


def _require_exact_keys(
    value: Mapping[str, object], *, required: set[str], allowed: set[str], label: str,
) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - allowed
    if missing or extra:
        raise BootstrapError(f"{label} does not match the canonical wire shape")


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise BootstrapError(f"{label} must be a boolean")
    return value


def _is_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def assert_safe_path_components(
    value: str | os.PathLike[str], *, require_leaf: bool = False,
) -> Path:
    """Reject lexical symlink/reparse components before any checkout write."""

    path = Path(os.path.abspath(os.fspath(value)))
    parts = path.parts
    if not parts:
        raise BootstrapError("Checkout path is invalid")
    current = Path(parts[0])
    for index in range(len(parts)):
        if index:
            current = current / parts[index]
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise BootstrapError("Checkout path could not be inspected safely") from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
            raise BootstrapError("Checkout path cannot contain links or reparse points")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise BootstrapError("Checkout path contains a non-directory component")
    if require_leaf and not path.exists():
        raise BootstrapError("Checkout path does not exist")
    return path


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise BootstrapError(f"{label} must be an object")
    return dict(value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BootstrapError(f"{label} must be a non-empty string")
    return value.strip()


def _normalize_repo_path(value: str) -> str:
    path = "/" + value.replace("\\", "/").lstrip("/")
    if not _SAFE_REPO_PATH_RE.fullmatch(path):
        raise BootstrapError("Repository URL path contains unsupported characters")
    normalized = posixpath.normpath(path)
    if normalized in {"/", "/."} or normalized.startswith("/../"):
        raise BootstrapError("Repository URL must contain a valid repository path")
    return normalized


def canonicalize_repository_url(url: str) -> dict[str, object]:
    """Return a credential-free canonical repository-location mapping."""

    candidate = _require_string(url, "Repository URL")
    scp = _SCP_RE.fullmatch(candidate) if "://" not in candidate else None
    if scp:
        user = scp.group("user") or "git"
        if user != "git":
            raise BootstrapError("SSH repository URLs may use only the 'git' username")
        host = scp.group("host").lower()
        path = _normalize_repo_path(scp.group("path"))
        return {"url": f"ssh://git@{host}{path}", "scheme": "ssh", "host": host, "path": path}

    parsed = urlsplit(candidate)
    scheme = parsed.scheme.lower()
    if scheme not in {"https", "ssh"}:
        raise BootstrapError("Repository URL scheme must be https or ssh")
    if parsed.query or parsed.fragment:
        raise BootstrapError("Repository URLs cannot contain query strings or fragments")
    try:
        port = parsed.port
    except ValueError as exc:
        raise BootstrapError("Repository URL has an invalid port") from exc
    if not parsed.hostname:
        raise BootstrapError("Repository URL must contain a host")
    host = parsed.hostname.lower()
    path = _normalize_repo_path(parsed.path)
    if scheme == "https":
        if parsed.username is not None or parsed.password is not None:
            raise BootstrapError("HTTPS repository URLs cannot contain userinfo")
        netloc = host if port in {None, 443} else f"{host}:{port}"
    else:
        if parsed.password is not None:
            raise BootstrapError("SSH repository URLs cannot contain passwords")
        if parsed.username not in {None, "git"}:
            raise BootstrapError("SSH repository URLs may use only the 'git' username")
        netloc = f"git@{host}"
        if port not in {None, 22}:
            netloc += f":{port}"
    return {
        "url": urlunsplit((scheme, netloc, path, "", "")),
        "scheme": scheme,
        "host": host,
        "path": path,
    }


def _location(value: object, policy: Mapping[str, object]) -> dict[str, object]:
    document = _require_mapping(value, "Repository location")
    _require_exact_keys(
        document, required={"url", "scheme", "host", "path"},
        allowed=_LOCATION_KEYS, label="Repository location",
    )
    canonical = canonicalize_repository_url(_require_string(document.get("url"), "Repository URL"))
    for key in ("scheme", "host", "path"):
        declared = document.get(key)
        if declared is not None and declared != canonical[key]:
            raise BootstrapError(f"Repository {key} does not match its canonical URL")
    hosts = policy["allowed_hosts"]
    schemes = policy["allowed_schemes"]
    if canonical["host"] not in hosts:
        raise BootstrapError("Repository host is not allowed")
    if canonical["scheme"] not in schemes:
        raise BootstrapError("Repository scheme is not allowed")
    credential = document.get("credential")
    if credential is not None:
        reference = _require_mapping(credential, "Credential reference")
        _require_exact_keys(
            reference, required=_CREDENTIAL_KEYS, allowed=_CREDENTIAL_KEYS,
            label="Credential reference",
        )
        provider = _require_string(reference.get("provider"), "Credential provider")
        name = _require_string(reference.get("name"), "Credential name")
        if provider not in {"env", "provider_secret", "credential_helper"}:
            raise BootstrapError("Credential provider is unsupported")
        if not _SECRET_NAME_RE.fullmatch(name):
            raise BootstrapError("Credential name is invalid")
        canonical["credential"] = {"provider": provider, "name": name}
    return canonical


def _resolve_relative_repository_url(url: str, parent: Mapping[str, object]) -> dict[str, object]:
    if "://" in url or _SCP_RE.fullmatch(url):
        return canonicalize_repository_url(url)
    if not url or url.startswith(("/", "\\")):
        raise BootstrapError("Relative repository URL must be repository-relative")
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(str(parent["path"])), url))
    if joined.startswith("../"):
        raise BootstrapError("Relative repository URL escapes the repository host root")
    if parent["scheme"] == "ssh":
        return canonicalize_repository_url(f"ssh://git@{parent['host']}/{joined.lstrip('/')}")
    return canonicalize_repository_url(f"https://{parent['host']}/{joined.lstrip('/')}")


def normalize_policy(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the primitive checkout-policy wire document."""

    document = _require_mapping(value, "Checkout policy")
    _require_exact_keys(
        document,
        required={"allowed_hosts", "allowed_schemes", "nested_submodules", "max_submodule_depth"},
        allowed=_POLICY_KEYS,
        label="Checkout policy",
    )
    raw_hosts = document.get("allowed_hosts")
    raw_schemes = document.get("allowed_schemes", ["https", "ssh"])
    if not isinstance(raw_hosts, list) or not raw_hosts:
        raise BootstrapError("Cloud checkout requires at least one allowed repository host")
    if not isinstance(raw_schemes, list) or not raw_schemes:
        raise BootstrapError("Cloud checkout requires approved repository schemes")
    host_values = [_require_string(item, "Allowed repository host") for item in raw_hosts]
    scheme_values = [_require_string(item, "Allowed repository scheme") for item in raw_schemes]
    hosts = sorted(set(host_values))
    schemes = sorted(set(scheme_values))
    if host_values != hosts or any(host != host.lower() for host in host_values):
        raise BootstrapError("Allowed repository hosts must be sorted, unique, and lowercase")
    if scheme_values != schemes or any(scheme != scheme.lower() for scheme in scheme_values):
        raise BootstrapError("Allowed repository schemes must be sorted, unique, and lowercase")
    if not set(schemes) <= {"https", "ssh"}:
        raise BootstrapError("Cloud checkout permits only approved https and ssh schemes")
    depth = document["max_submodule_depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 16:
        raise BootstrapError("Submodule depth must be between 0 and 16")
    normalized: dict[str, object] = {
        "allowed_hosts": hosts,
        "allowed_schemes": schemes,
        "nested_submodules": _require_bool(document["nested_submodules"], "nested_submodules"),
        "max_submodule_depth": depth,
    }
    ssh = document.get("ssh")
    if ssh is not None:
        ssh_document = _require_mapping(ssh, "Controlled SSH policy")
        _require_exact_keys(
            ssh_document, required=_SSH_KEYS, allowed=_SSH_KEYS,
            label="Controlled SSH policy",
        )
        executable = Path(_require_string(ssh_document.get("executable"), "SSH executable"))
        known_hosts = Path(_require_string(ssh_document.get("known_hosts"), "SSH known_hosts"))
        agent_socket = _require_string(ssh_document.get("agent_socket"), "SSH agent socket")
        if not executable.is_absolute() or not known_hosts.is_absolute():
            raise BootstrapError("Controlled SSH paths must be absolute")
        if not executable.is_file() or executable.is_symlink():
            raise BootstrapError("Controlled SSH executable must be an absolute regular file")
        if not known_hosts.is_file() or known_hosts.is_symlink():
            raise BootstrapError("Controlled SSH requires an explicit regular known_hosts file")
        normalized["ssh"] = {
            "executable": str(executable.resolve()),
            "agent_socket": agent_socket,
            "known_hosts": str(known_hosts.resolve()),
        }
    return normalized


def normalize_source_lock(
    value: Mapping[str, object], policy: Mapping[str, object], *,
    allow_legacy_metadata: bool = False,
) -> dict[str, object]:
    """Validate the canonical source-lock wire document for reconstruction."""

    document = _require_mapping(value, "Source lock")
    _require_exact_keys(
        document, required=_SOURCE_LOCK_KEYS, allowed=_SOURCE_LOCK_KEYS,
        label="Source lock",
    )
    if document.get("schema_version") != SOURCE_LOCK_SCHEMA:
        raise BootstrapError("Unsupported source-lock schema version")
    _require_string(document.get("run_id"), "Source lock run_id")
    created_at = _require_string(document.get("created_at"), "Source lock created_at")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BootstrapError("Source lock created_at must be an ISO date-time") from exc
    if parsed_created_at.tzinfo is None:
        raise BootstrapError("Source lock created_at must include a timezone")
    mode = document.get("mode")
    if mode not in {"standalone", "superproject", "dual_clone"}:
        raise BootstrapError("Unsupported source-lock mode")
    sources = _require_mapping(document.get("sources"), "Source lock sources")
    _require_exact_keys(
        sources, required={"project", "engine"}, allowed={"project", "engine"},
        label="Source lock sources",
    )
    project = _require_mapping(document.get("project"), "Source lock project")
    configuration = _require_mapping(document.get("configuration"), "Source lock configuration")
    for key in ("runtime", "outputs"):
        _require_mapping(document.get(key), f"Source lock {key}")
    if not allow_legacy_metadata:
        for key in ("manifest_uri", "manifest_sha256", "engine_requires"):
            _require_string(project.get(key), f"Source lock project.{key}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(project["manifest_sha256"])):
            raise BootstrapError("Source lock project.manifest_sha256 must be lowercase SHA-256")
        for key in ("resolved_uri", "resolved_sha256"):
            _require_string(configuration.get(key), f"Source lock configuration.{key}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(configuration["resolved_sha256"])):
            raise BootstrapError("Source lock configuration.resolved_sha256 must be lowercase SHA-256")
        documents = configuration.get("documents")
        if not isinstance(documents, list) or any(not isinstance(item, Mapping) for item in documents):
            raise BootstrapError("Source lock configuration.documents must be an array of objects")
    for key in ("plugins", "inputs"):
        sequence = document.get(key)
        if not isinstance(sequence, list) or any(not isinstance(item, Mapping) for item in sequence):
            raise BootstrapError(f"Source lock {key} must be an array of objects")

    def source(label: str) -> dict[str, object]:
        item = _require_mapping(sources.get(label), f"Source lock {label} source")
        _require_exact_keys(
            item, required={"url", "commit", "dirty", "pushed"},
            allowed=_SOURCE_KEYS, label=f"Source lock {label} source",
        )
        raw_location: dict[str, object] = canonicalize_repository_url(
            _require_string(item.get("url"), "Repository URL")
        )
        if item["url"] != raw_location["url"]:
            raise BootstrapError("Repository URL must use its canonical form")
        if item.get("credential") is not None:
            credential = _require_mapping(item["credential"], "Credential reference")
            _require_exact_keys(
                credential, required=_CREDENTIAL_KEYS, allowed=_CREDENTIAL_KEYS,
                label="Credential reference",
            )
            raw_location["credential"] = credential
        location = _location(raw_location, policy)
        commit = _require_string(item.get("commit"), f"{label} commit").lower()
        if not _COMMIT_RE.fullmatch(commit):
            raise BootstrapError("Source commit must be a full 40- or 64-character hash")
        if item["commit"] != commit:
            raise BootstrapError("Source commit must use lowercase canonical form")
        if "branch" in item and not isinstance(item["branch"], str):
            raise BootstrapError("Source branch must be a string")
        if _require_bool(item["dirty"], f"{label} dirty"):
            raise BootstrapError(f"Cloud checkout requires a clean {label} worktree")
        if not _require_bool(item["pushed"], f"{label} pushed"):
            raise BootstrapError(f"Cloud checkout requires the exact {label} commit to be pushed")
        result = dict(location)
        result.update({"commit": commit, "dirty": False, "pushed": True})
        submodule_path = item.get("submodule_path")
        gitlink = item.get("gitlink_commit")
        if submodule_path is not None:
            path = _require_string(submodule_path, "Engine submodule path").replace("\\", "/")
            if path.startswith("/") or ".." in path.split("/"):
                raise BootstrapError("Submodule path must be a contained relative path")
            result["submodule_path"] = path
        if gitlink is not None:
            gitlink_value = _require_string(gitlink, "Engine gitlink commit").lower()
            if not _COMMIT_RE.fullmatch(gitlink_value):
                raise BootstrapError("Gitlink commit must be a full 40- or 64-character hash")
            if gitlink != gitlink_value:
                raise BootstrapError("Gitlink commit must use lowercase canonical form")
            result["gitlink_commit"] = gitlink_value
        return result

    project = source("project")
    engine = source("engine")
    if mode == "standalone":
        if project["commit"] != engine["commit"] or project["url"] != engine["url"]:
            raise BootstrapError("Standalone mode requires identical project and engine sources")
    else:
        if not engine.get("submodule_path") or engine.get("gitlink_commit") != engine["commit"]:
            raise BootstrapError(f"{mode} mode requires the locked engine gitlink")
    return {"schema_version": SOURCE_LOCK_SCHEMA, "mode": mode, "sources": {"project": project, "engine": engine}}


def redact(text: object, secrets: tuple[str, ...] = ()) -> str:
    rendered = str(text)
    for secret in secrets:
        if secret:
            raw = secret.encode("utf-8")
            variants = {
                secret,
                quote(secret, safe=""),
                quote(secret, safe="").lower(),
                quote_plus(secret, safe=""),
                quote_plus(secret, safe="").lower(),
                base64.b64encode(raw).decode("ascii"),
                base64.urlsafe_b64encode(raw).decode("ascii"),
            }
            for prefix in (b"x-access-token:", b"oauth2:"):
                encoded = base64.b64encode(prefix + raw).decode("ascii")
                variants.update({encoded, encoded.rstrip("=")})
            variants.update({value.rstrip("=") for value in tuple(variants) if len(value) > 4})
            for variant in sorted(variants, key=len, reverse=True):
                if variant:
                    rendered = rendered.replace(variant, "<redacted>")
    rendered = re.sub(r"(https?://)[^/@\s]+@", r"\1<redacted>@", rendered)
    return rendered


_NO_REPLACE_ENVIRONMENT_KEY = "GIT_NO_REPLACE_OBJECTS"


def _force_no_replace_objects(environment: Mapping[str, str]) -> dict[str, str]:
    """Return one canonical replacement-object guard without rewriting other keys."""

    normalized = {
        key: value for key, value in environment.items()
        if key.upper() != _NO_REPLACE_ENVIRONMENT_KEY
    }
    normalized[_NO_REPLACE_ENVIRONMENT_KEY] = "1"
    return normalized


def git_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a minimal cross-platform environment with no Git injection knobs."""

    source = dict(os.environ)
    if overrides:
        source.update(overrides)
    allowed = {
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
        "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE",
    }
    environment = {key: value for key, value in source.items() if key.upper() in allowed}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": "https:ssh",
        }
    )
    return _force_no_replace_objects(environment)


def run_git(
    arguments: list[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None,
    secrets: tuple[str, ...] = (),
) -> str:
    process_environment = _force_no_replace_objects(
        dict(env) if env is not None else git_environment()
    )
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=str(cwd) if cwd else None,
            env=process_environment, capture_output=True,
            text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BootstrapError(f"Git checkout operation failed: {redact(exc, secrets)}") from exc
    if completed.returncode:
        detail = redact(completed.stderr.strip() or completed.stdout.strip(), secrets)
        raise BootstrapError(f"Git checkout operation failed: {detail or 'unknown error'}")
    return completed.stdout.strip()


def _environment_credential(reference: Mapping[str, object], environment: Mapping[str, str]) -> str:
    provider = reference.get("provider")
    name = reference.get("name")
    if provider != "env" or not isinstance(name, str) or not name:
        raise BootstrapError("Remote bootstrap supports only environment-backed credential references")
    value = environment.get(name)
    if not value:
        raise BootstrapError("Required repository credential is unavailable")
    return value


@contextmanager
def credential_scope(
    location: Mapping[str, object], *, policy: Mapping[str, object],
    environment: Mapping[str, str] | None, credential_resolver: CredentialResolver | None,
) -> Iterator[tuple[list[str], dict[str, str], tuple[str, ...]]]:
    """Expose one credential only to a scoped helper and always remove it."""

    helper_dir = Path(tempfile.mkdtemp(prefix="synaptic-git-credential-"))
    try:
        home_dir = helper_dir / "home"
        template_dir = helper_dir / "template"
        hooks_dir = helper_dir / "hooks"
        for path in (home_dir, template_dir, hooks_dir):
            path.mkdir()
        process_env = git_environment(environment)
        process_env["HOME"] = str(home_dir)
        process_env["USERPROFILE"] = str(home_dir)
        config = [
            "-c", "credential.helper=", "-c", "protocol.file.allow=never", "-c",
            "protocol.ext.allow=never", "-c", f"core.hooksPath={hooks_dir}", "-c",
            f"init.templateDir={template_dir}",
        ]
        reference = location.get("credential")
        if reference is None:
            if location["scheme"] == "ssh":
                ssh = policy.get("ssh")
                if not isinstance(ssh, Mapping):
                    raise BootstrapError("SSH checkout requires an explicit controlled agent and known_hosts policy")
                process_env["SSH_AUTH_SOCK"] = str(ssh["agent_socket"])
                process_env["GIT_SSH_VARIANT"] = "ssh"
                arguments = [
                    str(ssh["executable"]), "-F", os.devnull, "-oBatchMode=yes",
                    "-oStrictHostKeyChecking=yes", f"-oUserKnownHostsFile={ssh['known_hosts']}",
                    "-oGlobalKnownHostsFile=none", "-oIdentityFile=none", "-oIdentitiesOnly=no",
                    "-oProxyCommand=none", "-oProxyJump=none", "-oForwardAgent=no",
                    "-oClearAllForwardings=yes", "-oPermitLocalCommand=no", "-oLocalCommand=none",
                    "-oRequestTTY=no",
                ]
                process_env["GIT_SSH_COMMAND"] = " ".join(shlex.quote(item) for item in arguments)
            yield config, process_env, ()
            return

        reference_mapping = _require_mapping(reference, "Credential reference")
        if location["scheme"] != "https":
            raise BootstrapError("SecretRef-backed checkout requires HTTPS; SSH must use an external agent")
        resolver = credential_resolver or (lambda item: _environment_credential(item, environment or os.environ))
        value = resolver(reference_mapping)
        if not isinstance(value, str) or not value:
            raise BootstrapError("Required repository credential is unavailable")
        helper = helper_dir / "credential_helper.py"
        helper.write_text(
            "import os, sys\n"
            "request = dict(line.rstrip('\\n').split('=', 1) for line in sys.stdin if '=' in line)\n"
            "if request.get('host', '').lower() != os.environ['SYNAPTIC_GIT_HOST'].lower():\n"
            "    raise SystemExit(1)\n"
            "print('username=x-access-token')\n"
            "print('password=' + os.environ['SYNAPTIC_GIT_SECRET'])\n",
            encoding="utf-8",
        )
        process_env["SYNAPTIC_GIT_HOST"] = str(location["host"])
        process_env["SYNAPTIC_GIT_SECRET"] = value
        config.extend(
            ["-c", f"credential.https://{location['host']}.helper=!\"{sys.executable}\" \"{helper}\"",
             "-c", "credential.useHttpPath=true"]
        )
        try:
            yield config, process_env, (value,)
        finally:
            process_env.pop("SYNAPTIC_GIT_SECRET", None)
            process_env.pop("SYNAPTIC_GIT_HOST", None)
    finally:
        shutil.rmtree(helper_dir, ignore_errors=True)


def _clone_exact(
    source: Mapping[str, object], destination: Path, *, policy: Mapping[str, object],
    clone_url_resolver: CloneUrlResolver | None, environment: Mapping[str, str] | None,
    credential_resolver: CredentialResolver | None, command_runner: CommandRunner,
) -> None:
    destination = assert_safe_path_components(destination)
    location = {key: source[key] for key in ("url", "scheme", "host", "path")}
    if "credential" in source:
        location["credential"] = source["credential"]
    clone_url = clone_url_resolver(location) if clone_url_resolver else str(location["url"])
    with credential_scope(location, policy=policy, environment=environment, credential_resolver=credential_resolver) as (
        config, process_env, secrets,
    ):
        if clone_url_resolver:
            fixture_config: list[str] = []
            for index in range(0, len(config), 2):
                if config[index + 1] != "protocol.file.allow=never":
                    fixture_config.extend(config[index:index + 2])
            config = [*fixture_config, "-c", "protocol.file.allow=always"]
            process_env["GIT_ALLOW_PROTOCOL"] = "https:ssh:file"
        template = next(item.split("=", 1)[1] for item in config if item.startswith("init.templateDir="))
        command_runner(
            [*config, "clone", f"--template={template}", "--no-checkout", "--no-recurse-submodules",
             clone_url, str(destination)], env=process_env, secrets=secrets,
        )
        process_env.pop("SYNAPTIC_GIT_SECRET", None)
        process_env.pop("SYNAPTIC_GIT_HOST", None)
        command_runner([*config, "checkout", "--detach", str(source["commit"])], cwd=destination, env=process_env, secrets=secrets)
        actual = command_runner(["rev-parse", "HEAD"], cwd=destination, env=process_env, secrets=secrets)
    if actual.lower() != source["commit"]:
        raise BootstrapError("Checkout HEAD does not match the exact locked commit")


def _read_submodules(
    repository: Path, parent_location: Mapping[str, object], *, policy: Mapping[str, object],
    depth: int, command_runner: CommandRunner,
) -> list[dict[str, object]]:
    document = repository / ".gitmodules"
    if not document.is_file():
        return []
    paths = command_runner(["config", "--file", str(document), "--get-regexp", r"^submodule\..*\.path$"], cwd=repository)
    if not paths:
        return []
    if depth > 0 and not policy["nested_submodules"]:
        raise BootstrapError("Nested submodules are disabled by project policy")
    if depth >= policy["max_submodule_depth"]:
        raise BootstrapError("Submodule graph exceeds the approved maximum depth")
    entries: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    seen_casefolded_paths: set[str] = set()
    for line in paths.splitlines():
        key, separator, raw_path = line.partition(" ")
        match = _GITMODULE_KEY.fullmatch(key)
        if not separator or not match:
            raise BootstrapError("Malformed .gitmodules submodule path entry")
        path = raw_path.strip().replace("\\", "/")
        if (
            not path or path.startswith("/") or ".." in path.split("/")
            or path in seen_paths or path.casefold() in seen_casefolded_paths
        ):
            raise BootstrapError("Submodule path must be unique and contained")
        seen_paths.add(path)
        seen_casefolded_paths.add(path.casefold())
        name = match.group("name")
        raw_url = command_runner(["config", "--file", str(document), "--get", f"submodule.{name}.url"], cwd=repository)
        try:
            location = (
                _resolve_relative_repository_url(raw_url, parent_location)
                if "://" not in raw_url and not _SCP_RE.fullmatch(raw_url)
                else canonicalize_repository_url(raw_url)
            )
            location = _location(location, policy)
        except BootstrapError as exc:
            raise BootstrapError("Rejected .gitmodules repository URL") from exc
        tree_line = command_runner(["ls-tree", "HEAD", "--", path], cwd=repository)
        fields = tree_line.split()
        if len(fields) < 3 or fields[0] != "160000":
            raise BootstrapError("Submodule declaration does not match a committed gitlink")
        entries.append({"name": name, "path": path, "location": location, "commit": fields[2].lower()})
    return entries


def _materialize_submodules(
    repository: Path, parent_location: Mapping[str, object], *, policy: Mapping[str, object], depth: int,
    locked_engine: Mapping[str, object], clone_url_resolver: CloneUrlResolver | None,
    environment: Mapping[str, str] | None, credential_resolver: CredentialResolver | None,
    command_runner: CommandRunner,
) -> None:
    entries = _read_submodules(repository, parent_location, policy=policy, depth=depth, command_runner=command_runner)
    for entry in entries:
        location = dict(entry["location"])
        if entry["path"] == locked_engine.get("submodule_path") and depth == 0:
            if entry["commit"] != locked_engine["commit"]:
                raise BootstrapError("Host gitlink does not match the locked engine commit")
            if location["url"] != locked_engine["url"]:
                raise BootstrapError("Host engine submodule URL does not match the source lock")
            if "credential" in locked_engine:
                location["credential"] = locked_engine["credential"]
        elif location["host"] == parent_location["host"] and "credential" in parent_location:
            location["credential"] = parent_location["credential"]
        child = {**location, "commit": entry["commit"], "dirty": False, "pushed": True}
        repository_root = assert_safe_path_components(repository, require_leaf=True)
        target = assert_safe_path_components(repository_root / str(entry["path"]))
        try:
            if os.path.commonpath((str(repository_root), str(target))) != str(repository_root):
                raise ValueError
        except (ValueError, OSError) as exc:
            raise BootstrapError("Submodule destination escapes its parent checkout") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        _clone_exact(child, target, policy=policy, clone_url_resolver=clone_url_resolver,
                     environment=environment, credential_resolver=credential_resolver, command_runner=command_runner)
        _materialize_submodules(
            target, location, policy=policy, depth=depth + 1, locked_engine=locked_engine,
            clone_url_resolver=clone_url_resolver, environment=environment,
            credential_resolver=credential_resolver, command_runner=command_runner,
        )


def reconstruct_source_lock(
    source_lock: Mapping[str, object], destination: str | os.PathLike[str], *,
    policy: Mapping[str, object], clone_url_resolver: CloneUrlResolver | None = None,
    environment: Mapping[str, str] | None = None,
    credential_resolver: CredentialResolver | None = None,
    command_runner: CommandRunner = run_git,
    allow_legacy_metadata: bool = False,
) -> dict[str, object]:
    """Reconstruct and verify a lock from primitive, JSON-compatible values."""

    normalized_policy = normalize_policy(policy)
    lock = normalize_source_lock(
        source_lock, normalized_policy, allow_legacy_metadata=allow_legacy_metadata,
    )
    project_source = lock["sources"]["project"]
    engine_source = lock["sources"]["engine"]
    root = assert_safe_path_components(destination)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise BootstrapError("Checkout destination must be empty")
    root.mkdir(parents=True, exist_ok=True)
    assert_safe_path_components(root, require_leaf=True)
    project_root = root / "project"
    _clone_exact(project_source, project_root, policy=normalized_policy,
                 clone_url_resolver=clone_url_resolver, environment=environment,
                 credential_resolver=credential_resolver, command_runner=command_runner)
    if lock["mode"] == "standalone":
        engine_root = project_root
    else:
        engine_path = str(engine_source["submodule_path"])
        entries = _read_submodules(project_root, project_source, policy=normalized_policy, depth=0, command_runner=command_runner)
        matching = [entry for entry in entries if entry["path"] == engine_path]
        if len(matching) != 1 or matching[0]["commit"] != engine_source["commit"]:
            raise BootstrapError("Locked host gitlink does not identify the locked engine commit")
        if matching[0]["location"]["url"] != engine_source["url"]:
            raise BootstrapError("Locked host submodule URL does not identify the locked engine source")
        if lock["mode"] == "superproject":
            _materialize_submodules(
                project_root, project_source, policy=normalized_policy, depth=0, locked_engine=engine_source,
                clone_url_resolver=clone_url_resolver, environment=environment,
                credential_resolver=credential_resolver, command_runner=command_runner,
            )
            engine_root = (project_root / engine_path).resolve()
        else:
            engine_root = root / "engine"
            _clone_exact(engine_source, engine_root, policy=normalized_policy,
                         clone_url_resolver=clone_url_resolver, environment=environment,
                         credential_resolver=credential_resolver, command_runner=command_runner)
            _materialize_submodules(
                engine_root, engine_source, policy=normalized_policy, depth=1, locked_engine=engine_source,
                clone_url_resolver=clone_url_resolver, environment=environment,
                credential_resolver=credential_resolver, command_runner=command_runner,
            )
    actual_engine = command_runner(["rev-parse", "HEAD"], cwd=engine_root)
    if actual_engine.lower() != engine_source["commit"]:
        raise BootstrapError("Reconstructed engine does not match the source lock")
    return {
        "schema_version": "synaptic-bootstrap-result/v1",
        "project_root": str(project_root),
        "engine_root": str(engine_root),
        "project_commit": project_source["commit"],
        "engine_commit": engine_source["commit"],
    }


def reconstruct_source_lock_json(
    source_lock_json: bytes, policy_json: bytes, destination: str | os.PathLike[str], *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Remote entrypoint: parse authenticated JSON bytes, then reconstruct."""

    class DuplicateObjectKey(ValueError):
        pass

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateObjectKey
            result[key] = value
        return result

    try:
        source_lock = json.loads(
            source_lock_json.decode("utf-8"), object_pairs_hook=unique_object,
        )
        policy = json.loads(
            policy_json.decode("utf-8"), object_pairs_hook=unique_object,
        )
    except DuplicateObjectKey:
        raise BootstrapError("Bootstrap input JSON contains duplicate object keys") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BootstrapError("Bootstrap input JSON is invalid") from None
    return reconstruct_source_lock(source_lock, destination, policy=policy, environment=environment)


__all__ = [
    "BootstrapError", "SOURCE_LOCK_SCHEMA", "canonicalize_repository_url", "credential_scope",
    "git_environment", "normalize_policy", "normalize_source_lock", "reconstruct_source_lock",
    "reconstruct_source_lock_json", "redact", "run_git",
]
