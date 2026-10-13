"""Explicit, present-only Docker CLI ports; no durable claims or container TTL.

The caller supplies an already validated launch authority and daemon identity.
Local attempt claims last only for this session. Unknown effects are never retried.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import stat
import time
from types import MappingProxyType

from tuner.cloud.derived_training_image import DockerLaunchAuthority
from tuner.cloud.hf_training_image_lock import CommandSpec, subprocess_runner
from .model import (
    DockerCreateDispositionV1 as Create, DockerCreateResultV1,
    DockerStartDispositionV1 as Start, DockerStartResultV1,
    DockerLookupDispositionV1 as Lookup, DockerLookupResultV1,
    DockerRunPhaseV1 as Phase, validated_profile_snapshot,
)

_ID = re.compile(r"[0-9a-f]{64}\Z")
_PREFIX = "ai.synapticlabs.docker-v1."
_INSPECT = '{{json .Id}}\n{{json .Name}}\n{{json .Image}}\n{{json .Config.Labels}}\n{{json .State.Status}}\n{{json .State.ExitCode}}'


class DockerCLISessionV1:
    """Injected image/control ports with explicit mount destinations and values.

    ``arguments[0]`` is the entrypoint; remaining arguments replace image CMD.
    ``deadline`` bounds CLI work, not the lifetime of a started container.
    Source and artifact maps contain (host directory, container directory) pairs.
    """

    def __init__(self, *, authority: DockerLaunchAuthority, endpoint: str,
                 daemon_id: str, image_refs, source_roots, artifact_roots,
                 environment, deadline: float, runner=subprocess_runner, profile=None, source_submit_check=None):
        if type(authority) is not DockerLaunchAuthority:
            raise ValueError("docker_cli_configuration_invalid")
        if type(endpoint) is not str or not endpoint.startswith("unix:///") or len(endpoint) > 4096 or any(c in endpoint for c in "\n\r\0"):
            raise ValueError("docker_cli_configuration_invalid")
        if not isinstance(daemon_id, str) or not daemon_id or len(daemon_id) > 256 or any(c in daemon_id for c in "\n\r\0"):
            raise ValueError("docker_cli_configuration_invalid")
        if isinstance(deadline, bool) or not math.isfinite(deadline):
            raise ValueError("docker_cli_configuration_invalid")
        self._authority, self._endpoint, self._daemon = authority, endpoint, daemon_id
        self._deadline, self._runner = deadline, runner
        images = dict(image_refs)
        if len(images) > 256 or any(type(k) is not str or type(v) is not str or
                                  len(k) > 4096 or len(v) > 4096 for k, v in images.items()):
            raise ValueError("docker_cli_configuration_invalid")
        self._images = MappingProxyType(images)
        self._sources = self._mounts(source_roots)
        self._artifacts = self._mounts(artifact_roots)
        self._mount_identities = MappingProxyType({
            host: self._directory_chain(host)
            for host, _ in (*self._sources.values(), *self._artifacts.values())
        })
        values = dict(environment)
        if len(values) > 64:
            raise ValueError("docker_cli_configuration_invalid")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or
               type(v) is not str or '\0' in v for k, v in values.items()):
            raise ValueError("docker_cli_configuration_invalid")
        if sum(len(k.encode()) + len(v.encode()) for k, v in values.items()) > 32768:
            raise ValueError("docker_cli_configuration_invalid")
        self._environment = MappingProxyType(values)
        if source_submit_check is not None and not callable(source_submit_check):
            raise ValueError("docker_cli_configuration_invalid")
        self._source_submit_check = source_submit_check
        self._profile = None if profile is None else validated_profile_snapshot(profile)
        self._created, self._started, self._owned = set(), set(), {}

    @staticmethod
    def _mounts(mapping):
        result = {}
        if len(mapping) > 256:
            raise ValueError("docker_cli_configuration_invalid")
        for ref, pair in mapping.items():
            host, destination = pair
            host = str(host)
            if type(destination) is not str or len(host) > 4096 or len(destination) > 4096:
                raise ValueError("docker_cli_configuration_invalid")
            if any(c in host + destination for c in ',\n\r\0') or not Path(host).is_absolute() or not Path(destination).is_absolute():
                raise ValueError("docker_cli_configuration_invalid")
            if '..' in Path(host).parts or '..' in Path(destination).parts:
                raise ValueError("docker_cli_configuration_invalid")
            if any(parent.is_symlink() for parent in (Path(host), *Path(host).parents)):
                raise ValueError("docker_cli_configuration_invalid")
            if not Path(host).is_dir() or Path(host).is_symlink():
                raise ValueError("docker_cli_configuration_invalid")
            result[ref] = (str(Path(host).resolve(strict=True)), str(Path(destination)))
        return MappingProxyType(result)

    @staticmethod
    def _directory_chain(host):
        """Observe directory identities; this is not atomic daemon path custody."""
        path = Path(host)
        identities = []
        for parent in (path, *path.parents):
            observed = parent.lstat()
            if not stat.S_ISDIR(observed.st_mode):
                raise ValueError("docker_cli_configuration_invalid")
            identities.append((str(parent), observed.st_dev, observed.st_ino))
        return tuple(identities)

    def _recheck_mounts(self, source, artifact):
        for host in (source, artifact):
            if self._directory_chain(host) != self._mount_identities[host]:
                raise ValueError("docker_cli_configuration_invalid")

    def _invoke(self, args):
        remaining = math.floor(self._deadline - time.monotonic())
        if remaining < 1:
            raise ValueError("docker_cli_deadline")
        spec = CommandSpec(tuple(self._authority.command(['--host', self._endpoint, *args])),
                           self._authority.environment, min(30, remaining), 65536)
        try:
            result = self._runner(spec)
        finally:
            self._authority.assert_current()
        if type(result.stdout) is not bytes or type(result.stderr) is not bytes or len(result.stdout) + len(result.stderr) > 65536:
            raise ValueError("docker_cli_reply_invalid")
        return result.stdout.decode('utf-8', errors='strict')

    def _identity(self):
        if self._invoke(['info', '--format', '{{.ID}}']).strip() != self._daemon:
            raise ValueError("docker_cli_identity_changed")

    def _checked(self, args, *, before=None):
        self._identity()
        try:
            if before is not None:
                before()
            return self._invoke(args)
        finally:
            self._identity()

    @staticmethod
    def _name(labels):
        return labels.container_name

    @staticmethod
    def _labels(labels):
        return {_PREFIX + k: v for k, v in labels.to_dict().items()}

    def _ids(self, labels):
        rows = self._checked(['container', 'ls', '--all', '--no-trunc', '--filter',
                              'name=^/' + self._name(labels) + '$', '--format', '{{.ID}}']).splitlines()
        if len(rows) > 2 or any(not _ID.fullmatch(row) for row in rows):
            raise ValueError("docker_cli_reply_invalid")
        return rows

    def _inspect(self, identifier, labels, image_id=None):
        if not _ID.fullmatch(identifier):
            raise ValueError("docker_cli_reply_invalid")
        rows = self._checked(['container', 'inspect', '--format', _INSPECT, identifier]).splitlines()
        if len(rows) != 6:
            raise ValueError("docker_cli_reply_invalid")
        cid, name, image, actual, status, exit_code = map(json.loads, rows)
        if cid != identifier or name != '/' + self._name(labels) or not isinstance(actual, dict):
            raise ValueError("docker_cli_identity_changed")
        if {k: v for k, v in actual.items() if k.startswith(_PREFIX)} != self._labels(labels):
            raise ValueError("docker_cli_identity_changed")
        if image_id is not None and image != image_id:
            raise ValueError("docker_cli_identity_changed")
        if type(exit_code) is not int:
            raise ValueError("docker_cli_reply_invalid")
        return status, exit_code

    def _image(self, image):
        ref = self._images[image.image_ref]
        if ref != image.image_digest and not ref.endswith('@' + image.image_digest):
            raise ValueError("docker_cli_image_invalid")
        rows = self._checked(['image', 'inspect', '--format', '{{json .Id}}\n{{json .RepoDigests}}', ref]).splitlines()
        if len(rows) != 2:
            raise ValueError("docker_cli_reply_invalid")
        identifier, digests = map(json.loads, rows)
        if type(identifier) is not str or not identifier.startswith('sha256:') or not _ID.fullmatch(identifier[7:]):
            raise ValueError("docker_cli_reply_invalid")
        if ref == image.image_digest:
            if identifier != image.image_digest:
                raise ValueError("docker_cli_image_invalid")
        elif not isinstance(digests, list) or ref not in digests:
            raise ValueError("docker_cli_image_invalid")
        return identifier

    def require_present(self, image):
        try:
            self._image(image)
            return True
        except Exception:
            return False

    def create_once(self, *, labels, image, runtime, workload, source_ref, artifact_ref):
        if labels.digest in self._created:
            return DockerCreateResultV1(Create.COLLISION)
        try:
            if tuple(self._environment) != workload.environment_keys and tuple(sorted(self._environment)) != workload.environment_keys:
                raise ValueError("docker_cli_configuration_invalid")
            source, source_dest = self._sources[source_ref]
            artifact, artifact_dest = self._artifacts[artifact_ref]
            self._recheck_mounts(source, artifact)
            if (Path(source) in (Path(artifact), *Path(artifact).parents) or
                Path(artifact) in Path(source).parents or
                Path(source_dest) in (Path(artifact_dest), *Path(artifact_dest).parents) or
                Path(artifact_dest) in Path(source_dest).parents):
                raise ValueError("docker_cli_configuration_invalid")
            args = ['container', 'create', '--pull=never', '--name', self._name(labels),
                    '--network=none', '--cpus', str(runtime.cpu_count), '--memory', str(runtime.memory_bytes),
                    '--mount', f'type=bind,src={source},dst={source_dest},readonly',
                    '--mount', f'type=bind,src={artifact},dst={artifact_dest}',
                    '--entrypoint', workload.arguments[0]]
            for key, value in self._labels(labels).items():
                args += ['--label', key + '=' + value]
            for key in workload.environment_keys:
                args += ['--env', key + '=' + self._environment[key]]
            if runtime.accelerator_devices.device_indices:
                args += ['--gpus', 'device=0']
            self._created.add(labels.digest)
            if self._ids(labels):
                return DockerCreateResultV1(Create.COLLISION)
            image_id = self._image(image)
            identifier = self._checked(
                [*args, image_id, *workload.arguments[1:]],
                before=lambda: self._check_source_before_create(labels, source_ref, source, artifact),
            ).strip()
            status, _ = self._inspect(identifier, labels, image_id)
            if status != 'created':
                raise ValueError("docker_cli_reply_invalid")
            self._owned[identifier] = (labels.digest, image_id)
            return DockerCreateResultV1(Create.CREATED, labels, identifier)
        except Exception:
            return DockerCreateResultV1(Create.INDETERMINATE)

    def _check_source_before_create(self, labels, source_ref, source, artifact):
        self._recheck_mounts(source, artifact)
        if self._source_submit_check is not None:
            if self._source_submit_check(labels=labels, source_ref=source_ref) is not None:
                raise ValueError("docker_cli_configuration_invalid")
        self._recheck_mounts(source, artifact)

    def start_once(self, container_ref, labels):
        key = (container_ref, labels.digest)
        if key in self._started or self._owned.get(container_ref, (None,))[0] != labels.digest:
            return DockerStartResultV1(Start.COLLISION)
        self._started.add(key)
        try:
            image = self._owned[container_ref][1]
            if self._inspect(container_ref, labels, image)[0] != 'created':
                return DockerStartResultV1(Start.COLLISION)
            if self._checked(['container', 'start', container_ref]).strip() != container_ref:
                raise ValueError("docker_cli_reply_invalid")
            status, _ = self._inspect(container_ref, labels, image)
            if status not in ('running', 'exited', 'dead'):
                raise ValueError("docker_cli_reply_invalid")
            return DockerStartResultV1(Start.STARTED, labels, container_ref)
        except Exception:
            return DockerStartResultV1(Start.INDETERMINATE)

    def _stop_exact(self, container_ref, labels, image, *, grace_seconds):
        """One owned stop request plus observed nonrunning state, not causal exit."""
        if type(grace_seconds) is not int or not 1 <= grace_seconds <= 10:
            raise ValueError("docker_cli_configuration_invalid")
        image_id = self._image(image)
        before, _ = self._inspect(container_ref, labels, image_id)
        if before != 'running':
            raise ValueError("docker_cli_stop_ineligible")
        def sufficient_budget():
            if math.floor(self._deadline - time.monotonic()) <= grace_seconds:
                raise ValueError("docker_cli_deadline")
        output = self._checked(
            ['container', 'stop', '--timeout', str(grace_seconds), container_ref],
            before=sufficient_budget,
        ).strip()
        if output != container_ref:
            raise ValueError("docker_cli_reply_invalid")
        after, exit_code = self._inspect(container_ref, labels, image_id)
        if after not in ('exited', 'dead'):
            raise ValueError("docker_cli_stop_unconfirmed")
        return {'container_ref': container_ref, 'labels_digest': labels.digest,
                'daemon_id': self._daemon, 'image_id': image_id, 'before': before,
                'after': after, 'exit_code': exit_code, 'grace_seconds': grace_seconds}

    def lookup(self, request):
        try:
            profile_image = None
            if self._profile is not None:
                profile, labels = self._profile, request.labels
                if (labels.provider_id != profile.provider.provider_id or
                    labels.profile_ref != profile.provider.profile_ref or
                    labels.account_ref != profile.scope.account_ref or
                    labels.namespace_ref != profile.scope.namespace_ref):
                    return DockerLookupResultV1(Lookup.INDETERMINATE)
                profile_image = self._image(profile.image)
            ids = self._ids(request.labels)
            if len(ids) > 1:
                return DockerLookupResultV1(Lookup.MULTIPLE)
            if not ids:
                return DockerLookupResultV1(Lookup.INDETERMINATE)
            expected = self._owned.get(ids[0])
            if expected is not None and expected[0] != request.labels.digest:
                return DockerLookupResultV1(Lookup.INDETERMINATE)
            image_id = expected[1] if expected is not None else profile_image
            if image_id is None or (profile_image is not None and image_id != profile_image):
                return DockerLookupResultV1(Lookup.INDETERMINATE)
            status, code = self._inspect(ids[0], request.labels, image_id)
            phase = {'created': Phase.CREATED, 'running': Phase.RUNNING,
                     'dead': Phase.FAILED}.get(status)
            if status == 'exited':
                phase = Phase.SUCCEEDED if code == 0 else Phase.FAILED
            if phase is None:
                return DockerLookupResultV1(Lookup.INDETERMINATE)
            return DockerLookupResultV1(Lookup.FOUND, request.labels, ids[0], phase)
        except Exception:
            return DockerLookupResultV1(Lookup.INDETERMINATE)
