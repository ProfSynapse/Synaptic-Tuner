"""Explicit, present-only Docker CLI ports; no durable claims or container TTL.

The caller supplies an already validated launch authority and daemon identity.
Local attempt claims last only for this session. Unknown effects are never retried.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from types import MappingProxyType

from tuner.cloud.derived_training_image import DockerLaunchAuthority
from tuner.cloud.hf_training_image_lock import CommandSpec, subprocess_runner
from .model import (
    DockerCreateDispositionV1 as Create, DockerCreateResultV1,
    DockerStartDispositionV1 as Start, DockerStartResultV1,
    DockerLookupDispositionV1 as Lookup, DockerLookupResultV1,
    DockerRunPhaseV1 as Phase,
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
                 environment, deadline: float, runner=subprocess_runner):
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
        values = dict(environment)
        if len(values) > 64:
            raise ValueError("docker_cli_configuration_invalid")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or
               type(v) is not str or '\0' in v for k, v in values.items()):
            raise ValueError("docker_cli_configuration_invalid")
        if sum(len(k.encode()) + len(v.encode()) for k, v in values.items()) > 32768:
            raise ValueError("docker_cli_configuration_invalid")
        self._environment = MappingProxyType(values)
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
            if not Path(host).is_dir() or Path(host).is_symlink():
                raise ValueError("docker_cli_configuration_invalid")
            result[ref] = (host, destination)
        return MappingProxyType(result)

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

    def _checked(self, args):
        self._identity()
        try:
            return self._invoke(args)
        finally:
            self._identity()

    @staticmethod
    def _name(labels):
        return 'synaptic-' + labels.digest

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
            if source_dest == artifact_dest or source == artifact:
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
            identifier = self._checked([*args, image_id, *workload.arguments[1:]]).strip()
            status, _ = self._inspect(identifier, labels, image_id)
            if status != 'created':
                raise ValueError("docker_cli_reply_invalid")
            self._owned[identifier] = (labels.digest, image_id)
            return DockerCreateResultV1(Create.CREATED, labels, identifier)
        except Exception:
            return DockerCreateResultV1(Create.INDETERMINATE)

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

    def lookup(self, request):
        try:
            ids = self._ids(request.labels)
            if len(ids) > 1:
                return DockerLookupResultV1(Lookup.MULTIPLE)
            if not ids:
                return DockerLookupResultV1(Lookup.INDETERMINATE)
            status, code = self._inspect(ids[0], request.labels)
            phase = {'created': Phase.CREATED, 'running': Phase.RUNNING,
                     'dead': Phase.FAILED}.get(status)
            if status == 'exited':
                phase = Phase.SUCCEEDED if code == 0 else Phase.FAILED
            if phase is None:
                return DockerLookupResultV1(Lookup.INDETERMINATE)
            return DockerLookupResultV1(Lookup.FOUND, request.labels, ids[0], phase)
        except Exception:
            return DockerLookupResultV1(Lookup.INDETERMINATE)
