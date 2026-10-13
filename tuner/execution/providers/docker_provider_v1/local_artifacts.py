"""Host-local Docker artifact reads, not container isolation or recovery.

The host supplies the admitted plan, original labels, role paths and signers.
Retained descriptors detect observed replacements; they do not freeze files or
make a filesystem transaction with daemon observations.
"""
from __future__ import annotations

from contextlib import ExitStack
from collections.abc import Mapping
import hashlib
import math
import os
import re
from pathlib import Path
import stat
import time
from types import MappingProxyType

from synaptic_tuner.api.v1.results import VerifiedArtifact
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from .cli_transport import DockerCLISessionV1
from .model import (
    PreparedDockerPlanV1, DockerLabelsV1, DockerArtifactEntryV1,
    DockerArtifactInventoryV1, DockerArtifactInventoryRequestV1,
    AuthenticatedDockerArtifactInventoryV1, DockerArtifactReadRequestV1,
    DockerArtifactChunkV1, DockerArtifactEOFV1, DockerLookupRequestV1,
    DockerLookupPurposeV1, DockerLookupDispositionV1, DockerRunPhaseV1,
    DockerProviderError, DockerDiagnosticCodeV1,
)

_DIR = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
_FILE = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_CLOEXEC', 0)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_digest(info):
    return domain_digest('synaptic-docker-local-file-identity/v1', canonical_bytes(list(_identity(info))))


class DockerLocalArtifactsV1:
    """Linux descriptor-based read port; logs intentionally remain unavailable.

    ``role_paths`` selects only declared artifacts, not every directory member.
    Per-file/total limits come from the exact prepared profile. ``deadline`` is
    one absolute cooperative host deadline, including hashing and streaming.
    Signers receive existing inventory/EOF values and return authenticated types.
    """
    def __init__(self, *, plan, labels, container_ref, control, root,
                 role_paths, deadline, sign_inventory, sign_eof):
        if type(plan) is not PreparedDockerPlanV1 or type(labels) is not DockerLabelsV1 or type(control) is not DockerCLISessionV1:
            raise ValueError('docker_artifact_configuration_invalid')
        if labels.effect_kind != 'submit' or labels.adapter_descriptor_digest != plan.profile.adapter_descriptor.digest:
            raise ValueError('docker_artifact_configuration_invalid')
        if (labels.provider_id, labels.profile_ref, labels.account_ref, labels.namespace_ref,
            labels.project_ref, labels.run_id, labels.plan_fingerprint, labels.preparation_digest) != (
            plan.profile.provider.provider_id, plan.profile.provider.profile_ref,
            plan.profile.scope.account_ref, plan.profile.scope.namespace_ref,
            plan.project_ref, plan.run_id, plan.plan_fingerprint, plan.preparation_digest):
            raise ValueError('docker_artifact_configuration_invalid')
        if (type(container_ref) is not str or re.fullmatch(r'[0-9a-f]{64}', container_ref) is None or
            not callable(sign_inventory) or not callable(sign_eof) or
            not isinstance(role_paths, Mapping) or len(role_paths) > len(plan.profile.artifacts.roles)):
            raise ValueError('docker_artifact_configuration_invalid')
        root = Path(root)
        paths = dict(role_paths)
        if not root.is_absolute() or '..' in root.parts or tuple(sorted(paths)) != plan.profile.artifacts.roles:
            raise ValueError('docker_artifact_configuration_invalid')
        if len(set(paths.values())) != len(paths):
            raise ValueError('docker_artifact_configuration_invalid')
        for path in paths.values():
            if type(path) is not str or len(path.encode()) > 4096 or '\\' in path or '\0' in path or path.startswith('/') or any(p in ('', '.', '..') for p in path.split('/')):
                raise ValueError('docker_artifact_configuration_invalid')
        if isinstance(deadline, bool) or not math.isfinite(deadline):
            raise ValueError('docker_artifact_configuration_invalid')
        if os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW') or os.open not in os.supports_dir_fd:
            raise ValueError('docker_artifact_platform_unavailable')
        self._plan = PreparedDockerPlanV1(**{n: getattr(plan, n) for n in plan.__dataclass_fields__})
        self._labels = DockerLabelsV1(**labels.to_dict())
        self._container, self._control = container_ref, control
        self._root, self._paths = root, MappingProxyType(paths)
        self._deadline, self._sign_inventory, self._sign_eof = deadline, sign_inventory, sign_eof
        self._stack, self._chain, self._inventory = ExitStack(), [], None
        try:
            fd = os.open(root.anchor, _DIR)
            self._stack.callback(os.close, fd)
            for part in root.parts[1:]:
                child = os.open(part, _DIR, dir_fd=fd)
                self._stack.callback(os.close, child)
                self._chain.append((fd, part, child))
                fd = child
            self._root_fd = fd
            self._check()
        except BaseException:
            self._stack.close()
            raise ValueError('docker_artifact_root_invalid') from None

    def close(self):
        self._stack.close()
        self._root_fd = None
        self._inventory = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _check(self):
        if self._root_fd is None or time.monotonic() >= self._deadline:
            raise ValueError('docker_artifact_unavailable')
        self._recheck(self._chain)

    @staticmethod
    def _recheck(chain):
        for parent, name, held in chain:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            original = os.fstat(held)
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
                raise ValueError('docker_artifact_root_changed')

    def lookup(self, request):
        return self._control.lookup(request)

    def logs(self, request):
        raise DockerProviderError(DockerDiagnosticCodeV1.MALFORMED_EVIDENCE)

    def _terminal(self, labels, generation):
        self._check()
        if labels != self._labels:
            raise ValueError('docker_artifact_binding_invalid')
        result = self._control.lookup(DockerLookupRequestV1(labels, DockerLookupPurposeV1.OBSERVE, generation))
        if (result.disposition is not DockerLookupDispositionV1.FOUND or
            result.labels != labels or result.container_ref != self._container or
            result.phase not in (DockerRunPhaseV1.SUCCEEDED, DockerRunPhaseV1.FAILED, DockerRunPhaseV1.CANCELLED)):
            raise ValueError('docker_artifact_not_terminal')
        self._check()

    def _open(self, path, stack, maximum):
        self._check()
        parent, chain = self._root_fd, []
        for part in path.split('/')[:-1]:
            child = os.open(part, _DIR, dir_fd=parent)
            stack.callback(os.close, child)
            chain.append((parent, part, child))
            parent = child
        name = path.split('/')[-1]
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        fd = os.open(name, _FILE, dir_fd=parent)
        stack.callback(os.close, fd)
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
            _identity(before) != _identity(opened) or not 0 <= opened.st_size <= maximum):
            raise ValueError('docker_artifact_file_invalid')
        return fd, parent, name, opened, chain

    def _unchanged(self, fd, parent, name, before, chain):
        self._check()
        self._recheck(chain)
        after, named = os.fstat(fd), os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or after.st_nlink != 1 or
            _identity(before) != _identity(after) or _identity(before) != _identity(named)):
            raise ValueError('docker_artifact_file_changed')

    def artifact_inventory(self, request):
        try:
            if type(request) is not DockerArtifactInventoryRequestV1:
                raise ValueError
            plan = self._plan
            if (request.profile_digest, request.prepared_plan_digest, request.artifact_contract_digest,
                request.artifact_root_ref) != (plan.profile.profile_digest, plan.digest,
                plan.profile.artifacts.digest, plan.profile.roots.artifact_ref):
                raise ValueError
            self._terminal(request.labels, request.generation)
            entries, total = [], 0
            for role, path in sorted(self._paths.items()):
                with ExitStack() as stack:
                    fd, parent, name, before, chain = self._open(path, stack, plan.profile.artifacts.maximum_artifact_bytes)
                    if total + before.st_size > plan.profile.artifacts.maximum_total_bytes:
                        raise ValueError
                    hashed, size = hashlib.sha256(), 0
                    while True:
                        self._check()
                        chunk = os.read(fd, min(1048576, plan.profile.artifacts.maximum_artifact_bytes - size + 1))
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > plan.profile.artifacts.maximum_artifact_bytes:
                            raise ValueError
                        hashed.update(chunk)
                    digest = hashed.hexdigest()
                    self._unchanged(fd, parent, name, before, chain)
                    total += size
                    if total > plan.profile.artifacts.maximum_total_bytes:
                        raise ValueError
                    entries.append(DockerArtifactEntryV1(VerifiedArtifact(role, digest, size), path, _file_digest(before)))
            self._terminal(request.labels, request.generation)
            content = DockerArtifactInventoryV1(request.labels, request.digest, request.generation,
                request.profile_digest, request.prepared_plan_digest, request.artifact_contract_digest,
                request.artifact_root_ref, tuple(entries), domain_digest('synaptic-docker-local-artifacts/v1',
                    canonical_bytes({'labels_digest': request.labels.digest, 'paths': dict(self._paths),
                                     'container_ref': self._container})))
            signed = self._sign_inventory(content)
            self._check()
            if type(signed) is not AuthenticatedDockerArtifactInventoryV1 or signed.content != content:
                raise ValueError
            self._inventory = content
            return signed
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.MALFORMED_EVIDENCE) from None

    def iter_artifact_events(self, request):
        try:
            content = self._inventory
            if type(request) is not DockerArtifactReadRequestV1 or content is None:
                raise ValueError
            entries = [e for e in content.entries if e.descriptor.role == request.role]
            if len(entries) != 1:
                raise ValueError
            entry = entries[0]
            if (request.labels, request.inventory_digest, request.relative_path, request.expected_size,
                request.expected_sha256, request.file_identity_digest, request.generation,
                request.profile_digest, request.prepared_plan_digest, request.artifact_contract_digest,
                request.artifact_root_ref) != (content.labels, content.content_digest, entry.relative_path,
                entry.descriptor.size_bytes, entry.descriptor.sha256, entry.file_identity_digest,
                content.generation, content.profile_digest, content.prepared_plan_digest,
                content.artifact_contract_digest, content.artifact_root_ref):
                raise ValueError
            self._terminal(request.labels, request.generation)
            with ExitStack() as stack:
                fd, parent, name, before, chain = self._open(request.relative_path, stack, request.maximum_bytes)
                if _file_digest(before) != request.file_identity_digest:
                    raise ValueError
                digest, total, sequence = hashlib.sha256(), 0, 0
                while True:
                    self._unchanged(fd, parent, name, before, chain)
                    chunk = os.read(fd, min(1048576, request.expected_size - total + 1))
                    if not chunk:
                        break
                    offset = total
                    total += len(chunk)
                    if total > request.expected_size or total > request.maximum_bytes:
                        raise ValueError
                    digest.update(chunk)
                    self._unchanged(fd, parent, name, before, chain)
                    yield DockerArtifactChunkV1(request.digest, sequence, offset, chunk)
                    sequence += 1
                self._unchanged(fd, parent, name, before, chain)
                self._terminal(request.labels, request.generation)
                if total != request.expected_size or digest.hexdigest() != request.expected_sha256:
                    raise ValueError
                eof = DockerArtifactEOFV1(request.digest, sequence, total, digest.hexdigest(),
                    request.file_identity_digest, content.evidence_digest, 'unsigned', 'unsigned', '0' * 64)
                signed = self._sign_eof(eof)
                self._unchanged(fd, parent, name, before, chain)
                if type(signed) is not DockerArtifactEOFV1 or any(getattr(signed, f) != getattr(eof, f)
                    for f in ('stream_digest', 'next_sequence', 'total_bytes', 'sha256', 'file_identity_digest', 'evidence_digest')):
                    raise ValueError
                yield signed
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.MALFORMED_EVIDENCE) from None
