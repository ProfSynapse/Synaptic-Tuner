"""Create-only local staging of the existing offline SFT worker and input.

This is process-local custody, not restart recovery, model preparation or atomic
custody of paths used later by Docker. Only the source mount is read-only to the
container; prepared input in writable state remains independently checked.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import time
from types import MappingProxyType

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.training.contracts import (
    TrainingPlan as RichTrainingPlan, RetainedTrainingInputStreamLease,
    execution_material_digest,
)
from tuner.runtime.dispatch import (
    WorkerBundleMaterializationV1, WorkerControlLocationV1,
    CanonicalWorkloadFileLocationV1, build_source_worker_invocation,
    materialize_worker_bundle,
)
from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from .local_artifacts import DockerLocalArtifactsV1, _DIR, _FILE, _identity
from .preparation import DockerTrainingPreparationBridgeV1
from .model import (
    PreparedDockerPlanV1, DockerSourceSealRequestV1, DockerSourceSealContentV1,
    AuthenticatedDockerSourceSealV1, DockerSourceSealLookupRequestV1,
    DockerSourceSealLookupResultV1, DockerLookupDispositionV1, DockerLabelsV1,
    DockerProviderError, DockerDiagnosticCodeV1,
)


class _Root:
    def __init__(self, path):
        self.path, self.stack, self.chain, self.owned_dirs = Path(path), ExitStack(), [], {}
        if not self.path.is_absolute() or '..' in self.path.parts:
            raise ValueError
        try:
            fd = os.open(self.path.anchor, _DIR)
            self.stack.callback(os.close, fd)
            for part in self.path.parts[1:]:
                child = os.open(part, _DIR, dir_fd=fd)
                self.stack.callback(os.close, child)
                self.chain.append((fd, part, child))
                fd = child
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ValueError
            self.fd = fd
            self.check()
        except BaseException:
            self.stack.close()
            raise

    def check(self):
        DockerLocalArtifactsV1._recheck(self.chain)
        info = os.fstat(self.fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError

    def parent(self, relative, stack, *, create=False, fresh_prefix=None):
        parent, chain = self.fd, []
        parts = PurePosixPath(relative).parts[:-1]
        for index, name in enumerate(parts):
            prefix = '/'.join(parts[:index + 1])
            created = False
            if create:
                try:
                    os.mkdir(name, 0o700, dir_fd=parent)
                    created = True
                    os.fsync(parent)
                except FileExistsError:
                    if prefix == fresh_prefix and prefix not in self.owned_dirs:
                        raise ValueError
            child = os.open(name, _DIR, dir_fd=parent)
            stack.callback(os.close, child)
            info = os.fstat(child)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ValueError
            identity = (info.st_dev, info.st_ino)
            if created:
                self.owned_dirs[prefix] = identity
            elif prefix in self.owned_dirs and self.owned_dirs[prefix] != identity:
                raise ValueError
            chain.append((parent, name, child))
            parent = child
        return parent, PurePosixPath(relative).name, chain


class DockerLocalSourceSealsV1:
    """Existing source port with an explicit submit-check callback.

    RichTrainingPlan is tuner.training.contracts.TrainingPlan, not the public
    digest-only planning.TrainingPlan. All bytes are bound before publication;
    the lease is consumed once and is never reconstructed or replayed.
    """
    def __init__(self, *, training_plan, prepared_plan, bundle, closure_members,
                 prepared_lease, source_root, source_container_root,
                 artifact_root, artifact_container_root, deadline, sign_seal,
                 coordinator_plan=None, coordinator_material=None, recipes=None):
        if (type(training_plan) is not RichTrainingPlan or
            type(prepared_plan) is not PreparedDockerPlanV1 or
            type(bundle) is not WorkerBundleMaterializationV1 or
            type(prepared_lease) is not RetainedTrainingInputStreamLease or
            not callable(sign_seal) or isinstance(deadline, bool) or not math.isfinite(deadline)):
            raise ValueError('docker_source_configuration_invalid')
        if os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW') or os.open not in os.supports_dir_fd:
            raise ValueError('docker_source_platform_unavailable')
        source_mount, artifact_mount = PurePosixPath(source_container_root), PurePosixPath(artifact_container_root)
        if any(not p.is_absolute() or '..' in p.parts for p in (source_mount, artifact_mount)) or source_mount in (artifact_mount, *artifact_mount.parents) or artifact_mount in source_mount.parents:
            raise ValueError('docker_source_mount_invalid')
        run = TrainingRunRef(prepared_plan.run_id, prepared_plan.project_ref)
        if prepared_plan.source_digest != execution_material_digest(training_plan.execution_source):
            raise ValueError('docker_source_identity_invalid')
        bridge = DockerTrainingPreparationBridgeV1(prepared_plan.profile)
        if (coordinator_plan is None) != (coordinator_material is None):
            raise ValueError('docker_source_plan_invalid')
        if coordinator_plan is None:
            if recipes is not None:
                raise ValueError('docker_source_plan_invalid')
            prep = bridge.prepare(training_plan, run, prepared_plan.source_digest)
            expected_plan = bridge.prepared(preparation=prep, plan=training_plan,
                run=run, source_digest=prepared_plan.source_digest)
        else:
            prep, expected_plan = bridge.coordinator_prepared(plan=training_plan,
                material=coordinator_material, coordinator_plan=coordinator_plan,
                run=run, recipes=recipes)
        if expected_plan != prepared_plan:
            raise ValueError('docker_source_plan_invalid')
        control = WorkerControlLocationV1(PurePosixPath('/source/control'))
        expected = materialize_worker_bundle(build_source_worker_invocation(
            training_plan, control, CanonicalWorkloadFileLocationV1(control.control_root)))
        if any(getattr(bundle, n) != getattr(expected, n) for n in expected.__dataclass_fields__):
            raise ValueError('docker_source_bundle_invalid')
        document = json.loads(expected.canonical_workload_bytes)
        if document['method'] != 'sft':
            raise ValueError('docker_source_method_unavailable')
        identity = PreparedTrainingInputIdentity.from_dict(document['configuration']['document']['dataset'])
        if prepared_lease.identity != identity:
            raise ValueError('docker_source_input_invalid')
        roots = document['execution_source']['runtime']['roots']
        manifest = parse_offline_sft_worker_manifest(expected.closure_manifest_bytes,
            source_ref=expected.closure_manifest_source,
            manifest_path=Path(expected.closure_manifest_runtime_path), expected_digest=expected.closure_digest)
        if not isinstance(closure_members, Mapping) or len(closure_members) != len(manifest.closure.members):
            raise ValueError('docker_source_members_invalid')
        files = {}
        def place(runtime_path):
            path = PurePosixPath(runtime_path)
            if path.is_relative_to(source_mount):
                kind, relative = 'source', path.relative_to(source_mount)
            elif path.is_relative_to(artifact_mount):
                kind, relative = 'artifact', path.relative_to(artifact_mount)
            else:
                raise ValueError('docker_source_path_unmapped')
            if not relative.parts or any(p in ('', '.', '..') for p in relative.parts):
                raise ValueError('docker_source_path_invalid')
            return kind, relative.as_posix()
        engine = PurePosixPath(roots['engine'])
        for member in manifest.closure.members:
            raw = closure_members.get(member.path.as_posix())
            if type(raw) is not bytes or len(raw) != member.size_bytes or hashlib.sha256(raw).hexdigest() != member.sha256:
                raise ValueError('docker_source_members_invalid')
            key = place(engine / member.path)
            if key[0] != 'source' or key in files:
                raise ValueError('docker_source_path_invalid')
            files[key] = raw
        for path, raw in ((expected.closure_manifest_runtime_path, expected.closure_manifest_bytes),
                          (control.control_root / 'workload.json', expected.canonical_workload_bytes)):
            key = place(path)
            if key[0] != 'source' or key in files:
                raise ValueError('docker_source_path_invalid')
            files[key] = raw
        # This exact SFT locator is consumed by Trainers/sft/runtime_v1.py;
        # there is no generic public prepared-input filename convention.
        input_key = place(PurePosixPath(roots['state']) / 'prepared-inputs' /
                          identity.content_digest / 'private-dataset.jsonl')
        if input_key in files:
            raise ValueError('docker_source_path_invalid')
        self._plan, self._files = prepared_plan, MappingProxyType(files)
        self._identity, self._lease, self._input_key = identity, prepared_lease, input_key
        self._deadline, self._sign = deadline, sign_seal
        self._attempted, self._seal, self._request, self._retained, self._held = False, None, None, {}, {}
        self._held_parents = {}
        self._roots = {}
        try:
            self._roots['source'] = _Root(source_root)
            self._roots['artifact'] = _Root(artifact_root)
            first, second = (r.path for r in self._roots.values())
            if first in (second, *second.parents) or second in first.parents:
                raise ValueError
            self._check()
        except BaseException:
            self.close()
            raise ValueError('docker_source_root_invalid') from None

    def close(self):
        for root in self._roots.values():
            root.stack.close()
        self._roots.clear()
        self._lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _check(self):
        if len(self._roots) != 2 or time.monotonic() >= self._deadline:
            raise ValueError
        for root in self._roots.values():
            root.check()

    def _hash(self, fd, maximum):
        digest, size = hashlib.sha256(), 0
        while True:
            self._check()
            chunk = os.read(fd, min(1048576, maximum - size + 1))
            if not chunk:
                return size, digest.hexdigest()
            size += len(chunk)
            if size > maximum:
                raise ValueError
            digest.update(chunk)

    def _verify(self, key, size, digest, *, retain=False):
        self._check()
        root = self._roots[key[0]]
        if not retain:
            DockerLocalArtifactsV1._recheck(self._held_parents.get(key, ()))
        with ExitStack() as stack:
            parent, name, chain = root.parent(key[1], stack)
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            fd = os.open(name, _FILE, dir_fd=parent)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or
                _identity(before) != _identity(info)):
                raise ValueError
            if not retain and (key not in self._held or _identity(os.fstat(self._held[key])) != _identity(info)):
                raise ValueError
            if self._hash(fd, size) != (size, digest):
                raise ValueError
            DockerLocalArtifactsV1._recheck(chain)
            after = os.stat(name, dir_fd=parent, follow_symlinks=False)
            final = os.fstat(fd)
            if (info.st_nlink != 1 or after.st_nlink != 1 or
                final.st_uid != os.getuid() or after.st_uid != os.getuid() or
                stat.S_IMODE(final.st_mode) != 0o600 or stat.S_IMODE(after.st_mode) != 0o600 or
                _identity(info) != _identity(final) or _identity(info) != _identity(after)):
                raise ValueError
            identity = (_identity(info), tuple((n, os.fstat(h).st_dev, os.fstat(h).st_ino) for _, n, h in chain))
            if retain:
                held = os.dup(fd)
                root.stack.callback(os.close, held)
                self._held[key] = held
                parents = []
                for parent_fd, part, child_fd in chain:
                    saved_parent = os.dup(parent_fd)
                    root.stack.callback(os.close, saved_parent)
                    saved_child = os.dup(child_fd)
                    root.stack.callback(os.close, saved_child)
                    parents.append((saved_parent, part, saved_child))
                self._held_parents[key] = tuple(parents)
                self._retained[key] = (size, digest, identity)
            elif self._retained.get(key) != (size, digest, identity):
                raise ValueError
        self._check()

    def _publish(self, key, stream, size, digest):
        self._check()
        root = self._roots[key[0]]
        with ExitStack() as stack:
            parent, name, chain = root.parent(key[1], stack, create=True,
                fresh_prefix=PurePosixPath(key[1]).parent.as_posix() if key == self._input_key else None)
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            stack.callback(os.close, fd)
            os.fchmod(fd, 0o600)
            remaining = size
            while remaining:
                self._check()
                requested = min(1048576, remaining)
                raw = stream.read(requested)
                if type(raw) is not bytes or not raw or len(raw) > requested:
                    raise ValueError
                view = memoryview(raw)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise ValueError
                    view = view[written:]
                remaining -= len(raw)
            if stream.read(1) != b'':
                raise ValueError
            os.fsync(fd)
            os.fsync(parent)
            DockerLocalArtifactsV1._recheck(chain)
        self._verify(key, size, digest, retain=True)

    def _source_inventory(self):
        expected = {p for kind, p in (*self._files, self._input_key) if kind == 'source'}
        allowed_dirs = {str(parent) for p in expected for parent in PurePosixPath(p).parents if str(parent) != '.'}
        observed = set()
        def walk(fd, prefix=''):
            self._check()
            with os.scandir(fd) as rows:
                count = 0
                for row in rows:
                    count += 1
                    if count > len(expected) + len(allowed_dirs):
                        raise ValueError
                    path = prefix + row.name
                    info = row.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        if path not in allowed_dirs:
                            raise ValueError
                        child = os.open(row.name, _DIR, dir_fd=fd)
                        try:
                            walk(child, path + '/')
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(info.st_mode) and path in expected:
                        observed.add(path)
                    else:
                        raise ValueError
        walk(self._roots['source'].fd)
        if observed != expected:
            raise ValueError

    def _revalidate(self):
        self._check()
        self._source_inventory()
        for key, (size, digest, _) in self._retained.items():
            self._verify(key, size, digest)
        if len(self._retained) != len(self._files) + 1:
            raise ValueError

    def seal_read_only(self, request):
        if self._attempted:
            raise DockerProviderError(DockerDiagnosticCodeV1.SOURCE_UNSEALED)
        try:
            if type(request) is not DockerSourceSealRequestV1 or request.identity.plan != self._plan or request.source_ref != self._plan.profile.roots.source_ref or request.source_digest != self._plan.source_digest:
                raise ValueError
            self._check()
            self._attempted, self._request = True, request
            with os.scandir(self._roots['source'].fd) as rows:
                if next(rows, None) is not None:
                    raise ValueError
            # The derived prepared-input slot is never adopted, even if empty.
            input_root = self._roots[self._input_key[0]]
            prefix = PurePosixPath(self._input_key[1]).parent
            with ExitStack() as stack:
                parent = input_root.fd
                for index, part in enumerate(prefix.parts):
                    try:
                        child = os.open(part, _DIR, dir_fd=parent)
                    except FileNotFoundError:
                        break
                    stack.callback(os.close, child)
                    if index == len(prefix.parts) - 1:
                        raise ValueError
                    parent = child
            # Take once even on failure; no destination writes before exact spool proof.
            stream = self._lease.take_stream()
            try:
                with tempfile.TemporaryFile(mode='w+b') as spool:
                    digest, remaining = hashlib.sha256(), self._identity.size_bytes
                    while remaining:
                        self._check()
                        requested = min(1048576, remaining)
                        raw = stream.read(requested)
                        if type(raw) is not bytes or not raw or len(raw) > requested:
                            raise ValueError
                        if spool.write(raw) != len(raw):
                            raise ValueError
                        digest.update(raw)
                        remaining -= len(raw)
                    if stream.read(1) != b'' or digest.hexdigest() != self._identity.content_digest:
                        raise ValueError
                    for key, raw in self._files.items():
                        import io
                        self._publish(key, io.BytesIO(raw), len(raw), hashlib.sha256(raw).hexdigest())
                    spool.seek(0)
                    self._publish(self._input_key, spool, self._identity.size_bytes, self._identity.content_digest)
            finally:
                self._lease.close()
            self._revalidate()
            evidence = domain_digest('synaptic-docker-local-source-evidence/v1', canonical_bytes({
                'plan_digest': self._plan.digest, 'members': [
                    [kind, path, size, digest] for (kind, path), (size, digest, _) in sorted(self._retained.items())]}))
            content = DockerSourceSealContentV1(request.digest, request.identity.digest,
                request.source_ref, request.source_digest, True, 'stage-' + request.digest, evidence)
            signed = self._sign(content)
            self._revalidate()
            if type(signed) is not AuthenticatedDockerSourceSealV1 or signed.content != content:
                raise ValueError
            self._seal = signed
            return signed
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.SOURCE_UNSEALED) from None

    def lookup(self, request):
        try:
            if type(request) is not DockerSourceSealLookupRequestV1 or request.source_request != self._request or self._seal is None:
                raise ValueError
            self._revalidate()
            return DockerSourceSealLookupResultV1(DockerLookupDispositionV1.FOUND, seal=self._seal)
        except Exception:
            return DockerSourceSealLookupResultV1(DockerLookupDispositionV1.INDETERMINATE)

    def revalidate_for_submit(self, *, labels, source_ref):
        try:
            p = self._plan
            if type(labels) is not DockerLabelsV1 or labels.effect_kind != 'submit' or self._seal is None or source_ref != p.profile.roots.source_ref:
                raise ValueError
            if (labels.provider_id, labels.profile_ref, labels.account_ref, labels.namespace_ref, labels.project_ref, labels.run_id, labels.plan_fingerprint, labels.preparation_digest, labels.adapter_descriptor_digest) != (p.profile.provider.provider_id, p.profile.provider.profile_ref, p.profile.scope.account_ref, p.profile.scope.namespace_ref, p.project_ref, p.run_id, p.plan_fingerprint, p.preparation_digest, p.profile.adapter_descriptor.digest):
                raise ValueError
            self._revalidate()
        except Exception:
            raise DockerProviderError(DockerDiagnosticCodeV1.SOURCE_UNSEALED) from None
