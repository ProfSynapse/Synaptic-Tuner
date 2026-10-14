"""Explicit SDK transport using host-owned records and one-shot retained material."""

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from queue import Queue, Empty
from contextlib import contextmanager
import shlex
import stat
import tempfile
from threading import Thread
import time

from synaptic_tuner.api.v1.ports import StoragePartition
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.identities import EffectKind
from .model import HFJobsError, document_bytes, MAX_ARTIFACT_BYTES, MAX_ARTIFACT_TOTAL_BYTES
from .preparation import sha


def _bounded(operation, deadline):
    """Bound SDK waiting; late remote/local effects remain possible and consumed."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise HFJobsError("deadline_exceeded")
    queue = Queue(maxsize=1)

    def execute():
        try:
            if time.monotonic() >= deadline:
                queue.put((False, None))
                return
            queue.put((True, operation()))
        except BaseException:
            queue.put((False, None))

    Thread(target=execute, daemon=True, name="hf-jobs-sdk").start()
    try:
        okay, value = queue.get(timeout=remaining)
    except Empty:
        raise HFJobsError("deadline_exceeded") from None
    if not okay:
        raise HFJobsError("read_unavailable")
    if time.monotonic() >= deadline:
        raise HFJobsError("deadline_exceeded")
    return value


class HFJobsTransport:
    """No implicit account/bucket/create/pull, retry, adoption or secret persistence.

    Foundation owns submission authority. Additional create-only record claims
    prevent reusing this retained one-use material through another command.
    Unknown SDK completion stays consumed; this is not restart continuation.
    """

    def __init__(self, *, preparation, sdk, api, records, secrets, scratch_root,
                 deadline_seconds=30):
        if sdk.__version__ != "1.33.0" or type(deadline_seconds) is not int or not 1 <= deadline_seconds <= 180:
            raise HFJobsError("input_invalid")
        root = Path(scratch_root)
        info = root.lstat()
        if (root.is_symlink() or not root.is_absolute() or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise HFJobsError("input_invalid")
        self.preparation, self.selection = preparation, preparation.selection
        self.sdk, self.api, self.records, self.secrets = sdk, api, records, secrets
        self.scratch_root, self.deadline_seconds = root, deadline_seconds
        self._scratch_identity = (info.st_dev, info.st_ino)
        self.bootstrap = preparation.source_files["parent/tuner/execution/providers/hf_jobs_provider_v1/bootstrap.py"]
        members = [{"path": path.removeprefix("parent/"), "size_bytes": len(raw), "sha256": sha(raw)}
                   for path, raw in sorted(preparation.source_files.items()) if path.startswith("parent/")]
        self.parent_inventory = {"members": members}
        self._cancel_access = None

    def bind_cancel_access(self, access):
        if self._cancel_access is not None or not callable(access):
            raise HFJobsError("input_invalid")
        self._cancel_access = access

    def _root(self):
        info = self.scratch_root.lstat()
        if (self.scratch_root.is_symlink() or (info.st_dev, info.st_ino) != self._scratch_identity
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise HFJobsError("binding_mismatch")

    def _secret(self, ref):
        value = self.secrets.resolve(ref)
        if type(value) is not str or not value.strip():
            raise HFJobsError("input_invalid")
        return value

    def _key(self, kind, digest):
        return "hf-jobs/" + self.preparation.material_digest + "/" + kind + "/" + digest

    def _claim_document(self, command):
        return document_bytes({"schema_version": "synaptic-hf-attempt/v1",
            "material_digest": self.preparation.material_digest,
            "command_digest": command.digest, "kind": command.operation.effect.kind.value})

    def _claim(self, command):
        value = self._claim_document(command)
        if self.records.create(partition=StoragePartition.EFFECTS.value,
            key=self._key("claim", command.operation.effect.kind.value), canonical=value) is not True:
            raise HFJobsError("submit_indeterminate")

    def _check_claim(self, command):
        retained = self.records.read(partition=StoragePartition.EFFECTS.value,
            key=self._key("claim", command.operation.effect.kind.value))
        if retained is None or retained.canonical != self._claim_document(command):
            raise HFJobsError("binding_mismatch")

    def _retain(self, kind, command, document):
        raw = document_bytes(document)
        if self.records.create(partition=StoragePartition.EFFECTS.value,
            key=self._key(kind, command.digest), canonical=raw) is not True:
            raise HFJobsError("binding_mismatch")
        return document

    def retained(self, kind, command_digest):
        record = self.records.read(partition=StoragePartition.EFFECTS.value,
                                   key=self._key(kind, command_digest))
        if record is None:
            raise HFJobsError("read_unavailable")
        document = json.loads(record.canonical)
        if document["command_digest"] != command_digest or document["material_digest"] != self.preparation.material_digest:
            raise HFJobsError("binding_mismatch")
        return document

    def _list(self, prefix, maximum, token, deadline):
        def collect():
            result = {}
            for index, item in enumerate(self.api.list_bucket_tree(self.selection.bucket_id,
                    prefix=prefix, recursive=True, token=token), 1):
                if index > maximum * 8 + 8 or not item.path.startswith(prefix + "/"):
                    raise ValueError
                relative = item.path[len(prefix) + 1:]
                if (not relative or "\\" in relative or any(p in {"", ".", ".."} for p in relative.split("/"))):
                    raise ValueError
                if item.type == "directory":
                    continue
                if item.type != "file" or relative in result or len(result) >= maximum or type(item.size) is not int or item.size < 0:
                    raise ValueError
                result[relative] = item
            return result
        return _bounded(collect, deadline)

    @contextmanager
    def _spool(self, item, maximum, token, deadline):
        """Verify an owned 0600 leaf in a 0700 root.

        SDK leaves are normalized through the checked FD after writing inside
        the private root. Uncertain SDK scratch is retained; file permissions do not attest remote durability.
        """
        self._root()
        if type(maximum) is not int or item.size > maximum:
            raise HFJobsError("artifact_invalid")
        root = Path(tempfile.mkdtemp(prefix="hf-read-", dir=self.scratch_root))
        leaf = root / "content"
        complete, descriptor = False, None
        root_info = root.lstat()
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            _bounded(lambda: self.api.download_bucket_files(self.selection.bucket_id,
                [(item, leaf)], token=token, raise_on_missing_files=True), deadline)
            self._root()
            descriptor = os.open("content", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.scandir(parent) as entries:
                names = []
                for entry in entries:
                    names.append(entry.name)
                    if len(names) > 1:
                        raise HFJobsError("artifact_invalid")
            if names != ["content"]:
                raise HFJobsError("artifact_invalid")
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or before.st_size != item.size or before.st_uid != os.getuid()):
                raise HFJobsError("artifact_invalid")
            # Normalize only our checked owned regular leaf through the exact
            # descriptor, independent of SDK/ambient umask. The directory was
            # private throughout SDK writing.
            os.fchmod(descriptor, 0o600)
            before = os.fstat(descriptor)
            def check():
                self._root()
                if time.monotonic() >= deadline:
                    raise HFJobsError("deadline_exceeded")
                held, named = os.fstat(parent), root.lstat()
                if ((held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
                        or held.st_uid != os.getuid() or stat.S_IMODE(held.st_mode) != 0o700
                        or named.st_uid != os.getuid() or stat.S_IMODE(named.st_mode) != 0o700):
                    raise HFJobsError("artifact_invalid")
                current, named = os.fstat(descriptor), os.stat("content", dir_fd=parent, follow_symlinks=False)
                identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                    value.st_mtime_ns, value.st_ctime_ns)
                if identity(current) != identity(before) or identity(named) != identity(before):
                    raise HFJobsError("artifact_invalid")
            digest, total = hashlib.sha256(), 0
            while True:
                check()
                chunk = os.read(descriptor, min(1048576, maximum - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise HFJobsError("artifact_invalid")
                digest.update(chunk)
            check()
            if total != item.size:
                raise HFJobsError("artifact_invalid")
            os.lseek(descriptor, 0, os.SEEK_SET)
            complete = True
            yield descriptor, digest.hexdigest(), total, check
        finally:
            # A timeout/failed SDK operation may still write later. Never remove
            # or reuse its private namespace as though completion were known.
            try:
                if complete:
                    self._root()
                    named = root.lstat()
                    if ((named.st_dev, named.st_ino) != (root_info.st_dev, root_info.st_ino)
                            or named.st_uid != os.getuid() or stat.S_IMODE(named.st_mode) != 0o700):
                        raise HFJobsError("artifact_invalid")
                    held_leaf = os.fstat(descriptor)
                    named_leaf = os.stat("content", dir_fd=parent, follow_symlinks=False)
                    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                        value.st_mtime_ns, value.st_ctime_ns)
                    if identity(held_leaf) != identity(before) or identity(named_leaf) != identity(before):
                        raise HFJobsError("artifact_invalid")
                    os.unlink("content", dir_fd=parent)
                    root.rmdir()
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                os.close(parent)

    def _download(self, item, maximum, token, deadline):
        if maximum > 64 * 1024 * 1024:
            raise HFJobsError("input_invalid")
        with self._spool(item, maximum, token, deadline) as (fd, digest, size, check):
            chunks = []
            while True:
                check()
                chunk = os.read(fd, min(1048576, size + 1))
                if not chunk:
                    break
                chunks.append(chunk)
            check()
            return b"".join(chunks), digest

    def stage(self, command):
        self._validate(command, EffectKind.STAGE)
        token = self._secret(self.selection.host_token)
        self._secret(self.selection.job_token)
        self._claim(command)
        deadline = time.monotonic() + self.deadline_seconds
        try:
            if (self._list(self.selection.source_prefix, len(self.preparation.source_files) + 1, token, deadline)
                    or self._list(self.selection.run_prefix, 2, token, deadline)):
                raise HFJobsError("binding_mismatch")
            identity = self.preparation.input_identity
            with self.preparation.input_lease.take_stream() as stream:
                raw = stream.read(identity.size_bytes + 1)
                if type(raw) is not bytes or len(raw) != identity.size_bytes or sha(raw) != identity.content_digest:
                    raise HFJobsError("binding_mismatch")
                source = dict(self.preparation.source_files)
                add = [(content, self.selection.source_prefix + "/" + name)
                       for name, content in sorted(source.items())]
                input_path = "input/" + identity.content_digest + "/private-dataset.jsonl"
                add.append((raw, self.selection.run_prefix + "/" + input_path))
                _bounded(lambda: self.api.batch_bucket_files(self.selection.bucket_id, add=add, token=token), deadline)
                for prefix, expected in ((self.selection.source_prefix, source),
                                         (self.selection.run_prefix, {input_path: raw})):
                    listed = self._list(prefix, len(expected) + 1, token, deadline)
                    if set(listed) != set(expected):
                        raise HFJobsError("binding_mismatch")
                    for name, value in expected.items():
                        observed, digest = self._download(listed[name], len(value), token, deadline)
                        if observed != value or digest != sha(value):
                            raise HFJobsError("binding_mismatch")
            receipt = self._retain("stage", command, {"schema_version": "synaptic-hf-stage-receipt/v1",
                "command_digest": command.digest, "material_digest": self.preparation.material_digest,
                "stage_ref": "hf-stage-" + command.digest, "effect_id": command.operation.effect.effect_id})
            if self.records.create(partition=StoragePartition.EFFECTS.value,
                key=self._key("stage-effect", command.operation.effect.effect_id),
                canonical=document_bytes(receipt)) is not True:
                raise HFJobsError("binding_mismatch")
            return receipt
        except BaseException:
            raise HFJobsError("stage_unavailable") from None

    def _validate(self, command, kind):
        rebuilt = parse_exact_command(command.canonical_bytes)
        if (type(rebuilt) is not type(command) or rebuilt.canonical_bytes != command.canonical_bytes
                or command.operation.effect.kind is not kind
                or command.preparation != self.preparation.prepare(self.preparation.plan,
                    TrainingRunRef(
                        command.preparation.run_id, command.preparation.project_ref), self.preparation.binding)
                or command.payload != self.preparation.payload(command.preparation, kind)):
            raise HFJobsError("binding_mismatch")
        if kind is EffectKind.CANCEL:
            return self._cancel_target(command)

    def _cancel_target(self, command):
        if self._cancel_access is None:
            raise HFJobsError("binding_mismatch")
        submit = self._cancel_access(command)
        self._validate(submit, EffectKind.SUBMIT)
        receipt = self.retained("submit", submit.digest)
        target = command.to_dict()["cancellation"]
        if receipt["job_id"] != target["provider_job_ref"]:
            raise HFJobsError("binding_mismatch")
        return submit, receipt

    def _cancel_receipt(self, command, submit, job_id):
        record = self.records.read(partition=StoragePartition.EFFECTS.value,
            key=self._key("cancel", command.digest))
        if record is None:
            return None
        self._check_claim(command)
        expected = {"schema_version": "synaptic-hf-cancel-receipt/v1",
            "command_digest": command.digest, "material_digest": self.preparation.material_digest,
            "submit_command_digest": submit.digest, "job_id": job_id,
            "reason_digest": command.to_dict()["cancellation"]["reason_digest"],
            "acknowledged": True}
        if record.canonical != document_bytes(expected):
            raise HFJobsError("binding_mismatch")
        return expected

    def cancel(self, command):
        """One acknowledged request, not a terminal/physical-stop assertion.

        The absolute deadline bounds provider work, not synchronous host store
        callbacks. A late provider thread cannot write the owning-thread journal.
        """
        submit, receipt = self._validate(command, EffectKind.CANCEL)
        retained = self._cancel_receipt(command, submit, receipt["job_id"])
        if retained is not None:
            return retained
        deadline = time.monotonic() + self.deadline_seconds
        phase, _ = self.observe(submit.digest, deadline=deadline)
        if phase == "DELETED":
            raise HFJobsError("read_unavailable")
        token = self._secret(self.selection.host_token)
        self._claim(command)
        acknowledged = _bounded(lambda: self.api.cancel_job(job_id=receipt["job_id"],
            namespace=self.selection.scope.namespace_ref, token=token), deadline)
        if acknowledged is not None:
            raise HFJobsError("read_unavailable")
        value = {"schema_version": "synaptic-hf-cancel-receipt/v1",
            "command_digest": command.digest, "material_digest": self.preparation.material_digest,
            "submit_command_digest": submit.digest, "job_id": receipt["job_id"],
            "reason_digest": command.to_dict()["cancellation"]["reason_digest"],
            "acknowledged": True}
        return self._retain("cancel", command, value)

    def lookup(self, command):
        """Read retained exact receipts only; absence never permits replay."""
        kind = command.operation.effect.kind
        target = self._validate(command, kind)
        if kind is EffectKind.CANCEL:
            submit, receipt = target
            acknowledged = self._cancel_receipt(command, submit, receipt["job_id"])
            if acknowledged is None:
                return None
            # Revalidate the original provider target without another stop.
            phase, _ = self.observe(submit.digest)
            return None if phase == "DELETED" else acknowledged
        record = self.records.read(partition=StoragePartition.EFFECTS.value,
            key=self._key(kind.value, command.digest))
        if record is None:
            return None
        self._check_claim(command)
        receipt = self.retained(kind.value, command.digest)
        if kind is EffectKind.SUBMIT:
            phase, _ = self.observe(command.digest)
            return None if phase == "DELETED" else receipt
        expected = {"schema_version": "synaptic-hf-stage-receipt/v1",
            "command_digest": command.digest, "material_digest": self.preparation.material_digest,
            "stage_ref": "hf-stage-" + command.digest, "effect_id": command.operation.effect.effect_id}
        index = self.records.read(partition=StoragePartition.EFFECTS.value,
            key=self._key("stage-effect", command.operation.effect.effect_id))
        if receipt != expected or index is None or index.canonical != document_bytes(expected):
            raise HFJobsError("binding_mismatch")
        return receipt

    def job_request(self, command):
        prep = self.preparation
        material = json.loads(prep.material_bytes)
        projection = json.loads(prep.bundle.canonical_projection_bytes)
        config = json.loads(prep.bundle.canonical_workload_bytes)["configuration"]["document"]
        worker = {"material_digest": prep.material_digest,
            "workload_fingerprint": prep.bundle.workload_fingerprint,
            "source_members": [row for row in material["members"] if not row["path"].startswith("parent/")],
            "input": prep.input_identity.to_dict(), "dispatch": projection["dispatch"],
            "source": prep.rich_plan.execution_source.to_dict(),
            "runtime_inventory_sha256": sha(self.selection.inventory_bytes),
            "model": config["model"], "timeout_seconds": self.selection.timeout_seconds}
        payload = document_bytes({"schema_version": "synaptic-hf-source-bootstrap/v1",
                                 "parent_members": self.parent_inventory["members"], "worker": worker})
        program = self.bootstrap.decode() + "\ntry:\n run(__import__('base64').b64decode('" + base64.b64encode(payload).decode() + "'))\nexcept BaseException:\n print('HF_JOB_FAILED')\n sys.exit(1)\n"
        executable = prep.rich_plan.execution_source.python_executable
        volumes = [self.sdk.Volume(type="bucket", source=self.selection.bucket_id,
            path=self.selection.source_prefix, mount_path="/source", read_only=True),
            self.sdk.Volume(type="bucket", source=self.selection.bucket_id,
            path=self.selection.run_prefix, mount_path="/artifacts", read_only=False)]
        return {"image": self.selection.runtime_profile.image,
            "command": [executable, "-I", "-c", shlex.quote(program)],
            "env": {}, "flavor": self.selection.flavor, "timeout": self.selection.timeout_seconds,
            "namespace": self.selection.scope.namespace_ref,
            "labels": {"synaptic-command": command.digest, "synaptic-material": prep.material_digest,
                       "name": "synaptic-" + command.digest[:32]},
            "volumes": volumes}

    def submit(self, command):
        self._validate(command, EffectKind.SUBMIT)
        index = self.records.read(partition=StoragePartition.EFFECTS.value,
            key=self._key("stage-effect", command.stage_predecessor.stage_effect_id))
        if index is None:
            raise HFJobsError("binding_mismatch")
        stage = self.retained("stage", json.loads(index.canonical)["command_digest"])
        if stage["effect_id"] != command.stage_predecessor.stage_effect_id:
            raise HFJobsError("binding_mismatch")
        token, job_token = self._secret(self.selection.host_token), self._secret(self.selection.job_token)
        arguments = self.job_request(command)
        self._claim(command)
        try:
            job = _bounded(lambda: self.api.run_job(**arguments, token=token,
                secrets={"HF_TOKEN": job_token}), time.monotonic() + self.deadline_seconds)
            self._check_job(job, arguments)
            return self._retain("submit", command, {"schema_version": "synaptic-hf-submit-receipt/v1",
                "command_digest": command.digest, "material_digest": self.preparation.material_digest,
                "job_id": job.id, "request": {**arguments,
                    "volumes": [volume.to_dict() for volume in arguments["volumes"]]}})
        except BaseException:
            raise HFJobsError("submit_indeterminate") from None

    def _check_job(self, job, request):
        if (type(job.id) is not str or not job.id or len(job.id) > 128
                or job.docker_image != request["image"] or job.command != request["command"]
                or job.arguments not in (None, []) or job.environment != request["env"]
                or job.flavor != request["flavor"] or job.labels != request["labels"]
                or job.owner.name != self.selection.scope.namespace_ref
                or [value.to_dict() for value in job.volumes] != [
                    value.to_dict() if hasattr(value, "to_dict") else value for value in request["volumes"]]):
            raise HFJobsError("binding_mismatch")

    def observe(self, command_digest, *, deadline=None):
        receipt = self.retained("submit", command_digest)
        catalog = self.retained("command", command_digest)
        command = parse_exact_command(base64.b64decode(catalog["command_base64"], validate=True))
        self._validate(command, EffectKind.SUBMIT)
        expected = self.job_request(command)
        expected["volumes"] = [value.to_dict() for value in expected["volumes"]]
        if receipt["request"] != expected:
            raise HFJobsError("binding_mismatch")
        token = self._secret(self.selection.host_token)
        job = _bounded(lambda: self.api.inspect_job(job_id=receipt["job_id"],
            namespace=self.selection.scope.namespace_ref, token=token),
            time.monotonic() + self.deadline_seconds if deadline is None else deadline)
        self._check_job(job, receipt["request"])
        if job.id != receipt["job_id"]:
            raise HFJobsError("binding_mismatch")
        phase = str(job.status.stage.value if hasattr(job.status.stage, "value") else job.status.stage)
        if phase not in {"SCHEDULING", "RUNNING", "COMPLETED", "ERROR", "CANCELED", "DELETED"}:
            raise HFJobsError("read_unavailable")
        return phase, receipt

    def native(self, command_digest):
        phase, receipt = self.observe(command_digest)
        if phase != "COMPLETED":
            raise HFJobsError("read_unavailable")
        token = self._secret(self.selection.host_token)
        deadline = time.monotonic() + self.deadline_seconds
        listed = self._list(self.selection.run_prefix, 8, token, deadline)
        if "completion.json" not in listed:
            raise HFJobsError("read_unavailable")
        raw, _ = self._download(listed["completion.json"], 65536, token, deadline)
        def unique(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise HFJobsError("artifact_invalid")
                result[name] = value
            return result
        def reject_constant(value):
            raise HFJobsError("artifact_invalid")
        completion = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
        if (type(completion) is not dict or set(completion) != {
                "schema_version", "status", "material_digest", "workload_fingerprint", "job_id", "artifacts"}
                or completion["schema_version"] != "synaptic-hf-native-completion/v1"
                or type(completion["job_id"]) is not str or not 1 <= len(completion["job_id"]) <= 128
                or type(completion["artifacts"]) is not list or len(completion["artifacts"]) != 5):
            raise HFJobsError("artifact_invalid")
        if (completion["status"] != "completed" or completion["material_digest"] != self.preparation.material_digest
                or completion["job_id"] != receipt["job_id"]
                or completion["workload_fingerprint"] != self.preparation.bundle.workload_fingerprint):
            raise HFJobsError("artifact_invalid")
        roles, paths, total = set(), set(), 0
        for member in completion["artifacts"]:
            if (type(member) is not dict or set(member) != {"role", "path", "size_bytes", "sha256"}
                    or type(member["role"]) is not str or type(member["path"]) is not str
                    or type(member["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", member["sha256"]) is None):
                raise HFJobsError("artifact_invalid")
            role, path, size = member["role"], member["path"], member["size_bytes"]
            if (role in roles or path in paths or not path.startswith("artifacts/")
                    or len(path.split("/")) != 2 or type(size) is not int
                    or not 0 < size <= MAX_ARTIFACT_BYTES or path not in listed or listed[path].size != size):
                raise HFJobsError("artifact_invalid")
            roles.add(role)
            paths.add(path)
            total += size
        from .model import EXACT_ROLES
        identity = self.preparation.input_identity
        input_path = "input/" + identity.content_digest + "/private-dataset.jsonl"
        if (roles != EXACT_ROLES or total > MAX_ARTIFACT_TOTAL_BYTES
                or set(listed) != paths | {"completion.json", input_path}):
            raise HFJobsError("artifact_invalid")
        return completion, listed

    def iter_artifact(self, command_digest, member):
        completion, listed = self.native(command_digest)
        if member not in completion["artifacts"]:
            raise HFJobsError("artifact_invalid")
        with self._spool(listed[member["path"]], member["size_bytes"],
                self._secret(self.selection.host_token), time.monotonic() + self.deadline_seconds) as (fd, digest, size, check):
            if digest != member["sha256"]:
                raise HFJobsError("artifact_invalid")
            emitted = 0
            while True:
                check()
                chunk = os.read(fd, min(1048576, size - emitted + 1))
                if not chunk:
                    break
                emitted += len(chunk)
                if emitted > size:
                    raise HFJobsError("artifact_invalid")
                yield chunk
            check()
            if emitted != size:
                raise HFJobsError("artifact_invalid")
