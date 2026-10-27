"""Host-owned, one-attempt native MLX consumer for the public v1 facades.

No install, download, implicit admission, retry, or capability promotion lives
here. ``inspect`` is read-only. ``start`` needs an external root signature over
the exact admission candidate and consumes one fresh private state directory.
"""

from __future__ import annotations

from dataclasses import asdict
from contextlib import ExitStack
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys

ENGINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounded_command import BoundedCommandError, run_bounded_command
from launch import Clock, NamedSecrets, _private, download

from synaptic_tuner.api.v1.planning import ResolvedTrainingRequest
from synaptic_tuner.api.v1.reference import ReferenceRequestPortsV1, compose_reference_authority, compose_reference_host
from synaptic_tuner.api.v1.reference.provider_family import ReferenceHostPortsV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement, TrainingRequest
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.native_mlx import (
    NativeArtifactBudget, NativeCommandCatalogV1, NativeCommandMaterialV1,
    NativeEvidenceAuthority, NativeFile, NativeInvocation, NativeMLXProfile,
    NativeMaterialAuthority, NativeModelSnapshot, NativePlanningV1,
    NativeProcessLimits, NativeRestrictedAdmissionV1, compile_native_sft, native_mlx_family,
    NativeMLXGRPOProfile, compile_native_grpo,
    NativeLifecycleGrantV1,
    native_lifecycle_session,
)
from tuner.dataset_prep.training_input_source import LocalPreparedTrainingInputSource
from tuner.dataset_prep.conversation_messages import (
    MAX_BYTES as CONVERSATION_MAX_BYTES, MAX_DOCUMENT_BYTES as CONVERSATION_MAX_MANIFEST_BYTES,
    ROW_SCHEMA as CONVERSATION_ROW_SCHEMA, verify_bytes as verify_conversation_bytes,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.runtime.native_files import NativeTree, file_identity
from tuner.runtime.native_mlx_input import read_retained_native_source
from tuner.runtime.native_mlx_material import canonical, native_runtime_spec, strict_json, verify_native_material
from tuner.runtime.packaged_worker_closure import stable_read
from tuner.training.coordinator_material import _domain, _runtime
from tuner.training.modal_host_authority import HMACAuthenticator
from tuner.training.contracts import RetainedTrainingInputStreamLease


class NativePreparedTrainingInputSource(LocalPreparedTrainingInputSource):
    """Host-only POSIX conversation read; legacy publication readers stay unchanged.

    Retain the directory through bounded reads and semantic verification, then
    recheck both named members before leasing the verified immutable byte copy.
    No publication, normalization, model loading or execution authority is added.
    """

    __slots__ = ()

    def open_lease(self):
        if self.identity.format != CONVERSATION_ROW_SCHEMA:
            return super().open_lease()
        with NativeTree(self._path, private=True) as tree:
            def inventory():
                tree.check()
                if stat.S_IMODE(os.fstat(tree.fd).st_mode) != 0o700:
                    raise ValueError("native conversation publication must be private")
                members = {}
                with os.scandir(tree.fd) as entries:
                    for entry in entries:
                        if entry.name not in {"manifest.json", "dataset.jsonl"}:
                            raise ValueError("native conversation inventory differs")
                        info = os.stat(entry.name, dir_fd=tree.fd, follow_symlinks=False)
                        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                                or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
                            raise ValueError("native conversation member custody differs")
                        members[entry.name] = file_identity(info)
                if set(members) != {"manifest.json", "dataset.jsonl"}:
                    raise ValueError("native conversation inventory differs")
                return members

            before = inventory()
            manifest = tree.read("manifest.json", maximum=CONVERSATION_MAX_MANIFEST_BYTES)
            content = tree.read("dataset.jsonl", maximum=CONVERSATION_MAX_BYTES)
            verified = verify_conversation_bytes(self._path, manifest, content)
            if inventory() != before:
                raise ValueError("native conversation publication changed during verification")
            tree.check()
        observed = verified.semantic_identity
        current = PreparedTrainingInputIdentity(
            ref=f"prepared://sha256/{observed.dataset_digest}",
            revision=observed.dataset_digest, content_digest=observed.dataset_sha256,
            size_bytes=observed.dataset_bytes, format=CONVERSATION_ROW_SCHEMA,
        )
        if current != self.identity:
            raise ValueError("prepared publication changed before staging")
        return RetainedTrainingInputStreamLease(current, BytesIO(content))


def _read(path, maximum):
    value = Path(path)
    if not value.is_absolute():
        raise ValueError("native input path must be absolute")
    return stable_read(value, maximum)


def _document(path, maximum=2 << 20):
    raw = _read(path, maximum)
    value = strict_json(raw, maximum=maximum)
    if canonical(value) != raw:
        raise ValueError("native host input must be canonical JSON")
    return value


def _create_private_dir(path):
    path = Path(path)
    if not path.is_absolute() or str(path) != os.path.normpath(str(path)):
        raise ValueError("native private directory path is invalid")
    with NativeTree(path.parent, private=True) as parent:
        os.mkdir(path.name, mode=0o700, dir_fd=parent.fd)
        parent.check()
    _private(path)


def _write_owned(root, name, raw, maximum):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum or "/" in name:
        raise ValueError("native retained input is invalid")
    with NativeTree(root, private=True) as tree:
        tree.check()
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=tree.fd)
        with os.fdopen(fd, "wb") as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
        tree.check()
        if tree.read(name, maximum=maximum) != raw:
            raise ValueError("native retained input readback differs")


def _command(argv, *, cwd=None, maximum=4096):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C",
           "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"}
    status, stdout = run_bounded_command(argv, cwd=cwd, environment=env,
                                         maximum=maximum, timeout=15)
    if status != 0:
        raise ValueError("native host command refused")
    return stdout.decode("utf-8").strip()


def source_record(config, *, check_remote=True):
    """Observe the real clean checkout and exact pushed ref, never a supplied SHA alone."""
    root = Path(config["engine_root"])
    if (not root.is_absolute() or root.resolve(strict=True) != root
            or root != ENGINE_ROOT):
        raise ValueError("native engine checkout path is invalid")
    if _command(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root):
        raise ValueError("native engine checkout is dirty")
    commit = _command(["git", "rev-parse", "HEAD"], cwd=root)
    remote = _command(["git", "remote", "get-url", "origin"], cwd=root)
    ref = config["engine_ref"]
    if (not re.fullmatch(r"refs/heads/[A-Za-z0-9._/-]+", ref)
            or ".." in ref or "//" in ref or ref.endswith("/")):
        raise ValueError("native engine ref is invalid")
    if commit != config["engine_commit"] or remote != config["engine_remote"]:
        raise ValueError("native local source selection differs")
    if check_remote:
        remote_line = _command(["git", "ls-remote", "--exit-code", "origin", ref], cwd=root)
        if remote_line != f"{commit}\t{ref}":
            raise ValueError("native pushed source ref differs")
    value = {"schema": "synaptic-native-pushed-source/v1", "url": remote,
             "ref": ref, "commit": commit}
    return value, hashlib.sha256(canonical(value)).hexdigest()


def hardware_record():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("native host requires Darwin arm64")
    value = {"schema": "synaptic-native-hardware-observation/v1",
             "machine": platform.machine(), "macos": platform.mac_ver()[0],
             "kernel": platform.release(),
             "model": _command(["sysctl", "-n", "hw.model"]),
             "cpu": _command(["sysctl", "-n", "machdep.cpu.brand_string"]),
             "memory_bytes": int(_command(["sysctl", "-n", "hw.memsize"]))}
    return value, hashlib.sha256(canonical(value)).hexdigest()


def _snapshot(path):
    value = _document(path)
    if type(value) is not dict or set(value) != {"repository", "revision", "root", "files"}:
        raise ValueError("native snapshot fields differ")
    snapshot = NativeModelSnapshot(value["repository"], value["revision"], value["root"],
                                   tuple(NativeFile(**item) for item in value["files"]))
    with NativeTree(Path(snapshot.root), private=True) as tree:
        snapshot.verify(tree)
    return snapshot


def snapshot_candidate(*, root, repository, revision):
    """Inventory existing private bytes; provenance still needs root review."""
    with NativeTree(Path(root), private=True) as tree:
        snapshot = NativeModelSnapshot(repository, revision, str(root), tree.inventory())
        snapshot.verify(tree)
        tree.check()
    return asdict(snapshot)


def profile_candidate(config):
    """Render unapproved exact profile parameters after source/runtime capture."""
    proposal = strict_json(_read(config["proposal_file"], 1 << 20))
    if (proposal.get("schema") != "synaptic-native-qualification-proposal/v1"
            or proposal.get("status") != "ROOT_REVIEW_REQUIRED_NOT_EXECUTION_AUTHORITY"):
        raise ValueError("native candidate proposal differs")
    _, source_digest = source_record(config)
    hardware, _ = hardware_record()
    if (hardware["cpu"] != proposal["runtime"]["hardware"]
            or hardware["memory_bytes"] != proposal["runtime"]["physical_memory_bytes"]):
        raise ValueError("native candidate hardware differs")
    capture = _read(config["runtime_capture_file"], 16 << 20)
    lock = _read(config["dependency_lock_file"], 1 << 20)
    verify_native_material(capture, expected_digest=hashlib.sha256(capture).hexdigest(),
                           dependency_lock=lock)
    observed = strict_json(capture)["runtime"]
    selected = proposal["runtime"]
    if (observed["python_version"] != selected["python_version"]
            or observed["dependency_lock_sha256"] != selected["dependency_lock_sha256"]
            or observed["distributions"].get("mlx") != selected["mlx_version"]
            or observed["distributions"].get("mlx-lm") != selected["mlx_lm_version"]):
        raise ValueError("native candidate runtime differs")
    selected_model, training, lora, optimizer = (
        proposal["model"], proposal["training"], proposal["lora"], proposal["optimizer"])
    profile_type = NativeMLXGRPOProfile if proposal.get("method") == "grpo" else NativeMLXProfile
    value = profile_type(
        profile_ref=proposal["profile_ref"], model_ref=selected_model["repository"],
        model_revision=selected_model["revision"],
        tokenizer_revision=selected_model["tokenizer_revision"],
        runtime_digest=hashlib.sha256(capture).hexdigest(), source_digest=source_digest,
        lora_layers=lora["layers"], lora_scale_policy="alpha_over_rank",
        beta1=optimizer["beta1"], beta2=optimizer["beta2"],
        epsilon=optimizer["epsilon"], weight_decay=optimizer["weight_decay"],
        bias_correction=optimizer["bias_correction"], schedule=optimizer["schedule"],
        use_fast_tokenizer=config["use_fast_tokenizer"],
        steps_per_report=training["steps_per_report"],
        steps_per_eval=training["steps_per_eval"],
        maximum_updates=training["iterations"],
        maximum_batch_tokens=training["max_batch_tokens"],
        **(proposal["grpo_profile"] if profile_type is NativeMLXGRPOProfile else {}),
    )
    return asdict(value)


class RootSignatureAuthority:
    """Verify an external signature using only a pinned public key on this host."""

    def __init__(self, public_key_path, expected_key_digest, signature_path):
        self.key_path = Path(public_key_path)
        self.key = _read(self.key_path, 16384)
        if hashlib.sha256(self.key).hexdigest() != expected_key_digest:
            raise ValueError("reviewed root public key differs")
        self.signature_path = Path(signature_path)
        self.signature = _read(self.signature_path, 16384)

    def authenticate(self, admission):
        if type(admission) is not NativeRestrictedAdmissionV1:
            return False
        try:
            if admission.tag != hashlib.sha256(self.signature).hexdigest():
                return False
            content = asdict(admission)
            content.pop("tag")
            status, stdout = run_bounded_command(
                ["openssl", "dgst", "-sha256", "-verify", str(self.key_path),
                 "-signature", str(self.signature_path)],
                input_bytes=canonical(content), maximum=4096, timeout=10,
                environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"},
            )
            return (status == 0 and stdout.strip() == b"Verified OK"
                    and _read(self.key_path, 16384) == self.key
                    and _read(self.signature_path, 16384) == self.signature)
        except (OSError, BoundedCommandError, ValueError):
            return False


class _Requests:
    def __init__(self, request, resolved, planning):
        self.request, self.resolved, self.planning = request, resolved, planning

    def load(self, raw):
        if raw != self.request.canonical_json:
            raise ValueError("native public request differs")
        return self.request

    def resolve(self, request):
        if request != self.request:
            raise ValueError("native request differs")
        return self.resolved

    def for_plan(self, plan):
        return self.planning.for_plan(plan)


class _Grants:
    def __init__(self, selection, catalog, material_authority):
        self.s, self.catalog, self.authority = selection, catalog, material_authority

    def authorize(self, requirements):
        allowed = (AuthorizationRequirement("training.start", False),)
        if tuple(requirements) != allowed:
            raise ValueError("native host authorization differs")
        return ExecutionGrant(self.s["grant_ref"])

    def bind(self, grant, *, operation, requirements):
        if grant != self.authorize(requirements):
            raise ValueError("native host grant differs")
        command = parse_exact_command(operation.canonical_bytes)
        _, source_digest = source_record(self.s["config"])
        _, hardware_digest = hardware_record()
        admission = self.s["admission"]
        if (source_digest != admission.source_digest
                or hardware_digest != admission.hardware_digest
                or self.s["root_authority"].authenticate(admission) is not True):
            raise ValueError("native source or hardware admission changed")
        if (command.preparation.run_id != self.s["run_id"]
                or command.preparation.project_ref != self.s["project_ref"]):
            raise ValueError("native host command scope differs")
        invocation = NativeInvocation(
            self.s["attempt"], command.digest, self.s["executable"],
            self.s["executable_digest"], self.s["guardian"], self.s["guardian_digest"],
            self.s["worker"], self.s["worker_digest"], self.s["request_path"],
            self.s["request_digest"], self.s["private_root"], self.s["limits"],
        )
        unsigned = NativeCommandMaterialV1(
            operation.canonical_bytes, invocation, self.s["model"], self.s["tokenizer"],
            self.s["tokenizer_members"], self.s["output"], self.s["budget"],
            self.s["material_authority_ref"], self.s["material_key_ref"], "0" * 64,
        )
        self.catalog.publish_once(self.authority.issue(unsigned))
        return grant


class _ReadOnlyGrants:
    """Cold public reads cannot mint or bind a new training grant."""

    def authorize(self, requirements):
        raise ValueError("native retained read has no grant authority")

    def bind(self, grant, *, operation, requirements):
        raise ValueError("native retained read has no dispatch authority")


def inspect(config, *, require_admission=False, trusted_root_key_sha256=None,
            retained_read=False):
    """Rebuild source, runtime, prepared publication and snapshot bindings."""
    fields = {
        "schema", "proposal_file", "engine_root", "engine_remote", "engine_ref", "engine_commit",
        "runtime_capture_file", "dependency_lock_file", "profile_file", "training_input_file",
        "prepared_identity_file", "prepared_publication", "model_snapshot_file",
        "tokenizer_snapshot_file", "admission_file", "admission_signature_file",
        "root_public_key_file", "root_public_key_sha256", "limits", "scope", "run_id",
        "project_ref", "request_id", "attempt", "attempt_kind", "state_root", "host",
        "tokenizer_members", "artifact_budget", "use_fast_tokenizer",
    }
    lifecycle = None
    if type(config) is dict and config.get("schema") == "synaptic-native-host-config/v2":
        from native_host_lifecycle import validate_policy
        fields.add("lifecycle")
        lifecycle = validate_policy(config["lifecycle"])
        if any(type(config.get(name)) is not str or len(config[name].encode("utf-8")) > 128
               for name in ("run_id", "project_ref")):
            raise ValueError("native lifecycle report scope exceeds host bound")
    if (type(config) is not dict or set(config) != fields
            or config.get("schema") not in {"synaptic-native-host-config/v1", "synaptic-native-host-config/v2"}):
        raise ValueError("native host configuration invalid")
    if (type(config["host"]) is not dict or set(config["host"]) != {
            "authority_secret", "evidence_secret", "material_secret", "grant_ref",
            "reader_authority_ref", "reader_key_ref", "material_authority_ref", "material_key_ref",
    } or type(config["tokenizer_members"]) is not list
            or tuple(config["tokenizer_members"]) != tuple(sorted(set(config["tokenizer_members"])))):
        raise ValueError("native host authority or tokenizer selection invalid")
    if type(retained_read) is not bool or (retained_read and not require_admission):
        raise ValueError("native retained read requires signed selection")
    state_root = Path(config["state_root"])
    if (not state_root.is_absolute() or str(state_root) != os.path.normpath(str(state_root))
            or str(state_root) != config["state_root"]):
        raise ValueError("native state root is noncanonical")
    source, source_digest = source_record(config, check_remote=not retained_read)
    hardware, hardware_digest = hardware_record()
    proposal = strict_json(_read(config["proposal_file"], 1 << 20))
    if (proposal.get("schema") != "synaptic-native-qualification-proposal/v1"
            or proposal.get("status") != "ROOT_REVIEW_REQUIRED_NOT_EXECUTION_AUTHORITY"
            or hardware["cpu"] != proposal["runtime"]["hardware"]
            or hardware["memory_bytes"] != proposal["runtime"]["physical_memory_bytes"]):
        raise ValueError("native hardware differs from reviewed candidate")
    capture = _read(config["runtime_capture_file"], 16 << 20)
    lock = _read(config["dependency_lock_file"], 1 << 20)
    capture_digest = hashlib.sha256(capture).hexdigest()
    verify_native_material(capture, expected_digest=capture_digest, dependency_lock=lock)
    observed_runtime = strict_json(capture)["runtime"]
    selected_runtime = proposal["runtime"]
    if (observed_runtime["python_version"] != selected_runtime["python_version"]
            or observed_runtime["dependency_lock_sha256"] != selected_runtime["dependency_lock_sha256"]
            or observed_runtime["distributions"].get("mlx") != selected_runtime["mlx_version"]
            or observed_runtime["distributions"].get("mlx-lm") != selected_runtime["mlx_lm_version"]):
        raise ValueError("native runtime differs from reviewed candidate")
    runtime = native_runtime_spec(capture)
    if config["engine_root"] + "/tuner" not in strict_json(capture)["config"]["sources"]:
        raise ValueError("native captured source is not selected checkout")
    training = TrainingInputV1.from_dict(_document(config["training_input_file"]))
    grpo = training.method.value == "grpo"
    if grpo != (proposal.get("method") == "grpo"):
        raise ValueError("native proposal method differs from request")
    profile_type = NativeMLXGRPOProfile if grpo else NativeMLXProfile
    profile = profile_type.from_dict(_document(config["profile_file"]))
    selected_model = proposal["model"]
    if (profile.source_digest != source_digest or profile.runtime_digest != capture_digest
            or profile.profile_ref != proposal["profile_ref"]
            or profile.model_ref != selected_model["repository"]
            or profile.model_revision != selected_model["revision"]
            or profile.tokenizer_revision != selected_model["tokenizer_revision"]
            or profile.use_fast_tokenizer is not config["use_fast_tokenizer"]):
        raise ValueError("native profile source/runtime differs")
    identity = PreparedTrainingInputIdentity.from_dict(_document(config["prepared_identity_file"]))
    if grpo:
        # Existing publication format, with Darwin-compatible native custody.
        # This is a read of host-prepared bytes, never a new normalization path.
        from tuner.training.grpo_preparation import verify_grpo_publication_bytes, _identity
        with NativeTree(Path(config["prepared_publication"]), private=True) as publication:
            if publication.members() != ("dataset.jsonl", "manifest.json"):
                raise ValueError("native GRPO publication inventory differs")
            dataset = publication.read("dataset.jsonl", maximum=65536)
            manifest = verify_grpo_publication_bytes(publication.path,
                publication.read("manifest.json", maximum=4096), dataset)
            if (_identity(manifest) != identity
                    or manifest["prompt_column"] != training.hyperparameters.prompt_column):
                raise ValueError("native GRPO prepared identity differs")
            publication.check()
    else:
        prepared = NativePreparedTrainingInputSource(Path(config["prepared_publication"]), identity)
        dataset = read_retained_native_source(prepared, identity)
    rows = [strict_json(line, maximum=1 << 20) for line in dataset.splitlines()]
    counts = {name: sum(row.get("split") == name for row in rows)
              for name in ("train", "validation")}
    if not grpo and (counts != proposal["input_review"]["rows_per_attempt"]
            or sum(counts.values()) != len(rows)):
        raise ValueError("native synthetic publication split coverage differs")
    kind = config["attempt_kind"]
    hp = training.hyperparameters
    if grpo:
        if (kind != "grpo_on_policy" or len(rows) != proposal["input_review"]["prompt_rows"]
                or identity.content_digest != proposal["input_review"]["publication_sha256"]):
            raise ValueError("native GRPO reviewed input differs")
    elif kind == "raw_full_sequence":
        if (identity.format != "syntunia-sft-row/v1" or hp.dataset_format != "raw_text"
                or hp.prompt_render != "full_conversation" or hp.chat_template_kwargs is not None):
            raise ValueError("native raw attempt input differs")
    elif kind == "prompt_completion_assistant":
        if (identity.format not in {"syntunia-sft-row/v2", "synaptic-conversation-sft-row/v1"}
                or hp.dataset_format != "messages" or hp.prompt_render != "prompt_completion"
                or hp.chat_template_kwargs != {"enable_thinking": False}):
            raise ValueError("native prompt/completion attempt input differs")
    else:
        raise ValueError("native attempt kind is invalid")
    model, tokenizer = _snapshot(config["model_snapshot_file"]), _snapshot(config["tokenizer_snapshot_file"])
    if (model.repository != selected_model["repository"]
            or model.revision != selected_model["revision"]
            or tokenizer.repository != selected_model["tokenizer_repository"]
            or tokenizer.revision != selected_model["tokenizer_revision"]
            or sum(item.size for item in model.files) > selected_model["download_budget_bytes"]
            or tuple(config["tokenizer_members"]) != tuple(item.path for item in tokenizer.files)):
        raise ValueError("native complete model or tokenizer snapshot differs")
    projection = (compile_native_grpo if grpo else compile_native_sft)(training, identity, profile)
    if grpo:
        from tuner.runtime.native_mlx_grpo import prepared_prompts
        prepared_prompts(dataset, projection)
        if (asdict(projection.controls) != proposal["grpo_controls"]
                or any(asdict(profile)[key] != value for key, value in proposal["grpo_profile"].items())
                or set(proposal["grpo_profile"]) != set(NativeMLXGRPOProfile.__dataclass_fields__) - set(NativeMLXProfile.__dataclass_fields__)):
            raise ValueError("native GRPO controls differ from reviewed candidate")
    proposed_training, proposed_lora, proposed_optimizer = (
        proposal["training"], proposal["lora"], proposal["optimizer"])
    if (projection.controls.iterations != proposed_training["iterations"]
            or projection.controls.batch_size != proposed_training["batch_size"]
            or projection.controls.max_seq_length != proposed_training["max_seq_length"]
            or projection.controls.seed != proposed_training["seed"]
            or (not grpo and projection.controls.steps_per_save != proposed_training["steps_per_save"])
            or projection.lora.layers != proposed_lora["layers"]
            or projection.lora.rank != proposed_lora["rank"]
            or projection.lora.scale != proposed_lora["scale"]
            or projection.lora.dropout != proposed_lora["dropout"]
            or tuple(projection.lora.targets) != tuple(proposed_lora["targets"])
            or projection.optimizer.learning_rate != proposed_optimizer["learning_rate"]
            or projection.optimizer.beta1 != proposed_optimizer["beta1"]
            or projection.optimizer.beta2 != proposed_optimizer["beta2"]
            or projection.optimizer.epsilon != proposed_optimizer["epsilon"]
            or projection.optimizer.weight_decay != proposed_optimizer["weight_decay"]
            or projection.optimizer.bias_correction != proposed_optimizer["bias_correction"]
            or projection.optimizer.schedule != proposed_optimizer["schedule"]):
        raise ValueError("native attempt controls differ from reviewed candidate")
    resolved = ResolvedTrainingRequest(
        "synaptic-resolved-training-request/v1", config["request_id"], config["project_ref"],
        source_digest, hashlib.sha256(training.canonical_json().encode()).hexdigest(),
        projection.workload.fingerprint, _domain("synaptic-coordinator-runtime/v1", _runtime(runtime)),
        _domain("synaptic-coordinator-artifact-policy/v1", training.to_dict()["artifacts"]),
    )
    limits = NativeProcessLimits(**config["limits"])
    proposed_bounds = proposal["bounds"]
    if (limits.timeout_seconds != proposed_bounds["timeout_seconds_per_attempt"]
            or limits.termination_grace_seconds != proposed_bounds["termination_grace_seconds"]
            or limits.maximum_output_bytes != proposed_bounds["maximum_output_bytes_per_stream"]
            or limits.maximum_file_bytes != proposed_bounds["maximum_artifact_file_bytes"]
            or config["artifact_budget"] != {
                "per_file": proposed_bounds["maximum_artifact_file_bytes"],
                "aggregate": proposed_bounds["maximum_artifact_set_bytes"],
            }):
        raise ValueError("native attempt bounds differ from reviewed candidate")
    scope = ExecutionScopeV1(**config["scope"])
    run = TrainingRunRef(config["run_id"], config["project_ref"])
    base = dict(resolved=resolved, run=run, workload=projection.workload.canonical_bytes,
                runtime_capture=capture, limits=limits, scope=scope, clock=Clock())
    if lifecycle is not None:
        from native_host_lifecycle import validate_controls
        validate_controls(config["lifecycle"], projection, limits)
        base["lifecycle"] = lifecycle
    candidate = NativePlanningV1(**base)
    selection_digest = hashlib.sha256(canonical({
        "schema": "synaptic-native-host-execution-selection/v1",
        "config": config, "source": source, "hardware": hardware,
        "proposal": proposal, "profile": asdict(profile),
        "training_input": training.to_dict(), "prepared_identity": identity.to_dict(),
        "prepared_digest": hashlib.sha256(dataset).hexdigest(),
        "capture_digest": capture_digest, "dependency_lock_digest": hashlib.sha256(lock).hexdigest(),
        "model": asdict(model), "tokenizer": asdict(tokenizer),
        "workload_digest": projection.workload.fingerprint,
    })).hexdigest()
    admission_fields = dict(
        profile_digest=candidate.profile_digest, source_digest=source_digest,
        runtime_digest=resolved.runtime_digest, hardware_digest=hardware_digest,
        prepared_digest=identity.content_digest, model_digest=model.digest,
        tokenizer_digest=tokenizer.digest, run_id=run.run_id, attempt=config["attempt"],
        workload_digest=projection.workload.fingerprint, selection_digest=selection_digest,
        maximum_updates=projection.controls.iterations,
    )
    if not require_admission:
        return {"source": source, "hardware": hardware, "admission_candidate": admission_fields,
                "capture_digest": capture_digest, "prepared_rows_bytes": len(dataset)}, None
    admission_value = _document(config["admission_file"])
    if set(admission_value) != set(NativeRestrictedAdmissionV1.__dataclass_fields__):
        raise ValueError("restricted admission fields differ")
    admission = NativeRestrictedAdmissionV1(**admission_value)
    if any(getattr(admission, name) != value for name, value in admission_fields.items()):
        raise ValueError("restricted admission selection differs")
    if trusted_root_key_sha256 is None or trusted_root_key_sha256 != config["root_public_key_sha256"]:
        raise ValueError("independent root key pin is required")
    root = RootSignatureAuthority(config["root_public_key_file"],
                                  trusted_root_key_sha256, config["admission_signature_file"])
    if not root.authenticate(admission):
        raise ValueError("external restricted admission signature invalid")
    options = dict(restricted_admission=admission, restricted_authority=root,
                   selection_digest=selection_digest, retained_read=retained_read)
    if lifecycle is not None:
        from native_host_lifecycle import RestrictedLifecycleAuthority
        grant = NativeLifecycleGrantV1(
            admission.profile_digest, admission.source_digest, admission.runtime_digest,
            hashlib.sha256(canonical(lifecycle.to_dict())).hexdigest(), admission.hardware_digest,
            "restricted", (), admission.selection_digest, admission.run_id, admission.attempt,
            admission.authority_ref, admission.key_ref, admission.tag,
        )
        options.update(lifecycle_grant=grant,
                       lifecycle_authority=RestrictedLifecycleAuthority(grant, admission, root))
    planning = NativePlanningV1(**base, **options)
    if not (planning.restricted_selection() if retained_read else planning.restricted()):
        raise ValueError("restricted native admission expired or refused")
    return {"source": source, "hardware": hardware, "admission_candidate": admission_fields,
            "capture_digest": capture_digest, "prepared_rows_bytes": len(dataset)}, (
        planning, training, projection, dataset, lock, model, tokenizer,
    )


def operate(config, *, action="start", destination=None, trusted_root_key_sha256=None):
    """Start once, or recompose the same selected run for a cold public read."""
    lifecycle = config.get("schema") == "synaptic-native-host-config/v2"
    if action not in {"start", "qualify", "recover", "logs", "show", "observe", "verify", "reverify", "download"}:
        raise ValueError("native public action is unavailable")
    if (lifecycle and action == "start") or (not lifecycle and action in {"qualify", "recover", "logs"}):
        raise ValueError("native action requires its selected host workflow")
    starting = action in {"start", "qualify"}
    state = Path(config["state_root"])
    if not starting:
        _private(state)
        if _document(state / "config.json", 4 << 20) != config:
            raise ValueError("retained native host configuration differs")
    _, selected = inspect(config, require_admission=True,
                          trusted_root_key_sha256=trusted_root_key_sha256,
                          retained_read=not starting)
    planning, training, projection, dataset, lock, model, tokenizer = selected
    if starting:
        _create_private_dir(state)
    _private(state)
    private = state / "native"
    if starting:
        _create_private_dir(private)
    _private(private)
    output = private / "output"
    if starting:
        _create_private_dir(output)
    _private(output)
    if starting:
        _write_owned(private, "capture.json", planning.runtime_capture, 16 << 20)
        _write_owned(private, "dependency.lock", lock, 1 << 20)
        _write_owned(private, "dataset.jsonl", dataset, 64 << 20)
    elif (_read(private / "capture.json", 16 << 20) != planning.runtime_capture
          or _read(private / "dependency.lock", 1 << 20) != lock
          or _read(private / "dataset.jsonl", 64 << 20) != dataset):
        raise ValueError("retained native input differs")
    request_raw = canonical({
        "schema": "synaptic-native-child-request/v2" if lifecycle else "synaptic-native-child-request/v1",
        **({"lifecycle": planning.lifecycle.to_dict()} if lifecycle else {}),
        "workload": projection.workload.document,
        "runtime_capture": str(private / "capture.json"), "dependency_lock": str(private / "dependency.lock"),
        "dataset": str(private / "dataset.jsonl"), "output": str(output),
        "model_snapshot": asdict(model), "tokenizer_snapshot": asdict(tokenizer),
        "tokenizer_members": config["tokenizer_members"],
        "artifact_budget": config["artifact_budget"],
    })
    selection_document = {"plan": planning.plan.to_dict(),
                          "admission": asdict(planning._restricted_admission)}
    if starting:
        _write_owned(private, "request.json", request_raw, 2 << 20)
        _write_owned(state, "config.json", canonical(config), 4 << 20)
        _write_owned(state, "selection.json", canonical(selection_document), 2 << 20)
        _create_private_dir(state / "scratch")
    elif (_read(private / "request.json", 2 << 20) != request_raw
          or _document(state / "config.json", 4 << 20) != config
          or _document(state / "selection.json") != selection_document):
        raise ValueError("retained native selection differs")
    _private(state / "scratch")
    secrets, clock = NamedSecrets(), Clock()
    with ExitStack() as lifetime:
        store = lifetime.enter_context(SQLiteDurableStoreV1(state / "workflow.sqlite3", create=starting))
        key = secrets.resolve(SecretRef.from_dict(config["host"]["evidence_secret"])).encode()
        material_key = secrets.resolve(SecretRef.from_dict(config["host"]["material_secret"])).encode()
        host = config["host"]
        material_purposes = {"native-command-material/v1"}
        evidence_purposes = {"provider-run-observation/v1", "provider-log-page/v1",
                             "native-artifact-verification/v1", "native-process-outcome/v1"}
        if lifecycle:
            material_purposes.add("native-cancellation-material/v1")
            evidence_purposes.update({"native-lifecycle-capability/v1", "native-stage-ack/v1",
                                      "native-cancellation-ack/v1", "native-progress-metadata/v1",
                                      "native-quiescence/v1"})
        material_authority = NativeMaterialAuthority(host["material_authority_ref"],
            host["material_key_ref"], HMACAuthenticator({host["material_key_ref"]: material_key},
                allowed_purposes=frozenset(material_purposes)))
        evidence = NativeEvidenceAuthority(host["reader_authority_ref"], host["reader_key_ref"],
            HMACAuthenticator({host["reader_key_ref"]: key}, allowed_purposes=frozenset(evidence_purposes)))
        catalog = NativeCommandCatalogV1(store, material_authority)
        capture = strict_json(planning.runtime_capture)
        members = {str(Path(tree["root"]) / item["path"]): item["sha256"]
                   for tree in capture["trees"] for item in tree["members"]}
        worker = config["engine_root"] + "/tuner/runtime/native_mlx_child.py"
        guardian = config["engine_root"] + "/tuner/runtime/native_process_guardian.py"
        selection = dict(run_id=config["run_id"], project_ref=config["project_ref"],
            grant_ref=host["grant_ref"], attempt=config["attempt"], config=config,
            admission=planning._restricted_admission,
            root_authority=planning._restricted_authority,
            executable=capture["executable"], executable_digest=capture["runtime"]["executable_sha256"],
            worker=worker, worker_digest=members[worker], guardian=guardian, guardian_digest=members[guardian],
            request_path=str(private / "request.json"), request_digest=hashlib.sha256(request_raw).hexdigest(),
            private_root=str(private), output=str(output), limits=planning.limits,
            model=model, tokenizer=tokenizer, tokenizer_members=tuple(config["tokenizer_members"]),
            budget=NativeArtifactBudget(**config["artifact_budget"]),
            material_authority_ref=host["material_authority_ref"], material_key_ref=host["material_key_ref"])
        grants = (_Grants(selection, catalog, material_authority) if starting else _ReadOnlyGrants())
        if lifecycle and starting:
            from native_host_lifecycle import LifecycleGrants
            grants = LifecycleGrants(grants, selection, catalog, material_authority)
        ports = ReferenceHostPortsV1(store, store, clock, grants, secrets)
        authority = compose_reference_authority(
            clock=clock, secrets=secrets,
            authority_secret=SecretRef.from_dict(host["authority_secret"]),
        )
        session_options = {}
        session = None
        if lifecycle:
            session = native_lifecycle_session(records=store, launch=planning.lifecycle,
                                               authority=evidence, streams=store)
            lifetime.callback(session.close)
            session_options["lifecycle_session"] = session
        family = native_mlx_family(
            planning=planning, catalog=catalog, material_authority=material_authority,
            records=store, evidence_authority=evidence,
            foundation_authenticator=authority.foundation_authenticator,
            assessment_authenticator=authority.assessment_authority, clock=clock, **session_options,
        )
        request = TrainingRequest(config["request_id"], config["project_ref"], training.canonical_json())
        requests = _Requests(request, planning.resolved, planning)
        api = compose_reference_host(
            family=family, ports=ports,
            requests=ReferenceRequestPortsV1(requests, requests, requests), authority=authority,
            **({"allow_current_cancel_authorization": True} if lifecycle else {}),
        ).api()
        if starting:
            plan = api.training.plan(api.training.resolve(api.training.load(request.canonical_json)), planning.provider)
            preflight = api.training.preflight(plan)
            if not preflight.ready or not preflight.binds(plan):
                raise ValueError("restricted native public preflight refused")
            started = api.training.start(plan, preflight)
            if lifecycle:
                from native_host_lifecycle import qualify
                return qualify(config, api, started, state, _write_owned)
            return {"run": started.run.to_dict(), "accepted": started.accepted,
                    "stored": api.runs.show(started.run).to_dict()}
        run = planning.run
        if action in {"recover", "logs"}:
            from native_host_lifecycle import recover, logs
            return (recover(config, api, run, state, _write_owned) if action == "recover"
                    else logs(config, api, run))
        if action == "show":
            return api.runs.show(run).to_dict()
        if action == "observe":
            return api.runs.outcome(run).to_dict()
        if action in {"verify", "reverify"}:
            report = (api.runs.verify(run) if action == "verify"
                      else api.runs.reverify(run))
            return {"run": report.run.to_dict(), "verified": report.verified,
                    "checked_at": report.checked_at}
        if destination is None:
            raise ValueError("native artifact destination required")
        return {"paths": [str(path) for path in download(api.runs, run, destination,
                                                          maximum_bytes=64 << 20)]}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("canonicalize", "profile-candidate", "inspect",
                                          "admission-candidate", "seal-admission", "start", "qualify", "recover", "logs",
                                          "show", "observe", "verify", "reverify", "download",
                                          "snapshot-candidate"))
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--trusted-root-key-sha256")
    parser.add_argument("--expires-at")
    parser.add_argument("--authority-ref")
    parser.add_argument("--key-ref")
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--repository")
    parser.add_argument("--revision")
    args = parser.parse_args(argv)
    try:
        if args.action == "snapshot-candidate":
            if args.config is not None or args.snapshot_root is None or not args.repository or not args.revision:
                raise ValueError("native snapshot candidate arguments differ")
            result = snapshot_candidate(root=args.snapshot_root, repository=args.repository,
                                        revision=args.revision)
        else:
            if args.config is None:
                raise ValueError("native host configuration required")
            config = (strict_json(_read(args.config, 4 << 20), maximum=4 << 20)
                      if args.action == "canonicalize" else _document(args.config, 4 << 20))
        if args.action == "snapshot-candidate":
            pass
        elif args.action == "canonicalize":
            result = config
            if args.destination is not None:
                destination = args.destination
                raw = canonical(config)
                _write_owned(destination.parent, destination.name, raw, 4 << 20)
                result = {"path": str(destination), "sha256": hashlib.sha256(raw).hexdigest()}
        elif args.action == "profile-candidate":
            result = profile_candidate(config)
        elif args.action == "inspect":
            result = inspect(config)[0]
        elif args.action == "admission-candidate":
            if not all((args.expires_at, args.authority_ref, args.key_ref)):
                raise ValueError("reviewer admission fields required")
            result = {**inspect(config)[0]["admission_candidate"],
                      "expires_at": args.expires_at, "authority_ref": args.authority_ref,
                      "key_ref": args.key_ref}
            NativeRestrictedAdmissionV1(**result, tag="0" * 64)
        elif args.action == "seal-admission":
            if args.signature is None:
                raise ValueError("external signature file required")
            if set(config) != set(NativeRestrictedAdmissionV1.__dataclass_fields__) - {"tag"}:
                raise ValueError("unsigned admission fields differ")
            result = {**config, "tag": hashlib.sha256(_read(args.signature, 16384)).hexdigest()}
            NativeRestrictedAdmissionV1(**result)
        else:
            result = operate(config, action=args.action, destination=args.destination,
                             trusted_root_key_sha256=args.trusted_root_key_sha256)
    except Exception:
        print(json.dumps({"diagnostic_code": "native_host_refused"}))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
