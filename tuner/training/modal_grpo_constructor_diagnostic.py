"""Restricted one-shot Modal GRPO constructor capture beside CPU qualification.

This private operator shares the packaged deployment, source, attempt journal,
catalog and bounded Sandbox capture boundaries. Its result cannot be parsed as
a CPU or training qualification receipt and grants no TrainingAPI authority.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
from pathlib import Path
import queue
import threading
import time
from zipfile import ZipFile

from scripts.capture_modal_inference_runtime import _OperationOwnership, _bounded_call
from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.execution.providers.modal.facade import MODAL_VOLUME_V1
from tuner.execution.providers.modal.model_snapshot import authoritative_model_members, _bind_hub_api
from tuner.execution.providers.modal.runtime_release_qualification import (
    QUALIFICATION_HMAC_KEY_REF, ModalRuntimeQualificationHmacAuthenticator,
    _observe_current_layout, _validate_deployment_release,
)
from tuner.project.source_bundle import inspect_git_source, _git
from tuner.runtime.packaged_grpo_constructor_child import (
    DISPATCH_SCHEMA, canonical, validate_dispatch, validate_model_inventory, validate_result,
)
from tuner.runtime.packaged_grpo_worker_closure import load_packaged_grpo_worker_closure
from tuner.runtime.offline_grpo_worker import load_packaged_offline_grpo_worker_manifest
from tuner.runtime.packaged_worker_closure import stable_read
from tuner.training.modal_host_qualification import _CurrentQualificationObserver
from tuner.training.modal_host_runtime import ModalHostRuntimeV1, quote_modal_runtime_for_host

_SEAL = object()
_PURPOSE = "modal-grpo-constructor-diagnostic/v1"
_CLAIM = "synaptic-modal-grpo-constructor-claim/v1"
_RECEIPT = "synaptic-modal-grpo-constructor-receipt/v1"
_OUTPUT_MAX = 8192


class ModalGRPOConstructorUnavailable(RuntimeError):
    """Closed diagnostic failure; claims remain consumed on uncertain effects."""

    def __init__(self, phase):
        if phase not in {"ADMISSION", "CLAIM", "QUOTE", "APP", "LAYOUT", "VOLUME", "IMAGE",
                         "CREATE", "INPUT", "OUTPUT", "TERMINAL", "CLEANUP", "RECEIPT"}:
            raise ValueError
        self.phase = phase
        super().__init__("modal_grpo_constructor_diagnostic_unavailable")


@dataclass(frozen=True, slots=True)
class ModalGRPOConstructorAdmission:
    dispatch_bytes: bytes
    attempt_ref: str
    source_commit: str
    host_source_commit: str
    wheel_sha256: str
    runtime_release_digest: str
    deployment_facts_digest: str
    image_id: str
    snapshot_deployment_facts_digest: str
    model_cache_volume_id: str
    model_cache_volume_name: str
    quote_digest: str
    _seal: object = field(repr=False)


def _model_role(runtime):
    matches = tuple(item.volume_id for item in runtime.deployment_facts.volumes
                    if item.spec.role == "model_cache")
    if len(matches) != 1 or matches[0] != runtime.facts.model_cache_volume_id:
        raise ValueError
    return matches[0]


def _pinned_model_info(model):
    """One exact public metadata read, independent of the retained Volume."""
    from huggingface_hub import HfApi, snapshot_download
    _bind_hub_api(HfApi, snapshot_download)
    return HfApi(endpoint="https://huggingface.co", token=False).model_info(
        model["ref"], revision=model["revision"], files_metadata=True, token=False)


def _source_pair(engine_repository, host_repository):
    engine = inspect_git_source(engine_repository)
    host = inspect_git_source(host_repository)
    if (engine.dirty or not engine.pushed or host.dirty or not host.pushed
            or engine.commit is None or host.commit is None):
        raise ValueError
    line = _git(host_repository, "ls-tree", "HEAD", "--", "synaptic-tuner")
    if line != f"160000 commit {engine.commit}\tsynaptic-tuner":
        raise ValueError
    return engine.commit, host.commit


def _verify_wheel_sources(wheel, repository):
    """Compare the complete fixed GRPO child source closure to wheel bytes."""
    offline = load_packaged_offline_grpo_worker_manifest()
    outer = load_packaged_grpo_worker_closure()
    members = {item.path.as_posix() for item in offline.closure.members}
    members.update(item["path"] for item in outer.members)
    with ZipFile(BytesIO(wheel)) as archive:
        names = set(archive.namelist())
        if not members <= names:
            raise ValueError
        for name in sorted(members):
            raw = stable_read(repository / name, 32 * 1024 * 1024)
            if archive.read(name) != raw:
                raise ValueError
    return outer.digest


def admit_modal_grpo_constructor(*, runtime, snapshot_runtime,
                                  engine_repository: Path, host_repository: Path,
                                  wheel_path: Path, recipe_path: Path,
                                  plan_ready_path: Path, dataset_path: Path,
                                  profiles_root: Path,
                                  input_bytes: bytes, model: dict,
                                  library_sources: dict, attempt_ref: str):
    """Exact selection with one pinned public metadata read; no model download.

    The expected SHA256s are an independently reviewed manifest; the SDK
    metadata and Sandbox both revalidate upstream Git/LFS identities.
    """
    try:
        if (type(runtime) is not ModalHostRuntimeV1 or type(snapshot_runtime) is not ModalHostRuntimeV1
                or runtime.facts.schema_version != "synaptic-modal-packaged-runtime-facts/v2"
                or snapshot_runtime.facts.schema_version != "synaptic-modal-packaged-runtime-facts/v2"):
            raise ValueError
        engine_commit, host_commit = _source_pair(engine_repository, host_repository)
        closure = load_packaged_grpo_worker_closure()
        release, deployment = runtime.release, runtime.deployment_facts
        if (release.worker_closure_digest != closure.digest
                or release.manifest_digest != deployment.runtime_release_digest
                or runtime.facts.image_id != deployment.image_id
                or runtime.facts.package_digest != release.package_digest
                or runtime.facts.worker_closure_digest != closure.digest):
            raise ValueError
        _validate_deployment_release(deployment, release)
        _validate_deployment_release(snapshot_runtime.deployment_facts, snapshot_runtime.release)
        if (runtime.facts.account_ref != "synaptic-labs"
                or runtime.facts.workspace_ref != "synaptic-labs"
                or runtime.facts.environment_ref != "synaptic-smoke-v1"
                or deployment.client_binding.environment_ref != runtime.facts.environment_ref
                or snapshot_runtime.deployment_facts.client_binding != deployment.client_binding
                or snapshot_runtime.facts.account_ref != runtime.facts.account_ref
                or snapshot_runtime.facts.workspace_ref != runtime.facts.workspace_ref
                or snapshot_runtime.facts.environment_ref != runtime.facts.environment_ref):
            raise ValueError
        if (runtime.quote.resource.accelerator != "L40S"
                or runtime.quote.resource.accelerator_count != 1
                or runtime.quote.resource.timeout_seconds != 120
                or runtime.quote.maximum_cost_minor_units > 100
                or runtime.quote.gpu_only_timeout_estimate_minor_units > 100):
            raise ValueError
        wheel = stable_read(wheel_path, 256 * 1024 * 1024)
        wheel_sha = hashlib.sha256(wheel).hexdigest()
        if wheel_sha != release.package_digest:
            raise ValueError
        if _verify_wheel_sources(wheel, engine_repository) != closure.digest:
            raise ValueError
        expected = validate_model_inventory(model)
        upstream = authoritative_model_members(_pinned_model_info(model), model_ref=model["ref"],
                                                revision=model["revision"])
        if (len(upstream) != len(expected)
                or tuple(sorted(upstream)) != tuple(item[0] for item in expected)
                or any(upstream[item["path"]] !=
                       (item["size_bytes"], item["kind"], item["upstream_digest"])
                       for item in model["members"])):
            raise ValueError
        from tuner.cloud.grpo_constructor_diagnostic import verify_probe_input
        from tuner.cloud.grpo_constructor_diagnostic import derive_probe_input
        if input_bytes != derive_probe_input(recipe_path=recipe_path,
                plan_ready_path=plan_ready_path, dataset_path=dataset_path,
                profiles_root=profiles_root):
            raise ValueError
        probe = verify_probe_input(input_bytes)
        workload_model = probe["workload"]["model"]
        from tuner.runtime_profiles import load_runtime_profile
        profile = load_runtime_profile(probe["runtime_profile"], profiles_root)
        if (workload_model["model_id"] != model["ref"]
                or workload_model["revision"] != model["revision"]
                or workload_model["tokenizer_revision"] != model["revision"]
                or profile.profile_sha256 != probe["profile_sha256"]
                or profile.image != release.image_ref):
            raise ValueError
        from tuner.runtime.packaged_grpo_constructor_child import _SOURCES
        if type(library_sources) is not dict or set(library_sources) != set(_SOURCES):
            raise ValueError
        dispatch = {
            "schema_version": DISPATCH_SCHEMA, "attempt_ref": attempt_ref,
            "source_commit": engine_commit, "host_source_commit": host_commit,
            "engine_gitlink": engine_commit, "release": release.to_dict(),
            "deployment_facts_digest": deployment.facts_digest,
            "image_id": deployment.image_id,
            "input_base64": base64.b64encode(input_bytes).decode("ascii"),
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "model": model, "library_sources": library_sources,
        }
        raw = canonical(dispatch)
        validate_dispatch(raw)
        volume_id = _model_role(snapshot_runtime)
        volume_name = dict(snapshot_runtime.volume_names_by_id).get(volume_id)
        if type(volume_name) is not str:
            raise ValueError
        return ModalGRPOConstructorAdmission(raw, attempt_ref, engine_commit, host_commit,
            wheel_sha,
            release.manifest_digest, deployment.facts_digest, deployment.image_id,
            snapshot_runtime.deployment_facts.facts_digest, volume_id, volume_name,
            runtime.quote.quote_digest, _SEAL)
    except Exception:
        raise ModalGRPOConstructorUnavailable("ADMISSION") from None


def _read_limited(stream, limit, output):
    captured = bytearray()
    try:
        for chunk in stream:
            if type(chunk) is str:
                chunk = chunk.encode("utf-8")
            if type(chunk) is not bytes or len(chunk) > limit - len(captured):
                raise ValueError
            captured.extend(chunk)
        output.put((True, bytes(captured)))
    except BaseException:
        output.put((False, None))


def _terminal_by_id(sdk, client, sandbox_id, *, deadline):
    def read():
        handle = sdk.Sandbox.from_id(sandbox_id, client=client)
        if getattr(handle, "object_id", None) != sandbox_id:
            raise ValueError
        return handle.poll()
    return _bounded_call(read, deadline=deadline, timeout_code="constructor_terminal_unavailable")


def _cleanup_owned(sdk, client, sandbox, *, expected_id, deadline):
    """A terminate response is never proof; fresh-ID terminal inspection is."""
    if sandbox is None or getattr(sandbox, "object_id", None) != expected_id:
        return "INDETERMINATE"
    try:
        state = _terminal_by_id(sdk, client, expected_id, deadline=deadline)
        if state is None:
            _bounded_call(lambda: sandbox.terminate(wait=False), deadline=deadline,
                          timeout_code="constructor_cleanup_unavailable")
        while time.monotonic() < deadline:
            state = _terminal_by_id(sdk, client, expected_id, deadline=deadline)
            if type(state) is int:
                return "TERMINAL_CONFIRMED"
            time.sleep(0.05)
    except BaseException:
        pass
    return "INDETERMINATE"


def _catalog(storage, name):
    return storage.catalog(name, encode=lambda value: canonical_bytes(value),
                           decode=lambda raw: parse_canonical_object(raw, name=name))


def _receipt(runtime, admission, *, sandbox_id, primary, cleanup, result, terminal,
             failure_phase=None):
    if (primary not in {"COMPLETED", "REJECTED", "INDETERMINATE"}
            or cleanup not in {"NOT_STARTED", "TERMINAL_CONFIRMED", "INDETERMINATE"}
            or failure_phase not in {None, "QUOTE", "APP", "LAYOUT", "VOLUME", "IMAGE",
                                  "CREATE", "INPUT", "OUTPUT", "TERMINAL"}
            or (primary == "COMPLETED") != (failure_phase is None)
            or (primary == "COMPLETED" and (result is None or terminal != 0))):
        raise ValueError
    body = {"schema_version": _RECEIPT, "diagnostic_type": "GRPO_CONSTRUCTOR_ONLY",
            "attempt_ref": admission.attempt_ref, "dispatch_sha256": hashlib.sha256(admission.dispatch_bytes).hexdigest(),
            "source_commit": admission.source_commit,
            "host_source_commit": admission.host_source_commit,
            "wheel_sha256": admission.wheel_sha256,
            "runtime_release_digest": admission.runtime_release_digest,
            "deployment_facts_digest": admission.deployment_facts_digest,
            "image_id": admission.image_id,
            "snapshot_deployment_facts_digest": admission.snapshot_deployment_facts_digest,
            "model_cache_volume_id": admission.model_cache_volume_id,
            "quote_digest": admission.quote_digest, "sandbox_id": sandbox_id,
            "primary": primary, "cleanup": cleanup, "terminal": terminal,
            "failure_phase": failure_phase,
            "result_sha256": hashlib.sha256(result).hexdigest() if result is not None else None,
            "result": validate_result(result) if result is not None else None,
            "cpu_qualified": False, "training_qualified": False, "training_executed": False}
    auth = ModalRuntimeQualificationHmacAuthenticator(runtime.qualification_key)
    tag = auth.sign(_PURPOSE, canonical_bytes(body), QUALIFICATION_HMAC_KEY_REF)
    return {"body": body, "tag_hex": tag.hex()}


def read_modal_grpo_constructor_receipt(runtime, storage, attempt_ref):
    """Fresh journal readback; this receipt cannot satisfy CPU qualification."""
    value = _catalog(storage, "modal-grpo-constructor-results").resolve(attempt_ref)
    if type(value) is not dict or set(value) != {"body", "tag_hex"}:
        raise ModalGRPOConstructorUnavailable("RECEIPT")
    body, tag_hex = value["body"], value["tag_hex"]
    try:
        tag = bytes.fromhex(tag_hex)
        auth = ModalRuntimeQualificationHmacAuthenticator(runtime.qualification_key)
        if (type(body) is not dict or body.get("schema_version") != _RECEIPT
                or body.get("diagnostic_type") != "GRPO_CONSTRUCTOR_ONLY"
                or body.get("attempt_ref") != attempt_ref
                or body.get("cpu_qualified") is not False
                or body.get("training_qualified") is not False
                or body.get("training_executed") is not False
                or body.get("primary") not in {"COMPLETED", "REJECTED", "INDETERMINATE"}
                or body.get("cleanup") not in {"NOT_STARTED", "TERMINAL_CONFIRMED", "INDETERMINATE"}
                or body.get("runtime_release_digest") != runtime.release.manifest_digest
                or body.get("deployment_facts_digest") != runtime.deployment_facts.facts_digest
                or body.get("image_id") != runtime.deployment_facts.image_id
                or not auth.verify(_PURPOSE, canonical_bytes(body), tag, QUALIFICATION_HMAC_KEY_REF)):
            raise ValueError
        result = body.get("result")
        if (result is None) != (body.get("result_sha256") is None):
            raise ValueError
        if result is not None:
            raw = canonical(result)
            validate_result(raw)
            if hashlib.sha256(raw).hexdigest() != body["result_sha256"]:
                raise ValueError
        return value
    except Exception:
        raise ModalGRPOConstructorUnavailable("RECEIPT") from None


def execute_modal_grpo_constructor_once(*, sdk, client, client_binding, runtime,
                                         snapshot_runtime,
                                         private_storage, admission,
                                         engine_repository: Path, host_repository: Path,
                                         wheel_path: Path, recipe_path: Path,
                                         plan_ready_path: Path, dataset_path: Path,
                                         profiles_root: Path):
    """One claimed Sandbox. No retry, training Function or CPU self-check."""
    phase = "ADMISSION"
    if (type(admission) is not ModalGRPOConstructorAdmission or admission._seal is not _SEAL
            or type(runtime) is not ModalHostRuntimeV1
            or type(snapshot_runtime) is not ModalHostRuntimeV1
            or client_binding != runtime.deployment_facts.client_binding
            or getattr(sdk, "__version__", None) != "1.5.4"):
        raise ModalGRPOConstructorUnavailable(phase)
    # Revalidate source and wheel before granting a durable one-shot effect.
    try:
        engine_commit, host_commit = _source_pair(engine_repository, host_repository)
    except Exception:
        raise ModalGRPOConstructorUnavailable(phase) from None
    if (engine_commit != admission.source_commit
            or host_commit != admission.host_source_commit
            or hashlib.sha256(stable_read(wheel_path, 256 * 1024 * 1024)).hexdigest() != admission.wheel_sha256
            or _verify_wheel_sources(stable_read(wheel_path, 256 * 1024 * 1024), engine_repository)
               != runtime.release.worker_closure_digest
            or runtime.release.manifest_digest != admission.runtime_release_digest
            or runtime.deployment_facts.facts_digest != admission.deployment_facts_digest
            or runtime.deployment_facts.image_id != admission.image_id
            or snapshot_runtime.deployment_facts.facts_digest != admission.snapshot_deployment_facts_digest
            or snapshot_runtime.deployment_facts.client_binding != client_binding
            or _model_role(snapshot_runtime) != admission.model_cache_volume_id
            or dict(snapshot_runtime.volume_names_by_id).get(admission.model_cache_volume_id)
               != admission.model_cache_volume_name):
        raise ModalGRPOConstructorUnavailable(phase)
    try:
        _, input_bytes = validate_dispatch(admission.dispatch_bytes)
        from tuner.cloud.grpo_constructor_diagnostic import derive_probe_input
        if input_bytes != derive_probe_input(recipe_path=recipe_path,
                plan_ready_path=plan_ready_path, dataset_path=dataset_path,
                profiles_root=profiles_root):
            raise ValueError
    except Exception:
        raise ModalGRPOConstructorUnavailable(phase) from None
    deadline = time.monotonic() + 240  # Begins before the first provider effect.
    claim = canonical_bytes({"schema_version": _CLAIM, "attempt_ref": admission.attempt_ref,
        "dispatch_sha256": hashlib.sha256(admission.dispatch_bytes).hexdigest(),
        "deployment_facts_digest": admission.deployment_facts_digest,
        "image_id": admission.image_id, "volume_id": admission.model_cache_volume_id,
        "snapshot_deployment_facts_digest": admission.snapshot_deployment_facts_digest,
        "source_commit": admission.source_commit, "wheel_sha256": admission.wheel_sha256,
        "host_source_commit": admission.host_source_commit,
        "quote_digest": admission.quote_digest})
    phase = "CLAIM"
    private_storage.attempts.claim(admission.attempt_ref, claim)
    if private_storage.attempts.resolve(admission.attempt_ref) != claim:
        raise ModalGRPOConstructorUnavailable(phase)
    sandbox, sandbox_id, result, terminal = None, None, None, None
    primary, cleanup = "INDETERMINATE", "NOT_STARTED"
    failure_phase = None
    ownership = _OperationOwnership()
    try:
        phase = "QUOTE"
        live_quote = _bounded_call(lambda: quote_modal_runtime_for_host(
            sdk=sdk, client=client, client_binding=client_binding,
            recipe_resource=runtime.quote.resource,
            maximum_cost_minor_units=runtime.quote.maximum_cost_minor_units),
            deadline=deadline, timeout_code="constructor_quote_unavailable")
        if live_quote.quote_digest != admission.quote_digest:
            raise ValueError
        phase = "APP"
        app = _bounded_call(lambda: sdk.App.lookup(runtime.facts.app_name, create_if_missing=False,
            client=client, environment_name=runtime.facts.environment_ref),
            deadline=deadline, timeout_code="constructor_app_unavailable")
        phase = "LAYOUT"
        observer = _CurrentQualificationObserver(sdk=sdk, client=client,
            binding=client_binding, runtime=runtime)
        _bounded_call(lambda: _observe_current_layout(observer, runtime.deployment_facts),
                      deadline=deadline, timeout_code="constructor_layout_unavailable")
        snapshot_observer = _CurrentQualificationObserver(sdk=sdk, client=client,
            binding=client_binding, runtime=snapshot_runtime)
        _bounded_call(lambda: _observe_current_layout(
            snapshot_observer, snapshot_runtime.deployment_facts),
            deadline=deadline, timeout_code="constructor_layout_unavailable")
        phase = "VOLUME"
        name = admission.model_cache_volume_name
        if type(name) is not str:
            raise ValueError
        volume = _bounded_call(lambda: sdk.Volume.from_name(name,
            environment_name=runtime.facts.environment_ref, create_if_missing=False,
            version=MODAL_VOLUME_V1, client=client), deadline=deadline,
            timeout_code="constructor_volume_unavailable")
        _bounded_call(lambda: volume.hydrate(client), deadline=deadline,
                      timeout_code="constructor_volume_unavailable")
        if getattr(volume, "object_id", None) != admission.model_cache_volume_id:
            raise ValueError
        readonly = volume.with_mount_options(read_only=True)
        if (getattr(readonly, "object_id", None) != admission.model_cache_volume_id
                or getattr(readonly, "_read_only", None) is not True):
            raise ValueError
        phase = "IMAGE"
        image = _bounded_call(lambda: sdk.Image.from_id(admission.image_id, client=client),
                              deadline=deadline, timeout_code="constructor_image_unavailable")
        if getattr(image, "object_id", None) != admission.image_id:
            raise ValueError
        phase = "CREATE"
        cleanup = "INDETERMINATE"
        def late_cleanup(handle):
            identity = getattr(handle, "object_id", None)
            if type(identity) is str:
                _cleanup_owned(sdk, client, handle, expected_id=identity,
                               deadline=time.monotonic() + 30)
        sandbox = _bounded_call(lambda: sdk.Sandbox.create(
            runtime.release.python_executable, "-I", "-B", "-m",
            "tuner.runtime.packaged_grpo_constructor_child", "--supervise",
            app=app, name=admission.attempt_ref, image=image,
            gpu="L40S", cpu=2.0, memory=(8192, 8192), timeout=120,
            idle_timeout=120, block_network=True, secrets=[],
            volumes={"/mnt/synaptic-model-cache": readonly}, client=client),
            deadline=deadline, timeout_code="constructor_create_indeterminate",
            late_result=late_cleanup, ownership=ownership)
        ownership.transfer(sandbox)
        sandbox_id = getattr(sandbox, "object_id", None)
        if type(sandbox_id) is not str or not sandbox_id.startswith("sb-"):
            raise ValueError
        call_catalog = _catalog(private_storage, "modal-grpo-constructor-sandboxes")
        if (not call_catalog.publish_if_absent(admission.attempt_ref,
                {"sandbox_id": sandbox_id, "dispatch_sha256": hashlib.sha256(admission.dispatch_bytes).hexdigest()})
                or call_catalog.resolve(admission.attempt_ref)["sandbox_id"] != sandbox_id):
            raise ValueError
        phase = "INPUT"
        _bounded_call(lambda: (sandbox.stdin.write(admission.dispatch_bytes),
                               sandbox.stdin.write_eof(), sandbox.stdin.drain()),
                      deadline=deadline, timeout_code="constructor_input_indeterminate")
        phase = "OUTPUT"
        out, err = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        threading.Thread(target=_read_limited, args=(sandbox.stdout, _OUTPUT_MAX, out), daemon=True).start()
        threading.Thread(target=_read_limited, args=(sandbox.stderr, _OUTPUT_MAX, err), daemon=True).start()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        ok, data = out.get(timeout=remaining)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        err_ok, err_data = err.get(timeout=remaining)
        if not ok or not err_ok or err_data or data is None:
            raise ValueError
        validate_result(data)
        result = data
        phase = "TERMINAL"
        terminal = _terminal_by_id(sdk, client, sandbox_id, deadline=deadline)
        if terminal != 0:
            raise ValueError
        primary = "COMPLETED"
    except BaseException:
        failure_phase = phase
        primary = "INDETERMINATE" if phase in {"CREATE", "INPUT", "OUTPUT", "TERMINAL"} else "REJECTED"
    finally:
        if sandbox is not None and sandbox_id is not None:
            cleanup = _cleanup_owned(sdk, client, sandbox, expected_id=sandbox_id,
                                      deadline=time.monotonic() + 30)
        receipt = _receipt(runtime, admission, sandbox_id=sandbox_id,
                           primary=primary, cleanup=cleanup, result=result, terminal=terminal,
                           failure_phase=failure_phase)
        phase = "RECEIPT"
        catalog = _catalog(private_storage, "modal-grpo-constructor-results")
        if (not catalog.publish_if_absent(admission.attempt_ref, receipt)
                or catalog.resolve(admission.attempt_ref) != receipt):
            raise ModalGRPOConstructorUnavailable(phase)
    return read_modal_grpo_constructor_receipt(runtime, private_storage, admission.attempt_ref)
