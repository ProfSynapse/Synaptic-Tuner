"""Private, review-gated GRPO constructor diagnostic consumer.

Each effect phase consumes a separately reviewed canonical record. This is
not a TrainingAPI operation and never invokes the training Function.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

ENGINE = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ENGINE))

from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.cloud.grpo_constructor_diagnostic import derive_probe_input
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.packaged_binding import ModalPackagedRuntimeFactsV1
from tuner.execution.providers.modal.runtime_release_deployment import ModalRuntimeReleaseDeploymentFactsV1
from tuner.project.context import ProjectContext
from tuner.runtime.packaged_grpo_constructor_child import pinned_library_sources, validate_model_inventory
from tuner.runtime.offline_grpo_worker import load_packaged_offline_grpo_worker_manifest
from tuner.runtime.packaged_grpo_worker_closure import load_packaged_grpo_worker_closure
from tuner.runtime.packaged_worker_closure import stable_read
from tuner.runtime.releases import parse_packaged_runtime_release
from tuner.training.contracts import ResourceSpec
from tuner.training.modal_grpo_constructor_diagnostic import (
    admit_modal_grpo_constructor, execute_modal_grpo_constructor_once,
    prepare_pinned_model_manifest, read_modal_grpo_constructor_receipt, _source_pair,
)
from tuner.training.modal_host_runtime import (
    ModalHostRuntimeV1, ModalRuntimeQuoteV1, _CurrentPackagedDeploymentReader,
)
from tuner.training.modal_host_storage import ModalHostStorageV1
from tuner.training.modal_local_preparation import private_directory, validate_local_launcher
from tuner.training.modal_recipe import ModalGRPORecipePlanV1, plan_modal_recipe
from tuner.training.modal_run_retention import load_retained_modal_run
from tuner.training.modal_retained_reader import _typed_inputs
from tuner.training.modal_standalone_runner import prepare_modal_standalone_local, _qualified_runtime

_SCHEMA = "synaptic-private-grpo-constructor-consumer/v1"
_FIELDS = frozenset({"schema_version", "engine_root", "host_root", "project_root", "recipe_path",
    "plan_ready_path", "dataset_path", "profiles_root", "private_root", "retained_root",
    "retained_run", "modal_profile", "modal_environment", "hf_token_file", "model_ref", "model_revision",
    "cpu_attempt_ref", "diagnostic_attempt_ref"})
_HEX = frozenset("0123456789abcdef")
_MAX_CAPTURE = 2 * 1024 * 1024


class ConsumerUnavailable(RuntimeError):
    def __init__(self, phase):
        self.phase = phase
        super().__init__("grpo_constructor_consumer_unavailable")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ConsumerUnavailable("ARGUMENTS")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit):
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError
    return stable_read(path, limit)


def _document(path, limit=_MAX_CAPTURE):
    raw = _read(path, limit)
    value = json.loads(raw)
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise ValueError
    return value, _hash(raw)


def _write(path, value):
    raw = canonical_bytes(value)
    if len(raw) > _MAX_CAPTURE:
        raise ValueError
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return _hash(raw)


def _config(path):
    value, digest = _document(path, 16384)
    if set(value) != _FIELDS or value["schema_version"] != _SCHEMA:
        raise ValueError
    for name in _FIELDS - {"schema_version", "retained_run", "modal_profile", "modal_environment",
                           "model_ref", "model_revision", "cpu_attempt_ref", "diagnostic_attempt_ref"}:
        selected = value[name]
        if type(selected) is not str or not Path(selected).is_absolute():
            raise ValueError
        value[name] = Path(selected)
    run = TrainingRunRef.from_dict(value["retained_run"])
    value["retained_run"] = run
    for name in ("modal_profile", "modal_environment", "model_ref", "model_revision"):
        if type(value[name]) is not str or not value[name]:
            raise ValueError
    if (type(value["cpu_attempt_ref"]) is not str
            or not value["cpu_attempt_ref"].startswith("cpu-qual-")
            or len(value["cpu_attempt_ref"]) != len("cpu-qual-") + 24
            or any(char not in _HEX for char in value["cpu_attempt_ref"][len("cpu-qual-"):])
            or type(value["diagnostic_attempt_ref"]) is not str
            or not value["diagnostic_attempt_ref"].startswith("grpo-diag-")
            or len(value["diagnostic_attempt_ref"]) != len("grpo-diag-") + 24
            or any(char not in _HEX for char in value["diagnostic_attempt_ref"][len("grpo-diag-"):])):
        raise ValueError
    return value, digest


def _paths(config):
    return config["private_root"]


def _hf_token(config):
    raw = _read(config["hf_token_file"], 4096)
    if (not 1 <= len(raw) <= 512 or not raw.startswith(b"hf_")
            or any(byte not in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
                   for byte in raw)):
        raise ValueError
    return raw.decode("ascii")


def _plan(config, config_digest):
    engine, host = _source_pair(config["engine_root"], config["host_root"])
    plan = plan_modal_recipe(config["recipe_path"], project_root=config["project_root"],
        profiles_root=config["profiles_root"], source_root=config["engine_root"])
    if (type(plan) is not ModalGRPORecipePlanV1
            or plan.recipe.model.ref != config["model_ref"]
            or plan.recipe.model.revision != config["model_revision"]):
        raise ValueError
    resource = ResourceSpec(plan.recipe.accelerator, plan.recipe.accelerator_count,
                            plan.recipe.timeout_seconds)
    if (resource.accelerator != "L40S" or resource.accelerator_count != 1
            or resource.timeout_seconds != 120 or plan.recipe.maximum_cost_minor_units is None
            or plan.recipe.maximum_cost_minor_units > 100):
        raise ValueError
    probe = derive_probe_input(recipe_path=config["recipe_path"],
        plan_ready_path=config["plan_ready_path"], dataset_path=config["dataset_path"],
        profiles_root=config["profiles_root"])
    retained = load_retained_modal_run(config["retained_root"], expected_run=config["retained_run"])
    old_inputs, old_release, old_deployment, old_facts, old_scope, *_ = _typed_inputs(retained)
    old_model = old_facts.model_cache_volume_id
    if (old_scope.environment_ref != config["modal_environment"]
            or old_scope.account_ref != "synaptic-labs"
            or old_scope.workspace_ref != "synaptic-labs"
            or old_model not in dict(tuple(tuple(pair) for pair in old_inputs["volume_names"]))):
        raise ValueError
    candidate = {"schema_version": "synaptic-private-grpo-constructor-build-admission/v1",
        "config_sha256": config_digest, "engine_commit": engine, "host_commit": host,
        "offline_grpo_closure_digest": load_packaged_offline_grpo_worker_manifest().closure.closure_digest,
        "outer_grpo_closure_digest": load_packaged_grpo_worker_closure().digest,
        "modal_runtime_lock_sha256": _hash(stable_read(config["engine_root"] /
            "tuner/execution/providers/modal/modal-runtime-v1.lock.json", 2 * 1024 * 1024)),
        "python_version": "3.11.14", "modal_sdk_version": "1.5.4",
        "recipe_plan_sha256": _hash(canonical_bytes(plan.to_dict())),
        "probe_sha256": _hash(probe), "library_sources_sha256": _hash(canonical_bytes(pinned_library_sources())),
        "profile_sha256": plan.profile_sha256, "runtime_intent_digest": plan.runtime_material_intent_digest,
        "base_image": plan.profile_base_image,
        "resource": {"accelerator": "L40S", "accelerator_count": 1, "timeout_seconds": 120},
        "maximum_cost_minor_units": plan.recipe.maximum_cost_minor_units,
        "old_release_digest": old_release.manifest_digest,
        "old_deployment_facts_digest": old_deployment.facts_digest,
        "old_image_id": old_facts.image_id, "old_model_volume_id": old_model,
        "old_model_volume_name": dict(tuple(tuple(pair) for pair in old_inputs["volume_names"]))[old_model],
        "model_ref": config["model_ref"], "model_revision": config["model_revision"],
        "cpu_attempt_ref": config["cpu_attempt_ref"],
        "diagnostic_attempt_ref": config["diagnostic_attempt_ref"],
        "account_ref": old_scope.account_ref, "workspace_ref": old_scope.workspace_ref,
        "environment_ref": old_scope.environment_ref}
    return candidate, plan, retained


def _checked(path, expected):
    _, digest = _document(path)
    if type(expected) is not str or len(expected) != 64 or digest != expected:
        raise ValueError
    return digest


def _quote(value):
    if type(value) is not dict or set(value) != {"schema_version", "rate_key", "gpu_hourly_usd",
            "gpu_only_timeout_estimate_minor_units", "maximum_cost_minor_units", "resource",
            "excluded_billing_dimensions", "authorization_semantics", "quote_digest"}:
        raise ValueError
    if value["schema_version"] != "synaptic-modal-runtime-quote/v1":
        raise ValueError
    statement = {key: item for key, item in value.items() if key != "quote_digest"}
    if _hash(canonical_bytes(statement)) != value["quote_digest"]:
        raise ValueError
    return ModalRuntimeQuoteV1(value["rate_key"], value["gpu_hourly_usd"],
        value["gpu_only_timeout_estimate_minor_units"], value["maximum_cost_minor_units"],
        ResourceSpec(**value["resource"]), value["quote_digest"],
        tuple(value["excluded_billing_dimensions"]), value["authorization_semantics"])


def _runtime(value, *, sdk, client, quote):
    release, deployment, facts, volumes, key = _runtime_components(value)
    binding = deployment.client_binding
    reader = _CurrentPackagedDeploymentReader(sdk=sdk, client=client,
        binding=binding, facts=facts,
        names={item.spec.role: item.spec.name for item in deployment.volumes},
        secret_ids=tuple(sorted((item.spec.name, item.secret_id, item.spec.required_keys)
            for item in deployment.secrets)))
    return ModalHostRuntimeV1(release, deployment, facts, quote, volumes, key, reader)


def _runtime_components(value):
    release = parse_packaged_runtime_release(value["release"])
    deployment = ModalRuntimeReleaseDeploymentFactsV1.from_dict(value["deployment_facts"])
    facts = ModalPackagedRuntimeFactsV1.from_dict(value["facts"])
    binding = deployment.client_binding
    if (deployment.runtime_release_digest != release.manifest_digest
            or ModalPackagedRuntimeFactsV1.from_release_deployment(deployment) != facts
            or facts.client_binding != binding or facts.account_ref != "synaptic-labs"
            or facts.workspace_ref != "synaptic-labs" or facts.environment_ref != "synaptic-smoke-v1"):
        raise ValueError
    volumes = tuple(tuple(pair) for pair in value["volume_names_by_id"])
    if dict(volumes) != {item.volume_id: item.spec.name for item in deployment.volumes}:
        raise ValueError
    key = bytes.fromhex(value["qualification_key_hex"])
    if len(key) != 32:
        raise ValueError
    return release, deployment, facts, volumes, key


def _readback(config, config_digest, root, reviewed, reviewed_capture, reviewed_model):
    build, build_sha = _document(root / "build-admission.json")
    if (build.get("schema_version") != "synaptic-private-grpo-constructor-build-admission/v1"
            or build.get("config_sha256") != config_digest
            or build.get("diagnostic_attempt_ref") != config["diagnostic_attempt_ref"]
            or build.get("cpu_attempt_ref") != config["cpu_attempt_ref"]
            or build.get("model_ref") != config["model_ref"]
            or build.get("model_revision") != config["model_revision"]
            or build.get("environment_ref") != config["modal_environment"]):
        raise ValueError
    _checked(root / "runtime-capture.json", reviewed_capture)
    capture, capture_sha = _document(root / "runtime-capture.json")
    if (capture.get("schema_version") != "synaptic-private-grpo-constructor-runtime-capture/v1"
            or capture.get("build_admission_sha256") != build_sha
            or capture.get("cpu_qualification", {}).get("training_executed") is not False):
        raise ValueError
    _quote(capture["quote"])
    release, deployment, facts, volumes, key = _runtime_components(capture["runtime"])
    if (release.package_digest != capture.get("wheel_sha256")
            or build.get("account_ref") != facts.account_ref
            or build.get("workspace_ref") != facts.workspace_ref
            or build.get("environment_ref") != facts.environment_ref):
        raise ValueError
    _checked(root / "execution-admission.json", reviewed)
    admission, _ = _document(root / "execution-admission.json")
    model, model_sha = _document(root / "model-manifest.json")
    validate_model_inventory(model)
    if reviewed_model != model_sha:
        raise ValueError
    expected = {"schema_version": "synaptic-private-grpo-constructor-execution-admission/v1",
        "capture_sha256": capture_sha, "model_manifest_sha256": model_sha,
        "attempt_ref": config["diagnostic_attempt_ref"],
        "source_commit": build["engine_commit"], "host_source_commit": build["host_commit"],
        "wheel_sha256": capture["wheel_sha256"],
        "runtime_release_digest": release.manifest_digest,
        "deployment_facts_digest": deployment.facts_digest, "image_id": deployment.image_id,
        "snapshot_deployment_facts_digest": build["old_deployment_facts_digest"],
        "old_model_volume_id": build["old_model_volume_id"],
        "old_model_volume_name": build["old_model_volume_name"],
        "quote_digest": capture["quote"]["quote_digest"],
        "account_ref": "synaptic-labs", "workspace_ref": "synaptic-labs",
        "environment_ref": config["modal_environment"],
        "gpu": "L40S", "gpu_count": 1, "cpu": 2, "memory_mib": 8192,
        "provider_timeout_seconds": 120, "host_deadline_seconds": 240,
        "cleanup_seconds": 30, "result_limit_bytes": 8192, "block_network": True,
        "secrets": [], "training_executed": False}
    if (set(admission) != set(expected) | {"dispatch_sha256"}
            or any(admission.get(name) != value for name, value in expected.items())
            or type(admission.get("dispatch_sha256")) is not str
            or len(admission["dispatch_sha256"]) != 64
            or any(char not in _HEX for char in admission["dispatch_sha256"])):
        raise ValueError
    runtime = SimpleNamespace(release=release, deployment_facts=deployment,
                              qualification_key=key)
    with ModalHostStorageV1(root / "journal.sqlite3", "grpo-constructor-diagnostic", create=False) as storage:
        receipt = read_modal_grpo_constructor_receipt(runtime, storage, config["diagnostic_attempt_ref"])
    body = receipt["body"]
    bindings = {"dispatch_sha256": "dispatch_sha256", "source_commit": "source_commit",
        "host_source_commit": "host_source_commit", "wheel_sha256": "wheel_sha256",
        "runtime_release_digest": "runtime_release_digest",
        "deployment_facts_digest": "deployment_facts_digest", "image_id": "image_id",
        "snapshot_deployment_facts_digest": "snapshot_deployment_facts_digest",
        "model_cache_volume_id": "old_model_volume_id", "quote_digest": "quote_digest"}
    if any(body.get(receipt_key) != admission[record_key]
           for receipt_key, record_key in bindings.items()):
        raise ValueError
    return _receipt_summary(receipt)


def _client(config):
    import modal
    if modal.__version__ != "1.5.4":
        raise ValueError
    from modal.config import config as modal_config
    token_id = modal_config.get("token_id", profile=config["modal_profile"], use_env=False)
    token_secret = modal_config.get("token_secret", profile=config["modal_profile"], use_env=False)
    if any(type(value) is not str or not value.strip() for value in (token_id, token_secret)):
        raise ValueError
    return modal, modal.Client.from_credentials(token_id, token_secret)


def _read_capture(config, reviewed_capture, *, sdk, client):
    path = _paths(config) / "runtime-capture.json"
    _checked(path, reviewed_capture)
    capture, _ = _document(path)
    quote = _quote(capture["quote"])
    runtime = _runtime(capture["runtime"], sdk=sdk, client=client, quote=quote)
    if capture["cpu_qualification"]["training_executed"] is not False:
        raise ValueError
    retained = load_retained_modal_run(config["retained_root"], expected_run=config["retained_run"])
    inputs, old_release, old_deployment, old_facts, old_scope, *_ = _typed_inputs(retained)
    if (old_scope != runtime.deployment_facts.client_binding
            or config["modal_environment"] != old_scope.environment_ref):
        raise ValueError
    snapshot = _runtime({"release": old_release.to_dict(),
        "deployment_facts": old_deployment.to_dict(), "facts": old_facts.to_dict(),
        "volume_names_by_id": inputs["volume_names"],
        "qualification_key_hex": retained.qualification_key.hex()},
        sdk=sdk, client=client, quote=quote)
    return capture, runtime, snapshot


def _admission(config, capture, runtime, snapshot, model):
    return admit_modal_grpo_constructor(runtime=runtime, snapshot_runtime=snapshot,
        engine_repository=config["engine_root"], host_repository=config["host_root"],
        wheel_path=_paths(config) / "source-wheel.whl", recipe_path=config["recipe_path"],
        plan_ready_path=config["plan_ready_path"], dataset_path=config["dataset_path"],
        profiles_root=config["profiles_root"],
        input_bytes=derive_probe_input(recipe_path=config["recipe_path"],
            plan_ready_path=config["plan_ready_path"], dataset_path=config["dataset_path"],
            profiles_root=config["profiles_root"]), model=model,
        library_sources=pinned_library_sources(), attempt_ref=config["diagnostic_attempt_ref"])


def _admission_record(config, admission, capture_sha, model_sha):
    return {"schema_version": "synaptic-private-grpo-constructor-execution-admission/v1",
        "capture_sha256": capture_sha, "model_manifest_sha256": model_sha,
        "attempt_ref": admission.attempt_ref, "source_commit": admission.source_commit,
        "host_source_commit": admission.host_source_commit, "wheel_sha256": admission.wheel_sha256,
        "runtime_release_digest": admission.runtime_release_digest,
        "deployment_facts_digest": admission.deployment_facts_digest,
        "image_id": admission.image_id,
        "snapshot_deployment_facts_digest": admission.snapshot_deployment_facts_digest,
        "old_model_volume_id": admission.model_cache_volume_id,
        "old_model_volume_name": admission.model_cache_volume_name,
        "quote_digest": admission.quote_digest, "dispatch_sha256": _hash(admission.dispatch_bytes),
        "account_ref": "synaptic-labs", "workspace_ref": "synaptic-labs",
        "environment_ref": config["modal_environment"],
        "gpu": "L40S", "gpu_count": 1, "cpu": 2, "memory_mib": 8192,
        "provider_timeout_seconds": 120, "host_deadline_seconds": 240,
        "cleanup_seconds": 30, "result_limit_bytes": 8192, "block_network": True,
        "secrets": [], "training_executed": False}


def _receipt_summary(receipt):
    body = receipt["body"]
    return {"sha256": _hash(canonical_bytes(receipt)), "primary": body["primary"],
        "cleanup": body["cleanup"], "failure_phase": body["failure_phase"],
        "diagnostic": body["result"]}


def run(command, config_path, reviewed=None, reviewed_capture=None, reviewed_model=None):
    config, config_digest = _config(config_path)
    root = private_directory(_paths(config), create=command in {"plan", "prepare-model"})
    if command == "readback":
        return _readback(config, config_digest, root, reviewed, reviewed_capture, reviewed_model)
    if command == "prepare-model":
        model = prepare_pinned_model_manifest(config["model_ref"], config["model_revision"])
        return _write(root / "model-manifest.json", model)
    candidate, plan, _ = _plan(config, config_digest)
    if command == "plan":
        return _write(root / "build-admission.json", candidate)
    build_record, build_digest = _document(root / "build-admission.json")
    if build_record != candidate or (command == "prepare" and build_digest != reviewed):
        raise ValueError
    if command == "prepare":
        if _document(root / "build-admission.json")[0] != candidate:
            raise ValueError
        validate_local_launcher()
        context = ProjectContext.host(engine_root=config["engine_root"], project_root=config["project_root"])
        local = prepare_modal_standalone_local(plan=plan, context=context, state_root=root)
        try:
            wheel_bytes = stable_read(local.wheel, 256 * 1024 * 1024)
            hf_token = _hf_token(config)
            with ModalHostStorageV1(root / "journal.sqlite3", "grpo-constructor-diagnostic") as storage:
                storage.attempts.claim("build-admission-" + config["cpu_attempt_ref"],
                    canonical_bytes(candidate))
                modal, _ = _client(config)
                _, _, _, runtime, qualification = _qualified_runtime(
                    modal=modal, plan=plan, context=context,
                    resources=ResourceSpec("L40S", 1, 120), storage=storage,
                    effect_id=config["cpu_attempt_ref"], modal_profile=config["modal_profile"],
                    modal_environment=config["modal_environment"],
                    builder_cache_root=root / "modal-wheel-builder", local_inputs=local,
                    hf_token=hf_token)
            if _hash(wheel_bytes) != runtime.release.package_digest:
                raise ValueError
            wheel_path = root / "source-wheel.whl"
            fd = os.open(wheel_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                with os.fdopen(fd, "wb", closefd=False) as stream:
                    stream.write(wheel_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(fd)
            capture = {"schema_version": "synaptic-private-grpo-constructor-runtime-capture/v1",
                "build_admission_sha256": reviewed,
                "cpu_qualification": {"effect_id": config["cpu_attempt_ref"],
                    "output_sha256": qualification.output_sha256,
                    "provider_call_id": qualification.provider_call_id,
                    "training_executed": False, "gpu_qualified": False,
                    "receipt": qualification.receipt.to_dict()},
                "quote": runtime.quote.to_dict(), "runtime": {
                    "release": runtime.release.to_dict(),
                    "deployment_facts": runtime.deployment_facts.to_dict(),
                    "facts": runtime.facts.to_dict(),
                    "volume_names_by_id": [list(pair) for pair in runtime.volume_names_by_id],
                    "qualification_key_hex": runtime.qualification_key.hex()},
                "wheel_sha256": _hash(wheel_bytes)}
            return _write(root / "runtime-capture.json", capture)
        finally:
            local.close()
    _checked(root / "runtime-capture.json", reviewed_capture)
    if command == "execute-once":
        _checked(root / "execution-admission.json", reviewed)
    model, model_sha = _document(root / "model-manifest.json")
    validate_model_inventory(model)
    if reviewed_model != model_sha:
        raise ValueError
    modal, client = _client(config)
    capture, runtime, snapshot = _read_capture(config, reviewed_capture, sdk=modal, client=client)
    if (capture["build_admission_sha256"] != _document(root / "build-admission.json")[1]
            or capture["wheel_sha256"] != _hash(_read(root / "source-wheel.whl", 256 * 1024 * 1024))):
        raise ValueError
    admission = _admission(config, capture, runtime, snapshot, model)
    record = _admission_record(config, admission, reviewed_capture, model_sha)
    if command == "admit":
        return _write(root / "execution-admission.json", record)
    if command != "execute-once":
        raise ValueError
    if _document(root / "execution-admission.json")[0] != record:
        raise ValueError
    with ModalHostStorageV1(root / "journal.sqlite3", "grpo-constructor-diagnostic", create=False) as storage:
        receipt = execute_modal_grpo_constructor_once(sdk=modal, client=client,
            client_binding=runtime.deployment_facts.client_binding,
            runtime=runtime, snapshot_runtime=snapshot, private_storage=storage,
            admission=admission, engine_repository=config["engine_root"],
            host_repository=config["host_root"], wheel_path=root / "source-wheel.whl",
            recipe_path=config["recipe_path"], plan_ready_path=config["plan_ready_path"],
            dataset_path=config["dataset_path"], profiles_root=config["profiles_root"])
    return _receipt_summary(receipt)


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("command", choices=("plan", "prepare-model", "prepare", "admit", "execute-once", "readback"))
    parser.add_argument("--reviewed-sha256")
    parser.add_argument("--reviewed-capture-sha256")
    parser.add_argument("--reviewed-model-sha256")
    try:
        args = parser.parse_args(argv)
        result = run(args.command, args.config, args.reviewed_sha256,
                     args.reviewed_capture_sha256, args.reviewed_model_sha256)
        print(json.dumps({"status": "RECORDED", **(result if type(result) is dict else
            {"sha256": result})}, sort_keys=True))
        return 0
    except SystemExit as error:
        if error.code == 0:
            return 0
        print(json.dumps({"status": "UNAVAILABLE", "phase": "ARGUMENTS"}, sort_keys=True))
        return 1
    except BaseException as error:
        phase = error.phase if type(error) is ConsumerUnavailable else "UNAVAILABLE"
        print(json.dumps({"status": "UNAVAILABLE", "phase": phase}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
