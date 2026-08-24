from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from tuner.cloud.hf_run_approval import validate_hf_run_approval
from tuner.cloud.hf_training_smoke_contract import (
    ARTIFACT_SLOT_INPUT_SCHEMA,
    RUNTIME_PYTHON_IMPLEMENTATION,
    RUNTIME_PYTHON_VERSION,
    document_sha256,
    derive_hf_training_artifact_prefix,
    derive_hf_training_artifact_slot,
    validate_approval,
    validate_cancellation_event,
    validate_observation_event,
    validate_preflight,
    validate_result,
    validate_runtime_lock,
    seal_training_document,
    validate_submission_event,
    validate_training_document,
)
from tuner.core.exceptions import CloudProviderError


SHA = "a" * 64
SHA2 = "b" * 64
GIT = "c" * 40
TS = "2026-08-20T12:00:00Z"
CHILD_MEDIA = "application/vnd.docker.distribution.manifest.v2+json"


def _raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def image_evidence(kind: str = "manifest") -> tuple[dict, bytes, bytes, bytes]:
    config_raw = _raw(
        {
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": [f"sha256:{'5' * 64}"]},
        }
    )
    layer = {
        "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
        "digest": f"sha256:{'4' * 64}",
        "size": 200,
    }
    child_raw = _raw(
        {
            "schemaVersion": 2,
            "mediaType": CHILD_MEDIA,
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": _digest(config_raw),
                "size": len(config_raw),
            },
            "layers": [layer],
        }
    )
    child_digest = _digest(child_raw)
    if kind == "index":
        requested_raw = _raw(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
                "manifests": [
                    {
                        "mediaType": CHILD_MEDIA,
                        "digest": child_digest,
                        "size": len(child_raw),
                        "platform": {"os": "linux", "architecture": "amd64"},
                    }
                ],
            }
        )
        index_digest: str | None = _digest(requested_raw)
        index_media_type: str | None = "application/vnd.docker.distribution.manifest.list.v2+json"
    else:
        requested_raw = child_raw
        index_digest = None
        index_media_type = None
    image = {
        "registry_repository": "docker.io/unsloth/unsloth",
        "provider_repository": "unsloth/unsloth",
        "requested_digest": _digest(requested_raw),
        "requested_media_type": index_media_type or CHILD_MEDIA,
        "requested_kind": kind,
        "index_digest": index_digest,
        "index_media_type": index_media_type,
        "child_digest": child_digest,
        "child_media_type": CHILD_MEDIA,
        "config_digest": _digest(config_raw),
        "config_media_type": "application/vnd.docker.container.image.v1+json",
        "config_size": len(config_raw),
        "platform": "linux/amd64",
        "layers": [
            {"media_type": layer["mediaType"], "digest": layer["digest"], "size": layer["size"]}
        ],
        "provider_reference": f"unsloth/unsloth@{child_digest}",
    }
    return image, requested_raw, child_raw, config_raw


def _seal(document: dict, field: str) -> dict:
    document[field] = "0" * 64
    document[field] = document_sha256({key: value for key, value in document.items() if key != field})
    return document


def runtime_lock(kind: str = "manifest", *, image: dict | None = None) -> dict:
    image = image or image_evidence(kind)[0]
    return _seal(
        {
            "schema_version": "synaptic-hf-training-runtime-lock/v1",
            "created_at": TS,
            "image": image,
            "runtime": {
                "python_implementation": RUNTIME_PYTHON_IMPLEMENTATION,
                "python": RUNTIME_PYTHON_VERSION,
                "packages": {"torch": "2.9.0"},
                "signatures": {"model_loader": "revision,token"},
            },
            "anonymous_loading": {
                "token": False,
                "trust_remote_code": False,
                "use_safetensors": True,
            },
        },
        "lock_id",
    )


def preflight() -> dict:
    image, _, _, _ = image_evidence()
    slot_input = {
        "schema_version": ARTIFACT_SLOT_INPUT_SCHEMA,
        "experiment_id": "exp-1", "run_id": "run-1", "tracking_root_id": SHA,
        "source_lock_sha256": SHA2, "workload_digest": SHA,
        "runtime_lock_sha256": SHA2, "artifact_bucket_id": "owner/bucket",
        "artifact_base_prefix": "training/artifacts",
    }
    slot_id = derive_hf_training_artifact_slot(slot_input)
    return _seal(
        {
            "schema_version": "synaptic-hf-training-preflight/v1",
            "experiment_id": "exp-1",
            "run_id": "run-1",
            "tracking_root_id": SHA,
            "occurred_at": TS,
            "status": "PASS",
            "source": {
                "descriptor": {"uri": "tracking://descriptor.json", "sha256": SHA},
                "source_lock": {"uri": "tracking://source-lock.json", "sha256": SHA2},
                "provisioning_evidence": {"uri": "tracking://evidence.json", "sha256": SHA},
                "bundle_sha256": SHA2, "capsule_manifest_sha256": SHA,
                "checkout_policy_sha256": SHA2, "project_commit": GIT, "engine_commit": "d" * 40,
            },
            "runtime_lock": {"uri": "tracking://runtime-lock.json", "sha256": SHA2},
            "workload_digest": SHA,
            "model": {"repository": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": GIT},
            "dataset": {"path": "Datasets/smoke.jsonl", "sha256": SHA, "git_blob": GIT, "bytes": 10, "row_count": 1, "row_sha256": SHA2},
            "image": image,
            "hardware": {"endpoint": "https://huggingface.co", "flavor": "a10g-small", "unit_cost_micro_usd": 16000, "unit_label": "minute", "hourly_cost_micro_usd": 960000, "timeout_cost_micro_usd": 480000, "fetched_at": TS},
            "artifact_slot_input": slot_input, "artifact_slot_id": slot_id,
            "volumes": [
                {"bucket_id": "owner/bucket", "prefix": "source/capsule", "mount_path": "/workspace/synaptic-bootstrap-input", "read_only": True},
                {"bucket_id": "owner/bucket", "prefix": derive_hf_training_artifact_prefix("training/artifacts", slot_id), "mount_path": "/workspace/artifacts", "read_only": False},
            ],
            "command": {"remote_argv_sha256": SHA, "provider_command_sha256": SHA2},
            "launcher_auth": {"mode": "explicit_file", "expected_namespace": "owner"},
            "job_secrets": [],
        },
        "preflight_id",
    )


def approval(accepted_preflight: dict) -> dict:
    return _seal(
        {
            "schema_version": "synaptic-hf-training-approval/v1",
            "kind": "hf.training-smoke",
            "experiment_id": "exp-1",
            "run_id": "run-1",
            "tracking_root_id": SHA,
            "preflight": {"uri": "tracking://preflight.json", "sha256": document_sha256(accepted_preflight)},
            "user_authorization_reference": "conversation-2026-08-20-training-smoke",
            "issued_at": TS,
            "expires_at": "2026-08-20T13:00:00Z",
            "hardware": "a10g-small",
            "hardware_quote": {"preflight_sha256": document_sha256(accepted_preflight), "unit_cost_micro_usd": 16000, "hourly_cost_micro_usd": 960000, "timeout_cost_micro_usd": 480000, "fetched_at": TS},
            "provider_timeout_seconds": 1800,
            "cancel_after_seconds": 1500,
            "observe_until_seconds": 2100,
            "maximum_submissions": 1,
            "maximum_retries": 0,
            "publication": False,
            "ssh": False,
            "ports": False,
            "wandb": False,
            "launcher_auth": {"mode": "explicit_file", "expected_namespace": "owner"},
            "job_secrets": [],
            "bindings": {
                "source_lock_sha256": SHA2, "workload_digest": SHA, "runtime_lock_sha256": SHA2,
                "model_revision": GIT, "dataset_sha256": SHA,
                "image_child_digest": accepted_preflight["image"]["child_digest"], "remote_argv_sha256": SHA,
                "provider_command_sha256": SHA2, "source_bucket_id": "owner/bucket",
                "source_prefix": "source/capsule", "artifact_bucket_id": "owner/bucket",
                "artifact_base_prefix": "training/artifacts",
                "artifact_prefix": accepted_preflight["volumes"][1]["prefix"],
                "artifact_slot_id": accepted_preflight["artifact_slot_id"],
            },
        },
        "authorization_id",
    )


def event_base(schema: str, accepted_approval: dict) -> dict:
    return {
        "schema_version": schema,
        "authorization_id": accepted_approval["authorization_id"],
        "approval": {"uri": "tracking://approval.json", "sha256": document_sha256(accepted_approval)},
        "experiment_id": "exp-1",
        "run_id": "run-1",
        "tracking_root_id": SHA,
        "occurred_at": TS,
    }


def test_closed_runtime_preflight_and_approval_contracts_do_not_mutate_inputs():
    lock = runtime_lock()
    pf = preflight()
    auth = approval(pf)
    originals = deepcopy((lock, pf, auth))
    assert validate_runtime_lock(lock)["lock_id"] == lock["lock_id"]
    assert validate_preflight(pf)["status"] == "PASS"
    assert validate_approval(auth, preflight=pf)["maximum_retries"] == 0
    unsealed = {key: value for key, value in lock.items() if key != "lock_id"}
    assert seal_training_document(unsealed) == lock
    assert "lock_id" not in unsealed
    assert (lock, pf, auth) == originals

    hostile = deepcopy(pf)
    hostile["unexpected"] = True
    with pytest.raises(CloudProviderError):
        validate_preflight(hostile)
    assert hostile["unexpected"] is True


def test_artifact_slot_is_domain_separated_closed_and_prefix_bound():
    pf = preflight()
    slot_input = pf["artifact_slot_input"]
    expected = hashlib.sha256(
        b"synaptic-hf-training-artifact-slot/v1\x00"
        + (json.dumps(slot_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    assert derive_hf_training_artifact_slot(slot_input) == expected == pf["artifact_slot_id"]
    assert pf["volumes"][1]["prefix"] == f"training/artifacts/{expected}"

    for mutation in ("extra", "identity", "base", "volume"):
        hostile = deepcopy(pf)
        if mutation == "extra":
            hostile["artifact_slot_input"]["unexpected"] = True
        elif mutation == "identity":
            hostile["artifact_slot_input"]["run_id"] = "run-2"
        elif mutation == "base":
            hostile["artifact_slot_input"]["artifact_base_prefix"] = "training/other"
        else:
            hostile["volumes"][1]["prefix"] = "training/artifacts"
        hostile = _seal(hostile, "preflight_id")
        with pytest.raises(CloudProviderError):
            validate_preflight(hostile)

    separate_buckets = deepcopy(pf)
    separate_buckets["volumes"][0]["bucket_id"] = "source-owner/source-bucket"
    separate_buckets = _seal(separate_buckets, "preflight_id")
    assert validate_preflight(separate_buckets)["volumes"][0]["bucket_id"] == "source-owner/source-bucket"


@pytest.mark.parametrize("mutation", ["bool", "arithmetic", "stale", "future", "binding"])
def test_hardware_quote_is_integer_exact_fresh_ordered_and_preflight_bound(mutation: str):
    pf = preflight()
    auth = approval(pf)
    if mutation == "bool":
        auth["hardware_quote"]["unit_cost_micro_usd"] = True
    elif mutation == "arithmetic":
        auth["hardware_quote"]["hourly_cost_micro_usd"] -= 1
    elif mutation == "stale":
        auth["issued_at"] = "2026-08-20T12:15:01Z"
    elif mutation == "future":
        auth["hardware_quote"]["fetched_at"] = "2026-08-20T12:00:01Z"
    else:
        auth["hardware_quote"]["preflight_sha256"] = "f" * 64
    auth = _seal(auth, "authorization_id")
    with pytest.raises(CloudProviderError):
        validate_approval(auth, preflight=pf)


def test_live_a10g_quote_rounding_is_accepted_exactly() -> None:
    pf = preflight()
    pf["hardware"] = {
        **pf["hardware"],
        "unit_cost_micro_usd": 16_667,
        "hourly_cost_micro_usd": 1_000_020,
        "timeout_cost_micro_usd": 500_010,
    }
    pf = _seal(pf, "preflight_id")
    assert validate_preflight(pf)["hardware"] == pf["hardware"]

    auth = approval(pf)
    auth["hardware_quote"] = {
        "preflight_sha256": document_sha256(pf),
        "unit_cost_micro_usd": 16_667,
        "hourly_cost_micro_usd": 1_000_020,
        "timeout_cost_micro_usd": 500_010,
        "fetched_at": TS,
    }
    auth = _seal(auth, "authorization_id")
    assert validate_approval(auth, preflight=pf)["hardware_quote"] == auth["hardware_quote"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python_implementation", "PyPy"),
        ("python", "3.11.13"),
    ],
)
def test_runtime_lock_child_runtime_identity_is_content_addressed(
    field: str, value: str,
):
    lock = runtime_lock()
    assert lock["runtime"] == {
        "python_implementation": "CPython",
        "python": "3.11.14",
        "packages": {"torch": "2.9.0"},
        "signatures": {"model_loader": "revision,token"},
    }

    hostile = deepcopy(lock)
    hostile["runtime"][field] = value
    unsealed = {key: value for key, value in hostile.items() if key != "lock_id"}
    assert document_sha256(unsealed) != lock["lock_id"]


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("missing_implementation", None),
        ("missing_python", None),
        ("extra", True),
        ("bool_implementation", True),
        ("bool_python", True),
        ("wrong_implementation", "PyPy"),
        ("wrong_python", "3.11.13"),
        ("host_python", "3.12.7"),
    ],
)
def test_runtime_lock_rejects_hostile_child_runtime_identity(mutation: str, value: object):
    hostile = runtime_lock()
    runtime = hostile["runtime"]
    if mutation == "missing_implementation":
        del runtime["python_implementation"]
    elif mutation == "missing_python":
        del runtime["python"]
    elif mutation == "extra":
        runtime["unexpected"] = value
    elif mutation in {"bool_implementation", "wrong_implementation"}:
        runtime["python_implementation"] = value
    else:
        runtime["python"] = value
    hostile = _seal(hostile, "lock_id")
    with pytest.raises(CloudProviderError, match="exact schema"):
        validate_runtime_lock(hostile)


@pytest.mark.parametrize("kind", ["manifest", "index"])
def test_runtime_lock_authenticates_direct_manifest_and_index_chains(kind: str):
    image, requested_raw, child_raw, config_raw = image_evidence(kind)
    lock = runtime_lock(kind, image=image)
    accepted = validate_runtime_lock(
        lock,
        requested_raw=requested_raw,
        child_raw=None if kind == "manifest" else child_raw,
        config_raw=config_raw,
    )
    assert accepted["image"]["requested_kind"] == kind
    assert accepted["image"]["provider_reference"] == f"unsloth/unsloth@{image['child_digest']}"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registry_repository", "registry.example/unsloth/unsloth"),
        ("provider_repository", "docker.io/unsloth/unsloth"),
        ("provider_reference", f"unsloth/unsloth@sha256:{'9' * 64}"),
        ("child_media_type", "application/vnd.oci.image.manifest.v1+json"),
        ("config_size", 1),
    ],
)
def test_runtime_lock_rejects_identity_media_size_and_provider_reference_drift(field: str, value: object):
    image, requested_raw, _, config_raw = image_evidence()
    image[field] = value
    lock = runtime_lock(image=image)
    with pytest.raises(CloudProviderError):
        validate_runtime_lock(lock, requested_raw=requested_raw, config_raw=config_raw)


def test_runtime_lock_rejects_requested_bytes_and_incomplete_evidence():
    image, requested_raw, _, config_raw = image_evidence()
    lock = runtime_lock(image=image)
    with pytest.raises(CloudProviderError, match="requested bytes"):
        validate_runtime_lock(lock, requested_raw=requested_raw + b" ", config_raw=config_raw)
    with pytest.raises(CloudProviderError, match="incomplete"):
        validate_runtime_lock(lock, requested_raw=requested_raw)


def test_direct_oci_manifest_is_rejected_by_frozen_docker_media_contract():
    image, _, _, _ = image_evidence()
    image["requested_media_type"] = "application/vnd.oci.image.manifest.v1+json"
    image["child_media_type"] = "application/vnd.oci.image.manifest.v1+json"
    with pytest.raises(CloudProviderError):
        validate_runtime_lock(runtime_lock(image=image))


def test_requested_document_rejects_duplicate_json_keys():
    image, _, _, config_raw = image_evidence()
    duplicate_raw = b'{"schemaVersion":2,"schemaVersion":2}'
    image["requested_digest"] = _digest(duplicate_raw)
    image["child_digest"] = _digest(duplicate_raw)
    image["provider_reference"] = f"unsloth/unsloth@{image['child_digest']}"
    with pytest.raises(CloudProviderError, match="strict JSON"):
        validate_runtime_lock(
            runtime_lock(image=image), requested_raw=duplicate_raw, config_raw=config_raw
        )


def test_runtime_lock_rejects_config_platform_rootfs_and_layer_drift():
    image, requested_raw, _, config_raw = image_evidence()
    for config_mutation in ("platform", "variant", "rootfs", "diff-count"):
        hostile_config = json.loads(config_raw)
        if config_mutation == "platform":
            hostile_config["architecture"] = "arm64"
        elif config_mutation == "variant":
            hostile_config["variant"] = "v8"
        elif config_mutation == "rootfs":
            hostile_config["rootfs"]["unexpected"] = True
        else:
            hostile_config["rootfs"]["diff_ids"] = []
        hostile_raw = _raw(hostile_config)
        hostile_image = deepcopy(image)
        hostile_image["config_digest"] = _digest(hostile_raw)
        hostile_image["config_size"] = len(hostile_raw)
        child = json.loads(requested_raw)
        child["config"]["digest"] = _digest(hostile_raw)
        child["config"]["size"] = len(hostile_raw)
        hostile_child_raw = _raw(child)
        hostile_image["requested_digest"] = _digest(hostile_child_raw)
        hostile_image["child_digest"] = _digest(hostile_child_raw)
        hostile_image["provider_reference"] = f"unsloth/unsloth@{hostile_image['child_digest']}"
        hostile_lock = runtime_lock(image=hostile_image)
        with pytest.raises(CloudProviderError):
            validate_runtime_lock(
                hostile_lock, requested_raw=hostile_child_raw, config_raw=hostile_raw
            )


def test_runtime_lock_rejects_duplicate_or_reordered_layer_identity():
    image, requested_raw, _, config_raw = image_evidence()
    duplicate = deepcopy(image)
    duplicate["layers"].append(deepcopy(duplicate["layers"][0]))
    with pytest.raises(CloudProviderError):
        validate_runtime_lock(runtime_lock(image=duplicate))

    drifted = deepcopy(image)
    drifted["layers"][0]["size"] += 1
    with pytest.raises(CloudProviderError, match="ordered layer"):
        validate_runtime_lock(
            runtime_lock(image=drifted), requested_raw=requested_raw, config_raw=config_raw
        )


def test_index_requires_exactly_one_closed_linux_amd64_child():
    image, requested_raw, child_raw, config_raw = image_evidence("index")
    index = json.loads(requested_raw)
    index["manifests"].append(deepcopy(index["manifests"][0]))
    hostile_requested = _raw(index)
    image["requested_digest"] = _digest(hostile_requested)
    image["index_digest"] = _digest(hostile_requested)
    with pytest.raises(CloudProviderError, match="exactly one"):
        validate_runtime_lock(
            runtime_lock("index", image=image),
            requested_raw=hostile_requested,
            child_raw=child_raw,
            config_raw=config_raw,
        )


def test_direct_manifest_requires_null_index_fields_and_requested_equals_child():
    image, requested_raw, _, config_raw = image_evidence()
    image["index_digest"] = image["child_digest"]
    with pytest.raises(CloudProviderError):
        validate_runtime_lock(runtime_lock(image=image))

    image, requested_raw, _, config_raw = image_evidence()
    image["requested_digest"] = f"sha256:{'8' * 64}"
    with pytest.raises(CloudProviderError, match="direct-manifest"):
        validate_runtime_lock(runtime_lock(image=image))


def test_bootstrap_and_training_documents_cross_reject():
    pf = preflight()
    with pytest.raises(CloudProviderError, match="Bootstrap"):
        validate_training_document({"schema_version": "synaptic-hf-run-approval/v1"})
    with pytest.raises(Exception):
        validate_hf_run_approval(pf)


def test_submission_and_cancellation_transitions_bind_predecessors_and_effects():
    auth = approval(preflight())
    submitting = _seal({**event_base("synaptic-hf-training-submission-event/v1", auth), "state": "SUBMITTING", "sequence": 1, "previous_event": None, "provider_job": None, "reason_code": None, "provider_effect_possible": True}, "event_id")
    submitted = _seal({**event_base("synaptic-hf-training-submission-event/v1", auth), "state": "SUBMITTED", "sequence": 2, "previous_event": {"uri": "tracking://submitting.json", "sha256": document_sha256(submitting)}, "provider_job": {"namespace": "owner", "job_id": "job-1", "created_at": TS}, "reason_code": None, "provider_effect_possible": True}, "event_id")
    assert validate_submission_event(submitted, approval=auth, previous_event=submitting)["state"] == "SUBMITTED"

    claimed = _seal({**event_base("synaptic-hf-training-cancellation-event/v1", auth), "submission": {"uri": "tracking://submitted.json", "sha256": document_sha256(submitted)}, "provider_job": submitted["provider_job"], "state": "CLAIMED", "sequence": 1, "previous_event": None, "reason_code": None, "provider_effect_possible": True}, "event_id")
    ambiguous = _seal({**claimed, "state": "AMBIGUOUS", "sequence": 2, "previous_event": {"uri": "tracking://claimed.json", "sha256": document_sha256(claimed)}, "reason_code": "INTERRUPTED_AFTER_CLAIM"}, "event_id")
    assert validate_cancellation_event(ambiguous, previous_event=claimed)["state"] == "AMBIGUOUS"

    bad = deepcopy(submitted)
    bad["provider_effect_possible"] = False
    bad = _seal(bad, "event_id")
    with pytest.raises(CloudProviderError):
        validate_submission_event(bad, approval=auth, previous_event=submitting)

    ambiguous_submission = _seal({**event_base("synaptic-hf-training-submission-event/v1", auth), "state": "AMBIGUOUS", "sequence": 2, "previous_event": {"uri": "tracking://submitting.json", "sha256": document_sha256(submitting)}, "provider_job": None, "reason_code": "PROVIDER_OUTCOME_AMBIGUOUS", "provider_effect_possible": True}, "event_id")
    recovered = _seal({**event_base("synaptic-hf-training-submission-event/v1", auth), "state": "SUBMITTED", "sequence": 3, "previous_event": {"uri": "tracking://ambiguous.json", "sha256": document_sha256(ambiguous_submission)}, "provider_job": {"namespace": "owner", "job_id": "job-1", "created_at": TS}, "reason_code": "RECOVERY_CONFIRMED_SUBMITTED", "provider_effect_possible": True}, "event_id")
    assert validate_submission_event(recovered, approval=auth, previous_event=ambiguous_submission)["sequence"] == 3
    hostile = deepcopy(recovered)
    hostile["state"] = "NOT_SUBMITTED"
    hostile["provider_job"] = None
    hostile["provider_effect_possible"] = False
    hostile["reason_code"] = "LOCAL_PRECALL_FAILURE"
    hostile = _seal(hostile, "event_id")
    with pytest.raises(CloudProviderError):
        validate_submission_event(hostile, approval=auth, previous_event=ambiguous_submission)


@pytest.mark.parametrize(
    "reason_code",
    [
        "PROVIDER_OUTCOME_AMBIGUOUS",
        "PROVIDER_REQUEST_REJECTED",
        "PROVIDER_AUTH_REJECTED",
        "PROVIDER_PAYMENT_REJECTED",
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_SERVICE_ERROR",
    ],
)
def test_ambiguous_submission_accepts_only_bounded_provider_failure_classes(reason_code):
    auth = approval(preflight())
    submitting = _seal(
        {
            **event_base("synaptic-hf-training-submission-event/v1", auth),
            "state": "SUBMITTING",
            "sequence": 1,
            "previous_event": None,
            "provider_job": None,
            "reason_code": None,
            "provider_effect_possible": True,
        },
        "event_id",
    )
    ambiguous = _seal(
        {
            **event_base("synaptic-hf-training-submission-event/v1", auth),
            "state": "AMBIGUOUS",
            "sequence": 2,
            "previous_event": {
                "uri": "tracking://submitting.json",
                "sha256": document_sha256(submitting),
            },
            "provider_job": None,
            "reason_code": reason_code,
            "provider_effect_possible": True,
        },
        "event_id",
    )

    assert validate_submission_event(
        ambiguous,
        approval=auth,
        previous_event=submitting,
    )["reason_code"] == reason_code

def test_stopped_observation_is_nonterminal_and_can_refine():
    auth = approval(preflight())
    base = event_base("synaptic-hf-training-observation-event/v1", auth)
    common = {**base, "submission": {"uri": "tracking://submitted.json", "sha256": SHA}, "provider_job": {"namespace": "owner", "job_id": "job-1", "created_at": TS}, "status_intervals": [{"status": "RUNNING", "started_at": TS, "ended_at": "2026-08-20T12:30:00Z"}], "hourly_price_usd": "1.00", "estimated_cost_usd": "0.50"}
    stopped = _seal({**common, "state": "STOPPED", "terminal": False, "previous_event": None, "cost_bounded_completion": False}, "event_id")
    completed = _seal({**common, "state": "COMPLETED", "terminal": True, "previous_event": {"uri": "tracking://stopped.json", "sha256": document_sha256(stopped)}, "cost_bounded_completion": True}, "event_id")
    assert validate_observation_event(completed, previous_event=stopped)["terminal"] is True


def test_verified_result_requires_exact_optimizer_update_and_nonzero_delta():
    auth = approval(preflight())
    slot_id = auth["bindings"]["artifact_slot_id"]
    inventory = [{"path": "final_model/adapter.safetensors", "bytes": 10, "sha256": SHA}]
    inventory_digest = document_sha256(inventory)
    base = {
        **event_base("synaptic-hf-training-result/v1", auth),
        "submission": {"uri": "tracking://submitted.json", "sha256": SHA},
        "observation": {"uri": "tracking://observation.json", "sha256": SHA2},
        "provider_job": {"namespace": "owner", "job_id": "job-1", "created_at": TS},
        "artifact_prefix": {"bucket_id": "owner/bucket", "base_prefix": "training/artifacts", "slot_id": slot_id, "prefix": derive_hf_training_artifact_prefix("training/artifacts", slot_id), "pre_download_inventory_sha256": None, "post_download_inventory_sha256": None, "verified_inventory_sha256": None},
        "inventory": [], "publication": False, "ssh": False, "ports": False, "wandb": False, "job_secrets": [],
    }
    verifying = _seal({**base, "state": "VERIFYING", "previous_result": None, "optimizer_proof": None, "reason_code": None}, "result_id")
    proof = {"optimizer_boundaries": 1, "global_step": 1, "optimizer_step": 1, "scheduler_step": 1, "loss": 1.25, "max_steps": 1, "gradient_accumulation_steps": 1, "pre_adapter_sha256": SHA, "post_adapter_sha256": SHA2, "checkpoint_adapter_sha256": SHA2, "final_adapter_sha256": SHA2, "trainable_weight_delta": 0.01}
    provider_inventory_digest = "c" * 64
    verified_prefix = {**base["artifact_prefix"], "pre_download_inventory_sha256": provider_inventory_digest, "post_download_inventory_sha256": provider_inventory_digest, "verified_inventory_sha256": inventory_digest}
    verified = _seal({**base, "artifact_prefix": verified_prefix, "inventory": inventory, "state": "VERIFIED", "previous_result": {"uri": "tracking://verifying.json", "sha256": document_sha256(verifying)}, "optimizer_proof": proof, "reason_code": None}, "result_id")
    assert validate_result(verified, previous_result=verifying)["state"] == "VERIFIED"

    zero = deepcopy(verified)
    zero["optimizer_proof"]["trainable_weight_delta"] = 0.0
    zero = _seal(zero, "result_id")
    with pytest.raises(CloudProviderError):
        validate_result(zero, previous_result=verifying)

    for field in (
        "optimizer_boundaries", "global_step", "optimizer_step", "scheduler_step",
        "max_steps", "gradient_accumulation_steps", "loss", "trainable_weight_delta",
    ):
        boolean = deepcopy(verified)
        boolean["optimizer_proof"][field] = True
        boolean = _seal(boolean, "result_id")
        with pytest.raises(CloudProviderError):
            validate_result(boolean, previous_result=verifying)

    unstable = deepcopy(verified)
    unstable["artifact_prefix"]["post_download_inventory_sha256"] = "d" * 64
    unstable = _seal(unstable, "result_id")
    with pytest.raises(CloudProviderError, match="changed during download"):
        validate_result(unstable, previous_result=verifying)

    forged_inventory = deepcopy(verified)
    forged_inventory["artifact_prefix"]["verified_inventory_sha256"] = "e" * 64
    forged_inventory = _seal(forged_inventory, "result_id")
    with pytest.raises(CloudProviderError, match="verifier inventory digest"):
        validate_result(forged_inventory, previous_result=verifying)

    unordered = deepcopy(verified)
    unordered["inventory"] = [
        {"path": "z/file", "bytes": 1, "sha256": SHA},
        {"path": "a/file", "bytes": 1, "sha256": SHA2},
    ]
    unordered["artifact_prefix"]["verified_inventory_sha256"] = document_sha256(unordered["inventory"])
    unordered = _seal(unordered, "result_id")
    with pytest.raises(CloudProviderError, match="normalized and ordered"):
        validate_result(unordered, previous_result=verifying)
