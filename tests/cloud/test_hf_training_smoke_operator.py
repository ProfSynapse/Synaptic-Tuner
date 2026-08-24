from __future__ import annotations

import builtins
from datetime import datetime, timezone
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.cloud.hf_training_smoke_operator import (
    FIXED_NONSECRET_ENV,
    HF_ENDPOINT,
    HF_HUB_VERSION,
    ProviderJobExpectation,
    create_provider,
    _run_download_child,
    _advance_status_intervals,
    normalize_artifact_inventory,
)
import tuner.cloud.hf_training_smoke_operator as operator
from tests.cloud.test_hf_training_smoke_contract import approval as accepted_approval, preflight as accepted_preflight
from tuner.core.exceptions import CloudProviderError


class _Client:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def close(self):
        self.closed = True


class _Volume:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"name": self.name}


class _Api:
    def __init__(self, *, endpoint, token):
        self.init = (endpoint, token)
        self.submit = None

    def whoami(self, *, token):
        self.token = token
        return {"name": "synaptic"}

    def list_jobs_hardware(self, *, token):
        self.hardware_token = token
        return [{"name": "a10g-small", "unit_cost_micro_usd": 1000, "unit_label": "minute"}]

    def run_job(self, *, image, command, env, secrets, flavor, timeout, name, labels, volumes, expose, ssh, namespace, token):
        self.run_kwargs = locals()
        return SimpleNamespace(id="job-1", owner=SimpleNamespace(name=namespace), created_at=datetime(2026, 8, 21, 12, tzinfo=timezone.utc))

    def inspect_job(self, *, job_id, namespace, token):
        self.inspect_args = (job_id, namespace, token)
        if hasattr(self, "run_kwargs"):
            values = self.run_kwargs
            return {
                "id": job_id, "owner": {"name": namespace}, "created_at": "2026-08-21T12:00:00Z",
                "status": {"stage": "RUNNING", "expose_urls": [], "ssh_url": None},
                "docker_image": values["image"], "space_id": None,
                "command": values["command"], "arguments": [], "environment": values["env"],
                "secrets": None, "flavor": values["flavor"],
                "labels": {**values["labels"], "name": values["name"]},
                "volumes": values["volumes"], "endpoint": HF_ENDPOINT,
            }
        return {"id": job_id, "owner": {"name": namespace}, "created_at": "2026-08-21T12:00:00Z", "status": {"stage": "RUNNING"}}

    def cancel_job(self, *, job_id, namespace, token):
        self.cancelled = (job_id, namespace, token)

    def list_jobs(self, *, labels, timeout, namespace, token):
        self.list_args = (labels, timeout, namespace, token)
        return []

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
        self.bucket_args = (bucket_id, prefix, recursive, token)
        return []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, "PROVIDER_REQUEST_REJECTED"),
        (401, "PROVIDER_AUTH_REJECTED"),
        (402, "PROVIDER_PAYMENT_REJECTED"),
        (403, "PROVIDER_AUTH_REJECTED"),
        (413, "PROVIDER_REQUEST_REJECTED"),
        (422, "PROVIDER_REQUEST_REJECTED"),
        (429, "PROVIDER_RATE_LIMITED"),
        (500, "PROVIDER_SERVICE_ERROR"),
        (599, "PROVIDER_SERVICE_ERROR"),
        (418, "PROVIDER_OUTCOME_AMBIGUOUS"),
        (None, "PROVIDER_OUTCOME_AMBIGUOUS"),
        ("400", "PROVIDER_OUTCOME_AMBIGUOUS"),
    ],
)
def test_provider_failure_reason_is_bounded_and_status_only(status, expected) -> None:
    error = RuntimeError("must never be persisted")
    error.response = SimpleNamespace(status_code=status)
    assert operator._provider_failure_reason(error) == expected

@pytest.mark.parametrize(
    ("action", "experiment"),
    [
        (operator._execute_action, SimpleNamespace(hf_training_submission_state="PREFLIGHTED")),
        (operator._recover_action, SimpleNamespace(hf_training_submission_state="SUBMITTED")),
        (operator._observe_action, SimpleNamespace(hf_training_submission_state="APPROVED")),
    ],
)
def test_provider_sdk_import_follows_durable_action_gate(monkeypatch, action, experiment) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise AssertionError("provider SDK imported before durable action gate")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(operator, "_cli_state", lambda args, context: (object(), experiment, object()))
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(CloudProviderError):
        action(SimpleNamespace(), SimpleNamespace())


def _modules():
    made = []

    class HTTPX:
        class Timeout:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        @staticmethod
        def Client(**kwargs):
            value = _Client(**kwargs)
            made.append(value)
            return value

    hub = SimpleNamespace(__version__=HF_HUB_VERSION, HfApi=_Api)
    hub.set_client_factory = lambda factory: setattr(hub, "factory", factory)

    return hub, HTTPX, made


def test_provider_is_pinned_isolated_and_passes_token_explicitly() -> None:
    hub, httpx, made = _modules()
    provider = create_provider("hf_secret", environment={}, huggingface_hub=hub, httpx=httpx)
    assert provider.authenticate_namespace("synaptic") == "synaptic"
    quote = provider.quote_a10g(now=lambda: "2026-08-21T12:00:00Z")
    assert quote.unit_cost_micro_usd == 1000
    assert quote.hourly_cost_micro_usd == 60_000
    assert quote.timeout_cost_micro_usd == 30_000
    job = provider.submit(
        image="unsloth/unsloth@sha256:" + "a" * 64,
        command=("python", "-I", "-c", "pass"),
        name="synaptic-hf-training-smoke-" + "a" * 12,
        labels={
            "synaptic-kind": "hf-training-smoke", "synaptic-auth": "a" * 48,
        }, volumes=(_Volume("source"), _Volume("artifact")), namespace="synaptic",
    )
    assert job.job_id == "job-1"
    assert provider._api.run_kwargs["secrets"] == {}
    assert provider._api.run_kwargs["env"] == FIXED_NONSECRET_ENV
    assert provider._api.run_kwargs["expose"] == [] and provider._api.run_kwargs["ssh"] is False
    assert provider._api.run_kwargs["token"] == "hf_secret"
    assert provider.inspect("job-1", namespace="synaptic").status == "RUNNING"
    provider.cancel("job-1", namespace="synaptic")
    assert provider.list_jobs(namespace="synaptic", labels={"synaptic-auth": "a" * 48}) == ()
    assert provider.list_bucket_tree(bucket_id="synaptic/artifacts", prefix="training/slot") == ()
    assert provider._api.inspect_args[-1] == "hf_secret"
    assert provider._api.cancelled[-1] == "hf_secret"
    assert provider._api.list_args[-1] == "hf_secret"
    assert provider._api.bucket_args[-1] == "hf_secret"
    client = hub.factory()
    assert client.kwargs["base_url"] == HF_ENDPOINT
    assert client.kwargs["trust_env"] is False
    provider.close()
    assert made[0].closed


@pytest.mark.parametrize("name", ["HF_TOKEN", "HTTPS_PROXY", "SSL_CERT_FILE", "HF_ENDPOINT"])
def test_provider_rejects_ambient_authority(name: str) -> None:
    hub, httpx, _ = _modules()
    with pytest.raises(CloudProviderError, match="not isolated"):
        create_provider("hf_secret", environment={name: "hostile"}, huggingface_hub=hub, httpx=httpx)


@pytest.mark.parametrize("unit", [True, 0, 16668, 1.5])
def test_quote_rejects_invalid_exact_microusd(unit: object) -> None:
    hub, httpx, _ = _modules()
    provider = create_provider("hf_secret", environment={}, huggingface_hub=hub, httpx=httpx)
    provider._api.list_jobs_hardware = lambda *, token: [
        {"name": "a10g-small", "unit_cost_micro_usd": unit, "unit_label": "minute"}
    ]
    with pytest.raises(CloudProviderError):
        provider.quote_a10g()


def test_quote_accepts_live_a10g_integer_rounding() -> None:
    hub, httpx, _ = _modules()
    provider = create_provider("hf_secret", environment={}, huggingface_hub=hub, httpx=httpx)
    provider._api.list_jobs_hardware = lambda *, token: [
        {"name": "a10g-small", "unit_cost_micro_usd": 16_667, "unit_label": "minute"}
    ]
    quote = provider.quote_a10g(now=lambda: "2026-08-21T12:00:00Z")
    assert quote.hourly_cost_micro_usd == 1_000_020
    assert quote.timeout_cost_micro_usd == 500_010


def test_wrong_hub_version_fails_before_client_construction() -> None:
    hub, httpx, made = _modules()
    hub.__version__ = "1.28.0"
    with pytest.raises(CloudProviderError, match="wrong version"):
        create_provider("hf_secret", environment={}, huggingface_hub=hub, httpx=httpx)
    assert made == []


def test_inspection_reauthenticates_full_immutable_job_spec() -> None:
    hub, httpx, _ = _modules()
    provider = create_provider("hf_secret", environment={}, huggingface_hub=hub, httpx=httpx)
    volumes = (_Volume("source"), _Volume("artifact"))
    labels = {
        "synaptic-kind": "hf-training-smoke", "synaptic-auth": "a" * 48,
    }
    provider.submit(
        image="unsloth/unsloth@sha256:" + "d" * 64,
        command=("python", "-I", "-c", "pass"),
        name="synaptic-hf-training-smoke-" + "a" * 12,
        labels=labels, volumes=volumes, namespace="synaptic",
    )
    expected = ProviderJobExpectation(
        "unsloth/unsloth@sha256:" + "d" * 64, ("python", "-I", "-c", "pass"),
        "synaptic-hf-training-smoke-" + "a" * 12, tuple(sorted(labels.items())),
        volumes, "synaptic",
    )
    provider._api.run_kwargs["env"] = {**FIXED_NONSECRET_ENV, "HOSTILE": "1"}
    with pytest.raises(CloudProviderError, match="not approval-authenticated"):
        provider.inspect("job-1", namespace="synaptic", expected=expected)


def test_provider_job_identity_uses_conservative_provider_labels() -> None:
    approval = accepted_approval(accepted_preflight())
    name, labels = operator.provider_job_identity(approval)

    assert name == f"synaptic-hf-training-smoke-{approval['authorization_id'][:12]}"
    assert labels == {
        "synaptic-kind": "hf-training-smoke",
        "synaptic-auth": approval["authorization_id"][:48],
    }
    assert all(1 <= len(value) <= 63 for item in labels.items() for value in item)
    assert all("_" not in value for item in labels.items() for value in item)


@pytest.mark.parametrize(
    "labels",
    [
        {"synaptic_auth": "a" * 48},
        {"synaptic-auth": "a" * 64},
        {"synaptic-auth": "A" * 48},
    ],
)
def test_provider_rejects_nonconservative_recovery_labels(labels) -> None:
    hub, httpx, _ = _modules()
    provider = create_provider("hf_secret", environment={}, huggingface_hub=hub, httpx=httpx)
    with pytest.raises(CloudProviderError, match="recovery query"):
        provider.list_jobs(namespace="synaptic", labels=labels)


def test_inspection_rejects_swapped_job_id() -> None:
    hub, httpx, _ = _modules()
    provider = create_provider("hf_secret", environment={}, huggingface_hub=hub, httpx=httpx)
    provider._api.inspect_job = lambda *, job_id, namespace, token: {
        "id": "job-swapped", "owner": {"name": namespace},
        "created_at": "2026-08-21T12:00:00Z", "status": {"stage": "RUNNING"},
    }
    with pytest.raises(CloudProviderError, match="another job identity"):
        provider.inspect("job-1", namespace="synaptic")


def test_owned_download_child_propagates_cancellation_after_cleanup(monkeypatch) -> None:
    events = []
    cancellation = KeyboardInterrupt()

    class Process:
        pid = 42
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def wait(self, timeout):
            events.append(("wait", timeout))
            raise cancellation

        def poll(self):
            events.append("poll")
            return 1

    process = Process()
    monkeypatch.setattr(operator.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(operator, "_terminate_download_tree", lambda owned: events.append("terminate"))
    with pytest.raises(KeyboardInterrupt) as caught:
        _run_download_child(("python",), bytearray(b"secret"))
    assert caught.value is cancellation
    assert events.count("terminate") >= 2
    assert "poll" in events
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed


def test_owned_download_child_closes_process_when_reader_start_is_cancelled(monkeypatch) -> None:
    events = []

    class Process:
        pid = 42
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def wait(self, timeout):
            events.append(("wait", timeout))
            return 1

        def poll(self):
            return 1

    process = Process()
    monkeypatch.setattr(operator.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(operator, "_terminate_download_tree", lambda owned: events.append("terminate"))
    monkeypatch.setattr(operator.threading.Thread, "start", lambda self: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _run_download_child(("python",), bytearray(b"secret"))
    assert "terminate" in events
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed


def test_owned_download_child_rejects_output_overflow_without_leaking(tmp_path) -> None:
    with pytest.raises(CloudProviderError, match="rejected"):
        _run_download_child(
            (sys.executable, "-I", "-c", "import os;os.write(1,b'x'*5000)"),
            bytearray(b"secret"),
        )


def test_provider_status_intervals_preserve_actual_transitions() -> None:
    intervals = []
    stage = None
    started = "2026-08-21T12:00:00Z"
    stage, started = _advance_status_intervals(intervals, stage, started, "SCHEDULING", "2026-08-21T12:00:01Z")
    stage, started = _advance_status_intervals(intervals, stage, started, "RUNNING", "2026-08-21T12:01:00Z")
    stage, started = _advance_status_intervals(intervals, stage, started, "COMPLETED", "2026-08-21T12:02:00Z")
    intervals.append({"status": stage, "started_at": started, "ended_at": "2026-08-21T12:02:01Z"})
    assert [item["status"] for item in intervals] == ["SCHEDULING", "RUNNING", "COMPLETED"]
    with pytest.raises(CloudProviderError, match="unknown job stage"):
        _advance_status_intervals([], "RUNNING", started, "HOSTILE", "2026-08-21T12:03:00Z")


def test_provider_close_preserves_pending_cancellation() -> None:
    class Provider:
        def close(self):
            raise RuntimeError("close-detail")

    original = KeyboardInterrupt("cancel-detail")
    caught = None
    try:
        try:
            raise original
        finally:
            operator._close_provider_preserving_pending(Provider())
    except BaseException as exc:
        caught = exc
    assert caught is original

    with pytest.raises(RuntimeError, match="close-detail"):
        operator._close_provider_preserving_pending(Provider())


def test_approval_builder_uses_exact_observation_deadline_field() -> None:
    document = operator.build_approval_document(
        preflight=accepted_preflight(), preflight_uri="tracking://preflight.json",
        user_authorization_reference="conversation-1",
        issued_at="2026-08-20T12:00:00Z", expires_at="2026-08-20T13:00:00Z",
    )
    assert document["observe_until_seconds"] == 2100
    assert "observe_until_secoeconds" not in document


def test_normalizes_exact_v127_bucket_files_and_folders() -> None:
    from tuner.cloud.hf_training_smoke_artifacts import EXPECTED_PATHS
    prefix = "training/slot"
    values = [
        SimpleNamespace(type="directory", path=f"{prefix}/checkpoint-1"),
        SimpleNamespace(type="directory", path=f"{prefix}/final_model"),
        *[
            SimpleNamespace(type="file", path=f"{prefix}/{path}", size=index + 1, xet_hash=f"{index + 1:064x}")
            for index, path in enumerate(EXPECTED_PATHS)
        ],
    ]
    inventory = normalize_artifact_inventory(values, prefix=prefix)
    assert [item["path"] for item in inventory] == sorted(EXPECTED_PATHS)
    with pytest.raises(CloudProviderError, match="folder inventory"):
        normalize_artifact_inventory(values[1:], prefix=prefix)


def _execute_fakes(
    monkeypatch, tmp_path: Path, *, submit_error: Exception | None = None,
    slot_nonempty: bool = False,
):
    events = []
    approval = accepted_approval(accepted_preflight())
    experiment = SimpleNamespace(
        experiment_id="exp-1", hf_training_submission_state="APPROVED",
        hf_training_approval_uri="tracking://approval.json",
    )
    volume = _Volume("source")
    source_lock = SimpleNamespace(
        project_source=SimpleNamespace(commit="c" * 40),
        engine_source=SimpleNamespace(commit="d" * 40), mode="dual_clone",
    )
    preparation = SimpleNamespace(
        volume_spec=SimpleNamespace(), source_lock=source_lock,
        physical_project_root="/workspace/source/project",
        physical_engine_root="/workspace/source/engine",
    )
    preparation.require_consumable = lambda: events.append("consumable")
    preparation.prove_volume = lambda hub: SimpleNamespace(provider_volume=volume)
    claim_doc = None

    class Tracking:
        def claim_hf_training_submission(self, experiment, document):
            nonlocal claim_doc
            events.append("claim")
            claim_doc = document
            return SimpleNamespace(
                provider_attempt_authorized=True, state="SUBMITTING", document=document,
                uri="tracking://claim.json",
            )

        def record_hf_training_submission_terminal(self, experiment, document):
            events.append(("terminal", document["state"]))

    class Provider:
        sdk = object()

        def authenticate_namespace(self, namespace):
            events.append("auth")

        def list_bucket_tree(self, *, bucket_id, prefix):
            events.append("slot_check")
            return (object(),) if slot_nonempty else ()

        def submit(self, **kwargs):
            events.append("submit")
            if submit_error:
                raise submit_error
            return operator.ProviderJob("owner", "job-1", "2026-08-20T12:00:00Z", "RUNNING")

        def close(self):
            events.append("close")

    monkeypatch.setattr(operator, "_cli_state", lambda args, context: (Tracking(), experiment, preparation))
    monkeypatch.setattr(operator, "_load_training_document", lambda tracking, uri: approval)
    monkeypatch.setattr(operator, "_utc_now", lambda: "2026-08-20T12:00:00Z")
    monkeypatch.setattr(operator, "probe_provider_contract", lambda hub: events.append("probe"))
    monkeypatch.setattr(operator, "create_provider", lambda token: (events.append("create") or Provider()))
    import tuner.cloud.hf_training_smoke_workload as workload_module
    monkeypatch.setattr(workload_module, "build_workload", lambda *args, **kwargs: SimpleNamespace(
        workload_sha256="a" * 64, remote_argv_sha256="a" * 64,
        provider_command_sha256="b" * 64,
        image="unsloth/unsloth@" + str(approval["bindings"]["image_child_digest"]),
        provider_command=("python", "-I", "-c", "pass"),
    ))
    import tuner.cloud.hf_volume_transport as volume_module
    monkeypatch.setattr(volume_module, "prove_writable_artifact_volume", lambda hub, spec: SimpleNamespace(provider_volume=_Volume("artifact")))
    import tuner.handlers._hf_secret_file as secret_module
    monkeypatch.setattr(secret_module, "preflight_hf_secret_file", lambda path, context: (events.append("secret_preflight") or object()))
    monkeypatch.setattr(secret_module, "read_claimed_hf_token", lambda claim: (events.append("token") or "hf_secret"))
    return events, approval


def test_execute_claims_once_before_token_and_submits_once(monkeypatch, tmp_path) -> None:
    events, _ = _execute_fakes(monkeypatch, tmp_path)
    result = operator._execute_action(SimpleNamespace(env_file="secret.env"), SimpleNamespace(project_root=tmp_path))
    assert result["status"] == "SUBMITTED"
    assert events.count("claim") == 1 and events.count("submit") == 1
    assert events.index("claim") < events.index("token") < events.index("slot_check") < events.index("submit")
    assert ("terminal", "SUBMITTED") in events


def test_execute_provider_uncertainty_records_ambiguous_without_retry(monkeypatch, tmp_path) -> None:
    events, _ = _execute_fakes(monkeypatch, tmp_path, submit_error=RuntimeError("secret"))
    with pytest.raises(CloudProviderError, match="submission was rejected") as caught:
        operator._execute_action(SimpleNamespace(env_file="secret.env"), SimpleNamespace(project_root=tmp_path))
    assert "secret" not in str(caught.value)
    assert events.count("submit") == 1
    assert ("terminal", "AMBIGUOUS") in events


def test_execute_rechecks_empty_slot_after_claim_before_submit(monkeypatch, tmp_path) -> None:
    events, _ = _execute_fakes(monkeypatch, tmp_path, slot_nonempty=True)
    result = operator._execute_action(
        SimpleNamespace(env_file="secret.env"), SimpleNamespace(project_root=tmp_path),
    )
    assert result == {
        "status": "NOT_SUBMITTED", "submitted": False,
        "retry_allowed": False, "reason_code": "PREFIX_NOT_EMPTY",
    }
    assert events.count("claim") == 1 and events.count("slot_check") == 1
    assert "submit" not in events
    assert ("terminal", "NOT_SUBMITTED") in events


def test_execute_rejects_future_issued_approval_before_provider_authority(monkeypatch, tmp_path) -> None:
    events, approval = _execute_fakes(monkeypatch, tmp_path)
    approval.pop("authorization_id")
    approval["issued_at"] = "2026-08-20T12:00:01Z"
    approval.update(operator.seal_training_document(approval))
    with pytest.raises(CloudProviderError, match="approval or quote is stale"):
        operator._execute_action(
            SimpleNamespace(env_file="secret.env"), SimpleNamespace(project_root=tmp_path),
        )
    assert "claim" not in events
    assert "create" not in events


def test_preflight_snapshots_lock_quotes_and_proves_empty_slot(monkeypatch, tmp_path) -> None:
    events = []
    repo = Path(__file__).resolve().parents[2]
    experiment = SimpleNamespace(experiment_id="exp-1")
    source_lock = SimpleNamespace(
        run_id="run-1", mode="dual_clone",
        project_source=SimpleNamespace(commit="c" * 40),
        engine_source=SimpleNamespace(commit="d" * 40),
    )
    source_spec = SimpleNamespace(source="owner/source", path="prepared/run-1")
    descriptor = {
        "bundle": {"content_sha256": "1" * 64},
        "capsule": {"manifest": {"sha256": "2" * 64}},
        "checkout_policy": {"sha256": "3" * 64},
    }
    preparation = SimpleNamespace(
        volume_spec=source_spec, source_lock=source_lock, source_lock_sha256="4" * 64,
        source_lock_uri="tracking://source-lock.json",
        descriptor_uri="tracking://descriptor.json", descriptor_sha256="5" * 64,
        provisioning_evidence_uri="tracking://evidence.json", provisioning_evidence_sha256="6" * 64,
        physical_project_root="/workspace/source/project", physical_engine_root="/workspace/source/engine",
        consumable_transport=SimpleNamespace(prepared=SimpleNamespace(descriptor=descriptor)),
    )
    preparation.require_consumable = lambda: events.append("consumable")
    preparation.prove_volume = lambda hub: events.append("source_volume")

    class Tracking:
        base_dir = repo

        def snapshot_hf_training_runtime_lock(self, experiment, runtime_lock):
            events.append("snapshot")
            from tuner.cloud.hf_training_smoke_contract import canonical_json_bytes
            import hashlib
            return {"uri": "tracking://runtime-lock.json", "sha256": hashlib.sha256(canonical_json_bytes(runtime_lock)).hexdigest()}

        def record_hf_training_preflight(self, experiment, preflight):
            events.append("record")
            self.preflight = preflight

    tracking = Tracking()
    monkeypatch.setattr(operator, "_cli_state", lambda args, context: (tracking, experiment, preparation))
    monkeypatch.setattr(operator, "_utc_now", lambda: "2026-08-21T12:00:00Z")
    import shared.experiment_tracking.root_identity as root_module
    monkeypatch.setattr(root_module, "ensure_tracking_root_identity", lambda root: {"root_id": "7" * 64})
    import tuner.cloud.hf_training_smoke_workload as workload_module
    repository = tmp_path / "repo"
    runtime_lock = repository / workload_module.RUNTIME_LOCK_PATH
    runtime_lock.parent.mkdir(parents=True)
    runtime_lock.write_bytes((repo / workload_module.RUNTIME_LOCK_PATH).read_bytes())
    dataset = repository / workload_module.DATASET
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(
        (repo / workload_module.DATASET).read_bytes().replace(b"\r\n", b"\n")
    )

    def workload(*args, **kwargs):
        slot = kwargs["artifact_slot"]
        return SimpleNamespace(
            workload_sha256="8" * 64,
            remote_argv=("--artifact-slot", slot), remote_argv_sha256="9" * 64,
            provider_command=("python",), provider_command_sha256="a" * 64,
        )

    monkeypatch.setattr(workload_module, "build_workload", workload)
    import tuner.cloud.hf_volume_transport as volume_module
    monkeypatch.setattr(volume_module, "validate_disjoint_volume_prefixes", lambda source, artifact: events.append("disjoint"))
    monkeypatch.setattr(volume_module, "prove_writable_artifact_volume", lambda hub, spec: events.append("artifact_volume"))
    import tuner.handlers._hf_secret_file as secret_module
    monkeypatch.setattr(secret_module, "preflight_hf_secret_file", lambda path, context: object())
    monkeypatch.setattr(secret_module, "read_claimed_hf_token", lambda claim: "hf_secret")

    class Provider:
        sdk = object()

        def authenticate_namespace(self, namespace): events.append("auth")
        def quote_a10g(self):
            events.append("quote")
            return operator.HardwareQuote(operator.HF_ENDPOINT, "a10g-small", 1000, "minute", 60000, 30000, "2026-08-21T12:00:00Z")
        def list_bucket_tree(self, **kwargs): events.append("empty"); return ()
        def close(self): events.append("close")

    monkeypatch.setattr(operator, "create_provider", lambda token: Provider())
    result = operator._preflight_action(
        SimpleNamespace(
            source_bucket_id="owner/source", source_prefix="prepared/run-1",
            artifact_bucket_id="owner/artifacts", artifact_prefix="training/artifacts",
            expected_namespace="owner", env_file="secret.env",
        ),
        SimpleNamespace(project_root=repository),
    )
    assert result["status"] == "PASS"
    assert events.count("snapshot") == events.count("quote") == events.count("empty") == events.count("record") == 1
    assert events.index("empty") < events.index("record")


@pytest.mark.parametrize(("candidate_count", "inspect_fails", "recovered"), [(0, False, False), (2, False, False), (1, True, False), (1, False, True)])
def test_recovery_only_accepts_one_fully_authenticated_job(
    monkeypatch, tmp_path, candidate_count: int, inspect_fails: bool, recovered: bool,
) -> None:
    events = []
    approval = accepted_approval(accepted_preflight())
    claim = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json",
        state="SUBMITTING", sequence=1, occurred_at="2026-08-20T12:00:00Z",
    )
    previous = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json", state="AMBIGUOUS", sequence=2,
        occurred_at="2026-08-20T12:00:01Z", previous_event=claim,
        previous_event_uri="tracking://claim.json", reason_code="PROVIDER_OUTCOME_AMBIGUOUS",
    )
    experiment = SimpleNamespace(
        experiment_id="exp-1", hf_training_submission_state="AMBIGUOUS",
        hf_training_approval_uri="tracking://approval.json",
        hf_training_submission_event_uri="tracking://ambiguous.json",
    )
    source_lock = SimpleNamespace(
        project_source=SimpleNamespace(commit="c" * 40),
        engine_source=SimpleNamespace(commit="d" * 40), mode="dual_clone",
    )
    preparation = SimpleNamespace(
        volume_spec=SimpleNamespace(), source_lock=source_lock,
        physical_project_root="/workspace/source/project", physical_engine_root="/workspace/source/engine",
    )
    preparation.require_consumable = lambda: None
    preparation.prove_volume = lambda hub: SimpleNamespace(provider_volume=_Volume("source"))

    class Tracking:
        def recover_hf_training_submission(self, experiment, document):
            events.append(("recover", document["sequence"], document["reason_code"]))

    monkeypatch.setattr(operator, "_cli_state", lambda args, context: (Tracking(), experiment, preparation))
    monkeypatch.setattr(operator, "_load_training_document", lambda tracking, uri: approval if uri.endswith("approval.json") else previous)
    monkeypatch.setattr(operator, "_utc_now", lambda: "2026-08-20T12:00:02Z")
    import tuner.cloud.hf_training_smoke_workload as workload_module
    monkeypatch.setattr(workload_module, "build_workload", lambda *args, **kwargs: SimpleNamespace(
        workload_sha256=str(approval["bindings"]["workload_digest"]),
        remote_argv_sha256=str(approval["bindings"]["remote_argv_sha256"]),
        provider_command_sha256=str(approval["bindings"]["provider_command_sha256"]),
        image="unsloth/unsloth@" + str(approval["bindings"]["image_child_digest"]),
        provider_command=("python", "-I", "-c", "pass"),
    ))
    import tuner.cloud.hf_volume_transport as volume_module
    monkeypatch.setattr(volume_module, "prove_writable_artifact_volume", lambda hub, spec: SimpleNamespace(provider_volume=_Volume("artifact")))
    import tuner.handlers._hf_secret_file as secret_module
    monkeypatch.setattr(secret_module, "preflight_hf_secret_file", lambda path, context: object())
    monkeypatch.setattr(secret_module, "read_claimed_hf_token", lambda claim: "hf_secret")

    class Provider:
        sdk = object()
        def authenticate_namespace(self, namespace): events.append("auth")
        def list_jobs(self, **kwargs): events.append("list"); return tuple(SimpleNamespace(id=f"job-{i}") for i in range(candidate_count))
        def inspect(self, job_id, **kwargs):
            events.append("inspect")
            if inspect_fails: raise CloudProviderError("drift")
            return operator.ProviderJob("owner", job_id, "2026-08-20T12:00:00Z", "RUNNING")
        def close(self): events.append("close")

    monkeypatch.setattr(operator, "create_provider", lambda token: Provider())
    result = operator._recover_action(SimpleNamespace(env_file="secret.env"), SimpleNamespace(project_root=tmp_path))
    assert result["recovered"] is recovered
    assert events.count("list") == 1
    assert sum(isinstance(item, tuple) and item[0] == "recover" for item in events) == (1 if recovered else 0)


def test_verify_claims_pre_downloads_postlists_and_records_distinct_digests(monkeypatch, tmp_path) -> None:
    events = []
    approval = accepted_approval(accepted_preflight())
    submitted = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json", state="SUBMITTING",
        sequence=1, occurred_at="2026-08-20T12:00:00Z",
    )
    submission = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json", state="SUBMITTED",
        sequence=2, occurred_at="2026-08-20T12:00:01Z", previous_event=submitted,
        previous_event_uri="tracking://claim.json",
        provider_job=operator.ProviderJob("owner", "job-1", "2026-08-20T12:00:00Z"),
    )
    observation = operator._observation_document(
        approval=approval, approval_uri="tracking://approval.json",
        submission=submission, submission_uri="tracking://submission.json",
        state="COMPLETED", started_at="2026-08-20T12:00:00Z", ended_at="2026-08-20T12:01:00Z",
        status_intervals=[{"status": "COMPLETED", "started_at": "2026-08-20T12:00:00Z", "ended_at": "2026-08-20T12:01:00Z"}],
    )
    experiment = SimpleNamespace(
        experiment_id="exp-1", hf_training_observation_state="COMPLETED",
        hf_training_approval_uri="tracking://approval.json",
        hf_training_submission_event_uri="tracking://submission.json",
        hf_training_observation_event_uri="tracking://observation.json",
    )

    class Tracking:
        def claim_hf_training_verification(self, experiment, document):
            events.append("claim")
            return SimpleNamespace(provider_attempt_authorized=True, state="VERIFYING", document=document, uri="tracking://verify.json")
        def record_hf_training_result(self, experiment, document):
            events.append(("result", document))

    tracking = Tracking()
    monkeypatch.setattr(operator, "_cli_state", lambda args, context: (tracking, experiment, object()))
    preflight = accepted_preflight()
    from tuner.cloud.hf_training_smoke_workload import validate_runtime_lock, RUNTIME_LOCK_PATH
    runtime_lock, _ = validate_runtime_lock(Path(__file__).resolve().parents[2] / RUNTIME_LOCK_PATH)

    def load(tracking, uri):
        if uri.endswith("approval.json"): return approval
        if uri.endswith("submission.json"): return submission
        if uri.endswith("observation.json"): return observation
        if uri == str(approval["preflight"]["uri"]): return preflight
        return runtime_lock

    monkeypatch.setattr(operator, "_load_training_document", load)
    monkeypatch.setattr(operator, "_utc_now", lambda: "2026-08-20T12:02:00Z")
    import tuner.handlers._hf_secret_file as secret_module
    monkeypatch.setattr(secret_module, "preflight_hf_secret_file", lambda path, context: object())
    monkeypatch.setattr(secret_module, "read_claimed_hf_token", lambda claim: (events.append("token") or "hf_secret"))
    from tuner.cloud.hf_training_smoke_artifacts import EXPECTED_PATHS
    prefix = str(approval["bindings"]["artifact_prefix"])
    listing = (
        SimpleNamespace(type="directory", path=f"{prefix}/checkpoint-1"),
        SimpleNamespace(type="directory", path=f"{prefix}/final_model"),
        *tuple(SimpleNamespace(type="file", path=f"{prefix}/{path}", size=index + 1, xet_hash=f"{index + 1:064x}") for index, path in enumerate(EXPECTED_PATHS)),
    )

    class Provider:
        def authenticate_namespace(self, namespace): events.append("auth")
        def list_bucket_tree(self, **kwargs): events.append("list"); return listing
        def close(self): events.append("close")

    monkeypatch.setattr(operator, "create_provider", lambda token: Provider())

    def download(**kwargs):
        events.append("download")

    monkeypatch.setattr(operator, "download_exact_artifacts", download)
    import tuner.cloud.hf_training_smoke_artifacts as artifact_module
    proof = {
        "optimizer_boundaries": 1, "global_step": 1, "optimizer_step": 1,
        "scheduler_step": 1, "loss": 1.0, "max_steps": 1,
        "gradient_accumulation_steps": 1,
        "pre_adapter_sha256": "b" * 64,
        "post_adapter_sha256": "a" * 64,
        "checkpoint_adapter_sha256": "a" * 64,
        "final_adapter_sha256": "a" * 64,
        "trainable_weight_delta": 0.5,
    }
    monkeypatch.setattr(
        artifact_module, "verify_artifact_tree",
        lambda root, expectation: (
            events.append("verify_tree") or {
                "adapter_identity": "a" * 64, "optimizer_proof": proof,
            }
        ),
    )
    monkeypatch.setattr(artifact_module, "build_inventory", lambda root: (events.append("local_inventory") or {"files": [
        {"path": path, "size": index + 1, "sha256": f"{index + 1:064x}"}
        for index, path in enumerate(sorted(EXPECTED_PATHS))
    ]}))
    result = operator._verify_action(SimpleNamespace(env_file="secret.env"), SimpleNamespace())
    assert result["status"] == "VERIFIED"
    assert events.index("claim") < events.index("token") < events.index("list") < events.index("download")
    assert events.count("list") == 2
    document = next(item[1] for item in events if isinstance(item, tuple) and item[0] == "result")
    assert document["artifact_prefix"]["pre_download_inventory_sha256"] == document["artifact_prefix"]["post_download_inventory_sha256"]
    assert document["artifact_prefix"]["verified_inventory_sha256"] != document["artifact_prefix"]["pre_download_inventory_sha256"]
    assert document["optimizer_proof"]["post_adapter_sha256"] == "a" * 64

    events.clear()
    monkeypatch.setattr(secret_module, "read_claimed_hf_token", lambda claim: (_ for _ in ()).throw(RuntimeError("secret-value")))
    with pytest.raises(CloudProviderError, match="verification was rejected") as caught:
        operator._verify_action(SimpleNamespace(env_file="secret.env"), SimpleNamespace())
    assert "secret-value" not in str(caught.value)
    assert "download" not in events
    failed = next(item[1] for item in events if isinstance(item, tuple) and item[0] == "result")
    assert failed["state"] == "INCONCLUSIVE"


def test_verify_reclaim_binds_durable_inconclusive_predecessor(monkeypatch) -> None:
    approval = accepted_approval(accepted_preflight())
    submitted = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json", state="SUBMITTING",
        sequence=1, occurred_at="2026-08-20T12:00:00Z",
    )
    submission = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json", state="SUBMITTED",
        sequence=2, occurred_at="2026-08-20T12:00:01Z", previous_event=submitted,
        previous_event_uri="tracking://claim.json",
        provider_job=operator.ProviderJob("owner", "job-1", "2026-08-20T12:00:00Z"),
    )
    observation = operator._observation_document(
        approval=approval, approval_uri="tracking://approval.json",
        submission=submission, submission_uri="tracking://submission.json",
        state="COMPLETED", started_at="2026-08-20T12:00:00Z",
        ended_at="2026-08-20T12:01:00Z",
        status_intervals=[{
            "status": "COMPLETED", "started_at": "2026-08-20T12:00:00Z",
            "ended_at": "2026-08-20T12:01:00Z",
        }],
    )
    verifying = operator._result_document(
        approval=approval, approval_uri="tracking://approval.json",
        submission=submission, submission_uri="tracking://submission.json",
        observation=observation, observation_uri="tracking://observation.json",
        state="VERIFYING",
    )
    inconclusive = operator._result_document(
        approval=approval, approval_uri="tracking://approval.json",
        submission=submission, submission_uri="tracking://submission.json",
        observation=observation, observation_uri="tracking://observation.json",
        state="INCONCLUSIVE", previous=verifying, previous_uri="tracking://verify.json",
        reason_code="VERIFICATION_INCONCLUSIVE",
    )
    experiment = SimpleNamespace(
        experiment_id="exp-1", hf_training_observation_state="COMPLETED",
        hf_training_approval_uri="tracking://approval.json",
        hf_training_submission_event_uri="tracking://submission.json",
        hf_training_observation_event_uri="tracking://observation.json",
        hf_training_result_state="INCONCLUSIVE",
        hf_training_result_uri="tracking://inconclusive.json",
        hf_training_result_sha256=operator.document_sha256(inconclusive),
    )
    captured = []

    class Tracking:
        def claim_hf_training_verification(self, experiment, document):
            captured.append(document)
            return SimpleNamespace(provider_attempt_authorized=False, state="VERIFYING")

    documents = {
        "tracking://approval.json": approval,
        "tracking://submission.json": submission,
        "tracking://observation.json": observation,
        "tracking://inconclusive.json": inconclusive,
    }
    monkeypatch.setattr(
        operator, "_cli_state", lambda args, context: (Tracking(), experiment, object()),
    )
    monkeypatch.setattr(
        operator, "_load_training_document", lambda tracking, uri: documents[uri],
    )
    result = operator._verify_action(SimpleNamespace(), SimpleNamespace())
    assert result == {"status": "VERIFYING", "verified": False}
    assert captured[0]["previous_result"] == {
        "uri": "tracking://inconclusive.json",
        "sha256": operator.document_sha256(inconclusive),
    }


@pytest.mark.parametrize("state", [None, "STOPPED", "ERROR", "CANCELLED"])
def test_verify_requires_durable_completed_observation(monkeypatch, state) -> None:
    experiment = SimpleNamespace(hf_training_observation_state=state)
    monkeypatch.setattr(
        operator, "_cli_state", lambda args, context: (object(), experiment, object()),
    )
    with pytest.raises(CloudProviderError, match="completion is unavailable"):
        operator._verify_action(SimpleNamespace(), SimpleNamespace())


@pytest.mark.parametrize(("stages", "clock_values", "expected_state"), [
    (["SCHEDULING", "RUNNING", "COMPLETED"], [10], "COMPLETED"),
    (["RUNNING", "HOSTILE"], [1501], None),
    (["RUNNING"], [2101], "STOPPED"),
    (["RUNNING", "RUNNING"], [1501, 2101, 2101], "STOPPED"),
])
def test_observe_preserves_stages_and_rejects_unknown_before_cancel(
    monkeypatch, tmp_path, stages, clock_values, expected_state,
) -> None:
    events = []
    approval = accepted_approval(accepted_preflight())
    claim_event = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json", state="SUBMITTING",
        sequence=1, occurred_at="2026-08-20T12:00:00Z",
    )
    durable_job = operator.ProviderJob("owner", "job-1", "2026-08-20T12:00:00Z")
    submission = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json", state="SUBMITTED",
        sequence=2, occurred_at="2026-08-20T12:05:00Z", previous_event=claim_event,
        previous_event_uri="tracking://claim.json", provider_job=durable_job,
    )
    experiment = SimpleNamespace(
        experiment_id="exp-1", hf_training_submission_state="SUBMITTED",
        hf_training_approval_uri="tracking://approval.json",
        hf_training_submission_event_uri="tracking://submission.json",
        hf_training_observation_state=None,
        hf_training_observation_event_uri=None,
        hf_training_observation_event_sha256=None,
    )
    source_lock = SimpleNamespace(
        project_source=SimpleNamespace(commit="c" * 40),
        engine_source=SimpleNamespace(commit="d" * 40), mode="dual_clone",
    )
    preparation = SimpleNamespace(
        volume_spec=SimpleNamespace(), source_lock=source_lock,
        physical_project_root="/workspace/source/project", physical_engine_root="/workspace/source/engine",
    )
    preparation.require_consumable = lambda: None
    preparation.prove_volume = lambda hub: SimpleNamespace(provider_volume=_Volume("source"))

    class Tracking:
        def claim_hf_training_cancellation(self, experiment, document):
            events.append("cancel_claim")
            return SimpleNamespace(provider_attempt_authorized=True, document=document, uri="tracking://cancel.json")
        def record_hf_training_cancellation_terminal(self, experiment, document): events.append(("cancel_terminal", document))
        def record_hf_training_observation(self, experiment, document): events.append(("observation", document))

    monkeypatch.setattr(operator, "_cli_state", lambda args, context: (Tracking(), experiment, preparation))
    documents = {
        "tracking://approval.json": approval,
        "tracking://submission.json": submission,
        "tracking://claim.json": claim_event,
    }
    monkeypatch.setattr(operator, "_load_training_document", lambda tracking, uri: documents[uri])
    timestamps = iter([f"2026-08-20T12:00:0{i}Z" for i in range(1, 10)])
    monkeypatch.setattr(operator, "_utc_now", lambda: next(timestamps))
    import time as time_module
    start_epoch = datetime(2026, 8, 20, 12, tzinfo=timezone.utc).timestamp()
    clock = iter(clock_values)
    last_clock = clock_values[-1]

    def observed_time():
        nonlocal last_clock
        try:
            last_clock = next(clock)
        except StopIteration:
            pass
        return start_epoch + last_clock

    monkeypatch.setattr(time_module, "time", observed_time)
    monkeypatch.setattr(time_module, "sleep", lambda seconds: None)
    import tuner.cloud.hf_training_smoke_workload as workload_module
    monkeypatch.setattr(workload_module, "build_workload", lambda *args, **kwargs: SimpleNamespace(
        workload_sha256=str(approval["bindings"]["workload_digest"]),
        remote_argv_sha256=str(approval["bindings"]["remote_argv_sha256"]),
        provider_command_sha256=str(approval["bindings"]["provider_command_sha256"]),
        image="unsloth/unsloth@" + str(approval["bindings"]["image_child_digest"]),
        provider_command=("python", "-I", "-c", "pass"),
    ))
    import tuner.cloud.hf_volume_transport as volume_module
    monkeypatch.setattr(volume_module, "prove_writable_artifact_volume", lambda hub, spec: SimpleNamespace(provider_volume=_Volume("artifact")))
    import tuner.handlers._hf_secret_file as secret_module
    monkeypatch.setattr(secret_module, "preflight_hf_secret_file", lambda path, context: object())
    monkeypatch.setattr(secret_module, "read_claimed_hf_token", lambda claim: "hf_secret")

    class Provider:
        def __init__(self): self.remaining = list(stages)
        def authenticate_namespace(self, namespace): pass
        def inspect(self, job_id, **kwargs):
            events.append("inspect")
            return operator.ProviderJob("owner", job_id, "2026-08-20T12:00:00Z", self.remaining.pop(0))
        def cancel(self, job_id, **kwargs): events.append("cancel")
        def close(self): events.append("close")

    monkeypatch.setattr(operator, "create_provider", lambda token: Provider())
    if expected_state is None:
        with pytest.raises(CloudProviderError, match="unknown job stage"):
            operator._observe_action(SimpleNamespace(env_file="secret.env"), SimpleNamespace(project_root=tmp_path))
        assert "cancel" not in events
    else:
        result = operator._observe_action(SimpleNamespace(env_file="secret.env"), SimpleNamespace(project_root=tmp_path))
        assert result["status"] == expected_state
        document = next(item[1] for item in events if isinstance(item, tuple) and item[0] == "observation")
        assert [item["status"] for item in document["status_intervals"]] == list(dict.fromkeys(stages))
        if clock_values[0] >= 2100:
            assert "cancel_claim" not in events and "cancel" not in events
        if clock_values[:2] == [1501, 2101]:
            assert "cancel_claim" in events and "cancel" not in events


def test_observation_document_binds_stopped_predecessor() -> None:
    approval = accepted_approval(accepted_preflight())
    claim_event = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json", state="SUBMITTING",
        sequence=1, occurred_at="2026-08-20T12:00:00Z",
    )
    submission = operator.build_submission_event(
        approval=approval, approval_uri="tracking://approval.json", state="SUBMITTED",
        sequence=2, occurred_at="2026-08-20T12:00:01Z", previous_event=claim_event,
        previous_event_uri="tracking://claim.json",
        provider_job=operator.ProviderJob("owner", "job-1", "2026-08-20T12:00:00Z"),
    )
    stopped = operator._observation_document(
        approval=approval, approval_uri="tracking://approval.json",
        submission=submission, submission_uri="tracking://submission.json",
        state="STOPPED", started_at="2026-08-20T12:00:00Z",
        ended_at="2026-08-20T12:35:00Z",
        status_intervals=[{
            "status": "RUNNING", "started_at": "2026-08-20T12:00:00Z",
            "ended_at": "2026-08-20T12:35:00Z",
        }],
    )
    completed = operator._observation_document(
        approval=approval, approval_uri="tracking://approval.json",
        submission=submission, submission_uri="tracking://submission.json",
        state="COMPLETED", started_at="2026-08-20T12:00:00Z",
        ended_at="2026-08-20T12:36:00Z",
        status_intervals=[{
            "status": "COMPLETED", "started_at": "2026-08-20T12:35:00Z",
            "ended_at": "2026-08-20T12:36:00Z",
        }], previous=stopped, previous_uri="tracking://stopped.json",
    )
    from tuner.cloud.hf_training_smoke_contract import validate_observation_event
    assert validate_observation_event(completed, previous_event=stopped) == completed
    assert completed["previous_event"] == {
        "uri": "tracking://stopped.json", "sha256": operator.document_sha256(stopped),
    }
