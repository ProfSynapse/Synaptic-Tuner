from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tuner.cloud import (
    CloudJobSpec,
    HFJobExecutor,
    RepoCheckoutSpec,
    build_bash_command,
    build_hf_job_secrets,
    build_repo_checkout_steps,
    decode_hf_job_label,
    format_timeout_hours,
    resolve_hf_bucket_id,
    sanitize_hf_job_labels,
    observe_submitted_bootstrap_smoke,
    submit_approved_bootstrap_smoke,
    HFBootstrapSmokeSubmission,
)
from tuner.cloud.hf_run_approval import build_hf_run_approval
from tuner.cloud.hf_jobs import _normalize_job_info
from tuner.core.exceptions import CloudProviderError


_REPO_ROOT = Path(__file__).resolve().parents[2]
_HEAVY_PROVIDER_ROOTS = {
    "boto3",
    "botocore",
    "huggingface_hub",
    "modal",
    "runpod",
    "transformers",
}


def _fresh_imported_provider_modules(module_name: str) -> list[str]:
    code = (
        "import importlib,json,sys;"
        f"importlib.import_module({module_name!r});"
        f"roots={sorted(_HEAVY_PROVIDER_ROOTS)!r};"
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.split('.')[0] in roots)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_hf_jobs_primitives_and_cloud_facade_are_import_light_in_fresh_processes():
    for module_name in ("tuner.cloud.hf_jobs", "tuner.cloud"):
        assert _fresh_imported_provider_modules(module_name) == []


def test_build_repo_checkout_steps_pins_exact_commit():
    steps = build_repo_checkout_steps(
        RepoCheckoutSpec(
            url="https://github.com/test/repo.git",
            branch="main",
            commit="abc12345def67890",
        )
    )

    assert len(steps) == 2
    assert "if command -v git" in steps[0]
    assert "git clone --branch main --depth 1 https://github.com/test/repo.git /workspace/repo" in steps[0]
    assert "https://github.com/test/repo/archive/abc12345def67890.tar.gz" in steps[0]
    assert steps[1] == (
        "if [ -d /workspace/repo/.git ]; then cd /workspace/repo && "
        "git fetch --depth 1 origin abc12345def67890 && git checkout abc12345def67890; fi"
    )


def test_format_timeout_hours_normalizes_integers_and_floats():
    assert format_timeout_hours(4.0) == "4h"
    assert format_timeout_hours(2.5) == "2.5h"
    assert format_timeout_hours(None) is None


def test_build_hf_job_secrets_returns_both_key_names(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")

    assert build_hf_job_secrets() == {
        "HF_TOKEN": "hf_test_token_12345",
        "HF_API_KEY": "hf_test_token_12345",
    }


def test_hf_job_executor_submits_shared_job_spec():
    mock_hub = MagicMock()
    mock_hub.run_job.return_value = SimpleNamespace(id="job-123", url="https://hf.co/jobs/job-123")

    submission = HFJobExecutor(mock_hub).submit(
        CloudJobSpec(
            provider="hf_jobs",
            image="unsloth/test:latest",
            command=build_bash_command(["echo hi"]),
            flavor="a10g-small",
            timeout_hours=4.0,
            secrets={"HF_TOKEN": "hf_test_token_12345"},
            env={"FOO": "bar"},
            labels={"task": "gym"},
        )
    )

    assert submission.job_id == "job-123"
    assert submission.job_url == "https://hf.co/jobs/job-123"
    kwargs = mock_hub.run_job.call_args.kwargs
    assert kwargs["image"] == "unsloth/test:latest"
    assert kwargs["command"] == ["bash", "-c", "echo hi"]
    assert kwargs["flavor"] == "a10g-small"
    assert kwargs["timeout"] == "4h"
    assert kwargs["secrets"] == {"HF_TOKEN": "hf_test_token_12345"}
    assert kwargs["env"] == {"FOO": "bar"}
    assert kwargs["labels"] == {"task": "gym"}


def test_sanitize_hf_job_labels_encodes_slash_values():
    """Labels containing slashes are encoded, not dropped."""
    labels = sanitize_hf_job_labels(
        {
            "task": "training",
            "provider": "hf_jobs",
            "bucket_id": "professorsynapse/toolset-training-artifacts",
            "artifact_prefix": "runs/hf_jobs/sft/20260322_103451-eafd2a89",
            "run_prefix": "20260322_103451-eafd2a89",
        }
    )

    assert labels == {
        "task": "training",
        "provider": "hf_jobs",
        "bucket_id": "professorsynapse=2F=toolset-training-artifacts",
        "artifact_prefix": "runs=2F=hf_jobs=2F=sft=2F=20260322_103451-eafd2a89",
        "run_prefix": "20260322_103451-eafd2a89",
    }


def test_decode_hf_job_label_roundtrip():
    """Encoding then decoding recovers the original value."""
    originals = [
        "professorsynapse/toolset-training-artifacts",
        "runs/hf_jobs/sft/20260322_103451-eafd2a89",
        "simple-value",
    ]
    encoded = sanitize_hf_job_labels(
        {f"k{i}": v for i, v in enumerate(originals)}
    )
    for key, enc_value in encoded.items():
        idx = int(key[1:])
        assert decode_hf_job_label(enc_value) == originals[idx]


def test_resolve_hf_bucket_id_returns_namespaced_bucket():
    mock_hub = MagicMock()
    mock_hub.create_bucket.return_value = SimpleNamespace(bucket_id="test-user/toolset-training-artifacts")

    bucket_id = resolve_hf_bucket_id(
        mock_hub,
        "toolset-training-artifacts",
        token="hf_test_token_12345",
    )

    assert bucket_id == "test-user/toolset-training-artifacts"
    mock_hub.create_bucket.assert_called_once_with(
        "toolset-training-artifacts",
        exist_ok=True,
        private=True,
        token="hf_test_token_12345",
    )


def _smoke_approval(*, hardware="cpu-basic"):
    return build_hf_run_approval(
        experiment_id="exp-1", run_id="run-1",
        descriptor_uri="tracking://runs/run-1/descriptor.json", descriptor_sha256="d" * 64,
        provisioning_evidence_uri="tracking://runs/run-1/evidence.json",
        provisioning_evidence_sha256="e" * 64,
        source_lock_uri="tracking://runs/run-1/source-lock.json", source_lock_sha256="a" * 64,
        bundle_sha256="b" * 64, capsule_manifest_sha256="c" * 64,
        checkout_policy_sha256="f" * 64,
        hardware_flavor=hardware, user_authorization_reference="thread:approved-smoke-1",
        issued_at="2026-08-20T12:00:00Z", expires_at="2026-08-20T13:00:00Z",
        hourly_price_usd="0.01", projected_cost_usd="0.01", quoted_at="2026-08-20T12:00:00Z",
    )


def _smoke_fakes(events, *, malformed=False):
    descriptor = {
        "bundle": {"content_sha256": "b" * 64},
        "capsule": {"manifest": {"sha256": "c" * 64}},
        "checkout_policy": {"sha256": "f" * 64},
    }
    spec = object()
    volume = object()
    preparation = SimpleNamespace(
        source_lock=SimpleNamespace(run_id="run-1"),
        descriptor_uri="tracking://runs/run-1/descriptor.json", descriptor_sha256="d" * 64,
        provisioning_evidence_uri="tracking://runs/run-1/evidence.json",
        provisioning_evidence_sha256="e" * 64,
        source_lock_uri="tracking://runs/run-1/source-lock.json", source_lock_sha256="a" * 64,
        volume_spec=spec,
        consumable_transport=SimpleNamespace(prepared=SimpleNamespace(descriptor=descriptor)),
    )
    preparation.prove_volume = lambda hub: (events.append("volume") or SimpleNamespace(
        provider_volume=volume, spec=spec
    ))
    experiment = SimpleNamespace(
        experiment_id="exp-1", hf_run_approval_uri="tracking://runs/run-1/approval.json",
        hf_submission_state="APPROVED", hf_submission_event_uri=None,
    )

    class Tracking:
        def claim_hf_submission(self, current, event):
            events.append("claim")
            current.hf_submission_state = "SUBMITTING"
            current.hf_submission_event_uri = "tracking://runs/run-1/submitting.json"

        def record_hf_submission_terminal(self, current, event):
            events.append(event.state.value)
            current.hf_submission_state = event.state.value

    def run_job(**kwargs):
        events.append("run_job")
        run_job.kwargs = kwargs
        if malformed:
            return SimpleNamespace(id="bad/id/extra")
        return SimpleNamespace(id="owner/job-1", owner=SimpleNamespace(name="owner"))

    provider = SimpleNamespace(run_job=run_job)
    return experiment, preparation, Tracking(), provider, volume, run_job


def test_protected_smoke_claims_before_token_sdk_volume_and_exact_single_submission(monkeypatch):
    events = []
    experiment, preparation, tracking, provider, volume, run_job = _smoke_fakes(events)
    monkeypatch.setattr("tuner.handlers.stages._util.hf_verified_source_steps", lambda value: ["verify"])
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.revalidate_hf_verified_volume",
        lambda value: SimpleNamespace(volume_spec=preparation.volume_spec),
    )

    result = submit_approved_bootstrap_smoke(
        tracking_service=tracking, experiment=experiment, approval=_smoke_approval(),
        preparation=preparation,
        token_factory=lambda: (events.append("token") or "provider-secret"),
        provider_factory=lambda token: (events.append("sdk") or provider),
        now=lambda: datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
    )

    assert result == HFBootstrapSmokeSubmission("owner", "job-1", _smoke_approval().authorization_id)
    assert events == ["claim", "token", "sdk", "volume", "run_job", "SUBMITTED"]
    assert run_job.kwargs == {
        "image": "python:3.12", "command": ["bash", "-c", "verify && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/engine $(command -v python3 || command -v python) -m tuner.cloud.hf_bootstrap_smoke"],
        "flavor": "cpu-basic", "timeout": "10m",
        "secrets": {"HF_TOKEN": "provider-secret"}, "token": "provider-secret",
        "volumes": [volume],
    }


def test_protected_smoke_claim_conflict_has_no_provider_effect(monkeypatch):
    events = []
    experiment, preparation, tracking, provider, _, _ = _smoke_fakes(events)
    tracking.claim_hf_submission = lambda *args: (_ for _ in ()).throw(RuntimeError("claimed"))
    monkeypatch.setattr("tuner.handlers.stages._util.hf_verified_source_steps", lambda value: ["verify"])
    with pytest.raises(RuntimeError, match="claimed"):
        submit_approved_bootstrap_smoke(
            tracking_service=tracking, experiment=experiment, approval=_smoke_approval(),
            preparation=preparation, token_factory=lambda: events.append("token") or "secret",
            provider_factory=lambda token: events.append("sdk") or provider,
            now=lambda: datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
        )
    assert events == []


def test_expensive_hardware_with_forged_cheap_quote_fails_before_claim(monkeypatch):
    events = []
    experiment, preparation, tracking, provider, _, _ = _smoke_fakes(events)
    monkeypatch.setattr("tuner.handlers.stages._util.hf_verified_source_steps", lambda value: ["verify"])
    with pytest.raises(CloudProviderError):
        approval = _smoke_approval(hardware="a100-large")
        submit_approved_bootstrap_smoke(
            tracking_service=tracking, experiment=experiment,
            approval=approval, preparation=preparation,
            token_factory=lambda: events.append("token") or "secret",
            provider_factory=lambda token: events.append("sdk") or provider,
            now=lambda: datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
        )
    assert events == []


@pytest.mark.parametrize(("section", "key", "value"), [
    ("execution", "image", "attacker/image:latest"),
    ("workload", "sha256", "9" * 64),
])
def test_wrong_image_or_workload_fails_before_claim(monkeypatch, section, key, value):
    events = []
    experiment, preparation, tracking, provider, _, _ = _smoke_fakes(events)
    document = _smoke_approval().to_dict()
    document[section][key] = value
    monkeypatch.setattr("tuner.handlers.stages._util.hf_verified_source_steps", lambda value: ["verify"])
    with pytest.raises(CloudProviderError):
        submit_approved_bootstrap_smoke(
            tracking_service=tracking, experiment=experiment, approval=document,
            preparation=preparation, token_factory=lambda: events.append("token") or "secret",
            provider_factory=lambda token: events.append("sdk") or provider,
            now=lambda: datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
        )
    assert events == []


def test_post_claim_malformed_job_is_ambiguous_and_never_retried(monkeypatch):
    events = []
    experiment, preparation, tracking, provider, _, _ = _smoke_fakes(events, malformed=True)
    monkeypatch.setattr("tuner.handlers.stages._util.hf_verified_source_steps", lambda value: ["verify"])
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.revalidate_hf_verified_volume",
        lambda value: SimpleNamespace(volume_spec=preparation.volume_spec),
    )
    with pytest.raises(CloudProviderError, match="ambiguous"):
        submit_approved_bootstrap_smoke(
            tracking_service=tracking, experiment=experiment, approval=_smoke_approval(),
            preparation=preparation, token_factory=lambda: "secret",
            provider_factory=lambda token: provider,
            now=lambda: datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
        )
    assert events.count("run_job") == 1
    assert experiment.hf_submission_state == "AMBIGUOUS"


@pytest.mark.parametrize(
    "job",
    [
        SimpleNamespace(
            id="embedded-owner/job-1",
            owner=SimpleNamespace(name="declared-owner"),
        ),
        SimpleNamespace(id="owner/job/extra", owner=SimpleNamespace(name="owner")),
        SimpleNamespace(id=" owner/job-1", owner=SimpleNamespace(name=" owner")),
        SimpleNamespace(id="owner/job 1", owner=SimpleNamespace(name="owner")),
        SimpleNamespace(id="owner/job:1", owner=SimpleNamespace(name="owner")),
        SimpleNamespace(id="owner-/job-1", owner=SimpleNamespace(name="owner-")),
        SimpleNamespace(id="owner/job-", owner=SimpleNamespace(name="owner")),
        SimpleNamespace(id="owner/", owner=SimpleNamespace(name="owner")),
        SimpleNamespace(id="owner/job-1", owner=SimpleNamespace(name="")),
        SimpleNamespace(id=f"{'o' * 97}/job-1", owner=None),
        SimpleNamespace(id=f"owner/{'j' * 257}", owner=None),
    ],
)
def test_job_info_identity_rejects_contradictory_or_malformed_representations(job):
    with pytest.raises(CloudProviderError, match="JobInfo"):
        _normalize_job_info(job)


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        (
            SimpleNamespace(id="owner/job-1", owner=SimpleNamespace(name="owner")),
            ("owner", "job-1"),
        ),
        (SimpleNamespace(id="owner/job-1", owner=None), ("owner", "job-1")),
        (
            SimpleNamespace(id="job-1", owner=SimpleNamespace(name="owner")),
            ("owner", "job-1"),
        ),
    ],
)
def test_job_info_identity_accepts_only_agreeing_normalized_representations(job, expected):
    assert _normalize_job_info(job) == expected


def test_post_claim_provider_secret_is_not_chained_or_persisted(monkeypatch):
    events = []
    experiment, preparation, tracking, provider, _, _ = _smoke_fakes(events)
    secret = "provider-secret-must-not-escape"
    provider.run_job = lambda **kwargs: (_ for _ in ()).throw(RuntimeError(secret))
    monkeypatch.setattr("tuner.handlers.stages._util.hf_verified_source_steps", lambda value: ["verify"])
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.revalidate_hf_verified_volume",
        lambda value: SimpleNamespace(volume_spec=preparation.volume_spec),
    )
    with pytest.raises(CloudProviderError) as caught:
        submit_approved_bootstrap_smoke(
            tracking_service=tracking, experiment=experiment, approval=_smoke_approval(),
            preparation=preparation, token_factory=lambda: secret,
            provider_factory=lambda token: provider,
            now=lambda: datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
        )
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert experiment.hf_submission_state == "AMBIGUOUS"


def test_post_claim_secret_parse_failure_is_ambiguous_before_sdk(monkeypatch):
    events = []
    experiment, preparation, tracking, provider, _, _ = _smoke_fakes(events)
    monkeypatch.setattr("tuner.handlers.stages._util.hf_verified_source_steps", lambda value: ["verify"])
    with pytest.raises(CloudProviderError, match="ambiguous"):
        submit_approved_bootstrap_smoke(
            tracking_service=tracking, experiment=experiment, approval=_smoke_approval(),
            preparation=preparation,
            token_factory=lambda: (_ for _ in ()).throw(
                CloudProviderError("strict secret document rejected")
            ),
            provider_factory=lambda token: events.append("sdk") or provider,
            now=lambda: datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
        )
    assert events == ["claim", "AMBIGUOUS"]
    assert experiment.hf_submission_state == "AMBIGUOUS"


def test_observation_cancels_at_most_once_and_stops_at_outer_bound(monkeypatch):
    current = [0.0]
    cancels = []
    provider = SimpleNamespace(
        inspect_job=lambda **kwargs: SimpleNamespace(status=SimpleNamespace(stage="RUNNING")),
        cancel_job=lambda **kwargs: cancels.append(kwargs),
    )
    approval = _smoke_approval()
    experiment = SimpleNamespace(
        hf_submission_state="SUBMITTED",
        hf_authorization_id=approval.authorization_id,
        hf_submission_event_uri="tracking://runs/run-1/submitted.json",
    )
    tracking = SimpleNamespace(
        verify_hf_submission_provenance=lambda value: None,
        resolve_uri=lambda value: Path("submitted.json"),
        build_hf_cancellation_attempt_event=lambda *args, **kwargs: {"event": "cancel"},
        claim_hf_cancellation=lambda *args, **kwargs: SimpleNamespace(
            provider_attempt_authorized=True
        ),
    )
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.load_canonical_json",
        lambda *args, **kwargs: {
            "provider_job": {"namespace": "owner", "job_id": "job-1"},
            "occurred_at": "2026-08-20T12:01:00Z",
        },
    )
    result = observe_submitted_bootstrap_smoke(
        HFBootstrapSmokeSubmission("owner", "job-1", approval.authorization_id),
        tracking_service=tracking, experiment=experiment, approval=approval,
        token_factory=lambda: "secret", provider_factory=lambda token: provider,
        wall_clock=lambda: datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
        monotonic=lambda: current[0], sleep=lambda seconds: current.__setitem__(0, current[0] + seconds),
        poll_seconds=720,
    )
    assert result.elapsed_seconds == 900
    assert result.cancel_attempted is True
    assert len(cancels) == 1
    assert cancels[0]["job_id"] == "job-1"


def test_two_resumed_observers_share_one_durable_cancellation_attempt(monkeypatch):
    cancels = []
    claims = []
    order = []
    approval = _smoke_approval()
    experiment = SimpleNamespace(
        hf_submission_state="SUBMITTED",
        hf_authorization_id=approval.authorization_id,
        hf_submission_event_uri="tracking://runs/run-1/submitted.json",
    )

    class Tracking:
        def verify_hf_submission_provenance(self, value):
            return None

        def resolve_uri(self, value):
            return Path("submitted.json")

        def build_hf_cancellation_attempt_event(self, value, *, occurred_at):
            return {"occurred_at": occurred_at, "provider_job": ("owner", "job-1")}

        def claim_hf_cancellation(self, value, event):
            claims.append(event)
            order.append("claim")
            return SimpleNamespace(provider_attempt_authorized=len(claims) == 1)

    provider = SimpleNamespace(
        inspect_job=lambda **kwargs: SimpleNamespace(
            status=SimpleNamespace(stage="RUNNING")
        ),
        cancel_job=lambda **kwargs: (
            cancels.append(kwargs)
            or order.append("cancel")
            or (_ for _ in ()).throw(RuntimeError("ambiguous provider cancel"))
        ),
    )
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.load_canonical_json",
        lambda *args, **kwargs: {
            "provider_job": {"namespace": "owner", "job_id": "job-1"},
            "occurred_at": "2026-08-20T12:01:00Z",
        },
    )

    results = []
    for _ in range(2):
        current = [0.0]
        results.append(observe_submitted_bootstrap_smoke(
            HFBootstrapSmokeSubmission("owner", "job-1", approval.authorization_id),
            tracking_service=Tracking(), experiment=experiment, approval=approval,
            token_factory=lambda: order.append("token") or "secret",
            provider_factory=lambda token: order.append("provider") or provider,
            wall_clock=lambda: datetime(2026, 8, 20, 12, 13, tzinfo=timezone.utc),
            monotonic=lambda: current[0],
            sleep=lambda seconds: current.__setitem__(0, current[0] + seconds),
            poll_seconds=180,
        ))

    assert len(claims) == 2
    assert claims[0] == claims[1]
    assert len(cancels) == 1
    assert all(result.cancel_attempted is True for result in results)
    assert order == [
        "claim", "token", "provider", "cancel",
        "claim", "token", "provider",
    ]


def test_late_observer_claim_failure_has_no_token_provider_or_cancel_effect(monkeypatch):
    effects = []
    approval = _smoke_approval()
    experiment = SimpleNamespace(
        hf_submission_state="SUBMITTED",
        hf_authorization_id=approval.authorization_id,
        hf_submission_event_uri="tracking://runs/run-1/submitted.json",
    )
    tracking = SimpleNamespace(
        verify_hf_submission_provenance=lambda value: None,
        resolve_uri=lambda value: Path("submitted.json"),
        build_hf_cancellation_attempt_event=lambda *args, **kwargs: {"event": "cancel"},
        claim_hf_cancellation=lambda *args, **kwargs: (
            effects.append("claim")
            or (_ for _ in ()).throw(RuntimeError("durable claim unavailable"))
        ),
    )
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.load_canonical_json",
        lambda *args, **kwargs: {
            "provider_job": {"namespace": "owner", "job_id": "job-1"},
            "occurred_at": "2026-08-20T12:01:00Z",
        },
    )

    with pytest.raises(CloudProviderError, match="durably claimed"):
        observe_submitted_bootstrap_smoke(
            HFBootstrapSmokeSubmission("owner", "job-1", approval.authorization_id),
            tracking_service=tracking, experiment=experiment, approval=approval,
            token_factory=lambda: effects.append("token") or "secret",
            provider_factory=lambda token: effects.append("provider"),
            wall_clock=lambda: datetime(2026, 8, 20, 12, 13, tzinfo=timezone.utc),
        )

    assert effects == ["claim"]


def test_observer_has_no_claim_token_or_provider_effect_before_cancel_boundary(monkeypatch):
    effects = []
    approval = _smoke_approval()
    experiment = SimpleNamespace(
        hf_submission_state="SUBMITTED",
        hf_authorization_id=approval.authorization_id,
        hf_submission_event_uri="tracking://runs/run-1/submitted.json",
    )
    tracking = SimpleNamespace(
        verify_hf_submission_provenance=lambda value: None,
        resolve_uri=lambda value: Path("submitted.json"),
        build_hf_cancellation_attempt_event=lambda *args, **kwargs: effects.append("build"),
        claim_hf_cancellation=lambda *args, **kwargs: effects.append("claim"),
    )
    monkeypatch.setattr(
        "tuner.cloud.hf_provisioning.load_canonical_json",
        lambda *args, **kwargs: {
            "provider_job": {"namespace": "owner", "job_id": "job-1"},
            "occurred_at": "2026-08-20T12:01:00Z",
        },
    )

    with pytest.raises(RuntimeError, match="paused before boundary"):
        observe_submitted_bootstrap_smoke(
            HFBootstrapSmokeSubmission("owner", "job-1", approval.authorization_id),
            tracking_service=tracking, experiment=experiment, approval=approval,
            token_factory=lambda: effects.append("token") or "secret",
            provider_factory=lambda token: effects.append("provider"),
            wall_clock=lambda: datetime(2026, 8, 20, 12, 1, tzinfo=timezone.utc),
            sleep=lambda seconds: (_ for _ in ()).throw(
                RuntimeError("paused before boundary")
            ),
        )

    assert effects == []


def test_observation_rejects_untracked_job_before_token_or_provider():
    events = []
    approval = _smoke_approval()
    experiment = SimpleNamespace(
        hf_submission_state="APPROVED", hf_authorization_id=approval.authorization_id,
        hf_submission_event_uri=None,
    )
    with pytest.raises(CloudProviderError, match="SUBMITTED"):
        observe_submitted_bootstrap_smoke(
            HFBootstrapSmokeSubmission("owner", "other-job", approval.authorization_id),
            tracking_service=SimpleNamespace(verify_hf_submission_provenance=lambda value: None),
            experiment=experiment, approval=approval,
            token_factory=lambda: events.append("token") or "secret",
            provider_factory=lambda token: events.append("sdk"),
        )
    assert events == []
