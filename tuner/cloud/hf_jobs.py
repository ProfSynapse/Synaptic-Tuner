"""Generic Hugging Face Jobs primitives for cloud task execution."""

from __future__ import annotations

import importlib
import shlex
import re
import time
from urllib.parse import urlparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, NoReturn, Optional

from shared.utilities.env import get_env_var, get_hf_token
from tuner.core.exceptions import CloudProviderError

HF_BUCKET_SYNC_OVERLAY_PACKAGES: tuple[str, ...] = (
    "huggingface_hub>=1.5.0",
    "hf_transfer",
    "hf_xet",
)


@dataclass(frozen=True)
class RepoCheckoutSpec:
    """Exact repository source needed to reproduce a cloud job."""

    url: str
    branch: str
    commit: str
    clone_dir: str = "/workspace/repo"


@dataclass(frozen=True)
class CloudJobSpec:
    """Provider-agnostic cloud job description."""

    provider: str
    image: str
    command: List[str]
    flavor: str
    timeout_hours: Optional[float] = None
    env: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    namespace: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    # Generic jobs may omit volumes. Secure source-launch routes must supply a
    # semantically proven read-only volume from hf_volume_transport.
    volumes: tuple[Any, ...] = ()


@dataclass(frozen=True)
class HFJobSubmission:
    """Normalized response from a submitted HF Job."""

    job_id: str
    job_url: Optional[str] = None
    raw: Any = None


@dataclass(frozen=True)
class HFBootstrapSmokeSubmission:
    """Sanitized identity of the one protected bootstrap-smoke submission."""

    namespace: str
    job_id: str
    authorization_id: str


@dataclass(frozen=True)
class HFBootstrapSmokeObservation:
    """Bounded, sanitized observation result for one exact submitted job."""

    namespace: str
    job_id: str
    stage: str
    elapsed_seconds: int
    cancel_attempted: bool
    result: Mapping[str, object] | None = None


def require_current_hf_source_submission_authorization(*, route: str) -> NoReturn:
    """Fail closed until an exact-run secure-source approval contract exists.

    This is the single current-authorization barrier for every HF route that
    would consume a verified source volume.  Callers must invoke it before SDK
    import, credential resolution, bucket access, provider-volume construction,
    or config-derived command compilation.  Generic jobs without source
    volumes remain isolated in :class:`HFJobExecutor`'s no-volume seam.
    """

    if not isinstance(route, str) or not route.strip():
        raise CloudProviderError("HF secure source authorization route is invalid.")
    raise CloudProviderError(
        "HF secure source submission requires a separately authorized exact-run approval; "
        "no approval contract is implemented and no provider-facing operation was performed."
    )


def load_huggingface_hub(*, require_apis: Iterable[str] = ()) -> Any:
    """Import and validate the Hugging Face Hub SDK."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise CloudProviderError(
            "huggingface_hub not installed. Install with: pip install -r requirements-cloud.txt"
        ) from exc

    missing = [name for name in require_apis if not hasattr(huggingface_hub, name)]
    if missing:
        version = getattr(huggingface_hub, "__version__", "unknown")
        labels = []
        if "run_job" in missing:
            labels.append("Jobs API (run_job)")
        if "create_bucket" in missing:
            labels.append("Buckets API (create_bucket)")
        for name in missing:
            if name not in {"run_job", "create_bucket"}:
                labels.append(name)
        raise CloudProviderError(
            f"huggingface_hub {version} does not support required APIs: {', '.join(labels)}"
        )

    return huggingface_hub


def build_hf_job_secrets(token: Optional[str] = None) -> Dict[str, str]:
    """Build the standard HF secret payload for remote jobs."""
    resolved = (token or get_hf_token() or "").strip()
    if not resolved:
        return {}
    return {
        "HF_TOKEN": resolved,
        "HF_API_KEY": resolved,
    }


def build_secrets_from_env(secret_names: Iterable[str]) -> Dict[str, str]:
    """Build a secrets payload from selected local environment variables."""
    secrets: Dict[str, str] = {}
    for name in secret_names:
        key = str(name).strip()
        if not key:
            continue
        value = get_env_var(key)
        if value is None:
            continue
        value = value.strip()
        if value:
            secrets[key] = value
    return secrets


def format_timeout_hours(timeout_hours: Optional[float]) -> Optional[str]:
    """Format timeout hours for the Jobs API."""
    if timeout_hours is None:
        return None
    timeout = float(timeout_hours)
    if timeout.is_integer():
        return f"{int(timeout)}h"
    return f"{timeout}h"


def build_repo_checkout_steps(repo: RepoCheckoutSpec) -> List[str]:
    """Build shell steps that clone and pin the exact requested commit."""
    if not repo.url or not repo.branch or not repo.commit:
        raise CloudProviderError("Cloud jobs require exact repo source metadata.")

    quoted_branch = shlex.quote(repo.branch)
    quoted_url = shlex.quote(repo.url)
    quoted_commit = shlex.quote(repo.commit)
    quoted_dir = shlex.quote(repo.clone_dir)
    archive_url = _github_archive_url(repo.url, repo.commit)
    python_cmd = _shell_python_command()
    if archive_url:
        clone_or_download = (
            f"if command -v git >/dev/null 2>&1; then "
            f"git clone --branch {quoted_branch} --depth 1 {quoted_url} {quoted_dir}; "
            f"else "
            f"{python_cmd} -c \"import io, pathlib, shutil, tarfile, urllib.request; "
            f"url={archive_url!r}; "
            f"target=pathlib.Path({repo.clone_dir!r}); "
            f"target.parent.mkdir(parents=True, exist_ok=True); "
            f"data=urllib.request.urlopen(url).read(); "
            f"tmp=target.parent / (target.name + '-tmp'); "
            f"shutil.rmtree(tmp, ignore_errors=True); "
            f"tmp.mkdir(parents=True, exist_ok=True); "
            f"archive=tarfile.open(fileobj=io.BytesIO(data), mode='r:gz'); "
            f"archive.extractall(tmp); "
            f"entries=[p for p in tmp.iterdir() if p.is_dir()]; "
            f"root=entries[0] if len(entries)==1 else tmp; "
            f"shutil.rmtree(target, ignore_errors=True); "
            f"shutil.move(str(root), str(target)); "
            f"shutil.rmtree(tmp, ignore_errors=True)\"; "
            f"fi"
        )
    else:
        clone_or_download = f"git clone --branch {quoted_branch} --depth 1 {quoted_url} {quoted_dir}"
    return [
        clone_or_download,
        f"if [ -d {quoted_dir}/.git ]; then cd {quoted_dir} && git fetch --depth 1 origin {quoted_commit} && git checkout {quoted_commit}; fi",
    ]


def _github_archive_url(repo_url: str, commit: str) -> Optional[str]:
    """Return a tarball URL for GitHub repos, or None when unsupported."""
    cleaned = str(repo_url or "").strip()
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    parsed = urlparse(cleaned)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        return None
    path = parsed.path.strip("/")
    if path.count("/") != 1:
        return None
    return f"https://github.com/{path}/archive/{commit}.tar.gz"


def _shell_python_command() -> str:
    """Resolve a Python interpreter path safely in HF job shell snippets."""
    return '$(command -v python3 || command -v python)'


def build_bash_command(steps: Iterable[str]) -> List[str]:
    """Wrap shell steps in a bash command suitable for Jobs APIs."""
    return ["bash", "-c", " && ".join(step for step in steps if step)]


def resolve_hf_bucket_id(
    huggingface_hub: Any,
    bucket_id: str,
    *,
    token: Optional[str] = None,
    private: bool = True,
) -> str:
    """Resolve or create a Hugging Face bucket and return its namespaced id."""
    # Import only after the secure launch authorization boundary.  The shared
    # artifact module carries optional ML/provider dependencies that must not
    # load merely because an HF handler or this primitives module is imported.
    from shared.cloud_artifacts import normalize_hf_bucket_id

    requested_bucket_id = normalize_hf_bucket_id(bucket_id)
    if not requested_bucket_id:
        raise CloudProviderError("HF bucket identifier is required.")
    if not hasattr(huggingface_hub, "create_bucket"):
        version = getattr(huggingface_hub, "__version__", "unknown")
        raise CloudProviderError(
            f"huggingface_hub {version} does not support Buckets API. "
            "Upgrade with: pip install --upgrade huggingface_hub>=1.5.0"
        )

    try:
        try:
            bucket_info = huggingface_hub.create_bucket(
                requested_bucket_id,
                exist_ok=True,
                private=private,
                token=token,
            )
        except TypeError:
            bucket_info = huggingface_hub.create_bucket(
                requested_bucket_id,
                exist_ok=True,
                token=token,
            )
    except Exception as exc:
        error_msg = str(exc)
        if "hf_" in error_msg:
            error_msg = "check credentials and subscription"
        raise CloudProviderError(
            f"Failed to create or resolve HF bucket '{requested_bucket_id}': {error_msg}"
        ) from exc

    resolved = getattr(bucket_info, "bucket_id", None) or getattr(bucket_info, "id", None) or requested_bucket_id
    return normalize_hf_bucket_id(str(resolved))


class HFJobExecutor:
    """Shared HF Jobs submitter for training, evaluation, and future tasks."""

    def __init__(self, huggingface_hub: Any):
        self.huggingface_hub = huggingface_hub

    def submit(self, spec: CloudJobSpec) -> HFJobSubmission:
        """Submit a generic cloud job to Hugging Face Jobs."""
        if spec.volumes:
            from tuner.cloud.hf_volume_transport import HFVerifiedVolume
            from tuner.cloud.hf_provisioning import revalidate_hf_verified_volume

            # Local provenance validation is not authorization and has no
            # provider effect.  It may precede the barrier so malformed inputs
            # retain a precise fail-closed error, but nothing provider-facing
            # is assembled or invoked before current authorization.
            for volume in spec.volumes:
                if not isinstance(volume, HFVerifiedVolume):
                    raise CloudProviderError(
                        "HF source volumes require exact CONSUMABLE descriptor/evidence binding."
                    )
                revalidate_hf_verified_volume(volume)
            require_current_hf_source_submission_authorization(route="executor.submit")

        kwargs: Dict[str, Any] = {
            "image": spec.image,
            "command": spec.command,
            "flavor": spec.flavor,
        }
        timeout = format_timeout_hours(spec.timeout_hours)
        if timeout:
            kwargs["timeout"] = timeout
        if spec.secrets:
            kwargs["secrets"] = spec.secrets
        if spec.env:
            kwargs["env"] = spec.env
        if spec.namespace:
            kwargs["namespace"] = spec.namespace
        if spec.labels:
            sanitized_labels = sanitize_hf_job_labels(spec.labels)
            if sanitized_labels:
                kwargs["labels"] = sanitized_labels
        try:
            job = self.huggingface_hub.run_job(**kwargs)
        except Exception as exc:
            error_msg = str(exc)
            for secret in spec.secrets.values():
                if secret:
                    error_msg = error_msg.replace(secret, "[REDACTED]")
            if "hf_" in error_msg:
                error_msg = "Job submission failed (check credentials and subscription)"
            raise CloudProviderError(f"Failed to submit HF Job: {error_msg}") from exc

        job_id = job.id if hasattr(job, "id") else str(job)
        return HFJobSubmission(
            job_id=job_id,
            job_url=getattr(job, "url", None),
            raw=job,
        )


def submit_approved_bootstrap_smoke(
    *,
    tracking_service: Any,
    experiment: Any,
    approval: Any,
    preparation: Any,
    token_factory: Callable[[], str],
    provider_factory: Callable[[str], Any] | None = None,
    now: Callable[[], datetime] | None = None,
) -> HFBootstrapSmokeSubmission:
    """Consume one exact approval and submit only the fixed bootstrap smoke.

    Provider-free validation and deterministic command compilation precede the
    durable claim.  Every provider-facing operation, including token lookup,
    SDK import, and ``Volume`` construction, happens only after SUBMITTING is
    durably recorded.  There is deliberately no retry path.
    """

    from tuner.cloud.hf_bootstrap_smoke import WORKLOAD_SHA256
    from tuner.cloud.hf_run_approval import (
        HFSubmissionState,
        build_hf_ambiguous_event,
        build_hf_submitted_event,
        build_hf_submitting_event,
        validate_hf_run_approval,
    )
    from tuner.cloud.hf_provisioning import revalidate_hf_verified_volume
    from tuner.handlers.stages._util import hf_verified_source_steps

    clock = now or (lambda: datetime.now(timezone.utc))
    accepted = validate_hf_run_approval(approval, at=clock())
    _validate_bootstrap_smoke_bindings(
        experiment=experiment,
        approval=accepted,
        preparation=preparation,
        workload_sha256=WORKLOAD_SHA256,
    )
    steps = hf_verified_source_steps(preparation)
    steps.append(
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/engine "
        "$(command -v python3 || command -v python) -m tuner.cloud.hf_bootstrap_smoke"
    )
    command = build_bash_command(steps)
    approval_uri = str(experiment.hf_run_approval_uri)
    submitting = build_hf_submitting_event(
        accepted,
        approval_uri=approval_uri,
        occurred_at=clock(),
    )
    tracking_service.claim_hf_submission(experiment, submitting)
    submitting_uri = str(experiment.hf_submission_event_uri)

    try:
        if experiment.hf_submission_state != HFSubmissionState.SUBMITTING.value:
            raise CloudProviderError("HF bootstrap smoke claim was not durably projected.")
        token = token_factory()
        if not isinstance(token, str) or not token.strip():
            raise CloudProviderError("HF bootstrap smoke requires the approved HF_TOKEN SecretRef.")
        token = token.strip()
        hub = (provider_factory or _load_hf_bootstrap_smoke_provider)(token)
        verified_volume = preparation.prove_volume(hub)
        consumed = revalidate_hf_verified_volume(verified_volume)
        _validate_bootstrap_smoke_bindings(
            experiment=experiment,
            approval=accepted,
            preparation=preparation,
            workload_sha256=WORKLOAD_SHA256,
            consumed=consumed,
        )
        kwargs = {
            "image": "python:3.12",
            "command": command,
            "flavor": "cpu-basic",
            "timeout": "10m",
            "secrets": {"HF_TOKEN": token},
            "token": token,
            "volumes": [verified_volume.provider_volume],
        }
        job = hub.run_job(**kwargs)
        namespace, job_id = _normalize_job_info(job)
        terminal = build_hf_submitted_event(
            accepted,
            approval_uri=approval_uri,
            previous_event=submitting,
            previous_event_uri=submitting_uri,
            occurred_at=clock(),
            provider_namespace=namespace,
            provider_job_id=job_id,
        )
        tracking_service.record_hf_submission_terminal(experiment, terminal)
        return HFBootstrapSmokeSubmission(
            namespace=namespace,
            job_id=job_id,
            authorization_id=accepted.authorization_id,
        )
    except Exception as exc:
        ambiguous = build_hf_ambiguous_event(
            accepted,
            approval_uri=approval_uri,
            previous_event=submitting,
            previous_event_uri=submitting_uri,
            occurred_at=clock(),
            reason_code=_ambiguous_reason(exc),
        )
        try:
            tracking_service.record_hf_submission_terminal(experiment, ambiguous)
        except Exception as terminal_exc:
            raise CloudProviderError(
                "HF bootstrap smoke outcome is ambiguous and its terminal record could not be persisted."
            ) from terminal_exc
        raise CloudProviderError(
            "HF bootstrap smoke outcome is ambiguous; the authorization is consumed and must not be retried."
        ) from None


def observe_submitted_bootstrap_smoke(
    submission: HFBootstrapSmokeSubmission,
    *,
    tracking_service: Any,
    experiment: Any,
    approval: Any,
    token_factory: Callable[[], str],
    provider_factory: Callable[[str], Any] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 15.0,
) -> HFBootstrapSmokeObservation:
    """Observe one exact job, cancel at most once after 12m, and stop by 15m."""

    from tuner.cloud.hf_bootstrap_smoke import canonical_result_bytes
    from tuner.cloud.hf_provisioning import load_canonical_json
    from tuner.cloud.hf_run_approval import validate_hf_run_approval

    accepted = validate_hf_run_approval(approval)
    tracking_service.verify_hf_submission_provenance(experiment)
    if (
        experiment.hf_submission_state != "SUBMITTED"
        or experiment.hf_authorization_id != submission.authorization_id
        or accepted.authorization_id != submission.authorization_id
    ):
        raise CloudProviderError("HF bootstrap smoke observation is not bound to SUBMITTED state.")
    event_path = tracking_service.resolve_uri(str(experiment.hf_submission_event_uri))
    event = load_canonical_json(event_path, maximum_bytes=64 * 1024)
    if event.get("provider_job") != {
        "namespace": submission.namespace,
        "job_id": submission.job_id,
    }:
        raise CloudProviderError("HF bootstrap smoke observation job binding changed.")
    occurred_at = event.get("occurred_at")
    try:
        submitted_at = datetime.fromisoformat(str(occurred_at).replace("Z", "+00:00"))
        current_wall = (wall_clock or (lambda: datetime.now(timezone.utc)))()
        if submitted_at.tzinfo is None or current_wall.tzinfo is None:
            raise ValueError("timezone missing")
        initial_elapsed = max(0, int((current_wall - submitted_at).total_seconds()))
    except Exception:
        raise CloudProviderError("HF bootstrap smoke submitted timestamp is invalid.") from None
    if accepted.document.get("timeouts") != {
        "provider_seconds": 600,
        "cancel_after_seconds": 720,
        "observe_until_seconds": 900,
    }:
        raise CloudProviderError("HF bootstrap smoke observation limits changed.")

    token: str | None = None
    hub: Any | None = None

    def require_provider() -> tuple[str, Any]:
        nonlocal token, hub
        if hub is None:
            resolved_token = token_factory()
            if not isinstance(resolved_token, str) or not resolved_token.strip():
                raise CloudProviderError("HF bootstrap smoke observation requires HF_TOKEN.")
            token = resolved_token.strip()
            hub = (provider_factory or _load_hf_bootstrap_smoke_provider)(token)
        return token or "", hub

    start = monotonic()
    cancel_attempted = False
    cancellation_claim_checked = False
    cancellation_authorized = False
    stage = "UNKNOWN"
    terminal = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED"}
    while True:
        elapsed = initial_elapsed + max(0, int(monotonic() - start))
        if elapsed >= 900:
            return HFBootstrapSmokeObservation(
                namespace=submission.namespace,
                job_id=submission.job_id,
                stage=stage,
                elapsed_seconds=elapsed,
                cancel_attempted=cancel_attempted,
            )
        if elapsed < 720:
            sleep(max(0.1, min(float(poll_seconds), 720 - elapsed)))
            continue
        if elapsed >= 720 and not cancellation_claim_checked:
            cancellation_claim_checked = True
            cancellation_at = (submitted_at + timedelta(seconds=720)).astimezone(
                timezone.utc
            ).isoformat().replace("+00:00", "Z")
            try:
                cancellation_event = tracking_service.build_hf_cancellation_attempt_event(
                    experiment,
                    occurred_at=cancellation_at,
                )
                cancellation_claim = tracking_service.claim_hf_cancellation(
                    experiment,
                    cancellation_event,
                )
                authorized = cancellation_claim.provider_attempt_authorized
                if type(authorized) is not bool:
                    raise TypeError("invalid cancellation claim result")
                cancellation_authorized = authorized
                cancel_attempted = True
            except Exception:
                raise CloudProviderError(
                    "HF bootstrap smoke cancellation authority could not be durably claimed."
                ) from None
        active_token, active_hub = require_provider()
        try:
            job = active_hub.inspect_job(
                job_id=submission.job_id,
                namespace=submission.namespace,
                token=active_token,
            )
            stage = _normalized_job_stage(job)
        except Exception:
            stage = "UNKNOWN"
        if stage in terminal:
            result = None
            if stage == "COMPLETED":
                result = _read_sanitized_smoke_result(
                    active_hub,
                    token=active_token,
                    namespace=submission.namespace,
                    job_id=submission.job_id,
                    validator=canonical_result_bytes,
                )
            return HFBootstrapSmokeObservation(
                namespace=submission.namespace,
                job_id=submission.job_id,
                stage=stage,
                elapsed_seconds=elapsed,
                cancel_attempted=cancel_attempted,
                result=result,
            )
        if cancellation_authorized:
            cancellation_authorized = False
            try:
                active_hub.cancel_job(
                    job_id=submission.job_id,
                    namespace=submission.namespace,
                    token=active_token,
                )
            except Exception:
                pass
        sleep(max(0.1, min(float(poll_seconds), 900 - elapsed)))


def _validate_bootstrap_smoke_bindings(
    *, experiment: Any, approval: Any, preparation: Any, workload_sha256: str,
    consumed: Any | None = None,
) -> None:
    document = approval.document
    expected = {
        "experiment_id": experiment.experiment_id,
        "run_id": preparation.source_lock.run_id,
        "descriptor": {
            "uri": preparation.descriptor_uri,
            "sha256": preparation.descriptor_sha256,
        },
        "provisioning_evidence": {
            "uri": preparation.provisioning_evidence_uri,
            "sha256": preparation.provisioning_evidence_sha256,
        },
        "source_lock": {
            "uri": preparation.source_lock_uri,
            "sha256": preparation.source_lock_sha256,
        },
    }
    mismatch = next((key for key, value in expected.items() if document.get(key) != value), None)
    if mismatch:
        raise CloudProviderError(f"HF bootstrap smoke approval binding changed: {mismatch}.")
    workload = document.get("workload", {})
    if (
        not isinstance(workload, Mapping)
        or workload.get("kind") != "bootstrap_verification"
        or workload.get("sha256") != workload_sha256
    ):
        raise CloudProviderError("HF bootstrap smoke approval binding changed: workload.")
    descriptor = preparation.consumable_transport.prepared.descriptor
    if document.get("bundle_sha256") != descriptor["bundle"]["content_sha256"]:
        raise CloudProviderError("HF bootstrap smoke bundle approval binding changed.")
    if document.get("capsule_manifest_sha256") != descriptor["capsule"]["manifest"]["sha256"]:
        raise CloudProviderError("HF bootstrap smoke capsule approval binding changed.")
    if document.get("checkout_policy_sha256") != descriptor["checkout_policy"]["sha256"]:
        raise CloudProviderError("HF bootstrap smoke policy approval binding changed.")
    execution = document.get("execution", {})
    timeouts = document.get("timeouts", {})
    if (
        execution.get("image") != "python:3.12"
        or execution.get("hardware", {}).get("flavor") != "cpu-basic"
        or execution.get("ssh") is not False
        or execution.get("ports") != []
        or execution.get("maximum_submissions") != 1
        or execution.get("retry_count") != 0
        or timeouts != {
            "provider_seconds": 600,
            "cancel_after_seconds": 720,
            "observe_until_seconds": 900,
        }
    ):
        raise CloudProviderError("HF bootstrap smoke execution envelope changed.")
    if consumed is not None and consumed.volume_spec != preparation.volume_spec:
        raise CloudProviderError("HF bootstrap smoke verified volume changed after claim.")


def _load_hf_bootstrap_smoke_provider(token: str) -> Any:
    """Import the exact isolated Jobs client only after the durable claim."""

    try:
        hub = importlib.import_module("huggingface_hub")
        if str(getattr(hub, "__version__", "")) != "1.27.0":
            raise ValueError("version drift")
        for name in ("Volume", "run_job", "inspect_job", "cancel_job", "fetch_job_logs"):
            if not callable(getattr(hub, name, None)):
                raise ValueError("API drift")
    except Exception:
        raise CloudProviderError("The isolated HF bootstrap-smoke launcher is unavailable or incompatible.") from None
    return hub


def _normalize_job_info(job: Any) -> tuple[str, str]:
    raw_id = getattr(job, "id", None)
    declared_owner = getattr(getattr(job, "owner", None), "name", None)
    embedded_owner: str | None = None
    if isinstance(raw_id, str) and "/" in raw_id:
        if raw_id.count("/") != 1:
            raise CloudProviderError("HF Jobs returned malformed JobInfo identity.")
        embedded_owner, raw_id = raw_id.split("/", 1)
    if embedded_owner is not None and declared_owner is not None:
        if declared_owner != embedded_owner:
            raise CloudProviderError("HF Jobs returned contradictory JobInfo ownership.")
    owner = declared_owner if declared_owner is not None else embedded_owner
    if not _valid_provider_identity_segment(owner, maximum=96):
        raise CloudProviderError("HF Jobs returned malformed JobInfo ownership.")
    if not _valid_provider_identity_segment(raw_id, maximum=256):
        raise CloudProviderError("HF Jobs returned malformed JobInfo identity.")
    return owner, raw_id


def _valid_provider_identity_segment(value: Any, *, maximum: int) -> bool:
    """Accept one exact bounded ASCII provider-identity segment."""

    return (
        isinstance(value, str)
        and 1 <= len(value) <= maximum
        and value == value.strip()
        and re.fullmatch(
            r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?",
            value,
        ) is not None
    )


def _normalized_job_stage(job: Any) -> str:
    stage = getattr(getattr(job, "status", None), "stage", None)
    normalized = str(getattr(stage, "value", stage) or "UNKNOWN").upper()
    return normalized if re.fullmatch(r"[A-Z][A-Z0-9_]{0,31}", normalized) else "UNKNOWN"


def _read_sanitized_smoke_result(
    hub: Any, *, token: str, namespace: str, job_id: str, validator: Callable[[Any], bytes],
) -> Mapping[str, object] | None:
    import json

    try:
        logs = hub.fetch_job_logs(
            job_id=job_id, namespace=namespace, follow=False, token=token
        )
        lines = logs.splitlines() if isinstance(logs, str) else list(logs)
        for line in reversed(lines[-64:]):
            text = str(line).strip()
            if not text.startswith("{") or len(text.encode("utf-8")) > 4096:
                continue
            parsed = json.loads(text)
            validator(parsed)
            return parsed
    except Exception:
        return None
    return None


def _ambiguous_reason(exc: Exception) -> str:
    if isinstance(exc, CloudProviderError) and "malformed JobInfo" in str(exc):
        return "PROVIDER_RESPONSE_UNKNOWN"
    return "PROVIDER_CALL_OUTCOME_UNKNOWN"


_HF_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_=-]+$")

_LABEL_SLASH_REPLACEMENT = "=2F="


def _encode_label_value(value: str) -> str:
    """Encode a label value so it conforms to HF Jobs label validation rules.

    Only encodes forward slashes (the primary invalid character in bucket IDs
    and artifact prefixes). Uses '=2F=' as the replacement so the value stays
    within HF's current label charset while remaining reversible.
    """
    return value.replace("/", _LABEL_SLASH_REPLACEMENT)


def _decode_label_value(value: str) -> str:
    """Reverse the encoding applied by _encode_label_value."""
    return value.replace(_LABEL_SLASH_REPLACEMENT, "/")


def sanitize_hf_job_labels(labels: Dict[str, str]) -> Dict[str, str]:
    """Encode label entries to conform to HF Jobs label validation rules.

    Values containing slashes (e.g. bucket ids, artifact prefixes) are encoded
    so they pass HF validation while remaining recoverable via decode_hf_job_label().
    Entries that cannot be made valid after encoding are dropped.
    """
    sanitized: Dict[str, str] = {}
    for raw_key, raw_value in labels.items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if not key or not value:
            continue
        if not _HF_LABEL_PATTERN.fullmatch(key):
            continue
        encoded = _encode_label_value(value)
        if not _HF_LABEL_PATTERN.fullmatch(encoded):
            continue
        sanitized[key] = encoded
    return sanitized


def decode_hf_job_label(value: str) -> str:
    """Decode a label value that was encoded by sanitize_hf_job_labels."""
    return _decode_label_value(value)
