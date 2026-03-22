"""Generic Hugging Face Jobs primitives for cloud task execution."""

from __future__ import annotations

import shlex
import re
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from shared.cloud_artifacts import normalize_hf_bucket_id
from shared.utilities.env import get_env_var, get_hf_token
from tuner.core.exceptions import CloudProviderError


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


@dataclass(frozen=True)
class HFJobSubmission:
    """Normalized response from a submitted HF Job."""

    job_id: str
    job_url: Optional[str] = None
    raw: Any = None


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
            if "hf_" in error_msg:
                error_msg = "Job submission failed (check credentials and subscription)"
            raise CloudProviderError(f"Failed to submit HF Job: {error_msg}") from exc

        job_id = job.id if hasattr(job, "id") else str(job)
        return HFJobSubmission(
            job_id=job_id,
            job_url=getattr(job, "url", None),
            raw=job,
        )


_HF_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def sanitize_hf_job_labels(labels: Dict[str, str]) -> Dict[str, str]:
    """Drop label entries that do not conform to HF Jobs label validation rules.

    We intentionally omit slash-heavy values like bucket ids or artifact prefixes here.
    Downstream tooling can recover those from the submitted command args when needed.
    """
    sanitized: Dict[str, str] = {}
    for raw_key, raw_value in labels.items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if not key or not value:
            continue
        if not _HF_LABEL_PATTERN.fullmatch(key):
            continue
        if not _HF_LABEL_PATTERN.fullmatch(value):
            continue
        sanitized[key] = value
    return sanitized
