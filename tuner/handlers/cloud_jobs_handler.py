"""Inspect and manage live Hugging Face Jobs."""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from shared.utilities.env import get_hf_token, load_env_file
from tuner.cloud import load_huggingface_hub
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler
from tuner.ui import BOX, confirm, print_config, print_error, print_header, print_info, print_success, print_table


class CloudJobsHandler(BaseHandler):
    """List, inspect, tail logs for, and cancel HF Jobs."""

    @property
    def name(self) -> str:
        return "cloud-jobs"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _validate_environment(self):
        load_env_file()
        token = get_hf_token()
        if not token:
            raise CloudProviderError(
                "HF_TOKEN not set. Required for cloud-jobs. Set HF_TOKEN (or HF_API_KEY) in your .env file or environment."
            )
        hub = load_huggingface_hub(require_apis=("list_jobs", "inspect_job", "fetch_job_logs", "cancel_job"))
        return hub, token

    def _resolve_subcommand(self) -> str:
        subcommand = (getattr(self.args, "subcommand", None) or "list").strip().lower()
        if subcommand not in {"list", "show", "logs", "cancel"}:
            raise CloudProviderError(
                f"Unsupported cloud-jobs subcommand '{subcommand}'. Available: list, show, logs, cancel"
            )
        return subcommand

    def _split_job_ref(self, job_ref: str, namespace: Optional[str]) -> Tuple[str, Optional[str]]:
        raw = (job_ref or "").strip()
        if not raw:
            raise CloudProviderError("cloud-jobs requires --job for show/logs/cancel.")
        if "/" in raw:
            ref_namespace, ref_job_id = raw.split("/", 1)
            return ref_job_id.strip(), ref_namespace.strip() or namespace
        return raw, namespace

    def _normalize_job(self, job: Any) -> Dict[str, Any]:
        status = getattr(job, "status", None)
        owner = getattr(job, "owner", None)
        created_at = getattr(job, "created_at", None)
        command = getattr(job, "command", None) or []
        return {
            "id": getattr(job, "id", ""),
            "stage": getattr(status, "stage", None),
            "message": getattr(status, "message", None),
            "created_at": created_at.isoformat() if isinstance(created_at, datetime) else (str(created_at) if created_at else None),
            "owner": getattr(owner, "name", None),
            "image": getattr(job, "docker_image", None) or getattr(job, "space_id", None),
            "flavor": getattr(job, "flavor", None),
            "url": getattr(job, "url", None),
            "labels": getattr(job, "labels", None) or {},
            "command": command,
        }

    def _list_jobs(self, hub, token: str, namespace: Optional[str], limit: int) -> List[Dict[str, Any]]:
        jobs = hub.list_jobs(namespace=namespace, token=token)
        normalized = [self._normalize_job(job) for job in jobs]
        normalized.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return normalized[: max(limit, 1)]

    def _show_job(self, hub, token: str, job_id: str, namespace: Optional[str]) -> Dict[str, Any]:
        job = hub.inspect_job(job_id=job_id, namespace=namespace, token=token)
        return self._normalize_job(job)

    def _fetch_logs(
        self,
        hub,
        token: str,
        job_id: str,
        namespace: Optional[str],
        *,
        tail: int,
        follow: bool,
    ) -> List[str]:
        logs = hub.fetch_job_logs(job_id=job_id, namespace=namespace, follow=follow, token=token)
        if isinstance(logs, str):
            lines = logs.splitlines()
        else:
            lines = [str(line).rstrip("\n") for line in logs]
        if follow:
            return lines
        return lines[-max(tail, 1):]

    def handle(self) -> int:
        try:
            hub, token = self._validate_environment()
            subcommand = self._resolve_subcommand()
            namespace = getattr(self.args, "namespace", None)
            tail = int(getattr(self.args, "tail", 200) or 200)
            limit = int(getattr(self.args, "limit", 20) or 20)
            follow = bool(getattr(self.args, "follow", False))
        except Exception as exc:
            if self.json_mode:
                self.output_error(str(exc), code="CLOUD_JOBS_ENV_ERROR")
                return 1
            print_error(str(exc))
            return 1

        try:
            if subcommand == "list":
                jobs = self._list_jobs(hub, token, namespace, limit)
                if self.json_mode:
                    self.output({"jobs": jobs, "namespace": namespace})
                    return 0

                print_header("CLOUD JOBS", "Recent Hugging Face Jobs")
                if not jobs:
                    print_info("No HF Jobs found.")
                    return 0
                rows = []
                for job in jobs:
                    rows.append(
                        [
                            job["id"],
                            job.get("stage") or "-",
                            job.get("owner") or "-",
                            job.get("flavor") or "-",
                            job.get("created_at") or "-",
                            job.get("image") or "-",
                        ]
                    )
                print_table(
                    rows,
                    headers=["Job ID", "Stage", "Owner", "Flavor", "Created", "Image"],
                    title="Hugging Face Jobs",
                )
                return 0

            job_id, resolved_namespace = self._split_job_ref(getattr(self.args, "job", None), namespace)

            if subcommand == "show":
                job = self._show_job(hub, token, job_id, resolved_namespace)
                if self.json_mode:
                    self.output({"job": job, "namespace": resolved_namespace})
                    return 0
                print_header("CLOUD JOB", "Hugging Face Job Details")
                print_config(
                    {
                        "Job": job["id"],
                        "Namespace": resolved_namespace or job.get("owner") or "-",
                        "Stage": job.get("stage") or "-",
                        "Message": job.get("message") or "-",
                        "Created": job.get("created_at") or "-",
                        "Flavor": job.get("flavor") or "-",
                        "Image": job.get("image") or "-",
                        "URL": job.get("url") or "-",
                        "Labels": ", ".join(f"{k}={v}" for k, v in sorted((job.get("labels") or {}).items())) or "-",
                    },
                    "HF Job Summary",
                )
                if job.get("command"):
                    print_info("Command:")
                    print(job["command"][-1] if len(job["command"]) >= 3 else " ".join(job["command"]))
                return 0

            if subcommand == "logs":
                if self.json_mode and follow:
                    raise CloudProviderError("--follow is not supported with --json for cloud-jobs logs.")
                lines = self._fetch_logs(
                    hub,
                    token,
                    job_id,
                    resolved_namespace,
                    tail=tail,
                    follow=follow,
                )
                if self.json_mode:
                    self.output({"job": job_id, "namespace": resolved_namespace, "logs": lines})
                    return 0
                print_header("CLOUD JOB LOGS", f"{resolved_namespace + '/' if resolved_namespace else ''}{job_id}")
                if not lines:
                    print_info("No logs returned.")
                    return 0
                for line in lines:
                    print(line)
                return 0

            if subcommand == "cancel":
                if not getattr(self.args, "auto_confirm", False) and not confirm(
                    f"Cancel HF job {resolved_namespace + '/' if resolved_namespace else ''}{job_id}?"
                ):
                    print_info("Cloud job cancellation cancelled.")
                    return 0
                hub.cancel_job(job_id=job_id, namespace=resolved_namespace, token=token)
                if self.json_mode:
                    self.output({"job": job_id, "namespace": resolved_namespace, "cancelled": True})
                else:
                    print_success(f"Cancelled HF job {resolved_namespace + '/' if resolved_namespace else ''}{job_id}")
                return 0

        except Exception as exc:
            if self.json_mode:
                self.output_error(str(exc), code="CLOUD_JOBS_ERROR")
                return 1
            print_error(str(exc))
            return 1

        return 0
