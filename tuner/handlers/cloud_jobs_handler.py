"""Inspect and manage live Hugging Face Jobs."""

from __future__ import annotations

import json
import shlex
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from shared.utilities.env import get_hf_token, load_env_file
from tuner.cloud import decode_hf_job_label, load_huggingface_hub
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

    def _extract_command_text(self, job: Dict[str, Any]) -> str:
        command = job.get("command") or []
        if not command:
            return ""
        if len(command) >= 3 and str(command[1]).strip() in {"-c", "-lc"}:
            return str(command[2])
        return " ".join(str(part) for part in command)

    def _find_cli_arg(self, tokens: List[str], *names: str) -> Optional[str]:
        for index, token in enumerate(tokens):
            for name in names:
                if token == name and index + 1 < len(tokens):
                    return tokens[index + 1]
                if token.startswith(f"{name}="):
                    return token.split("=", 1)[1]
        return None

    def _resolve_stage_artifact_root(self, job: Dict[str, Any]) -> Optional[Dict[str, str]]:
        labels = job.get("labels") or {}
        bucket_id = (
            labels.get("artifact_bucket")
            or labels.get("bucket_id")
            or labels.get("bucket")
        )
        prefix = (
            labels.get("eval_prefix")
            or labels.get("results_prefix")
            or labels.get("artifact_prefix")
        )
        if bucket_id:
            bucket_id = decode_hf_job_label(str(bucket_id))
        if prefix:
            prefix = decode_hf_job_label(str(prefix))
        if bucket_id and prefix:
            return {"bucket_id": bucket_id.strip("/"), "prefix": prefix.strip("/")}

        command_text = self._extract_command_text(job)
        if not command_text:
            return None
        try:
            tokens = shlex.split(command_text)
        except ValueError:
            tokens = command_text.split()

        bucket_id = bucket_id or self._find_cli_arg(tokens, "--bucket-id", "--artifact-bucket")
        prefix = prefix or self._find_cli_arg(tokens, "--eval-prefix", "--results-prefix", "--artifact-prefix")
        if not prefix:
            prefix = self._find_cli_arg(tokens, "--run-prefix")
        if not bucket_id or not prefix:
            return None
        return {"bucket_id": str(bucket_id).strip("/"), "prefix": str(prefix).strip("/")}

    def _stage_summary_paths(self, artifact_root: Dict[str, str]) -> Dict[str, str]:
        bucket_id = artifact_root["bucket_id"].strip("/")
        prefix = artifact_root["prefix"].strip("/")
        base_uri = f"hf://buckets/{bucket_id}/{prefix}"
        return {
            "bucket_id": bucket_id,
            "prefix": prefix,
            "artifact_uri": base_uri,
            "summary_uri": f"{base_uri}/logs/stage_summary.json",
            "events_uri": f"{base_uri}/logs/stage_events.jsonl",
        }

    def _load_stage_summary(self, hub, token: str, job: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
        artifact_root = self._resolve_stage_artifact_root(job)
        if artifact_root is None or not hasattr(hub, "HfFileSystem"):
            return None, None

        artifact_paths = self._stage_summary_paths(artifact_root)
        try:
            fs = hub.HfFileSystem(token=token)
            with fs.open(artifact_paths["summary_uri"], "r") as handle:
                payload = json.load(handle)
        except Exception:
            return None, artifact_paths

        if not isinstance(payload, dict):
            return None, artifact_paths
        return payload, artifact_paths

    def _stringify_stage_summary_value(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return str(value)

    def _build_stage_summary_display(
        self,
        summary: Dict[str, Any],
        artifact_paths: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        display: Dict[str, str] = {}
        preferred_fields = [
            ("Stage", summary.get("stage")),
            ("Health", summary.get("health")),
            ("Status", summary.get("status")),
            ("Message", summary.get("message")),
            ("Last Event", summary.get("last_event") or summary.get("event")),
            ("Updated", summary.get("updated_at") or summary.get("timestamp")),
        ]
        for label, value in preferred_fields:
            if value not in (None, "", []):
                display[label] = self._stringify_stage_summary_value(value)

        if artifact_paths:
            display["Artifacts"] = artifact_paths["artifact_uri"]
            display["Stage Summary"] = artifact_paths["summary_uri"]
            display["Stage Events"] = artifact_paths["events_uri"]

        consumed_keys = {"stage", "health", "status", "message", "last_event", "event", "updated_at", "timestamp"}
        for key in sorted(summary):
            if key in consumed_keys:
                continue
            label = key.replace("_", " ").title()
            display[label] = self._stringify_stage_summary_value(summary.get(key))
        return display

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
                stage_summary, artifact_paths = self._load_stage_summary(hub, token, job)
                if self.json_mode:
                    payload: Dict[str, Any] = {
                        "job": job,
                        "namespace": resolved_namespace,
                        "stage_summary": stage_summary,
                    }
                    if artifact_paths:
                        payload["stage_artifacts"] = artifact_paths
                    if stage_summary is None:
                        payload["log_tail"] = self._fetch_logs(
                            hub,
                            token,
                            job_id,
                            resolved_namespace,
                            tail=min(max(tail, 1), 40),
                            follow=False,
                        )
                    self.output(payload)
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
                if stage_summary is not None:
                    print_config(
                        self._build_stage_summary_display(stage_summary, artifact_paths),
                        "Structured Stage Health",
                    )
                else:
                    print_info("Structured stage summary not available; falling back to recent raw logs.")
                    lines = self._fetch_logs(
                        hub,
                        token,
                        job_id,
                        resolved_namespace,
                        tail=min(max(tail, 1), 40),
                        follow=False,
                    )
                    if not lines:
                        print_info("No logs returned.")
                    else:
                        print_info("Recent raw logs:")
                        for line in lines:
                            print(line)
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
