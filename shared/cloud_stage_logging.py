"""
Canonical structured stage-event logging for cloud producers.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional
from shared.experiment_tracking.experiment import _atomic_write_text
from shared.experiment_tracking.registry import _PathLock

if TYPE_CHECKING:
    from tuner.project import ProjectContext

SCHEMA_VERSION = 1
STAGE_EVENTS_FILENAME = "stage_events.jsonl"
STAGE_SUMMARY_FILENAME = "stage_summary.json"

ENV_STAGE_NAME = "CLOUD_STAGE_NAME"
ENV_STAGE_PROVIDER = "CLOUD_STAGE_PROVIDER"
ENV_STAGE_RUN_PREFIX = "CLOUD_STAGE_RUN_PREFIX"
ENV_STAGE_JOB_REF = "CLOUD_STAGE_JOB_REF"
ENV_STAGE_BUCKET_ID = "CLOUD_STAGE_BUCKET_ID"
ENV_STAGE_LOG_DIR = "CLOUD_STAGE_LOG_DIR"
ENV_SOURCE_LOCK_ID = "SYNAPTIC_SOURCE_LOCK_ID"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_cloud_job_ref() -> Optional[str]:
    for key in (ENV_STAGE_JOB_REF, "HF_JOB_ID", "JOB_ID", "MODAL_TASK_ID", "RUNPOD_POD_ID"):
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    return None


def apply_stage_logging_env(
    *,
    stage: str,
    provider: Optional[str] = None,
    run_prefix: Optional[str] = None,
    job_ref: Optional[str] = None,
    bucket_id: Optional[str] = None,
    log_dir: Optional[Path | str] = None,
    source_lock_id: Optional[str] = None,
) -> None:
    os.environ[ENV_STAGE_NAME] = stage
    if provider:
        os.environ[ENV_STAGE_PROVIDER] = provider
    if run_prefix:
        os.environ[ENV_STAGE_RUN_PREFIX] = run_prefix
    if job_ref:
        os.environ[ENV_STAGE_JOB_REF] = job_ref
    if bucket_id:
        os.environ[ENV_STAGE_BUCKET_ID] = bucket_id
    if log_dir:
        os.environ[ENV_STAGE_LOG_DIR] = str(log_dir)
    if source_lock_id:
        os.environ[ENV_SOURCE_LOCK_ID] = source_lock_id


def normalize_failure(
    error: BaseException | str,
    *,
    traceback_text: Optional[str] = None,
) -> dict[str, Any]:
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        error_message = str(error) or repr(error)
    else:
        error_type = "RuntimeError"
        error_message = str(error)

    payload: dict[str, Any] = {
        "error_type": error_type,
        "error_message": error_message,
    }
    if traceback_text:
        payload["traceback"] = traceback_text
    return payload


def _resource_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {}
    if not torch.cuda.is_available():
        return {}
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return {}
    used_bytes = max(int(total_bytes) - int(free_bytes), 0)
    return {
        "gpu_mem_used_gb": round(used_bytes / (1024 ** 3), 3),
        "gpu_mem_free_gb": round(int(free_bytes) / (1024 ** 3), 3),
        "gpu_mem_total_gb": round(int(total_bytes) / (1024 ** 3), 3),
        "gpu_count": int(torch.cuda.device_count()),
    }


class CloudStageLogger:
    def __init__(
        self,
        log_dir: Path | str,
        *,
        stage: str,
        provider: Optional[str] = None,
        run_prefix: Optional[str] = None,
        job_ref: Optional[str] = None,
        bucket_id: Optional[str] = None,
        source_lock_id: Optional[str] = None,
        project_context: "ProjectContext | None" = None,
    ) -> None:
        resolved_log_dir = Path(log_dir)
        if project_context is not None and project_context.mode == "host":
            if not resolved_log_dir.is_absolute():
                resolved_log_dir = project_context.state_root / "logs" / resolved_log_dir
            resolved_log_dir = resolved_log_dir.resolve(strict=False)
            if not any(
                resolved_log_dir.is_relative_to(root.resolve(strict=False))
                for root in project_context.writable_roots
            ):
                raise ValueError("Cloud stage log directory must remain below a project writable root")
        self.log_dir = resolved_log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.provider = provider
        self.run_prefix = run_prefix
        self.job_ref = job_ref
        self.bucket_id = bucket_id
        self.source_lock_id = source_lock_id
        self._lock = threading.RLock()
        self.events_path = self.log_dir / STAGE_EVENTS_FILENAME
        self.summary_path = self.log_dir / STAGE_SUMMARY_FILENAME

    @classmethod
    def from_env(
        cls,
        log_dir: Path | str,
        *,
        stage: Optional[str] = None,
        provider: Optional[str] = None,
        run_prefix: Optional[str] = None,
        job_ref: Optional[str] = None,
    ) -> "CloudStageLogger":
        resolved_stage = stage or str(os.environ.get(ENV_STAGE_NAME, "")).strip() or "evaluation"
        resolved_provider = provider or str(os.environ.get(ENV_STAGE_PROVIDER, "")).strip() or None
        resolved_run_prefix = run_prefix or str(os.environ.get(ENV_STAGE_RUN_PREFIX, "")).strip() or None
        resolved_job_ref = job_ref or detect_cloud_job_ref()
        return cls(
            log_dir,
            stage=resolved_stage,
            provider=resolved_provider,
            run_prefix=resolved_run_prefix,
            job_ref=resolved_job_ref,
            bucket_id=str(os.environ.get(ENV_STAGE_BUCKET_ID, "")).strip() or None,
            source_lock_id=str(os.environ.get(ENV_SOURCE_LOCK_ID, "")).strip() or None,
        )

    def emit(
        self,
        event: str,
        *,
        status: str = "running",
        message: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": _utcnow_iso(),
            "stage": self.stage,
            "provider": self.provider,
            "job_ref": self.job_ref,
            "run_prefix": self.run_prefix,
            "bucket_id": self.bucket_id,
            "source_lock_id": self.source_lock_id,
            "event": event,
            "status": status,
            "message": message,
            "details": dict(details or {}),
        }
        with self._lock:
            line = (
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            self._append_event_transaction(line, payload)
        return payload

    def _append_event_transaction(
        self, line: bytes, summary_payload: Mapping[str, Any]
    ) -> None:
        """Append one complete event and its summary as a rollback-safe unit."""

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with _PathLock(self.events_path):
            fd = os.open(
                self.events_path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0),
                0o600,
            )
            try:
                boundary = self._last_complete_boundary(fd)
                end = os.lseek(fd, 0, os.SEEK_END)
                if end != boundary:
                    os.ftruncate(fd, boundary)
                    self._fsync_event_file(fd)
                os.lseek(fd, boundary, os.SEEK_SET)
                try:
                    remaining = memoryview(line)
                    while remaining:
                        written = self._write_event_bytes(fd, remaining)
                        if written <= 0:
                            raise OSError("Zero-byte write while appending cloud stage event")
                        remaining = remaining[written:]
                    self._fsync_event_file(fd)
                    self._write_summary(summary_payload)
                except BaseException:
                    try:
                        os.ftruncate(fd, boundary)
                        self._fsync_event_file(fd)
                    except BaseException as rollback_error:
                        raise RuntimeError(
                            "Cloud stage event rollback could not be persisted"
                        ) from rollback_error
                    raise
            finally:
                os.close(fd)

    @staticmethod
    def _write_event_bytes(fd: int, data: memoryview) -> int:
        return os.write(fd, data)

    @staticmethod
    def _fsync_event_file(fd: int) -> None:
        os.fsync(fd)

    @staticmethod
    def _last_complete_boundary(fd: int) -> int:
        end = os.lseek(fd, 0, os.SEEK_END)
        if end == 0:
            return 0
        position = end
        while position > 0:
            size = min(position, 64 * 1024)
            position -= size
            os.lseek(fd, position, os.SEEK_SET)
            chunk = os.read(fd, size)
            index = chunk.rfind(b"\n")
            if index >= 0:
                return position + index + 1
        return 0

    def emit_failure(
        self,
        error: BaseException | str,
        *,
        message: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
        traceback_text: Optional[str] = None,
    ) -> dict[str, Any]:
        failure = normalize_failure(error, traceback_text=traceback_text)
        merged_details = dict(details or {})
        merged_details.update(failure)
        return self.emit(
            "failed",
            status="failed",
            message=message or failure["error_message"],
            details=merged_details,
        )

    def emit_resource_snapshot(
        self,
        *,
        message: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        merged_details = dict(details or {})
        merged_details.update(_resource_snapshot())
        return self.emit("resource_snapshot", message=message, details=merged_details)

    def _default_summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": self.stage,
            "provider": self.provider,
            "job_ref": self.job_ref,
            "run_prefix": self.run_prefix,
            "bucket_id": self.bucket_id,
            "source_lock_id": self.source_lock_id,
            "event": None,
            "status": "pending",
            "health": "unknown",
            "message": None,
            "details": {},
            "event_count": 0,
            "updated_at": None,
            "last_progress_at": None,
        }

    def _load_summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return self._default_summary()
        try:
            payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_summary()
        summary = self._default_summary()
        summary.update(payload)
        if not isinstance(summary.get("details"), dict):
            summary["details"] = {}
        return summary

    @staticmethod
    def _health_for_status(status: str) -> str:
        if status == "failed":
            return "failed"
        if status in {"completed", "running"}:
            return "healthy"
        return "unknown"

    def _write_summary(self, payload: Mapping[str, Any]) -> None:
        summary = self._load_summary()
        details = dict(summary.get("details", {}))
        details.update(dict(payload.get("details") or {}))

        summary.update(
            {
                "schema_version": SCHEMA_VERSION,
                "stage": payload.get("stage"),
                "provider": payload.get("provider"),
                "job_ref": payload.get("job_ref"),
                "run_prefix": payload.get("run_prefix"),
                "bucket_id": payload.get("bucket_id"),
                "source_lock_id": payload.get("source_lock_id"),
                "event": payload.get("event"),
                "status": payload.get("status"),
                "health": self._health_for_status(str(payload.get("status", ""))),
                "message": payload.get("message"),
                "details": details,
                "event_count": int(summary.get("event_count", 0)) + 1,
                "updated_at": payload.get("timestamp"),
            }
        )
        if payload.get("event") == "progress":
            summary["last_progress_at"] = payload.get("timestamp")
        _atomic_write_text(self.summary_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    def emit_sync(self, *, path: str) -> dict[str, Any]:
        return self.emit("artifacts_synced", details={"last_sync_path": path})


StageLogger = CloudStageLogger


def stage_logger_from_env(log_dir: Optional[Path | str] = None) -> Optional[CloudStageLogger]:
    resolved_log_dir = log_dir or str(os.environ.get(ENV_STAGE_LOG_DIR, "")).strip()
    if not resolved_log_dir:
        return None
    return CloudStageLogger.from_env(resolved_log_dir)
