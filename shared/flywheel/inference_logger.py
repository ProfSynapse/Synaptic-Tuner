"""
shared/flywheel/inference_logger.py

Async JSONL writer for inference request/response pairs. Captures
OpenAI-format chat completions, writes them to date-partitioned JSONL files,
and indexes them in the LogCatalog for downstream pipeline queries.

Thread-safe via asyncio.Queue. Non-blocking: callers enqueue log entries
and return immediately. A background task drains the queue and writes.

Used by: services/proxy/app.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import InferenceLogRecord, LogCatalog
from .config import FlywheelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credential / PII scrubber
# ---------------------------------------------------------------------------

_SCRUB_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"Bearer [a-zA-Z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
    (re.compile(r'"password"\s*:\s*"[^"]{4,}"'), '"password": "[REDACTED]"'),
]


def _scrub_credentials(text: str) -> str:
    """Replace API keys, bearer tokens, and passwords in *text*."""
    for pattern, replacement in _SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _scrub_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep-copy of *messages* with credentials scrubbed from content."""
    scrubbed: list[dict[str, Any]] = []
    for msg in messages:
        msg_copy = dict(msg)
        if isinstance(msg_copy.get("content"), str):
            msg_copy["content"] = _scrub_credentials(msg_copy["content"])
        scrubbed.append(msg_copy)
    return scrubbed


def _extract_completion_token_logprobs(
    choice: dict[str, Any],
) -> tuple[list[int] | None, list[float] | None]:
    """Best-effort extraction of completion token ids + per-token logprobs.

    Reads the OpenAI/vLLM chat-completions ``logprobs.content`` array. Per-token
    logprobs are taken directly. Token ids are only available when vLLM was asked
    with ``return_tokens_as_token_ids: true`` (then each ``token`` is rendered as
    ``"token_id:<int>"``); if any token is not in that form, token ids are
    reported as None (logprobs may still be returned). Never raises — capture is
    best-effort and absence is normal.
    """
    logprobs_obj = choice.get("logprobs") or {}
    content = logprobs_obj.get("content")
    if not content:
        return None, None

    logprob_values: list[float] = []
    token_ids: list[int] = []
    token_ids_ok = True
    for entry in content:
        if not isinstance(entry, dict):
            return None, None
        lp = entry.get("logprob")
        logprob_values.append(float(lp) if lp is not None else 0.0)
        token = entry.get("token", "")
        if isinstance(token, str) and token.startswith("token_id:"):
            try:
                token_ids.append(int(token.split(":", 1)[1]))
            except ValueError:
                token_ids_ok = False
        else:
            token_ids_ok = False

    return (token_ids if token_ids_ok and token_ids else None), (logprob_values or None)


class InferenceLogger:
    """Async JSONL writer for inference request/response pairs.

    Args:
        log_dir: Directory for JSONL files (e.g., "inference_logs/")
        catalog: LogCatalog instance for indexing
        config: FlywheelConfig for rotation settings
        enabled: When False, all log calls are no-ops (zero overhead)
    """

    def __init__(
        self,
        log_dir: Path,
        catalog: LogCatalog,
        config: FlywheelConfig,
        enabled: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._catalog = catalog
        self._config = config
        self._enabled = enabled
        self._queue: asyncio.Queue[InferenceLogRecord] = asyncio.Queue()
        self._writer_task: asyncio.Task | None = None
        self._running = False
        self._total_logged = 0
        self._today_count = 0

    async def start(self) -> None:
        """Start the background writer task. Call once at proxy startup."""
        if not self._enabled:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop())
        logger.info("InferenceLogger started, writing to %s", self._log_dir)

    async def stop(self) -> None:
        """Flush pending writes and stop the background task."""
        if not self._enabled or not self._running:
            return
        self._running = False
        # Drain remaining items
        await self._queue.join()
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
        logger.info(
            "InferenceLogger stopped. Total logged: %d", self._total_logged,
        )

    async def log_inference(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        latency_ms: float,
        model_id: str,
        adapter_name: str | None = None,
    ) -> str:
        """Enqueue an inference log entry. Returns the log_id immediately.

        Args:
            request: OpenAI chat completion request body
            response: OpenAI chat completion response body
            latency_ms: End-to-end latency in milliseconds
            model_id: Model identifier from the request
            adapter_name: LoRA adapter name if applicable

        Returns:
            The generated log_id (UUID4 string)
        """
        if not self._enabled:
            return ""

        record = self._build_record(
            request, response, latency_ms, model_id, adapter_name,
        )
        await self._queue.put(record)
        return record.log_id

    @property
    def queue_depth(self) -> int:
        """Current number of items waiting to be written."""
        return self._queue.qsize()

    @property
    def total_logged(self) -> int:
        """Total inference logs written since start."""
        return self._total_logged

    def _build_record(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        latency_ms: float,
        model_id: str,
        adapter_name: str | None,
    ) -> InferenceLogRecord:
        """Extract fields from OpenAI-format request/response."""
        messages = request.get("messages", [])
        tools = request.get("tools", [])
        tools_requested = bool(tools)

        # Extract response content and tool calls
        choices = response.get("choices", [])
        response_content = ""
        tool_calls_list: list[dict] = []
        finish_reason = "stop"

        completion_token_ids: list[int] | None = None
        completion_logprobs: list[float] | None = None
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            response_content = message.get("content", "") or ""
            tool_calls_list = message.get("tool_calls", []) or []
            finish_reason = choice.get("finish_reason", "stop") or "stop"
            if getattr(self._config, "capture_token_ids", False):
                completion_token_ids, completion_logprobs = _extract_completion_token_logprobs(choice)

        # Extract usage
        usage = response.get("usage", {})
        prompt_toks = usage.get("prompt_tokens", 0)
        completion_toks = usage.get("completion_tokens", 0)

        return InferenceLogRecord(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=model_id,
            adapter_name=adapter_name,
            messages=_scrub_messages(messages),
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens", 1024),
            tools=tools,
            tools_requested=tools_requested,
            response_content=_scrub_credentials(response_content),
            tool_calls=tool_calls_list,
            finish_reason=finish_reason,
            prompt_tokens=prompt_toks,
            completion_tokens=completion_toks,
            completion_token_ids=completion_token_ids,
            completion_logprobs=completion_logprobs,
            latency_ms=latency_ms,
        )

    def _get_log_file(self) -> Path:
        """Return today's JSONL file path (date-partitioned)."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._log_dir / f"{date_str}.jsonl"

    async def _writer_loop(self) -> None:
        """Background loop: drain queue, batch-write to JSONL, index in catalog."""
        batch: list[InferenceLogRecord] = []
        flush_interval = self._config.flush_interval_seconds

        while self._running or not self._queue.empty():
            try:
                # Collect items for up to flush_interval seconds
                try:
                    record = await asyncio.wait_for(
                        self._queue.get(), timeout=flush_interval,
                    )
                    batch.append(record)
                    self._queue.task_done()

                    # Drain any additional ready items
                    while not self._queue.empty():
                        record = self._queue.get_nowait()
                        batch.append(record)
                        self._queue.task_done()

                except asyncio.TimeoutError:
                    pass

                if not batch:
                    continue

                # Write batch to JSONL
                log_file = self._get_log_file()
                await self._write_batch(log_file, batch)

                # Index in catalog
                try:
                    await self._catalog.insert_logs_batch(batch)
                except Exception as exc:
                    logger.error(
                        "Failed to index %d logs in catalog: %s",
                        len(batch), exc,
                    )

                self._total_logged += len(batch)
                self._today_count += len(batch)
                batch = []

            except asyncio.CancelledError:
                # Final flush on cancellation
                if batch:
                    log_file = self._get_log_file()
                    await self._write_batch(log_file, batch)
                    try:
                        await self._catalog.insert_logs_batch(batch)
                    except Exception:
                        pass
                    self._total_logged += len(batch)
                raise
            except Exception as exc:
                logger.error("Writer loop error: %s", exc)
                batch = []

    async def _write_batch(
        self, log_file: Path, batch: list[InferenceLogRecord],
    ) -> None:
        """Write a batch of records to a JSONL file.

        File I/O is offloaded to a thread via asyncio.to_thread() to avoid
        blocking the event loop. This is important as JSONL files grow over
        a day's worth of inference logging.
        """
        try:
            await asyncio.to_thread(self._write_batch_sync, log_file, batch)
        except Exception as exc:
            logger.error("Failed to write %d logs to %s: %s", len(batch), log_file, exc)

    def _write_batch_sync(
        self, log_file: Path, batch: list[InferenceLogRecord],
    ) -> None:
        """Synchronous file write, called from a thread pool."""
        start_line = 0
        if log_file.exists():
            start_line = sum(1 for _ in open(log_file, "r", encoding="utf-8"))

        with open(log_file, "a", encoding="utf-8") as f:
            for i, record in enumerate(batch):
                record.source_file = str(log_file)
                record.line_number = start_line + i
                f.write(record.to_json() + "\n")
