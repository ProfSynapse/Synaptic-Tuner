"""Provider-agnostic evaluation backend.

Supports local inference and cloud providers (HF Jobs, Modal, RunPod).
Cloud providers are injected via the CloudProvider protocol.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result from a single evaluation run."""

    eval_score: float
    results_path: str | None = None
    raw_results: dict | None = None


@runtime_checkable
class EvalBackend(Protocol):
    """Provider-agnostic eval interface."""

    async def run_eval(self, adapter_path: str, scenario: str) -> EvalResult: ...


class LocalEvalBackend:
    """Run evaluation on local GPU via Evaluator CLI.

    Hardware-gated: checks GPU VRAM before running.
    """

    def __init__(self, min_vram_gb: int = 8):
        self.min_vram_gb = min_vram_gb

    def check_hardware(self) -> bool:
        """Check if local GPU has sufficient VRAM."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            vram_mb = int(result.stdout.strip().split("\n")[0])
            vram_gb = vram_mb / 1024
            return vram_gb >= self.min_vram_gb
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            return False

    async def run_eval(self, adapter_path: str, scenario: str) -> EvalResult:
        """Run Evaluator CLI on local GPU."""
        if not self.check_hardware():
            raise RuntimeError(
                f"Insufficient GPU VRAM (need {self.min_vram_gb}GB). "
                f"Use eval_backend='cloud' with a cloud provider instead."
            )

        cmd = [
            "python",
            "-m",
            "Evaluator.cli",
            "--model",
            adapter_path,
            "--prompt-set",
            scenario,
        ]
        logger.info(f"Running local eval: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error(f"Eval failed: {result.stderr[:500]}")
            raise RuntimeError(f"Evaluation failed: {result.stderr[:200]}")

        # Parse eval results — look for JSON output or results file
        try:
            # Try parsing stdout as JSON
            raw = json.loads(result.stdout)
            score = raw.get("results_summary", {}).get("overall_pass_rate", 0.0)
            return EvalResult(eval_score=score, raw_results=raw)
        except json.JSONDecodeError:
            # Fall back to searching for results file
            logger.warning("Could not parse eval output as JSON, returning 0.0")
            return EvalResult(eval_score=0.0)


@runtime_checkable
class CloudProvider(Protocol):
    """Adapter interface for cloud compute providers.

    Implement this for HF Jobs, Modal, RunPod, etc.
    """

    async def submit_eval_job(
        self, adapter_path: str, scenario: str
    ) -> str: ...

    async def wait_for_result(self, job_id: str) -> EvalResult: ...

    async def upload_adapter(self, local_path: str) -> str: ...


class CloudEvalBackend:
    """Delegates evaluation to a cloud provider.

    Provider is injected, not hardcoded. Supports any CloudProvider implementation.
    Enforces a timeout to prevent stuck jobs from blocking indefinitely.
    """

    def __init__(self, provider: CloudProvider, timeout_seconds: int = 3600):
        self.provider = provider
        self.timeout_seconds = timeout_seconds

    async def run_eval(self, adapter_path: str, scenario: str) -> EvalResult:
        """Upload adapter and run eval on cloud with timeout."""
        import asyncio

        remote_path = await self.provider.upload_adapter(adapter_path)
        job_id = await self.provider.submit_eval_job(remote_path, scenario)
        logger.info(f"Submitted cloud eval job: {job_id}")
        try:
            return await asyncio.wait_for(
                self.provider.wait_for_result(job_id),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Cloud eval timed out after {self.timeout_seconds}s for job {job_id}. "
                f"Increase timeout_seconds or check cloud provider status."
            )


def create_eval_backend(
    backend_type: str = "local",
    cloud_provider: CloudProvider | None = None,
    min_vram_gb: int = 8,
) -> EvalBackend:
    """Factory for creating eval backends."""
    if backend_type == "local":
        return LocalEvalBackend(min_vram_gb=min_vram_gb)
    elif backend_type == "cloud":
        if cloud_provider is None:
            raise ValueError("cloud_provider required when eval_backend='cloud'")
        return CloudEvalBackend(provider=cloud_provider)
    else:
        raise ValueError(f"Unknown eval backend: {backend_type}")
