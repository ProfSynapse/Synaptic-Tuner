"""
shared/flywheel/orchestrator.py

FlywheelOrchestrator: coordinates the full flywheel pipeline and retrain
lifecycle. Stages: ingest -> clean -> tag -> stage -> (optional) retrain.

Retrain modes:
- GPU_MUTEX: Single GPU. Stops vLLM, trains, restarts with new adapter.
- HOT_SWAP: Train elsewhere, hot-swap adapter via vLLM API.
- CLOUD: Offload to cloud backend (HF Jobs, RunPod, Modal).

Used by: CLI (flywheel run-cycle), automated cron triggers
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .catalog import DatasetVersion, LogCatalog
from .cleaner import CleaningResult, DataCleaner
from .config import FlywheelConfig
from .readiness import ReadinessChecker, ReadinessReport
from .stager import DatasetStager, StagingResult
from .tagger import AutoTagger, TaggingResult

logger = logging.getLogger(__name__)


class RetrainMode(Enum):
    """How to handle GPU resources during retraining."""
    GPU_MUTEX = "gpu_mutex"
    HOT_SWAP = "hot_swap"
    CLOUD = "cloud"


@dataclass
class TrainingResult:
    """Outcome of a training run."""
    success: bool = False
    run_id: str = ""
    adapter_path: str = ""
    training_type: str = ""
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class CycleResult:
    """Summary of a full flywheel cycle."""
    cleaning: CleaningResult | None = None
    tagging: TaggingResult | None = None
    staging: StagingResult | None = None
    training: TrainingResult | None = None
    hot_swap_success: bool | None = None
    total_duration_seconds: float = 0.0


@dataclass
class FlywheelStatus:
    """Current state of the flywheel system."""
    vllm_running: bool = False
    vllm_model: str = ""
    active_adapter: str | None = None
    total_logs: int = 0
    unprocessed_logs: int = 0
    last_cycle_at: str | None = None
    last_dataset_version: str | None = None
    readiness: ReadinessReport | None = None


class FlywheelOrchestrator:
    """Coordinates the full flywheel pipeline and retrain lifecycle.

    Args:
        catalog: LogCatalog instance
        config: FlywheelConfig
        cleaner: DataCleaner instance
        tagger: AutoTagger instance
        stager: DatasetStager instance
    """

    def __init__(
        self,
        catalog: LogCatalog,
        config: FlywheelConfig,
        cleaner: DataCleaner,
        tagger: AutoTagger,
        stager: DatasetStager,
    ) -> None:
        self._catalog = catalog
        self._config = config
        self._cleaner = cleaner
        self._tagger = tagger
        self._stager = stager
        self._readiness = ReadinessChecker(catalog, config)

    async def run_cycle(
        self,
        skip_retrain: bool = False,
        retrain_mode: RetrainMode | None = None,
        dry_run: bool = False,
    ) -> CycleResult:
        """Execute a full flywheel cycle: clean -> tag -> stage -> retrain.

        Args:
            skip_retrain: If True, stop after staging (prepare data only)
            retrain_mode: Override config's default retrain mode
            dry_run: If True, show what would happen without executing

        Returns:
            CycleResult with per-stage summaries
        """
        start = time.monotonic()
        result = CycleResult()
        mode = retrain_mode or RetrainMode(self._config.retrain_mode)

        if dry_run:
            logger.info("DRY RUN: would execute clean -> tag -> stage")
            readiness = await self.check_readiness()
            logger.info("Readiness: %s", readiness.reasons)
            result.total_duration_seconds = time.monotonic() - start
            return result

        # Stage 1: Clean
        logger.info("Flywheel cycle: stage 1/4 - cleaning")
        try:
            result.cleaning = await self._cleaner.clean_logs()
        except Exception as exc:
            logger.error("Cleaning failed: %s", exc)

        # Short-circuit: if cleaning produced zero scored logs,
        # downstream stages (tag, stage) will produce zero results.
        if result.cleaning and result.cleaning.scored == 0:
            logger.warning(
                "Cleaning scored 0 logs (processed=%d, errors=%d); "
                "downstream stages will produce empty results",
                result.cleaning.total_processed, result.cleaning.errors,
            )

        # Stage 2: Tag
        logger.info("Flywheel cycle: stage 2/4 - tagging")
        try:
            result.tagging = await self._tagger.tag_logs()
        except Exception as exc:
            logger.error("Tagging failed: %s", exc)

        # Stage 3: Stage
        logger.info("Flywheel cycle: stage 3/4 - staging")
        try:
            result.staging = await self._stager.stage_dataset()
        except Exception as exc:
            logger.error("Staging failed: %s", exc)

        # Stage 4: Retrain (optional)
        if not skip_retrain and result.staging and result.staging.version_id:
            logger.info("Flywheel cycle: stage 4/4 - retraining (%s)", mode.value)
            try:
                # Get the dataset version for training
                version = await self._catalog.get_dataset_version(
                    result.staging.version_id,
                )
                if version:
                    result.training = await self._run_training(version, mode)

                    # Hot-swap adapter if training succeeded
                    if result.training and result.training.success:
                        if mode == RetrainMode.GPU_MUTEX:
                            result.hot_swap_success = await self._start_vllm(
                                result.training.adapter_path,
                            )
                        elif mode == RetrainMode.HOT_SWAP:
                            result.hot_swap_success = await self._hot_swap_adapter(
                                result.training.adapter_path,
                                self._config.vllm_adapter_name,
                            )
            except Exception as exc:
                logger.error("Retraining failed: %s", exc)
                result.training = TrainingResult(
                    success=False, error=str(exc),
                )
        else:
            logger.info("Flywheel cycle: skipping retrain")

        result.total_duration_seconds = time.monotonic() - start
        logger.info(
            "Flywheel cycle complete in %.1fs", result.total_duration_seconds,
        )
        return result

    async def check_readiness(self) -> ReadinessReport:
        """Check if enough data has accumulated to justify a retrain cycle."""
        return await self._readiness.check()

    async def _stop_vllm(self) -> bool:
        """Stop the vLLM server process. Returns True if stopped."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pkill", "-f", "vllm.entrypoints",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            # Wait for port to free up
            await asyncio.sleep(3)
            logger.info("vLLM server stopped")
            return True
        except Exception as exc:
            logger.error("Failed to stop vLLM: %s", exc)
            return False

    async def _start_vllm(self, adapter_path: str | None = None) -> bool:
        """Start vLLM with optional LoRA adapter."""
        cfg = self._config
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", cfg.vllm_base_model or "",
            "--port", str(cfg.vllm_port),
            "--gpu-memory-utilization", str(cfg.vllm_gpu_memory_utilization),
        ]

        if adapter_path and cfg.vllm_enable_runtime_lora:
            cmd.extend([
                "--enable-lora",
                "--lora-modules",
                f"{cfg.vllm_adapter_name}={adapter_path}",
                "--max-lora-rank", str(cfg.vllm_max_lora_rank),
            ])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            # Wait a bit for server startup
            await asyncio.sleep(10)
            logger.info("vLLM server started (PID %s)", proc.pid)
            return True
        except Exception as exc:
            logger.error("Failed to start vLLM: %s", exc)
            return False

    async def _hot_swap_adapter(
        self, adapter_path: str, adapter_name: str,
    ) -> bool:
        """Hot-swap LoRA adapter via vLLM API."""
        try:
            import httpx

            cfg = self._config
            url = f"http://{cfg.vllm_host}:{cfg.vllm_port}/v1/load_lora_adapter"
            payload = {
                "lora_name": adapter_name,
                "lora_path": adapter_path,
                "load_inplace": True,
            }

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()

            logger.info("LoRA adapter hot-swapped: %s", adapter_name)
            return True

        except Exception as exc:
            logger.error("Hot-swap failed: %s", exc)
            return False

    async def _run_training(
        self,
        dataset_version: DatasetVersion,
        retrain_mode: RetrainMode,
    ) -> TrainingResult:
        """Execute training using the staged dataset."""
        start = time.monotonic()

        if retrain_mode == RetrainMode.GPU_MUTEX:
            # Stop vLLM first to free GPU
            stopped = await self._stop_vllm()
            if not stopped:
                return TrainingResult(
                    success=False,
                    error="Failed to stop vLLM for GPU mutex",
                )

        script_path, args = self._select_trainer(dataset_version)

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", script_path, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            duration = time.monotonic() - start

            if proc.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")[:500]
                logger.error("Training failed: %s", error_msg)
                return TrainingResult(
                    success=False,
                    training_type=self._config.retrain_trainer,
                    duration_seconds=duration,
                    error=error_msg,
                )

            logger.info("Training completed in %.1fs", duration)
            return TrainingResult(
                success=True,
                training_type=self._config.retrain_trainer,
                duration_seconds=duration,
                adapter_path=self._config.vllm_adapter_path or "",
            )

        except Exception as exc:
            return TrainingResult(
                success=False,
                error=str(exc),
                duration_seconds=time.monotonic() - start,
            )

    def _select_trainer(
        self, dataset_version: DatasetVersion,
    ) -> tuple[str, list[str]]:
        """Select trainer script and args based on dataset composition."""
        counts = dataset_version.record_counts
        trainer = self._config.retrain_trainer

        if trainer == "auto":
            if counts.get("sft", 0) > 0:
                trainer = "sft"
            elif counts.get("kto_pos", 0) > 0 or counts.get("kto_neg", 0) > 0:
                trainer = "kto"
            else:
                trainer = "sft"

        file_paths = dataset_version.file_paths

        if trainer == "sft":
            dataset_file = file_paths.get("sft", "")
            return (
                "Trainers/sft/train_sft.py",
                ["--dataset-file", dataset_file],
            )
        elif trainer == "kto":
            dataset_file = file_paths.get("kto", "")
            return (
                "Trainers/kto/train_kto.py",
                ["--dataset-file", dataset_file],
            )
        else:
            # Default to SFT
            dataset_file = file_paths.get("sft", "")
            return (
                "Trainers/sft/train_sft.py",
                ["--dataset-file", dataset_file],
            )

    def status(self) -> FlywheelStatus:
        """Return current flywheel status (sync method for CLI use)."""
        import asyncio as _aio

        status = FlywheelStatus()

        # Check vLLM
        try:
            import httpx
            cfg = self._config
            resp = httpx.get(
                f"http://{cfg.vllm_host}:{cfg.vllm_port}/v1/models",
                timeout=5,
            )
            if resp.status_code == 200:
                status.vllm_running = True
                data = resp.json()
                models = data.get("data", [])
                if models:
                    status.vllm_model = models[0].get("id", "")
        except Exception:
            status.vllm_running = False

        return status
