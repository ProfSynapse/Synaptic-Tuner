"""
Flywheel CLI handler for the Synaptic Tuner.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/flywheel_handler.py
Purpose: Route and execute flywheel subcommands (status, run-cycle, configure, etc.)
Used by: Router when 'flywheel' command is invoked; MainMenuHandler for interactive menu

Integrates with shared.flywheel for pipeline operations and configuration.
"""

import asyncio
import logging
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from tuner.handlers.base import BaseHandler

logger = logging.getLogger(__name__)


class FlywheelHandler(BaseHandler):
    """Handler for ``tuner flywheel`` CLI subcommands."""

    _SUBCOMMANDS = {
        "status": "_handle_status",
        "run-cycle": "_handle_run_cycle",
        "configure": "_handle_configure",
        "readiness": "_handle_readiness",
        "stage": "_handle_stage",
        "logs": "_handle_logs",
        "versions": "_handle_versions",
    }

    def __init__(self, args: Optional[Namespace] = None):
        super().__init__(args=args)

    @property
    def name(self) -> str:
        return "flywheel"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _load_config(self) -> Any:
        """Load FlywheelConfig, respecting --flywheel-config override."""
        from shared.flywheel.config import load_flywheel_config
        config_path = getattr(self.args, "flywheel_config", None) if self.args else None
        return load_flywheel_config(config_path)

    async def _build_orchestrator_async(self, config: Any) -> Any:
        """Construct a FlywheelOrchestrator with all pipeline components.

        Async because create_catalog() requires await for DB initialization.
        """
        from shared.flywheel.catalog import create_catalog
        from shared.flywheel.cleaner import DataCleaner
        from shared.flywheel.orchestrator import FlywheelOrchestrator
        from shared.flywheel.stager import DatasetStager
        from shared.flywheel.tagger import AutoTagger

        catalog = await create_catalog(
            backend=config.catalog_backend,
            path=config.catalog_path,
            url=config.catalog_url,
            tenant_id=config.tenant_id,
        )
        return FlywheelOrchestrator(
            catalog=catalog, config=config,
            cleaner=DataCleaner(catalog=catalog, config=config),
            tagger=AutoTagger(catalog=catalog, config=config),
            stager=DatasetStager(catalog=catalog, config=config),
        )

    def _get_orchestrator(self) -> tuple:
        """Load config and build orchestrator. Returns (config, orchestrator)."""
        config = self._load_config()
        return config, asyncio.run(self._build_orchestrator_async(config))

    def handle(self) -> int:
        """Route to the appropriate flywheel subcommand handler."""
        action = getattr(self.args, "subcommand", None) if self.args else None
        if not action:
            if self.json_mode:
                self.output({"command": "flywheel", "subcommands": list(self._SUBCOMMANDS)})
                return 0
            return self._show_interactive_menu()

        method_name = self._SUBCOMMANDS.get(action)
        if not method_name:
            self.output_error(f"Unknown flywheel subcommand: {action}", code="UNKNOWN_SUBCOMMAND")
            return 1
        return getattr(self, method_name)()

    def _show_interactive_menu(self) -> int:
        """Present an interactive menu when no subcommand is given."""
        try:
            from tuner.ui import print_header, print_menu
        except ImportError:
            print("Available subcommands: " + ", ".join(self._SUBCOMMANDS))
            return 0

        print_header("DATA FLYWHEEL", "Self-improving training pipeline")
        menu_items = [
            ("1", "Status     - Show flywheel system status"),
            ("2", "Run Cycle  - Execute flywheel pipeline"),
            ("3", "Readiness  - Check retrain readiness"),
            ("4", "Stage      - Stage dataset version (no retrain)"),
            ("5", "Logs       - Show inference log statistics"),
            ("6", "Versions   - List staged dataset versions"),
            ("7", "Configure  - View flywheel settings"),
            ("0", "Back"),
        ]
        choice = print_menu(menu_items, "Select action:")
        dispatch = {"1": self._handle_status, "2": self._handle_run_cycle,
                     "3": self._handle_readiness, "4": self._handle_stage,
                     "5": self._handle_logs, "6": self._handle_versions,
                     "7": self._handle_configure}
        return dispatch[choice]() if choice in dispatch else 0

    def _handle_status(self) -> int:
        """Display flywheel status: vLLM state, log counts, readiness."""
        try:
            _, orch = self._get_orchestrator()
            status = orch.status()
        except Exception as exc:
            logger.error("Failed to get flywheel status: %s", exc, exc_info=True)
            self.output_error(f"Failed to get status: {exc}", code="STATUS_ERROR")
            return 1

        if self.json_mode:
            self.output(asdict(status))
            return 0

        try:
            from tuner.ui import print_header
        except ImportError:
            print_header = print

        print_header("FLYWHEEL STATUS", "Current system state")
        print(f"  vLLM Server:        {'RUNNING' if status.vllm_running else 'STOPPED'}")
        if status.vllm_model:
            print(f"  Active Model:       {status.vllm_model}")
        if status.active_adapter:
            print(f"  Active Adapter:     {status.active_adapter}")
        print(f"  Total Logs:         {status.total_logs}")
        print(f"  Unprocessed Logs:   {status.unprocessed_logs}")
        print(f"  Last Cycle:         {status.last_cycle_at or 'Never'}")
        print(f"  Last Dataset Ver:   {status.last_dataset_version or 'None'}")
        if status.readiness:
            r = status.readiness
            print(f"\n  Retrain Ready:      {'YES' if r.ready else 'NO'}")
            print(f"  Est. SFT/KTO/GRPO:  {r.estimated_sft}/{r.estimated_kto}/{r.estimated_grpo}")
            print(f"  Avg Quality Score:  {r.avg_quality_score:.2f}")
            for reason in (r.reasons or []):
                print(f"    - {reason}")
        print()
        return 0

    def _handle_run_cycle(self) -> int:
        """Execute a full flywheel cycle (clean -> tag -> stage -> retrain)."""
        skip_retrain = getattr(self.args, "skip_retrain", False) if self.args else False
        retrain_mode = getattr(self.args, "retrain_mode", None) if self.args else None
        dry_run = getattr(self.args, "dry_run", False) if self.args else False

        try:
            _, orch = self._get_orchestrator()
            result = asyncio.run(orch.run_cycle(
                skip_retrain=skip_retrain, retrain_mode=retrain_mode, dry_run=dry_run,
            ))
        except Exception as exc:
            logger.error("Flywheel cycle failed: %s", exc, exc_info=True)
            self.output_error(f"Cycle failed: {exc}", code="CYCLE_ERROR")
            return 1

        if self.json_mode:
            self.output(asdict(result))
            return 0

        self.output_info(f"Flywheel cycle completed in {result.total_duration_seconds:.1f}s")
        if result.cleaning:
            print(f"  Cleaned:  {result.cleaning.total_processed} logs ({result.cleaning.scored} scored)")
        if result.tagging:
            print(f"  Tagged:   {result.tagging.total_processed} examples (sft={result.tagging.sft_count} kto={result.tagging.kto_count})")
        if result.staging:
            print(f"  Staged:   version {getattr(result.staging, 'version_id', '?')}")
        if result.training:
            t = result.training
            print(f"  Training: {'SUCCESS' if t.success else 'FAILED'} ({t.training_type}, {t.duration_seconds:.0f}s)")
        if result.hot_swap_success is not None:
            print(f"  Hot Swap: {'SUCCESS' if result.hot_swap_success else 'FAILED'}")
        print()
        return 0

    def _handle_configure(self) -> int:
        """Display current flywheel configuration."""
        try:
            config = self._load_config()
        except Exception as exc:
            self.output_error(f"Failed to load config: {exc}", code="CONFIG_ERROR")
            return 1

        if self.json_mode:
            self.output(asdict(config))
            return 0

        try:
            from tuner.ui import print_config, print_header, print_info
        except ImportError:
            print_config = lambda d, t: [print(f"  {k}: {v}") for k, v in d.items()]
            print_header = print_info = print

        print_header("FLYWHEEL CONFIGURATION", "Current settings")
        print_config({
            "SFT Threshold": str(config.sft_threshold),
            "KTO Min Threshold": str(config.kto_min_threshold),
            "Catalog Backend": config.catalog_backend,
            "Catalog Path": config.catalog_path,
            "Datasets Dir": config.datasets_dir,
            "vLLM Host:Port": f"{config.vllm_host}:{config.vllm_port}",
            "Proxy Port": str(config.proxy_port),
            "Retrain Mode": config.retrain_mode,
            "Min New Examples": str(config.min_new_examples),
            "Min SFT Examples": str(config.min_sft_examples),
            "Min Quality Score": str(config.min_quality_score),
        }, "Flywheel Configuration")
        print_info("Edit configs/flywheel/default.yaml to change settings.")
        return 0

    def _handle_readiness(self) -> int:
        """Check and display retrain readiness report."""
        try:
            _, orch = self._get_orchestrator()
            report = asyncio.run(orch.check_readiness())
        except Exception as exc:
            logger.error("Readiness check failed: %s", exc, exc_info=True)
            self.output_error(f"Readiness check failed: {exc}", code="READINESS_ERROR")
            return 1

        if self.json_mode:
            self.output(asdict(report))
            return 0

        self.output_info(f"Retrain readiness: {'READY' if report.ready else 'NOT READY'}")
        print(f"  New logs:           {report.new_log_count}")
        print(f"  Est. SFT/KTO/GRPO:  {report.estimated_sft}/{report.estimated_kto}/{report.estimated_grpo}")
        print(f"  Avg Quality Score:  {report.avg_quality_score:.2f}")
        for reason in (report.reasons or []):
            print(f"    - {reason}")
        print()
        return 0

    def _handle_stage(self) -> int:
        """Stage a new dataset version without retraining."""
        try:
            _, orch = self._get_orchestrator()
            result = asyncio.run(orch.run_cycle(skip_retrain=True, dry_run=False))
        except Exception as exc:
            logger.error("Staging failed: %s", exc, exc_info=True)
            self.output_error(f"Staging failed: {exc}", code="STAGE_ERROR")
            return 1

        if self.json_mode:
            self.output(asdict(result))
            return 0

        if result.staging:
            self.output_info(f"Dataset version {getattr(result.staging, 'version_id', 'unknown')} staged.")
        else:
            self.output_info("Staging completed (no new data to stage).")
        return 0

    def _handle_logs(self) -> int:
        """Show inference log statistics."""
        try:
            config, orch = self._get_orchestrator()
            status = orch.status()
        except Exception as exc:
            logger.error("Failed to get log stats: %s", exc, exc_info=True)
            self.output_error(f"Failed to get log stats: {exc}", code="LOGS_ERROR")
            return 1

        data = {"total_logs": status.total_logs, "unprocessed_logs": status.unprocessed_logs,
                "log_dir": config.log_dir}
        if self.json_mode:
            self.output(data)
            return 0

        self.output_info("Inference Log Statistics")
        print(f"  Total Logs:    {status.total_logs}")
        print(f"  Unprocessed:   {status.unprocessed_logs}")
        print(f"  Log Directory: {config.log_dir}")
        print()
        return 0

    def _handle_versions(self) -> int:
        """List all staged dataset versions with record counts."""
        try:
            config = self._load_config()
        except Exception as exc:
            self.output_error(f"Failed to load config: {exc}", code="CONFIG_ERROR")
            return 1

        datasets_dir = Path(self.repo_root / config.datasets_dir)
        if not datasets_dir.is_dir():
            if self.json_mode:
                self.output_list([], "dataset_versions")
            else:
                self.output_info(f"No dataset versions found in {datasets_dir}")
            return 0

        versions = []
        for vdir in sorted(datasets_dir.iterdir()):
            if not vdir.is_dir():
                continue
            jsonl_files = list(vdir.glob("*.jsonl"))
            total_lines = 0
            for jf in jsonl_files:
                try:
                    with open(jf) as f:
                        total_lines += sum(1 for _ in f)
                except OSError:
                    pass
            versions.append({"name": vdir.name, "path": str(vdir),
                             "files": len(jsonl_files), "total_examples": total_lines})

        if self.json_mode:
            self.output_list(versions, "dataset_versions")
            return 0
        if not versions:
            self.output_info("No staged dataset versions found.")
            return 0

        self.output_info(f"Staged Dataset Versions ({len(versions)} total)")
        print(f"  {'Version':<20} {'Files':<8} {'Examples':<10}")
        print(f"  {'-' * 38}")
        for v in versions:
            print(f"  {v['name']:<20} {v['files']:<8} {v['total_examples']:<10}")
        print()
        return 0
