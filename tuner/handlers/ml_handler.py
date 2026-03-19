"""
ML training handler for the Synaptic Tuner CLI.

Location: /mnt/f/Code/Toolset-Training/tuner/handlers/ml_handler.py
Purpose: Orchestrate the ML (traditional/sklearn) training workflow via TUI and CLI
Used by: Router when 'ml' command is invoked; MainMenuHandler for interactive menu

Supports two sub-commands:
  - train: Select a config YAML, preview it, confirm, and run training
  - list-configs: Show available config templates

Supports --json flag for AI-parseable output. In JSON mode:
  - Returns structured status/results without interactive menus
  - All output is JSON formatted for programmatic parsing
"""

import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

import yaml

from tuner.handlers.base import BaseHandler
from tuner.ui import (
    print_menu,
    print_header,
    print_config,
    print_error,
    print_info,
    confirm,
    BOX,
)

logger = logging.getLogger(__name__)

# Template directory relative to repo root
TEMPLATES_REL = Path("Trainers") / "ml" / "configs" / "templates"


class MLHandler(BaseHandler):
    """
    Handler for traditional ML training workflow.

    Orchestrates:
    1. Config selection (from templates or user-provided path)
    2. Config summary display
    3. User confirmation
    4. Training execution via Trainers.ml.train.main()
    5. Results display

    JSON Mode (--json flag):
        Returns structured info about available configs and training status.

    Example:
        handler = MLHandler()
        exit_code = handler.handle()

        # CLI with sub-command
        handler = MLHandler(args=args)  # args.ml_subcommand = "train"
        exit_code = handler.handle()
    """

    def __init__(self, args: Optional[Namespace] = None):
        """Initialize handler with optional args."""
        super().__init__(args=args)

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "ml"

    def can_handle_direct_mode(self) -> bool:
        """Can be invoked as 'python -m tuner ml'."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _templates_dir(self) -> Path:
        """Return absolute path to the config templates directory."""
        return self.repo_root / TEMPLATES_REL

    def _discover_templates(self) -> list[dict]:
        """
        Discover available YAML config templates.

        Returns:
            List of dicts with 'name', 'path', and 'task_type' keys.
        """
        templates_dir = self._templates_dir()
        if not templates_dir.is_dir():
            return []

        templates = []
        for yaml_file in sorted(templates_dir.glob("*.yaml")):
            entry = {
                "name": yaml_file.stem,
                "path": str(yaml_file),
            }
            # Try to extract task type from YAML without full validation
            try:
                with open(yaml_file) as f:
                    raw = yaml.safe_load(f)
                entry["task_type"] = raw.get("task", {}).get("type", "unknown")
                entry["algorithm"] = raw.get("algorithm", {}).get("name", "unknown")
            except Exception:
                entry["task_type"] = "unknown"
                entry["algorithm"] = "unknown"
            templates.append(entry)
        return templates

    def _load_config_summary(self, config_path: Path) -> dict:
        """
        Load a YAML config and return a human-readable summary dict.

        Args:
            config_path: Absolute path to the YAML config file.

        Returns:
            Dict of key-value pairs for display via print_config.
        """
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        task = raw.get("task", {})
        data = raw.get("data", {})
        algo = raw.get("algorithm", {})
        features = raw.get("features", {})
        output = raw.get("output", {})

        # Count feature columns
        num_cols = len(features.get("numeric", {}).get("columns", []))
        cat_cols = len(features.get("categorical", {}).get("columns", []))

        summary = {
            "Task": task.get("name", "unnamed"),
            "Type": task.get("type", "unknown"),
            "Target": task.get("target_column", "unknown"),
            "Metric": task.get("eval_metric", "default"),
            "Algorithm": algo.get("name", "lightgbm"),
            "Data": data.get("train_path", "unknown"),
            "Test Split": str(data.get("test_size", 0.2)),
            "Features": f"{num_cols} numeric, {cat_cols} categorical",
            "Output": output.get("dir", "./ml_output"),
            "Config": str(config_path.relative_to(self.repo_root)),
        }
        return summary

    def _run_training(self, config_path: str) -> int:
        """
        Execute training via Trainers.ml.train.main().

        Args:
            config_path: Path string to the YAML config file.

        Returns:
            Exit code (0 = success, 1 = failure).
        """
        try:
            from Trainers.ml.train import main as ml_train_main

            run_dir = ml_train_main(config_path)
            print_info(f"Training complete. Output: {run_dir}")
            return 0
        except Exception as e:
            logger.error("ML training failed: %s", e, exc_info=True)
            print_error(f"Training failed: {e}")
            return 1

    # ------------------------------------------------------------------
    # JSON mode helpers
    # ------------------------------------------------------------------

    def _get_ml_status(self) -> dict:
        """
        Get ML training status for JSON output.

        Returns dict with available templates and capabilities.
        """
        templates = self._discover_templates()
        return {
            "command": "ml",
            "status": "ready" if templates else "no_templates",
            "templates": templates,
            "templates_dir": str(self._templates_dir()),
        }

    # ------------------------------------------------------------------
    # Sub-command: list-configs
    # ------------------------------------------------------------------

    def _handle_list_configs(self) -> int:
        """List available config templates."""
        templates = self._discover_templates()

        if self.json_mode:
            self.output_list(templates, "ml_configs")
            return 0

        if not templates:
            print_info("No config templates found.")
            print_info(f"Add YAML configs to: {self._templates_dir()}")
            return 0

        print_header("ML CONFIG TEMPLATES", "Available training configurations")
        for tpl in templates:
            task_type = tpl.get("task_type", "unknown")
            algorithm = tpl.get("algorithm", "unknown")
            print(f"  {BOX['bullet']} {tpl['name']}  ({task_type} / {algorithm})")
            print(f"    {tpl['path']}")
        print()
        return 0

    # ------------------------------------------------------------------
    # Sub-command: train (interactive or --config)
    # ------------------------------------------------------------------

    def _handle_train(self) -> int:
        """Run the interactive or CLI-driven training flow."""
        # CLI mode: --config <path> provided
        config_path_arg = getattr(self.args, "ml_config", None) if self.args else None
        if config_path_arg:
            config_path = Path(config_path_arg)
            if not config_path.exists():
                msg = f"Config file not found: {config_path}"
                if self.json_mode:
                    self.output_error(msg, code="CONFIG_NOT_FOUND")
                else:
                    print_error(msg)
                return 1

            if self.json_mode:
                summary = self._load_config_summary(config_path)
                self.output({"action": "train", "config": summary})
                return self._run_training(str(config_path))

            summary = self._load_config_summary(config_path)
            print_config(summary, "ML Training Configuration")
            if not confirm("Start ML training with this configuration?"):
                print_info("Training cancelled.")
                return 0
            return self._run_training(str(config_path))

        # JSON mode without --config: return available templates
        if self.json_mode:
            status = self._get_ml_status()
            self.output(status)
            return 0

        # Interactive mode: select from templates
        templates = self._discover_templates()
        if not templates:
            print_error("No config templates found.")
            print_info(f"Add YAML configs to: {self._templates_dir()}")
            return 1

        print_header("ML TRAINING", "Select a configuration to train")

        menu_items = [
            (
                str(i),
                f"{BOX['bullet']} {tpl['name']}  ({tpl.get('task_type', '?')} / {tpl.get('algorithm', '?')})",
            )
            for i, tpl in enumerate(templates)
        ]
        choice = print_menu(menu_items, "Select config:")
        if not choice:
            return 0

        selected = templates[int(choice)]
        config_path = Path(selected["path"])

        # Show config summary
        try:
            summary = self._load_config_summary(config_path)
        except Exception as e:
            print_error(f"Failed to load config: {e}")
            return 1

        print_config(summary, "ML Training Configuration")

        if not confirm("Start ML training with this configuration?"):
            print_info("Training cancelled.")
            return 0

        return self._run_training(str(config_path))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle(self) -> int:
        """
        Execute ML workflow.

        Routes to sub-commands: train (default), list-configs.

        Returns:
            int: Exit code (0 = success, non-zero = failure)
        """
        ml_sub = getattr(self.args, "ml_subcommand", None) if self.args else None

        if ml_sub == "list-configs":
            return self._handle_list_configs()

        # Default to train
        return self._handle_train()
