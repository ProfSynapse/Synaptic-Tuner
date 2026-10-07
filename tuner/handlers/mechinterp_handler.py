"""
Mech-interp handler for the Synaptic Tuner CLI.

Purpose: orchestrate the MechInterp verbs (extract, probe-fit, steer,
dose-calibrate, score-gates)
Used by: router when the 'mechinterp' command is invoked.

Each sub-command is recipe-YAML driven, mirroring the other config-first verbs.
The handler is thin: it loads and validates the recipe, then delegates to the
MechInterp CLI layer. GPU-touching verbs (extract, steer) require an explicit
acknowledgement flag.

Sub-commands:
  run          run a multi-stage mechinterp pipeline from one config
  extract      generate + capture hidden states to safetensors + manifest
  probe-fit    fit a linear readout and freeze a direction JSON
  steer        run the six-block steer cell (smoke-gated)
  dose-calibrate resumable dose ladder over one or more frozen readouts
  score-gates  evaluate a gates.yaml against a per-row output JSONL
  list-configs show bundled example recipes
"""

import logging
import os
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Optional

from tuner.handlers.base import BaseHandler
from tuner.project import resolve_path

logger = logging.getLogger(__name__)

TEMPLATES_REL = Path("MechInterp") / "configs" / "templates"


class MechInterpHandler(BaseHandler):
    """Handler for mechanistic-interpretation cells."""

    def __init__(self, args: Optional[Namespace] = None):
        super().__init__(args=args)

    @property
    def name(self) -> str:
        return "mechinterp"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _templates_dir(self) -> Path:
        return self.repo_root / TEMPLATES_REL

    def _arg(self, key: str, default=None):
        return getattr(self.args, key, default) if self.args else default

    def handle(self) -> int:
        sub = self._arg("subcommand")
        if sub == "list-configs":
            return self._handle_list_configs()
        if sub == "run":
            return self._handle_run()
        if sub == "extract":
            return self._handle_extract()
        if sub == "probe-fit":
            return self._handle_probe_fit()
        if sub == "steer":
            return self._handle_steer()
        if sub == "dose-calibrate":
            return self._handle_dose_calibrate()
        if sub == "score-gates":
            return self._handle_score_gates()

        msg = (
            "mechinterp requires a sub-command: run, extract, probe-fit, steer, "
            "dose-calibrate, score-gates, list-configs"
        )
        if self.json_mode:
            self.output_error(msg, code="SUBCOMMAND_REQUIRED")
        else:
            print(msg)
        return 1

    def _handle_list_configs(self) -> int:
        tdir = self._templates_dir()
        templates = (
            [{"name": p.stem, "path": str(p)} for p in sorted(tdir.glob("*.yaml"))]
            if tdir.is_dir()
            else []
        )
        if self.json_mode:
            self.output_list(templates, "mechinterp_configs")
            return 0
        if not templates:
            print(f"No recipes found. Add YAML recipes to: {tdir}")
            return 0
        print("MechInterp example recipes:")
        for t in templates:
            print(f"  - {t['name']}  ({t['path']})")
        return 0

    def _require(self, key: str, label: str) -> Optional[str]:
        val = self._arg(key)
        if not val:
            msg = f"{label} is required"
            if self.json_mode:
                self.output_error(msg, code="ARG_REQUIRED")
            else:
                print(f"Error: {msg}")
            return None
        return val

    def _pipeline_config_arg(self) -> Optional[str]:
        return (
            self._arg("pipeline_config")
            or self._arg("ml_config")
            or self._arg("mechinterp_config")
        )

    def _input_path(self, value: str) -> Path:
        return resolve_path(value, self.context, from_cli=True, access="read")

    def _handle_run(self) -> int:
        from MechInterp.pipeline import (
            build_pipeline_plan,
            load_pipeline_config,
            run_local_pipeline,
        )

        config_path = self._pipeline_config_arg()
        if not config_path:
            self.output_error(
                "mechinterp run requires --config <pipeline.yaml>",
                code="ARG_REQUIRED",
            )
            return 1

        try:
            resolved_config_path = self._input_path(config_path)
            cfg = load_pipeline_config(resolved_config_path, context=self.context)
            provider = self._arg("provider") or cfg.runtime.provider
            plan = build_pipeline_plan(
                cfg,
                repo_root=self.repo_root,
                context=self.context,
                provider=provider,
                only_step=self._arg("only_step"),
                from_step=self._arg("from_step"),
                skip_steps=self._arg("skip_step") or [],
                gpu_ack=bool(self._arg("i_know_this_runs_on_gpu", False)),
                force=bool(self._arg("force_full_run", False)),
            )
        except Exception as exc:
            self.output_error(str(exc), code="MECHINTERP_PIPELINE_CONFIG")
            return 1

        if self._arg("dry_run", False):
            self.output(plan, "MechInterp pipeline dry-run plan:")
            return 0

        if provider == "local":
            if not self._arg("auto_confirm", False):
                self.output_error(
                    "Refusing to run without --yes. Use --dry-run to inspect the plan first.",
                    code="CONFIRMATION_REQUIRED",
                )
                return 2
            try:
                return run_local_pipeline(
                    cfg,
                    repo_root=self.repo_root,
                    context=self.context,
                    only_step=self._arg("only_step"),
                    from_step=self._arg("from_step"),
                    skip_steps=self._arg("skip_step") or [],
                    gpu_ack=bool(self._arg("i_know_this_runs_on_gpu", False)),
                    force=bool(self._arg("force_full_run", False)),
                )
            except Exception as exc:
                self.output_error(str(exc), code="MECHINTERP_PIPELINE_RUN_FAILED")
                return 1

        if provider == "modal":
            if not self._arg("auto_confirm", False):
                self.output_error(
                    "Refusing to submit Modal work without --yes. Use --dry-run first.",
                    code="CONFIRMATION_REQUIRED",
                )
                return 2
            return self._submit_modal_pipeline(str(resolved_config_path), cfg)

        self.output_error(f"Unsupported mechinterp provider: {provider}", code="BAD_PROVIDER")
        return 1

    def _git_value(self, args: list[str], default: str = "") -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=self.repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return default

    def _submit_modal_pipeline(self, config_path: str, cfg) -> int:
        repo_url = self._arg("repo_url") or cfg.repo.url or self._git_value(["config", "--get", "remote.origin.url"])
        repo_branch = self._arg("repo_branch") or cfg.repo.branch or self._git_value(["branch", "--show-current"], "main")
        repo_commit = self._arg("repo_commit") or cfg.repo.commit or self._git_value(["rev-parse", "HEAD"])
        if not repo_url or not repo_commit:
            self.output_error(
                "Modal mechinterp runs require a repo URL and exact commit. Pass --repo-url/--repo-commit or set repo.* in config.",
                code="REPO_SOURCE_REQUIRED",
            )
            return 1

        modal_cfg = cfg.modal
        env = {
            **os.environ,
            "MI_MODAL_APP_NAME": modal_cfg.app_name or f"mechinterp-{cfg.name}",
            "MI_MODAL_IMAGE": modal_cfg.image,
            "MI_MODAL_GPU": self._arg("gpu") or modal_cfg.gpu,
            "MI_MODAL_TIMEOUT_HOURS": str(self._arg("timeout_hours") or modal_cfg.timeout_hours),
            "MI_MODAL_CHECKPOINT_INTERVAL_SEC": str(modal_cfg.checkpoint_interval_sec),
            "MI_MODAL_VOLUME": modal_cfg.volume_name,
            "MI_MODAL_MOUNT": modal_cfg.mount_path,
            "MI_MODAL_PIP": "\n".join(modal_cfg.pip),
            "MI_MODAL_APT": "\n".join(modal_cfg.apt),
        }

        runner = self.repo_root / "MechInterp" / "cloud" / "modal_runner.py"
        cmd = [
            "modal",
            "run",
            "--detach",
            str(runner),
            "--config",
            config_path,
            "--repo-url",
            repo_url,
            "--repo-branch",
            repo_branch,
            "--repo-commit",
            repo_commit,
        ]
        if self._arg("only_step"):
            cmd.extend(["--only-step", self._arg("only_step")])
        if self._arg("from_step"):
            cmd.extend(["--from-step", self._arg("from_step")])
        for step in self._arg("skip_step") or []:
            cmd.extend(["--skip-step", step])
        if self._arg("i_know_this_runs_on_gpu", False):
            cmd.append("--i-know-this-runs-on-gpu")
        if self._arg("force_full_run", False):
            cmd.append("--force-full-run")

        try:
            return subprocess.run(cmd, cwd=self.repo_root, env=env).returncode
        except FileNotFoundError:
            self.output_error("Modal CLI not found. Install with: pip install modal", code="MODAL_NOT_FOUND")
            return 1

    def _handle_extract(self) -> int:
        from MechInterp.cli import run_extract
        from MechInterp.config import load_extract_config

        config_path = self._require("mechinterp_config", "--mi-config")
        model = self._require("model", "--model")
        if not config_path or not model:
            return 1
        config = load_extract_config(
            self._input_path(config_path), context=self.context
        )
        return run_extract(
            config,
            model_name=model,
            revision=self._arg("model_revision"),
            adapter=self._arg("adapter"),
            gpu_ack=bool(self._arg("i_know_this_runs_on_gpu", False)),
        )

    def _handle_probe_fit(self) -> int:
        from MechInterp.cli import run_probe_fit
        from MechInterp.config import load_probe_fit_config

        config_path = self._require("mechinterp_config", "--mi-config")
        if not config_path:
            return 1
        config = load_probe_fit_config(
            self._input_path(config_path), context=self.context
        )
        return run_probe_fit(config)

    def _handle_steer(self) -> int:
        from MechInterp.cli import run_steer
        from MechInterp.config import load_steer_config

        config_path = self._require("mechinterp_config", "--mi-config")
        model = self._require("model", "--model")
        if not config_path or not model:
            return 1
        config = load_steer_config(
            self._input_path(config_path), context=self.context
        )
        render_fn = self._arg("render_fn") or config.execution.render_fn
        if not render_fn:
            self.output_error(
                "steer requires execution.render_fn in the cell config or --render-fn",
                code="ARG_REQUIRED",
            )
            return 1
        return run_steer(
            config,
            model_name=model,
            revision=self._arg("model_revision"),
            adapter=self._arg("adapter"),
            render_fn_spec=render_fn,
            gpu_ack=bool(self._arg("i_know_this_runs_on_gpu", False)),
            force=bool(self._arg("force_full_run", False)),
            project_context=self.context,
        )

    def _handle_dose_calibrate(self) -> int:
        from MechInterp.cli import run_dose_calibration
        from MechInterp.config import load_dose_calibration_config

        config_path = self._require("mechinterp_config", "--mi-config")
        model = self._require("model", "--model")
        if not config_path or not model:
            return 1
        config = load_dose_calibration_config(
            self._input_path(config_path), context=self.context
        )
        render_fn = self._arg("render_fn") or config.execution.render_fn
        if not render_fn:
            self.output_error(
                "dose-calibrate requires execution.render_fn in the config or --render-fn",
                code="ARG_REQUIRED",
            )
            return 1
        return run_dose_calibration(
            config,
            model_name=model,
            adapter=self._arg("adapter"),
            render_fn_spec=render_fn,
            gpu_ack=bool(self._arg("i_know_this_runs_on_gpu", False)),
            project_context=self.context,
        )

    def _handle_score_gates(self) -> int:
        from MechInterp.cli import run_score_gates

        gates_path = self._require("gates_config", "--gates-config")
        rows_path = self._require("rows_path", "--rows-path")
        if not gates_path or not rows_path:
            return 1
        return run_score_gates(
            str(self._input_path(gates_path)),
            str(self._input_path(rows_path)),
            arm_field=self._arg("arm_field", "arm") or "arm",
        )
