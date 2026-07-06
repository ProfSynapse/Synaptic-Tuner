"""
Mech-interp handler for the Synaptic Tuner CLI.

Purpose: orchestrate the MechInterp verbs (extract, probe-fit, steer, score-gates)
Used by: router when the 'mechinterp' command is invoked.

Each sub-command is recipe-YAML driven, mirroring the other config-first verbs.
The handler is thin: it loads and validates the recipe, then delegates to the
MechInterp CLI layer. GPU-touching verbs (extract, steer) require an explicit
acknowledgement flag.

Sub-commands:
  extract      generate + capture hidden states to safetensors + manifest
  probe-fit    fit a linear readout and freeze a direction JSON
  steer        run the six-block steer cell (smoke-gated)
  score-gates  evaluate a gates.yaml against a per-row output JSONL
  list-configs show bundled example recipes
"""

import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

from tuner.handlers.base import BaseHandler

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
        if sub == "extract":
            return self._handle_extract()
        if sub == "probe-fit":
            return self._handle_probe_fit()
        if sub == "steer":
            return self._handle_steer()
        if sub == "score-gates":
            return self._handle_score_gates()

        msg = (
            "mechinterp requires a sub-command: extract, probe-fit, steer, "
            "score-gates, list-configs"
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

    def _handle_extract(self) -> int:
        from MechInterp.cli import run_extract
        from MechInterp.config import load_extract_config

        config_path = self._require("mechinterp_config", "--mi-config")
        model = self._require("model", "--model")
        if not config_path or not model:
            return 1
        config = load_extract_config(config_path)
        return run_extract(
            config,
            model_name=model,
            adapter=self._arg("adapter"),
            gpu_ack=bool(self._arg("i_know_this_runs_on_gpu", False)),
        )

    def _handle_probe_fit(self) -> int:
        from MechInterp.cli import run_probe_fit
        from MechInterp.config import load_probe_fit_config

        config_path = self._require("mechinterp_config", "--mi-config")
        if not config_path:
            return 1
        config = load_probe_fit_config(config_path)
        return run_probe_fit(config)

    def _handle_steer(self) -> int:
        from MechInterp.cli import run_steer
        from MechInterp.config import load_steer_config

        config_path = self._require("mechinterp_config", "--mi-config")
        model = self._require("model", "--model")
        render_fn = self._require("render_fn", "--render-fn")
        if not config_path or not model or not render_fn:
            return 1
        config = load_steer_config(config_path)
        return run_steer(
            config,
            model_name=model,
            adapter=self._arg("adapter"),
            render_fn_spec=render_fn,
            gpu_ack=bool(self._arg("i_know_this_runs_on_gpu", False)),
            force=bool(self._arg("force_full_run", False)),
        )

    def _handle_score_gates(self) -> int:
        from MechInterp.cli import run_score_gates

        gates_path = self._require("gates_config", "--gates-config")
        rows_path = self._require("rows_path", "--rows-path")
        if not gates_path or not rows_path:
            return 1
        return run_score_gates(
            gates_path, rows_path, arm_field=self._arg("arm_field", "arm") or "arm"
        )
