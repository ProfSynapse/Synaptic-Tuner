"""Blind hardware planning handler for experiment specs."""

from __future__ import annotations

from pathlib import Path

from shared.experiment_tracking import ExperimentSpec, load_experiment_spec
from tuner.cloud.hardware_planner import format_stage_plan_json, plan_experiment_hardware
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler
from tuner.ui import print_config, print_error, print_header, print_info


class HardwarePlanHandler(BaseHandler):
    @property
    def name(self) -> str:
        return "plan-hardware"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _load_spec(self) -> tuple[ExperimentSpec, Path]:
        spec_path = getattr(self.args, "experiment_spec", None)
        if not spec_path:
            raise CloudProviderError("plan-hardware requires --experiment-spec <path>.")
        path = Path(spec_path).expanduser().resolve()
        if not path.exists():
            raise CloudProviderError(f"Experiment spec not found: {path}")
        return load_experiment_spec(path), path

    def handle(self) -> int:
        try:
            spec, spec_path = self._load_spec()
            plans = plan_experiment_hardware(
                spec=spec,
                optimize_for=getattr(self.args, "optimize_for", "balanced"),
                max_hourly_price=getattr(self.args, "max_hourly_price", None),
            )
        except Exception as exc:
            if self.json_mode:
                self.output_error(str(exc), code="HARDWARE_PLAN_ERROR")
                return 1
            print_error(str(exc))
            return 1

        payload = {
            "experiment_spec": str(spec_path),
            "optimize_for": getattr(self.args, "optimize_for", "balanced"),
            "max_hourly_price": getattr(self.args, "max_hourly_price", None),
            "plans": {stage: format_stage_plan_json(plan) for stage, plan in plans.items()},
        }
        if self.json_mode:
            self.output(payload)
            return 0

        print_header("HARDWARE PLAN", f"Blind stage hardware planning for {spec.name}")
        print_info(f"Objective: {payload['optimize_for']}")
        if payload["max_hourly_price"] is not None:
            print_info(f"Hourly price cap: ${payload['max_hourly_price']:.2f}")

        for stage_name, stage_plan in plans.items():
            recommendation = stage_plan.recommendation
            if recommendation is None:
                print_config({"Recommendation": "No feasible GPU found"}, f"{stage_name.title()} Hardware")
                continue

            plan_rows = {
                "Recommended Flavor": recommendation.flavor,
                "GPU": recommendation.pretty_name,
                "Price/hr": f"${recommendation.price_hr:.2f}",
                "Estimated Memory": f"{recommendation.estimated_memory_gb:.1f} GB",
                "Estimated Headroom": f"{recommendation.estimated_headroom_gb:.1f} GB",
                "Estimated Hours": f"{recommendation.estimated_hours:.2f}" if recommendation.estimated_hours is not None else "relative only",
                "Estimated Cost": f"${recommendation.estimated_cost:.2f}" if recommendation.estimated_cost is not None else "relative only",
            }
            if recommendation.recommended_batch_size:
                plan_rows["Batch Size"] = str(recommendation.recommended_batch_size)
            if recommendation.recommended_gradient_accumulation:
                plan_rows["Grad Accum"] = str(recommendation.recommended_gradient_accumulation)
            print_config(plan_rows, f"{stage_name.title()} Hardware")

        return 0
