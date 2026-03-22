from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.experiment_tracking import load_experiment

from .base import BaseHandler


class ExperimentAnalysisHandler(BaseHandler):
    """Inspect a saved experiment bundle and surface recommendation outputs."""

    @property
    def name(self) -> str:
        return "analyze-experiment"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _resolve_experiment_id(self) -> str:
        requested = getattr(self.args, "experiment_id", None) or "latest"
        if requested != "latest":
            return requested

        experiments_root = self.repo_root / getattr(self.args, "base_dir", ".tracking") / "experiments"
        if not experiments_root.exists():
            raise FileNotFoundError(f"No experiments found under {experiments_root}")

        latest_id: str | None = None
        latest_created_at = ""
        for exp_dir in experiments_root.iterdir():
            if not exp_dir.is_dir():
                continue
            experiment_file = exp_dir / "experiment.json"
            if not experiment_file.exists():
                continue
            payload = json.loads(experiment_file.read_text(encoding="utf-8"))
            created_at = str(payload.get("created_at", ""))
            if created_at >= latest_created_at:
                latest_created_at = created_at
                latest_id = exp_dir.name
        if latest_id is None:
            raise FileNotFoundError(f"No experiments found under {experiments_root}")
        return latest_id

    @staticmethod
    def _load_json_if_exists(path_str: str | None) -> dict[str, Any] | None:
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def handle(self) -> int:
        try:
            experiment_id = self._resolve_experiment_id()
            base_dir = self.repo_root / getattr(self.args, "base_dir", ".tracking")
            experiment = load_experiment(experiment_id, base_dir=base_dir)
        except Exception as exc:
            self.output_error(str(exc), code="EXPERIMENT_NOT_FOUND")
            return 1

        analysis_dir = base_dir / "experiments" / experiment.experiment_id / "analysis"
        summary_path = experiment.derived_outputs.get("experiment_summary_json") or str(analysis_dir / "experiment_summary.json")
        next_candidates_path = experiment.derived_outputs.get("next_run_candidates_json") or str(analysis_dir / "next_run_candidates.json")
        hypothesis_path = experiment.derived_outputs.get("hypothesis_context_json") or str(analysis_dir / "hypothesis_context.json")

        summary_payload = self._load_json_if_exists(summary_path)
        candidates_payload = self._load_json_if_exists(next_candidates_path) or {"candidates": []}
        hypothesis_payload = self._load_json_if_exists(hypothesis_path)

        data = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "provider": experiment.provider,
            "method": experiment.method,
            "objective": experiment.objective,
            "stage_statuses": experiment.stage_statuses,
            "artifact_roots": experiment.artifact_roots,
            "derived_outputs": experiment.derived_outputs,
            "summary": summary_payload,
            "top_candidates": candidates_payload.get("candidates", [])[:5],
            "hypothesis_context": hypothesis_payload,
        }

        if self.json_mode:
            self.output(data, success=True)
            return 0

        lines = [
            f"Experiment: {experiment.experiment_id}",
            f"Name: {experiment.name}",
            f"Status: {experiment.status}",
            f"Provider/Method: {experiment.provider or '-'} / {experiment.method or '-'}",
            f"Stages: {', '.join(f'{stage}={status}' for stage, status in experiment.stage_statuses.items()) or '-'}",
        ]
        if summary_payload and summary_payload.get("eval_summary"):
            eval_summary = summary_payload["eval_summary"]
            lines.append(
                "Eval Summary: "
                f"{eval_summary.get('passed', 0)} passed, "
                f"{eval_summary.get('warned', 0)} warned, "
                f"{eval_summary.get('failed', 0)} failed"
            )
        if candidates_payload.get("candidates"):
            lines.append("Top Recommendations:")
            for candidate in candidates_payload["candidates"][:3]:
                lines.append(
                    f"- [{candidate.get('rank', '?')}] {candidate.get('hypothesis', 'n/a')}"
                )
        if next_candidates_path:
            lines.append(f"Next-run candidates: {next_candidates_path}")
        draft_spec = (
            experiment.derived_outputs.get("draft_next_spec_yaml")
            or experiment.derived_outputs.get("draft_next_experiment_yaml")
        )
        if draft_spec:
            lines.append(f"Draft next spec: {draft_spec}")
        self.output(data, "\n".join(lines), success=True)
        return 0
