"""
Cloud pipeline handler for training followed by HF Jobs cloud evaluation.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Dict, Optional

from tuner.backends.registry import TrainingBackendRegistry
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler
from tuner.handlers.cloud_eval_handler import CloudEvalHandler
from tuner.handlers.cloud_train_handler import CloudTrainHandler, PROVIDER_INFO
from tuner.ui import BOX, confirm, print_config, print_error, print_header, print_info, print_success


class CloudPipelineHandler(BaseHandler):
    """Train on HF Jobs, then evaluate the resulting run on HF Jobs via vLLM."""

    @property
    def name(self) -> str:
        return "cloud-pipeline"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _select_method(self, backend) -> Optional[str]:
        requested_method = getattr(self.args, "method", None)
        methods = backend.get_available_methods()
        if requested_method:
            if requested_method not in methods:
                raise CloudProviderError(
                    f"Unsupported training method '{requested_method}'. Available: {', '.join(methods)}"
                )
            return requested_method

        method_labels = CloudTrainHandler(args=self.args).load_method_labels()
        from tuner.ui import print_menu

        options = [(m, f"{BOX['bullet']} {method_labels.get(m, m.upper())}") for m in methods]
        if len(options) == 1:
            return options[0][0]
        return print_menu(options, "Select training method:")

    def _build_pipeline_display(
        self,
        *,
        training_config,
        eval_preset: Optional[str],
        eval_scenarios: Optional[list[str]],
        eval_tags: Optional[str],
    ) -> Dict[str, str]:
        display = CloudTrainHandler(args=self.args).build_config_display(
            training_config,
            PROVIDER_INFO["hf_jobs"],
        )
        display["Eval Preset"] = eval_preset or "-"
        display["Eval Scenarios"] = ", ".join(eval_scenarios) if eval_scenarios else "-"
        display["Eval Tags"] = eval_tags or "-"
        return display

    def _resolve_eval_args(self, *, training_config, artifact_prefix: str) -> Namespace:
        preset = getattr(self.args, "preset", None) or ("full" if not getattr(self.args, "scenario", None) else None)
        scenarios = getattr(self.args, "scenario", None)
        return Namespace(
            json=False,
            run=artifact_prefix,
            method=training_config.method,
            bucket=training_config.artifact_identifier,
            preset=preset,
            scenario=scenarios,
            tags=getattr(self.args, "tags", None),
            env_backend=getattr(self.args, "env_backend", None),
            env_template=getattr(self.args, "env_template", None),
            env_tool_schema=getattr(self.args, "env_tool_schema", None),
            env_exec_config=getattr(self.args, "env_exec_config", None),
            upload_to_hf=getattr(self.args, "upload_to_hf", None),
            update_model_card=bool(getattr(self.args, "update_model_card", False)),
            gpu=getattr(self.args, "gpu", None),
            timeout_hours=getattr(self.args, "timeout_hours", None),
            auto_confirm=True,
        )

    def handle(self) -> int:
        if self.json_mode:
            try:
                backend = TrainingBackendRegistry.get("hf_jobs", repo_root=self.repo_root)
                is_valid, error = backend.validate_environment()
                self.output(
                    {
                        "provider": "hf_jobs",
                        "status": "ready" if is_valid else "invalid_env",
                        "supports_eval": True,
                        "error": error or None,
                    }
                )
                return 0
            except Exception as exc:
                self.output_error(str(exc), code="CLOUD_PIPELINE_ERROR")
                return 1

        print_header("CLOUD PIPELINE", "Train on HF Jobs, then evaluate on HF Jobs via vLLM")

        try:
            backend = TrainingBackendRegistry.get("hf_jobs", repo_root=self.repo_root)
            is_valid, error = backend.validate_environment()
            if not is_valid:
                print_error(f"Environment validation failed: {error}")
                return 1

            method = self._select_method(backend)
            if not method:
                return 0

            training_config = CloudTrainHandler(args=self.args).apply_training_overrides(
                backend.load_config(method)
            )
            eval_preset = getattr(self.args, "preset", None) or ("full" if not getattr(self.args, "scenario", None) else None)
            eval_scenarios = CloudEvalHandler(args=self.args).resolve_display_scenarios(
                preset=eval_preset,
                scenarios=getattr(self.args, "scenario", None),
            )
            print_config(
                self._build_pipeline_display(
                    training_config=training_config,
                    eval_preset=eval_preset,
                    eval_scenarios=eval_scenarios,
                    eval_tags=getattr(self.args, "tags", None),
                ),
                "Cloud Pipeline Configuration",
            )
        except Exception as exc:
            print_error(str(exc))
            return 1

        if not getattr(self.args, "auto_confirm", False) and not confirm("Start cloud training and then evaluate the resulting run?"):
            print_info("Cloud pipeline cancelled.")
            return 0

        try:
            backend.show_post_training_actions = False
            print_header("STEP 1: CLOUD TRAINING")
            train_exit_code = backend.execute(training_config, python_path="")
        except Exception as exc:
            print_error(f"Cloud training failed: {exc}")
            return 1

        if train_exit_code != 0:
            print_error(f"Cloud training failed with exit code: {train_exit_code}")
            return train_exit_code

        artifact_prefix = getattr(backend, "last_artifact_prefix", None)
        if not artifact_prefix:
            print_error("Cloud training completed but did not report an artifact prefix for evaluation.")
            return 1

        print_success("Cloud training completed successfully.")
        print_header("STEP 2: CLOUD EVALUATION")
        eval_handler = CloudEvalHandler(args=self._resolve_eval_args(training_config=training_config, artifact_prefix=artifact_prefix))
        eval_handler._repo_root = self.repo_root
        return eval_handler.handle()
