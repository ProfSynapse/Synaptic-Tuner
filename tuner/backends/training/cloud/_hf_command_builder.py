"""
HF Jobs command builder mixin.

Location: tuner/backends/training/cloud/_hf_command_builder.py
Purpose: Build shell commands for HF Jobs Docker containers
Used by: HFJobsBackend (via mixin inheritance) in hf_jobs_backend.py

Encapsulates all logic for constructing the bash command that runs inside
the HF Jobs container: dependency installation, repo checkout, training
script invocation, and GRPO virtual-env bootstrapping.
"""

import shlex
import yaml

from shared.utilities.paths import get_canonical_trainer_dir_name
from shared.utilities.unique_ids import unique_utc_timestamp
from tuner.cloud import RepoCheckoutSpec, build_repo_checkout_steps
from tuner.core.config import CloudTrainingConfig
from tuner.core.exceptions import CloudProviderError

from .base_cloud import load_project_deps


class HFCommandBuilderMixin:
    """Methods for building training commands sent to HF Jobs containers."""

    ENV_GRPO_CONFIG_NAME = "env_config.yaml"
    ENV_GRPO_SCRIPT_NAME = "train_env_grpo.py"

    @classmethod
    def _config_filename_for_method(cls, method: str) -> str:
        return cls.ENV_GRPO_CONFIG_NAME if method == "grpo" else "config.yaml"

    @classmethod
    def _script_name_for_method(cls, method: str) -> str:
        return cls.ENV_GRPO_SCRIPT_NAME if method == "grpo" else f"train_{method}.py"

    @staticmethod
    def _dataset_display(dataset_config: dict) -> str:
        local_file = dataset_config.get("local_file")
        if local_file:
            return str(local_file)

        dataset_name = str(dataset_config.get("dataset_name") or "").strip()
        dataset_file = str(dataset_config.get("dataset_file") or "").strip()
        if dataset_name and dataset_file:
            return f"{dataset_name}/{dataset_file}"
        if dataset_file:
            return dataset_file
        if dataset_name:
            return dataset_name
        return "Unknown"

    @staticmethod
    def _new_run_timestamp() -> str:
        """Return a launch timestamp with a short nonce to avoid same-second collisions."""
        return unique_utc_timestamp()

    def _build_artifact_prefix(self, config: CloudTrainingConfig, timestamp: str) -> str:
        """Build the canonical artifact prefix used by HF Jobs runs."""
        if not config.repo_commit:
            raise CloudProviderError("HF Jobs requires exact repo source metadata.")
        run_slug = f"{timestamp}-{config.repo_commit[:8]}"
        return f"runs/{config.provider}/{config.method}/{run_slug}"

    def _build_training_command(self, config: CloudTrainingConfig, timestamp: str | None = None) -> str:
        """
        Build a shell command string for the HF Jobs container.

        The command installs training dependencies, clones the repo,
        and runs the training script. HF auth is passed via job secrets
        when the job is submitted.

        Args:
            config: Cloud training configuration
            timestamp: Optional run timestamp (generated if not provided)

        Returns:
            Shell command string to pass as ["bash", "-c", command]
        """
        if not config.repo_url or not config.repo_branch or not config.repo_commit:
            raise CloudProviderError("HF Jobs requires exact repo source metadata.")
        if not config.artifact_identifier:
            raise CloudProviderError("HF Jobs requires an artifact bucket identifier.")

        timestamp = timestamp or self._new_run_timestamp()
        artifact_prefix = self._build_artifact_prefix(config, timestamp)

        # Read project-specific deps from cloud_config.yaml (single source of truth)
        cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        project_deps = load_project_deps(cloud_config_path)
        checkout_steps = build_repo_checkout_steps(
            RepoCheckoutSpec(
                url=config.repo_url,
                branch=config.repo_branch,
                commit=config.repo_commit,
            )
        )

        python_cmd = "$(command -v python3 || command -v python)"
        parts = [
            # Install project-specific deps only; unsloth, trl, transformers,
            # datasets, peft, and PyTorch are pre-installed in the Docker image
            f"{python_cmd} -m pip install --upgrade {' '.join(project_deps)}",
            "mkdir -p /tmp/hf-bucket-sync-site",
            f"{python_cmd} -m pip install --upgrade --target /tmp/hf-bucket-sync-site huggingface_hub>=1.5.0 hf_transfer",
            f"export HF_BUCKET_SYNC_PYTHON={python_cmd}",
            "export HF_BUCKET_SYNC_PYTHONPATH=/tmp/hf-bucket-sync-site",
            f"export CLOUD_PROVIDER={config.provider}",
            f"export CLOUD_GPU_TYPE={config.hf_flavor or config.gpu_type}",
            # Enable fast HF transfers
            "export HF_HUB_ENABLE_HF_TRANSFER=1",
            *checkout_steps,
        ]
        if config.pip_packages:
            quoted_pip_packages = " ".join(shlex.quote(pkg) for pkg in config.pip_packages)
            parts.append(f"{python_cmd} -m pip install --upgrade {quoted_pip_packages}")

        training_args: list[str] = []
        if config.model_name:
            training_args.extend(["--model-name", config.model_name])
        if config.dataset_name:
            training_args.extend(["--dataset-name", config.dataset_name])
        if config.dataset_file:
            dataset_file_arg = config.dataset_file
            if config.dataset_name and dataset_file_arg.startswith(f"{config.dataset_name}/"):
                dataset_file_arg = dataset_file_arg[len(config.dataset_name) + 1 :]
            training_args.extend(["--dataset-file", dataset_file_arg])
        if config.batch_size is not None:
            training_args.extend(["--batch-size", str(config.batch_size)])
        if config.save_steps is not None:
            training_args.extend(["--save-steps", str(config.save_steps)])
        if config.save_total_limit is not None:
            training_args.extend(["--save-total-limit", str(config.save_total_limit)])
        if config.gradient_accumulation_steps is not None:
            training_args.extend(["--gradient-accumulation", str(config.gradient_accumulation_steps)])
        if config.learning_rate:
            training_args.extend(["--learning-rate", str(config.learning_rate)])
        if config.seed is not None:
            training_args.extend(["--seed", str(config.seed)])
        if config.method in ("dpo", "kto") and config.beta is not None:
            training_args.extend(["--beta", str(config.beta)])
        if config.epochs is not None:
            training_args.extend(["--num-epochs", str(config.epochs)])
        if config.max_steps is not None:
            training_args.extend(["--max-steps", str(config.max_steps)])
        if config.max_seq_length is not None:
            training_args.extend(["--max-seq-length", str(config.max_seq_length)])
        if config.method == "sft" and config.load_in_4bit is not None:
            training_args.append("--load-in-4bit" if config.load_in_4bit else "--no-load-in-4bit")
        if config.method == "sft" and config.lora_r is not None:
            training_args.extend(["--lora-r", str(config.lora_r)])
        if config.method == "sft" and config.lora_alpha is not None:
            training_args.extend(["--lora-alpha", str(config.lora_alpha)])
        if config.method == "sft" and config.lora_dropout is not None:
            training_args.extend(["--lora-dropout", str(config.lora_dropout)])
        if config.method == "sft" and config.use_dora:
            training_args.append("--use-dora")
        if config.method == "sft" and config.use_rslora:
            training_args.append("--use-rslora")
        if config.method == "sft" and config.init_lora_weights is not None:
            training_args.extend(["--init-lora-weights", str(config.init_lora_weights)])
        if config.method == "sft" and config.lora_target_modules:
            target_modules_arg = (
                config.lora_target_modules
                if isinstance(config.lora_target_modules, str)
                else ",".join(config.lora_target_modules)
            )
            training_args.extend(["--lora-target-modules", target_modules_arg])
        if config.method == "sft" and config.evolutionary_enabled:
            training_args.append("--evolutionary-enabled")
        if config.method == "sft" and config.evolutionary_candidates is not None:
            training_args.extend(["--evolutionary-candidates", str(config.evolutionary_candidates)])
        if config.method == "sft" and config.evolutionary_eval_batch_size is not None:
            training_args.extend(["--evolutionary-eval-batch-size", str(config.evolutionary_eval_batch_size)])
        if config.method == "sft" and config.evolutionary_validation_config:
            training_args.extend(["--evolutionary-validation-config", str(config.evolutionary_validation_config)])
        if config.method == "sft" and config.evolutionary_strategy:
            training_args.extend(["--evolutionary-strategy", str(config.evolutionary_strategy)])
        if config.method == "sft" and config.evolutionary_noise_scale is not None:
            training_args.extend(["--evolutionary-noise-scale", str(config.evolutionary_noise_scale)])
        if config.method == "sft" and config.evolutionary_max_grad_norm is not None:
            training_args.extend(["--evolutionary-max-grad-norm", str(config.evolutionary_max_grad_norm)])
        if config.method == "sft" and config.evolutionary_scale_factors:
            training_args.extend(
                ["--evolutionary-scale-factors", ",".join(str(value) for value in config.evolutionary_scale_factors)]
            )
        if config.method == "sft" and config.evolutionary_selection_method:
            training_args.extend(["--evolutionary-selection-method", str(config.evolutionary_selection_method)])
        if config.method == "sft" and config.evolutionary_min_improvement is not None:
            training_args.extend(["--evolutionary-min-improvement", str(config.evolutionary_min_improvement)])
        if config.method == "sft" and config.evolutionary_min_relative_improvement is not None:
            training_args.extend(["--evolutionary-min-relative-improvement", str(config.evolutionary_min_relative_improvement)])
        if config.method == "sft" and config.evolutionary_noise_floor_epsilon is not None:
            training_args.extend(["--evolutionary-noise-floor-epsilon", str(config.evolutionary_noise_floor_epsilon)])
        if config.method == "sft" and config.evolutionary_eval_frequency is not None:
            training_args.extend(["--evolutionary-eval-frequency", str(config.evolutionary_eval_frequency)])
        if config.method == "sft" and config.evolutionary_warmup_steps is not None:
            training_args.extend(["--evolutionary-warmup-steps", str(config.evolutionary_warmup_steps)])
        if config.method == "sft" and config.evolutionary_cache_baseline is not None:
            training_args.append("--evolutionary-cache-baseline" if config.evolutionary_cache_baseline else "--evolutionary-no-cache-baseline")
        if config.method == "sft" and config.evolutionary_log_candidates is not None:
            training_args.append("--evolutionary-log-candidates" if config.evolutionary_log_candidates else "--evolutionary-no-log-candidates")
        if config.method == "sft" and config.evolutionary_log_selected is not None:
            training_args.append("--evolutionary-log-selected" if config.evolutionary_log_selected else "--evolutionary-no-log-selected")
        training_args_str = ""
        if config.method == "grpo":
            output_dir = f"/workspace/repo/Trainers/{get_canonical_trainer_dir_name(config.method)}/env_grpo_output/{timestamp}"
            training_args.extend(["--output-dir", output_dir])
        if training_args:
            training_args_str = " " + " ".join(shlex.quote(arg) for arg in training_args)

        if config.method == "grpo":
            parts.extend(
                self._build_env_grpo_steps(
                    config=config,
                    timestamp=timestamp,
                    artifact_prefix=artifact_prefix,
                    python_cmd=python_cmd,
                    training_args=training_args_str,
                )
            )
        else:
            parts.append(
                f"cd /workspace/repo/Trainers/{get_canonical_trainer_dir_name(config.method)} && "
                "python "
                f"{self._script_name_for_method(config.method)} "
                f"--run-timestamp {timestamp} "
                f"--output-root {config.artifact_mount_path} "
                f"--cloud-provider {config.provider} "
                f"--artifact-backend {config.artifact_backend} "
                f"--artifact-bucket {config.artifact_identifier} "
                f"--artifact-prefix {artifact_prefix} "
                f"{'--publish-final-model' if config.publish_final_model else ''} "
                f"{f'--publish-target-repo {config.publish_target_repo}' if config.publish_target_repo else ''}"
                f"{training_args_str}"
            )

        return " && ".join(parts)

    def _build_env_grpo_steps(
        self,
        *,
        config: CloudTrainingConfig,
        timestamp: str,
        artifact_prefix: str,
        python_cmd: str,
        training_args: str,
    ) -> list[str]:
        with open(config.config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

        runtime_cfg = ((raw_config.get("env_training") or {}).get("runtime") or {})
        venv_dir = str(runtime_cfg.get("isolated_venv_dir") or "/workspace/.venvs/grpo-openenv")
        python_packages = list(runtime_cfg.get("python_packages") or [])
        project_pip_deps = list(runtime_cfg.get("project_pip_deps") or [])
        install_args = " ".join(shlex.quote(str(part)) for part in ["pip", "setuptools", "wheel", *project_pip_deps, *python_packages])
        trainer_dir = f"/workspace/repo/Trainers/{get_canonical_trainer_dir_name(config.method)}"
        output_dir = f"{trainer_dir}/env_grpo_output/{timestamp}"
        remote_uri = self._build_remote_run_uri(config, artifact_prefix)

        return [
            "mkdir -p /tmp/grpo-openenv-bootstrap",
            f"{python_cmd} -m pip install --upgrade --target /tmp/grpo-openenv-bootstrap virtualenv",
            "export PYTHONPATH=/tmp/grpo-openenv-bootstrap:$PYTHONPATH",
            f"{python_cmd} -m virtualenv --no-download {shlex.quote(venv_dir)}",
            f". {shlex.quote(venv_dir)}/bin/activate",
            f"python -m pip install --upgrade {install_args}",
            f"cd {trainer_dir} && python {self.ENV_GRPO_SCRIPT_NAME} --config {trainer_dir}/configs/{self.ENV_GRPO_CONFIG_NAME}{training_args}",
            (
                f"if [ -d {shlex.quote(output_dir)} ]; then "
                f"cd /workspace/repo && "
                f"PYTHONPATH=/tmp/hf-bucket-sync-site:$PYTHONPATH python -m shared.hf_bucket_sync_helper "
                f"{shlex.quote(output_dir)} {shlex.quote(remote_uri)}; "
                "fi"
            ),
        ]
