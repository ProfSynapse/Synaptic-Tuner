"""
Cloud training workflow handler.

Location: tuner/handlers/cloud_train_handler.py
Purpose: Orchestrate cloud training workflow (provider selection, config, job submission)
Used by: Router when 'cloud' command is invoked, MainMenuHandler for cloud training option

Manages the user workflow for submitting training jobs to cloud GPU providers:
1. Select a legacy training backend (HF Jobs or RunPod)
2. Validate provider credentials/environment
3. Select training method (SFT, KTO)
4. Load and display configuration with cost estimate
5. Confirm with user
6. Submit job and stream logs

Supports --json flag for AI-parseable output.
"""

import logging
import os
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tuner.backends.training.cloud.base_cloud import resolve_cloud_image
from tuner.cloud import (
    build_runtime_layout,
    build_source_lock,
    checkout_policy_from_context,
    ssh_checkout_policy_from_environment,
    standalone_credential_from_environment,
)
from tuner.cloud.hf_jobs import require_current_hf_source_submission_authorization
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler
from tuner.backends.registry import TrainingBackendRegistry
from tuner.ui import (
    print_menu,
    print_header,
    print_config,
    print_error,
    print_info,
    print_success,
    confirm,
    BOX,
)

logger = logging.getLogger(__name__)

# Provider display metadata
PROVIDER_INFO = {
    "hf_jobs": {
        "name": "HuggingFace Jobs",
        "description": "Managed GPU training via HF infrastructure",
        "install_hint": "pip install --upgrade huggingface_hub>=0.27.0",
        "env_var": "HF_TOKEN",
    },
    "runpod": {
        "name": "RunPod",
        "description": "On-demand GPU pods with Docker support",
        "install_hint": "pip install runpod",
        "env_var": "RUNPOD_API_KEY",
    },
}


class CloudTrainHandler(BaseHandler):
    """
    Handler for cloud training workflow.

    Orchestrates the process of submitting training jobs to cloud GPU providers.
    Follows the same pattern as TrainHandler but adds provider selection,
    cost estimation, and cloud-specific configuration.

    Graceful degradation: providers whose SDKs aren't installed are shown
    in the menu with "(not installed)" annotation rather than being hidden,
    so users know what options exist.

    Example:
        handler = CloudTrainHandler()
        exit_code = handler.handle()

        # With JSON mode
        handler = CloudTrainHandler(args=args)  # args.json = True
        exit_code = handler.handle()  # Returns JSON status
    """

    def __init__(self, args: Optional[Namespace] = None):
        """Initialize handler with optional args."""
        super().__init__(args=args)

    @property
    def name(self) -> str:
        """Handler identifier."""
        return "cloud"

    def can_handle_direct_mode(self) -> bool:
        """Can be invoked as 'python tuner.py cloud'."""
        return True

    def _get_provider_status(self, *, validate_environment: bool = True) -> List[Dict]:
        """
        Check availability and status of each cloud provider.

        ``validate_environment=False`` is the inspection-only path used by
        JSON status output. It reports registry metadata without constructing
        a backend or resolving provider credentials.

        Returns:
            List of dicts with provider id, name, status, and details
        """
        providers = []

        for provider_id, info in PROVIDER_INFO.items():
            status = {
                "id": provider_id,
                "name": info["name"],
                "registered": provider_id in TrainingBackendRegistry.list(),
                "env_ready": False,
                "detail": "",
            }

            if status["registered"] and validate_environment:
                try:
                    backend = TrainingBackendRegistry.get(provider_id, repo_root=self.repo_root)
                    is_valid, error = backend.validate_environment()
                    status["env_ready"] = is_valid
                    status["detail"] = "" if is_valid else error
                except Exception as e:
                    status["detail"] = str(e)
            elif status["registered"]:
                status["detail"] = "Registered; credentials not checked in inspection mode"
            else:
                status["detail"] = f"Not installed (run: {info['install_hint']})"

            providers.append(status)

        return providers

    def _get_cloud_status(self) -> dict:
        """
        Get cloud training status for JSON output.

        Returns dict with available providers, their status, and methods.
        """
        providers = self._get_provider_status(validate_environment=False)
        return {
            "command": "cloud",
            "status": "inspection_only",
            "submission_enabled": False,
            "credentials_checked": False,
            "providers": providers,
        }

    def _prepare_source_contract(self):
        """Build and validate source/layout provenance before provider choice."""

        run_id = getattr(self.args, "run_id", None) or (
            "cloud-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        standalone_credential = standalone_credential_from_environment(os.environ)
        ssh_policy = ssh_checkout_policy_from_environment(os.environ)
        source_lock = build_source_lock(
            self.context,
            run_id=run_id,
            mode=getattr(self.args, "source_mode", None),
            environment=os.environ,
            provider_secret=getattr(self, "_source_provider_secret", None),
            credential_helper=getattr(self, "_source_credential_helper", None),
            standalone_credential=standalone_credential,
            ssh_policy=ssh_policy,
        )
        policy = checkout_policy_from_context(
            self.context,
            ssh_policy=ssh_policy,
            source_lock=source_lock,
        )
        policy.validate(source_lock.project_source.location)
        policy.validate(source_lock.engine_source.location)
        return source_lock, build_runtime_layout(self.context), policy

    def _build_provider_menu(self, providers: List[Dict]) -> List[Tuple[str, str]]:
        """
        Build menu options for provider selection.

        Shows all known providers with status indicators:
        - Ready: provider name with checkmark
        - Installed but not configured: provider name with warning
        - Not installed: provider name with install hint

        Args:
            providers: List of provider status dicts from _get_provider_status()

        Returns:
            List of (provider_id, display_string) tuples for print_menu
        """
        menu_options = []

        for provider in providers:
            info = PROVIDER_INFO[provider["id"]]

            if provider["env_ready"]:
                label = f"{BOX['star']} {info['name']} (ready)"
            elif provider["registered"]:
                # SDK installed but credentials not configured
                short_detail = provider["detail"].split(".")[0] if provider["detail"] else "needs setup"
                label = f"{BOX['bullet']} {info['name']} ({short_detail})"
            else:
                label = f"{BOX['bullet']} {info['name']} (not installed -- run: {info['install_hint']})"

            menu_options.append((provider["id"], label))

        return menu_options

    def handle(self) -> int:
        """
        Execute cloud training workflow.

        In JSON mode, returns cloud provider status without interactive prompts.

        Returns:
            int: Exit code (0 = success, non-zero = failure)
        """
        # JSON mode: return status information
        if self.json_mode:
            status = self._get_cloud_status()
            self.output(status)
            return 0

        print_header("CLOUD TRAINING", "Train models on cloud GPU providers")

        # Source identity, cleanliness, pushed state, policy, and filesystem
        # layout are established before provider selection or paid execution.
        try:
            source_lock, runtime_layout, checkout_policy = self._prepare_source_contract()
        except Exception as exc:
            print_error(f"Cloud source preflight failed: {exc}")
            return 1

        # Source preflight establishes what would run, but does not authorize
        # provider interaction. Fail closed before capability probes, menus,
        # SDK/backend construction, credential resolution, or compilation.
        try:
            require_current_hf_source_submission_authorization(
                route="cloud-train.handle"
            )
        except CloudProviderError as exc:
            print_error(f"Cloud launch authorization failed: {exc}")
            return 1

        # Step 1: Check provider availability
        providers = self._get_provider_status()

        # Step 2: Show provider selection menu
        menu_options = self._build_provider_menu(providers)
        provider_choice = print_menu(menu_options, "Select cloud provider:")

        if not provider_choice:
            return 0  # User selected back/exit

        # Step 3: Check if provider is usable
        provider_status = next(
            (p for p in providers if p["id"] == provider_choice), None
        )

        if not provider_status:
            print_error(f"Unknown provider: {provider_choice}")
            return 1

        if not provider_status["registered"]:
            info = PROVIDER_INFO[provider_choice]
            print_error(
                f"{info['name']} SDK not installed.\n"
                f"  Install with: {info['install_hint']}"
            )
            return 1

        # Step 4: Get backend and validate environment
        try:
            backend = TrainingBackendRegistry.get(provider_choice, repo_root=self.repo_root)
        except ValueError as e:
            print_error(str(e))
            return 1

        is_valid, error = backend.validate_environment()
        if not is_valid:
            print_error(f"Environment validation failed: {error}")
            return 1

        # Step 5: Select training method
        methods = backend.get_available_methods()
        method_labels = self._load_method_labels()
        method_options = [
            (m, f"{BOX['bullet']} {method_labels.get(m, m.upper())}") for m in methods
        ]

        if len(methods) > 1:
            method = print_menu(method_options, "Select training method:")
            if not method:
                return 0
        else:
            method = methods[0]
            print_info(f"Using method: {method.upper()}")

        # Step 6: Load configuration
        try:
            config = backend.load_config(method)
            config = self._apply_training_overrides(config)
            # Provider integrations consume these canonical objects as they
            # migrate; attaching rather than re-modeling keeps one source SSOT.
            config.source_lock = source_lock
            config.runtime_layout = runtime_layout
            config.checkout_policy = checkout_policy
        except Exception as e:
            print_error(f"Failed to load configuration: {e}")
            return 1

        # Step 7: Display configuration with cost estimate
        info = PROVIDER_INFO[provider_choice]
        config_display = self._build_config_display(config, info)
        print_config(config_display, "Cloud Training Configuration")

        # Step 8: Confirm with user
        if not getattr(self.args, "auto_confirm", False) and not confirm("Start cloud training with this configuration?"):
            print_info("Cloud training cancelled.")
            return 0

        # Step 9: Execute training
        print_info(f"Submitting job to {info['name']}...")
        print()

        try:
            exit_code = backend.execute(config, python_path="")
        except Exception as e:
            print_error(f"Cloud training failed: {e}")
            return 1

        if exit_code == 0:
            print_success("Cloud training completed successfully.")
        else:
            print_error(f"Cloud training failed with exit code: {exit_code}")

        return exit_code

    def _apply_training_overrides(self, config):
        """Apply direct CLI overrides to a loaded cloud training config."""
        args = self.args
        if not args:
            return config

        train_model_name = getattr(args, "train_model_name", None)
        if train_model_name:
            config.model_name = train_model_name

        train_dataset_name = getattr(args, "train_dataset_name", None)
        if train_dataset_name:
            config.dataset_name = train_dataset_name

        train_dataset_file = getattr(args, "train_dataset_file", None)
        if train_dataset_file:
            config.dataset_file = train_dataset_file

        train_batch_size = getattr(args, "train_batch_size", None)
        if train_batch_size is not None:
            config.batch_size = train_batch_size

        train_save_steps = getattr(args, "train_save_steps", None)
        if train_save_steps is not None:
            config.save_steps = train_save_steps

        train_save_total_limit = getattr(args, "train_save_total_limit", None)
        if train_save_total_limit is not None:
            config.save_total_limit = train_save_total_limit

        train_gradient_accumulation = getattr(args, "train_gradient_accumulation", None)
        if train_gradient_accumulation is not None:
            config.gradient_accumulation_steps = train_gradient_accumulation

        train_learning_rate = getattr(args, "train_learning_rate", None)
        if train_learning_rate is not None:
            config.learning_rate = train_learning_rate

        train_seed = getattr(args, "train_seed", None)
        if train_seed is not None:
            config.seed = train_seed

        train_beta = getattr(args, "train_beta", None)
        if train_beta is not None and config.method in ("dpo", "kto"):
            config.beta = train_beta

        train_num_epochs = getattr(args, "train_num_epochs", None)
        if train_num_epochs is not None:
            config.epochs = train_num_epochs

        train_max_steps = getattr(args, "train_max_steps", None)
        if train_max_steps is not None:
            config.max_steps = train_max_steps

        train_max_seq_length = getattr(args, "train_max_seq_length", None)
        if train_max_seq_length is not None:
            config.max_seq_length = train_max_seq_length

        if getattr(args, "train_load_in_4bit", None) is not None:
            config.load_in_4bit = args.train_load_in_4bit

        train_lora_r = getattr(args, "train_lora_r", None)
        if train_lora_r is not None:
            config.lora_r = train_lora_r

        train_lora_alpha = getattr(args, "train_lora_alpha", None)
        if train_lora_alpha is not None:
            config.lora_alpha = train_lora_alpha

        train_lora_dropout = getattr(args, "train_lora_dropout", None)
        if train_lora_dropout is not None:
            config.lora_dropout = train_lora_dropout

        if getattr(args, "train_use_dora", False):
            config.use_dora = True

        if getattr(args, "train_use_rslora", False):
            config.use_rslora = True

        train_init_lora_weights = getattr(args, "train_init_lora_weights", None)
        if train_init_lora_weights is not None:
            config.init_lora_weights = train_init_lora_weights

        train_lora_target_modules = getattr(args, "train_lora_target_modules", None)
        if train_lora_target_modules:
            normalized = train_lora_target_modules.strip()
            if normalized == "all-linear":
                config.lora_target_modules = normalized
            else:
                config.lora_target_modules = [
                    module.strip()
                    for module in normalized.split(",")
                    if module.strip()
                ]

        if getattr(args, "train_evolutionary_enabled", False):
            config.evolutionary_enabled = True

        train_evolutionary_candidates = getattr(args, "train_evolutionary_candidates", None)
        if train_evolutionary_candidates is not None:
            config.evolutionary_candidates = train_evolutionary_candidates

        train_evolutionary_eval_batch_size = getattr(args, "train_evolutionary_eval_batch_size", None)
        if train_evolutionary_eval_batch_size is not None:
            config.evolutionary_eval_batch_size = train_evolutionary_eval_batch_size

        train_evolutionary_validation_config = getattr(args, "train_evolutionary_validation_config", None)
        if train_evolutionary_validation_config is not None:
            config.evolutionary_validation_config = train_evolutionary_validation_config

        train_evolutionary_strategy = getattr(args, "train_evolutionary_strategy", None)
        if train_evolutionary_strategy is not None:
            config.evolutionary_strategy = train_evolutionary_strategy

        train_evolutionary_noise_scale = getattr(args, "train_evolutionary_noise_scale", None)
        if train_evolutionary_noise_scale is not None:
            config.evolutionary_noise_scale = train_evolutionary_noise_scale

        train_evolutionary_max_grad_norm = getattr(args, "train_evolutionary_max_grad_norm", None)
        if train_evolutionary_max_grad_norm is not None:
            config.evolutionary_max_grad_norm = train_evolutionary_max_grad_norm

        train_evolutionary_scale_factors = getattr(args, "train_evolutionary_scale_factors", None)
        if train_evolutionary_scale_factors:
            config.evolutionary_scale_factors = [
                float(value.strip())
                for value in train_evolutionary_scale_factors.split(",")
                if value.strip()
            ]

        train_evolutionary_selection_method = getattr(args, "train_evolutionary_selection_method", None)
        if train_evolutionary_selection_method is not None:
            config.evolutionary_selection_method = train_evolutionary_selection_method

        train_evolutionary_min_improvement = getattr(args, "train_evolutionary_min_improvement", None)
        if train_evolutionary_min_improvement is not None:
            config.evolutionary_min_improvement = train_evolutionary_min_improvement

        train_evolutionary_min_relative_improvement = getattr(args, "train_evolutionary_min_relative_improvement", None)
        if train_evolutionary_min_relative_improvement is not None:
            config.evolutionary_min_relative_improvement = train_evolutionary_min_relative_improvement

        train_evolutionary_noise_floor_epsilon = getattr(args, "train_evolutionary_noise_floor_epsilon", None)
        if train_evolutionary_noise_floor_epsilon is not None:
            config.evolutionary_noise_floor_epsilon = train_evolutionary_noise_floor_epsilon

        train_evolutionary_eval_frequency = getattr(args, "train_evolutionary_eval_frequency", None)
        if train_evolutionary_eval_frequency is not None:
            config.evolutionary_eval_frequency = train_evolutionary_eval_frequency

        train_evolutionary_warmup_steps = getattr(args, "train_evolutionary_warmup_steps", None)
        if train_evolutionary_warmup_steps is not None:
            config.evolutionary_warmup_steps = train_evolutionary_warmup_steps

        if getattr(args, "train_evolutionary_cache_baseline", None) is not None:
            config.evolutionary_cache_baseline = args.train_evolutionary_cache_baseline

        if getattr(args, "train_evolutionary_log_candidates", None) is not None:
            config.evolutionary_log_candidates = args.train_evolutionary_log_candidates

        if getattr(args, "train_evolutionary_log_selected", None) is not None:
            config.evolutionary_log_selected = args.train_evolutionary_log_selected

        train_gpu = getattr(args, "train_gpu", None)
        if train_gpu:
            config.gpu_type = train_gpu
            if hasattr(config, "hf_flavor"):
                config.hf_flavor = train_gpu

        train_timeout_hours = getattr(args, "train_timeout_hours", None)
        if train_timeout_hours is not None:
            config.timeout_hours = train_timeout_hours

        train_cloud_image = getattr(args, "train_cloud_image", None)
        if train_cloud_image:
            config.cloud_image = train_cloud_image
            config.cloud_image_profile = None

        train_image_profile = getattr(args, "train_image_profile", None)
        if train_image_profile:
            cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
            config.cloud_image, config.cloud_image_profile = resolve_cloud_image(
                cloud_config_path,
                requested_profile=train_image_profile,
                fallback_image=config.cloud_image,
            )

        if config.dataset_name and config.dataset_file and "/" not in config.dataset_file:
            config.dataset_file = f"{config.dataset_name}/{config.dataset_file}"

        return config

    def _load_method_labels(self) -> Dict[str, str]:
        """
        Load training method display labels from Trainers/methods.yaml.

        Reads the method_labels section of the dedicated, backend-agnostic
        methods.yaml so that adding or changing a label only requires a YAML
        edit, not a code change (mirrors _load_gpu_tiers). A method with no
        entry falls back to its uppercased code at the call site (see the menu
        builder's `method_labels.get(m, m.upper())`).

        Returns:
            Dict mapping method codes to human-readable labels.
        """
        from tuner.backends.training.cloud.base_cloud import load_method_labels

        config_path = self.repo_root / "Trainers" / "methods.yaml"
        return load_method_labels(config_path)

    def _load_gpu_tiers(self) -> Dict[str, Dict]:
        """
        Load GPU tier definitions from cloud_config.yaml.

        Reads the gpu_tiers section so that adding or changing tiers
        only requires a YAML edit, not a code change.

        Returns:
            Dict mapping tier names to their config (description,
            provider GPU identifiers, approximate cost).
        """
        from tuner.backends.training.cloud.base_cloud import load_gpu_tiers

        config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        return load_gpu_tiers(config_path)

    def _build_config_display(
        self, config, provider_info: Dict
    ) -> Dict[str, str]:
        """
        Build configuration display dict for print_config.

        Args:
            config: CloudTrainingConfig instance
            provider_info: Provider metadata from PROVIDER_INFO

        Returns:
            Ordered dict of config key-value pairs for display
        """
        from tuner.backends.training.cloud.base_cloud import (
            estimate_cost,
            get_gpu_display_name,
        )

        display = {
            "Provider": provider_info["name"],
            "Method": config.method.upper(),
        }

        # Model name (strip org prefix for display)
        model_display = config.model_name
        if "/" in model_display:
            model_display = model_display.split("/")[-1]
        display["Model"] = model_display

        # Dataset (just filename)
        if config.dataset_file and config.dataset_file != "Unknown":
            display["Dataset"] = Path(config.dataset_file).name
        else:
            display["Dataset"] = "Unknown"

        # GPU info
        if hasattr(config, "gpu_type") and config.gpu_type:
            gpu_name = get_gpu_display_name(config.provider, config.gpu_type)
            display["GPU"] = gpu_name

        # Timeout
        if hasattr(config, "timeout_hours"):
            display["Timeout"] = f"{config.timeout_hours:.0f} hours"

        if getattr(config, "cloud_image_profile", None):
            display["Image Profile"] = config.cloud_image_profile

        if getattr(config, "cloud_image", None):
            image = config.cloud_image
            display["Image"] = image if len(image) <= 72 else f"{image[:69]}..."

        # Cost estimate
        if hasattr(config, "gpu_type") and hasattr(config, "timeout_hours"):
            cost = estimate_cost(config.provider, config.gpu_type, config.timeout_hours)
            if cost:
                display["Est. Cost"] = cost

        # Training params
        display["Epochs"] = str(config.epochs)
        display["Batch Size"] = str(config.batch_size)
        if getattr(config, "gradient_accumulation_steps", None) is not None:
            display["Grad Accum"] = str(config.gradient_accumulation_steps)
        display["Learning Rate"] = str(config.learning_rate)
        if getattr(config, "save_steps", None) is not None:
            display["Save Steps"] = str(config.save_steps)
        if getattr(config, "save_total_limit", None) is not None:
            display["Save Total Limit"] = str(config.save_total_limit)
        if getattr(config, "max_steps", None) is not None:
            display["Max Steps"] = str(config.max_steps)
        if getattr(config, "max_seq_length", None) is not None:
            display["Max Seq Len"] = str(config.max_seq_length)
        if getattr(config, "load_in_4bit", None) is not None:
            display["4-bit Load"] = "yes" if config.load_in_4bit else "no"

        return display
