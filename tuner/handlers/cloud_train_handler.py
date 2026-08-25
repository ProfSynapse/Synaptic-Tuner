"""
Cloud training workflow handler.

Location: tuner/handlers/cloud_train_handler.py
Purpose: Orchestrate cloud training workflow (provider selection, config, job submission)
Used by: Router when 'cloud' command is invoked, MainMenuHandler for cloud training option

Manages the user workflow for submitting training jobs to cloud GPU providers:
1. Select cloud provider (HF Jobs, Modal, RunPod)
2. Validate provider credentials/environment
3. Select training method (SFT, KTO)
4. Load and display configuration with cost estimate
5. Confirm with user
6. Submit job and stream logs

Supports --json flag for AI-parseable output.
"""

import logging
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from synaptic_tuner.api.v1 import (
    CloudSourceContract,
    CloudTrainingAPI,
    CloudTrainingRequest,
)

from tuner.cloud.hf_jobs import require_current_hf_source_submission_authorization
from tuner.backends.registry import TrainingBackendRegistry  # compatibility patch seam
from tuner.handlers.base import BaseHandler
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
    "modal": {
        "name": "Modal",
        "description": "Serverless GPU compute with auto-scaling",
        "install_hint": "pip install modal && modal setup",
        "env_var": None,  # Uses OAuth or MODAL_TOKEN_ID
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
        """Return provider discovery through the public training API."""

        return CloudTrainingAPI(self.context).provider_statuses(
            validate_environment=validate_environment
        )

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
        """Delegate source/layout provenance to the public training API."""

        contract = CloudTrainingAPI(self.context).prepare_source(
            run_id=getattr(self.args, "run_id", None),
            mode=getattr(self.args, "source_mode", None),
            provider_secret=getattr(self, "_source_provider_secret", None),
            credential_helper=getattr(self, "_source_credential_helper", None),
        )
        return (
            contract.source_lock,
            contract.runtime_layout,
            contract.checkout_policy,
        )

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

        api = CloudTrainingAPI(
            self.context,
            hf_authorizer=require_current_hf_source_submission_authorization,
        )

        # Provider discovery and configuration stay behind the API boundary.
        try:
            methods = api.provider_methods(provider_choice)
        except Exception as exc:
            print_error(f"Environment validation failed: {exc}")
            return 1

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

        try:
            request = self._build_training_request(provider_choice, method)
            plan = api.prepare(
                request,
                source=CloudSourceContract(
                    source_lock=source_lock,
                    runtime_layout=runtime_layout,
                    checkout_policy=checkout_policy,
                ),
                validate_environment=False,
            )
            config = plan._config
        except Exception as exc:
            print_error(f"Failed to prepare cloud training: {exc}")
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
            result = api.submit(plan)
            exit_code = result.exit_code
        except Exception as e:
            print_error(f"Cloud training failed: {e}")
            return 1

        if exit_code == 0:
            print_success("Cloud training completed successfully.")
        else:
            print_error(f"Cloud training failed with exit code: {exit_code}")

        return exit_code

    def _build_training_request(
        self, provider: str, method: str
    ) -> CloudTrainingRequest:
        """Translate argparse fields into the stable API request contract."""

        args = self.args or Namespace()
        training = {}
        for argument, field_name in {
            "train_batch_size": "batch_size",
            "train_save_steps": "save_steps",
            "train_save_total_limit": "save_total_limit",
            "train_gradient_accumulation": "gradient_accumulation_steps",
            "train_learning_rate": "learning_rate",
            "train_seed": "seed",
            "train_num_epochs": "epochs",
            "train_max_steps": "max_steps",
            "train_max_seq_length": "max_seq_length",
            "train_load_in_4bit": "load_in_4bit",
            "train_evolutionary_candidates": "evolutionary_candidates",
            "train_evolutionary_eval_batch_size": "evolutionary_eval_batch_size",
            "train_evolutionary_validation_config": "evolutionary_validation_config",
            "train_evolutionary_strategy": "evolutionary_strategy",
            "train_evolutionary_noise_scale": "evolutionary_noise_scale",
            "train_evolutionary_max_grad_norm": "evolutionary_max_grad_norm",
            "train_evolutionary_selection_method": "evolutionary_selection_method",
            "train_evolutionary_min_improvement": "evolutionary_min_improvement",
            "train_evolutionary_min_relative_improvement": "evolutionary_min_relative_improvement",
            "train_evolutionary_noise_floor_epsilon": "evolutionary_noise_floor_epsilon",
            "train_evolutionary_eval_frequency": "evolutionary_eval_frequency",
            "train_evolutionary_warmup_steps": "evolutionary_warmup_steps",
            "train_evolutionary_cache_baseline": "evolutionary_cache_baseline",
            "train_evolutionary_log_candidates": "evolutionary_log_candidates",
            "train_evolutionary_log_selected": "evolutionary_log_selected",
        }.items():
            value = getattr(args, argument, None)
            if value is not None:
                training[field_name] = value
        if method in {"dpo", "kto"} and getattr(args, "train_beta", None) is not None:
            training["beta"] = args.train_beta
        if getattr(args, "train_evolutionary_enabled", False):
            training["evolutionary_enabled"] = True
        scale_factors = getattr(args, "train_evolutionary_scale_factors", None)
        if scale_factors:
            training["evolutionary_scale_factors"] = [
                float(value.strip())
                for value in scale_factors.split(",")
                if value.strip()
            ]

        lora = {}
        for argument, field_name in {
            "train_lora_r": "r",
            "train_lora_alpha": "alpha",
            "train_lora_dropout": "dropout",
            "train_init_lora_weights": "init_lora_weights",
        }.items():
            value = getattr(args, argument, None)
            if value is not None:
                lora[field_name] = value
        if getattr(args, "train_use_dora", False):
            lora["use_dora"] = True
        if getattr(args, "train_use_rslora", False):
            lora["use_rslora"] = True
        target_modules = getattr(args, "train_lora_target_modules", None)
        if target_modules:
            normalized = target_modules.strip()
            lora["target_modules"] = (
                normalized
                if normalized == "all-linear"
                else [item.strip() for item in normalized.split(",") if item.strip()]
            )

        runtime = {}
        for argument, field_name in {
            "train_gpu": "gpu_type",
            "train_timeout_hours": "timeout_hours",
            "train_cloud_image": "cloud_image",
            "train_image_profile": "image_profile",
        }.items():
            value = getattr(args, argument, None)
            if value is not None:
                runtime[field_name] = value

        return CloudTrainingRequest(
            provider=provider,
            method=method,
            model_name=getattr(args, "train_model_name", None),
            dataset_name=getattr(args, "train_dataset_name", None),
            dataset_file=getattr(args, "train_dataset_file", None),
            training=training,
            lora=lora,
            runtime=runtime,
            run_id=getattr(args, "run_id", None),
        )

    def _apply_training_overrides(self, config):
        """Compatibility adapter backed by the public request mapper."""

        request = self._build_training_request(
            getattr(config, "provider", None) or getattr(config, "platform", ""),
            config.method,
        )
        return CloudTrainingAPI.apply_request(config, request)

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
