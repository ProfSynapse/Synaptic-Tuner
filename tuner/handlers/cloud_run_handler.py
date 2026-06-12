"""Generic cloud job runner using provider-backed executors."""

from __future__ import annotations

import shlex
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from shared.utilities.unique_ids import unique_utc_timestamp
from tuner.backends.training.cloud.base_cloud import load_cloud_config, load_project_deps, resolve_repo_source
from tuner.cloud import (
    CloudJobSpec,
    HF_BUCKET_SYNC_OVERLAY_PACKAGES,
    HFJobExecutor,
    RepoCheckoutSpec,
    build_bash_command,
    build_secrets_from_env,
    build_repo_checkout_steps,
    load_huggingface_hub,
    resolve_hf_bucket_id,
)
from tuner.core.exceptions import CloudProviderError
from tuner.discovery.recipes import load_recipe
from tuner.handlers.base import BaseHandler
from tuner.ui import BOX, confirm, print_config, print_error, print_header, print_info, print_menu

_HF_SYNC_OVERLAY = "/tmp/hf-bucket-sync-site"


class _SafeTemplateDict(dict):
    def __missing__(self, key):
        raise CloudProviderError(f"Missing template variable '{key}' in cloud job config.")


class CloudRunHandler(BaseHandler):
    """Submit config-driven cloud jobs, starting with HF Jobs."""

    @property
    def name(self) -> str:
        return "cloud-run"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _jobs_dir(self) -> Path:
        return self.repo_root / "Trainers" / "recipes"

    def _cloud_config_path(self) -> Path:
        return self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"

    def _load_cloud_config(self) -> dict:
        return load_cloud_config(self._cloud_config_path())

    def _hf_defaults(self) -> dict:
        return self._load_cloud_config().get("hf_jobs", {})

    def _list_job_configs(self) -> List[Path]:
        jobs_dir = self._jobs_dir()
        if not jobs_dir.exists():
            return []
        return sorted(path for path in jobs_dir.glob("*.yaml") if path.is_file())

    def _resolve_job_config_path(self, requested: Optional[str]) -> Path:
        if requested:
            candidate = Path(requested)
            if not candidate.is_absolute():
                candidate = self.repo_root / requested
                if not candidate.exists():
                    candidate = self._jobs_dir() / requested
            if candidate.exists():
                return candidate.resolve()
            raise CloudProviderError(f"Cloud job config not found: {requested}")

        configs = self._list_job_configs()
        if self.json_mode:
            raise CloudProviderError("JSON mode requires --job-config for cloud-run.")
        if not configs:
            raise CloudProviderError(f"No cloud job configs found under {self._jobs_dir()}")

        options = [(str(path), f"{BOX['bullet']} {path.stem}") for path in configs]
        choice = print_menu(options, "Select cloud job config:")
        if not choice:
            raise CloudProviderError("Cloud run cancelled.")
        return Path(choice)

    def _load_job_config(self, path: Path) -> Dict[str, Any]:
        try:
            data = load_recipe(path, "cloud")
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise CloudProviderError(
                f"Cloud job config must be a YAML object: {path} ({exc})"
            ) from exc
        if not isinstance(data, dict):
            raise CloudProviderError(f"Cloud job config must be a YAML object: {path}")
        return data

    def _render_value(self, value: Any, variables: Dict[str, str]) -> Any:
        if isinstance(value, str):
            return value.format_map(_SafeTemplateDict(variables))
        if isinstance(value, list):
            return [self._render_value(item, variables) for item in value]
        if isinstance(value, dict):
            return {str(k): self._render_value(v, variables) for k, v in value.items()}
        return value

    @staticmethod
    def _new_job_timestamp() -> str:
        return unique_utc_timestamp()

    def _compile_hf_job(self, config_path: Path, config: Dict[str, Any]) -> tuple[CloudJobSpec, Dict[str, Any]]:
        provider = str(config.get("provider", "hf_jobs")).strip().lower()
        if provider != "hf_jobs":
            raise CloudProviderError(f"Unsupported provider for cloud-run: {provider}")

        huggingface_hub = load_huggingface_hub(require_apis=("run_job", "create_bucket"))
        repo_source = resolve_repo_source(self.repo_root)
        hf_defaults = self._hf_defaults()
        job_cfg = config.get("job", {}) if isinstance(config.get("job"), dict) else {}
        repo_cfg = config.get("repo", {}) if isinstance(config.get("repo"), dict) else {}
        setup_cfg = config.get("setup", {}) if isinstance(config.get("setup"), dict) else {}
        run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
        artifacts_cfg = config.get("artifacts", {}) if isinstance(config.get("artifacts"), dict) else {}

        timestamp = self._new_job_timestamp()
        job_name = str(config.get("name") or config_path.stem).strip()
        repo_dir = str(repo_cfg.get("clone_dir", "/workspace/repo")).strip() or "/workspace/repo"
        output_dir = str(artifacts_cfg.get("output_dir", "/workspace/outputs")).strip() or "/workspace/outputs"
        template_vars = {
            "provider": provider,
            "job_name": job_name,
            "timestamp": timestamp,
            "commit": repo_source.commit,
            "commit8": repo_source.commit[:8],
            "repo_dir": repo_dir,
            "output_dir": output_dir,
            "config_path": str(config_path),
        }
        extra_vars = run_cfg.get("template_vars", {}) if isinstance(run_cfg.get("template_vars"), dict) else {}
        template_vars.update({str(k): str(v) for k, v in extra_vars.items()})

        resolved_bucket = None
        artifact_prefix = None
        if artifacts_cfg.get("bucket"):
            resolved_bucket = resolve_hf_bucket_id(
                huggingface_hub,
                str(artifacts_cfg["bucket"]),
                token=build_secrets_from_env(["HF_TOKEN", "HF_API_KEY"]).get("HF_TOKEN"),
                private=True,
            )
            prefix_template = str(
                artifacts_cfg.get("prefix", "runs/{provider}/custom/{job_name}/{timestamp}-{commit8}")
            )
            artifact_prefix = prefix_template.format_map(_SafeTemplateDict(template_vars))
            template_vars["artifact_bucket"] = resolved_bucket
            template_vars["artifact_prefix"] = artifact_prefix
            template_vars["artifact_uri"] = f"hf://buckets/{resolved_bucket}/{artifact_prefix}"

        steps: List[str] = []
        if bool(repo_cfg.get("checkout", True)):
            steps.extend(
                build_repo_checkout_steps(
                    RepoCheckoutSpec(
                        url=repo_source.url,
                        branch=repo_source.branch,
                        commit=repo_source.commit,
                        clone_dir=repo_dir,
                    )
                )
            )

        if bool(setup_cfg.get("use_project_deps", True)):
            project_deps = load_project_deps(self._cloud_config_path())
            steps.append(f"cd {shlex.quote(repo_dir)} && python -m pip install --upgrade {' '.join(project_deps)}")

        extra_pip = setup_cfg.get("pip", [])
        if isinstance(extra_pip, list) and extra_pip:
            quoted = " ".join(shlex.quote(str(item)) for item in extra_pip)
            steps.append(f"cd {shlex.quote(repo_dir)} && python -m pip install --upgrade {quoted}")

        if resolved_bucket:
            sync_deps = " ".join(shlex.quote(dep) for dep in HF_BUCKET_SYNC_OVERLAY_PACKAGES)
            steps.extend(
                [
                    f"mkdir -p {_HF_SYNC_OVERLAY}",
                    f"cd {shlex.quote(repo_dir)} && python -m pip install --upgrade --target {_HF_SYNC_OVERLAY} {sync_deps}",
                    f"mkdir -p {shlex.quote(output_dir)}",
                ]
            )

        for step in setup_cfg.get("steps", []) if isinstance(setup_cfg.get("steps"), list) else []:
            steps.append(self._render_value(str(step), template_vars))

        env_vars = self._render_value(run_cfg.get("env", {}), template_vars)
        if not isinstance(env_vars, dict):
            env_vars = {}
        for key, value in env_vars.items():
            steps.append(f"export {key}={shlex.quote(str(value))}")

        run_steps = run_cfg.get("steps", [])
        if not isinstance(run_steps, list) or not run_steps:
            raise CloudProviderError("Cloud job config must define run.steps as a non-empty list.")
        for step in run_steps:
            steps.append(self._render_value(str(step), template_vars))

        if resolved_bucket and artifact_prefix:
            steps.append(
                " && ".join(
                    [
                        f"cd {shlex.quote(repo_dir)}",
                        f"PYTHONPATH={shlex.quote(_HF_SYNC_OVERLAY)}:$PYTHONPATH python -m shared.hf_bucket_sync_helper {shlex.quote(output_dir)} {shlex.quote(template_vars['artifact_uri'])}",
                    ]
                )
            )

        secret_names: List[str] = []
        if bool(run_cfg.get("include_hf_token", True)):
            secret_names.extend(["HF_TOKEN", "HF_API_KEY"])
        from_env = run_cfg.get("secrets_from_env", [])
        if isinstance(from_env, list):
            secret_names.extend(str(item) for item in from_env)
        secrets = build_secrets_from_env(secret_names)

        spec = CloudJobSpec(
            provider=provider,
            image=str(job_cfg.get("image") or hf_defaults.get("image") or ""),
            command=build_bash_command(steps),
            flavor=str(getattr(self.args, "gpu", None) or job_cfg.get("flavor") or hf_defaults.get("flavor") or "a10g-small"),
            timeout_hours=float(getattr(self.args, "timeout_hours", None) or job_cfg.get("timeout_hours") or 4.0),
            secrets=secrets,
            namespace=str(job_cfg["namespace"]) if job_cfg.get("namespace") else None,
            labels=self._render_value(job_cfg.get("labels", {}), template_vars) if isinstance(job_cfg.get("labels"), dict) else {},
        )
        metadata = {
            "name": job_name,
            "description": str(config.get("description", "")).strip(),
            "artifact_bucket": resolved_bucket,
            "artifact_prefix": artifact_prefix,
            "artifact_uri": template_vars.get("artifact_uri"),
            "command": spec.command[2] if len(spec.command) >= 3 else "",
            "config_path": str(config_path),
        }
        return spec, metadata

    def handle(self) -> int:
        try:
            config_path = self._resolve_job_config_path(getattr(self.args, "job_config", None))
            config = self._load_job_config(config_path)
        except Exception as exc:
            if self.json_mode:
                self.output_error(str(exc), code="CLOUD_RUN_CONFIG_ERROR")
                return 1
            print_error(str(exc))
            return 1

        if self.json_mode:
            try:
                spec, metadata = self._compile_hf_job(config_path, config)
            except Exception as exc:
                self.output_error(str(exc), code="CLOUD_RUN_COMPILE_ERROR")
                return 1
            self.output(
                {
                    "provider": spec.provider,
                    "image": spec.image,
                    "flavor": spec.flavor,
                    "timeout_hours": spec.timeout_hours,
                    "labels": spec.labels,
                    "secrets": sorted(spec.secrets.keys()),
                    "metadata": metadata,
                }
            )
            return 0

        print_header("CLOUD RUN", "Submit a config-driven cloud job")

        try:
            huggingface_hub = load_huggingface_hub(require_apis=("run_job", "create_bucket"))
            spec, metadata = self._compile_hf_job(config_path, config)
        except Exception as exc:
            print_error(str(exc))
            return 1

        print_config(
            {
                "Config": metadata["config_path"],
                "Name": metadata["name"],
                "Provider": spec.provider,
                "Image": spec.image,
                "GPU": spec.flavor,
                "Timeout": f"{spec.timeout_hours:.1f}h" if spec.timeout_hours is not None else "-",
                "Secrets": ", ".join(sorted(spec.secrets.keys())) or "-",
                "Artifacts": metadata.get("artifact_uri") or "-",
            },
            "Cloud Run Configuration",
        )

        if not getattr(self.args, "auto_confirm", False) and not confirm("Start cloud run with this configuration?"):
            print_info("Cloud run cancelled.")
            return 0

        try:
            submission = HFJobExecutor(huggingface_hub).submit(spec)
        except Exception as exc:
            print_error(f"Failed to submit cloud run: {exc}")
            return 1

        print_info(f"Cloud job submitted: {submission.job_id}")
        if submission.job_url:
            print_info(f"Monitor at: {submission.job_url}")
        if metadata.get("artifact_uri"):
            print_info(f"Artifacts will sync to: {metadata['artifact_uri']}")
        return 0
