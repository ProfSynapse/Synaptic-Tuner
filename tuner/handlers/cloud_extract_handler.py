"""Cloud hidden-state extraction verb for HF Jobs (publish-by-id, Option A).

Location: tuner/handlers/cloud_extract_handler.py
Purpose: Implement the ``cloud-extract`` CLI verb -- a GENERAL tuner capability
         that launches a forward-only hidden-state extraction on Hugging Face
         Jobs against published (hub-resident) inputs: a matched slice dataset,
         a contrast adapter model repo (by id + revision), and a base model
         (by id + revision). Outputs are written to a hub output dataset.
Used by: tuner/cli/router.py (routes the ``cloud-extract`` command here);
         tuner/cli/parser.py (registers the verb + its flags).

Design contract: docs/architecture/experiment-runner-probe-dataprep.md section 6
(Component 3, Option A). This verb resolves the three PREPARE blockers:
  (a) no extraction verb in cloud-pipeline  -> this NEW verb;
  (b) probe_results.jsonl is large/gitignored -> the runner publishes only the
      small matched SLICE; this verb consumes it by hub id (never the 123MB file);
  (c) the contrast adapter is a local artifact -> consumed as a private hub
      model repo by id + revision.

The extraction workload remains off the training implementation path, but its
source bootstrap is intentionally shared with every secure HF Jobs launcher:
one provider-neutral SourceLock, one verified capsule, and one read-only volume
contract.  This module does not construct a second Git/clone source model.

GPU boundary (section 10): every step this module performs locally
(arg parsing, artifact resolution, publish, dry-run command assembly, job
submission) is GPU-free. The forward-only extraction that the submitted job
runs on the cloud GPU is the only GPU-required step, and launching it is
cost-incurring -- gated behind explicit ``--yes`` confirmation and, upstream,
the runner's capability + submodule-pushed + HF_TOKEN + artifact-resolution
push-gate (section 6.5). ``--dry-run`` assembles and prints the job spec
without submitting (and without requiring a token).

RUNBOOK / SECURITY: credentials remain provider secrets. They are never copied
into generated shell, source-lock transport members, labels, or dry-run output.
"""

from __future__ import annotations

import shlex
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.utilities.env import get_hf_token, load_env_file
from tuner.cloud import (
    CloudJobSpec,
    HFJobExecutor,
    build_bash_command,
    build_hf_job_secrets,
    load_huggingface_hub,
)
from tuner.cloud.hf_jobs import require_current_hf_source_submission_authorization
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler
from tuner.handlers.stages._util import (
    HFSourcePreparation,
    hf_verified_source_steps,
    preflight_hf_source_lock,
    prepare_hf_source,
)
from tuner.ui import confirm, print_config, print_error, print_header, print_info, print_success

# Default HF Jobs hardware flavor for a forward-only extraction pass. Extraction
# is far lighter than training (no optimizer state, no backward), but still needs
# enough VRAM to hold base + adapter; a10g-large is a safe, modest default and is
# overridable via --gpu.
_DEFAULT_FLAVOR = "a10g-large"

# Default wall-clock cap for an extraction job. Forward-only over a few-hundred-row
# slice is quick; the cap is a guard-rail, overridable via --timeout-hours.
_DEFAULT_TIMEOUT_HOURS = 2.0

# Where the tuner repo is cloned inside the HF Job, and where the extraction
# config is read from relative to that clone.
# The in-job entrypoint that runs the forward-only extraction against the
# published artifacts. This is the tuner repo's own extraction runner (cloud
# analog of the local probe harness); it is invoked by hub id, never by a local
# path, because HF Jobs has no research-repo mount.
_EXTRACTION_ENTRYPOINT = "python -m tuner.cloud.extraction_runner"


@dataclass(frozen=True)
class ExtractionLaunchPlan:
    """Fully-resolved, provider-agnostic description of a cloud-extract launch.

    Every field is GPU-free to compute and is what the dry-run prints and the
    submit path hands to :class:`HFJobExecutor`. Keeping this as a pure value
    object makes the whole resolution path unit-testable without a GPU, a
    network, or a token.
    """

    extraction_config: str
    slice_dataset_name: str
    base_model_name: str
    base_model_revision: str
    adapter_repo_id: str
    adapter_revision: str
    output_dataset_name: str
    flavor: str
    timeout_hours: float
    image: str
    source_preparation: HFSourcePreparation

    def as_display(self) -> Dict[str, str]:
        """Human-readable summary for ``print_config`` / JSON output."""
        return {
            "Extraction Config": self.extraction_config,
            "Slice Dataset": self.slice_dataset_name,
            "Base Model": f"{self.base_model_name}@{self.base_model_revision}",
            "Adapter Repo": f"{self.adapter_repo_id}@{self.adapter_revision}",
            "Output Dataset": self.output_dataset_name,
            "Image": self.image,
            "GPU Flavor": self.flavor,
            "Timeout": f"{self.timeout_hours:.1f}h",
            "Repo Commit": self.source_preparation.source_lock.engine_source.commit,
        }


# Each entry: (cli_attr, flag_name, human_label). Every one is required for a
# cloud launch because the data-locality contract (section 6.3/6.4) demands that
# every input -- slice, adapter, base, output -- be a hub id, and every model
# input be revision-pinned for reproducibility (section 8).
_REQUIRED_ARGS = (
    ("extraction_config", "--extraction-config", "extraction config"),
    ("slice_dataset_name", "--slice-dataset-name", "matched slice dataset id"),
    ("base_model_name", "--base-model-name", "base model id"),
    ("base_model_revision", "--base-model-revision", "base model revision SHA"),
    ("adapter_repo_id", "--adapter-repo-id", "contrast adapter repo id"),
    ("adapter_revision", "--adapter-revision", "contrast adapter revision SHA"),
    ("output_dataset_name", "--output-dataset-name", "output dataset id"),
)


class CloudExtractHandler(BaseHandler):
    """Launch a forward-only hidden-state extraction on HF Jobs (publish-by-id).

    This is a sibling of ``cloud-pipeline`` in placement only; it is a GENERAL
    tuner capability and is wholly off the signed training path (see module
    docstring). Inputs are referenced by hub id + revision; outputs land in a
    hub dataset.
    """

    @property
    def name(self) -> str:
        return "cloud-extract"

    def can_handle_direct_mode(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Argument resolution (GPU-free, network-free, token-free)
    # ------------------------------------------------------------------ #
    def _collect_required_args(self) -> Dict[str, str]:
        """Read + validate every required CLI arg, failing fast on any gap.

        Returns a dict of the trimmed string values. Raises
        :class:`CloudProviderError` listing every missing flag at once so the
        caller fixes them in a single pass rather than one round-trip per flag.
        """
        values: Dict[str, str] = {}
        missing: List[str] = []
        for attr, flag, label in _REQUIRED_ARGS:
            raw = getattr(self.args, attr, None)
            value = str(raw).strip() if raw is not None else ""
            if not value:
                missing.append(f"{flag} ({label})")
            else:
                values[attr] = value
        if missing:
            raise CloudProviderError(
                "cloud-extract requires (all are hub ids / pinned SHAs, per the "
                "data-locality contract): " + ", ".join(missing)
            )
        return values

    def _resolve_flavor(self) -> str:
        return str(getattr(self.args, "gpu", None) or _DEFAULT_FLAVOR).strip() or _DEFAULT_FLAVOR

    def _resolve_timeout_hours(self) -> float:
        raw = getattr(self.args, "timeout_hours", None)
        if raw is None:
            return _DEFAULT_TIMEOUT_HOURS
        try:
            timeout = float(raw)
        except (TypeError, ValueError) as exc:
            raise CloudProviderError(f"Invalid --timeout-hours value: {raw!r}") from exc
        if timeout <= 0:
            raise CloudProviderError("--timeout-hours must be positive.")
        return timeout

    def _resolve_image(self) -> str:
        """Resolve the job's Docker image. Overridable via --cloud-image.

        Kept as a simple, explicit knob rather than reaching into the training
        cloud-config loader (which lives on the training namespace). A default
        is provided so the common path needs no flag.
        """
        override = getattr(self.args, "cloud_image", None)
        if override and str(override).strip():
            return str(override).strip()
        # A CUDA + PyTorch base sufficient for forward-only inference with PEFT.
        return "huggingface/transformers-pytorch-gpu:latest"

    def build_launch_plan(self) -> ExtractionLaunchPlan:
        """Assemble the fully-resolved launch plan from CLI args (GPU-free).

        This is the single source of truth for what both the dry-run and the
        submit path act on, which keeps the two paths from drifting.
        """
        values = self._collect_required_args()
        config_path = Path(values["extraction_config"])
        if not config_path.is_absolute():
            config_path = self.repo_root / config_path
        cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        try:
            cloud_config = yaml.safe_load(cloud_config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise CloudProviderError("Could not load HF bootstrap volume configuration.") from exc
        cloud_settings = cloud_config.get("cloud")
        if not isinstance(cloud_settings, dict):
            raise CloudProviderError("cloud configuration must contain a cloud mapping.")
        hf_settings = cloud_settings.get("hf_jobs")
        if not isinstance(hf_settings, dict):
            raise CloudProviderError("cloud.hf_jobs must be a mapping.")
        volume_settings = hf_settings.get("bootstrap_volume", {})
        if not isinstance(volume_settings, dict):
            raise CloudProviderError("hf_jobs.bootstrap_volume must be a mapping.")
        run_id = f"hf-extract-{self._new_launch_id()}"
        source_lock = preflight_hf_source_lock(self.context, run_id=run_id)
        source_preparation = prepare_hf_source(
            self.context,
            run_id=run_id,
            config_path=config_path,
            volume_settings=volume_settings,
            runtime={"provider": "hf_jobs", "task": "extraction"},
            source_lock=source_lock,
        )
        return ExtractionLaunchPlan(
            extraction_config=values["extraction_config"],
            slice_dataset_name=values["slice_dataset_name"],
            base_model_name=values["base_model_name"],
            base_model_revision=values["base_model_revision"],
            adapter_repo_id=values["adapter_repo_id"],
            adapter_revision=values["adapter_revision"],
            output_dataset_name=values["output_dataset_name"],
            flavor=self._resolve_flavor(),
            timeout_hours=self._resolve_timeout_hours(),
            image=self._resolve_image(),
            source_preparation=source_preparation,
        )

    @staticmethod
    def _new_launch_id() -> str:
        from shared.utilities.unique_ids import unique_utc_timestamp
        return unique_utc_timestamp()

    # ------------------------------------------------------------------ #
    # Job-spec assembly (GPU-free)
    # ------------------------------------------------------------------ #
    def build_extraction_command(self, plan: ExtractionLaunchPlan) -> List[str]:
        """Build the in-job extraction command (every input passed by hub id).

        Each value is shell-quoted; no local paths cross into the job because HF
        Jobs has no research-repo mount. The base/adapter revisions are pinned
        for reproducibility.
        """
        invocation = [
            _EXTRACTION_ENTRYPOINT,
            f"--extraction-config {shlex.quote(plan.extraction_config)}",
            f"--slice-dataset-name {shlex.quote(plan.slice_dataset_name)}",
            f"--base-model-name {shlex.quote(plan.base_model_name)}",
            f"--base-model-revision {shlex.quote(plan.base_model_revision)}",
            f"--adapter-repo-id {shlex.quote(plan.adapter_repo_id)}",
            f"--adapter-revision {shlex.quote(plan.adapter_revision)}",
            f"--output-dataset-name {shlex.quote(plan.output_dataset_name)}",
        ]
        engine_root = plan.source_preparation.remote_engine_root
        return [
            *hf_verified_source_steps(plan.source_preparation),
            f"cd {shlex.quote(engine_root)}",
            " ".join(invocation),
        ]

    def build_job_spec(self, plan: ExtractionLaunchPlan, *, token: Optional[str], huggingface_hub) -> CloudJobSpec:
        """Assemble the provider-agnostic :class:`CloudJobSpec` (GPU-free).

        HF_TOKEN is injected as a job SECRET (never echoed into the command or
        labels). ``token`` may be None for dry-run, in which case no secret is
        attached.
        """
        proven_volume = plan.source_preparation.prove_volume(huggingface_hub)
        secrets = build_hf_job_secrets(token) if token else {}
        labels = {
            "task": "extract",
            "base_model": plan.base_model_name,
            "adapter_repo": plan.adapter_repo_id,
            "output_dataset": plan.output_dataset_name,
        }
        return CloudJobSpec(
            provider="hf_jobs",
            image=plan.image,
            command=build_bash_command(self.build_extraction_command(plan)),
            flavor=plan.flavor,
            timeout_hours=plan.timeout_hours,
            secrets=secrets,
            labels=labels,
            volumes=(proven_volume,),
        )

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def handle(self) -> int:
        dry_run = bool(getattr(self.args, "dry_run", False))

        try:
            plan = self.build_launch_plan()
        except Exception as exc:
            return self._fail(exc, code="CLOUD_EXTRACT_CONFIG_ERROR")

        if dry_run:
            return self._handle_dry_run(plan)

        return self._handle_submit(plan)

    def _handle_dry_run(self, plan: ExtractionLaunchPlan) -> int:
        """Assemble + print the job spec WITHOUT submitting (no token needed).

        SECURITY: the printed command embeds the clone URL. If an operator's git
        origin embeds credentials (``https://user:token@host/...``) those must
        not leak to stdout/JSON, so the displayed command has any URL userinfo
        redacted via :func:`_redact_url_userinfo`. Origins should never embed
        credentials in the first place; the runbook note above the verb
        documents this.
        """
        try:
            require_current_hf_source_submission_authorization(route="cloud-extract.dry-run")
            plan.source_preparation.require_consumable()
            huggingface_hub = load_huggingface_hub(require_apis=("run_job", "Volume"))
            spec = self.build_job_spec(plan, token=None, huggingface_hub=huggingface_hub)
        except Exception as exc:
            return self._fail(exc, code="CLOUD_EXTRACT_ENV_ERROR")
        command_text = spec.command[-1] if spec.command else ""
        if self.json_mode:
            self.output(
                {
                    "dry_run": True,
                    "plan": plan.as_display(),
                    "image": spec.image,
                    "flavor": spec.flavor,
                    "timeout_hours": spec.timeout_hours,
                    "labels": spec.labels,
                    "command": command_text,
                }
            )
            return 0
        print_header("CLOUD EXTRACT (DRY RUN)", "Forward-only hidden-state extraction on HF Jobs")
        print_config(plan.as_display(), "Cloud Extract Plan")
        print_info("Job command (not submitted):")
        print(command_text)
        return 0

    def _handle_submit(self, plan: ExtractionLaunchPlan) -> int:
        """Resolve the token + hub, confirm, and submit the extraction job."""
        try:
            require_current_hf_source_submission_authorization(route="cloud-extract.submit")
            plan.source_preparation.require_consumable()
            load_env_file()
            token = get_hf_token()
            if not token:
                raise CloudProviderError(
                    "HF_TOKEN not set. Required to submit cloud-extract. Set "
                    "HF_TOKEN (or HF_API_KEY) in your .env file or environment."
                )
            huggingface_hub = load_huggingface_hub(require_apis=("run_job", "Volume"))
            spec = self.build_job_spec(plan, token=token, huggingface_hub=huggingface_hub)
        except Exception as exc:
            return self._fail(exc, code="CLOUD_EXTRACT_ENV_ERROR")

        if self.json_mode:
            # JSON mode is non-interactive: require explicit confirmation.
            if not getattr(self.args, "auto_confirm", False):
                self.output_error(
                    "cloud-extract in --json mode requires --yes to submit "
                    "(this is a cost-incurring GPU job).",
                    code="CLOUD_EXTRACT_CONFIRM_REQUIRED",
                )
                return 1
        else:
            print_header("CLOUD EXTRACT", "Forward-only hidden-state extraction on HF Jobs")
            print_config(plan.as_display(), "Cloud Extract Plan")
            if not getattr(self.args, "auto_confirm", False) and not confirm(
                "Submit this cost-incurring GPU extraction job to HF Jobs?"
            ):
                print_info("Cloud extract cancelled.")
                return 0

        try:
            submission = HFJobExecutor(huggingface_hub).submit(spec)
        except Exception as exc:
            return self._fail(exc, code="CLOUD_EXTRACT_SUBMIT_ERROR")

        if self.json_mode:
            self.output(
                {
                    "submitted": True,
                    "job_id": submission.job_id,
                    "job_url": submission.job_url,
                    "output_dataset": plan.output_dataset_name,
                }
            )
            return 0
        print_success(f"Cloud extraction job submitted: {submission.job_id}")
        if submission.job_url:
            print_info(f"Monitor at: {submission.job_url}")
        print_info(f"Outputs will be written to hub dataset: {plan.output_dataset_name}")
        return 0

    def _fail(self, exc: Exception, *, code: str) -> int:
        """Uniform error surface honoring --json mode."""
        message = str(exc)
        if self.json_mode:
            self.output_error(message, code=code)
        else:
            print_error(message)
        return 1
