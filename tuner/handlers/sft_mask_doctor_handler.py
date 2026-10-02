"""
Handler for ``python tuner.py doctor sft-mask``.

Location: tuner/handlers/sft_mask_doctor_handler.py
Purpose: Resolve the SFT trainer's preprocessing settings the way
    ``Trainers/sft/train_sft.py`` does (trainer config, then explicit flags),
    load only the tokenizer, and run the SFT mask doctor
    (``Trainers/sft/src/mask_doctor.py``) over a dataset sample.

Exit codes: 0 = no hard failures, 1 = hard failures found, 2 = setup error
(bad flags/config, tokenizer or dataset could not be loaded).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from tuner.handlers.base import BaseHandler
from tuner.project import ProjectContext

PROMPT_RENDERS = ("full_conversation", "prompt_completion")
DEFAULT_SFT_CONFIG = "Trainers/sft/configs/config.yaml"

EXIT_OK = 0
EXIT_HARD_FAILURE = 1
EXIT_SETUP_ERROR = 2


class MaskDoctorSetupError(Exception):
    """Raised for invalid flags/config before any row is analyzed."""


def import_sft_modules(engine_root: Path) -> tuple[ModuleType, ModuleType]:
    """Import the SFT trainer's config loader and the mask doctor without the GPU stack.

    The trainer modules use bare imports (``from preprocessing import ...``), so
    ``Trainers/sft/src`` goes on ``sys.path`` exactly as ``train_sft.py`` does.
    The config loader is loaded by file path under a unique name because several
    trainers ship a module called ``config_loader``.
    """
    sft_root = engine_root / "Trainers" / "sft"
    for path in (str(engine_root), str(sft_root / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)

    module_name = "sft_trainer_config_loader"
    config_loader = sys.modules.get(module_name)
    if config_loader is None:
        spec = importlib.util.spec_from_file_location(
            module_name, sft_root / "configs" / "config_loader.py"
        )
        config_loader = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = config_loader
        spec.loader.exec_module(config_loader)

    import mask_doctor  # noqa: E402  (bare import, see docstring)

    return config_loader, mask_doctor


def load_trainer_config(config_loader: ModuleType, path: Optional[Path]) -> Any:
    """Load the SFT trainer config the way ``train_sft.py`` does.

    No path -> the trainer's default YAML (``Trainers/sft/configs/config.yaml``).
    A ``.py`` path -> the module's ``Config()`` (``train_sft.py --config``).
    Any other path -> ``config_loader.load_config(path)`` (trainer YAML schema).
    """
    if path is None:
        return config_loader.load_config()
    if not path.is_file():
        raise MaskDoctorSetupError(f"--sft-config not found: {path}")
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("custom_sft_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Config()
    return config_loader.load_config(str(path))


def _parse_json_object(raw: str, flag: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MaskDoctorSetupError(f"{flag} must be a JSON object string: {exc}") from exc
    if not isinstance(value, dict):
        raise MaskDoctorSetupError(f"{flag} must decode to a JSON object (got {type(value).__name__}).")
    return value


def _parse_preview_rows(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    try:
        return [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise MaskDoctorSetupError(f"--preview-rows must be comma-separated integers: {raw!r}") from exc


def resolve_settings(
    args: Namespace,
    *,
    config_loader: ModuleType,
    mask_doctor: ModuleType,
    doctor_config: Any,
    cwd: Path,
    sft_root: Path,
):
    """Resolve preprocessing settings with the trainer's precedence: config < flags."""

    def _path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else cwd / path

    sft_config = getattr(args, "sft_config", None)
    config = load_trainer_config(config_loader, _path(sft_config) if sft_config else None)
    sources = [f"trainer config {sft_config or DEFAULT_SFT_CONFIG}"]

    model_name = config.model.model_name
    max_seq_length = config.training.max_seq_length
    chat_template_kwargs = config.training.chat_template_kwargs
    prompt_render = config.training.prompt_render
    local_file = config.dataset.local_file
    if local_file and not Path(local_file).is_absolute():
        # train_sft.py runs from Trainers/sft, so config-relative files resolve there.
        local_file = str(sft_root / local_file)

    flag_sources = []
    if getattr(args, "model", None):
        model_name = args.model
        flag_sources.append("--model")
    if getattr(args, "max_seq_length", None) is not None:
        max_seq_length = args.max_seq_length
        flag_sources.append("--max-seq-length")
    if getattr(args, "chat_template_kwargs", None) is not None:
        chat_template_kwargs = _parse_json_object(args.chat_template_kwargs, "--chat-template-kwargs")
        flag_sources.append("--chat-template-kwargs")
    if getattr(args, "prompt_render", None):
        prompt_render = args.prompt_render
        flag_sources.append("--prompt-render")
    if getattr(args, "dataset_path", None):
        local_file = str(_path(args.dataset_path))
        flag_sources.append("--dataset-path")
    if flag_sources:
        sources.append("flags " + " ".join(flag_sources))

    if prompt_render not in PROMPT_RENDERS:
        raise MaskDoctorSetupError(f"prompt_render must be one of {PROMPT_RENDERS}, got {prompt_render!r}.")
    if not local_file and not config.dataset.dataset_name:
        raise MaskDoctorSetupError("No dataset resolved; pass --dataset-path or an --sft-config with a dataset.")
    if local_file and not Path(local_file).is_file():
        raise MaskDoctorSetupError(f"Dataset file not found: {local_file}")

    # Same derivations as train_sft.py before load_and_prepare_tokenized_dataset.
    completion_only = bool(config.training.completion_only_loss) and not getattr(args, "no_completion_only", False)
    aux_head = getattr(config, "aux_head", None)
    aux_enabled = bool(aux_head is not None and aux_head.enabled)
    sample_size = getattr(args, "sample_size", None)
    seed = getattr(args, "seed", None)
    preview_count = getattr(args, "preview_count", None)
    return mask_doctor.MaskDoctorSettings(
        model_name=model_name,
        max_seq_length=int(max_seq_length),
        loss_mask_mode="assistant_only" if completion_only else "full_sequence",
        prompt_render=prompt_render,
        chat_template_kwargs=chat_template_kwargs,
        local_file=local_file,
        dataset_name=config.dataset.dataset_name,
        dataset_file=config.dataset.dataset_file,
        filter_desirable=bool(config.dataset.filter_desirable),
        assistant_only_loss_requested=bool(config.training.assistant_only_loss),
        aux_token_position=aux_head.token_position if aux_enabled else None,
        use_preassigned_splits=bool(getattr(config.dataset, "use_preassigned_splits", False)),
        sample_size=int(sample_size if sample_size is not None else doctor_config.sample_size),
        seed=int(seed if seed is not None else doctor_config.seed),
        preview_rows=_parse_preview_rows(getattr(args, "preview_rows", None)),
        preview_count=int(preview_count if preview_count is not None else doctor_config.preview_count),
        sources=sources,
    )


def load_tokenizer(model_name: str, *, revision: Optional[str], trust_remote_code: bool) -> Any:
    """Load only the tokenizer (or processor, for multimodal models) with a chat template."""
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    except Exception:  # noqa: BLE001 - fall through to the explicit error below
        processor = None
    if processor is not None and getattr(processor, "chat_template", None):
        return processor
    raise MaskDoctorSetupError(
        f"{model_name} ships no chat template. The SFT trainer would apply an Unsloth "
        "fallback template, which the doctor cannot reproduce without Unsloth; "
        "use a tokenizer that carries its chat template."
    )


class SFTMaskDoctorHandler(BaseHandler):
    """Diagnose SFT loss masking for a dataset with only the tokenizer loaded."""

    def __init__(
        self,
        args: Optional[Namespace] = None,
        context: Optional[ProjectContext] = None,
        tokenizer: Any = None,
    ):
        super().__init__(args=args, context=context)
        # Injected tokenizer: lets tests and callers that already hold one skip
        # the transformers load.
        self._tokenizer = tokenizer

    @property
    def name(self) -> str:
        return "doctor sft-mask"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        args = self.args or Namespace()
        if getattr(args, "doctor_fix", False):
            self.output_error("--fix is not supported by doctor sft-mask.", code="INVALID_ARGS")
            return EXIT_SETUP_ERROR

        # Trainer-path modules print progress; keep stdout pure JSON in --json mode.
        log_stream = sys.stderr if self.json_mode else sys.stdout
        try:
            with contextlib.redirect_stdout(log_stream):
                config_loader, mask_doctor = import_sft_modules(self.context.engine_root)
                doctor_config = mask_doctor.load_doctor_config()
                settings = resolve_settings(
                    args,
                    config_loader=config_loader,
                    mask_doctor=mask_doctor,
                    doctor_config=doctor_config,
                    cwd=self.context.invocation_cwd,
                    sft_root=self.context.engine_root / "Trainers" / "sft",
                )
                tokenizer = self._tokenizer or load_tokenizer(
                    settings.model_name,
                    revision=getattr(args, "tokenizer_revision", None),
                    trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
                )
                report = mask_doctor.run_mask_doctor(
                    tokenizer=tokenizer,
                    settings=settings,
                    doctor_config=doctor_config,
                )
        except MaskDoctorSetupError as exc:
            self.output_error(str(exc), code="MASK_DOCTOR_SETUP")
            return EXIT_SETUP_ERROR
        except (ImportError, OSError, ValueError) as exc:
            self.output_error(f"{type(exc).__name__}: {exc}", code="MASK_DOCTOR_SETUP")
            return EXIT_SETUP_ERROR

        exit_code = EXIT_OK if report["success"] else EXIT_HARD_FAILURE
        if self.json_mode:
            print(json.dumps({**report, "exit_code": exit_code}, indent=2, default=str))
        else:
            print(mask_doctor.format_report(report))
        return exit_code
