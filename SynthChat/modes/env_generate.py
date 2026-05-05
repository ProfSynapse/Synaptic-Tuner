"""SynthChat environment-generation diagnostic mode.

Location: SynthChat/modes/env_generate.py
Purpose: Run only the configured environment_generation stage for one scenario
         and write the resolved seed bundle for inspection.
Usage: Called by SynthChat.run.main() when command is 'env-generate'.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..config.privacy import resolve_privacy_settings
from ..engine import ImprovementEngine
from ..generator import SynthChatGenerator
from ..result_writer import DebugEventWriter
from ..utils.logger import get_logger


def env_generate_mode(args, *, load_settings, create_llm_client):
    """Run only environment generation for a configured scenario."""
    print("=== SynthChat: Env Generate Mode ===\n")

    config_dir = Path(args.config_dir or "SynthChat/config")
    scenarios_dir = Path(args.scenarios_dir or "SynthChat/scenarios")
    rubrics_dir = Path(args.rubrics_dir or "SynthChat/rubrics")
    output_file = Path(args.output or "SynthChat/output/env_generation_debug.json")
    debug_path = _resolve_debug_artifact_path(args.debug_artifacts, output_file)

    settings = load_settings(config_dir)
    if args.llm_timeout is not None:
        settings.setdefault("llm", {}).setdefault("generation", {})["timeout_seconds"] = args.llm_timeout
        settings.setdefault("llm", {}).setdefault("improvement", {})["timeout_seconds"] = args.llm_timeout
    if args.disable_provider_routing:
        settings.setdefault("llm", {}).setdefault("generation", {}).pop("provider_routing", None)
        settings.setdefault("llm", {}).setdefault("improvement", {}).pop("provider_routing", None)
    settings["privacy_preprocess"] = resolve_privacy_settings(settings, {})
    logger = get_logger("synthchat_env_generate")

    print("Initializing LLM clients...")
    gen_client = create_llm_client(
        settings,
        mode="generation",
        provider_override=args.provider,
        model_override=args.model,
    )
    improve_client = create_llm_client(
        settings,
        mode="improvement",
        provider_override=args.provider,
        model_override=args.model,
    )

    engine = ImprovementEngine(
        llm_client=improve_client,
        rubrics_dir=rubrics_dir,
        config_path=config_dir / "validation.yaml",
        logger=logger,
        enable_interactions=settings.get("logging", {}).get("save_interactions", True),
    )

    with _optional_debug_writer(debug_path, settings) as debug_writer:
        generator = SynthChatGenerator(
            config_dir=config_dir,
            scenarios_dir=scenarios_dir,
            rubrics_dir=rubrics_dir,
            llm_client=gen_client,
            engine=engine,
            environment_validator=None,
            enable_stage_validation=settings.get("generation", {}).get("stage_validation", True),
            logger=logger,
            privacy_settings=settings.get("privacy_preprocess"),
            debug_event_writer=(debug_writer.write_event if debug_writer else None),
        )

        scenario = generator.scenario_loader.get_scenario(args.scenario)
        if scenario is None:
            available = sorted(generator.scenario_loader.list_scenarios())
            raise ValueError(
                f"Unknown scenario '{args.scenario}'. "
                f"Available scenario count: {len(available)}"
            )
        if args.max_retries is not None:
            scenario = dict(scenario)
            generation_cfg = dict(scenario.get("environment_generation") or {})
            generation_cfg["max_retries"] = args.max_retries
            scenario["environment_generation"] = generation_cfg
        if args.max_tokens is not None:
            scenario = dict(scenario)
            generation_cfg = dict(scenario.get("environment_generation") or {})
            generation_cfg["max_tokens"] = args.max_tokens
            scenario["environment_generation"] = generation_cfg

        if debug_writer:
            print(f"Debug artifacts enabled: {debug_path}")
            debug_writer.write_event(
                "env_generate_start",
                {
                    "scenario": args.scenario,
                    "seed_id": args.seed_id,
                    "output": str(output_file),
                    "provider": args.provider,
                    "model": args.model,
                    "llm_timeout": args.llm_timeout,
                    "max_retries": args.max_retries,
                    "max_tokens": args.max_tokens,
                    "disable_provider_routing": args.disable_provider_routing,
                },
            )

        started_at = time.monotonic()
        seed_metadata = {}
        if args.seed_id:
            seed_metadata["seed_id"] = args.seed_id
        if args.seed_index is not None:
            seed_metadata["seed_index"] = args.seed_index
            seed_metadata["seed_number"] = args.seed_index + 1
        if args.seed_count is not None:
            seed_metadata["seed_count"] = args.seed_count

        bundle = generator.prepare_seed_bundle(
            scenario_key=args.scenario,
            seed_id=args.seed_id,
            scenario=scenario,
            randomize_params=not args.deterministic,
            seed_metadata=seed_metadata,
        )
        elapsed_s = round(time.monotonic() - started_at, 3)

        result = {
            "scenario": args.scenario,
            "seed_id": args.seed_id,
            "elapsed_s": elapsed_s,
            "environment_mode": bundle.get("environment_mode"),
            "seed_context": bundle.get("seed_context"),
            "generated_environment": bundle.get("generated_environment"),
            "resolved_environment_config": bundle.get("resolved_environment_config"),
            "resolved_system_context": bundle.get("resolved_system_context"),
            "resolved_task_context": bundle.get("resolved_task_context"),
            "stage_reviews": bundle.get("stage_reviews"),
            "stage_failures": bundle.get("stage_failures"),
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        if debug_writer:
            debug_writer.write_event(
                "env_generate_done",
                {
                    "scenario": args.scenario,
                    "seed_id": args.seed_id,
                    "output": str(output_file),
                    "elapsed_s": elapsed_s,
                    "environment_mode": result.get("environment_mode"),
                    "generated_keys": sorted((result.get("generated_environment") or {}).keys()),
                    "stage_failures": result.get("stage_failures"),
                },
            )

    print(f"\nEnvironment generation complete in {elapsed_s}s")
    print(f"Output: {output_file}")
    if debug_path:
        print(f"Debug: {debug_path}")


class _NullDebugWriter:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _optional_debug_writer(debug_path: Optional[Path], settings: Dict[str, Any]):
    if debug_path is None:
        return _NullDebugWriter()
    return DebugEventWriter(debug_path, settings)


def _resolve_debug_artifact_path(raw_value: Optional[str], output_file: Path) -> Optional[Path]:
    if raw_value is None:
        return None
    if raw_value == "auto" or not str(raw_value).strip():
        return output_file.with_name(f"{output_file.stem}.debug_events.jsonl")
    return Path(raw_value)
