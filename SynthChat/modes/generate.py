"""SynthChat Generate Mode - Create new datasets from scenarios.

Location: SynthChat/modes/generate.py
Purpose: Implements the 'generate' CLI command. Loads settings and scenarios,
         creates LLM clients and generator, dispatches sequential or parallel
         generation, streams results to disk, and prints a summary.
Usage: Called by SynthChat.run.main() when command is 'generate'.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.privacy import resolve_privacy_settings
from ..utils.logger import get_logger
from ..engine import ImprovementEngine
from ..generator import SynthChatGenerator
from ..targets import _extract_shared_seed_spec, _normalize_target_spec
from ..result_writer import StreamingResultWriter, generate_output_path, print_summary
from ..parallel.workers import (
    run_parallel_generation,
    serialize_environment_options,
)


def generate_mode(args, *, load_settings, create_llm_client, create_environment_validator):
    """Generate new dataset from scenarios.

    Args:
        args: Parsed CLI arguments.
        load_settings: Callable to load settings.yaml.
        create_llm_client: Callable to create LLM clients.
        create_environment_validator: Callable to create environment validator.

    Flow:
        1. Load settings and scenarios
        2. Create LLM clients (generation + improvement)
        3. Create generator with improvement engine
        4. Generate examples
        5. Save results
    """
    print("=== SynthChat: Generate Mode ===\n")

    # Load configuration
    config_dir = Path(args.config_dir or "SynthChat/config")
    settings = load_settings(config_dir)
    privacy_overrides: Dict[str, Any] = {}
    if getattr(args, "privacy_profile", None):
        privacy_overrides = {"enabled": True, "profile": args.privacy_profile}
    settings["privacy_preprocess"] = resolve_privacy_settings(settings, privacy_overrides)
    logger = get_logger("synthchat_generate")

    scenarios_dir = Path(args.scenarios_dir or "SynthChat/scenarios")
    rubrics_dir = Path(args.rubrics_dir or "SynthChat/rubrics")

    # Create LLM clients (CLI args override settings.yaml)
    print("Initializing LLM clients...")
    gen_client = create_llm_client(settings, mode="generation",
                                   provider_override=args.provider, model_override=args.model)
    improve_client = create_llm_client(settings, mode="improvement",
                                       provider_override=args.provider, model_override=args.model)
    environment_validator = create_environment_validator(settings, args)
    if environment_validator is not None:
        print(f"Environment validation enabled (backend={environment_validator.backend})")

    # Create improvement engine
    validation_config = config_dir / "validation.yaml"
    engine = ImprovementEngine(
        llm_client=improve_client,
        rubrics_dir=rubrics_dir,
        config_path=validation_config,
        logger=logger,
        enable_interactions=settings["logging"]["save_interactions"]
    )

    # Create generator
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=gen_client,
        engine=engine,
        environment_validator=environment_validator,
        enable_stage_validation=settings["generation"]["stage_validation"],
        logger=logger,
        privacy_settings=settings.get("privacy_preprocess"),
    )
    prompt_opt_metadata = _prepare_prompt_optimization(
        args=args,
        generator=generator,
        scenarios_dir=scenarios_dir,
        config_dir=config_dir,
    )

    # Load targets
    if args.targets_file:
        with open(args.targets_file) as f:
            targets = json.load(f)
    else:
        # Use default targets from settings
        targets = settings["defaults"]["targets"]

    shared_seed_spec, targets = _extract_shared_seed_spec(targets)

    # Filter targets if specific scenarios requested
    if args.scenarios:
        targets = {k: v for k, v in targets.items() if k in args.scenarios}
        if shared_seed_spec and shared_seed_spec.get("targets"):
            shared_seed_spec = {
                **shared_seed_spec,
                "targets": [key for key in shared_seed_spec["targets"] if key in targets],
            }

    print(f"\nGeneration targets:")
    total_examples = 0
    if shared_seed_spec:
        shared_targets = set(shared_seed_spec.get("targets") or targets.keys())
        print(
            f"  [shared seed] scenario={shared_seed_spec['scenario']} "
            f"seed_count={shared_seed_spec['seed_count']} targets={len(shared_targets)}"
        )
    for scenario_key, raw_target in targets.items():
        target_spec = _normalize_target_spec(raw_target)
        if shared_seed_spec and (not shared_seed_spec.get("targets") or scenario_key in set(shared_seed_spec.get("targets") or [])):
            scenario_total = shared_seed_spec["seed_count"] * target_spec["rollouts_per_seed"]
        else:
            scenario_total = target_spec["seed_count"] * target_spec["rollouts_per_seed"]
        total_examples += scenario_total
        if target_spec["rollouts_per_seed"] == 1:
            print(f"  {scenario_key}: {scenario_total}")
        else:
            print(
                f"  {scenario_key}: {target_spec['seed_count']} seed(s) x "
                f"{target_spec['rollouts_per_seed']} rollout(s) = {scenario_total}"
            )
    print(f"Total: {total_examples} examples\n")

    # Load docs if provided
    docs: List = []
    if args.docs:
        from ..utils.docs_loader import DocsLoader
        print(f"Loading docs from: {args.docs}")
        docs = DocsLoader().load(args.docs)
        print(f"Loaded {len(docs)} document(s)")
        print(f"Will generate {args.per_doc} example(s) per doc\n")
        if settings.get("privacy_preprocess", {}).get("enabled") and settings.get("privacy_preprocess", {}).get("profile"):
            print(f"Privacy seed preprocessing enabled (profile={settings['privacy_preprocess']['profile']})")

    # Generate
    max_iterations = args.max_iterations or settings["improvement"]["max_iterations"]
    args.workers = max(1, args.workers)
    num_workers = args.workers
    if prompt_opt_metadata and num_workers > 1:
        print("Prompt optimization overlays are applied in memory; using 1 worker for this generation run.")
        num_workers = 1
        args.workers = 1
    results = []

    # Determine output path early so we can stream results to disk
    output_file = Path(args.output) if args.output else generate_output_path(settings)

    with StreamingResultWriter(output_file, settings) as writer:
        result_writer = _PromptOptimizationResultWriter(writer.write, prompt_opt_metadata)
        if docs and num_workers > 1:
            # Parallel docs-based generation with multiple workers
            print(f"Using {num_workers} parallel workers for {len(docs)} doc(s)\n")
            work_items = _build_docs_work_items(
                docs, args, generator, targets, shared_seed_spec,
                environment_validator, create_llm_client, settings,
                max_iterations,
            )
            results.extend(run_parallel_generation(work_items, num_workers, result_writer))
        elif docs:
            # Sequential docs-based generation (single worker)
            total_docs = len(docs)
            for doc_idx, doc in enumerate(docs, 1):
                print(f"\n--- Document {doc_idx}/{total_docs}: {doc.path} ---")
                for rep in range(args.per_doc):
                    if args.per_doc > 1:
                        print(f"  Repetition {rep + 1}/{args.per_doc}")
                    batch_results = generator.generate_batch(
                        targets=targets,
                        max_iterations=max_iterations,
                        randomize_params=True,
                        doc_context=doc,
                        on_result=result_writer.write,
                        shared_seed_spec=shared_seed_spec,
                    )
                    results.extend(batch_results)
        elif num_workers > 1:
            # Parallel generation with multiple workers (no docs)
            print(f"Using {num_workers} parallel workers\n")
            work_items = _build_nodocs_work_items(
                generator, targets, shared_seed_spec,
                args, settings, environment_validator, create_llm_client,
                max_iterations,
            )
            results.extend(run_parallel_generation(work_items, num_workers, result_writer))
        else:
            # Standard sequential generation (no docs)
            results = generator.generate_batch(
                targets=targets,
                max_iterations=max_iterations,
                randomize_params=True,
                on_result=result_writer.write,
                shared_seed_spec=shared_seed_spec,
            )

        print(f"\nStreamed {writer.count} examples to {output_file}")

    # Print summary
    print_summary(results, output_file)


def _prepare_prompt_optimization(args, *, generator, scenarios_dir: Path, config_dir: Path) -> Optional[Dict[str, Any]]:
    """Run or load prompt optimization artifacts and apply overlays in memory."""
    config_path = getattr(args, "prompt_opt_config", None)
    artifact_path = getattr(args, "prompt_opt_artifact", None)
    if not config_path and not artifact_path:
        return None

    result = None
    if config_path:
        try:
            from shared.prompt_optimization import PromptOptimizationService
        except ImportError as exc:
            raise RuntimeError(
                "Prompt optimization requested, but shared.prompt_optimization is not importable."
            ) from exc
        result = PromptOptimizationService.from_config(config_path, overrides=None).run()
        artifact_path = result.output_dir
        print(f"Prompt optimization artifact: {artifact_path}")

    overlay_path = _resolve_prompt_overlay_path(Path(artifact_path))
    overlays = _load_prompt_overlays(overlay_path)
    applied_subjects = _apply_prompt_overlays(
        overlays=overlays,
        generator=generator,
        scenarios_dir=scenarios_dir,
        config_dir=config_dir,
    )
    if overlays.get("subjects") and not applied_subjects:
        raise ValueError(
            "Prompt optimization was requested, but none of the overlay subjects "
            "matched the loaded SynthChat scenario/config surfaces."
        )
    selected_candidate_id = overlays.get("selected_candidate_id")
    metadata = {
        "artifact_path": str(overlay_path.parent),
        "overlays_path": str(overlay_path),
        "selected_candidate_id": selected_candidate_id,
        "applied_subjects": applied_subjects,
    }
    if result is not None:
        metadata["run_id"] = result.run_id
        metadata["candidate_count"] = result.candidate_count

    print(
        "Applied prompt optimization overlays: "
        f"{len(applied_subjects)} subject(s)"
        + (f", selected_candidate_id={selected_candidate_id}" if selected_candidate_id else "")
    )
    return metadata


def _resolve_prompt_overlay_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "overlays.json"
    if not resolved.exists():
        raise FileNotFoundError(f"Prompt optimization overlays not found: {resolved}")
    if resolved.name != "overlays.json":
        raise ValueError(
            "Prompt optimization artifact must be a directory containing overlays.json "
            "or a direct path to overlays.json."
        )
    return resolved


def _load_prompt_overlays(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt optimization overlays must be a JSON object: {path}")
    if not isinstance(payload.get("subjects"), dict):
        raise ValueError(f"Prompt optimization overlays missing subjects mapping: {path}")
    return payload


def _apply_prompt_overlays(
    *,
    overlays: Dict[str, Any],
    generator,
    scenarios_dir: Path,
    config_dir: Path,
) -> List[str]:
    applied: List[str] = []
    for subject_id, overlay in (overlays.get("subjects") or {}).items():
        if not isinstance(overlay, dict):
            raise ValueError(f"Prompt optimization overlay subject must be an object: {subject_id}")
        dotted_path = str(overlay.get("dotted_path") or "").strip()
        if not dotted_path:
            raise ValueError(f"Prompt optimization overlay missing dotted_path: {subject_id}")
        if "optimized_prompt" not in overlay:
            raise ValueError(f"Prompt optimization overlay missing optimized_prompt: {subject_id}")

        target = _resolve_overlay_target(
            source_path=str(overlay.get("source_path") or ""),
            source_path_absolute=str(overlay.get("source_path_absolute") or ""),
            source_path_repo_relative=str(overlay.get("source_path_repo_relative") or ""),
            dotted_path=dotted_path,
            generator=generator,
            scenarios_dir=scenarios_dir,
            config_dir=config_dir,
        )
        if target is None:
            print(
                "Warning: Prompt optimization overlay target not loaded in SynthChat generate; "
                f"skipping subject={subject_id} source={overlay.get('source_path')} path={dotted_path}"
            )
            continue

        container, key = target
        container[key] = _coerce_overlay_value(container.get(key), overlay["optimized_prompt"])
        applied.append(str(subject_id))
    return applied


def _resolve_overlay_target(
    *,
    source_path: str,
    dotted_path: str,
    source_path_absolute: str = "",
    source_path_repo_relative: str = "",
    generator,
    scenarios_dir: Path,
    config_dir: Path,
):
    normalized_sources = _normalized_source_candidates(
        source_path=source_path,
        source_path_absolute=source_path_absolute,
        source_path_repo_relative=source_path_repo_relative,
        config_dir=config_dir,
    )
    scenarios_prefix = _normalize_repo_path(str(scenarios_dir))

    if any(source.startswith(scenarios_prefix + "/") for source in normalized_sources):
        parts = dotted_path.split(".")
        if parts and parts[0] == "scenarios":
            parts = parts[1:]
        return _resolve_container(generator.scenario_loader.scenarios, parts)

    if _normalize_repo_path(str(config_dir / "tool_call_formats.yaml")) in normalized_sources:
        parts = dotted_path.split(".")
        if parts and parts[0] == "formats":
            parts = parts[1:]
        return _resolve_container(generator._tool_call_formats, parts)

    if _normalize_repo_path(str(config_dir / "workspace_formats.yaml")) in normalized_sources:
        parts = dotted_path.split(".")
        if parts and parts[0] == "formats":
            parts = parts[1:]
        return _resolve_container(generator._workspace_formats, parts)

    if _normalize_repo_path(str(config_dir / "label_mappings.yaml")) in normalized_sources:
        return _resolve_container(generator._label_mappings, dotted_path.split("."))

    return None


def _normalized_source_candidates(
    *,
    source_path: str,
    source_path_absolute: str,
    source_path_repo_relative: str,
    config_dir: Path,
) -> set[str]:
    candidates = set()
    for value in [source_path_absolute, source_path]:
        normalized = _normalize_repo_path(value)
        if normalized:
            candidates.add(normalized)

    repo_root = _find_repo_root(config_dir)
    for value in [source_path_repo_relative, source_path]:
        if value and not Path(value).expanduser().is_absolute():
            candidates.add(_normalize_repo_path(str(repo_root / value)))
    return candidates


def _find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "tuner.py").exists():
            return candidate
    return Path.cwd().resolve()


def _resolve_container(root: Any, parts: List[str]):
    current = root
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None

    if not parts:
        return None
    key = parts[-1]
    if isinstance(current, dict):
        if key not in current:
            return None
        return current, key
    if isinstance(current, list):
        try:
            index = int(key)
        except ValueError:
            return None
        if index < 0 or index >= len(current):
            return None
        return current, index
    return None


def _coerce_overlay_value(existing: Any, optimized_prompt: Any) -> Any:
    if isinstance(existing, list) and isinstance(optimized_prompt, str):
        return optimized_prompt.splitlines()
    return deepcopy(optimized_prompt)


def _normalize_repo_path(path: str) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve()).replace("\\", "/").lower()


class _PromptOptimizationResultWriter:
    def __init__(self, write_fn, metadata: Optional[Dict[str, Any]]):
        self._write_fn = write_fn
        self._metadata = metadata

    def write(self, result) -> bool:
        if self._metadata and getattr(result, "example", None):
            metadata = result.example.setdefault("metadata", {})
            metadata["prompt_optimization"] = deepcopy(self._metadata)
        return self._write_fn(result)


def _build_docs_work_items(docs, args, generator, targets, shared_seed_spec,
                           environment_validator, create_llm_client, settings,
                           max_iterations):
    """Build work item tuples for parallel docs-based generation."""
    config_dir = Path(args.config_dir or "SynthChat/config")
    scenarios_dir = Path(args.scenarios_dir or "SynthChat/scenarios")
    rubrics_dir = Path(args.rubrics_dir or "SynthChat/rubrics")

    work_items = []
    task_id = 0
    env_options = serialize_environment_options(environment_validator)

    for doc in docs:
        for rep in range(args.per_doc):
            shared_targets = set((shared_seed_spec or {}).get("targets") or [])
            if shared_seed_spec:
                shared_scenario = generator.scenario_loader.get_scenario(shared_seed_spec["scenario"])
                if not shared_scenario:
                    raise ValueError(f"Shared seed scenario not found: {shared_seed_spec['scenario']}")
                for shared_seed_index in range(shared_seed_spec["seed_count"]):
                    shared_seed_id = f"{shared_seed_spec['scenario']}:shared_seed:{shared_seed_index + 1}"
                    shared_seed_bundle = generator.prepare_seed_bundle(
                        scenario_key=shared_seed_spec["scenario"],
                        seed_id=shared_seed_id,
                        scenario=shared_scenario,
                        randomize_params=True,
                        doc_context=doc,
                    )
                    for scenario_key, raw_target in targets.items():
                        if shared_targets and scenario_key not in shared_targets:
                            continue
                        scenario = generator.scenario_loader.get_scenario(scenario_key)
                        if not scenario:
                            print(f"Warning: Scenario not found: {scenario_key}")
                            continue
                        target_spec = _normalize_target_spec(raw_target)
                        for rollout_index in range(target_spec["rollouts_per_seed"]):
                            work_items.append((
                                scenario_key, scenario, max_iterations,
                                config_dir, scenarios_dir, rubrics_dir,
                                settings, args.provider, args.model, doc, task_id,
                                env_options, create_llm_client,
                                shared_seed_bundle,
                                {
                                    "seed_id": shared_seed_id,
                                    "seed_index": shared_seed_index,
                                    "seed_count": shared_seed_spec["seed_count"],
                                    "rollout_index": rollout_index,
                                    "rollouts_per_seed": target_spec["rollouts_per_seed"],
                                    "shared_seed_source": shared_seed_spec["scenario"],
                                    "shared_across_scenarios": True,
                                },
                            ))
                            task_id += 1
            for scenario_key, raw_target in targets.items():
                if shared_seed_spec and (not shared_seed_spec.get("targets") or scenario_key in set(shared_seed_spec.get("targets") or [])):
                    continue
                scenario = generator.scenario_loader.get_scenario(scenario_key)
                if not scenario:
                    print(f"Warning: Scenario not found: {scenario_key}")
                    continue
                target_spec = _normalize_target_spec(raw_target)
                for seed_index in range(target_spec["seed_count"]):
                    seed_id = f"{scenario_key}:seed:{seed_index + 1}"
                    seed_bundle = generator.prepare_seed_bundle(
                        scenario_key=scenario_key,
                        seed_id=seed_id,
                        scenario=scenario,
                        randomize_params=True,
                        doc_context=doc,
                    )
                    for rollout_index in range(target_spec["rollouts_per_seed"]):
                        work_items.append((
                            scenario_key, scenario, max_iterations,
                            config_dir, scenarios_dir, rubrics_dir,
                            settings, args.provider, args.model, doc, task_id,
                            env_options, create_llm_client,
                            seed_bundle,
                            {
                                "seed_id": seed_id,
                                "seed_index": seed_index,
                                "seed_count": target_spec["seed_count"],
                                "rollout_index": rollout_index,
                                "rollouts_per_seed": target_spec["rollouts_per_seed"],
                            },
                        ))
                        task_id += 1

    return work_items


def _build_nodocs_work_items(generator, targets, shared_seed_spec,
                             args, settings, environment_validator,
                             create_llm_client, max_iterations):
    """Build work item tuples for parallel generation without docs."""
    config_dir = Path(args.config_dir or "SynthChat/config")
    scenarios_dir = Path(args.scenarios_dir or "SynthChat/scenarios")
    rubrics_dir = Path(args.rubrics_dir or "SynthChat/rubrics")

    work_items = []
    task_id = 0
    env_options = serialize_environment_options(environment_validator)
    shared_targets = set((shared_seed_spec or {}).get("targets") or [])

    if shared_seed_spec:
        shared_scenario = generator.scenario_loader.get_scenario(shared_seed_spec["scenario"])
        if not shared_scenario:
            raise ValueError(f"Shared seed scenario not found: {shared_seed_spec['scenario']}")
        for shared_seed_index in range(shared_seed_spec["seed_count"]):
            shared_seed_id = f"{shared_seed_spec['scenario']}:shared_seed:{shared_seed_index + 1}"
            shared_seed_bundle = generator.prepare_seed_bundle(
                scenario_key=shared_seed_spec["scenario"],
                seed_id=shared_seed_id,
                scenario=shared_scenario,
                randomize_params=True,
            )
            for scenario_key, raw_target in targets.items():
                if shared_targets and scenario_key not in shared_targets:
                    continue
                scenario = generator.scenario_loader.get_scenario(scenario_key)
                if not scenario:
                    print(f"Warning: Scenario not found: {scenario_key}")
                    continue
                target_spec = _normalize_target_spec(raw_target)
                for rollout_index in range(target_spec["rollouts_per_seed"]):
                    work_items.append((
                        scenario_key, scenario, max_iterations,
                        config_dir, scenarios_dir, rubrics_dir,
                        settings, args.provider, args.model, None, task_id,
                        env_options, create_llm_client,
                        shared_seed_bundle,
                        {
                            "seed_id": shared_seed_id,
                            "seed_index": shared_seed_index,
                            "seed_count": shared_seed_spec["seed_count"],
                            "rollout_index": rollout_index,
                            "rollouts_per_seed": target_spec["rollouts_per_seed"],
                            "shared_seed_source": shared_seed_spec["scenario"],
                            "shared_across_scenarios": True,
                        },
                    ))
                    task_id += 1

    for scenario_key, raw_target in targets.items():
        if shared_seed_spec and (not shared_seed_spec.get("targets") or scenario_key in set(shared_seed_spec.get("targets") or [])):
            continue
        scenario = generator.scenario_loader.get_scenario(scenario_key)
        if not scenario:
            print(f"Warning: Scenario not found: {scenario_key}")
            continue
        target_spec = _normalize_target_spec(raw_target)
        for seed_index in range(target_spec["seed_count"]):
            seed_id = f"{scenario_key}:seed:{seed_index + 1}"
            seed_bundle = generator.prepare_seed_bundle(
                scenario_key=scenario_key,
                seed_id=seed_id,
                scenario=scenario,
                randomize_params=True,
            )
            for rollout_index in range(target_spec["rollouts_per_seed"]):
                work_items.append((
                    scenario_key, scenario, max_iterations,
                    config_dir, scenarios_dir, rubrics_dir,
                    settings, args.provider, args.model, None, task_id,
                    env_options, create_llm_client,
                    seed_bundle,
                    {
                        "seed_id": seed_id,
                        "seed_index": seed_index,
                        "seed_count": target_spec["seed_count"],
                        "rollout_index": rollout_index,
                        "rollouts_per_seed": target_spec["rollouts_per_seed"],
                    },
                ))
                task_id += 1

    return work_items
