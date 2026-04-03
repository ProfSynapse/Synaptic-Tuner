"""SynthChat Generate Mode - Create new datasets from scenarios.

Location: SynthChat/modes/generate.py
Purpose: Implements the 'generate' CLI command. Loads settings and scenarios,
         creates LLM clients and generator, dispatches sequential or parallel
         generation, streams results to disk, and prints a summary.
Usage: Called by SynthChat.run.main() when command is 'generate'.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    # Generate
    max_iterations = args.max_iterations or settings["improvement"]["max_iterations"]
    args.workers = max(1, args.workers)
    num_workers = args.workers
    results = []

    # Determine output path early so we can stream results to disk
    output_file = Path(args.output) if args.output else generate_output_path(settings)

    with StreamingResultWriter(output_file, settings) as writer:
        if docs and num_workers > 1:
            # Parallel docs-based generation with multiple workers
            print(f"Using {num_workers} parallel workers for {len(docs)} doc(s)\n")
            work_items = _build_docs_work_items(
                docs, args, generator, targets, shared_seed_spec,
                environment_validator, create_llm_client, settings,
                max_iterations,
            )
            results.extend(run_parallel_generation(work_items, num_workers, writer))
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
                        on_result=writer.write,
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
            results.extend(run_parallel_generation(work_items, num_workers, writer))
        else:
            # Standard sequential generation (no docs)
            results = generator.generate_batch(
                targets=targets,
                max_iterations=max_iterations,
                randomize_params=True,
                on_result=writer.write,
                shared_seed_spec=shared_seed_spec,
            )

        print(f"\nStreamed {writer.count} examples to {output_file}")

    # Print summary
    print_summary(results, output_file)


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
