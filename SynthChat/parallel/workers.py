"""SynthChat Parallel Workers - Thread-pool execution for parallel generation.

Location: SynthChat/parallel/workers.py
Purpose: Execute generation work items in parallel using ThreadPoolExecutor.
         Each worker creates its own SynthChatGenerator instance for thread
         safety. Handles progress tracking, error reporting, streaming to disk,
         and graceful shutdown on interrupts.
Usage: Called by SynthChat.modes.generate when --workers > 1.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from shared.environments import EnvironmentValidator

from ..utils.logger import get_logger
from ..engine import ImprovementEngine
from ..generator import SynthChatGenerator


def serialize_environment_options(environment_validator: Optional[EnvironmentValidator]) -> Dict:
    """Serialize environment validator configuration for worker fan-out."""
    if environment_validator is None:
        return {"backend": "none"}
    return {
        "backend": environment_validator.backend,
        "template": environment_validator.e2b_template,
        "api_key": environment_validator.e2b_api_key,
        "timeout_seconds": environment_validator.timeout_seconds,
        "tool_schema_path": str(environment_validator.tool_schema_path),
        "execution_config_path": str(environment_validator.execution_config_path),
    }


def create_environment_validator_from_options(options: Dict) -> Optional[EnvironmentValidator]:
    """Recreate environment validator from serialized options."""
    backend = (options or {}).get("backend", "none")
    if backend == "none":
        return None
    return EnvironmentValidator(
        backend=backend,
        e2b_template=(options or {}).get("template"),
        e2b_api_key=(options or {}).get("api_key"),
        timeout_seconds=float((options or {}).get("timeout_seconds", 120.0)),
        tool_schema_path=(options or {}).get("tool_schema_path"),
        execution_config_path=(options or {}).get("execution_config_path"),
    )


def create_worker_generator(
    config_dir: Path,
    scenarios_dir: Path,
    rubrics_dir: Path,
    settings: Dict,
    create_llm_client: Callable,
    provider: str = None,
    model: str = None,
    environment_options: Optional[Dict] = None,
):
    """Create a new generator instance for a worker thread.

    Args:
        config_dir: Path to config directory.
        scenarios_dir: Path to scenarios directory.
        rubrics_dir: Path to rubrics directory.
        settings: Loaded settings dict.
        create_llm_client: Factory function for LLM clients (from run.py).
        provider: Optional provider override.
        model: Optional model override.
        environment_options: Serialized environment validator config.
    """
    logger = get_logger("synthchat_generate_worker")
    gen_client = create_llm_client(settings, mode="generation",
                                   provider_override=provider, model_override=model)
    improve_client = create_llm_client(settings, mode="improvement",
                                       provider_override=provider, model_override=model)

    validation_config = config_dir / "validation.yaml"
    engine = ImprovementEngine(
        llm_client=improve_client,
        rubrics_dir=rubrics_dir,
        config_path=validation_config,
        logger=logger,
        enable_interactions=settings["logging"]["save_interactions"]
    )

    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=gen_client,
        engine=engine,
        environment_validator=create_environment_validator_from_options(environment_options or {}),
        enable_stage_validation=settings["generation"]["stage_validation"],
        logger=logger,
        privacy_settings=settings.get("privacy_preprocess"),
    )
    return generator


def generate_single_example(args_tuple):
    """Worker function to generate a single example.

    The args_tuple includes a create_llm_client callable at index 12
    (after environment_options) to avoid circular imports.

    Returns:
        Tuple of (result, error, task_id) where task_id preserves input ordering.
    """
    (scenario_key, scenario, max_iterations, config_dir, scenarios_dir,
     rubrics_dir, settings, provider, model, doc_context, task_id,
     environment_options, create_llm_client_fn, seed_bundle,
     rollout_metadata) = args_tuple

    try:
        # Each worker creates its own generator (thread-safe LLM clients)
        generator = create_worker_generator(
            config_dir, scenarios_dir, rubrics_dir, settings,
            create_llm_client=create_llm_client_fn,
            provider=provider, model=model,
            environment_options=environment_options
        )

        result = generator.generate_single(
            scenario_key,
            scenario,
            max_iterations,
            True,
            doc_context,
            seed_bundle=seed_bundle,
            rollout_metadata=rollout_metadata,
        )
        return result, None, task_id
    except Exception as e:
        return None, f"Task {task_id} error for {scenario_key}: {e}", task_id


def run_parallel_generation(work_items: List, num_workers: int,
                            writer=None) -> List:
    """Execute work items in parallel using a thread pool.

    Shared execution logic for both docs-based and non-docs parallel generation.
    Handles progress tracking, error reporting, streaming to disk, and graceful
    shutdown on interrupts.

    Args:
        work_items: List of tuples to pass to generate_single_example.
        num_workers: Number of parallel worker threads.
        writer: Optional StreamingResultWriter to stream results as they complete.

    Returns:
        List of GenerationResult objects (successful results only),
        sorted by task_id to preserve input ordering.
    """
    if not work_items:
        print("No work items to process (check scenario names)")
        return []

    total = len(work_items)
    completed = 0
    lock = threading.Lock()
    indexed_results = []  # List of (task_id, result) for ordering

    def update_progress():
        nonlocal completed
        with lock:
            completed += 1
            pct = (completed / total * 100) if total > 0 else 0
            print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end="", flush=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(generate_single_example, item): item for item in work_items}

        try:
            for future in as_completed(futures):
                result, error, task_id = future.result()
                if error:
                    print(f"\n{error}")
                if result:
                    if writer:
                        writer.write(result)
                    indexed_results.append((task_id, result))
                update_progress()
        except BaseException as e:
            print(f"\nInterrupted: {e}. Shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    print()  # Newline after progress

    # Sort by task_id to preserve input document order
    indexed_results.sort(key=lambda x: x[0])
    return [result for _, result in indexed_results]
