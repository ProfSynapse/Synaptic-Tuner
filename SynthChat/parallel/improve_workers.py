"""SynthChat Parallel Improve Workers - Thread-pool execution for improvement mode.

Location: SynthChat/parallel/improve_workers.py
Purpose: Execute dataset improvement work items in parallel using ThreadPoolExecutor.
         Each worker creates its own ImprovementEngine for thread safety.
Usage: Called by SynthChat.modes.improve when --workers > 1.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..engine import ImprovementEngine
from ..utils.logger import get_logger


def create_worker_engine(
    config_dir: Path,
    rubrics_dir: Path,
    settings: Dict,
    create_llm_client: Callable,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> ImprovementEngine:
    """Create a fresh improvement engine for a worker thread."""
    logger = get_logger("synthchat_improve_worker")
    improve_client = create_llm_client(
        settings,
        mode="improvement",
        provider_override=provider,
        model_override=model,
    )
    validation_config = config_dir / "validation.yaml"
    return ImprovementEngine(
        llm_client=improve_client,
        rubrics_dir=rubrics_dir,
        config_path=validation_config,
        logger=logger,
        enable_interactions=settings["logging"]["save_interactions"],
    )


def improve_single_example(args_tuple):
    """Worker function for improving a single example."""
    (
        task_id,
        example,
        rubric_keys,
        max_iterations,
        config_dir,
        rubrics_dir,
        settings,
        provider,
        model,
        create_llm_client_fn,
    ) = args_tuple

    try:
        engine = create_worker_engine(
            config_dir=config_dir,
            rubrics_dir=rubrics_dir,
            settings=settings,
            create_llm_client=create_llm_client_fn,
            provider=provider,
            model=model,
        )
        result = engine.run(
            example=example,
            rubric_keys=rubric_keys,
            max_iterations=max_iterations,
        )
        return result, None, task_id
    except Exception as exc:  # pragma: no cover - worker error path
        return None, f"Task {task_id} improvement error: {exc}", task_id


def run_parallel_improvement(work_items: List[Tuple], num_workers: int):
    """Execute improvement work items in parallel and preserve input ordering."""
    if not work_items:
        print("No work items to process")
        return []

    total = len(work_items)
    completed = 0
    indexed_results: Dict[int, object] = {}
    lock = threading.Lock()

    def update_progress():
        nonlocal completed
        with lock:
            completed += 1
            pct = (completed / total * 100) if total > 0 else 0.0
            print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end="", flush=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(improve_single_example, item): item for item in work_items}
        try:
            for future in as_completed(futures):
                result, error, task_id = future.result()
                if error:
                    print(f"\n{error}")
                indexed_results[task_id] = result
                update_progress()
        except BaseException as exc:  # pragma: no cover - interrupt path
            print(f"\nInterrupted: {exc}. Shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    print()
    return [indexed_results.get(task_id) for task_id in range(total)]
