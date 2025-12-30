"""Handler for dataset improvement workflow."""

import os
from pathlib import Path

from SynthChat.services.rubric_runner import RubricRunner
from SynthChat.utils.dataset_scanner import DatasetScanner
from SynthChat.utils.yaml_loader import load_config
from SynthChat.utils.logger import ImproveLogger
from shared.llm import create_client, LLMError
from shared.ui import (
    print_menu,
    print_header,
    print_config,
    print_info,
    print_error,
    print_success,
    confirm,
    prompt,
    spinner,
    BOX,
)
from tuner.utils import load_env_file


def handle_improve():
    """Handle dataset improvement workflow via interactive CLI."""

    print_header("DATASET IMPROVEMENT ENGINE", "Improve dataset quality with LLM feedback")

    # Load environment variables from root .env
    env_path = Path(__file__).parent.parent.parent / ".env"
    env_loaded = load_env_file(env_path)
    if not env_loaded:
        print_info("Note: .env file not found; improvement settings may be incomplete")

    # LLM defaults from settings.yaml (cloud-only defaults) with .env fallback
    try:
        cfg = load_config("settings")
        llm_defaults = cfg.get("llm", {}).get("improvement", {}) if isinstance(cfg, dict) else {}
    except Exception:
        llm_defaults = {}

    default_backend = llm_defaults.get("provider", os.getenv("IMPROVEMENT_BACKEND", "openrouter"))
    default_model = llm_defaults.get("model", os.getenv("IMPROVEMENT_MODEL", "local-model"))
    default_temp = float(llm_defaults.get("temperature", 0.3))
    default_max_tokens = int(llm_defaults.get("max_tokens", 2048))

    backend = print_menu([
        ("openrouter", f"{BOX['bullet']} OpenRouter (cloud, uses OPENROUTER_API_KEY)"),
        ("lmstudio", f"{BOX['bullet']} LM Studio (local, uses loaded model)"),
        ("ollama", f"{BOX['bullet']} Ollama (local, uses loaded model)")
    ], title=f"Select LLM provider (default: {default_backend})") or default_backend

    if not backend:
        print_info("Cancelled")
        return

    # List available models for LM Studio/Ollama (and OpenRouter if key present)
    model = default_model
    # For local providers, list models; for OpenRouter, stick to config/env default
    if backend in ("lmstudio", "ollama"):
        try:
            with spinner(f"Connecting to {backend}..."):
                client = create_client(
                    provider=backend,
                    model=default_model,
                    config_defaults=llm_defaults
                )
                models = client.list_models()
            if models:
                menu_items = [(m, f"{BOX['bullet']} {m}") for m in models]
                chosen = print_menu(menu_items, title="Select model to use:") or default_model
                if chosen:
                    model = chosen
        except LLMError as e:
            print_info(f"Could not list models for {backend}: {e}")
        except Exception:
            # Silent fallback to config/env default
            pass

    # Use defaults for temp/max_tokens without prompting
    temperature = default_temp
    max_tokens = default_max_tokens

    if backend == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        print_error("OPENROUTER_API_KEY not found in .env file")
        print_info("Please add your OpenRouter API key to the root .env file")
        return

    # Scan datasets
    with spinner("Scanning datasets..."):
        scanner = DatasetScanner()
        datasets = scanner.scan()

    # Select category (folder)
    categories = list(datasets.keys())
    cat_menu = []
    for cat in categories:
        total_agents = len(datasets[cat])
        cat_menu.append((cat, f"{BOX['bullet']} {cat} ({total_agents} datasets)"))
    category = print_menu(cat_menu, title="Select dataset category:")
    if not category:
        print_info("Cancelled")
        return

    # Select dataset (agent)
    agents = sorted(datasets[category].keys())
    agent_menu = []
    for agent in agents:
        versions = datasets[category][agent]
        agent_menu.append((agent, f"{BOX['bullet']} {agent} ({len(versions)} versions)"))
    agent = print_menu(agent_menu, title="Select dataset:")
    if not agent:
        print_info("Cancelled")
        return

    # Select version
    versions = datasets[category][agent]
    version_menu = []
    for version in versions:
        file_path = scanner.get_file_path(category, agent, version)
        count = scanner.count_examples(file_path)
        version_menu.append((str(version), f"{BOX['bullet']} v{version} ({count} examples)"))
    version = print_menu(version_menu, title="Select version to improve:")
    if not version:
        print_info("Cancelled")
        return

    # Get file paths
    input_file = scanner.get_file_path(category, agent, version)
    next_version = scanner.get_next_version(version)
    output_file = scanner.get_file_path(category, agent, next_version)

    # Get settings
    try:
        if backend in ("lmstudio", "ollama"):
            batch_size = 1
            print_info("Using batch size 1 for local backend (LM Studio/Ollama)")
        else:
            batch_str = prompt("Batch size", "10")
            batch_size = int(batch_str) if batch_str else 10

        start_str = prompt("Start from line", "1")
        start_line = int(start_str) if start_str else 1

        total_lines = scanner.count_examples(input_file)
        end_str = prompt(f"End at line (total: {total_lines})", str(total_lines))
        end_line = int(end_str) if end_str else total_lines

        dry_run = confirm("Dry run first?")

    except ValueError:
        print_error("Invalid input")
        return

    # Configuration summary
    print_config({
        "Category": category,
        "Dataset": agent,
        "Input": f"v{version} ({scanner.count_examples(input_file)} examples)",
        "Output": f"v{next_version}",
        "Backend": backend,
        "Model": model,
        "Temperature": str(temperature),
        "Max tokens": str(max_tokens),
        "Batch size": str(batch_size),
        "Range": f"Lines {start_line}-{end_line}",
        "Mode": "DRY RUN" if dry_run else "LIVE",
    }, "Configuration Summary")

    if not confirm("Proceed with improvement?"):
        print_info("Cancelled")
        return

    # Get rubrics directory
    rubrics_dir = Path(__file__).parent.parent.parent / "SynthChat" / "rubrics"

    # Initialize RubricRunner
    runner = RubricRunner(
        rubrics_dir=rubrics_dir,
        backend=backend,
        host=None,
        port=None
    )

    # Select rubrics (use default or interactive)
    rubric_keys = runner.select_rubrics_interactive()
    if not rubric_keys:
        print_info("No rubrics selected. Cancelled.")
        return

    # Set max iterations
    max_iterations = cfg.get("improvement", {}).get("max_iterations", 3) if cfg else 3

    print_info(f"Starting improvement with {len(rubric_keys)} rubrics...")
    print()

    try:
        runner.run_on_file(
            file_path=input_file,
            output_path=output_file,
            rubric_keys=rubric_keys,
            start_line=start_line,
            end_line=end_line,
            max_iterations=max_iterations
        )
        print()
        print_success(f"Improvement complete! Output saved to: {output_file}")
    except KeyboardInterrupt:
        print_info("\nInterrupted by user")
    except Exception as e:
        print_error(f"Error: {e}")

    print()
