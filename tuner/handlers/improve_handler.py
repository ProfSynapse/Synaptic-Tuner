"""Handler for dataset improvement workflow."""

import os
from pathlib import Path

from SynthChat.services.rubric_runner import RubricRunner
from SynthChat.utils.dataset_scanner import DatasetScanner
from SynthChat.utils.yaml_loader import load_config
from SynthChat.utils.logger import ImproveLogger
from shared.llm import create_client, LLMError
from shared.ui import print_menu, print_header
from tuner.utils import load_env_file


def handle_improve():
    """Handle dataset improvement workflow via interactive CLI."""

    print("\n" + "=" * 50)
    print("   Dataset Improvement Engine")
    print("=" * 50 + "\n")

    # Load environment variables from root .env
    env_path = Path(__file__).parent.parent.parent / ".env"
    env_loaded = load_env_file(env_path)
    if not env_loaded:
        print("WARN: .env file not found; improvement settings may be incomplete")

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

    print_header("LLM BACKEND", "Select provider (defaults from config/env)")
    backend = print_menu([
        ("openrouter", "OpenRouter (cloud, uses OPENROUTER_API_KEY)"),
        ("lmstudio", "LM Studio (local, uses loaded model)"),
        ("ollama", "Ollama (local, uses loaded model)")
    ], title=f"Choose provider (default: {default_backend})") or default_backend

    # List available models for LM Studio/Ollama (and OpenRouter if key present)
    model = default_model
    # For local providers, list models; for OpenRouter, stick to config/env default
    if backend in ("lmstudio", "ollama"):
        try:
            client = create_client(
                provider=backend,
                model=default_model,
                config_defaults=llm_defaults
            )
            models = client.list_models()
            if models:
                menu_items = [(m, m) for m in models]
                chosen = print_menu(menu_items, title="Select model to use:") or default_model
                model = chosen
        except LLMError as e:
            print(f"WARN: Could not list models for {backend}: {e}")
        except Exception:
            # Silent fallback to config/env default
            pass

    # Use defaults for temp/max_tokens without prompting
    temperature = default_temp
    max_tokens = default_max_tokens

    if backend == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found in .env file")
        print("Please add your OpenRouter API key to the root .env file")
        return

    # Scan datasets
    scanner = DatasetScanner()
    datasets = scanner.scan()

    # Select category (folder)
    print_header("DATASETS", "Select category")
    categories = list(datasets.keys())
    cat_menu = []
    for cat in categories:
        total_agents = len(datasets[cat])
        cat_menu.append((cat, f"{cat} ({total_agents} datasets)"))
    category = print_menu(cat_menu, title="Choose category:") or None
    if not category:
        print("Cancelled")
        return

    # Select dataset (agent)
    agents = sorted(datasets[category].keys())
    agent_menu = []
    for agent in agents:
        versions = datasets[category][agent]
        agent_menu.append((agent, f"{agent} ({len(versions)} versions)"))
    agent = print_menu(agent_menu, title="Choose dataset:") or None
    if not agent:
        print("Cancelled")
        return

    # Select version
    versions = datasets[category][agent]
    version_menu = []
    for version in versions:
        file_path = scanner.get_file_path(category, agent, version)
        count = scanner.count_examples(file_path)
        version_menu.append((str(version), f"v{version} ({count} examples)"))
    version = print_menu(version_menu, title="Choose version to improve:") or None
    if not version:
        print("Cancelled")
        return

    # Get file paths
    input_file = scanner.get_file_path(category, agent, version)
    next_version = scanner.get_next_version(version)
    output_file = scanner.get_file_path(category, agent, next_version)

    # Get settings
    print("\n" + "=" * 50)
    print("   Improvement Settings")
    print("=" * 50)

    try:
        if backend in ("lmstudio", "ollama"):
            batch_size = 1
            print("Using batch size 1 for local backend (LM Studio/Ollama)")
        else:
            batch_size = int(input(f"Batch size [default: 10]: ") or "10")

        start_line = int(input(f"Start from line [default: 1]: ") or "1")

        total_lines = scanner.count_examples(input_file)
        end_input = input(f"End at line [default: {total_lines} (all)]: ")
        end_line = int(end_input) if end_input else total_lines

        dry_run_input = input("Dry run first? [y/n]: ").lower()
        dry_run = dry_run_input == 'y'

    except ValueError:
        print("Invalid input")
        return

    # Configuration summary
    print("\n" + "=" * 50)
    print("   Configuration Summary")
    print("=" * 50)
    print(f"Category:   {category}")
    print(f"Agent:      {agent}")
    print(f"Input:      v{version} ({scanner.count_examples(input_file)} examples)")
    print(f"Output:     v{next_version}")
    print(f"Backend:    {backend}")
    print(f"Model:      {model}")
    print(f"Temp:       {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Batch size: {batch_size}")
    print(f"Range:      Lines {start_line}-{end_line}")
    print(f"Mode:       {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 50)

    proceed = input("\nProceed? [y/n]: ").lower()
    if proceed != 'y':
        print("Cancelled")
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
    print_header("RUBRICS", "Select rubrics to apply")
    rubric_keys = runner.select_rubrics_interactive()
    if not rubric_keys:
        print("No rubrics selected. Cancelled.")
        return

    # Set max iterations
    max_iterations = cfg.get("improvement", {}).get("max_iterations", 3) if cfg else 3

    try:
        runner.run_on_file(
            file_path=input_file,
            output_path=output_file,
            rubric_keys=rubric_keys,
            start_line=start_line,
            end_line=end_line,
            max_iterations=max_iterations
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")

    print()
