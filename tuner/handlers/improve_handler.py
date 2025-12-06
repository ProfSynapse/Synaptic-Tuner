"""Handler for dataset improvement workflow."""

import os
from pathlib import Path
from dotenv import load_dotenv

from Datasets.improvement_engine.core.models import ImprovementConfig
from Datasets.improvement_engine.services.improvement_service import ImprovementService
from Datasets.improvement_engine.utils.logger import get_logger
from Datasets.improvement_engine.utils.dataset_scanner import DatasetScanner


def handle_improve():
    """Handle dataset improvement workflow via interactive CLI."""

    print("\n" + "=" * 50)
    print("   Dataset Improvement Engine")
    print("=" * 50 + "\n")

    # Load environment variables from improvement_engine/.env
    env_path = Path(__file__).parent.parent.parent / "Datasets" / "improvement_engine" / ".env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in .env file")
        print("Please create Datasets/improvement_engine/.env with your API key")
        return

    # Scan datasets
    scanner = DatasetScanner()
    datasets = scanner.scan()

    # Select category
    print("Select dataset category:")
    categories = list(datasets.keys())
    for i, cat in enumerate(categories, 1):
        count = sum(len(agents) for agents in datasets[cat].values())
        print(f"  {i}. {cat} ({count} agents)")

    try:
        cat_choice = int(input("\nChoice: ")) - 1
        if cat_choice < 0 or cat_choice >= len(categories):
            print("Invalid choice")
            return
        category = categories[cat_choice]
    except (ValueError, IndexError):
        print("Invalid input")
        return

    # Select agent
    print(f"\nSelect agent from {category}:")
    agents = sorted(datasets[category].keys())
    for i, agent in enumerate(agents, 1):
        versions = datasets[category][agent]
        print(f"  {i}. {agent} ({len(versions)} versions)")

    try:
        agent_choice = int(input("\nChoice: ")) - 1
        if agent_choice < 0 or agent_choice >= len(agents):
            print("Invalid choice")
            return
        agent = agents[agent_choice]
    except (ValueError, IndexError):
        print("Invalid input")
        return

    # Select version
    print(f"\nAvailable {agent} versions:")
    versions = datasets[category][agent]
    for i, version in enumerate(versions, 1):
        file_path = scanner.get_file_path(category, agent, version)
        count = scanner.count_examples(file_path)
        print(f"  {i}. v{version} ({count} examples)")

    try:
        ver_choice = int(input("\nSelect version to improve: ")) - 1
        if ver_choice < 0 or ver_choice >= len(versions):
            print("Invalid choice")
            return
        version = versions[ver_choice]
    except (ValueError, IndexError):
        print("Invalid input")
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
    print(f"Model:      {os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')}")
    print(f"Batch size: {batch_size}")
    print(f"Range:      Lines {start_line}-{end_line}")
    print(f"Mode:       {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 50)

    proceed = input("\nProceed? [y/n]: ").lower()
    if proceed != 'y':
        print("Cancelled")
        return

    # Create configuration
    config = ImprovementConfig(
        input_file=input_file,
        output_file=output_file,
        batch_size=batch_size,
        start_line=start_line,
        end_line=end_line,
        dry_run=dry_run,
        model=os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini'),
        api_key=api_key
    )

    # Run improvement
    logger = get_logger()
    service = ImprovementService(config=config, logger=logger)

    try:
        service.run()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")

    print()
