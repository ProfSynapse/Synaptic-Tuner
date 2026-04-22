"""SynthChat CLI - Unified entry point for dataset generation and improvement.

Location: SynthChat/run.py
Purpose: Single CLI for both "generate", "improve", and "validate" modes.
         Contains only the argument parser, factory functions for LLM clients
         and environment validators, and the main() dispatcher. Mode logic
         lives in SynthChat.modes.{generate,improve,validate}.
Usage: python -m SynthChat.run [generate|improve|validate] [options]

Commands:
    generate - Create new dataset from scenarios
    improve  - Improve existing dataset with rubrics
    validate - Check if dataset passes rubrics (no improvement)
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from shared.llm import create_client
from shared.environments import EnvironmentValidator
from .utils.yaml_loader import load_yaml

from .modes.generate import generate_mode
from .modes.improve import improve_mode
from .modes.validate import validate_mode


def load_settings(config_dir: Path) -> Dict:
    """Load settings.yaml configuration."""
    settings_path = config_dir / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings not found: {settings_path}")
    return load_yaml(settings_path)


def create_llm_client(config: Dict, mode: str = "generation",
                      provider_override: str = None, model_override: str = None):
    """Create LLM client for generation or improvement.

    Args:
        config: Settings configuration.
        mode: "generation" or "improvement".
        provider_override: CLI override for provider (optional).
        model_override: CLI override for model (optional).

    Returns:
        LLM client from shared.llm.
    """
    llm_config = config["llm"].get(mode, {})
    provider = provider_override or llm_config.get("provider", "lmstudio")
    model = model_override or llm_config.get("model", "local-model")

    # Build config defaults from settings
    config_defaults = {
        "provider": provider,
        "model": model,
        "temperature": llm_config.get("temperature", 0.7),
        "max_tokens": llm_config.get("max_tokens", 2048),
    }

    # Provider-specific config
    if provider == "unsloth":
        config_defaults["max_seq_length"] = llm_config.get("max_seq_length", 4096)
        config_defaults["load_in_4bit"] = llm_config.get("load_in_4bit", True)
        config_defaults["top_p"] = llm_config.get("top_p", 0.9)
    elif "provider_routing" in llm_config:
        config_defaults["provider_routing"] = llm_config["provider_routing"]

    # Create client using shared.llm factory
    client = create_client(config_defaults=config_defaults)
    setattr(client, "default_max_tokens", llm_config.get("max_tokens"))
    return client


def create_environment_validator(settings: Dict, args) -> Optional[EnvironmentValidator]:
    """Create optional environment validator from settings and CLI overrides."""
    env_settings = settings.get("environment", {})
    env_enabled = env_settings.get("enabled", False)

    backend = args.env_backend
    if backend is None:
        backend = env_settings.get("backend", "local" if env_enabled else "none")
    backend = backend.lower()

    if backend == "none":
        return None

    template = args.env_template if args.env_template is not None else env_settings.get("template")
    timeout_seconds = (
        args.env_timeout if args.env_timeout is not None else env_settings.get("timeout_seconds", 120.0)
    )
    api_key = args.env_api_key
    tool_schema_path = (
        args.env_tool_schema
        if args.env_tool_schema is not None
        else env_settings.get("tool_schema_path")
    )
    execution_config_path = (
        args.env_exec_config
        if args.env_exec_config is not None
        else env_settings.get("execution_config_path")
    )

    return EnvironmentValidator(
        backend=backend,
        e2b_template=template,
        e2b_api_key=api_key,
        timeout_seconds=float(timeout_seconds),
        tool_schema_path=tool_schema_path,
        execution_config_path=execution_config_path,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SynthChat - Synthetic dataset generation and improvement"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate new dataset")
    generate_parser.add_argument("--config-dir", help="Config directory path")
    generate_parser.add_argument("--scenarios-dir", help="Scenarios directory path")
    generate_parser.add_argument("--rubrics-dir", help="Rubrics directory path")
    generate_parser.add_argument("--output", "-o", help="Output file path")
    generate_parser.add_argument("--targets-file", help="JSON file with generation targets")
    generate_parser.add_argument("--scenarios", nargs="+", help="Specific scenarios to generate")
    generate_parser.add_argument("--max-iterations", type=int, help="Max improvement iterations")
    generate_parser.add_argument("--provider", help="LLM provider (overrides settings.yaml)")
    generate_parser.add_argument("--model", help="Model name (overrides settings.yaml)")
    generate_parser.add_argument("--docs", help="Path to doc file or folder (seed data for generation)")
    generate_parser.add_argument("--per-doc", type=int, default=1, help="Examples to generate per doc (default: 1)")
    generate_parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")
    generate_parser.add_argument(
        "--env-backend",
        choices=["none", "local", "e2b"],
        default=None,
        help="Environment execution backend (default: settings.yaml or disabled)",
    )
    generate_parser.add_argument(
        "--env-template",
        help="E2B template ID (for --env-backend e2b)",
    )
    generate_parser.add_argument(
        "--env-timeout",
        type=float,
        default=None,
        help="Environment command timeout in seconds",
    )
    generate_parser.add_argument(
        "--env-api-key",
        help="E2B API key override (default: E2B_API_KEY env var)",
    )
    generate_parser.add_argument(
        "--env-tool-schema",
        default=None,
        help="Path to tool schema YAML for environment execution",
    )
    generate_parser.add_argument(
        "--env-exec-config",
        default=None,
        help="Path to environment execution rules YAML",
    )

    # Improve command
    improve_parser = subparsers.add_parser("improve", help="Improve existing dataset")
    improve_parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    improve_parser.add_argument("--output", "-o", help="Output file path")
    improve_parser.add_argument("--config-dir", help="Config directory path")
    improve_parser.add_argument("--rubrics-dir", help="Rubrics directory path")
    improve_parser.add_argument("--rubrics", help="Comma-separated rubric names")
    improve_parser.add_argument("--start-line", type=int, help="Start line (1-indexed)")
    improve_parser.add_argument("--end-line", type=int, help="End line (inclusive)")
    improve_parser.add_argument(
        "--lines",
        help="Comma-separated 1-indexed line selectors (for example: 3,7,10-15)",
    )
    improve_parser.add_argument(
        "--line-file",
        help="Text file containing 1-indexed line selectors, one per line or comma-separated",
    )
    improve_parser.add_argument("--max-iterations", type=int, help="Max improvement iterations")
    improve_parser.add_argument("--provider", help="LLM provider (overrides settings.yaml)")
    improve_parser.add_argument("--model", help="Model name (overrides settings.yaml)")
    improve_parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset (no improvement)")
    validate_parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    validate_parser.add_argument("--config-dir", help="Config directory path")
    validate_parser.add_argument("--rubrics-dir", help="Rubrics directory path")
    validate_parser.add_argument("--rubrics", help="Comma-separated rubric names")
    validate_parser.add_argument("--provider", help="LLM provider (overrides settings.yaml)")
    validate_parser.add_argument("--model", help="Model name (overrides settings.yaml)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate mode, injecting factory functions
    if args.command == "generate":
        generate_mode(
            args,
            load_settings=load_settings,
            create_llm_client=create_llm_client,
            create_environment_validator=create_environment_validator,
        )
    elif args.command == "improve":
        improve_mode(
            args,
            load_settings=load_settings,
            create_llm_client=create_llm_client,
        )
    elif args.command == "validate":
        validate_mode(
            args,
            load_settings=load_settings,
            create_llm_client=create_llm_client,
        )


if __name__ == "__main__":
    main()
