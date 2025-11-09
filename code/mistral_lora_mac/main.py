#!/usr/bin/env python3
"""
File: /Users/jrosenbaum/Documents/Code/Synthetic Conversations/code/mistral_lora_mac/main.py

Main Entry Point for MLX Fine-Tuning System

This script orchestrates the complete fine-tuning pipeline:
1. Load and validate configuration
2. Setup logging and monitoring
3. Initialize data pipeline
4. Load model and apply LoRA
5. Create optimizer and scheduler
6. Run training loop
7. Save final model
8. Run final evaluation
9. Generate training report

Usage:
    python main.py [--config path/to/config.yaml] [--resume path/to/checkpoint.npz]

Dependencies:
- All src modules (config, data, model, trainer, evaluator, utils)
- MLX framework
- transformers

Related Files:
- config/config.yaml: Default configuration
- All modules in src/
"""

import argparse
import sys
import traceback
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

# Import with absolute paths
import importlib.util

def load_module(name, path):
    """Load a module from an absolute path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load config manager
config_manager = load_module("config_manager", current_dir / "config" / "config_manager.py")
ConfigurationManager = config_manager.ConfigurationManager
ConfigurationError = config_manager.ConfigurationError

# Load utils
utils = load_module("utils", current_dir / "src" / "utils.py")
setup_logging = utils.setup_logging
MemoryMonitor = utils.MemoryMonitor
get_device_info = utils.get_device_info
ensure_dir = utils.ensure_dir
format_time = utils.format_time
seed_everything = utils.seed_everything
check_metal_availability = utils.check_metal_availability

# Load data pipeline
data_pipeline_mod = load_module("data_pipeline", current_dir / "src" / "data_pipeline.py")
DataPipeline = data_pipeline_mod.DataPipeline

# Load model manager
model_manager_mod = load_module("model_manager", current_dir / "src" / "model_manager.py")
ModelManager = model_manager_mod.ModelManager

# Load trainer
trainer_mod = load_module("trainer", current_dir / "src" / "trainer.py")
Trainer = trainer_mod.Trainer

# Load evaluator
evaluator_mod = load_module("evaluator", current_dir / "src" / "evaluator.py")
Evaluator = evaluator_mod.Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MLX Fine-Tuning System for Mistral-7B-Instruct-v0.3"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (optional)"
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (requires --resume)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path from config"
    )

    return parser.parse_args()


def print_banner(logger):
    """Print startup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           MLX Fine-Tuning System v1.0.0                      ║
    ║           Mistral-7B-Instruct-v0.3 + LoRA                    ║
    ║           Optimized for Apple Silicon (M4)                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    logger.info(banner)


def check_system_requirements(logger):
    """Check system requirements."""
    logger.info("Checking system requirements...")

    # Get device info
    device_info = get_device_info()

    logger.info(f"Platform: {device_info['platform']}")
    logger.info(f"Python Version: {device_info['python_version']}")
    logger.info(f"Total RAM: {device_info['total_ram_gb']:.2f} GB")
    logger.info(f"Metal Available: {device_info['metal_available']}")

    if not device_info['metal_available']:
        logger.warning("Metal GPU not available! Training will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    if device_info['total_ram_gb'] < 16:
        logger.warning(f"System has only {device_info['total_ram_gb']:.2f} GB RAM. "
                      f"16+ GB recommended for 7B model fine-tuning.")

    logger.info("System requirements check passed.")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    try:
        # Load configuration
        print(f"Loading configuration from: {config_path}")
        config_manager = ConfigurationManager(project_root=Path(__file__).parent)
        config = config_manager.load(str(config_path))

        # Override dataset path if provided
        if args.dataset:
            config.data.dataset_path = args.dataset

        # Ensure output directories exist
        ensure_dir(config.output.checkpoint_dir)
        ensure_dir(config.output.final_model_dir)
        ensure_dir(config.output.logs_dir)
        ensure_dir(config.output.metrics_dir)

        # Setup logging
        # Create a modified logging config with logs_dir from output config
        logging_config = config.logging
        logging_config.logs_dir = config.output.logs_dir
        logger = setup_logging(logging_config)

        print_banner(logger)

        # Check system requirements
        check_system_requirements(logger)

        # Set random seeds
        logger.info(f"Setting random seed: {config.training.seed}")
        seed_everything(config.training.seed)

        # Initialize memory monitor
        memory_monitor = MemoryMonitor(logger)
        memory_monitor.log_memory("startup")

        # Resolve dataset path
        dataset_path = Path(config.data.dataset_path)
        if not dataset_path.is_absolute():
            # Try relative to project root
            dataset_path = Path(__file__).parent.parent / dataset_path
            if not dataset_path.exists():
                # Try current directory
                dataset_path = Path.cwd() / config.data.dataset_path

        logger.info(f"Dataset path: {dataset_path}")
        config.data.dataset_path = str(dataset_path)

        # Initialize data pipeline
        logger.info("=" * 80)
        logger.info("Initializing Data Pipeline")
        logger.info("=" * 80)

        data_pipeline = DataPipeline(config, logger)
        data_pipeline.initialize()

        # Load and prepare data
        train_loader, val_loader = data_pipeline.load_and_prepare()

        memory_monitor.log_memory("after_data_loading")

        # Initialize model manager
        logger.info("=" * 80)
        logger.info("Initializing Model")
        logger.info("=" * 80)

        model_manager = ModelManager(config, logger)

        # Load base model
        model = model_manager.load_base_model()

        memory_monitor.log_memory("after_model_loading")

        # Apply LoRA
        model = model_manager.apply_lora()

        memory_monitor.log_memory("after_lora_application")

        # Get tokenizer
        tokenizer = data_pipeline.tokenizer

        # Initialize evaluator
        evaluator = Evaluator(model, tokenizer, config, logger)

        # Eval-only mode
        if args.eval_only:
            if not args.resume:
                logger.error("--eval-only requires --resume with checkpoint path")
                sys.exit(1)

            logger.info("Running evaluation only...")
            model_manager.load_adapters(args.resume)

            metrics = evaluator.evaluate(val_loader)

            logger.info("Evaluation complete.")
            logger.info(f"Validation Loss: {metrics.val_loss:.4f}")
            logger.info(f"Perplexity: {metrics.perplexity:.2f}")

            return

        # Initialize trainer
        logger.info("=" * 80)
        logger.info("Initializing Trainer")
        logger.info("=" * 80)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            logger=logger,
            memory_monitor=memory_monitor
        )

        # Resume from checkpoint if provided
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            model_manager.load_adapters(args.resume)

        # Run training
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)

        final_state = trainer.train()

        # Save final model
        logger.info("=" * 80)
        logger.info("Saving Final Model")
        logger.info("=" * 80)

        final_model_path = Path(config.output.final_model_dir) / "lora_adapters.npz"
        model_manager.save_adapters(str(final_model_path))

        logger.info(f"Final model saved to: {final_model_path}")

        # Final evaluation
        logger.info("=" * 80)
        logger.info("Running Final Evaluation")
        logger.info("=" * 80)

        final_metrics = evaluator.evaluate(val_loader)

        # Generate training report
        logger.info("=" * 80)
        logger.info("Training Summary")
        logger.info("=" * 80)

        logger.info(f"Total Epochs: {final_state.epoch + 1}")
        logger.info(f"Total Steps: {final_state.global_step}")
        logger.info(f"Best Validation Loss: {final_state.best_val_loss:.4f}")
        logger.info(f"Final Validation Loss: {final_metrics.val_loss:.4f}")
        logger.info(f"Final Perplexity: {final_metrics.perplexity:.2f}")

        peak_memory = memory_monitor.get_peak_usage()
        logger.info(f"Peak RAM Usage: {peak_memory['peak_ram_gb']:.2f} GB")
        if peak_memory['peak_metal_gb']:
            logger.info(f"Peak Metal Usage: {peak_memory['peak_metal_gb']:.2f} GB")

        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)

        # Save training report
        report_path = Path(config.output.metrics_dir) / "training_report.json"
        save_training_report(
            report_path,
            config,
            final_state,
            final_metrics,
            peak_memory
        )

        logger.info(f"Training report saved to: {report_path}")

    except ConfigurationError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def save_training_report(report_path, config, final_state, final_metrics, peak_memory):
    """Save training report to JSON file."""
    import json
    from dataclasses import asdict

    report = {
        'configuration': {
            'model': config.model.name,
            'lora_rank': config.lora.rank,
            'lora_alpha': config.lora.alpha,
            'learning_rate': config.training.learning_rate,
            'batch_size': config.training.per_device_batch_size,
            'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
            'num_epochs': config.training.num_epochs,
        },
        'training_results': {
            'total_epochs': final_state.epoch + 1,
            'total_steps': final_state.global_step,
            'best_val_loss': final_state.best_val_loss,
            'final_val_loss': final_metrics.val_loss,
            'final_perplexity': final_metrics.perplexity,
        },
        'resource_usage': {
            'peak_ram_gb': peak_memory['peak_ram_gb'],
            'peak_metal_gb': peak_memory.get('peak_metal_gb'),
        },
        'sample_generations': [
            {
                'prompt': s.prompt,
                'generated': s.generated_text,
                'time': s.generation_time
            }
            for s in final_metrics.samples
        ]
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
