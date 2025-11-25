#!/usr/bin/env python3
"""
Verification script for MLX Fine-Tuning System installation.

This script checks:
1. Python version
2. Required dependencies
3. MLX and Metal GPU availability
4. Configuration loading
5. Dataset accessibility
6. Directory structure

Run with: python verify_installation.py
"""

import sys
import os
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text.center(80)}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    print(f"{RED}✗{RESET} {text}")

def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")

def check_python_version():
    """Check Python version >= 3.9"""
    print_header("Checking Python Version")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"Python version: {version_str}")

    if version.major >= 3 and version.minor >= 9:
        print_success(f"Python {version_str} is compatible")
        return True
    else:
        print_error(f"Python {version_str} is too old. Requires Python 3.9+")
        return False

def check_dependencies():
    """Check required Python packages"""
    print_header("Checking Dependencies")

    required_packages = [
        ('mlx', 'mlx.core'),
        ('transformers', 'transformers'),
        ('yaml', 'yaml'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
        ('psutil', 'psutil'),
    ]

    all_present = True

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print_success(f"{package_name} is installed")
        except ImportError:
            print_error(f"{package_name} is NOT installed")
            all_present = False

    if not all_present:
        print(f"\n{YELLOW}Install missing packages with:{RESET}")
        print("  pip install -r requirements.txt")

    return all_present

def check_mlx_metal():
    """Check MLX and Metal GPU availability"""
    print_header("Checking MLX and Metal GPU")

    try:
        import mlx.core as mx

        print_success("MLX is installed")

        # Check Metal availability
        try:
            # Try to create a small array
            x = mx.array([1.0, 2.0, 3.0])
            mx.eval(x)
            print_success("Metal GPU is available and working")
            return True
        except Exception as e:
            print_error(f"Metal GPU is not available: {e}")
            print_warning("Training will be very slow on CPU")
            return False

    except ImportError:
        print_error("MLX is not installed")
        print(f"\n{YELLOW}Install MLX with:{RESET}")
        print("  pip install mlx mlx-lm")
        return False

def check_configuration():
    """Check configuration file loads correctly"""
    print_header("Checking Configuration")

    config_path = Path(__file__).parent / "config" / "config.yaml"

    if not config_path.exists():
        print_error(f"Configuration file not found: {config_path}")
        return False

    print_success(f"Configuration file found: {config_path}")

    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))

        from config.config_manager import ConfigurationManager

        config_manager = ConfigurationManager()
        config = config_manager.load(str(config_path))

        print_success("Configuration loaded successfully")
        print(f"  Model: {config.model.name}")
        print(f"  LoRA rank: {config.lora.rank}")
        print(f"  Batch size: {config.training.per_device_batch_size}")
        print(f"  Learning rate: {config.training.learning_rate}")

        return True

    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        return False

def check_dataset():
    """Check dataset file accessibility"""
    print_header("Checking Dataset")

    # Try to find dataset
    dataset_name = "syngen_toolset_v1.0.0_claude.jsonl"

    search_paths = [
        Path(__file__).parent / dataset_name,
        Path(__file__).parent.parent / dataset_name,
        Path.cwd() / dataset_name,
    ]

    for path in search_paths:
        if path.exists():
            print_success(f"Dataset found: {path}")

            # Count lines
            try:
                with open(path, 'r') as f:
                    num_lines = sum(1 for _ in f)
                print(f"  Total examples: {num_lines}")

                # Try to parse first line
                with open(path, 'r') as f:
                    first_line = f.readline()
                    import json
                    data = json.loads(first_line)

                    if 'conversations' in data and 'label' in data:
                        print_success("Dataset format is valid")
                    else:
                        print_warning("Dataset may have incorrect format")

                return True

            except Exception as e:
                print_error(f"Error reading dataset: {e}")
                return False

    print_error(f"Dataset not found: {dataset_name}")
    print(f"\n{YELLOW}Searched locations:{RESET}")
    for path in search_paths:
        print(f"  - {path}")

    print(f"\n{YELLOW}To fix:{RESET}")
    print(f"  1. Place dataset in: {Path(__file__).parent / dataset_name}")
    print(f"  2. Or create symlink: ln -s /path/to/{dataset_name} {dataset_name}")

    return False

def check_directory_structure():
    """Check required directories exist"""
    print_header("Checking Directory Structure")

    base_path = Path(__file__).parent

    required_dirs = [
        "config",
        "src",
        "src/config",
        "src/data",
        "src/model",
        "src/training",
        "src/evaluation",
        "src/utils",
    ]

    all_present = True

    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print_success(f"{dir_name}/ exists")
        else:
            print_error(f"{dir_name}/ is missing")
            all_present = False

    # Check auto-created directories
    auto_dirs = ["logs", "checkpoints", "outputs"]

    print(f"\n{BLUE}Auto-created directories:{RESET}")
    for dir_name in auto_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print_success(f"{dir_name}/ exists")
        else:
            print_warning(f"{dir_name}/ will be created on first run")

    return all_present

def check_memory():
    """Check system memory"""
    print_header("Checking System Resources")

    try:
        import psutil

        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024 ** 3)
        available_gb = vm.available / (1024 ** 3)

        print(f"Total RAM: {total_gb:.2f} GB")
        print(f"Available RAM: {available_gb:.2f} GB")

        if total_gb >= 16:
            print_success("System has sufficient RAM (16+ GB)")
        else:
            print_warning(f"System has only {total_gb:.2f} GB RAM. 16+ GB recommended")

        if available_gb < 8:
            print_warning(f"Only {available_gb:.2f} GB RAM available. Close other applications")

        return True

    except ImportError:
        print_warning("psutil not installed, cannot check memory")
        return True

def main():
    """Run all verification checks"""
    print(f"\n{GREEN}{'*' * 80}{RESET}")
    print(f"{GREEN}{'MLX Fine-Tuning System - Installation Verification'.center(80)}{RESET}")
    print(f"{GREEN}{'*' * 80}{RESET}")

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("MLX and Metal GPU", check_mlx_metal),
        ("Configuration", check_configuration),
        ("Dataset", check_dataset),
        ("Directory Structure", check_directory_structure),
        ("System Resources", check_memory),
    ]

    results = {}

    for check_name, check_func in checks:
        results[check_name] = check_func()

    # Summary
    print_header("Verification Summary")

    passed = sum(results.values())
    total = len(results)

    for check_name, result in results.items():
        if result:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")

    print(f"\n{BLUE}Overall: {passed}/{total} checks passed{RESET}")

    if passed == total:
        print(f"\n{GREEN}✓ System is ready for training!{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print("  1. Review configuration: config/config.yaml")
        print("  2. Run quick test: python main.py --dataset test_dataset.jsonl")
        print("  3. Run full training: python main.py")
        return 0
    else:
        print(f"\n{RED}✗ Some checks failed. Please fix the issues above.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
