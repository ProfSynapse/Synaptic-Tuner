#!/bin/bash
# Toolset-Training Unified CLI - Bash wrapper
# Usage: ./run.sh [train|upload|eval|pipeline]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Standard environment
UNSLOTH_ENV="unsloth_latest"

# Source conda
CONDA_SH=""
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    CONDA_SH=~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/.conda/etc/profile.d/conda.sh ]; then
    CONDA_SH=~/.conda/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    CONDA_SH=/opt/conda/etc/profile.d/conda.sh
fi

if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
    if conda env list | grep -q "$UNSLOTH_ENV"; then
        conda activate "$UNSLOTH_ENV"
        echo "✓ Using $UNSLOTH_ENV environment"
    else
        echo "⚠ Environment $UNSLOTH_ENV not found."
        read -p "Would you like to run setup now? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            bash setup_env.sh
            source "$CONDA_SH"
            conda activate "$UNSLOTH_ENV"
        else
            echo "✗ Setup cancelled. Cannot continue."
            exit 1
        fi
    fi
else
    echo "✗ Conda not found"
    exit 1
fi

# ============================================================================
# DEPENDENCY CHECK - Auto-install missing packages
# ============================================================================
check_and_install_deps() {
    local MISSING_DEPS=()
    local NEED_INSTALL=false

    # Check for unsloth
    if ! python -c "import unsloth" 2>/dev/null; then
        MISSING_DEPS+=("unsloth")
        NEED_INSTALL=true
    fi

    # Check for FastVisionModel (VL support) - CRITICAL for VL models
    if ! python -c "from unsloth import FastVisionModel" 2>/dev/null; then
        MISSING_DEPS+=("unsloth_zoo (Vision Model support)")
        NEED_INSTALL=true
    fi

    # Check for xformers
    if ! python -c "import xformers" 2>/dev/null; then
        MISSING_DEPS+=("xformers")
        NEED_INSTALL=true
    fi

    if [ "$NEED_INSTALL" = true ]; then
        echo ""
        echo "⚠ Missing dependencies detected:"
        for dep in "${MISSING_DEPS[@]}"; do
            echo "  - $dep"
        done
        echo ""

        # Check if we're in an interactive terminal
        if [ -t 0 ]; then
            # Interactive - ask user
            read -p "Install missing dependencies? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                DO_INSTALL=true
            else
                DO_INSTALL=false
            fi
        else
            # Non-interactive (e.g., from PowerShell via WSL) - auto-install
            echo "Non-interactive mode detected - auto-installing dependencies..."
            DO_INSTALL=true
        fi

        if [ "$DO_INSTALL" = true ]; then
            echo "Installing dependencies (this may take a minute)..."
            pip install --upgrade unsloth unsloth_zoo xformers
            echo ""

            # Verify installation
            if python -c "from unsloth import FastVisionModel" 2>/dev/null; then
                echo "✓ Dependencies installed successfully"
                echo "✓ Vision Model support (FastVisionModel) available"
            else
                echo "⚠ FastVisionModel still not available after install"
                echo "  Try manually: pip install --upgrade --force-reinstall unsloth unsloth_zoo"
                exit 1
            fi
        else
            echo "⚠ Skipping dependency installation"
            echo "  VL model operations will fail without FastVisionModel"
        fi
        echo ""
    fi
}

# Run dependency check
check_and_install_deps

# Run CLI
python tuner.py "$@"
