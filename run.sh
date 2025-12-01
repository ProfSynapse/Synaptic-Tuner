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

# Run CLI
python tuner.py "$@"
