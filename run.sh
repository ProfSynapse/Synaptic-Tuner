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
    if conda activate "$UNSLOTH_ENV" 2>/dev/null; then
        echo "✓ Using $UNSLOTH_ENV environment"
    else
        echo "✗ Could not activate $UNSLOTH_ENV environment"
        echo "  Please run setup first: cd Trainers && source activate_unsloth_latest.sh"
        exit 1
    fi
else
    echo "✗ Conda not found"
    exit 1
fi

# Run CLI
python tuner.py "$@"
