#!/bin/bash
# Toolset-Training Unified CLI - Bash wrapper
# Usage: ./run.sh [train|upload|eval|pipeline]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source conda if available
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate unsloth_env 2>/dev/null || true
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate unsloth_env 2>/dev/null || true
fi

# Run CLI
python cli.py "$@"
