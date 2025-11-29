#!/bin/bash
# Universal upload script for all trainers
# Uses the shared upload framework

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINERS_DIR="$(dirname "$SCRIPT_DIR")"

# Detect trainer type from current directory or argument
detect_trainer_type() {
    local current_dir=$(basename "$(pwd)")
    case "$current_dir" in
        rtx3090_sft)
            echo "sft"
            ;;
        rtx3090_kto)
            echo "kto"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

TRAINER_TYPE=$(detect_trainer_type)

# Source conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# Activate environment
conda activate unsloth_env 2>/dev/null || {
    echo "Warning: Could not activate unsloth_env"
}

# Run the universal upload script
python "$SCRIPT_DIR/upload_model.py" "$@"
