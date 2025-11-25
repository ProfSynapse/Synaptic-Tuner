#!/usr/bin/env bash
# Training wrapper for rtx3090_sft
# Usage: ./train.sh --model-size 7b [options]

set -e  # Exit on error

# Source conda if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Activate environment if it exists
if conda env list | grep -q "^unsloth_env "; then
    conda activate unsloth_env
    echo "✓ Activated conda environment: unsloth_env"
fi

# Run training script with all arguments passed through
python train_sft.py "$@"
