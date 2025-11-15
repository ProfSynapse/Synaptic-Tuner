#!/bin/bash
# Wrapper script to run training with the correct Python environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate base conda environment (has CUDA support)
conda activate base

echo "============================================"
echo "KTO TRAINING"
echo "============================================"
echo "All settings from configs/training_config.py"
echo ""
echo "Using: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run training with all passed arguments
python "${SCRIPT_DIR}/train_kto.py" "$@"
