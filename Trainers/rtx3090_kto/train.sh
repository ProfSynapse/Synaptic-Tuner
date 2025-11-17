#!/bin/bash
# Wrapper script to run training with the correct Python environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate unsloth_env (has Unsloth + CUDA support)
conda activate unsloth_env || conda activate base

echo "============================================"
echo "KTO TRAINING - Qwen 2.5 3B Instruct"
echo "============================================"
echo "All settings from configs/training_config.py"
echo ""
echo "Using: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run training with local dataset and all passed arguments
python "${SCRIPT_DIR}/train_kto.py" --local-file "../../Datasets/syngen_tools_11.14.25.jsonl" "$@"
