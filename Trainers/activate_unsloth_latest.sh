#!/bin/bash
# Activate the latest Unsloth environment (2025.11.4)
# Usage: source activate_unsloth_latest.sh

# Detect conda location
if [ -d "/home/profsynapse/miniconda3" ]; then
    CONDA_BASE="/home/profsynapse/miniconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "Error: Could not find conda installation"
    return 1 2>/dev/null || exit 1
fi

# Initialize conda for this shell
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the environment
conda activate unsloth_latest

# Verify activation
if [ "$CONDA_DEFAULT_ENV" = "unsloth_latest" ]; then
    echo "✓ Activated unsloth_latest environment"
    python -c "import unsloth; print(f'  Unsloth: {unsloth.__version__}')"
    python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
    python -c "import torch; print(f'  CUDA: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"
else
    echo "Error: Failed to activate unsloth_latest"
    return 1 2>/dev/null || exit 1
fi
