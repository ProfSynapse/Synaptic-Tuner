#!/bin/bash
# Setup script for Toolset-Training Environment (unsloth_latest)
# Usage: ./setup_env.sh
#
# This installs the latest Unsloth + Transformers 5 for Ministral 3 support
# Last updated: December 2025

set -e

ENV_NAME="unsloth_latest"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "Toolset-Training Environment Setup"
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo ""
echo "This will install:"
echo "  - Transformers 5.0.0.dev0 (Ministral 3 support)"
echo "  - TRL 0.22.2"
echo "  - Unsloth 2025.11.6+"
echo "  - PyTorch 2.9+ with CUDA 12.8"
echo "=========================================="

# Detect conda
if [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -d "/opt/conda" ]; then
    CONDA_BASE="/opt/conda"
else
    echo "Error: Could not find conda installation"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if environment exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists."
    read -p "Recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME"
    else
        echo "Updating existing environment instead..."
        conda activate "$ENV_NAME"

        # Just update the critical packages
        echo ""
        echo "[1/4] Installing Transformers 5 (Ministral 3 branch)..."
        pip install git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce -q

        echo "[2/4] Installing TRL 0.22.2..."
        pip install --no-deps trl==0.22.2 -q

        echo "[3/4] Installing Unsloth (latest)..."
        pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo -q

        echo "[4/4] Installing xformers..."
        pip install --upgrade xformers -q

        echo ""
        echo "=========================================="
        echo "Update complete!"
        echo "=========================================="
        exit 0
    fi
fi

# Create environment
echo ""
echo "[1/7] Creating conda environment $ENV_NAME..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# Activate
conda activate "$ENV_NAME"

# Install dependencies
echo ""
echo "[2/7] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel -q

# Install PyTorch with CUDA
echo ""
echo "[3/7] Installing PyTorch with CUDA 12.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q

# Install core ML libraries
echo ""
echo "[4/7] Installing core ML libraries..."
pip install accelerate bitsandbytes peft datasets huggingface-hub sentencepiece protobuf -q

# Install Transformers 5 from special branch (required for Ministral 3)
echo ""
echo "[5/7] Installing Transformers 5 (Ministral 3 branch)..."
pip install git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce -q

# Install TRL 0.22.2 (compatible with Transformers 5 + Unsloth)
echo ""
echo "[6/7] Installing TRL 0.22.2..."
pip install --no-deps trl==0.22.2 -q

# Install Unsloth and Xformers
echo ""
echo "[7/7] Installing Unsloth and Xformers..."
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo -q
pip install --upgrade xformers -q

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "
import torch
import transformers
import trl
import unsloth

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Transformers: {transformers.__version__}')
print(f'TRL: {trl.__version__}')
print(f'Unsloth: {unsloth.__version__}')

# Check FastVisionModel (required for Ministral 3)
try:
    from unsloth import FastVisionModel
    print('FastVisionModel: Available (Ministral 3 ready)')
except ImportError:
    print('FastVisionModel: NOT AVAILABLE')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "You can now run the CLI using: ./run.sh"
echo "=========================================="
