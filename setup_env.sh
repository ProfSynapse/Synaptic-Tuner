#!/bin/bash
# Setup script for Toolset-Training Environment (unsloth_latest)
# Usage: ./setup_env.sh

set -e

ENV_NAME="unsloth_latest"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "Toolset-Training Environment Setup"
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
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
        echo "Skipping creation."
        exit 0
    fi
fi

# Create environment
echo "Creating conda environment $ENV_NAME..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# Activate
conda activate "$ENV_NAME"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel -q

# Install from requirements
if [ -f "Trainers/rtx3090_sft/requirements.txt" ]; then
    pip install -r Trainers/rtx3090_sft/requirements.txt
else
    echo "Error: Trainers/rtx3090_sft/requirements.txt not found"
    exit 1
fi

# Install Unsloth and Xformers
echo "Installing Unsloth and Xformers..."
pip install --no-deps unsloth==2024.9
pip install --no-deps xformers==0.0.27.post2

echo "=========================================="
echo "Setup complete!"
echo "You can now run the CLI using: ./run.sh"
echo "=========================================="
