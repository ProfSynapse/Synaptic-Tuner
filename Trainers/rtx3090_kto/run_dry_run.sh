#!/bin/bash
# Script to run training dry run

set -e

echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./venv

echo "Running KTO training dry run..."
python train_kto.py --model-size 7b --dry-run
