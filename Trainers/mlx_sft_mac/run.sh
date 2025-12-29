#!/bin/bash
# MLX SFT Training Runner for Apple Silicon
# Usage: ./run.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "MLX SFT Training for Apple Silicon"
echo "======================================"

# Check Python version
python3 --version

# Check if mlx is installed
if ! python3 -c "import mlx" 2>/dev/null; then
    echo "[ERROR] MLX not installed. Run: pip install -r requirements.txt"
    exit 1
fi

# Check if mlx_lm is installed
if ! python3 -c "import mlx_lm" 2>/dev/null; then
    echo "[ERROR] mlx_lm not installed. Run: pip install mlx-lm"
    exit 1
fi

echo "[OK] Dependencies verified"
echo ""

# Run training
python3 train_sft.py "$@"
