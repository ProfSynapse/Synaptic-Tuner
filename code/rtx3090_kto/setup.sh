#!/bin/bash
# Setup script for RTX 3090 KTO Training - IMPROVED
# Run with: bash setup.sh
# For quick setup: bash setup.sh --quick
# For flash-attn: bash setup.sh --with-flash-attn

set -e  # Exit on error

# Parse arguments
QUICK_MODE=false
INSTALL_FLASH_ATTN=false
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --with-flash-attn)
            INSTALL_FLASH_ATTN=true
            shift
            ;;
    esac
done

echo "=========================================="
echo "RTX 3090 KTO Training - Setup"
echo "=========================================="
echo "Mode: $([ "$QUICK_MODE" = true ] && echo "Quick (no verification)" || echo "Full")"
echo "Flash Attention: $([ "$INSTALL_FLASH_ATTN" = true ] && echo "Yes" || echo "No")"
echo "=========================================="

# Check Python version
echo -e "\n[1/8] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "ERROR: Python 3.9+ required (3.10+ recommended)"
    exit 1
fi

echo "✓ Python version OK"

# Check NVIDIA GPU
echo -e "\n[2/8] Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    echo "Required: Driver 535+ for CUDA 12.1"
else
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

    # Check compute capability
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    if [ "$(echo "$compute_cap < 8.6" | bc -l)" -eq 1 ]; then
        echo "WARNING: GPU compute capability $compute_cap (RTX 3090 is 8.6)"
    fi

    echo "✓ GPU detected"
fi

# Create virtual environment
echo -e "\n[3/8] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
    read -p "Recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ Virtual environment recreated"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n[4/8] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo -e "\n[5/8] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel -q
echo "✓ pip upgraded to $(pip --version | awk '{print $2}')"

# Install PyTorch and core dependencies
echo -e "\n[6/8] Installing PyTorch and core dependencies..."
echo "This may take 2-3 minutes..."
pip install -r requirements.txt
echo "✓ Core dependencies installed"

# Verify PyTorch CUDA
echo -e "\n[7/8] Verifying PyTorch CUDA support..."
if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    python -c "import torch; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
    python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
else
    echo "✗ ERROR: PyTorch CUDA not available!"
    echo "Troubleshooting:"
    echo "  1. Check NVIDIA drivers: nvidia-smi"
    echo "  2. Verify CUDA 12.1 installation"
    echo "  3. Reinstall PyTorch: pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

# Install Unsloth (special installation)
echo -e "\n[8/8] Installing Unsloth..."
echo "Installing Unsloth for PyTorch 2.4.1 + CUDA 12.1 + Ampere..."
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
echo "✓ Unsloth installed"

# Optional: Install Flash Attention
if [ "$INSTALL_FLASH_ATTN" = true ]; then
    echo -e "\n[OPTIONAL] Installing Flash Attention..."
    echo "This will take 5-10 minutes to compile..."

    # Check RAM before compilation
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_ram" -lt 32 ]; then
        echo "WARNING: Less than 32GB RAM detected. Using MAX_JOBS=2 to limit compilation."
        MAX_JOBS=2 pip install flash-attn==2.5.9 --no-build-isolation
    else
        MAX_JOBS=4 pip install flash-attn==2.5.9 --no-build-isolation
    fi

    echo "✓ Flash Attention installed"
fi

# Final verification (unless quick mode)
if [ "$QUICK_MODE" = false ]; then
    echo -e "\n=========================================="
    echo "Running Installation Tests..."
    echo "=========================================="
    python test_installation.py
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Installation Summary:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import trl; print(f'  TRL: {trl.__version__}')"
python -c "from unsloth import FastLanguageModel; print('  Unsloth: ✓ Installed')"
if [ "$INSTALL_FLASH_ATTN" = true ]; then
    python -c "import flash_attn; print(f'  Flash Attention: {flash_attn.__version__}')" 2>/dev/null || echo "  Flash Attention: ✗ Not installed"
fi

echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Test with dry run: python train_kto.py --model-size 7b --dry-run"
echo "  3. Start training: python train_kto.py --model-size 7b"
echo ""
echo "For help: python train_kto.py --help"
echo ""
