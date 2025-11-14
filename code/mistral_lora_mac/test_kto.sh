#!/bin/bash

# Test script for KTO training
# This runs a lightweight test to validate the KTO implementation

echo "=================================================="
echo "  KTO Training Test for MLX"
echo "=================================================="
echo ""
echo "Test Configuration:"
echo "  - Dataset: 100 examples (50 desirable, 50 undesirable)"
echo "  - Max steps: 50"
echo "  - Batch size: 2 (effective: 4 with accumulation)"
echo "  - LoRA rank: 8 (lightweight)"
echo "  - Sequence length: 512 tokens"
echo ""
echo "This test should complete in 5-10 minutes"
echo "=================================================="
echo ""

# Clean up previous test runs
rm -rf test_checkpoints test_outputs test_logs
mkdir -p test_checkpoints test_outputs test_logs

# Run training
python main.py --config config/config_test_kto.yaml

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "  ✓ KTO Training Test PASSED"
    echo "=================================================="
    echo ""
    echo "Check outputs:"
    echo "  - Logs: test_logs/training.log"
    echo "  - Checkpoints: test_checkpoints/"
    echo "  - Final model: test_outputs/final_model/"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "  ✗ KTO Training Test FAILED"
    echo "=================================================="
    echo ""
    echo "Check test_logs/training.log for errors"
    exit 1
fi
