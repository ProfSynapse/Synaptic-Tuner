#!/bin/bash

# RUN 8: Two-Stage Learning Rate Schedule
# Simply runs training with all settings from configs/training_config.py

echo "============================================"
echo "RUN 8: TWO-STAGE LEARNING RATE SCHEDULE"
echo "============================================"
echo ""
echo "Configuration:"
echo "  - LR: 5e-7 (steps 1-50) → 2.5e-7 (steps 51+)"
echo "  - Beta: 0.2"
echo "  - Batch: 8x4 (effective 32)"
echo "  - Reduction at: Step 50"
echo ""
echo "All settings pulled from configs/training_config.py"
echo "Edit config file to tune parameters"
echo "============================================"
echo ""

# Run training with config settings (no flags needed!)
python train_kto.py --model-size 7b --max-steps 145
