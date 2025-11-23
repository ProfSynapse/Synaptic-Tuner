#!/bin/bash
# Complete SFT → KTO Training Pipeline
# This script chains supervised fine-tuning with preference learning
# Uses YAML configs from each trainer's configs/ directory
#
# Usage:
#   ./train_sft_to_kto_pipeline.sh [--wandb] [--wandb-project PROJECT_NAME]

set -e  # Exit on error

# Default parameters
WANDB_FLAG=""
WANDB_PROJECT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --wandb)
      WANDB_FLAG="--wandb"
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="--wandb-project $2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--wandb] [--wandb-project PROJECT_NAME]"
      exit 1
      ;;
  esac
done

# Get repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         SFT → KTO Training Pipeline                           ║"
echo "║              Using YAML Configurations                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  SFT config: Trainers/rtx3090_sft/configs/config.yaml"
echo "  KTO config: Trainers/rtx3090_kto/configs/config.yaml"
echo "  W&B logging: $([ -n "$WANDB_FLAG" ] && echo 'Enabled' || echo 'Disabled')"
[ -n "$WANDB_PROJECT" ] && echo "  W&B project: ${WANDB_PROJECT#--wandb-project }"
echo ""
read -p "Press Enter to start training or Ctrl+C to cancel..."

# ============================================================================
# PHASE 1: SFT (Initial Training)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 1: SFT Training (Teaching Tool-Calling Syntax)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Config: configs/config.yaml"
echo "  Model: mistral-7b-instruct-v0.3-bnb-4bit (from YAML)"
echo "  Dataset: syngen_tools_sft_11.22.25.jsonl (from YAML)"
echo "  Learning rate: 2e-4 (from YAML)"
echo "  Epochs: 3 (from YAML)"
echo ""

cd Trainers/rtx3090_sft

# Run SFT with default YAML config (no CLI overrides)
python train_sft.py $WANDB_FLAG $WANDB_PROJECT

# Find the most recent SFT output directory
SFT_OUTPUT=$(ls -td sft_output_rtx3090/*/ | head -1)
SFT_FINAL_MODEL="${SFT_OUTPUT}final_model"

echo ""
echo "✓ Phase 1 complete!"
echo "  Output: $SFT_OUTPUT"
echo "  Model: $SFT_FINAL_MODEL"

# ============================================================================
# PHASE 2: KTO (Refinement)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Phase 2: KTO Training (Preference Learning Refinement)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Config: configs/config.yaml (will be temporarily modified)"
echo "  Base model: $SFT_FINAL_MODEL (SFT output)"
echo "  Dataset: syngen_tools_11.18.25.jsonl (from YAML)"
echo "  Learning rate: 2e-7 (from YAML)"
echo "  Epochs: 1 (from YAML)"
echo ""
read -p "Press Enter to start KTO refinement or Ctrl+C to skip..."

cd ../rtx3090_kto

# Backup original KTO config
KTO_CONFIG="configs/config.yaml"
KTO_CONFIG_BACKUP="configs/config.yaml.backup"
cp "$KTO_CONFIG" "$KTO_CONFIG_BACKUP"

# Get the absolute path to SFT final model
SFT_FINAL_MODEL_ABS="$(cd ../../Trainers/rtx3090_sft && realpath "$SFT_FINAL_MODEL")"

echo "  Updating KTO config to use SFT output model..."
echo "  Original model_name: $(grep 'model_name:' "$KTO_CONFIG" | head -1)"
echo "  New model_name: $SFT_FINAL_MODEL_ABS"

# Update the model_name in KTO config using Python (handles YAML formatting)
python3 - <<EOF
import yaml
from pathlib import Path

config_path = Path("$KTO_CONFIG")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update model_name to point to SFT output
config['model']['model_name'] = "$SFT_FINAL_MODEL_ABS"

# Write back with preserved formatting
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

print(f"✓ Updated KTO config: model.model_name = $SFT_FINAL_MODEL_ABS")
EOF

echo ""
echo "  Running KTO training with updated config..."

# Run KTO with modified YAML config (no CLI overrides needed)
python train_kto.py $WANDB_FLAG $WANDB_PROJECT

# Restore original KTO config
echo ""
echo "  Restoring original KTO config..."
mv "$KTO_CONFIG_BACKUP" "$KTO_CONFIG"
echo "  ✓ Config restored"

# Find the most recent KTO output directory
KTO_OUTPUT=$(ls -td kto_output_rtx3090/*/ | head -1)

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Complete SFT→KTO Pipeline Finished Successfully!          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Training Outputs:"
echo "  SFT Output:  $SFT_OUTPUT"
echo "  KTO Output:  $KTO_OUTPUT"
echo ""
echo "Configuration Files:"
echo "  SFT used: Trainers/rtx3090_sft/configs/config.yaml"
echo "  KTO used: Trainers/rtx3090_kto/configs/config.yaml (restored)"
echo ""
echo "Next Steps:"
echo "  1. Test the model:"
echo "     cd Evaluator"
echo "     python cli.py --model ${KTO_OUTPUT}final_model --prompt-set prompts/baseline.json"
echo ""
echo "  2. Upload to HuggingFace:"
echo "     cd Trainers/rtx3090_kto"
echo "     ./upload_model.sh"
echo ""
echo "  3. Create GGUF quantizations:"
echo "     Select 'Create GGUF' during upload or run:"
echo "     python src/upload_to_hf.py ${KTO_OUTPUT}final_model username/model-name --create-gguf"
echo ""
