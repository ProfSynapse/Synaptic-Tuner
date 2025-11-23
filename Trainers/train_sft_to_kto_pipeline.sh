#!/bin/bash
# Complete SFT → KTO Training Pipeline
# This script chains supervised fine-tuning with preference learning
#
# Usage:
#   ./train_sft_to_kto_pipeline.sh [--model-size 7b] [--wandb] [--wandb-project PROJECT_NAME]

set -e  # Exit on error

# Default parameters
MODEL_SIZE="7b"
WANDB_FLAG=""
WANDB_PROJECT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-size)
      MODEL_SIZE="$2"
      shift 2
      ;;
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
      echo "Usage: $0 [--model-size 7b] [--wandb] [--wandb-project PROJECT_NAME]"
      exit 1
      ;;
  esac
done

# Get repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         SFT → KTO Training Pipeline                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Model size: $MODEL_SIZE"
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
echo "  Dataset: syngen_tools_sft_11.18.25.jsonl (2,676 examples)"
echo "  Method: Supervised fine-tuning with positive examples"
echo "  Learning rate: 2e-4 (high for initial training)"
echo "  Epochs: 3"
echo ""

cd Trainers/rtx3090_sft

./train.sh --model-size "$MODEL_SIZE" \
  --local-file ../../Datasets/syngen_tools_sft_11.18.25.jsonl \
  $WANDB_FLAG $WANDB_PROJECT

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
echo "  Base model: $SFT_FINAL_MODEL"
echo "  Dataset: syngen_tools_11.14.25.jsonl (4,649 examples)"
echo "  Method: KTO preference learning with True/False labels"
echo "  Learning rate: 2e-7 (100x lower for refinement)"
echo "  Epochs: 1"
echo ""
read -p "Press Enter to start KTO refinement or Ctrl+C to skip..."

cd ../rtx3090_kto

python train_kto.py --model-size "$MODEL_SIZE" \
  --model-name "$SFT_FINAL_MODEL" \
  --local-file ../../Datasets/syngen_tools_11.14.25.jsonl \
  $WANDB_FLAG $WANDB_PROJECT

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
