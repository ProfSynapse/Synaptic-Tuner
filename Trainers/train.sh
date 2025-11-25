#!/bin/bash
# ============================================================================
# Interactive Training CLI for WSL/Linux
# Unified script for SFT, KTO, or SFT→KTO pipeline training
# ============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Display Header
# ============================================================================
clear
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║            Toolset-Training Interactive CLI                   ║${NC}"
echo -e "${CYAN}║         SFT & KTO Training for RTX 3090 / 4090                ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Main Menu
# ============================================================================
echo -e "${YELLOW}Select Training Mode:${NC}"
echo ""
echo "  1) SFT Only           - Supervised Fine-Tuning (teaches tool-calling)"
echo "  2) KTO Only           - Preference Learning (refines existing model)"
echo "  3) SFT → KTO Pipeline - Full training pipeline (recommended)"
echo "  4) Exit"
echo ""
read -p "Enter choice [1-4]: " CHOICE

case $CHOICE in
    1)
        MODE="sft"
        ;;
    2)
        MODE="kto"
        ;;
    3)
        MODE="pipeline"
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""

# ============================================================================
# W&B Configuration
# ============================================================================
echo -e "${YELLOW}Weights & Biases Logging:${NC}"
echo ""
read -p "Enable W&B logging? (y/n) [n]: " WANDB_ENABLE
WANDB_ENABLE=${WANDB_ENABLE:-n}

WANDB_FLAG=""
WANDB_PROJECT=""

if [[ "$WANDB_ENABLE" == "y" || "$WANDB_ENABLE" == "Y" ]]; then
    read -p "W&B project name [toolset-training]: " WANDB_PROJECT_NAME
    WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-toolset-training}

    WANDB_FLAG="--wandb"
    WANDB_PROJECT="--wandb-project $WANDB_PROJECT_NAME"
fi

echo ""

# ============================================================================
# Configuration Summary
# ============================================================================
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Configuration Summary${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Training Mode:    $([ "$MODE" == "sft" ] && echo "SFT Only" || [ "$MODE" == "kto" ] && echo "KTO Only" || echo "SFT → KTO Pipeline")"
echo "  W&B Logging:      $([ -n "$WANDB_FLAG" ] && echo "Enabled ($WANDB_PROJECT_NAME)" || echo "Disabled")"
echo ""

if [[ "$MODE" == "sft" || "$MODE" == "pipeline" ]]; then
    echo "  SFT Configuration:"
    echo "    - Config: rtx3090_sft/configs/config.yaml"
    echo "    - Model: mistral-7b-instruct-v0.3-bnb-4bit"
    echo "    - Dataset: syngen_tools_sft_11.22.25.jsonl"
    echo "    - Learning rate: 2e-4, Epochs: 3"
    echo ""
fi

if [[ "$MODE" == "kto" || "$MODE" == "pipeline" ]]; then
    echo "  KTO Configuration:"
    echo "    - Config: rtx3090_kto/configs/config.yaml"
    if [[ "$MODE" == "kto" ]]; then
        echo "    - Model: (from config.yaml)"
    else
        echo "    - Model: SFT output (auto-detected)"
    fi
    echo "    - Dataset: syngen_tools_11.18.25.jsonl"
    echo "    - Learning rate: 2e-7, Epochs: 1"
    echo ""
fi

read -p "Continue with this configuration? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""

# ============================================================================
# Execute Training
# ============================================================================

if [[ "$MODE" == "sft" ]]; then
    # ========================================================================
    # SFT Only
    # ========================================================================
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}SFT Training (Supervised Fine-Tuning)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    cd rtx3090_sft
    python train_sft.py $WANDB_FLAG $WANDB_PROJECT

    # Find output
    SFT_OUTPUT=$(ls -td sft_output_rtx3090/*/ 2>/dev/null | head -1)

    echo ""
    echo -e "${GREEN}✓ SFT Training Complete!${NC}"
    if [[ -n "$SFT_OUTPUT" ]]; then
        echo "  Output: $SFT_OUTPUT"
    fi

elif [[ "$MODE" == "kto" ]]; then
    # ========================================================================
    # KTO Only
    # ========================================================================
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}KTO Training (Preference Learning)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Note: KTO is designed for refinement. Ensure you're using an SFT-trained model.${NC}"
    echo ""

    cd rtx3090_kto
    python train_kto.py $WANDB_FLAG $WANDB_PROJECT

    # Find output
    KTO_OUTPUT=$(ls -td kto_output_rtx3090/*/ 2>/dev/null | head -1)

    echo ""
    echo -e "${GREEN}✓ KTO Training Complete!${NC}"
    if [[ -n "$KTO_OUTPUT" ]]; then
        echo "  Output: $KTO_OUTPUT"
    fi

elif [[ "$MODE" == "pipeline" ]]; then
    # ========================================================================
    # SFT → KTO Pipeline
    # ========================================================================
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}Phase 1: SFT Training (Teaching Tool-Calling Syntax)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    cd rtx3090_sft
    python train_sft.py $WANDB_FLAG $WANDB_PROJECT

    # Find SFT output
    SFT_OUTPUT=$(ls -td sft_output_rtx3090/*/ | head -1)
    SFT_FINAL_MODEL="${SFT_OUTPUT}final_model"

    echo ""
    echo -e "${GREEN}✓ Phase 1 Complete!${NC}"
    echo "  Output: $SFT_OUTPUT"
    echo "  Model: $SFT_FINAL_MODEL"
    echo ""

    # Ask to continue to KTO
    read -p "Continue to KTO refinement? (y/n): " CONTINUE_KTO
    if [[ "$CONTINUE_KTO" != "y" && "$CONTINUE_KTO" != "Y" ]]; then
        echo "Pipeline stopped after SFT."
        exit 0
    fi

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}Phase 2: KTO Training (Preference Learning Refinement)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    cd ../rtx3090_kto

    # Get absolute path to SFT final model
    SFT_FINAL_MODEL_ABS="$(cd ../rtx3090_sft && realpath "$SFT_FINAL_MODEL")"

    echo "  Updating KTO config to use SFT output model..."
    echo "  New model_name: $SFT_FINAL_MODEL_ABS"

    # Update KTO config with SFT model path
    python3 - <<EOF
import yaml
from pathlib import Path

config_path = Path("configs/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['model']['model_name'] = "$SFT_FINAL_MODEL_ABS"

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

print(f"✓ Updated KTO config")
EOF

    echo ""
    python train_kto.py $WANDB_FLAG $WANDB_PROJECT

    # Find KTO output
    KTO_OUTPUT=$(ls -td kto_output_rtx3090/*/ 2>/dev/null | head -1)

    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ Complete SFT→KTO Pipeline Finished Successfully!          ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Training Outputs:"
    echo "  SFT Output:  $SFT_OUTPUT"
    echo "  KTO Output:  $KTO_OUTPUT"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Next Steps${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "1. Test the model:"
echo "   cd Evaluator"
if [[ "$MODE" == "kto" || "$MODE" == "pipeline" ]]; then
    echo "   python cli.py --model ${KTO_OUTPUT}final_model --prompt-set prompts/baseline.json"
else
    echo "   python cli.py --model ${SFT_OUTPUT}final_model --prompt-set prompts/baseline.json"
fi
echo ""
echo "2. Upload to HuggingFace:"
if [[ "$MODE" == "kto" || "$MODE" == "pipeline" ]]; then
    echo "   cd Trainers/rtx3090_kto"
else
    echo "   cd Trainers/rtx3090_sft"
fi
echo "   ./upload_model.sh"
echo ""
echo "3. Create GGUF quantizations:"
echo "   Select 'Create GGUF' option during upload"
echo ""
