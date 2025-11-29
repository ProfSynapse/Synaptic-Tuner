#!/bin/bash
# SFT model upload script (uses shared framework)
# This is a thin wrapper that sets up the environment and calls the shared CLI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/sft_output_rtx3090"

# Source conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# Activate environment
conda activate unsloth_env 2>/dev/null || {
    echo "Warning: Could not activate unsloth_env"
}

# If no arguments provided, show interactive selection
if [ $# -eq 0 ]; then
    echo "======================================"
    echo "SFT Model Upload (Shared Framework)"
    echo "======================================"
    echo ""
    echo "Available training runs:"
    echo ""

    # List available runs
    if [ -d "$OUTPUT_DIR" ]; then
        runs=($(ls -d "$OUTPUT_DIR"/*/final_model 2>/dev/null | sort -r | head -10 | xargs -I {} dirname {}))
        if [ ${#runs[@]} -eq 0 ]; then
            echo "No training runs found in $OUTPUT_DIR"
            exit 1
        fi

        for i in "${!runs[@]}"; do
            run_name=$(basename "${runs[$i]}")
            echo "  [$((i+1))] $run_name"
        done

        echo ""
        read -p "Select training run (1-${#runs[@]}): " selection

        if [[ ! "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt ${#runs[@]} ]; then
            echo "Invalid selection"
            exit 1
        fi

        SELECTED_RUN="${runs[$((selection-1))]}"
        MODEL_PATH="$SELECTED_RUN/final_model"

        echo ""
        read -p "Enter HuggingFace repo ID (username/model-name): " REPO_ID

        echo ""
        echo "Save methods:"
        echo "  [1] merged_16bit (default, ~14GB)"
        echo "  [2] merged_4bit (~3.5GB)"
        echo "  [3] lora (~320MB)"
        read -p "Select save method (1-3) [1]: " save_choice

        case "$save_choice" in
            2) SAVE_METHOD="merged_4bit" ;;
            3) SAVE_METHOD="lora" ;;
            *) SAVE_METHOD="merged_16bit" ;;
        esac

        echo ""
        read -p "Create GGUF versions? (y/N): " create_gguf

        GGUF_FLAG=""
        if [[ "$create_gguf" =~ ^[Yy]$ ]]; then
            GGUF_FLAG="--create-gguf"
        fi

        echo ""
        echo "======================================"
        echo "Upload Configuration:"
        echo "  Model: $MODEL_PATH"
        echo "  Repo: $REPO_ID"
        echo "  Method: $SAVE_METHOD"
        echo "  GGUF: ${create_gguf:-no}"
        echo "======================================"
        echo ""
        read -p "Proceed with upload? (Y/n): " confirm

        if [[ "$confirm" =~ ^[Nn]$ ]]; then
            echo "Upload cancelled"
            exit 0
        fi

        # Run upload
        python "$SCRIPT_DIR/src/upload_to_hf_new.py" \
            "$MODEL_PATH" \
            "$REPO_ID" \
            --save-method "$SAVE_METHOD" \
            $GGUF_FLAG
    else
        echo "Output directory not found: $OUTPUT_DIR"
        exit 1
    fi
else
    # Pass all arguments to the upload script
    python "$SCRIPT_DIR/src/upload_to_hf_new.py" "$@"
fi
