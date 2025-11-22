#!/bin/bash
# Interactive upload script for trained models to HuggingFace
# Supports selecting training runs and configuring upload options

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HuggingFace Model Upload (Interactive)${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Find available training runs
OUTPUT_DIR="./kto_output_rtx3090"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}Error: Output directory not found: $OUTPUT_DIR${NC}"
    echo "No training runs available."
    exit 1
fi

# Get list of training runs that have final_model directory
echo -e "${BLUE}Available training runs:${NC}"
echo

runs=()
run_paths=()
index=1

for run_dir in "$OUTPUT_DIR"/*/; do
    if [ -d "${run_dir}final_model" ]; then
        run_name=$(basename "$run_dir")
        runs+=("$run_name")
        run_paths+=("${run_dir}final_model")
        echo "  [$index] $run_name"
        index=$((index + 1))
    fi
done

if [ ${#runs[@]} -eq 0 ]; then
    echo -e "${RED}No training runs with final_model found in $OUTPUT_DIR${NC}"
    exit 1
fi

echo
read -p "Select training run number: " run_selection

# Validate selection
if ! [[ "$run_selection" =~ ^[0-9]+$ ]] || [ "$run_selection" -lt 1 ] || [ "$run_selection" -gt ${#runs[@]} ]; then
    echo -e "${RED}Invalid selection${NC}"
    exit 1
fi

# Get selected run (arrays are 0-indexed)
selected_run="${runs[$((run_selection - 1))]}"
model_path="${run_paths[$((run_selection - 1))]}"

echo
echo -e "${GREEN}✓${NC} Selected: $selected_run"
echo -e "${GREEN}✓${NC} Model path: $model_path"
echo

# Get model name
echo -e "${BLUE}Enter model details:${NC}"
read -p "Model name (without username): " model_name

if [ -z "$model_name" ]; then
    echo -e "${RED}Error: Model name required${NC}"
    exit 1
fi

# Get username (default to professorsynapse)
read -p "HuggingFace username [professorsynapse]: " username
username=${username:-professorsynapse}

repo_id="$username/$model_name"

echo
echo -e "${GREEN}✓${NC} Repository: $repo_id"
echo

# Select save method
echo -e "${BLUE}Select save method:${NC}"
echo "  [1] merged_16bit (recommended, ~14GB, full quality)"
echo "  [2] merged_4bit (~3.5GB, quantized)"
echo "  [3] lora (~320MB, adapters only)"
echo
read -p "Select [1-3]: " save_method_selection

case $save_method_selection in
    1)
        save_method="merged_16bit"
        ;;
    2)
        save_method="merged_4bit"
        ;;
    3)
        save_method="lora"
        ;;
    *)
        echo -e "${RED}Invalid selection, using merged_16bit${NC}"
        save_method="merged_16bit"
        ;;
esac

echo -e "${GREEN}✓${NC} Save method: $save_method"
echo

# Ask about GGUF creation
read -p "Create GGUF quantizations? (y/n): " create_gguf

echo
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Upload Summary${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "Training run:  $selected_run"
echo "Model path:    $model_path"
echo "Repository:    $repo_id"
echo "Save method:   $save_method"
if [[ "$create_gguf" =~ ^[Yy]$ ]]; then
    echo "GGUF:          Yes (Q4_K_M, Q5_K_M, Q8_0)"
else
    echo "GGUF:          No"
fi
echo
echo "Output will be organized in:"
echo "  $OUTPUT_DIR/$selected_run/$model_name/"
echo "    ├── $save_method/"
if [[ "$create_gguf" =~ ^[Yy]$ ]]; then
    echo "    ├── gguf/"
fi
echo "    ├── upload_manifest.json"
echo "    └── README.md"
echo -e "${YELLOW}========================================${NC}"
echo

# Check if .env exists
ENV_PATH="../../.env"
if [ ! -f "$ENV_PATH" ]; then
    ENV_PATH=".env"
    if [ ! -f "$ENV_PATH" ]; then
        echo -e "${RED}Error: .env file not found${NC}"
        echo
        echo "Create .env file with your HuggingFace token:"
        echo "  In root directory: ../../.env"
        echo "  Or locally: .env"
        echo "  Add: HF_TOKEN=hf_your_token_here"
        echo
        echo "Get token from: https://huggingface.co/settings/tokens"
        exit 1
    fi
fi

# Load environment variables
export $(grep -v '^#' "$ENV_PATH" | xargs)

echo -e "${GREEN}✓${NC} .env file found: $ENV_PATH"
echo

# Final confirmation
read -p "Proceed with upload? (y/n): " -n 1 -r
echo
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 0
fi

# Activate conda environment
echo -e "${BLUE}Activating environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./venv

# Build upload command
upload_cmd="python src/upload_to_hf.py \"$model_path\" \"$repo_id\" --save-method $save_method"

if [[ "$create_gguf" =~ ^[Yy]$ ]]; then
    upload_cmd="$upload_cmd --create-gguf"
fi

# Run upload
echo
echo -e "${BLUE}Running upload...${NC}"
echo
eval $upload_cmd

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Upload complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo "View your model at:"
echo "  https://huggingface.co/$repo_id"
echo
echo "Local artifacts saved to:"
echo "  $OUTPUT_DIR/$selected_run/$model_name/"
echo
