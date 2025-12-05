#!/bin/bash
# Quick start script for self-play data generation
#
# Usage:
#   ./Tools/run_selfplay.sh                    # Interactive mode
#   ./Tools/run_selfplay.sh --quick            # Generate 100 examples (test)
#   ./Tools/run_selfplay.sh --standard         # Generate 1000 examples
#   ./Tools/run_selfplay.sh --large            # Generate 5000 examples

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Self-Play Synthetic Data Generator${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✓${NC} Loaded .env file"
else
    echo -e "${YELLOW}⚠${NC} No .env file found (LM Studio host may need manual config)"
fi

# Default values
MODEL=""
PROMPT_SET="Evaluator/prompts/tool_prompts.json"
NUM_EXAMPLES=1000
TEMPERATURE=0.7
NUM_VARIATIONS=3
OUTPUT_DIR="Datasets"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXECUTE_MCP=false

# Parse arguments
case "$1" in
    --quick)
        NUM_EXAMPLES=100
        echo -e "${GREEN}Quick mode:${NC} 100 examples"
        ;;
    --standard)
        NUM_EXAMPLES=1000
        echo -e "${GREEN}Standard mode:${NC} 1000 examples"
        ;;
    --large)
        NUM_EXAMPLES=5000
        echo -e "${GREEN}Large mode:${NC} 5000 examples"
        ;;
    --help|-h)
        echo "Usage: $0 [--quick|--standard|--large]"
        echo ""
        echo "Options:"
        echo "  --quick      Generate 100 examples (fast test)"
        echo "  --standard   Generate 1000 examples (default)"
        echo "  --large      Generate 5000 examples (full dataset)"
        echo "  --help       Show this help"
        echo ""
        exit 0
        ;;
esac

# Check LM Studio connection
echo ""
echo -e "${BLUE}Checking LM Studio connection...${NC}"

if [ -n "$LMSTUDIO_HOST" ]; then
    LMSTUDIO_URL="http://${LMSTUDIO_HOST}:1234"
else
    LMSTUDIO_URL="http://localhost:1234"
fi

if curl -s "${LMSTUDIO_URL}/v1/models" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} LM Studio is accessible at ${LMSTUDIO_URL}"
else
    echo -e "${RED}✗${NC} Cannot connect to LM Studio at ${LMSTUDIO_URL}"
    echo ""
    echo "Please ensure:"
    echo "  1. LM Studio is running"
    echo "  2. Server is started (Developer > Server > Start Server)"
    echo "  3. If using WSL, 'Serve on Local Network' is enabled"
    echo "  4. LMSTUDIO_HOST is set in .env (if using WSL)"
    echo ""
    exit 1
fi

# Get available models
echo ""
echo -e "${BLUE}Available models in LM Studio:${NC}"
MODELS=$(curl -s "${LMSTUDIO_URL}/v1/models" | jq -r '.data[].id' 2>/dev/null || echo "")

if [ -z "$MODELS" ]; then
    echo -e "${RED}✗${NC} Could not retrieve models from LM Studio"
    exit 1
fi

# Display models with numbers
i=1
declare -a MODEL_ARRAY
while IFS= read -r model; do
    echo "  ${i}. ${model}"
    MODEL_ARRAY[$i]=$model
    i=$((i+1))
done <<< "$MODELS"

# Prompt for model selection
echo ""
read -p "Select model number [1-$((i-1))]: " MODEL_NUM

if [ -z "$MODEL_NUM" ] || [ "$MODEL_NUM" -lt 1 ] || [ "$MODEL_NUM" -ge "$i" ]; then
    echo -e "${RED}Invalid selection${NC}"
    exit 1
fi

MODEL="${MODEL_ARRAY[$MODEL_NUM]}"
echo -e "${GREEN}Selected:${NC} $MODEL"

# Prompt for temperature (optional)
echo ""
echo -e "${BLUE}Temperature controls response diversity:${NC}"
echo "  0.3-0.5: Mostly correct, less diversity"
echo "  0.6-0.8: Balanced mix (recommended)"
echo "  0.9-1.2: Maximum diversity, more errors"
echo ""
read -p "Temperature [default: 0.7]: " TEMP_INPUT

if [ -n "$TEMP_INPUT" ]; then
    TEMPERATURE=$TEMP_INPUT
fi

# Confirm configuration
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo "  Model:            $MODEL"
echo "  Prompt set:       $PROMPT_SET"
echo "  Output:           ${OUTPUT_DIR}/syngen_selfplay_${TIMESTAMP}.jsonl"
echo "  Num examples:     $NUM_EXAMPLES"
echo "  Temperature:      $TEMPERATURE"
echo "  Variations:       $NUM_VARIATIONS"
echo "  Execute MCP:      $EXECUTE_MCP"
echo ""
read -p "Continue? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Run generator
echo ""
echo -e "${GREEN}Starting generation...${NC}"
echo ""

OUTPUT_FILE="${OUTPUT_DIR}/syngen_selfplay_${TIMESTAMP}.jsonl"

python Tools/selfplay_generator.py \
    --model "$MODEL" \
    --prompt-set "$PROMPT_SET" \
    --output "$OUTPUT_FILE" \
    --num-examples "$NUM_EXAMPLES" \
    --temperature "$TEMPERATURE" \
    --num-variations "$NUM_VARIATIONS" \
    ${LMSTUDIO_HOST:+--lmstudio-host "$LMSTUDIO_HOST"}

# Check if generation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Generation complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Output: $OUTPUT_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Validate dataset:"
    echo "     python tools/validate_syngen.py $OUTPUT_FILE"
    echo ""
    echo "  2. Train with KTO:"
    echo "     cd Trainers/rtx3090_kto"
    echo "     python train_kto.py --model-size 7b --local-file ../../$OUTPUT_FILE"
    echo ""
    echo "  3. Evaluate results:"
    echo "     python -m Evaluator.cli --model your-model --prompt-set $PROMPT_SET"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Generation failed${NC}"
    exit 1
fi
