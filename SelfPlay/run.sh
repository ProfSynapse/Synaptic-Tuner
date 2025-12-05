#!/bin/bash
# Run SelfPlay generation with LM Studio

# Default values
NUM_EXAMPLES=100
OUTPUT="SelfPlay/selfplay_output.jsonl"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            NUM_EXAMPLES=10
            shift
            ;;
        --standard)
            NUM_EXAMPLES=100
            shift
            ;;
        --large)
            NUM_EXAMPLES=1000
            shift
            ;;
        --num)
            NUM_EXAMPLES=$2
            shift 2
            ;;
        --output)
            OUTPUT=$2
            shift 2
            ;;
        --help)
            echo "Usage: ./SelfPlay/run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick       Generate 10 examples (test run)"
            echo "  --standard    Generate 100 examples (default)"
            echo "  --large       Generate 1000 examples"
            echo "  --num N       Generate N examples"
            echo "  --output FILE Output file path"
            echo "  --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./SelfPlay/run.sh --quick"
            echo "  ./SelfPlay/run.sh --num 500"
            echo "  ./SelfPlay/run.sh --large --output my_data.jsonl"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

echo "Starting SelfPlay generation..."
echo "Examples: $NUM_EXAMPLES"
echo "Output: $OUTPUT"
echo ""

python3 SelfPlay/run_generation.py \
    --num-examples $NUM_EXAMPLES \
    --output $OUTPUT
