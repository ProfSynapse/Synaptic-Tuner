# Quick start script for self-play data generation (PowerShell)
#
# Usage:
#   .\Tools\run_selfplay.ps1                    # Interactive mode
#   .\Tools\run_selfplay.ps1 -Quick             # Generate 100 examples (test)
#   .\Tools\run_selfplay.ps1 -Standard          # Generate 1000 examples
#   .\Tools\run_selfplay.ps1 -Large             # Generate 5000 examples

param(
    [switch]$Quick,
    [switch]$Standard,
    [switch]$Large,
    [switch]$Help
)

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

if ($Help) {
    Write-Output "Usage: .\Tools\run_selfplay.ps1 [-Quick|-Standard|-Large]"
    Write-Output ""
    Write-Output "Options:"
    Write-Output "  -Quick      Generate 100 examples (fast test)"
    Write-Output "  -Standard   Generate 1000 examples (default)"
    Write-Output "  -Large      Generate 5000 examples (full dataset)"
    Write-Output "  -Help       Show this help"
    Write-Output ""
    exit 0
}

Write-ColorOutput Blue "═══════════════════════════════════════════════════════════"
Write-ColorOutput Blue "   Self-Play Synthetic Data Generator"
Write-ColorOutput Blue "═══════════════════════════════════════════════════════════"
Write-Output ""

# Load environment variables from .env
$envFile = ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, [System.EnvironmentVariableTarget]::Process)
        }
    }
    Write-ColorOutput Green "✓ Loaded .env file"
} else {
    Write-ColorOutput Yellow "⚠ No .env file found (LM Studio host may need manual config)"
}

# Default values
$MODEL = ""
$PROMPT_SET = "Evaluator/prompts/tool_prompts.json"
$NUM_EXAMPLES = 1000
$TEMPERATURE = 0.7
$NUM_VARIATIONS = 3
$OUTPUT_DIR = "Datasets"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$EXECUTE_MCP = $false

# Parse arguments
if ($Quick) {
    $NUM_EXAMPLES = 100
    Write-ColorOutput Green "Quick mode: 100 examples"
} elseif ($Standard) {
    $NUM_EXAMPLES = 1000
    Write-ColorOutput Green "Standard mode: 1000 examples"
} elseif ($Large) {
    $NUM_EXAMPLES = 5000
    Write-ColorOutput Green "Large mode: 5000 examples"
}

# Check LM Studio connection
Write-Output ""
Write-ColorOutput Blue "Checking LM Studio connection..."

$LMSTUDIO_HOST = [System.Environment]::GetEnvironmentVariable("LMSTUDIO_HOST")
if ($LMSTUDIO_HOST) {
    $LMSTUDIO_URL = "http://${LMSTUDIO_HOST}:1234"
} else {
    $LMSTUDIO_URL = "http://localhost:1234"
}

try {
    $null = Invoke-RestMethod -Uri "$LMSTUDIO_URL/v1/models" -Method Get -TimeoutSec 2
    Write-ColorOutput Green "✓ LM Studio is accessible at $LMSTUDIO_URL"
} catch {
    Write-ColorOutput Red "✗ Cannot connect to LM Studio at $LMSTUDIO_URL"
    Write-Output ""
    Write-Output "Please ensure:"
    Write-Output "  1. LM Studio is running"
    Write-Output "  2. Server is started (Developer > Server > Start Server)"
    Write-Output "  3. If using WSL, 'Serve on Local Network' is enabled"
    Write-Output "  4. LMSTUDIO_HOST is set in .env (if using WSL)"
    Write-Output ""
    exit 1
}

# Get available models
Write-Output ""
Write-ColorOutput Blue "Available models in LM Studio:"

try {
    $response = Invoke-RestMethod -Uri "$LMSTUDIO_URL/v1/models" -Method Get
    $models = $response.data | ForEach-Object { $_.id }

    if ($models.Count -eq 0) {
        Write-ColorOutput Red "✗ No models loaded in LM Studio"
        exit 1
    }

    # Display models with numbers
    for ($i = 0; $i -lt $models.Count; $i++) {
        Write-Output "  $($i+1). $($models[$i])"
    }

    # Prompt for model selection
    Write-Output ""
    $MODEL_NUM = Read-Host "Select model number [1-$($models.Count)]"

    $MODEL_NUM = [int]$MODEL_NUM
    if ($MODEL_NUM -lt 1 -or $MODEL_NUM -gt $models.Count) {
        Write-ColorOutput Red "Invalid selection"
        exit 1
    }

    $MODEL = $models[$MODEL_NUM - 1]
    Write-ColorOutput Green "Selected: $MODEL"

} catch {
    Write-ColorOutput Red "✗ Could not retrieve models from LM Studio"
    exit 1
}

# Prompt for temperature
Write-Output ""
Write-ColorOutput Blue "Temperature controls response diversity:"
Write-Output "  0.3-0.5: Mostly correct, less diversity"
Write-Output "  0.6-0.8: Balanced mix (recommended)"
Write-Output "  0.9-1.2: Maximum diversity, more errors"
Write-Output ""
$TEMP_INPUT = Read-Host "Temperature [default: 0.7]"

if ($TEMP_INPUT) {
    $TEMPERATURE = [double]$TEMP_INPUT
}

# Confirm configuration
Write-Output ""
Write-ColorOutput Blue "═══════════════════════════════════════════════════════════"
Write-ColorOutput Blue "Configuration:"
Write-ColorOutput Blue "═══════════════════════════════════════════════════════════"
Write-Output "  Model:            $MODEL"
Write-Output "  Prompt set:       $PROMPT_SET"
Write-Output "  Output:           ${OUTPUT_DIR}/syngen_selfplay_${TIMESTAMP}.jsonl"
Write-Output "  Num examples:     $NUM_EXAMPLES"
Write-Output "  Temperature:      $TEMPERATURE"
Write-Output "  Variations:       $NUM_VARIATIONS"
Write-Output "  Execute MCP:      $EXECUTE_MCP"
Write-Output ""
$CONFIRM = Read-Host "Continue? (y/n)"

if ($CONFIRM -ne "y" -and $CONFIRM -ne "Y") {
    Write-Output "Aborted."
    exit 0
}

# Create output directory if needed
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

# Run generator
Write-Output ""
Write-ColorOutput Green "Starting generation..."
Write-Output ""

$OUTPUT_FILE = "${OUTPUT_DIR}/syngen_selfplay_${TIMESTAMP}.jsonl"

$args = @(
    "Tools/selfplay_generator.py",
    "--model", $MODEL,
    "--prompt-set", $PROMPT_SET,
    "--output", $OUTPUT_FILE,
    "--num-examples", $NUM_EXAMPLES,
    "--temperature", $TEMPERATURE,
    "--num-variations", $NUM_VARIATIONS
)

if ($LMSTUDIO_HOST) {
    $args += "--lmstudio-host", $LMSTUDIO_HOST
}

python @args

# Check if generation succeeded
if ($LASTEXITCODE -eq 0) {
    Write-Output ""
    Write-ColorOutput Green "═══════════════════════════════════════════════════════════"
    Write-ColorOutput Green "✓ Generation complete!"
    Write-ColorOutput Green "═══════════════════════════════════════════════════════════"
    Write-Output ""
    Write-Output "Output: $OUTPUT_FILE"
    Write-Output ""
    Write-Output "Next steps:"
    Write-Output "  1. Validate dataset:"
    Write-Output "     python tools/validate_syngen.py $OUTPUT_FILE"
    Write-Output ""
    Write-Output "  2. Train with KTO:"
    Write-Output "     cd Trainers/rtx3090_kto"
    Write-Output "     python train_kto.py --model-size 7b --local-file ../../$OUTPUT_FILE"
    Write-Output ""
    Write-Output "  3. Evaluate results:"
    Write-Output "     python -m Evaluator.cli --model your-model --prompt-set $PROMPT_SET"
    Write-Output ""
} else {
    Write-Output ""
    Write-ColorOutput Red "✗ Generation failed"
    exit 1
}
