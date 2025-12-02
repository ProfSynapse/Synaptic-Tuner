# Activate the latest Unsloth environment (2025.11.4) via WSL
# Usage: .\activate_unsloth_latest.ps1

param(
    [switch]$Test,
    [switch]$Shell,
    [string]$Command
)

$ErrorActionPreference = "Stop"

# WSL distribution to use
$WslDistro = "Ubuntu-22.04"

# Conda activation command
$CondaActivate = "source /home/profsynapse/miniconda3/etc/profile.d/conda.sh && conda activate unsloth_latest"

function Invoke-WslConda {
    param([string]$Cmd)
    $FullCmd = "$CondaActivate && $Cmd"
    wsl -d $WslDistro bash -c $FullCmd
}

if ($Test) {
    Write-Host "Testing unsloth_latest environment..." -ForegroundColor Cyan
    Write-Host ""

    # Use the test script to avoid escaping issues
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $TestScript = "/mnt/f/Code/Toolset-Training/Trainers/test_unsloth_env.py"

    Invoke-WslConda "python $TestScript"

    Write-Host ""
    Write-Host "Environment test complete!" -ForegroundColor Green

} elseif ($Shell) {
    Write-Host "Entering WSL shell with unsloth_latest activated..." -ForegroundColor Cyan
    wsl -d $WslDistro bash -c "$CondaActivate && bash"

} elseif ($Command) {
    Invoke-WslConda $Command

} else {
    Write-Host ""
    Write-Host "Unsloth Latest Environment" -ForegroundColor Cyan
    Write-Host "==========================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This environment is installed in WSL and should be used via WSL."
    Write-Host ""
    Write-Host "Usage from PowerShell:" -ForegroundColor Yellow
    Write-Host "  # Test the environment (checks Vision Model support)"
    Write-Host "  .\activate_unsloth_latest.ps1 -Test" -ForegroundColor White
    Write-Host ""
    Write-Host "  # Enter interactive shell"
    Write-Host "  .\activate_unsloth_latest.ps1 -Shell" -ForegroundColor White
    Write-Host ""
    Write-Host "  # Run a command"
    Write-Host "  .\activate_unsloth_latest.ps1 -Command 'python train_sft.py --model-size 7b'" -ForegroundColor White
    Write-Host ""
    Write-Host "Usage from WSL/Bash:" -ForegroundColor Yellow
    Write-Host "  source Trainers/activate_unsloth_latest.sh" -ForegroundColor White
    Write-Host ""
    Write-Host "Vision Model Support:" -ForegroundColor Yellow
    Write-Host "  Qwen-VL, LLaVA, Pixtral, PaliGemma" -ForegroundColor White
    Write-Host "  Run -Test to verify FastVisionModel is available" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To enable vision support:" -ForegroundColor Yellow
    Write-Host "  pip install --upgrade unsloth unsloth_zoo" -ForegroundColor White
    Write-Host ""
}
