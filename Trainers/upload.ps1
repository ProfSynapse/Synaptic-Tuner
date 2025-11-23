# ============================================================================
# Model Upload Launcher - Quick Access to SFT and KTO Upload Scripts
# ============================================================================

Clear-Host
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "            Model Upload - Select Trainer Type                            " -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Which model do you want to upload?" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1) SFT Model" -ForegroundColor White
Write-Host "     - Upload models from rtx3090_sft/sft_output_rtx3090/" -ForegroundColor Gray
Write-Host "     - Merge LoRA adapters and create GGUF versions" -ForegroundColor Gray
Write-Host ""
Write-Host "  2) KTO Model" -ForegroundColor White
Write-Host "     - Upload models from rtx3090_kto/kto_output_rtx3090/" -ForegroundColor Gray
Write-Host "     - Merge LoRA adapters and create GGUF versions" -ForegroundColor Gray
Write-Host ""
Write-Host "  3) Exit" -ForegroundColor White
Write-Host ""

$Choice = Read-Host "Enter choice [1-3]"

switch ($Choice) {
    "1" {
        Write-Host ""
        Write-Host "Launching SFT model upload..." -ForegroundColor Green
        Write-Host ""
        Set-Location "rtx3090_sft"
        & .\upload_model.ps1
    }
    "2" {
        Write-Host ""
        Write-Host "Launching KTO model upload..." -ForegroundColor Green
        Write-Host ""
        Set-Location "rtx3090_kto"
        & .\upload_model.ps1
    }
    "3" {
        Write-Host ""
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host ""
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}
