# Setup llama.cpp for GGUF conversion on Windows
# This script clones and builds llama.cpp with minimal dependencies

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "llama.cpp Setup for Windows" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$LlamaCppDir = "F:\Code\llama.cpp"

# Check if already exists
if (Test-Path $LlamaCppDir) {
    Write-Host "[OK] llama.cpp already exists at: $LlamaCppDir" -ForegroundColor Green
    
    # Check if build exists
    if (Test-Path "$LlamaCppDir\build\bin\Release\llama-quantize.exe") {
        Write-Host "[OK] llama-quantize.exe already built" -ForegroundColor Green
        Write-Host ""
        Write-Host "Setup complete! llama.cpp is ready to use." -ForegroundColor Green
        exit 0
    } else {
        Write-Host "[WARNING] llama.cpp exists but not built. Rebuilding..." -ForegroundColor Yellow
    }
} else {
    # Clone llama.cpp
    Write-Host "Cloning llama.cpp repository..." -ForegroundColor Yellow
    Set-Location "F:\Code"
    git clone https://github.com/ggerganov/llama.cpp
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to clone llama.cpp" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[OK] Repository cloned" -ForegroundColor Green
}

# Build llama.cpp
Write-Host ""
Write-Host "Building llama.cpp (this may take 5-10 minutes)..." -ForegroundColor Yellow
Write-Host "  Using CPU-only build (no CURL, no GPU)" -ForegroundColor Gray
Write-Host ""

Set-Location $LlamaCppDir

# Create build directory
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
New-Item -ItemType Directory -Path "build" | Out-Null

Set-Location "build"

# Configure with CMake (disable CURL and GPU to minimize dependencies)
Write-Host "Configuring with CMake..." -ForegroundColor Cyan
cmake .. -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] CMake configuration failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Configuration complete" -ForegroundColor Green
Write-Host ""

# Build (using all CPU cores)
Write-Host "Building binaries..." -ForegroundColor Cyan
$NumCores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "  Using $NumCores CPU cores" -ForegroundColor Gray

cmake --build . --config Release -j $NumCores

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Build failed" -ForegroundColor Red
    exit 1
}

# Verify build
if (Test-Path "bin\Release\llama-quantize.exe") {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "[SUCCESS] llama.cpp built successfully!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Binary location:" -ForegroundColor White
    Write-Host "  $LlamaCppDir\build\bin\Release\llama-quantize.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "You can now use GGUF creation in the upload script!" -ForegroundColor White
} else {
    Write-Host "[ERROR] llama-quantize.exe not found after build" -ForegroundColor Red
    exit 1
}
