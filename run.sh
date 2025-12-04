#!/bin/bash
# Toolset-Training Unified CLI - Bash wrapper
# Usage: ./run.sh [train|upload|eval|pipeline]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    # Export all variables, ignoring comments and empty lines
    set -a
    source .env
    set +a
fi

# Standard environment
UNSLOTH_ENV="unsloth_latest"

# Source conda
CONDA_SH=""
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    CONDA_SH=~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/.conda/etc/profile.d/conda.sh ]; then
    CONDA_SH=~/.conda/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    CONDA_SH=/opt/conda/etc/profile.d/conda.sh
fi

# ============================================================================
# DEPENDENCY CHECK - Auto-install missing packages for Ministral 3 / Transformers 5
# ============================================================================
check_and_install_deps() {
    local MISSING_DEPS=()
    local NEED_INSTALL=false

    # Check for unsloth (suppress all output including Unsloth banner)
    if ! python -c "import unsloth" &>/dev/null; then
        MISSING_DEPS+=("unsloth")
        NEED_INSTALL=true
    fi

    # Check for FastVisionModel (VL support) - CRITICAL for Ministral 3 and VL models
    if ! python -c "from unsloth import FastVisionModel" &>/dev/null; then
        MISSING_DEPS+=("unsloth_zoo (Vision Model support - required for Ministral 3)")
        NEED_INSTALL=true
    fi

    # Check for xformers
    if ! python -c "import xformers" &>/dev/null; then
        MISSING_DEPS+=("xformers")
        NEED_INSTALL=true
    fi

    # Check for uv (required by Unsloth for GGUF conversion)
    if ! command -v uv &>/dev/null && ! python -c "import uv" &>/dev/null; then
        MISSING_DEPS+=("uv (required for GGUF conversion)")
        NEED_INSTALL=true
    fi

    # Check for Transformers version (4.51.0+ required for Qwen3-VL, 5.x for Ministral 3)
    # Accept either 4.5x+ or 5.x - both work for most models
    TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "0")
    if [[ ! "$TRANSFORMERS_VERSION" =~ ^(4\.(5[1-9]|[6-9][0-9])|5\.) ]]; then
        MISSING_DEPS+=("transformers >=4.51.0 (current: $TRANSFORMERS_VERSION)")
        NEED_INSTALL=true
    fi

    # Check for TRL (accept 0.15.x or 0.22.x)
    TRL_VERSION=$(python -c "import trl; print(trl.__version__)" 2>/dev/null || echo "0")
    if [[ ! "$TRL_VERSION" =~ ^0\.(15|22)\. ]]; then
        MISSING_DEPS+=("trl 0.15.x or 0.22.x (current: $TRL_VERSION)")
        NEED_INSTALL=true
    fi

    if [ "$NEED_INSTALL" = true ]; then
        echo ""
        echo "⚠ Missing or outdated dependencies detected:"
        for dep in "${MISSING_DEPS[@]}"; do
            echo "  - $dep"
        done
        echo ""

        # Check if we're in an interactive terminal
        if [ -t 0 ]; then
            # Interactive - ask user
            read -p "Install/update dependencies for Ministral 3 support? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                DO_INSTALL=true
            else
                DO_INSTALL=false
            fi
        else
            # Non-interactive (e.g., from PowerShell via WSL) - auto-install
            echo "Non-interactive mode detected - auto-installing dependencies..."
            DO_INSTALL=true
        fi

        if [ "$DO_INSTALL" = true ]; then
            echo "Installing dependencies for Ministral 3 / Transformers 5 (this may take 2-3 minutes)..."
            echo ""

            # Install Transformers 5 from special branch (required for Ministral 3)
            echo "[1/6] Installing Transformers 5 (Ministral 3 branch)..."
            pip install git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce -q

            # Install TRL 0.22.2 (compatible with Transformers 5 + Unsloth)
            echo "[2/6] Installing TRL 0.22.2..."
            pip install --no-deps trl==0.22.2 -q

            # Install Unsloth (with --no-deps to avoid version conflicts)
            echo "[3/6] Installing Unsloth (latest)..."
            pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo -q

            # Install xformers
            echo "[4/6] Installing xformers..."
            pip install --upgrade xformers -q

            # Install uv (required by Unsloth for GGUF conversion)
            echo "[5/6] Installing uv (for GGUF conversion)..."
            pip install --upgrade uv -q

            # Install gguf from llama.cpp source (for Ministral 3 GGUF conversion)
            echo "[6/6] Installing gguf from llama.cpp source..."
            if [ -d "Trainers/llama.cpp" ]; then
                pip install -e Trainers/llama.cpp -q
                echo "  ✓ gguf installed from llama.cpp source"
            else
                echo "  ⚠ Warning: llama.cpp directory not found"
            fi

            echo ""

            # Verify installation
            if python -c "from unsloth import FastVisionModel" 2>/dev/null; then
                echo "✓ Dependencies installed successfully"
                echo "✓ FastVisionModel available (Ministral 3 ready)"
                python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
                python -c "import trl; print(f'✓ TRL: {trl.__version__}')"
            else
                echo "⚠ FastVisionModel still not available after install"
                echo "  Try running: ./setup_env.sh"
                exit 1
            fi
        else
            echo "⚠ Skipping dependency installation"
            echo "  Ministral 3 and VL model operations may fail"
        fi
        echo ""
    fi
}

if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH" 2>/dev/null
    if conda env list 2>/dev/null | grep -q "$UNSLOTH_ENV"; then
        conda activate "$UNSLOTH_ENV" 2>/dev/null

        # Animated startup with progress bar using Python/Rich
        set +e
        python -c "
import sys
import os
import contextlib
from time import sleep

# Ensure current directory is in path for Trainers import
sys.path.append(os.getcwd())

# Exit codes
EXIT_SUCCESS = 0
EXIT_MISSING_DEPS = 100
EXIT_FAILURE = 1

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.align import Align
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich.text import Text
    from Trainers.shared.ui.theme import get_animated_logo_frame, TAGLINE, COLORS

    console = Console()

    # Phase 1: Quick logo animation
    # Use Live display to animate in-place instead of clearing screen
    with Live(console=console, refresh_per_second=10, transient=False) as live:
        for i in range(8):
            frame_text = Text.from_markup(get_animated_logo_frame(i))
            tagline_align = Align.center(TAGLINE)
            live.update(Group(frame_text, tagline_align))
            sleep(0.1)

    # Phase 2: Real system checks with progress bar
    console.print()

    def run_check(name, import_cmd):
        # Run import in subprocess to avoid blocking the animation thread (GIL)
        # and to ensure we verify the environment state accurately.
        import subprocess
        try:
            subprocess.run(
                [sys.executable, \"-c\", import_cmd],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            return False

    checks = [
        ('Initializing neural pathways (Unsloth)...', 'import unsloth'),
        ('Loading model architectures (Vision)...', 'from unsloth import FastVisionModel'),
        ('Calibrating training loops (TRL)...', 'import trl'),
        ('Establishing GPU connection (xformers)...', 'import xformers'),
        ('Preparing interface (Transformers)...', 'import transformers'),
        ('Configuring GGUF converter (llama.cpp)...', 'from gguf.vocab import MistralTokenizerType'),
    ]

    missing_deps = False

    # Define colors to avoid f-string quoting issues
    purple = \"#93278F\"
    aqua = \"#00A99D\"

    with Progress(
        SpinnerColumn('dots', style=f\"bold {purple}\"),
        TextColumn('[bold cyan]{task.description}'),
        BarColumn(bar_width=30, style=purple, complete_style=aqua),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task('Starting...', total=len(checks))
        
        for desc, cmd in checks:
            progress.update(task, description=desc)
            
            # Artificial delay for "satisfying" animation feel (user requested "fun")
            # This also ensures the user has time to read the steps
            sleep(0.4)
            
            if not run_check(desc, cmd):
                missing_deps = True
                break
                
            progress.advance(task)

    if missing_deps:
        sys.exit(EXIT_MISSING_DEPS)

    # Final ready message
    ready = Text('✓ SYNAPTIC TUNER ready', style=f\"bold {aqua}\")
    console.print(Align.center(ready))
    console.print()
    sys.exit(EXIT_SUCCESS)

except ImportError:
    # Rich not installed or other error, fallback silently
    print('  SYNAPTIC TUNER ready')
    sys.exit(EXIT_SUCCESS)
except Exception as e:
    # Catch-all for other errors to prevent scary shell warnings
    # We print the error to stderr for debugging but exit success to allow CLI to try loading
    sys.stderr.write(f'Startup animation error: {e}\\n')
    sys.exit(EXIT_SUCCESS)
"

        EXIT_CODE=$?
        set -e
        
        if [ $EXIT_CODE -eq 100 ]; then
            echo "⚠ Dependencies missing or outdated. Starting installer..."
            check_and_install_deps
        elif [ $EXIT_CODE -ne 0 ]; then
            # This path should rarely be hit now due to the catch-all above
            echo "⚠ Startup check failed (Code $EXIT_CODE). Proceeding with caution..."
        fi
    else
        echo "⚠ Environment $UNSLOTH_ENV not found."

        read -p "Would you like to run setup now? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            bash setup_env.sh
            source "$CONDA_SH"
            conda activate "$UNSLOTH_ENV"
        else
            echo "✗ Setup cancelled. Cannot continue."
            exit 1
        fi
    fi
else
    echo "✗ Conda not found"
    exit 1
fi

# Run CLI
python tuner.py "$@"
