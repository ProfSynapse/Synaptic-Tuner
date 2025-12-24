#!/bin/bash
# Wrapper script to run MLC LLM commands in a clean Linux environment
# This avoids WSL path permission issues

set -e

# Use only Linux paths
export PATH="/home/profsynapse/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"

# Create a Python script that filters Windows paths
python3 << 'PYTHON_SCRIPT'
import os
import sys
import subprocess

# Filter Windows paths from environment
os.environ['PATH'] = '/home/profsynapse/miniconda3/bin:/usr/local/bin:/usr/bin:/bin'

# Remove Windows paths from PYTHONPATH if set
if 'PYTHONPATH' in os.environ:
    paths = os.environ['PYTHONPATH'].split(':')
    os.environ['PYTHONPATH'] = ':'.join(p for p in paths if not p.startswith('/mnt/c'))

# Get command line arguments
args = sys.argv[1:]
if not args:
    print("Usage: run_mlc_convert.sh <command> [args...]")
    print("Commands: convert_weight, gen_config, compile")
    sys.exit(1)

cmd = args[0]

if cmd == "convert_weight":
    from mlc_llm.cli import convert_weight
    convert_weight.main()
elif cmd == "gen_config":
    from mlc_llm.cli import gen_config
    gen_config.main()
elif cmd == "compile":
    from mlc_llm.cli import compile_model
    compile_model.main()
else:
    print(f"Unknown command: {cmd}")
    sys.exit(1)
PYTHON_SCRIPT
