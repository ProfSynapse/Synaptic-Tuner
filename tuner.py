#!/usr/bin/env python3
"""
Synaptic Tuner - Fine-tuning CLI for the Claudesidian MCP

This is a thin wrapper that delegates to the tuner package.

Run from repo root:
    python tuner.py          # Interactive mode
    python tuner.py train    # Training submenu
    python tuner.py upload   # Upload submenu
    python tuner.py eval     # Evaluation submenu
    python tuner.py pipeline # Full pipeline (train -> upload -> eval)

Or run as a module:
    python -m tuner          # Same as above
"""

if __name__ == "__main__":
    from tuner.cli.main import main
    main()
