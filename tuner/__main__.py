"""
Entry point for running tuner as a module: python -m tuner

This file allows the tuner package to be executed directly using Python's -m flag.
It imports and calls the main() function from the CLI module.

Usage:
    python -m tuner              # Interactive menu
    python -m tuner train        # Direct training menu
    python -m tuner upload       # Direct upload menu
    python -m tuner eval         # Direct evaluation menu
    python -m tuner pipeline     # Full pipeline
    python -m tuner --help       # Show help
"""

from __future__ import annotations

if __name__ == "__main__":
    from tuner.cli.main import main
    main()
