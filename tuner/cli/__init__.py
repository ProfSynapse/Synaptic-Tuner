"""
CLI layer for the Synaptic Tuner.

This module provides the command-line interface including:
- Argument parsing
- Command routing
- Main entry point
"""

from .main import main
from .parser import create_parser
from .router import route_command

__all__ = [
    'main',
    'create_parser',
    'route_command',
]
