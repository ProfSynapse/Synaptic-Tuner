"""SynthChat - Unified synthetic dataset generation and improvement system.

Location: SynthChat/__init__.py
Purpose: Package initialization and public API exports
Usage: from SynthChat import SynthChatGenerator, ImprovementEngine

Architecture:
    Single system for both generation and improvement of synthetic training data.
    Combines scenario-driven generation with quality improvement loops.
    Uses shared LLM infrastructure and rubric-based validation.

Components:
    - generator.py: Stage-by-stage dataset generation from scenarios
    - engine.py: Judge → improve quality control loop
    - run.py: CLI entry point for generate/improve/validate modes
    - config/: Settings and validation configuration
    - scenarios/: Generation templates (tools, behaviors, destructive)
    - rubrics/: Quality criteria for improvement
    - services/: Validators, scope handlers, parsing
"""

from .engine import ImprovementEngine, ImprovementResult
from .generator import SynthChatGenerator, ScenarioLoader, GenerationResult

__version__ = "1.0.0"

__all__ = [
    "ImprovementEngine",
    "ImprovementResult",
    "SynthChatGenerator",
    "ScenarioLoader",
    "GenerationResult",
]
