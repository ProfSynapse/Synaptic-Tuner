"""
shared/flywheel/

Enterprise Data Flywheel pipeline package. Captures inference logs from a
vLLM-served model, cleans and tags them by quality, stages versioned training
datasets, and optionally triggers retraining.

Used by: services/proxy/, tuner/handlers/flywheel_handler.py, CLI
"""

from .catalog import (
    DatasetVersion,
    InferenceLogRecord,
    LogCatalog,
    LogFilter,
    SQLiteLogCatalog,
    PostgresLogCatalog,
    create_catalog,
)
from .cleaner import CleaningResult, DataCleaner, NoOpPIIDetector
from .config import FlywheelConfig, load_flywheel_config
from .inference_logger import InferenceLogger
from .orchestrator import (
    CycleResult,
    FlywheelOrchestrator,
    FlywheelStatus,
    RetrainMode,
    TrainingResult,
)
from .readiness import ReadinessChecker, ReadinessReport
from .stager import DatasetStager, StagingResult
from .tagger import AutoTagger, TaggedExample, TaggingResult

__all__ = [
    # Config
    "FlywheelConfig",
    "load_flywheel_config",
    # Catalog
    "InferenceLogRecord",
    "LogFilter",
    "DatasetVersion",
    "LogCatalog",
    "SQLiteLogCatalog",
    "PostgresLogCatalog",
    "create_catalog",
    # Logger
    "InferenceLogger",
    # Pipeline stages
    "DataCleaner",
    "CleaningResult",
    "NoOpPIIDetector",
    "AutoTagger",
    "TaggedExample",
    "TaggingResult",
    "DatasetStager",
    "StagingResult",
    # Orchestration
    "FlywheelOrchestrator",
    "RetrainMode",
    "TrainingResult",
    "CycleResult",
    "FlywheelStatus",
    "ReadinessChecker",
    "ReadinessReport",
]
