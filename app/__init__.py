"""
Intelligent AutoML Platform.
"""
from app.core.config import Settings, get_settings
from app.core.logging import get_logger, setup_logging
from app.core.database import Base, get_db, create_tables
from app.data.loader import DataLoader
from app.data.validator import DataValidator, ValidationResult
from app.data.profiler import DataProfiler, DatasetProfile
from app.features.preprocessing import FeaturePreprocessor
from app.features.feature_engineering import FeatureEngineer
from app.features.selection import FeatureSelector
from app.models.classification import ClassificationModelLibrary
from app.models.regression import RegressionModelLibrary
from app.models.tuning import HyperparameterTuner
from app.models.registry import ModelRegistry
from app.evaluation.metrics import ModelEvaluator
from app.evaluation.reports import ReportGenerator
from app.pipeline.orchestrator import AutoMLOrchestrator
from app.pipeline.monitoring import ModelMonitor

# Backwards compatibility module aliases
import sys
from app.data import validator as data_validator
from app.data import profiler as data_profiler
from app.features import feature_engineering
from app.features import selection as feature_selection
from app.pipeline import orchestrator

sys.modules['app.data_validator'] = data_validator
sys.modules['app.data_profiler'] = data_profiler
sys.modules['app.feature_engineering'] = feature_engineering
sys.modules['app.feature_selection'] = feature_selection
sys.modules['app.orchestrator'] = orchestrator

__version__ = "1.0.0"
