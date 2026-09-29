"""
app/pipeline/orchestrator.py
----------------------------
Central orchestration engine for the Intelligent AutoML Platform.

The :class:`AutoMLOrchestrator` drives every stage of the end-to-end ML
pipeline:  data loading → validation → profiling → feature engineering →
preprocessing → feature selection → model benchmarking → champion tuning →
SHAP explanation → model registration → report generation.

All runs are tracked in MLflow and results are persisted in the model registry.
"""

from __future__ import annotations

import json
import logging
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import contextlib
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
try:
    import mlflow
    import mlflow.sklearn
    _HAS_MLFLOW = True
except ImportError:
    mlflow = None
    _HAS_MLFLOW = False
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPTUNA_AVAILABLE = False

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False

from app.core.config import Settings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases & lightweight data containers
# ---------------------------------------------------------------------------


@dataclass
class ColumnTypeInfo:
    """Stores column classification for a dataset."""

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    high_cardinality: List[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    """Lightweight descriptive profile of a dataset."""

    n_rows: int = 0
    n_cols: int = 0
    missing_pct: float = 0.0
    duplicate_pct: float = 0.0
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    datetime_cols: List[str] = field(default_factory=list)
    target_distribution: Dict[str, Any] = field(default_factory=dict)
    class_balance: Optional[Dict[str, float]] = None
    memory_mb: float = 0.0


@dataclass
class ValidationResult:
    """Outcome of data validation checks."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Per-model evaluation snapshot after cross-validation and hold-out testing."""

    model_name: str = ""
    model: Any = None
    primary_metric: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    training_time: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Full configuration for a single AutoML training run."""

    dataset_path: str = ""
    target_column: str = ""
    problem_type: Optional[str] = None  # None → auto-detect
    feature_selection_method: str = "auto"
    hyperparameter_method: str = "optuna"  # grid | random | optuna
    n_trials: int = 50
    cv_folds: int = 5
    test_size: float = 0.2
    models_to_train: Optional[List[str]] = None  # None → all models
    experiment_name: str = "automl_experiment"
    enable_shap: bool = True
    random_state: int = 42


@dataclass
class TrainingResult:
    """Aggregated result of a complete AutoML training run."""

    experiment_id: str = ""
    status: str = "success"  # success | failed | partial
    problem_type: str = ""
    dataset_profile: Optional[DatasetProfile] = None
    validation_result: Optional[ValidationResult] = None
    benchmark_results: List[EvaluationResult] = field(default_factory=list)
    champion: Optional[EvaluationResult] = None
    champion_model_id: str = ""
    feature_names: List[str] = field(default_factory=list)
    selected_features: List[str] = field(default_factory=list)
    shap_explanation: Optional[Dict[str, Any]] = None
    report_path: str = ""
    training_time: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: Preprocessor builder
# ---------------------------------------------------------------------------


class _DataPreprocessor:
    """Minimal scikit-learn-compatible preprocessing transformer."""

    def __init__(self, numeric_cols: List[str], categorical_cols: List[str],
                 scaler: str = "standard") -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.scaler_type = scaler
        self._scaler: Optional[BaseEstimator] = None
        self._encoder: Optional[OrdinalEncoder] = None
        self._num_fill: Optional[pd.Series] = None
        self._cat_fill: Optional[pd.Series] = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fit on *df* and return the transformed frame + final feature names."""
        df = df.copy()

        # ── Imputation ──────────────────────────────────────────────────────
        if self.numeric_cols:
            self._num_fill = df[self.numeric_cols].median()
            df[self.numeric_cols] = df[self.numeric_cols].fillna(self._num_fill)

        if self.categorical_cols:
            self._cat_fill = df[self.categorical_cols].mode().iloc[0]
            df[self.categorical_cols] = df[self.categorical_cols].fillna(self._cat_fill)

        # ── Encoding ────────────────────────────────────────────────────────
        if self.categorical_cols:
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            df[self.categorical_cols] = self._encoder.fit_transform(
                df[self.categorical_cols]
            )

        # ── Scaling ─────────────────────────────────────────────────────────
        if self.numeric_cols:
            self._scaler = (
                StandardScaler() if self.scaler_type == "standard" else MinMaxScaler()
            )
            df[self.numeric_cols] = self._scaler.fit_transform(df[self.numeric_cols])

        feature_names = self.numeric_cols + self.categorical_cols
        return df[feature_names], feature_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted transforms to *df*."""
        df = df.copy()

        if self.numeric_cols and self._num_fill is not None:
            df[self.numeric_cols] = df[self.numeric_cols].fillna(self._num_fill)

        if self.categorical_cols and self._cat_fill is not None:
            df[self.categorical_cols] = df[self.categorical_cols].fillna(self._cat_fill)

        if self.categorical_cols and self._encoder is not None:
            df[self.categorical_cols] = self._encoder.transform(df[self.categorical_cols])

        if self.numeric_cols and self._scaler is not None:
            df[self.numeric_cols] = self._scaler.transform(df[self.numeric_cols])

        return df[self.numeric_cols + self.categorical_cols]


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

_CLASSIFIERS: Dict[str, Callable[..., BaseEstimator]] = {
    "logistic_regression": lambda rs: LogisticRegression(
        max_iter=1000, random_state=rs, n_jobs=-1
    ),
    "decision_tree": lambda rs: DecisionTreeClassifier(random_state=rs),
    "random_forest": lambda rs: RandomForestClassifier(
        n_estimators=100, random_state=rs, n_jobs=-1
    ),
    "extra_trees": lambda rs: ExtraTreesClassifier(
        n_estimators=100, random_state=rs, n_jobs=-1
    ),
    "gradient_boosting": lambda rs: GradientBoostingClassifier(random_state=rs),
    "knn": lambda rs: KNeighborsClassifier(n_jobs=-1),
    "sgd_classifier": lambda rs: SGDClassifier(random_state=rs, n_jobs=-1),
    "mlp_classifier": lambda rs: MLPClassifier(max_iter=500, random_state=rs),
    "svm_classifier": lambda rs: SVC(probability=True, random_state=rs),
}

_REGRESSORS: Dict[str, Callable[..., BaseEstimator]] = {
    "ridge": lambda rs: Ridge(),
    "lasso": lambda rs: Lasso(),
    "elastic_net": lambda rs: ElasticNet(),
    "decision_tree": lambda rs: DecisionTreeRegressor(random_state=rs),
    "random_forest": lambda rs: RandomForestRegressor(
        n_estimators=100, random_state=rs, n_jobs=-1
    ),
    "extra_trees": lambda rs: ExtraTreesRegressor(
        n_estimators=100, random_state=rs, n_jobs=-1
    ),
    "gradient_boosting": lambda rs: GradientBoostingRegressor(random_state=rs),
    "knn": lambda rs: KNeighborsRegressor(n_jobs=-1),
    "sgd_regressor": lambda rs: SGDRegressor(random_state=rs),
    "mlp_regressor": lambda rs: MLPRegressor(max_iter=500, random_state=rs),
    "svr": lambda rs: SVR(),
}

if _LIGHTGBM_AVAILABLE:
    _CLASSIFIERS["lgbm_classifier"] = lambda rs: LGBMClassifier(
        random_state=rs, n_jobs=-1, verbose=-1
    )
    _REGRESSORS["lgbm_regressor"] = lambda rs: LGBMRegressor(
        random_state=rs, n_jobs=-1, verbose=-1
    )

if _XGBOOST_AVAILABLE:
    _CLASSIFIERS["xgb_classifier"] = lambda rs: XGBClassifier(
        random_state=rs, n_jobs=-1, verbosity=0, use_label_encoder=False,
        eval_metric="logloss",
    )
    _REGRESSORS["xgb_regressor"] = lambda rs: XGBRegressor(
        random_state=rs, n_jobs=-1, verbosity=0
    )


# ---------------------------------------------------------------------------
# Optuna hyperparameter search spaces
# ---------------------------------------------------------------------------

def _get_optuna_objective(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    problem_type: str,
    cv_folds: int,
    random_state: int,
) -> Callable[[Any], float]:
    """Return an Optuna objective function for *model_name*."""

    def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
        if problem_type == "classification":
            model = _build_classifier_for_trial(trial, model_name, random_state)
            scoring = "roc_auc" if len(np.unique(y_train)) == 2 else "accuracy"
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            model = _build_regressor_for_trial(trial, model_name, random_state)
            scoring = "r2"
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        if model is None:
            raise optuna.exceptions.TrialPruned()

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        return float(np.mean(scores))

    return objective


def _build_classifier_for_trial(trial: Any, model_name: str, rs: int) -> Optional[BaseEstimator]:
    """Build a classifier with trial-suggested hyperparameters."""
    if model_name in ("logistic_regression",):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        return LogisticRegression(C=C, max_iter=1000, random_state=rs, n_jobs=-1)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=rs, n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            random_state=rs, n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=rs,
        )
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            random_state=rs,
        )
    if _LIGHTGBM_AVAILABLE and model_name == "lgbm_classifier":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", -1, 15),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            random_state=rs, n_jobs=-1, verbose=-1,
        )
    if _XGBOOST_AVAILABLE and model_name == "xgb_classifier":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            random_state=rs, n_jobs=-1, verbosity=0, eval_metric="logloss",
        )
    return None


def _build_regressor_for_trial(trial: Any, model_name: str, rs: int) -> Optional[BaseEstimator]:
    """Build a regressor with trial-suggested hyperparameters."""
    if model_name == "ridge":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-4, 100.0, log=True))
    if model_name == "lasso":
        return Lasso(alpha=trial.suggest_float("alpha", 1e-4, 10.0, log=True))
    if model_name == "elastic_net":
        return ElasticNet(
            alpha=trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=rs, n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            random_state=rs, n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=rs,
        )
    if model_name == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            random_state=rs,
        )
    if _LIGHTGBM_AVAILABLE and model_name == "lgbm_regressor":
        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", -1, 15),
            random_state=rs, n_jobs=-1, verbose=-1,
        )
    if _XGBOOST_AVAILABLE and model_name == "xgb_regressor":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=rs, n_jobs=-1, verbosity=0,
        )
    return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class AutoMLOrchestrator:
    """
    Central brain of the Intelligent AutoML Platform.

    Drives the entire end-to-end ML pipeline from raw data to a registered,
    explainable, MLflow-tracked model.

    Parameters
    ----------
    config : Settings
        Platform-wide configuration (paths, limits, defaults).
    db_session : optional
        SQLAlchemy session used to persist experiment metadata.  When ``None``
        all results are kept in memory only.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        db_session: Any = None,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        if config is None:
            from app.core.config import get_settings
            config = get_settings()
        self.config = config
        self.db_session = db_session
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self.output_dir = output_dir
        self.default_progress_callback = progress_callback

        if output_dir is not None:
            self.config.MODEL_REGISTRY_PATH = str(Path(output_dir) / "models")
            self.config.REPORTS_PATH = str(Path(output_dir) / "reports")

        # Configure MLflow if available
        if _HAS_MLFLOW and mlflow is not None:
            try:
                mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
            except Exception as e:
                logger.warning("Could not set MLflow tracking URI: %s", e)

        # Ensure output directories exist
        Path(self.config.MODEL_REGISTRY_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.config.REPORTS_PATH).mkdir(parents=True, exist_ok=True)

        logger.info("AutoMLOrchestrator initialised — MLflow URI: %s", self.config.MLFLOW_TRACKING_URI)

    # ------------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience method that saves a temporary dataset and runs the full pipeline."""
        import tempfile

        temp_csv = Path(tempfile.gettempdir()) / f"automl_{uuid.uuid4().hex[:8]}.csv"
        df.to_csv(temp_csv, index=False)

        config = TrainingConfig(
            dataset_path=str(temp_csv),
            target_column=target_column,
            problem_type=problem_type,
            n_trials=kwargs.get("n_trials", 5),
            cv_folds=kwargs.get("cv_folds", 2),
            enable_shap=False,
            models_to_train=["random_forest", "logistic_regression", "ridge"],
        )
        cb_func = progress_callback or getattr(self, "default_progress_callback", None)
        if cb_func:
            def _cb(step, pct, msg):
                payload = {"step": step, "stage": step, "pct": pct, "progress": pct, "message": msg}
                try:
                    cb_func(payload)
                except TypeError:
                    try:
                        cb_func(pct, msg)
                    except TypeError:
                        cb_func(step, pct, msg)
        else:
            _cb = None

        res = self.train(config, progress_callback=_cb)

        try:
            temp_csv.unlink(missing_ok=True)
        except Exception:
            pass

        out_path = Path(self.config.MODEL_REGISTRY_PATH) / "model.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model_obj = getattr(self, "_last_model", None)
        if model_obj is not None:
            joblib.dump(model_obj, out_path)

        metrics = {}
        champion_obj = getattr(res, "champion", None)
        if champion_obj:
            if hasattr(champion_obj, "metrics") and isinstance(champion_obj.metrics, dict):
                metrics = champion_obj.metrics
            else:
                for k in ["accuracy", "f1_weighted", "r2", "rmse", "mae", "cv_mean"]:
                    val = getattr(champion_obj, k, None)
                    if val is not None:
                        metrics[k] = val

        return {
            "best_model": model_obj,
            "model": model_obj,
            "metrics": metrics,
            "model_path": str(out_path),
            "report_path": res.report_path,
            "result": res,
        }

    def train(
        self,
        training_config: TrainingConfig,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> TrainingResult:
        """
        Execute the full AutoML pipeline.

        Parameters
        ----------
        training_config : TrainingConfig
            All parameters governing this training run.
        progress_callback : Callable[[str, int, str], None], optional
            Called at each pipeline step with ``(step_name, pct_complete, message)``.

        Returns
        -------
        TrainingResult
            Comprehensive result object including champion model details.
        """
        experiment_id = str(uuid.uuid4())
        start_time = time.time()
        result = TrainingResult(experiment_id=experiment_id)

        # Record running state
        self._experiments[experiment_id] = {
            "id": experiment_id,
            "name": training_config.experiment_name,
            "status": "running",
            "started_at": time.time(),
            "config": training_config,
        }

        logger.info("Starting AutoML run [%s] for experiment '%s'",
                    experiment_id, training_config.experiment_name)

        mlflow.set_experiment(training_config.experiment_name)

        try:
            with mlflow.start_run(run_name=f"{training_config.experiment_name}_{experiment_id[:8]}") as run:
                mlflow.log_params({
                    "dataset_path": training_config.dataset_path,
                    "target_column": training_config.target_column,
                    "problem_type": training_config.problem_type or "auto",
                    "feature_selection_method": training_config.feature_selection_method,
                    "hyperparameter_method": training_config.hyperparameter_method,
                    "n_trials": training_config.n_trials,
                    "cv_folds": training_config.cv_folds,
                    "test_size": training_config.test_size,
                    "random_state": training_config.random_state,
                    "enable_shap": training_config.enable_shap,
                })

                # ── Step 1: Load data (5%) ─────────────────────────────
                self._notify_progress(progress_callback, "load_data", 5, "Loading dataset…")
                df = self._step_load_data(training_config)
                mlflow.log_metric("raw_rows", df.shape[0])
                mlflow.log_metric("raw_cols", df.shape[1])

                # ── Step 2: Validate (10%) ─────────────────────────────
                self._notify_progress(progress_callback, "validate", 10, "Validating dataset…")
                validation_result = self._step_validate(df, training_config.target_column)
                result.validation_result = validation_result
                if not validation_result.is_valid:
                    raise ValueError(
                        "Dataset validation failed: " + "; ".join(validation_result.errors)
                    )

                # ── Step 3: Profile (15%) ──────────────────────────────
                self._notify_progress(progress_callback, "profile", 15, "Profiling dataset…")
                profile = self._step_profile(df)
                result.dataset_profile = profile
                mlflow.log_metric("missing_pct", profile.missing_pct)
                mlflow.log_metric("duplicate_pct", profile.duplicate_pct)

                # Detect problem type
                problem_type = training_config.problem_type or self._detect_problem_type(
                    df[training_config.target_column]
                )
                result.problem_type = problem_type
                mlflow.log_param("detected_problem_type", problem_type)

                # ── Step 4: Engineer features (25%) ───────────────────
                self._notify_progress(progress_callback, "engineer_features", 25,
                                      "Engineering features…")
                df, col_types = self._step_engineer_features(df, training_config.target_column)

                # ── Step 5: Preprocess (30%) ───────────────────────────
                self._notify_progress(progress_callback, "preprocess", 30,
                                      "Preprocessing & splitting data…")
                X_train, X_test, y_train, y_test, preprocessor, feature_names = (
                    self._step_preprocess(df, training_config.target_column,
                                         col_types, training_config)
                )
                result.feature_names = feature_names
                mlflow.log_metric("n_features_raw", len(feature_names))
                mlflow.log_metric("train_samples", X_train.shape[0])
                mlflow.log_metric("test_samples", X_test.shape[0])

                # ── Step 6: Select features (35%) ─────────────────────
                self._notify_progress(progress_callback, "select_features", 35,
                                      "Selecting informative features…")
                X_train, X_test, selected_features = self._step_select_features(
                    X_train, y_train, X_test, feature_names,
                    problem_type, training_config
                )
                result.selected_features = selected_features
                mlflow.log_metric("n_features_selected", len(selected_features))

                # ── Step 7: Benchmark models (70%) ────────────────────
                self._notify_progress(progress_callback, "benchmark_models", 40,
                                      "Benchmarking candidate models…")
                benchmark_results = self._step_benchmark_models(
                    X_train, y_train, X_test, y_test,
                    problem_type, training_config,
                    progress_callback=progress_callback,
                )
                result.benchmark_results = benchmark_results

                if not benchmark_results:
                    raise RuntimeError("No models were successfully trained.")

                champion_result = benchmark_results[0]  # already sorted best-first

                # ── Step 8: Tune champion (85%) ────────────────────────
                self._notify_progress(progress_callback, "tune_champion", 70,
                                      f"Tuning champion '{champion_result.model_name}'…")
                tuned_model, tuned_result = self._step_tune_champion(
                    champion_result, X_train, y_train, X_test, y_test,
                    problem_type, training_config
                )
                result.champion = tuned_result
                mlflow.log_metrics({
                    f"champion_{k}": v
                    for k, v in tuned_result.metrics.items()
                    if isinstance(v, (int, float))
                })
                mlflow.sklearn.log_model(tuned_model, artifact_path="champion_model")

                # ── Step 9: SHAP (90%) ────────────────────────────────
                shap_explanation: Optional[Dict[str, Any]] = None
                if training_config.enable_shap and _SHAP_AVAILABLE:
                    self._notify_progress(progress_callback, "compute_shap", 85,
                                          "Computing SHAP explanations…")
                    try:
                        shap_explanation = self._step_compute_shap(
                            tuned_model, X_test, selected_features
                        )
                        result.shap_explanation = shap_explanation
                    except Exception as exc:  # pragma: no cover
                        logger.warning("SHAP computation failed: %s", exc)

                # ── Step 10: Register model (95%) ─────────────────────
                self._notify_progress(progress_callback, "register_model", 90,
                                      "Registering champion model…")
                metadata = {
                    "experiment_id": experiment_id,
                    "problem_type": problem_type,
                    "feature_names": selected_features,
                    "preprocessor": preprocessor,
                    "mlflow_run_id": run.info.run_id,
                    "metrics": tuned_result.metrics,
                    "params": tuned_result.params,
                }
                model_id = self._step_register_model(tuned_model, tuned_result, metadata)
                result.champion_model_id = model_id
                self._last_model = tuned_model
                self._last_preprocessor = preprocessor
                self._last_champion_id = model_id
                mlflow.log_param("champion_model_id", model_id)

                # ── Step 11: Generate report (100%) ───────────────────
                self._notify_progress(progress_callback, "generate_report", 95,
                                      "Generating HTML report…")
                result.training_time = time.time() - start_time
                report_path = self._step_generate_report(result)
                result.report_path = report_path
                mlflow.log_artifact(report_path)

                self._notify_progress(progress_callback, "complete", 100,
                                      "Training complete!")

                result.status = "success"
                self._experiments[experiment_id]["status"] = "success"
                self._experiments[experiment_id]["result"] = result

                logger.info(
                    "AutoML run [%s] completed in %.1fs — champion: %s (primary metric=%.4f)",
                    experiment_id, result.training_time,
                    tuned_result.model_name, tuned_result.primary_metric,
                )

        except Exception as exc:
            elapsed = time.time() - start_time
            tb = traceback.format_exc()
            logger.error("AutoML run [%s] failed after %.1fs:\n%s", experiment_id, elapsed, tb)
            result.status = "failed"
            result.error = str(exc)
            result.training_time = elapsed
            self._experiments[experiment_id]["status"] = "failed"
            self._experiments[experiment_id]["error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _step_load_data(self, config: TrainingConfig) -> pd.DataFrame:
        """
        Load a dataset from *config.dataset_path*.

        Supports CSV, TSV, Parquet, JSON, and Excel formats.
        """
        path = Path(config.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        suffix = path.suffix.lower()
        loaders: Dict[str, Callable] = {
            ".csv": lambda p: pd.read_csv(p),
            ".tsv": lambda p: pd.read_csv(p, sep="\t"),
            ".parquet": lambda p: pd.read_parquet(p),
            ".json": lambda p: pd.read_json(p),
            ".jsonl": lambda p: pd.read_json(p, lines=True),
            ".xlsx": lambda p: pd.read_excel(p),
            ".xls": lambda p: pd.read_excel(p),
        }
        loader = loaders.get(suffix)
        if loader is None:
            raise ValueError(f"Unsupported file format: {suffix!r}")

        df = loader(path)
        logger.info("Loaded dataset %s — shape: %s", path.name, df.shape)

        # Enforce platform limits
        if df.shape[0] > self.config.MAX_ROWS:
            logger.warning(
                "Dataset has %d rows, exceeds MAX_ROWS=%d; sampling down.",
                df.shape[0], self.config.MAX_ROWS,
            )
            df = df.sample(n=self.config.MAX_ROWS, random_state=config.random_state)

        if df.shape[1] > self.config.MAX_COLUMNS:
            raise ValueError(
                f"Dataset has {df.shape[1]} columns, exceeds MAX_COLUMNS={self.config.MAX_COLUMNS}."
            )

        return df.reset_index(drop=True)

    def _step_validate(self, df: pd.DataFrame, target_col: str) -> ValidationResult:
        """
        Run a battery of structural and statistical data-quality checks.

        Checks: target present, min rows/cols, missing rate, constant columns,
        duplicate rows, target leakage indicators.
        """
        vr = ValidationResult()

        # Target column exists
        if target_col not in df.columns:
            vr.errors.append(f"Target column '{target_col}' not found in dataset.")
            vr.is_valid = False
            return vr

        # Minimum viable size
        if len(df) < 20:
            vr.errors.append(f"Dataset has only {len(df)} rows; minimum required is 20.")
            vr.is_valid = False

        if df.shape[1] < 2:
            vr.errors.append("Dataset must have at least 2 columns (features + target).")
            vr.is_valid = False

        # Target nulls
        target_null_pct = df[target_col].isnull().mean() * 100
        if target_null_pct > 0:
            vr.errors.append(
                f"Target column '{target_col}' has {target_null_pct:.1f}% missing values."
            )
            vr.is_valid = False

        # High missing rate in features
        feature_cols = [c for c in df.columns if c != target_col]
        col_missing = df[feature_cols].isnull().mean()
        high_missing = col_missing[col_missing > 0.8].index.tolist()
        if high_missing:
            vr.warnings.append(
                f"{len(high_missing)} feature(s) have >80% missing values: "
                + ", ".join(high_missing[:5]) + ("…" if len(high_missing) > 5 else "")
            )

        # Constant columns
        constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
        if constant_cols:
            vr.warnings.append(
                f"{len(constant_cols)} constant/empty column(s) detected: "
                + ", ".join(constant_cols[:5])
            )

        # Duplicate rows
        dup_pct = df.duplicated().mean() * 100
        if dup_pct > 50:
            vr.warnings.append(f"{dup_pct:.1f}% of rows are duplicates.")

        # Sufficient unique target values
        n_unique_target = df[target_col].nunique()
        if n_unique_target == 1:
            vr.errors.append("Target column has only one unique value; training is impossible.")
            vr.is_valid = False

        vr.info.append(
            f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} cols. "
            f"Target unique values: {n_unique_target}."
        )

        logger.info(
            "Validation complete — valid=%s, errors=%d, warnings=%d",
            vr.is_valid, len(vr.errors), len(vr.warnings),
        )
        return vr

    def _step_profile(self, df: pd.DataFrame) -> DatasetProfile:
        """
        Compute a lightweight descriptive profile of *df*.

        Identifies numeric, categorical, and datetime columns; computes
        missing/duplicate rates; estimates memory usage.
        """
        profile = DatasetProfile()
        profile.n_rows, profile.n_cols = df.shape
        profile.missing_pct = float(df.isnull().mean().mean() * 100)
        profile.duplicate_pct = float(df.duplicated().mean() * 100)
        profile.memory_mb = float(df.memory_usage(deep=True).sum() / 1_048_576)

        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                profile.datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                profile.numeric_cols.append(col)
            else:
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().head(20))
                    profile.datetime_cols.append(col)
                except (ValueError, TypeError):
                    profile.categorical_cols.append(col)

        logger.info(
            "Profile — rows=%d, cols=%d, missing=%.1f%%, numeric=%d, categorical=%d, datetime=%d",
            profile.n_rows, profile.n_cols, profile.missing_pct,
            len(profile.numeric_cols), len(profile.categorical_cols), len(profile.datetime_cols),
        )
        return profile

    def _step_engineer_features(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, ColumnTypeInfo]:
        """
        Perform lightweight automated feature engineering.

        Operations:
        * Parse and expand datetime columns into year/month/day/weekday/hour.
        * Drop columns with >90% missing values.
        * Flag high-cardinality categoricals (>50 unique values) for exclusion.
        * Strip whitespace from string columns.

        Returns the transformed DataFrame and a :class:`ColumnTypeInfo` mapping.
        """
        col_types = ColumnTypeInfo()
        cols_to_drop: List[str] = []

        for col in df.columns:
            if col == target_col:
                continue

            # Drop >90% missing
            if df[col].isnull().mean() > 0.90:
                cols_to_drop.append(col)
                continue

            dtype = df[col].dtype

            # Datetime expansion
            if pd.api.types.is_datetime64_any_dtype(dtype):
                col_types.datetime.append(col)
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_weekday"] = df[col].dt.weekday
                df[f"{col}_hour"] = df[col].dt.hour
                cols_to_drop.append(col)
                for new_col in [f"{col}_year", f"{col}_month", f"{col}_day",
                                 f"{col}_weekday", f"{col}_hour"]:
                    col_types.numeric.append(new_col)
                continue

            if pd.api.types.is_numeric_dtype(dtype):
                col_types.numeric.append(col)
            else:
                # Try datetime parse
                try:
                    parsed = pd.to_datetime(df[col], errors="raise")
                    df[col] = parsed
                    col_types.datetime.append(col)
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_weekday"] = df[col].dt.weekday
                    cols_to_drop.append(col)
                    for new_col in [f"{col}_year", f"{col}_month", f"{col}_day",
                                    f"{col}_weekday"]:
                        col_types.numeric.append(new_col)
                    continue
                except (ValueError, TypeError):
                    pass

                # Categorical
                n_unique = df[col].nunique()
                if n_unique > 50:
                    col_types.high_cardinality.append(col)
                    # Hash-encode to keep it usable
                    df[col] = df[col].astype(str).apply(lambda x: hash(x) % 10_000)
                    col_types.numeric.append(col)
                else:
                    # Strip whitespace
                    df[col] = df[col].astype(str).str.strip()
                    col_types.categorical.append(col)

        if cols_to_drop:
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        logger.info(
            "Feature engineering — numeric=%d, categorical=%d, datetime=%d, high_cardinality=%d, dropped=%d",
            len(col_types.numeric), len(col_types.categorical),
            len(col_types.datetime), len(col_types.high_cardinality), len(cols_to_drop),
        )
        return df, col_types

    def _step_preprocess(
        self,
        df: pd.DataFrame,
        target_col: str,
        col_types: ColumnTypeInfo,
        training_config: TrainingConfig,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, _DataPreprocessor, List[str]]:
        """
        Encode, impute, scale, and split the dataset.

        Returns
        -------
        X_train, X_test, y_train, y_test : np.ndarray
        preprocessor : _DataPreprocessor
            Fitted transformer (serialised into the model package).
        feature_names : list[str]
            Ordered list of feature column names.
        """
        # Separate target
        y_raw = df[target_col].copy()
        feature_cols = [
            c for c in df.columns
            if c != target_col and (
                c in col_types.numeric or c in col_types.categorical
            )
        ]
        X_df = df[feature_cols].copy()

        # Encode target for classification
        label_encoder: Optional[LabelEncoder] = None
        if y_raw.dtype == object or str(y_raw.dtype) == "category":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw.astype(str))
        else:
            y = y_raw.values

        # Train/test split
        numeric_in_features = [c for c in col_types.numeric if c in feature_cols]
        categorical_in_features = [c for c in col_types.categorical if c in feature_cols]

        preprocessor = _DataPreprocessor(
            numeric_cols=numeric_in_features,
            categorical_cols=categorical_in_features,
        )
        X_processed, feature_names = preprocessor.fit_transform(X_df)

        X_arr = X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed
        y_arr = y.astype(float) if np.issubdtype(np.array(y).dtype, np.floating) else y

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr,
            test_size=training_config.test_size,
            random_state=training_config.random_state,
            stratify=y_arr if self._detect_problem_type(pd.Series(y_arr)) == "classification" else None,
        )

        # Attach label encoder to preprocessor for downstream use
        preprocessor._label_encoder = label_encoder  # type: ignore[attr-defined]

        logger.info(
            "Preprocessing — train=%d, test=%d, features=%d",
            X_train.shape[0], X_test.shape[0], len(feature_names),
        )
        return X_train, X_test, y_train, y_test, preprocessor, feature_names

    def _step_select_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        problem_type: str,
        training_config: TrainingConfig,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Reduce the feature space using the configured selection method.

        Methods:
        * ``auto`` — SelectFromModel with a RandomForest estimator.
        * ``kbest`` — SelectKBest with ANOVA F-score.
        * ``none`` — keep all features.

        Returns updated X_train, X_test, and the selected feature names.
        """
        method = training_config.feature_selection_method.lower()
        n_features = X_train.shape[1]

        if method == "none" or n_features <= 5:
            logger.info("Feature selection skipped (method=%s, n_features=%d)", method, n_features)
            return X_train, X_test, feature_names

        selected_mask: Optional[np.ndarray] = None

        if method in ("auto", "model"):
            if problem_type == "classification":
                selector_estimator = RandomForestClassifier(
                    n_estimators=50, random_state=training_config.random_state, n_jobs=-1
                )
            else:
                selector_estimator = RandomForestRegressor(
                    n_estimators=50, random_state=training_config.random_state, n_jobs=-1
                )
            selector = SelectFromModel(
                selector_estimator, threshold="median", prefit=False
            )
            selector.fit(X_train, y_train)
            selected_mask = selector.get_support()

        elif method == "kbest":
            k = max(5, n_features // 2)
            score_func = f_classif if problem_type == "classification" else f_regression
            selector = SelectKBest(score_func=score_func, k=min(k, n_features))
            selector.fit(X_train, y_train)
            selected_mask = selector.get_support()

        else:
            logger.warning("Unknown feature_selection_method '%s'; keeping all.", method)
            return X_train, X_test, feature_names

        if selected_mask is not None and selected_mask.sum() > 0:
            X_train = X_train[:, selected_mask]
            X_test = X_test[:, selected_mask]
            selected_features = [f for f, keep in zip(feature_names, selected_mask) if keep]
        else:
            logger.warning("Feature selection produced empty set; retaining all features.")
            selected_features = feature_names

        logger.info(
            "Feature selection — method=%s, kept %d / %d features",
            method, len(selected_features), n_features,
        )
        return X_train, X_test, selected_features

    def _step_benchmark_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        problem_type: str,
        training_config: TrainingConfig,
        progress_callback: Optional[Callable] = None,
    ) -> List[EvaluationResult]:
        """
        Train and cross-validate every candidate model.

        Models are returned sorted by primary metric (descending).

        Progress is reported incrementally between 40 % and 70 %.
        """
        catalogue = _CLASSIFIERS if problem_type == "classification" else _REGRESSORS

        if training_config.models_to_train:
            catalogue = {
                k: v for k, v in catalogue.items()
                if k in training_config.models_to_train
            }

        cv = (
            StratifiedKFold(
                n_splits=training_config.cv_folds, shuffle=True,
                random_state=training_config.random_state,
            )
            if problem_type == "classification"
            else KFold(
                n_splits=training_config.cv_folds, shuffle=True,
                random_state=training_config.random_state,
            )
        )

        scoring = (
            "roc_auc" if (problem_type == "classification" and len(np.unique(y_train)) == 2)
            else ("accuracy" if problem_type == "classification" else "r2")
        )

        results: List[EvaluationResult] = []
        n_models = len(catalogue)

        for idx, (name, factory) in enumerate(catalogue.items()):
            pct = 40 + int(30 * idx / max(n_models, 1))
            self._notify_progress(
                progress_callback, "benchmark_models", pct,
                f"Training {name} ({idx + 1}/{n_models})…",
            )
            try:
                model = factory(training_config.random_state)
                t0 = time.time()
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
                )
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                metrics = self._compute_metrics(model, X_test, y_test, problem_type)
                primary = float(np.mean(cv_scores))

                eval_result = EvaluationResult(
                    model_name=name,
                    model=model,
                    primary_metric=primary,
                    metrics=metrics,
                    cv_mean=float(np.mean(cv_scores)),
                    cv_std=float(np.std(cv_scores)),
                    training_time=train_time,
                    params=model.get_params(),
                )
                results.append(eval_result)
                logger.info(
                    "  %-30s CV=%.4f±%.4f  test_primary=%.4f  time=%.1fs",
                    name, primary, np.std(cv_scores),
                    metrics.get("accuracy" if problem_type == "classification" else "r2", 0.0),
                    train_time,
                )
            except Exception as exc:
                logger.warning("Model '%s' failed: %s", name, exc)

        # Sort: higher is better for both accuracy/ROC-AUC and R²
        results.sort(key=lambda r: r.primary_metric, reverse=True)
        logger.info("Benchmarking complete — best: %s (%.4f)", results[0].model_name, results[0].primary_metric)
        return results

    def _step_tune_champion(
        self,
        champion_result: EvaluationResult,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        problem_type: str,
        training_config: TrainingConfig,
    ) -> Tuple[BaseEstimator, EvaluationResult]:
        """
        Run hyper-parameter optimisation on the champion model.

        Falls back gracefully when Optuna is unavailable or when the model
        name has no defined search space.
        """
        method = training_config.hyperparameter_method.lower()
        model_name = champion_result.model_name

        if method == "optuna" and _OPTUNA_AVAILABLE:
            objective = _get_optuna_objective(
                model_name, X_train, y_train,
                problem_type, training_config.cv_folds,
                training_config.random_state,
            )
            study = optuna.create_study(direction="maximize")
            try:
                study.optimize(
                    objective,
                    n_trials=training_config.n_trials,
                    timeout=300,  # max 5 min per study
                    show_progress_bar=False,
                )
                best_params = study.best_params
                logger.info(
                    "Optuna tuning — best trial: %.4f with params: %s",
                    study.best_value, best_params,
                )
            except Exception as exc:
                logger.warning("Optuna study failed (%s); using default champion.", exc)
                best_params = {}
        else:
            logger.info("Hyperparameter method '%s' — skipping tuning; using champion as-is.", method)
            best_params = {}

        # Rebuild champion with best params if found
        if best_params:
            catalogue = _CLASSIFIERS if problem_type == "classification" else _REGRESSORS
            if model_name in catalogue:
                base_model = catalogue[model_name](training_config.random_state)
                valid_params = {k: v for k, v in best_params.items()
                                if k in base_model.get_params()}
                base_model.set_params(**valid_params)
                t0 = time.time()
                base_model.fit(X_train, y_train)
                train_time = time.time() - t0
                tuned_metrics = self._compute_metrics(base_model, X_test, y_test, problem_type)

                cv = (
                    StratifiedKFold(n_splits=training_config.cv_folds, shuffle=True,
                                    random_state=training_config.random_state)
                    if problem_type == "classification"
                    else KFold(n_splits=training_config.cv_folds, shuffle=True,
                               random_state=training_config.random_state)
                )
                scoring = (
                    "roc_auc" if (problem_type == "classification" and len(np.unique(y_train)) == 2)
                    else ("accuracy" if problem_type == "classification" else "r2")
                )
                cv_scores = cross_val_score(
                    base_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
                )

                tuned_result = EvaluationResult(
                    model_name=model_name,
                    model=base_model,
                    primary_metric=float(np.mean(cv_scores)),
                    metrics=tuned_metrics,
                    cv_mean=float(np.mean(cv_scores)),
                    cv_std=float(np.std(cv_scores)),
                    training_time=train_time,
                    params=base_model.get_params(),
                )

                # Only keep tuned if it improved
                if tuned_result.primary_metric >= champion_result.primary_metric:
                    logger.info(
                        "Tuning improved champion: %.4f → %.4f",
                        champion_result.primary_metric, tuned_result.primary_metric,
                    )
                    return base_model, tuned_result
                else:
                    logger.info(
                        "Tuned model did not improve (%.4f vs %.4f); keeping original.",
                        tuned_result.primary_metric, champion_result.primary_metric,
                    )

        # Fall through: refit original champion cleanly
        catalogue = _CLASSIFIERS if problem_type == "classification" else _REGRESSORS
        if model_name in catalogue:
            final_model = catalogue[model_name](training_config.random_state)
            final_model.fit(X_train, y_train)
        else:
            final_model = champion_result.model
            if not hasattr(final_model, "predict"):
                final_model.fit(X_train, y_train)

        final_metrics = self._compute_metrics(final_model, X_test, y_test, problem_type)
        final_result = EvaluationResult(
            model_name=model_name,
            model=final_model,
            primary_metric=champion_result.primary_metric,
            metrics=final_metrics,
            cv_mean=champion_result.cv_mean,
            cv_std=champion_result.cv_std,
            training_time=champion_result.training_time,
            params=final_model.get_params() if hasattr(final_model, "get_params") else {},
        )
        return final_model, final_result

    def _step_register_model(
        self,
        model: BaseEstimator,
        champion_result: EvaluationResult,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Serialise the champion model and its metadata to the model registry.

        The on-disk layout:
        ``<MODEL_REGISTRY_PATH>/<model_id>/model.joblib``
        ``<MODEL_REGISTRY_PATH>/<model_id>/metadata.json``
        ``<MODEL_REGISTRY_PATH>/<model_id>/preprocessor.joblib``

        Returns
        -------
        str
            The generated ``model_id``.
        """
        model_id = str(uuid.uuid4())
        model_dir = Path(self.config.MODEL_REGISTRY_PATH) / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Serialise artefacts
        joblib.dump(model, model_dir / "model.joblib")

        preprocessor = metadata.pop("preprocessor", None)
        if preprocessor is not None:
            joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
            metadata["has_preprocessor"] = True

        # JSON-safe metadata
        safe_meta: Dict[str, Any] = {}
        for k, v in metadata.items():
            try:
                json.dumps(v)
                safe_meta[k] = v
            except (TypeError, ValueError):
                safe_meta[k] = str(v)

        safe_meta["model_id"] = model_id
        safe_meta["model_name"] = champion_result.model_name
        safe_meta["registered_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        with open(model_dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(safe_meta, fh, indent=2)

        logger.info("Model registered — id=%s, path=%s", model_id, model_dir)
        return model_id

    def _step_compute_shap(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for global and per-feature importance.

        Uses TreeExplainer for tree-based models, otherwise falls back to
        KernelExplainer on a small background sample.

        Returns a dictionary with keys:
        * ``mean_abs_shap`` — dict mapping feature name → mean |SHAP|
        * ``top_features`` — top-10 features by mean |SHAP|
        * ``values`` — raw SHAP matrix (list of lists)
        """
        if not _SHAP_AVAILABLE:
            return {}

        n_background = min(self.config.SHAP_MAX_SAMPLES, X_test.shape[0])
        background = X_test[:n_background]

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(background)
        except Exception:
            try:
                explainer = shap.KernelExplainer(
                    model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                    background[:min(50, n_background)],
                )
                shap_values = explainer.shap_values(background, nsamples=100)
            except Exception as exc:
                logger.warning("SHAP explainer failed: %s", exc)
                return {}

        # Handle multi-class (list of arrays) vs binary
        if isinstance(shap_values, list):
            arr = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            arr = np.abs(shap_values)

        if arr.ndim > 2:
            arr = arr.mean(axis=0)

        mean_abs = arr.mean(axis=0) if arr.ndim == 2 else arr

        importance = {
            f: float(v)
            for f, v in zip(feature_names, mean_abs)
            if len(feature_names) == len(mean_abs)
        }
        top_features = sorted(importance, key=lambda k: importance[k], reverse=True)[:10]

        return {
            "mean_abs_shap": importance,
            "top_features": top_features,
            "values": arr.tolist() if hasattr(arr, "tolist") else [],
        }

    def _step_generate_report(self, result: TrainingResult) -> str:
        """
        Generate an HTML summary report for the training run.

        Writes to ``<REPORTS_PATH>/<experiment_id>.html`` and returns the path.
        """
        report_dir = Path(self.config.REPORTS_PATH)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = str(report_dir / f"{result.experiment_id}.html")

        champion = result.champion
        metrics_rows = ""
        if champion:
            for k, v in champion.metrics.items():
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                metrics_rows += f"<tr><td>{k}</td><td><strong>{val_str}</strong></td></tr>\n"

        benchmark_rows = ""
        for er in result.benchmark_results[:10]:
            benchmark_rows += (
                f"<tr><td>{er.model_name}</td>"
                f"<td>{er.cv_mean:.4f}±{er.cv_std:.4f}</td>"
                f"<td>{er.primary_metric:.4f}</td>"
                f"<td>{er.training_time:.1f}s</td></tr>\n"
            )

        shap_section = ""
        if result.shap_explanation and result.shap_explanation.get("top_features"):
            items = "".join(
                f"<li><code>{f}</code>: {result.shap_explanation['mean_abs_shap'].get(f, 0):.4f}</li>"
                for f in result.shap_explanation["top_features"]
            )
            shap_section = f"<h2>Top SHAP Features</h2><ol>{items}</ol>"

        profile = result.dataset_profile
        profile_html = ""
        if profile:
            profile_html = f"""
            <table>
              <tr><td>Rows</td><td>{profile.n_rows:,}</td></tr>
              <tr><td>Columns</td><td>{profile.n_cols}</td></tr>
              <tr><td>Missing %</td><td>{profile.missing_pct:.2f}%</td></tr>
              <tr><td>Duplicate %</td><td>{profile.duplicate_pct:.2f}%</td></tr>
              <tr><td>Memory</td><td>{profile.memory_mb:.2f} MB</td></tr>
            </table>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AutoML Report — {result.experiment_id[:8]}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; color: #333; }}
    h1 {{ color: #1a73e8; }}
    h2 {{ color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
              background: #1a73e8; color: #fff; font-size: 12px; }}
    .metric-good {{ color: #0a8b3c; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>🤖 AutoML Training Report</h1>
  <p><strong>Experiment ID:</strong> {result.experiment_id}</p>
  <p><strong>Status:</strong> <span class="badge">{result.status.upper()}</span></p>
  <p><strong>Problem Type:</strong> {result.problem_type}</p>
  <p><strong>Training Time:</strong> {result.training_time:.1f}s</p>
  <p><strong>Champion Model:</strong> {champion.model_name if champion else 'N/A'}</p>
  <p><strong>Champion Model ID:</strong> <code>{result.champion_model_id}</code></p>

  <h2>Dataset Profile</h2>
  {profile_html}

  <h2>Champion Model Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {metrics_rows}
  </table>

  <h2>Model Benchmark Results</h2>
  <table>
    <tr><th>Model</th><th>CV Score (mean±std)</th><th>Primary Metric</th><th>Train Time</th></tr>
    {benchmark_rows}
  </table>

  <h2>Selected Features ({len(result.selected_features)})</h2>
  <p>{', '.join(f'<code>{f}</code>' for f in result.selected_features[:30])}</p>

  {shap_section}

  <h2>Validation</h2>
  <ul>
    {''.join(f'<li>⚠️ {w}</li>' for w in (result.validation_result.warnings if result.validation_result else []))}
    {''.join(f'<li>ℹ️ {i}</li>' for i in (result.validation_result.info if result.validation_result else []))}
  </ul>

  <footer><p style="color:#999;font-size:12px;">
    Generated by Intelligent AutoML Platform • {time.strftime("%Y-%m-%d %H:%M:%S")}
  </p></footer>
</body>
</html>
"""
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("Report generated: %s", report_path)
        return report_path

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict(
        self,
        model_id_or_data: Any,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Run inference for a single sample or batch DataFrame.
        """
        if isinstance(model_id_or_data, (pd.DataFrame, np.ndarray)):
            df = (
                model_id_or_data
                if isinstance(model_id_or_data, pd.DataFrame)
                else pd.DataFrame(model_id_or_data)
            )
            if hasattr(self, "_last_model") and self._last_model is not None:
                if hasattr(self, "_last_preprocessor") and self._last_preprocessor is not None:
                    try:
                        X = self._last_preprocessor.transform(df)
                        if hasattr(X, "values"):
                            X = X.values
                    except Exception:
                        X = df.select_dtypes(include=[np.number]).fillna(0).values
                else:
                    X = df.select_dtypes(include=[np.number]).fillna(0).values
                return self._last_model.predict(X)
            elif hasattr(self, "_last_champion_id"):
                batch_df = self.predict_batch(self._last_champion_id, df)
                return batch_df["prediction"].to_numpy()

        model_id = str(model_id_or_data)
        input_data = input_data or {}
        model, preprocessor, metadata = self._load_model_artifacts(model_id)
        feature_names: List[str] = metadata.get("feature_names", [])
        problem_type: str = metadata.get("problem_type", "classification")

        df_input = pd.DataFrame([input_data])

        # Align columns
        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = np.nan

        df_input = df_input[feature_names] if feature_names else df_input

        if preprocessor is not None:
            try:
                X = preprocessor.transform(df_input).values
            except Exception:
                X = df_input.fillna(0).values
        else:
            X = df_input.fillna(0).values

        prediction = model.predict(X)[0]
        result: Dict[str, Any] = {
            "prediction": prediction.item() if hasattr(prediction, "item") else prediction,
            "model_version": model_id,
            "probability": None,
            "feature_contributions": {},
        }

        if problem_type == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            result["probability"] = {
                f"class_{i}": float(p) for i, p in enumerate(proba)
            }

        # SHAP for single prediction
        if _SHAP_AVAILABLE and feature_names:
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    contribs_arr = np.array(shap_vals[int(prediction)]).flatten()
                else:
                    contribs_arr = np.array(shap_vals).flatten()
                result["feature_contributions"] = {
                    f: float(v)
                    for f, v in zip(feature_names, contribs_arr)
                    if len(feature_names) == len(contribs_arr)
                }
            except Exception:
                pass

        return result

    def predict_batch(self, model_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run batch inference on a DataFrame.

        Adds ``prediction`` and (for classifiers) ``probability_max`` columns.

        Parameters
        ----------
        model_id : str
            Registered model identifier.
        df : pd.DataFrame
            Input feature frame.

        Returns
        -------
        pd.DataFrame
            Original frame with appended prediction columns.
        """
        model, preprocessor, metadata = self._load_model_artifacts(model_id)
        feature_names: List[str] = metadata.get("feature_names", [])
        problem_type: str = metadata.get("problem_type", "classification")

        df_out = df.copy()

        # Align columns
        df_features = df_out.copy()
        for col in feature_names:
            if col not in df_features.columns:
                df_features[col] = np.nan

        if feature_names:
            df_features = df_features[feature_names]

        if preprocessor is not None:
            try:
                X = preprocessor.transform(df_features).values
            except Exception:
                X = df_features.fillna(0).values
        else:
            X = df_features.fillna(0).values

        preds = model.predict(X)
        df_out["prediction"] = preds

        if problem_type == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            df_out["probability_max"] = proba.max(axis=1)

        logger.info("Batch prediction — model_id=%s, n_samples=%d", model_id, len(df))
        return df_out

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def get_training_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Return the current status of an experiment.

        Parameters
        ----------
        experiment_id : str
            The UUID returned by :meth:`train`.

        Returns
        -------
        dict
            Keys: ``id``, ``name``, ``status``, ``started_at``, optional ``error``.
        """
        record = self._experiments.get(experiment_id)
        if record is None:
            return {"id": experiment_id, "status": "not_found"}

        out = {
            "id": record["id"],
            "name": record.get("name"),
            "status": record["status"],
            "started_at": record.get("started_at"),
        }
        if "error" in record:
            out["error"] = record["error"]
        return out

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        Return a summary list of all experiments managed by this orchestrator.

        Returns
        -------
        list[dict]
            Each entry has ``id``, ``name``, ``status``, ``started_at``.
        """
        return [
            {
                "id": v["id"],
                "name": v.get("name"),
                "status": v["status"],
                "started_at": v.get("started_at"),
            }
            for v in self._experiments.values()
        ]

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _detect_problem_type(self, target: pd.Series) -> str:
        """
        Heuristically detect whether the target represents a classification
        or regression task.

        Decision rules (in order):
        1. Non-numeric dtype → classification.
        2. ≤20 unique integer values → classification.
        3. Fraction of unique values > 5 % of total rows → regression.
        4. Default → regression.
        """
        if not pd.api.types.is_numeric_dtype(target):
            return "classification"
        n_unique = target.nunique()
        if n_unique <= 2:
            return "classification"
        if n_unique <= 20 and pd.api.types.is_integer_dtype(target):
            return "classification"
        if n_unique / len(target) > 0.05:
            return "regression"
        return "regression"

    def _compute_metrics(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray,
        problem_type: str,
    ) -> Dict[str, float]:
        """Compute hold-out test metrics appropriate for *problem_type*."""
        preds = model.predict(X_test)
        metrics: Dict[str, float] = {}

        if problem_type == "classification":
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
            n_classes = len(np.unique(y_test))
            avg = "binary" if n_classes == 2 else "weighted"
            metrics["f1_score"] = float(f1_score(y_test, preds, average=avg, zero_division=0))
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)
                    if n_classes == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
                        )
                except Exception:
                    pass
        else:
            metrics["mae"] = float(mean_absolute_error(y_test, preds))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
            metrics["r2"] = float(r2_score(y_test, preds))
            ss_res = np.sum((y_test - preds) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            metrics["mape"] = float(
                np.mean(np.abs((y_test - preds) / np.where(y_test == 0, 1e-9, y_test))) * 100
            )

        return metrics

    def _load_model_artifacts(
        self, model_id: str
    ) -> Tuple[BaseEstimator, Optional[_DataPreprocessor], Dict[str, Any]]:
        """Load model, preprocessor, and metadata from the registry."""
        model_dir = Path(self.config.MODEL_REGISTRY_PATH) / model_id
        if not model_dir.exists():
            raise FileNotFoundError(f"No model found with id '{model_id}' in registry.")

        model = joblib.load(model_dir / "model.joblib")

        preprocessor_path = model_dir / "preprocessor.joblib"
        preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None

        meta_path = model_dir / "metadata.json"
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)

        return model, preprocessor, metadata

    def _notify_progress(
        self,
        callback: Optional[Callable],
        step: str,
        pct: int,
        message: str,
    ) -> None:
        """Invoke the caller-supplied progress callback (if any)."""
        logger.debug("[%d%%] %s — %s", pct, step, message)
        if callback is not None:
            try:
                callback(step, pct, message)
            except Exception as exc:
                logger.warning("Progress callback raised: %s", exc)
