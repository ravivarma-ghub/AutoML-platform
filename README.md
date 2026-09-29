# 🚀 Intelligent AutoML Platform

> **Automated End-to-End Machine Learning, Model Governance & Inference System**  
> *Production-ready tabular AutoML engine featuring automated data validation, profiling, feature engineering, Optuna hyperparameter optimization, model benchmarking, MLflow tracking, SHAP explainability, FastAPI serving, and real-time drift monitoring.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org)
[![Optuna](https://img.shields.io/badge/Optuna-3.4+-2980B9.svg)](https://optuna.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-0194E2.svg?logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED.svg?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Capabilities](#-key-capabilities)
- [Project Directory Structure](#-project-directory-structure)
- [Technology Stack](#-technology-stack)
- [Quick Start Guide](#-quick-start-guide)
  - [Option A: Docker Compose (Recommended)](#option-a-docker-compose-recommended)
  - [Option B: Local Development](#option-b-local-development)
- [Core Pipeline Walkthrough](#-core-pipeline-walkthrough)
  - [1. Data Validation Layer](#1-data-validation-layer)
  - [2. Automated Data Profiling](#2-automated-data-profiling)
  - [3. Feature Engineering & Preprocessing](#3-feature-engineering--preprocessing)
  - [4. Multi-Strategy Feature Selection](#4-multi-strategy-feature-selection)
  - [5. Model Library & Benchmarking](#5-model-library--benchmarking)
  - [6. Hyperparameter Optimization (Optuna)](#6-hyperparameter-optimization-optuna)
  - [7. Experiment Tracking & Model Registry](#7-experiment-tracking--model-registry)
  - [8. Model Explainability (SHAP)](#8-model-explainability-shap)
  - [9. Data & Prediction Drift Monitoring](#9-data--prediction-drift-monitoring)
- [API Reference](#-api-reference)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Resume & Interview Talking Points](#-resume--interview-talking-points)

---

## 🏛 System Architecture

```
                    CSV / Parquet Dataset
                             │
                             ▼
                    Data Validation Layer
       (Shape, Nulls, Dups, Leakage, Infinite, Cardinality)
                             │
                             ▼
                    Data Profiling Engine
           (Stats, Skewness, Outliers, Distributions)
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
            Classification          Regression
                  │                     │
                  └──────────┬──────────┘
                             ▼
                  Feature Pipeline Engine
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
    Numeric             Categorical             Datetime
 (IQR Clip + Scale)  (OHE / Ordinal / Freq)   (Date Parts)
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             ▼
                  Feature Selection Suite
       (Variance Threshold + Correlation Filter + KBest / Model)
                             │
                             ▼
                  Model Library Execution
       (8 Classifiers / 9 Regressors across K-Fold / StratifiedKFold)
                             │
                             ▼
               Hyperparameter Optimization
            (Optuna TPE / RandomSearch / GridSearch)
                             │
                             ▼
                  Model Benchmarking Rank
               (CV Mean ± Std + Holdout Test)
                             │
                             ▼
                   Champion Model Pick
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
            Model Registry       Evaluation Report
        (Dev → Champion → Prod)   (Standalone HTML + JSON)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
    FastAPI API        MLflow Server
 (Inference & REST)  (Experiment Tracking)
        │                   │
        ▼                   ▼
    Streamlit UI     Drift Monitoring
 (Interactive App)   (PSI & KS-Test Engine)
```

---

## ✨ Key Capabilities

1. **Intelligent Problem Type Detection:** Dynamically determines whether the problem is binary classification, multi-class classification, or regression based on unique target values, distributions, and dtypes (with manual override options).
2. **Robust 10-Point Data Validation:** Flags blocking errors and non-blocking warnings including target leakage, constant columns, duplicate rows, high cardinality, and extreme class imbalance.
3. **Automated Feature Engineering:**
   - **Datetime:** Automatically expands dates into `year`, `month`, `day`, `day_of_week`, `quarter`, `is_weekend`, `hour`, and `days_since_min`.
   - **Numeric:** Outlier clipping using IQR fences followed by median imputation and standard scaling.
   - **Categorical:** Dynamic dispatching: Low cardinality ($\le 10$) $\rightarrow$ One-Hot Encoding; Medium ($11-50$) $\rightarrow$ Ordinal Encoding; High ($> 50$) $\rightarrow$ Frequency Encoding.
4. **Multi-Stage Feature Selection:** Cascaded pruning (Variance Filter $\rightarrow$ Collinear Correlation Pruning $\rightarrow$ Model-based `SelectFromModel` with Random Forest).
5. **Comprehensive Algorithm Suite:**
   - **Classification:** Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, KNN, SVM.
   - **Regression:** Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting.
6. **Advanced Optimization with Optuna:** Bayesian hyperparameter search using Tree-structured Parzen Estimator (TPE) sampler and Median pruning for rapid convergence.
7. **Strict Cross-Validation:** Stratified 5-Fold for classification and standard K-Fold for regression; displays CV Mean and CV Standard Deviation for stability assessment.
8. **Enterprise Model Governance:**
   - Lifecycle stages: `development` $\rightarrow$ `validation` $\rightarrow$ `champion` $\rightarrow$ `production`.
   - Model artifacts, preprocessing pipelines, and schema metadata serialized as versioned assets.
9. **Explainable AI (XAI):** Built-in TreeSHAP / KernelSHAP calculating global feature importances and local per-prediction attribution waterfalls.
10. **Data & Concept Drift Monitoring:** Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistical tests for continuous surveillance, triggering retraining workflows.

---

## 📂 Project Directory Structure

```
automl-platform/
│
├── app/
│   ├── api/                     # FastAPI routing layer
│   │   ├── routes_models.py     # Registry, metadata & promotion endpoints
│   │   ├── routes_predict.py    # Single & batch inference + logging
│   │   └── routes_train.py      # Upload, async training & job status
│   │
│   ├── core/                    # System infrastructure
│   │   ├── config.py            # Pydantic Settings & environment loader
│   │   ├── database.py          # SQLAlchemy 2.x ORM models & session
│   │   └── logging.py           # Colorlog console & rotating file logging
│   │
│   ├── data/                    # Ingestion & validation
│   │   ├── loader.py            # CSV/Parquet/Excel multi-format loader
│   │   ├── profiler.py          # Column statistics & distribution profiling
│   │   └── validator.py         # 10-point data quality analysis
│   │
│   ├── evaluation/              # Metrics & reporting
│   │   ├── metrics.py           # Classification & regression score engine
│   │   └── reports.py           # Standalone HTML report & Matplotlib visualizer
│   │
│   ├── features/                # Preprocessing & selection
│   │   ├── feature_engineering.py # Datetime extraction & smart type mapping
│   │   ├── preprocessing.py     # Custom Sklearn transformers (IQR/Frequency)
│   │   └── selection.py         # Variance, Correlation, KBest, SelectFromModel
│   │
│   ├── models/                  # Modeling & lifecycle
│   │   ├── classification.py    # 8 classification estimators & hyperparam spaces
│   │   ├── regression.py        # 9 regression estimators & hyperparam spaces
│   │   ├── registry.py          # Versioned filesystem model registry
│   │   └── tuning.py            # Optuna, RandomizedSearch & GridSearch tuners
│   │
│   ├── pipeline/                # Orchestration & monitoring
│   │   ├── monitoring.py        # PSI & KS-test data drift detector
│   │   └── orchestrator.py      # End-to-end AutoML pipeline coordinator
│   │
│   └── main.py                  # FastAPI application entry point
│
├── frontend/
│   └── streamlit_app.py         # 7-page interactive Streamlit dashboard
│
├── tests/                       # Complete Pytest test suite
│   ├── conftest.py              # Test fixtures & synthetic datasets
│   ├── test_api.py              # FastAPI endpoint tests
│   ├── test_data_profiler.py    # Profiler tests
│   ├── test_data_validator.py   # Validation check tests
│   ├── test_feature_engineering.py # Preprocessing & engineering tests
│   ├── test_feature_selection.py   # Selection tests
│   ├── test_models.py           # Library & tuning tests
│   ├── test_monitoring.py       # PSI & drift monitoring tests
│   └── test_orchestrator.py     # Full end-to-end pipeline tests
│
├── configs/
│   └── config.yaml              # Global default configuration
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # Automated test, lint & coverage workflow
│       └── cd.yml               # Production deployment pipeline
│
├── alembic/                     # Database migrations
│   ├── env.py
│   └── script.py.mako
│
├── models/                      # Saved versioned models (.gitkeep)
├── reports/                     # Generated HTML reports (.gitkeep)
├── logs/                        # Application log rotation (.gitkeep)
│
├── Dockerfile                   # Multi-stage production container
├── docker-compose.yml           # Compose specification (API, UI, MLflow, DB)
├── Makefile                     # Developer productivity shortcuts
├── requirements.txt             # Pinned production dependencies
├── pytest.ini                   # Pytest test configuration
├── .env.example                 # Example environment template
└── README.md                    # Project documentation
```

---

## 🛠 Technology Stack

| Domain | Technology | Rationale |
|---|---|---|
| **Core Language** | Python 3.11 | High performance, modern type hinting, pattern matching |
| **Data Manipulation** | Pandas, NumPy, PyArrow | Fast tabular processing and parquet vectorization |
| **Machine Learning** | Scikit-learn | Proven, reliable estimator and transformer interface |
| **Optimization** | Optuna | State-of-the-art Bayesian optimization with pruning |
| **Explainability** | SHAP (SHapley Additive exPlanations) | Game-theoretic model interpretability |
| **Experiment Tracking** | MLflow | Model artifact versioning and run metric logging |
| **Backend REST API** | FastAPI, Uvicorn, Pydantic 2.0 | Asynchronous, auto-documented OpenAPI specification |
| **Frontend UI** | Streamlit, Plotly | Interactive reactive analytics dashboard |
| **Database & ORM** | PostgreSQL / SQLite, SQLAlchemy 2.x | ACID compliant logging and experiment audits |
| **Containerization** | Docker, Docker Compose | Multi-stage reproducible deployment builds |
| **DevOps & CI/CD** | GitHub Actions, Ruff, Pytest | Automated static analysis, linting, and regression testing |

---

## ⚡ Quick Start Guide

### Option A: Docker Compose (Recommended)

Run the entire platform (FastAPI, Streamlit, MLflow, PostgreSQL) in isolated containers:

```bash
# 1. Clone repository
git clone https://github.com/your-username/automl-platform.git
cd automl-platform

# 2. Configure environment
cp .env.example .env

# 3. Launch the container cluster
docker compose up -d --build
```

**Services will be live at:**
- 🖥 **Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)
- 🔌 **FastAPI REST API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- 📊 **MLflow Tracking Server:** [http://localhost:5000](http://localhost:5000)

---

### Option B: Local Development

```bash
# 1. Create and activate a Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Unix:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the FastAPI server (terminal 1)
make run-api
# or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Launch the Streamlit dashboard (terminal 2)
make run-ui
# or: streamlit run frontend/streamlit_app.py --server.port 8501
```

---

## 🔍 Core Pipeline Walkthrough

### 1. Data Validation Layer
Before training begins, incoming data is audited through `DataValidator`.

```
DATA VALIDATION REPORT
══════════════════════════════════════════════
  Rows                    50,000
  Columns                     38
  Duplicate Rows             124  ⚠
  Infinite Values              0  ✓
  Missing Value Columns       3   ⚠

MISSING VALUES
──────────────────────────────────────────────
  income                    2.1%  ⚠
  age                       0.3%  ✓
  occupation                1.7%  ⚠

CONSTANT COLUMNS
──────────────────────────────────────────────
  customer_id_hash                ✗ DROPPED

HIGH CARDINALITY COLUMNS
──────────────────────────────────────────────
  city               3,421 unique values  ⚠

POTENTIAL LEAKAGE COLUMNS
──────────────────────────────────────────────
  account_closed_date              ⚠ REVIEW

TARGET DISTRIBUTION (churn)
──────────────────────────────────────────────
  0 (No Churn)             82.0%
  1 (Churn)                18.0%
  Class Imbalance           ⚠

OVERALL STATUS: ⚠ WARNINGS - Validated with warnings
```

---

### 2. Automated Data Profiling
`DataProfiler` computes detailed statistical signatures for all attributes:
- **Numeric Features:** Mean, standard deviation, skewness, kurtosis, quartiles ($Q_1, Q_2, Q_3$), and IQR-based outlier counts.
- **Categorical Features:** Unique count, cardinalities, mode, and top-20 frequency distributions.
- **Collinearity Matrix:** Automated Pearson correlation ranking identifying tightly coupled variables ($|r| > 0.85$).

---

### 3. Feature Engineering & Preprocessing
`FeaturePreprocessor` constructs a clean Scikit-Learn `ColumnTransformer`:
- **Dates:** Transformed into cyclical and numeric components: `year`, `month`, `day`, `day_of_week`, `quarter`, `is_weekend`, `hour`.
- **Outliers:** Custom `OutlierClipper` handles extreme values via $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ boundaries.
- **Missing Data:** Median imputation for continuous variables; constant token (`"missing"`) for nominal values.
- **Encoding:** Low-cardinality nominals receive `OneHotEncoder(handle_unknown='ignore')`; high-cardinality features receive frequency-ratio encoding via custom `FrequencyEncoder`.

---

### 4. Multi-Strategy Feature Selection
Feature selection prunes noise and redundant attributes:
$$\text{Raw Features} \xrightarrow{\text{Variance} > 0.01} \text{Non-constant} \xrightarrow{\text{Collinear Filter} < 0.95} \text{Uncorrelated} \xrightarrow{\text{SelectFromModel}} \text{Optimal Subset}$$

---

### 5. Model Library & Benchmarking
The platform trains and evaluates models using stratified/standard K-Fold cross-validation:

```
MODEL BENCHMARKING RESULTS
══════════════════════════════════════════════════════════════════════
  Rank  Model                   CV Mean   CV Std   Test F1   Time(s)
══════════════════════════════════════════════════════════════════════
  1     Gradient Boosting        0.881     0.008     0.883    38.2
  2     Random Forest            0.872     0.011     0.874    21.4
  3     Hist Gradient Boosting   0.870     0.009     0.871    12.3
  4     Extra Trees              0.865     0.013     0.869    18.7
  5     Logistic Regression      0.821     0.014     0.819     2.1
══════════════════════════════════════════════════════════════════════
  ★ Champion: Gradient Boosting  (CV F1: 0.881 ± 0.008)
```

---

### 6. Hyperparameter Optimization (Optuna)
The platform uses Optuna's Tree-structured Parzen Estimator (TPE) algorithm to navigate non-convex parameter spaces.

```python
# Bayesian Search Space Definition (app/models/classification.py)
"random_forest": lambda trial: {
    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
    "max_depth": trial.suggest_int("max_depth", 3, 30),
    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
}
```

---

### 7. Experiment Tracking & Model Registry
- **Tracking:** Training jobs log parameters, metrics, model artifacts, and evaluation plots directly to **MLflow**.
- **Model Registry:** Organized file-based and DB-backed storage directory:
```
models/
└── churn_dataset/
    └── gradient_boosting/
        ├── v1/
        │   ├── model.pkl
        │   ├── preprocessor.pkl
        │   └── metadata.json
        └── v2/ (champion)
```
- **Lifecycle Promotion:** Models can be promoted through `development` $\rightarrow$ `validation` $\rightarrow$ `champion` $\rightarrow$ `production`.

---

### 8. Model Explainability (SHAP)
Integrated SHAP explanations provide both global and local visibility:
- **Global Feature Importance:** Summary bar charts and beeswarm plots.
- **Local Prediction Explanations:** Real-time feature attribution returned with each prediction request:
```json
{
  "prediction": 1,
  "probability": 0.874,
  "shap_contributions": {
    "MonthlyCharges": +0.312,
    "Contract_Month-to-month": +0.241,
    "Tenure": -0.194,
    "InternetService_Fiber": +0.121
  }
}
```

---

### 9. Data & Prediction Drift Monitoring
`ModelMonitor` detects shifts between baseline training data and live inference traffic using the **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests:

$$\text{PSI} = \sum \left( \text{Actual}\% - \text{Expected}\% \right) \times \ln\left(\frac{\text{Actual}\%}{\text{Expected}\%}\right)$$

| PSI Value | Status | Action |
|---|---|---|
| $\text{PSI} < 0.10$ | No Drift | Normal operation |
| $0.10 \le \text{PSI} < 0.20$ | Slight Drift | Flag for review |
| $\text{PSI} \ge 0.20$ | Significant Drift | Automatic retraining triggered |

---

## 📡 API Reference

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Cluster health check and system metadata |
| `POST` | `/api/v1/train/upload` | Multipart upload for training datasets |
| `POST` | `/api/v1/train/start` | Launch asynchronous AutoML training job |
| `GET` | `/api/v1/train/status/{job_id}` | Real-time training progress polling |
| `GET` | `/api/v1/train/experiments` | List all historical experiments |
| `GET` | `/api/v1/train/report/{exp_id}` | Fetch standalone HTML evaluation report |
| `POST` | `/api/v1/predict` | Single JSON record inference + SHAP explanations |
| `POST` | `/api/v1/predict/batch` | Bulk CSV inference processing |
| `GET` | `/api/v1/predict/logs` | Historical prediction audit trail |
| `GET` | `/api/v1/models` | List all registered model versions |
| `GET` | `/api/v1/models/champion` | Retrieve current production champion model |
| `POST` | `/api/v1/models/{id}/promote` | Update model stage (dev / champion / prod) |
| `GET` | `/api/v1/models/{id}/monitoring` | Run data drift diagnostics on live data |

---

## 🖥 Streamlit Dashboard

The frontend interface (`frontend/streamlit_app.py`) provides 7 distinct screens:

1. 🏠 **Executive Dashboard:** System health, active models, performance metrics, and recent training runs.
2. 🚀 **Model Training Studio:** Upload data, configure problem type, select algorithms, tune with Optuna, and track progress live.
3. 📊 **Model Comparison Matrix:** Multi-model radar charts, ROC curves, and cross-validation comparisons.
4. 🔮 **Prediction & SHAP Center:** Manual form input or batch CSV uploads with probability gauges and SHAP waterfall plots.
5. 🔍 **Data Profiler:** Column statistics, missing value matrices, and correlation heatmaps.
6. 📈 **Drift Monitoring:** PSI calculation, KS drift testing, and automated retraining triggers.
7. 📋 **Experiment Registry:** Historical runs with one-click model promotion controls.

---

## 🧪 Testing & Quality Assurance

The platform includes comprehensive test suites across every layer:

```bash
# Run all unit and integration tests
pytest

# Run tests with coverage report
pytest --cov=app --cov-report=term-missing --cov-report=html

# Run code style and lint check
ruff check .
```

---

## 💼 Resume & Interview Talking Points

### Recommended Resume Bullets:

- **Intelligent AutoML Platform | Python, Scikit-learn, FastAPI, MLflow, Optuna, SHAP, Streamlit, Docker**
  - Architected a production-grade AutoML platform with automated validation, statistical profiling, feature engineering (IQR outlier clipping, frequency/target encoding, datetime decomposition), and multi-stage selection.
  - Implemented automated model benchmarking across 17 estimators with stratified K-fold cross-validation and Optuna Bayesian hyperparameter search (TPE), tracking experiments and versioned artifacts in MLflow.
  - Built an asynchronous FastAPI inference service with SHAP explainability and a 7-page Streamlit analytics dashboard supporting single-sample and batch predictions.
  - Designed automated drift detection using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests, enabling continuous monitoring and automated retraining workflows.

### High-Value Interview Discussion Topics:
1. **Handling Edge Cases in Tabular Data:** How the custom `OutlierClipper` preserves distributional shape better than standard Winsorizing, and how `FrequencyEncoder` handles high-cardinality categorical features without exploding dimensionality.
2. **Preventing Data Leakage:** Validation checks specifically identify target leakage by analyzing timestamp sequences and high correlations before splitting.
3. **Bayesian Optimization vs Grid Search:** Why Optuna's TPE sampler paired with median pruning achieves higher CV scores in fewer iterations compared to exhaustive or random search.
4. **Drift Detection & Retraining Strategy:** How PSI thresholds ($\ge 0.2$) and KS-test p-values ($p < 0.05$) work together to differentiate between seasonal variation and true distribution drift before triggering automated retraining.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
# AutoML-platform
