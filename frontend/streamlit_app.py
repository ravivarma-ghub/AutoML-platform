"""
Intelligent AutoML Platform — Streamlit Frontend
=================================================
Multi-page Streamlit application that integrates with the FastAPI backend.

Pages
-----
1. 🏠  Home Dashboard
2. 🚀  Train Model
3. 📊  Model Comparison
4. 🔮  Make Predictions
5. 🔍  Data Profiler
6. 📈  Model Monitoring
7. 📋  Experiment Logs
"""

from __future__ import annotations

import io
import os
import time
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

API_BASE: str = os.getenv("API_URL", "http://localhost:8000")

PAGES = [
    "🏠 Home Dashboard",
    "🚀 Train Model",
    "📊 Model Comparison",
    "🔮 Make Predictions",
    "🔍 Data Profiler",
    "📈 Model Monitoring",
    "📋 Experiment Logs",
]

CLASSIFICATION_MODELS = [
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "LogisticRegression",
    "SVM",
    "KNN",
    "ExtraTreesClassifier",
    "GradientBoosting",
]

REGRESSION_MODELS = [
    "RandomForestRegressor",
    "XGBoostRegressor",
    "LightGBMRegressor",
    "CatBoostRegressor",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
]

FEATURE_SELECTION_METHODS = [
    "auto",
    "correlation",
    "mutual_info",
    "rfe",
    "lasso",
    "none",
]

HP_METHODS = ["optuna", "random", "grid"]

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="AutoML Platform",
    page_icon="🤖",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 60%, #16213e 100%);
    border-right: 1px solid #2d2d4e;
}
[data-testid="stSidebar"] * {
    color: #e0e0f0 !important;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 10px 14px !important;
    transition: background 0.2s;
    border: 1px solid transparent;
    font-size: 14px;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.25);
    border-color: #6366f1;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:has(input:checked) + div label {
    background: rgba(99,102,241,0.4);
    border-color: #818cf8;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1e1e3f 0%, #252550 100%);
    border: 1px solid #3d3d6b;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.3);
}
.metric-card .metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-card .metric-label {
    font-size: 0.85rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.metric-card .metric-delta {
    font-size: 0.8rem;
    color: #34d399;
    margin-top: 6px;
}

/* ── Section headers ─────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e0e0f0;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #3d3d6b;
}

/* ── Status badges ───────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-green  { background: #065f46; color: #6ee7b7; }
.badge-yellow { background: #78350f; color: #fcd34d; }
.badge-red    { background: #7f1d1d; color: #fca5a5; }
.badge-blue   { background: #1e3a5f; color: #93c5fd; }
.badge-purple { background: #3b0764; color: #d8b4fe; }

/* ── Champion banner ─────────────────────────────────────────────────────── */
.champion-banner {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 16px 24px;
    margin: 16px 0;
    text-align: center;
}
.champion-banner h2 {
    color: #6ee7b7;
    margin: 0;
    font-size: 1.4rem;
}

/* ── Prediction result ───────────────────────────────────────────────────── */
.prediction-result {
    background: linear-gradient(135deg, #1e3a5f, #1e1e5f);
    border: 2px solid #6366f1;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.prediction-value {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Health dot ──────────────────────────────────────────────────────────── */
.health-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
}
.health-dot.ok      { background: #10b981; box-shadow: 0 0 6px #10b981; }
.health-dot.warn    { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
.health-dot.error   { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

/* ── Drift cell colours ──────────────────────────────────────────────────── */
.drift-ok       { color: #34d399; font-weight: 600; }
.drift-warn     { color: #fbbf24; font-weight: 600; }
.drift-critical { color: #f87171; font-weight: 600; }

/* ── Info box ────────────────────────────────────────────────────────────── */
.info-box {
    background: rgba(99,102,241,0.1);
    border-left: 4px solid #6366f1;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #c7d2fe;
    font-size: 0.9rem;
}

/* ── Step tracker ────────────────────────────────────────────────────────── */
.step-tracker {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
}
.step-item {
    flex: 1;
    text-align: center;
    padding: 8px 4px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
}
.step-active   { background: #4f46e5; color: #fff; }
.step-done     { background: #065f46; color: #6ee7b7; }
.step-pending  { background: #1e1e3f; color: #6b7280; border: 1px solid #3d3d6b; }

/* ── Quick-start card ────────────────────────────────────────────────────── */
.qs-card {
    background: #1e1e3f;
    border: 1px solid #3d3d6b;
    border-radius: 10px;
    padding: 16px;
    height: 100%;
}
.qs-card h4 { color: #a78bfa; margin: 0 0 8px; }
.qs-card p  { color: #9ca3af; font-size: 0.85rem; margin: 0; }

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr.styled { border: none; border-top: 1px solid #2d2d4e; margin: 20px 0; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def api_get(endpoint: str, params: dict | None = None, timeout: int = 10) -> dict | None:
    """Perform a GET request against the FastAPI backend.

    Parameters
    ----------
    endpoint:
        Path relative to API_BASE (e.g. ``"/health"``).
    params:
        Optional query parameters.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    Parsed JSON response dict, or ``None`` on failure.
    """
    url = f"{API_BASE}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at **{API_BASE}**. Is the backend running?")
    except requests.exceptions.HTTPError as exc:
        st.error(f"❌ API error {exc.response.status_code}: {exc.response.text[:300]}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Unexpected error: {exc}")
    return None


def api_post(
    endpoint: str,
    data: dict | None = None,
    files: dict | None = None,
    timeout: int = 30,
) -> dict | None:
    """Perform a multipart/form-data POST request.

    Parameters
    ----------
    endpoint:
        Path relative to API_BASE.
    data:
        Form fields.
    files:
        Files to upload (requests-style dict).
    timeout:
        Request timeout in seconds.

    Returns
    -------
    Parsed JSON response dict, or ``None`` on failure.
    """
    url = f"{API_BASE}{endpoint}"
    try:
        resp = requests.post(url, data=data, files=files, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at **{API_BASE}**.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"❌ API error {exc.response.status_code}: {exc.response.text[:300]}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Unexpected error: {exc}")
    return None


def api_post_json(endpoint: str, data: dict, timeout: int = 30) -> dict | None:
    """Perform an ``application/json`` POST request.

    Parameters
    ----------
    endpoint:
        Path relative to API_BASE.
    data:
        JSON-serialisable payload.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    Parsed JSON response dict, or ``None`` on failure.
    """
    url = f"{API_BASE}{endpoint}"
    try:
        resp = requests.post(url, json=data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at **{API_BASE}**.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"❌ API error {exc.response.status_code}: {exc.response.text[:300]}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Unexpected error: {exc}")
    return None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def render_metric_card(label: str, value: Any, delta: str = "") -> str:
    """Return an HTML metric card string."""
    delta_html = f'<div class="metric-delta">↑ {delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


def badge_html(text: str, colour: str = "blue") -> str:
    """Return an inline HTML badge."""
    return f'<span class="badge badge-{colour}">{text}</span>'


def render_step_tracker(steps: list[str], current: int) -> str:
    """Render a horizontal step tracker.

    Parameters
    ----------
    steps:
        Step labels.
    current:
        0-based index of the active step.
    """
    items = []
    for i, label in enumerate(steps):
        if i < current:
            css = "step-done"
            prefix = "✓ "
        elif i == current:
            css = "step-active"
            prefix = ""
        else:
            css = "step-pending"
            prefix = ""
        items.append(f'<div class="step-item {css}">{prefix}{label}</div>')
    return f'<div class="step-tracker">{"".join(items)}</div>'


def load_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    """Load a CSV, Parquet, or Excel uploaded file into a DataFrame.

    Parameters
    ----------
    uploaded_file:
        Streamlit ``UploadedFile`` object.

    Returns
    -------
    DataFrame or ``None`` if loading fails.
    """
    if uploaded_file is None:
        return None
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".parquet"):
            return pd.read_parquet(uploaded_file)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        st.error("Unsupported file format. Please upload CSV, Parquet, or Excel.")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load file: {exc}")
        return None


def human_bytes(num: float) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def colour_psi(psi: float) -> str:
    """Return an HTML-coloured PSI string."""
    if psi < 0.1:
        return f'<span class="drift-ok">{psi:.4f} ✓</span>'
    if psi < 0.2:
        return f'<span class="drift-warn">{psi:.4f} ⚠</span>'
    return f'<span class="drift-critical">{psi:.4f} ✗</span>'


def psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """Calculate Population Stability Index.

    Parameters
    ----------
    reference:
        Reference distribution values.
    current:
        Current distribution values.
    buckets:
        Number of bins.

    Returns
    -------
    PSI score.
    """
    eps = 1e-8
    breakpoints = np.nanpercentile(reference, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    ref_pct = np.histogram(reference, breakpoints)[0] / len(reference) + eps
    cur_pct = np.histogram(current, breakpoints)[0] / len(current) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_stat(reference: np.ndarray, current: np.ndarray) -> float:
    """Compute a simple KS statistic between two samples.

    Parameters
    ----------
    reference:
        Reference distribution values.
    current:
        Current distribution values.

    Returns
    -------
    KS statistic (max absolute difference in CDFs).
    """
    combined = np.concatenate([reference, current])
    combined_sorted = np.sort(combined)
    cdf_ref = np.searchsorted(np.sort(reference), combined_sorted, side="right") / len(reference)
    cdf_cur = np.searchsorted(np.sort(current), combined_sorted, side="right") / len(current)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 24px 0 16px;'>
                <div style='font-size:3rem;'>🤖</div>
                <div style='font-size:1.2rem; font-weight:700; color:#a78bfa; letter-spacing:1px;'>
                    AutoML Platform
                </div>
                <div style='font-size:0.72rem; color:#6b7280; margin-top:4px;'>
                    Intelligent ML Automation
                </div>
            </div>
            <hr style='border-color:#2d2d4e; margin:0 0 16px;'>
            """,
            unsafe_allow_html=True,
        )

        selected = st.radio(
            "Navigation",
            PAGES,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#2d2d4e; margin:16px 0 12px;'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style='font-size:0.72rem; color:#4b5563; text-align:center;'>
                Backend: <code style='color:#6366f1;'>{API_BASE}</code>
            </div>
            <div style='font-size:0.68rem; color:#374151; text-align:center; margin-top:4px;'>
                v1.0.0 · AutoML Platform
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Page 1 — Home Dashboard
# ---------------------------------------------------------------------------


def page_home() -> None:
    """Render the Home Dashboard page."""
    st.markdown('<div class="section-header">🏠 Platform Overview</div>', unsafe_allow_html=True)

    # ── System health ──────────────────────────────────────────────────────
    health_data = api_get("/health")
    if health_data:
        status = health_data.get("status", "unknown").lower()
        dot_cls = "ok" if status == "ok" else "warn" if status == "degraded" else "error"
        status_label = status.upper()
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:20px;'>
                <span class="health-dot {dot_cls}"></span>
                <span style='color:#9ca3af;font-size:0.85rem;'>
                    System Status: <strong style='color:#e0e0f0;'>{status_label}</strong>
                </span>
                <span style='color:#4b5563;font-size:0.75rem;margin-left:auto;'>
                    Checked at {datetime.now().strftime('%H:%M:%S')}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">'
            '<span class="health-dot error"></span>'
            '<span style="color:#9ca3af;font-size:0.85rem;">System Status: <strong style="color:#ef4444;">OFFLINE</strong></span>'
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Overview metric cards ──────────────────────────────────────────────
    experiments_data = api_get("/train/experiments") or []
    total_experiments = len(experiments_data)
    total_models = sum(len(e.get("models", [])) for e in experiments_data) if experiments_data else 0

    best_accuracy: str = "—"
    if experiments_data:
        scores = [
            e.get("best_score", 0)
            for e in experiments_data
            if e.get("best_score") is not None
        ]
        if scores:
            best_accuracy = f"{max(scores):.4f}"

    datasets_processed = len({e.get("dataset_name", "") for e in experiments_data if e.get("dataset_name")})

    cols = st.columns(4)
    cards = [
        ("Total Experiments", total_experiments, ""),
        ("Best Score", best_accuracy, ""),
        ("Models Trained", total_models, ""),
        ("Datasets Used", datasets_processed, ""),
    ]
    for col, (label, value, delta) in zip(cols, cards):
        with col:
            st.markdown(render_metric_card(label, value, delta), unsafe_allow_html=True)

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Recent experiments ─────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown('<div class="section-header" style="font-size:1.1rem;">📋 Recent Experiments</div>', unsafe_allow_html=True)
        if experiments_data:
            rows = []
            for exp in experiments_data[-10:][::-1]:
                rows.append(
                    {
                        "ID": exp.get("experiment_id", "—")[:8] + "…" if len(str(exp.get("experiment_id", ""))) > 8 else exp.get("experiment_id", "—"),
                        "Dataset": exp.get("dataset_name", "—"),
                        "Task": exp.get("task_type", "—"),
                        "Best Model": exp.get("best_model", "—"),
                        "Score": f"{exp.get('best_score', 0):.4f}" if exp.get("best_score") is not None else "—",
                        "Status": exp.get("status", "—"),
                        "Started": exp.get("created_at", "—"),
                    }
                )
            df_recent = pd.DataFrame(rows)

            def _status_icon(s: str) -> str:
                icons = {"completed": "✅", "running": "⏳", "failed": "❌", "pending": "⏸"}
                return icons.get(s.lower(), s)

            df_recent["Status"] = df_recent["Status"].apply(_status_icon)
            st.dataframe(df_recent, use_container_width=True, hide_index=True)
        else:
            st.info("No experiments found. Start your first training run!")

    with col_right:
        st.markdown('<div class="section-header" style="font-size:1.1rem;">⚡ Quick Start</div>', unsafe_allow_html=True)
        quick_steps = [
            ("1️⃣ Upload", "Go to Train Model and upload your CSV dataset."),
            ("2️⃣ Configure", "Select your target column and tuning options."),
            ("3️⃣ Train", "Click Launch Training and watch the live progress."),
            ("4️⃣ Predict", "Use Make Predictions to run inference."),
        ]
        for icon, desc in quick_steps:
            st.markdown(
                f'<div class="qs-card" style="margin-bottom:8px;">'
                f"<h4>{icon}</h4><p>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    # ── Experiment status distribution ────────────────────────────────────
    if experiments_data:
        st.markdown("<hr class='styled'>", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="font-size:1.1rem;">📊 Experiment Summary</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        status_counts: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        for exp in experiments_data:
            s = exp.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
            t = exp.get("task_type", "unknown")
            task_counts[t] = task_counts.get(t, 0) + 1

        with c1:
            fig_status = px.pie(
                names=list(status_counts.keys()),
                values=list(status_counts.values()),
                title="Experiments by Status",
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                hole=0.4,
            )
            fig_status.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0f0",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_status, use_container_width=True)

        with c2:
            fig_task = px.bar(
                x=list(task_counts.keys()),
                y=list(task_counts.values()),
                title="Experiments by Task Type",
                color=list(task_counts.values()),
                color_continuous_scale="Viridis",
                labels={"x": "Task Type", "y": "Count"},
            )
            fig_task.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0f0",
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_task, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Train Model
# ---------------------------------------------------------------------------


def page_train() -> None:
    """Render the Train Model page with a 4-step wizard."""
    st.markdown('<div class="section-header">🚀 Train Model</div>', unsafe_allow_html=True)

    # Step state management
    if "train_step" not in st.session_state:
        st.session_state.train_step = 0
    if "train_df" not in st.session_state:
        st.session_state.train_df = None
    if "train_job_id" not in st.session_state:
        st.session_state.train_job_id = None
    if "train_results" not in st.session_state:
        st.session_state.train_results = None
    if "uploaded_file_bytes" not in st.session_state:
        st.session_state.uploaded_file_bytes = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None

    steps = ["📁 Upload", "⚙️ Configure", "⏳ Training", "📊 Results"]
    step = st.session_state.train_step
    st.markdown(render_step_tracker(steps, step), unsafe_allow_html=True)

    # ── Step 0 — Upload ───────────────────────────────────────────────────
    if step == 0:
        st.subheader("Step 1: Upload Dataset")
        uploaded = st.file_uploader(
            "Upload your dataset",
            type=["csv", "parquet", "xlsx", "xls"],
            help="Supported formats: CSV, Parquet, Excel",
        )
        if uploaded is not None:
            df = load_uploaded_file(uploaded)
            if df is not None:
                st.session_state.train_df = df
                st.session_state.uploaded_file_bytes = uploaded.getvalue()
                st.session_state.uploaded_file_name = uploaded.name

                st.success(f"✅ File loaded: **{uploaded.name}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{len(df):,}")
                c2.metric("Columns", str(df.shape[1]))
                c3.metric("Size", human_bytes(len(st.session_state.uploaded_file_bytes)))

                st.markdown("**Preview (first 5 rows):**")
                st.dataframe(df.head(), use_container_width=True)

                missing_pct = (df.isnull().sum().sum() / df.size * 100)
                if missing_pct > 0:
                    st.warning(f"⚠️ Dataset contains **{missing_pct:.1f}%** missing values.")

                col_types = df.dtypes.reset_index()
                col_types.columns = ["Column", "Type"]
                col_types["Nulls"] = df.isnull().sum().values
                col_types["Null %"] = (df.isnull().mean().values * 100).round(2)
                with st.expander("📋 Column Info"):
                    st.dataframe(col_types, use_container_width=True, hide_index=True)

                if st.button("Next: Configure Training →", type="primary"):
                    st.session_state.train_step = 1
                    st.rerun()

    # ── Step 1 — Configure ────────────────────────────────────────────────
    elif step == 1:
        st.subheader("Step 2: Configure Training")
        df: pd.DataFrame = st.session_state.train_df

        if df is None:
            st.error("No dataset loaded. Please go back to Step 1.")
            if st.button("← Back"):
                st.session_state.train_step = 0
                st.rerun()
            return

        col_names = df.columns.tolist()

        target_col = st.selectbox("🎯 Target Column", col_names, index=len(col_names) - 1)
        problem_type = st.selectbox(
            "📋 Problem Type",
            ["auto", "classification", "regression"],
            help="'auto' will infer from the target column.",
        )

        with st.expander("⚙️ Advanced Options", expanded=True):
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                feat_method = st.selectbox("Feature Selection Method", FEATURE_SELECTION_METHODS)
                hp_method = st.selectbox("Hyperparameter Tuning", HP_METHODS)
                optuna_trials = st.slider("Optuna Trials", 10, 200, 50, 5)
                enable_shap = st.toggle("Enable SHAP Explanations", value=True)
            with adv_c2:
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
                default_models = (
                    CLASSIFICATION_MODELS[:5]
                    if problem_type in ("auto", "classification")
                    else REGRESSION_MODELS[:5]
                )
                all_models = (
                    CLASSIFICATION_MODELS
                    if problem_type in ("auto", "classification")
                    else REGRESSION_MODELS
                )
                selected_models = st.multiselect(
                    "Models to Train",
                    all_models,
                    default=default_models,
                )

        st.markdown("<hr class='styled'>", unsafe_allow_html=True)
        bc1, bc2 = st.columns([1, 4])
        with bc1:
            if st.button("← Back"):
                st.session_state.train_step = 0
                st.rerun()
        with bc2:
            if st.button("🚀 Launch Training", type="primary"):
                if not selected_models:
                    st.error("Please select at least one model.")
                else:
                    config = {
                        "target_column": target_col,
                        "task_type": problem_type,
                        "feature_selection_method": feat_method,
                        "hp_method": hp_method,
                        "optuna_trials": optuna_trials,
                        "cv_folds": cv_folds,
                        "test_size": test_size,
                        "models": selected_models,
                        "enable_shap": enable_shap,
                    }
                    st.session_state.train_config = config
                    file_bytes = st.session_state.uploaded_file_bytes
                    file_name = st.session_state.uploaded_file_name

                    with st.spinner("Submitting training job…"):
                        result = api_post(
                            "/train/start",
                            data={k: str(v) for k, v in config.items()},
                            files={
                                "file": (
                                    file_name,
                                    io.BytesIO(file_bytes),
                                    "text/csv",
                                )
                            },
                        )
                    if result:
                        st.session_state.train_job_id = result.get("job_id") or result.get("experiment_id")
                        st.session_state.train_step = 2
                        st.rerun()
                    else:
                        st.error("Failed to start training. Check the backend logs.")

    # ── Step 2 — Training Progress ────────────────────────────────────────
    elif step == 2:
        st.subheader("Step 3: Training in Progress")
        job_id = st.session_state.train_job_id

        if not job_id:
            st.error("No active job found. Please restart training.")
            if st.button("← Restart"):
                st.session_state.train_step = 0
                st.rerun()
            return

        st.markdown(
            f'<div class="info-box">Job ID: <code>{job_id}</code></div>',
            unsafe_allow_html=True,
        )

        status_data = api_get(f"/train/status/{job_id}")

        if status_data:
            progress_val: float = status_data.get("progress", 0.0)
            status_str: str = status_data.get("status", "running")
            current_step: str = status_data.get("current_step", "Initialising…")
            logs: list[str] = status_data.get("logs", [])

            progress_bar = st.progress(min(progress_val, 1.0))

            col_status, col_step = st.columns([1, 3])
            col_status.metric("Status", status_str.capitalize())
            col_step.metric("Current Step", current_step)

            if logs:
                with st.expander("📜 Live Logs", expanded=True):
                    log_text = "\n".join(logs[-30:])
                    st.code(log_text, language="text")

            if status_str in ("completed", "failed", "error"):
                if status_str == "completed":
                    st.success("✅ Training completed!")
                    st.session_state.train_results = status_data.get("results") or status_data
                    if st.button("View Results →", type="primary"):
                        st.session_state.train_step = 3
                        st.rerun()
                else:
                    st.error(f"❌ Training failed: {status_data.get('error', 'Unknown error')}")
                    if st.button("← Start Over"):
                        for key in ("train_step", "train_job_id", "train_df", "train_results"):
                            st.session_state.pop(key, None)
                        st.rerun()
            else:
                with st.spinner("Training in progress… auto-refreshing every 3 seconds"):
                    time.sleep(3)
                st.rerun()
        else:
            st.warning("Waiting for job status…")
            time.sleep(3)
            st.rerun()

    # ── Step 3 — Results ──────────────────────────────────────────────────
    elif step == 3:
        st.subheader("Step 4: Training Results")
        results = st.session_state.train_results or {}

        champion: str = results.get("best_model", "—")
        best_score: float | None = results.get("best_score")
        metrics: dict = results.get("metrics", {})
        leaderboard: list[dict] = results.get("leaderboard", [])

        # Champion banner
        st.markdown(
            f"""
            <div class="champion-banner">
                <h2>🏆 Champion Model: {champion}</h2>
                <p style='color:#a7f3d0; margin:8px 0 0;'>
                    Best Score: <strong>{best_score:.4f}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric cards
        if metrics:
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                with col:
                    st.markdown(
                        render_metric_card(k.replace("_", " ").title(), f"{v:.4f}"),
                        unsafe_allow_html=True,
                    )

        st.markdown("<hr class='styled'>", unsafe_allow_html=True)

        # Leaderboard
        if leaderboard:
            st.markdown("**Model Leaderboard**")
            df_lb = pd.DataFrame(leaderboard)
            st.dataframe(df_lb, use_container_width=True, hide_index=True)

            # Bar chart
            if "model" in df_lb.columns and "score" in df_lb.columns:
                fig = px.bar(
                    df_lb,
                    x="model",
                    y="score",
                    color="score",
                    color_continuous_scale="Viridis",
                    title="Model Comparison",
                    labels={"model": "Model", "score": "Score"},
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0f0",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Download report
        report_payload = {
            "champion": champion,
            "best_score": best_score,
            "metrics": metrics,
            "leaderboard": leaderboard,
        }
        report_str = pd.DataFrame([report_payload]).to_csv(index=False)
        st.download_button(
            "⬇️ Download Report (CSV)",
            data=report_str,
            file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        if st.button("🔄 Start New Training"):
            for key in ("train_step", "train_job_id", "train_df", "train_results", "train_config"):
                st.session_state.pop(key, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Page 3 — Model Comparison
# ---------------------------------------------------------------------------


def page_comparison() -> None:
    """Render the Model Comparison page."""
    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

    experiments: list[dict] = api_get("/train/experiments") or []

    if not experiments:
        st.info("No experiments found yet. Train some models first!")
        return

    # ── Filters ───────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        all_tasks = sorted({e.get("task_type", "unknown") for e in experiments})
        all_datasets = sorted({e.get("dataset_name", "unknown") for e in experiments if e.get("dataset_name")})

        with fc1:
            sel_tasks = st.multiselect("Problem Type", all_tasks, default=all_tasks)
        with fc2:
            sel_datasets = st.multiselect("Dataset", all_datasets, default=all_datasets)
        with fc3:
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
            )

    filtered = [
        e for e in experiments
        if e.get("task_type", "unknown") in sel_tasks
        and e.get("dataset_name", "unknown") in sel_datasets
    ]

    if not filtered:
        st.warning("No experiments match the selected filters.")
        return

    # ── Metrics table ─────────────────────────────────────────────────────
    rows = []
    for exp in filtered:
        row: dict = {
            "Experiment ID": str(exp.get("experiment_id", ""))[:12],
            "Dataset": exp.get("dataset_name", "—"),
            "Task": exp.get("task_type", "—"),
            "Best Model": exp.get("best_model", "—"),
            "Best Score": exp.get("best_score"),
            "Status": exp.get("status", "—"),
            "Created": exp.get("created_at", "—"),
        }
        metrics = exp.get("metrics", {})
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)

    df_cmp = pd.DataFrame(rows)
    st.markdown("**All Experiments**")
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Bar chart — best score by experiment ──────────────────────────────
    df_plot = df_cmp.dropna(subset=["Best Score"]).copy()
    df_plot["Best Score"] = pd.to_numeric(df_plot["Best Score"], errors="coerce")

    if not df_plot.empty:
        st.subheader("Best Score per Experiment")
        fig_bar = px.bar(
            df_plot,
            x="Experiment ID",
            y="Best Score",
            color="Task",
            barmode="group",
            text="Best Model",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_bar.update_traces(textposition="outside", textfont_size=10)
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0f0",
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Radar chart for multi-metric comparison ───────────────────────────
    numeric_metrics = [c for c in df_cmp.columns if c not in (
        "Experiment ID", "Dataset", "Task", "Best Model", "Status", "Created"
    )]
    if len(numeric_metrics) >= 3 and len(df_cmp) >= 2:
        st.subheader("Multi-Metric Radar Chart")
        radar_cols = numeric_metrics[:6]
        fig_radar = go.Figure()
        for _, row_data in df_cmp.head(6).iterrows():
            vals = []
            for col in radar_cols:
                v = row_data.get(col)
                try:
                    vals.append(float(v) if v is not None else 0.0)
                except (ValueError, TypeError):
                    vals.append(0.0)
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=radar_cols + [radar_cols[0]],
                    fill="toself",
                    name=str(row_data.get("Experiment ID", "exp"))[:10],
                    opacity=0.6,
                )
            )
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(30,30,63,0.5)",
                radialaxis=dict(color="#6b7280", gridcolor="#2d2d4e"),
                angularaxis=dict(color="#9ca3af"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0f0",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────
    st.download_button(
        "⬇️ Export Comparison CSV",
        data=df_cmp.to_csv(index=False),
        file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Page 4 — Make Predictions
# ---------------------------------------------------------------------------


def page_predict() -> None:
    """Render the Make Predictions page."""
    st.markdown('<div class="section-header">🔮 Make Predictions</div>', unsafe_allow_html=True)

    # ── Model selection ───────────────────────────────────────────────────
    models_data: list[dict] = api_get("/models") or []

    if not models_data:
        st.info("No trained models available. Complete a training run first.")
        return

    # Champion first
    models_sorted = sorted(models_data, key=lambda m: m.get("is_champion", False), reverse=True)
    model_options = {
        f"{m.get('model_name', 'Unknown')} (v{m.get('version', '?')})"
        + (" 🏆" if m.get("is_champion") else ""): m.get("model_id") or m.get("id")
        for m in models_sorted
    }

    selected_label = st.selectbox("Select Model", list(model_options.keys()))
    selected_model_id = model_options[selected_label]

    # Model info
    model_info = next((m for m in models_sorted if (m.get("model_id") or m.get("id")) == selected_model_id), {})
    if model_info:
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Model", model_info.get("model_name", "—"))
        ic2.metric("Version", f"v{model_info.get('version', '?')}")
        ic3.metric("Task", model_info.get("task_type", "—"))
        score = model_info.get("best_score") or model_info.get("score")
        ic4.metric("Score", f"{score:.4f}" if score else "—")

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Input mode ────────────────────────────────────────────────────────
    input_mode = st.radio("Prediction Mode", ["Manual Input", "Batch Upload"], horizontal=True)

    features: list[dict] = model_info.get("features", [])
    feature_names: list[str] = (
        [f["name"] if isinstance(f, dict) else f for f in features]
        if features
        else model_info.get("feature_names", [])
    )

    # ── Manual input ──────────────────────────────────────────────────────
    if input_mode == "Manual Input":
        if not feature_names:
            st.info("No feature list available for this model. Please upload a batch file.")
        else:
            st.subheader("Enter Feature Values")
            form_cols = st.columns(min(3, len(feature_names)))
            input_values: dict[str, Any] = {}
            for i, feat in enumerate(feature_names):
                col = form_cols[i % len(form_cols)]
                feat_meta = next((f for f in features if isinstance(f, dict) and f.get("name") == feat), {})
                dtype = feat_meta.get("dtype", "float")
                if dtype in ("int", "integer"):
                    input_values[feat] = col.number_input(feat, value=0, step=1)
                elif dtype in ("float", "numeric"):
                    input_values[feat] = col.number_input(feat, value=0.0)
                elif dtype in ("bool", "boolean"):
                    input_values[feat] = col.checkbox(feat)
                else:
                    input_values[feat] = col.text_input(feat, value="")

            if st.button("🔮 Run Prediction", type="primary"):
                payload = {"model_id": selected_model_id, "features": input_values}
                with st.spinner("Running inference…"):
                    pred_result = api_post_json("/predict", payload)

                if pred_result:
                    _render_prediction_result(pred_result, model_info)

    # ── Batch upload ──────────────────────────────────────────────────────
    else:
        st.subheader("Batch Prediction via CSV Upload")
        batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
        if batch_file is not None:
            df_batch = load_uploaded_file(batch_file)
            if df_batch is not None:
                st.write(f"Loaded **{len(df_batch):,} rows** × {df_batch.shape[1]} columns")
                st.dataframe(df_batch.head(3), use_container_width=True)

                if st.button("🔮 Run Batch Prediction", type="primary"):
                    with st.spinner(f"Running predictions on {len(df_batch):,} rows…"):
                        result = api_post(
                            "/predict/batch",
                            data={"model_id": str(selected_model_id)},
                            files={
                                "file": (
                                    batch_file.name,
                                    io.BytesIO(batch_file.getvalue()),
                                    "text/csv",
                                )
                            },
                        )
                    if result:
                        predictions = result.get("predictions", [])
                        df_out = df_batch.copy()
                        df_out["prediction"] = predictions
                        if "probabilities" in result:
                            probs = result["probabilities"]
                            if isinstance(probs[0], list):
                                for ci, cls in enumerate(result.get("classes", [])):
                                    df_out[f"prob_{cls}"] = [p[ci] for p in probs]
                            else:
                                df_out["probability"] = probs
                        st.success(f"✅ Predictions complete for **{len(predictions):,}** rows.")
                        st.dataframe(df_out.head(20), use_container_width=True)
                        st.download_button(
                            "⬇️ Download Predictions CSV",
                            data=df_out.to_csv(index=False),
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )


def _render_prediction_result(result: dict, model_info: dict) -> None:
    """Render a single prediction result with gauge and SHAP chart.

    Parameters
    ----------
    result:
        API prediction response dict.
    model_info:
        Model metadata dict.
    """
    pred = result.get("prediction")
    probabilities: dict | list | None = result.get("probabilities")
    shap_values: dict | None = result.get("shap_values")

    st.markdown(
        f"""
        <div class="prediction-result">
            <div style='color:#9ca3af; font-size:0.85rem; margin-bottom:8px;'>PREDICTION</div>
            <div class="prediction-value">{pred}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probabilities is not None:
        st.markdown("**Class Probabilities**")
        if isinstance(probabilities, dict):
            classes = list(probabilities.keys())
            probs = list(probabilities.values())
        elif isinstance(probabilities, (list, np.ndarray)):
            classes_raw = result.get("classes", [f"Class {i}" for i in range(len(probabilities))])
            classes = classes_raw
            probs = list(probabilities)
        else:
            classes, probs = [], []

        if classes and probs:
            # Gauge for binary
            if len(classes) == 2:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=float(probs[1]) * 100,
                        title={"text": f"P({classes[1]})", "font": {"color": "#e0e0f0"}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#6366f1"},
                            "steps": [
                                {"range": [0, 30], "color": "#1e3a5f"},
                                {"range": [30, 70], "color": "#3b0764"},
                                {"range": [70, 100], "color": "#064e3b"},
                            ],
                            "threshold": {
                                "line": {"color": "#a78bfa", "width": 3},
                                "thickness": 0.8,
                                "value": 50,
                            },
                        },
                        number={"suffix": "%", "font": {"color": "#a78bfa"}},
                    )
                )
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0f0",
                    height=250,
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                fig_prob = px.bar(
                    x=classes,
                    y=[float(p) for p in probs],
                    labels={"x": "Class", "y": "Probability"},
                    color=[float(p) for p in probs],
                    color_continuous_scale="Viridis",
                    title="Class Probabilities",
                )
                fig_prob.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0f0",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

    # SHAP waterfall
    if shap_values:
        st.markdown("**SHAP Feature Contributions**")
        features_sorted = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        feat_names = [f[0] for f in features_sorted]
        feat_vals = [f[1] for f in features_sorted]
        colors = ["#34d399" if v > 0 else "#f87171" for v in feat_vals]

        fig_shap = go.Figure(
            go.Bar(
                x=feat_vals,
                y=feat_names,
                orientation="h",
                marker_color=colors,
            )
        )
        fig_shap.update_layout(
            title="SHAP Waterfall (Top 15 Features)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0f0",
            xaxis_title="SHAP Value",
            yaxis={"autorange": "reversed"},
            height=400,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Feature contribution table
        df_shap = pd.DataFrame({"Feature": feat_names, "SHAP Value": feat_vals})
        df_shap["Direction"] = df_shap["SHAP Value"].apply(lambda v: "↑ Positive" if v > 0 else "↓ Negative")
        st.dataframe(df_shap, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 5 — Data Profiler
# ---------------------------------------------------------------------------


def page_profiler() -> None:
    """Render the Data Profiler page."""
    st.markdown('<div class="section-header">🔍 Data Profiler</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload dataset to profile",
        type=["csv", "parquet", "xlsx", "xls"],
        key="profiler_upload",
    )

    if uploaded is None:
        st.info("Upload a dataset to start profiling.")
        return

    df = load_uploaded_file(uploaded)
    if df is None:
        return

    # ── Try API validation ────────────────────────────────────────────────
    try:
        val_result = api_post(
            "/data/validate",
            files={
                "file": (
                    uploaded.name,
                    io.BytesIO(uploaded.getvalue()),
                    "text/csv",
                )
            },
        )
        if val_result:
            with st.expander("📋 Validation Report", expanded=True):
                issues = val_result.get("issues", [])
                warnings = val_result.get("warnings", [])
                if not issues and not warnings:
                    st.success("✅ Dataset passed all validation checks.")
                if issues:
                    for issue in issues:
                        st.error(f"❌ {issue}")
                if warnings:
                    for warn in warnings:
                        st.warning(f"⚠️ {warn}")
    except Exception:  # noqa: BLE001
        pass  # Validation endpoint optional

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Basic stats ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", str(df.shape[1]))
    c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    c4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Column profile ─────────────────────────────────────────────────────
    st.subheader("Column Profile")
    profile_rows = []
    for col in df.columns:
        s = df[col]
        profile_rows.append(
            {
                "Column": col,
                "Type": str(s.dtype),
                "Non-Null": int(s.notna().sum()),
                "Null %": round(s.isna().mean() * 100, 2),
                "Unique": int(s.nunique()),
                "Mean": round(s.mean(), 4) if pd.api.types.is_numeric_dtype(s) else "—",
                "Std": round(s.std(), 4) if pd.api.types.is_numeric_dtype(s) else "—",
                "Min": s.min() if pd.api.types.is_numeric_dtype(s) else "—",
                "Max": s.max() if pd.api.types.is_numeric_dtype(s) else "—",
            }
        )
    df_profile = pd.DataFrame(profile_rows)
    st.dataframe(df_profile, use_container_width=True, hide_index=True)

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Correlation heatmap ───────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0f0",
            height=max(400, len(numeric_cols) * 40),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Distribution histograms ───────────────────────────────────────────
    if numeric_cols:
        st.subheader("Distribution Histograms")
        selected_hist_cols = st.multiselect(
            "Select columns to visualise",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))],
        )
        if selected_hist_cols:
            ncols_per_row = min(2, len(selected_hist_cols))
            for i in range(0, len(selected_hist_cols), ncols_per_row):
                row_cols = st.columns(ncols_per_row)
                for j, col_name in enumerate(selected_hist_cols[i : i + ncols_per_row]):
                    with row_cols[j]:
                        fig_hist = px.histogram(
                            df,
                            x=col_name,
                            nbins=40,
                            title=col_name,
                            color_discrete_sequence=["#6366f1"],
                            marginal="box",
                        )
                        fig_hist.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e0e0f0",
                            showlegend=False,
                            height=300,
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Missing values heatmap ────────────────────────────────────────────
    if df.isnull().any().any():
        st.subheader("Missing Values Heatmap")
        missing_matrix = df.isnull().astype(int)
        sample_size = min(200, len(missing_matrix))
        missing_sample = missing_matrix.head(sample_size)
        fig_mv = px.imshow(
            missing_sample.T,
            color_continuous_scale=["#1e1e3f", "#ef4444"],
            labels={"x": "Row Index", "y": "Column", "color": "Missing"},
            title=f"Missing Values (first {sample_size} rows)",
            aspect="auto",
        )
        fig_mv.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0f0",
            height=max(300, df.shape[1] * 22),
        )
        st.plotly_chart(fig_mv, use_container_width=True)
    else:
        st.success("✅ No missing values found in this dataset.")


# ---------------------------------------------------------------------------
# Page 6 — Model Monitoring
# ---------------------------------------------------------------------------


def page_monitoring() -> None:
    """Render the Model Monitoring page with drift analysis."""
    st.markdown('<div class="section-header">📈 Model Monitoring</div>', unsafe_allow_html=True)

    models_data: list[dict] = api_get("/models") or []

    if not models_data:
        st.info("No trained models found. Train a model first.")
        return

    model_options = {
        f"{m.get('model_name', 'Unknown')} (v{m.get('version', '?')})": m.get("model_id") or m.get("id")
        for m in models_data
    }
    selected_label = st.selectbox("Select Model to Monitor", list(model_options.keys()))
    selected_id = model_options[selected_label]

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    col_ref, col_cur = st.columns(2)
    with col_ref:
        st.markdown("**📚 Reference Dataset** (training data)")
        ref_file = st.file_uploader("Upload reference CSV", type=["csv"], key="ref_upload")
    with col_cur:
        st.markdown("**📡 Current Dataset** (production data)")
        cur_file = st.file_uploader("Upload current CSV", type=["csv"], key="cur_upload")

    if ref_file is None or cur_file is None:
        st.info("Upload both reference and current datasets to run drift analysis.")
        return

    df_ref = load_uploaded_file(ref_file)
    df_cur = load_uploaded_file(cur_file)

    if df_ref is None or df_cur is None:
        return

    if st.button("🔍 Run Drift Analysis", type="primary"):
        # ── Try API endpoint ───────────────────────────────────────────────
        api_result = api_post(
            f"/monitor/drift/{selected_id}",
            files={
                "reference_file": (ref_file.name, io.BytesIO(ref_file.getvalue()), "text/csv"),
                "current_file": (cur_file.name, io.BytesIO(cur_file.getvalue()), "text/csv"),
            },
        )

        if api_result and api_result.get("features"):
            drift_stats = api_result["features"]
        else:
            # Compute locally as fallback
            st.info("Computing drift statistics locally…")
            common_numeric = [
                c for c in df_ref.columns
                if c in df_cur.columns and pd.api.types.is_numeric_dtype(df_ref[c])
            ]
            drift_stats = []
            for col in common_numeric:
                ref_vals = df_ref[col].dropna().values
                cur_vals = df_cur[col].dropna().values
                if len(ref_vals) > 0 and len(cur_vals) > 0:
                    psi_val = psi(ref_vals, cur_vals)
                    ks_val = ks_stat(ref_vals, cur_vals)
                    drift_stats.append(
                        {"feature": col, "psi": psi_val, "ks": ks_val}
                    )

        if not drift_stats:
            st.warning("No numeric features found for drift analysis.")
            return

        # ── Build results table ────────────────────────────────────────────
        rows = []
        for feat in drift_stats:
            psi_v = feat.get("psi", 0.0)
            ks_v = feat.get("ks", 0.0)
            if psi_v < 0.1 and ks_v < 0.1:
                status = "✅ OK"
                css = "drift-ok"
            elif psi_v < 0.2 and ks_v < 0.2:
                status = "⚠️ WARNING"
                css = "drift-warn"
            else:
                status = "🚨 CRITICAL"
                css = "drift-critical"
            rows.append(
                {
                    "Feature": feat["feature"],
                    "PSI": round(psi_v, 4),
                    "KS Stat": round(ks_v, 4),
                    "Status": status,
                }
            )

        df_drift = pd.DataFrame(rows)

        # Overall recommendation
        critical_count = sum(1 for r in rows if "CRITICAL" in r["Status"])
        warn_count = sum(1 for r in rows if "WARNING" in r["Status"])
        if critical_count > 0:
            reco = "🚨 URGENT — Model drift detected. Immediate retraining recommended."
            reco_css = "badge-red"
        elif warn_count >= 2:
            reco = "⚠️ RETRAIN — Significant drift in multiple features. Schedule retraining."
            reco_css = "badge-yellow"
        else:
            reco = "✅ MONITOR — Model is stable. Continue regular monitoring."
            reco_css = "badge-green"

        st.markdown(
            f'<div style="margin:16px 0;">'
            f'<span class="badge {reco_css}" style="font-size:0.9rem;padding:8px 20px;">{reco}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Summary metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Features Analysed", len(rows))
        mc2.metric("⚠️ Warnings", warn_count)
        mc3.metric("🚨 Critical", critical_count)

        st.markdown("<hr class='styled'>", unsafe_allow_html=True)

        # Drift table with colour-coded HTML
        st.markdown("**Drift Statistics per Feature**")
        st.dataframe(df_drift, use_container_width=True, hide_index=True)

        st.markdown("<hr class='styled'>", unsafe_allow_html=True)

        # ── PSI bar chart ──────────────────────────────────────────────────
        fig_psi = px.bar(
            df_drift.sort_values("PSI", ascending=False),
            x="Feature",
            y="PSI",
            color="PSI",
            color_continuous_scale=["#34d399", "#fbbf24", "#ef4444"],
            range_color=[0, 0.3],
            title="PSI per Feature (threshold: 0.1 / 0.2)",
        )
        fig_psi.add_hline(y=0.1, line_dash="dash", line_color="#fbbf24", annotation_text="Warning")
        fig_psi.add_hline(y=0.2, line_dash="dash", line_color="#ef4444", annotation_text="Critical")
        fig_psi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0f0",
            coloraxis_showscale=False,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_psi, use_container_width=True)

        # ── Distribution comparison for worst feature ──────────────────────
        if rows:
            worst_feat = df_drift.sort_values("PSI", ascending=False).iloc[0]["Feature"]
            st.subheader(f"Distribution Shift: {worst_feat}")
            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Histogram(
                    x=df_ref[worst_feat].dropna(),
                    name="Reference",
                    opacity=0.6,
                    marker_color="#6366f1",
                    histnorm="probability",
                )
            )
            fig_dist.add_trace(
                go.Histogram(
                    x=df_cur[worst_feat].dropna(),
                    name="Current",
                    opacity=0.6,
                    marker_color="#f87171",
                    histnorm="probability",
                )
            )
            fig_dist.update_layout(
                barmode="overlay",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0f0",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 7 — Experiment Logs
# ---------------------------------------------------------------------------


def page_logs() -> None:
    """Render the Experiment Logs page."""
    st.markdown('<div class="section-header">📋 Experiment Logs</div>', unsafe_allow_html=True)

    experiments: list[dict] = api_get("/train/experiments") or []

    if not experiments:
        st.info("No experiments recorded yet.")
        return

    # ── Search and filter ──────────────────────────────────────────────────
    search_col, filter_col = st.columns([3, 1])
    with search_col:
        search_query = st.text_input("🔍 Search experiments", placeholder="model name, dataset, status…")
    with filter_col:
        statuses = sorted({e.get("status", "unknown") for e in experiments})
        status_filter = st.selectbox("Status", ["All"] + statuses)

    # Apply filters
    filtered_exps = experiments
    if search_query:
        q = search_query.lower()
        filtered_exps = [
            e for e in filtered_exps
            if any(
                q in str(v).lower()
                for v in [
                    e.get("experiment_id", ""),
                    e.get("dataset_name", ""),
                    e.get("best_model", ""),
                    e.get("status", ""),
                    e.get("task_type", ""),
                ]
            )
        ]
    if status_filter != "All":
        filtered_exps = [e for e in filtered_exps if e.get("status") == status_filter]

    st.markdown(f"Showing **{len(filtered_exps)}** of **{len(experiments)}** experiments")

    # ── Experiments table ──────────────────────────────────────────────────
    table_rows = []
    for exp in filtered_exps[::-1]:
        table_rows.append(
            {
                "Experiment ID": exp.get("experiment_id", "—"),
                "Dataset": exp.get("dataset_name", "—"),
                "Task": exp.get("task_type", "—"),
                "Best Model": exp.get("best_model", "—"),
                "Score": f"{exp.get('best_score', 0):.4f}" if exp.get("best_score") is not None else "—",
                "Status": exp.get("status", "—"),
                "Created": exp.get("created_at", "—"),
                "Duration (s)": exp.get("duration_seconds", "—"),
            }
        )

    df_logs = pd.DataFrame(table_rows)
    st.dataframe(df_logs, use_container_width=True, hide_index=True)

    # ── Expandable detail view ─────────────────────────────────────────────
    st.markdown("<hr class='styled'>", unsafe_allow_html=True)
    st.subheader("Experiment Detail Viewer")
    exp_ids = [e.get("experiment_id", "unknown") for e in filtered_exps]
    if exp_ids:
        selected_exp_id = st.selectbox("Select Experiment to Inspect", exp_ids)
        exp_detail = next((e for e in filtered_exps if e.get("experiment_id") == selected_exp_id), None)

        if exp_detail:
            with st.expander("📂 Full Experiment Details", expanded=True):
                detail_cols = st.columns(2)

                with detail_cols[0]:
                    st.markdown("**General Info**")
                    general_fields = [
                        "experiment_id", "dataset_name", "task_type",
                        "status", "created_at", "duration_seconds",
                    ]
                    for field in general_fields:
                        val = exp_detail.get(field, "—")
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                            f'border-bottom:1px solid #2d2d4e;">'
                            f'<span style="color:#9ca3af;font-size:0.85rem;">{field.replace("_", " ").title()}</span>'
                            f'<span style="color:#e0e0f0;font-size:0.85rem;font-weight:600;">{val}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                with detail_cols[1]:
                    st.markdown("**Model & Metrics**")
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2d2d4e;">'
                        f'<span style="color:#9ca3af;font-size:0.85rem;">Best Model</span>'
                        f'<span style="color:#a78bfa;font-size:0.85rem;font-weight:700;">{exp_detail.get("best_model", "—")}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2d2d4e;">'
                        f'<span style="color:#9ca3af;font-size:0.85rem;">Best Score</span>'
                        f'<span style="color:#34d399;font-size:0.85rem;font-weight:700;">'
                        f'{exp_detail.get("best_score", "—")}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    metrics = exp_detail.get("metrics", {})
                    for mk, mv in metrics.items():
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2d2d4e;">'
                            f'<span style="color:#9ca3af;font-size:0.85rem;">{mk}</span>'
                            f'<span style="color:#e0e0f0;font-size:0.85rem;">{mv}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # Configuration
                config = exp_detail.get("config", {})
                if config:
                    st.markdown("**Training Configuration**")
                    st.json(config)

                # Leaderboard
                leaderboard = exp_detail.get("leaderboard", [])
                if leaderboard:
                    st.markdown("**Model Leaderboard**")
                    st.dataframe(pd.DataFrame(leaderboard), use_container_width=True, hide_index=True)

                # Logs
                logs = exp_detail.get("logs", [])
                if logs:
                    st.markdown("**Execution Logs**")
                    st.code("\n".join(logs), language="text")

    # ── Export ─────────────────────────────────────────────────────────────
    st.markdown("<hr class='styled'>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Export All Logs (CSV)",
        data=df_logs.to_csv(index=False),
        file_name=f"experiment_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — render sidebar and route to the selected page."""
    page = render_sidebar()

    if page == "🏠 Home Dashboard":
        page_home()
    elif page == "🚀 Train Model":
        page_train()
    elif page == "📊 Model Comparison":
        page_comparison()
    elif page == "🔮 Make Predictions":
        page_predict()
    elif page == "🔍 Data Profiler":
        page_profiler()
    elif page == "📈 Model Monitoring":
        page_monitoring()
    elif page == "📋 Experiment Logs":
        page_logs()


if __name__ == "__main__":
    main()
