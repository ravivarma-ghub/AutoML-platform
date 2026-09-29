# =============================================================================
# Intelligent AutoML Platform — Multi-Stage Dockerfile
# Stage 1: builder  →  Stage 2: runtime (non-root, minimal image)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder — compile / install all Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL maintainer="automl-platform"
LABEL description="Intelligent AutoML Platform — builder stage"

# Build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libpq-dev \
        libffi-dev \
        libssl-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip / setuptools / wheel in the builder
RUN pip install --upgrade pip setuptools wheel

WORKDIR /build

# Copy dependency manifests first (layer-cache friendly)
COPY requirements.txt ./

# Install all packages into a user-local prefix so we can copy them cleanly
RUN pip install --user --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime — lean final image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="automl-platform"
LABEL description="Intelligent AutoML Platform — runtime stage"
LABEL org.opencontainers.image.title="automl-platform"
LABEL org.opencontainers.image.version="1.0.0"

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user & group for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Ensure local bin is on PATH
ENV PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    # Sensible defaults — override via docker-compose / .env
    APP_NAME="Intelligent AutoML Platform" \
    API_HOST="0.0.0.0" \
    API_PORT="8000" \
    STREAMLIT_SERVER_PORT="8501" \
    STREAMLIT_SERVER_ADDRESS="0.0.0.0" \
    MODEL_REGISTRY_PATH="/app/models" \
    REPORTS_PATH="/app/reports" \
    LOGS_PATH="/app/logs"

WORKDIR /app

# Copy application source code
COPY --chown=appuser:appgroup . .

# Create runtime directories and fix ownership
RUN mkdir -p /app/models /app/reports /app/logs /app/mlruns \
    && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default: run the FastAPI server via uvicorn
# Override in docker-compose for the Streamlit service
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
