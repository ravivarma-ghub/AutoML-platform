# =============================================================================
# Intelligent AutoML Platform — Makefile
#
# Usage:  make <target>
#         make help          → list all targets with descriptions
# =============================================================================

# ---------------------------------------------------------------------------
# Configurable variables (override on CLI: make run-api PORT=9000)
# ---------------------------------------------------------------------------
PYTHON        ?= python
PIP           ?= pip
UVICORN       ?= uvicorn
STREAMLIT     ?= streamlit
MLFLOW        ?= mlflow
DOCKER        ?= docker
COMPOSE       ?= docker compose
ALEMBIC       ?= alembic
RUFF          ?= ruff
PYTEST        ?= pytest

APP_MODULE    ?= app.main:app
API_HOST      ?= 0.0.0.0
API_PORT      ?= 8000
API_WORKERS   ?= 1

STREAMLIT_APP ?= app/ui/streamlit_app.py
STREAMLIT_PORT?= 8501

MLFLOW_HOST   ?= 0.0.0.0
MLFLOW_PORT   ?= 5000
MLFLOW_BACKEND?= sqlite:///mlruns/mlflow.db
MLFLOW_ARTIFACTS ?= ./mlruns

IMAGE_NAME    ?= automl-platform
IMAGE_TAG     ?= latest

COVERAGE_MIN  ?= 70

# Colour helpers
BOLD   := $(shell tput bold   2>/dev/null || echo "")
RESET  := $(shell tput sgr0   2>/dev/null || echo "")
GREEN  := $(shell tput setaf 2 2>/dev/null || echo "")
YELLOW := $(shell tput setaf 3 2>/dev/null || echo "")
CYAN   := $(shell tput setaf 6 2>/dev/null || echo "")

# Default target
.DEFAULT_GOAL := help

# All phony targets (no corresponding files)
.PHONY: help install install-dev run-api run-ui run-mlflow run-all \
        test lint format build docker-up docker-down docker-logs \
        clean migrate migrate-new migrate-history migrate-downgrade \
        check-env shell

# ---------------------------------------------------------------------------
# help — auto-generated from ## comments
# ---------------------------------------------------------------------------
help: ## Show this help message
	@echo ""
	@echo "$(BOLD)$(CYAN)Intelligent AutoML Platform$(RESET)"
	@echo "$(CYAN)================================$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Install production + development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt 2>/dev/null || \
	    $(PIP) install pytest pytest-cov pytest-asyncio pytest-mock httpx ruff mypy pre-commit
	@echo "$(GREEN)✔  Dev environment ready$(RESET)"

# ---------------------------------------------------------------------------
# Running services locally (outside Docker)
# ---------------------------------------------------------------------------
run-api: check-env ## Run FastAPI server (uvicorn, hot-reload)
	$(UVICORN) $(APP_MODULE) \
	    --host $(API_HOST) \
	    --port $(API_PORT) \
	    --workers $(API_WORKERS) \
	    --reload \
	    --log-level info

run-ui: check-env ## Run Streamlit dashboard
	$(STREAMLIT) run $(STREAMLIT_APP) \
	    --server.port $(STREAMLIT_PORT) \
	    --server.address 0.0.0.0 \
	    --server.headless true

run-mlflow: ## Run local MLflow tracking server
	@mkdir -p mlruns
	$(MLFLOW) server \
	    --backend-store-uri $(MLFLOW_BACKEND) \
	    --default-artifact-root $(MLFLOW_ARTIFACTS) \
	    --host $(MLFLOW_HOST) \
	    --port $(MLFLOW_PORT)

run-all: ## Run all services via docker-compose (alias for docker-up)
	$(MAKE) docker-up

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test: ## Run tests with coverage report
	$(PYTEST) tests/ \
	    --cov=app \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    --cov-fail-under=$(COVERAGE_MIN) \
	    -v \
	    --tb=short
	@echo "$(GREEN)✔  Coverage report: htmlcov/index.html$(RESET)"

test-fast: ## Run tests without coverage (faster iteration)
	$(PYTEST) tests/ -v --tb=short -x

test-ci: ## Run tests suitable for CI (xml coverage, no color)
	$(PYTEST) tests/ \
	    --cov=app \
	    --cov-report=xml \
	    --cov-fail-under=$(COVERAGE_MIN) \
	    -v \
	    --tb=short \
	    -p no:warnings

# ---------------------------------------------------------------------------
# Linting & Formatting
# ---------------------------------------------------------------------------
lint: ## Run ruff linter (check only, no fixes)
	$(RUFF) check . --output-format=concise
	@echo "$(GREEN)✔  Lint passed$(RESET)"

lint-fix: ## Run ruff linter and apply auto-fixes
	$(RUFF) check . --fix

format: ## Run ruff formatter (check + apply)
	$(RUFF) format .
	@echo "$(GREEN)✔  Formatting complete$(RESET)"

format-check: ## Check formatting without modifying files
	$(RUFF) format --check .

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
build: ## Build Docker image (runtime stage)
	$(DOCKER) build \
	    --target runtime \
	    -t $(IMAGE_NAME):$(IMAGE_TAG) \
	    -t $(IMAGE_NAME):latest \
	    -f Dockerfile \
	    .
	@echo "$(GREEN)✔  Image built: $(IMAGE_NAME):$(IMAGE_TAG)$(RESET)"

build-no-cache: ## Build Docker image without layer cache
	$(DOCKER) build \
	    --no-cache \
	    --target runtime \
	    -t $(IMAGE_NAME):$(IMAGE_TAG) \
	    -f Dockerfile \
	    .

docker-up: ## Start all Docker Compose services in detached mode
	@cp -n .env.example .env 2>/dev/null || true
	$(COMPOSE) up -d --build --remove-orphans
	@echo ""
	@echo "$(GREEN)✔  Services running:$(RESET)"
	@echo "   API      → http://localhost:$(API_PORT)"
	@echo "   Streamlit→ http://localhost:$(STREAMLIT_PORT)"
	@echo "   MLflow   → http://localhost:$(MLFLOW_PORT)"

docker-down: ## Stop and remove Docker Compose containers
	$(COMPOSE) down
	@echo "$(YELLOW)▸  Containers stopped$(RESET)"

docker-down-v: ## Stop containers AND remove volumes (destructive!)
	$(COMPOSE) down -v
	@echo "$(YELLOW)▸  Containers and volumes removed$(RESET)"

docker-restart: ## Restart all Docker Compose services
	$(COMPOSE) restart

docker-logs: ## Follow logs from all Docker Compose services
	$(COMPOSE) logs -f

docker-logs-api: ## Follow logs from the API service only
	$(COMPOSE) logs -f api

docker-ps: ## List running Docker Compose services
	$(COMPOSE) ps

# ---------------------------------------------------------------------------
# Database Migrations (Alembic)
# ---------------------------------------------------------------------------
migrate: ## Apply all pending Alembic migrations (upgrade head)
	$(ALEMBIC) upgrade head
	@echo "$(GREEN)✔  Migrations applied$(RESET)"

migrate-new: ## Create a new migration (usage: make migrate-new MSG="add users")
	@if [ -z "$(MSG)" ]; then \
	    echo "$(YELLOW)Usage: make migrate-new MSG=\"your migration message\"$(RESET)"; \
	    exit 1; \
	fi
	$(ALEMBIC) revision --autogenerate -m "$(MSG)"
	@echo "$(GREEN)✔  Migration created$(RESET)"

migrate-history: ## Show migration history
	$(ALEMBIC) history --verbose

migrate-current: ## Show current migration revision
	$(ALEMBIC) current

migrate-downgrade: ## Revert the last migration (usage: make migrate-downgrade N=1)
	$(ALEMBIC) downgrade -$(or $(N),1)
	@echo "$(YELLOW)▸  Reverted $(or $(N),1) migration(s)$(RESET)"

migrate-sql: ## Generate SQL for offline migration (stdout)
	$(ALEMBIC) upgrade head --sql

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
check-env: ## Check that .env exists (copy from .env.example if not)
	@if [ ! -f .env ]; then \
	    cp .env.example .env; \
	    echo "$(YELLOW)▸  .env created from .env.example — please review it$(RESET)"; \
	fi

shell: ## Open a Python REPL with the app context
	$(PYTHON) -c "import app; import IPython; IPython.embed()" 2>/dev/null || \
	$(PYTHON) -i -c "import app"

clean: ## Remove build artifacts, caches, and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"       -delete               2>/dev/null || true
	find . -type f -name "*.pyo"       -delete               2>/dev/null || true
	find . -type f -name "*.pyd"       -delete               2>/dev/null || true
	find . -type d -name "*.egg-info"  -exec rm -rf {} +     2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} +   2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} +     2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} +     2>/dev/null || true
	find . -type d -name "htmlcov"     -exec rm -rf {} +     2>/dev/null || true
	find . -type f -name "coverage.xml" -delete              2>/dev/null || true
	find . -type f -name ".coverage"   -delete               2>/dev/null || true
	find . -type d -name "dist"        -exec rm -rf {} +     2>/dev/null || true
	find . -type d -name "build"       -exec rm -rf {} +     2>/dev/null || true
	@echo "$(GREEN)✔  Clean complete$(RESET)"
