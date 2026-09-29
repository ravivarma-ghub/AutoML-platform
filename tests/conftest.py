"""
Pytest configuration and shared fixtures for the AutoML Platform test suite.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.database import Base, get_db
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classification_df(seed: int = 42) -> pd.DataFrame:
    """Return a 1000-row synthetic classification dataset."""
    rng = np.random.default_rng(seed)

    n = 1000

    age = rng.integers(18, 70, size=n).astype(float)
    income = rng.uniform(20_000, 150_000, size=n)
    education = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"], size=n, p=[0.3, 0.4, 0.2, 0.1]
    )
    occupation = rng.choice(
        ["Engineer", "Manager", "Sales", "Support", "Other"], size=n
    )

    base_date = datetime(2022, 1, 1)
    signup_date = [
        (base_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in rng.integers(0, 730, size=n)
    ]
    signup_date = pd.to_datetime(signup_date)

    score = rng.uniform(0, 100, size=n)

    # Binary target correlated with features
    logit = (
        -3
        + 0.02 * (age - 40)
        + 0.00001 * income
        + score * 0.04
        + rng.normal(0, 1, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "education": education,
            "occupation": occupation,
            "signup_date": signup_date,
            "score": score,
            "churn": churn,
        }
    )

    # Introduce ~2% missing values in age and income
    missing_idx_age = rng.choice(n, size=int(n * 0.02), replace=False)
    missing_idx_income = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[missing_idx_age, "age"] = np.nan
    df.loc[missing_idx_income, "income"] = np.nan

    # Append 5 duplicate rows
    duplicates = df.iloc[:5].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def _make_regression_df(seed: int = 42) -> pd.DataFrame:
    """Return a 1000-row synthetic regression dataset."""
    rng = np.random.default_rng(seed)
    n = 1000

    area = rng.uniform(500, 5000, size=n)
    bedrooms = rng.integers(1, 6, size=n).astype(float)
    bathrooms = rng.uniform(1, 4, size=n)
    age_years = rng.integers(0, 50, size=n).astype(float)
    location = rng.choice(["Urban", "Suburban", "Rural"], size=n, p=[0.5, 0.3, 0.2])
    garage = rng.choice([0, 1], size=n, p=[0.3, 0.7])

    price = (
        50_000
        + area * 120
        + bedrooms * 10_000
        + bathrooms * 8_000
        - age_years * 500
        + rng.normal(0, 15_000, size=n)
    )
    price = np.clip(price, 50_000, 2_000_000)

    df = pd.DataFrame(
        {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "location": location,
            "garage": garage,
            "price": price,
        }
    )

    # 2% missing in bedrooms
    missing_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[missing_idx, "bedrooms"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_classification_df() -> pd.DataFrame:
    """1005-row synthetic classification dataset (1000 + 5 duplicates)."""
    return _make_classification_df()


@pytest.fixture(scope="session")
def sample_regression_df() -> pd.DataFrame:
    """1000-row synthetic regression dataset."""
    return _make_regression_df()


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Default application settings for testing."""
    return Settings()


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """
    Provide an in-memory SQLite session for each test function.

    Tables are created fresh for every test and torn down afterwards so tests
    remain fully isolated.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient with the database dependency overridden to use the
    in-memory SQLite session created by :func:`db_session`.
    """

    def _override_get_db() -> Generator[Session, None, None]:
        try:
            yield db_session
        finally:
            pass  # session lifecycle managed by db_session fixture

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
