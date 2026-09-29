"""
Utility script to generate sample datasets for testing the AutoML platform:
1. customer_churn.csv - Tabular classification dataset with mixed types, dates, missing values, outliers
2. house_prices.csv - Tabular regression dataset with continuous target
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_datasets(output_dir: str = "./data"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    n_samples = 2500

    print("Generating customer_churn.csv (Classification)...")
    # Dates
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=int(x)) for x in np.random.uniform(0, 1000, n_samples)]
    
    # Numerical features
    age = np.random.normal(40, 14, n_samples).clip(18, 90)
    tenure_months = np.random.exponential(18, n_samples).clip(1, 72)
    monthly_charges = np.random.normal(70, 30, n_samples).clip(18, 150)
    total_charges = tenure_months * monthly_charges * np.random.uniform(0.9, 1.1, n_samples)
    support_tickets = np.random.poisson(2, n_samples)

    # Categorical features
    contract_types = np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.20], size=n_samples)
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], p=[0.4, 0.45, 0.15], size=n_samples)
    payment_method = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ], p=[0.35, 0.2, 0.25, 0.2], size=n_samples)
    city = np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"], size=n_samples)

    # Churn probability logic
    logit = (
        - 1.5
        + 0.03 * (monthly_charges - 70)
        - 0.05 * tenure_months
        + 0.4 * (contract_types == "Month-to-month")
        + 0.5 * (internet_service == "Fiber optic")
        + 0.25 * support_tickets
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob_churn).astype(int)

    churn_df = pd.DataFrame({
        "signup_date": dates,
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "contract": contract_types,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "city": city,
        "churn": churn
    })

    # Add realistic noise: 2% missing values
    churn_df.loc[np.random.choice(n_samples, int(0.02 * n_samples)), "monthly_charges"] = np.nan
    churn_df.loc[np.random.choice(n_samples, int(0.02 * n_samples)), "internet_service"] = np.nan

    churn_path = os.path.join(output_dir, "customer_churn.csv")
    churn_df.to_csv(churn_path, index=False)
    print(f" Saved: {churn_path} ({len(churn_df)} rows, {churn_df.shape[1]} columns)")

    print("\nGenerating house_prices.csv (Regression)...")
    sqft = np.random.normal(2100, 650, n_samples).clip(600, 6000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.25, 0.45, 0.2, 0.05], size=n_samples)
    bathrooms = (bedrooms * 0.75 + np.random.choice([0, 0.5, 1], size=n_samples)).clip(1, 5)
    year_built = np.random.randint(1950, 2024, size=n_samples)
    lot_size = np.random.exponential(8000, n_samples).clip(1500, 45000)
    neighborhood = np.random.choice(["Downtown", "Suburbs", "Northside", "WestEnd", "Lakeside", "Hillcrest"], size=n_samples)
    has_pool = np.random.choice([0, 1], p=[0.75, 0.25], size=n_samples)
    sale_date = [start_date + timedelta(days=int(x)) for x in np.random.uniform(0, 1000, n_samples)]

    price = (
        120000
        + 180 * sqft
        + 15000 * bedrooms
        + 25000 * bathrooms
        + 1200 * (year_built - 1950)
        + 45000 * has_pool
        + np.random.normal(0, 35000, n_samples)
    ).clip(80000, 1500000)

    house_df = pd.DataFrame({
        "sale_date": sale_date,
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "lot_size": lot_size,
        "neighborhood": neighborhood,
        "has_pool": has_pool,
        "price": price
    })

    # Missing values
    house_df.loc[np.random.choice(n_samples, int(0.015 * n_samples)), "lot_size"] = np.nan

    house_path = os.path.join(output_dir, "house_prices.csv")
    house_df.to_csv(house_path, index=False)
    print(f" Saved: {house_path} ({len(house_df)} rows, {house_df.shape[1]} columns)")
    print("\nDataset generation complete!")

if __name__ == "__main__":
    generate_datasets("./data")
