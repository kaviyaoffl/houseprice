import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

st.set_page_config(page_title="Train House Price Model", layout="centered")
st.title("🏗️ Train & Save House Price Prediction Model")

st.markdown("Click the button below to train a sample model and generate `house_price_predictor.pkl` for use in your main app.")

if st.button("Train Model & Save File"):
    # Simulated training data
    np.random.seed(42)
    n_samples = 500
    df = pd.DataFrame({
        "GrLivArea": np.random.randint(800, 3000, n_samples),
        "OverallQual": np.random.randint(1, 11, n_samples),
        "YearBuilt": np.random.randint(1950, 2025, n_samples),
        "Neighborhood": np.random.choice([
            "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst",
            "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes"
        ], n_samples),
        "RecentPriceTrend": np.random.uniform(-5, 5, n_samples),
        "PropertyTax": np.random.randint(1000, 5000, n_samples),
        "CrimeRate": np.random.uniform(1, 10, n_samples),
        "SchoolRating": np.random.randint(1, 11, n_samples),
        "Price_per_sqft": np.random.randint(0, 2, n_samples),
        "Distance_to_city": np.random.uniform(0.5, 30, n_samples),
    })

    # Create target variable
    df["SalePrice"] = (
        df["GrLivArea"] * 120 +
        df["OverallQual"] * 5000 +
        (2025 - df["YearBuilt"]) * -150 +
        df["RecentPriceTrend"] * 1000 +
        np.random.normal(0, 10000, n_samples)
    )

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    categorical = ["Neighborhood"]
    numeric = [col for col in X.columns if col not in categorical]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)

    joblib.dump(pipeline, "house_price_predictor.pkl")

    st.success("✅ Model trained and saved as `house_price_predictor.pkl`!")
    st.info("You can now use this file in your House Price Prediction App.")
