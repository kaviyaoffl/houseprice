import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Simulate some data
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

# Fake target
df["SalePrice"] = (
    df["GrLivArea"] * 120 +
    df["OverallQual"] * 5000 +
    (2025 - df["YearBuilt"]) * -150 +
    df["RecentPriceTrend"] * 1000 +
    np.random.normal(0, 10000, n_samples)
)

# Features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Categorical encoding
categorical = ["Neighborhood"]
numeric = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
model.fit(X, y)

# Save the model
joblib.dump(model, "house_price_predictor.pkl")
print("Model saved as house_price_predictor.pkl")
