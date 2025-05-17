import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained model
model_path = "house_price_predictor.pkl"
model = None

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file 'house_price_predictor.pkl' not found. Please upload the file to continue.")

st.title("🏡 Real-Time House Price Predictor")
st.write("Fill in the property details below to get an estimated house price.")

# Input fields
gr_liv_area = st.number_input("Living Area (in sq. ft)", value=1500)
overall_qual = st.selectbox("Overall Quality (1-10)", list(range(1, 11)), index=7)
year_built = st.number_input("Year Built", value=2000, min_value=1900, max_value=2025)
neighborhood = st.selectbox("Neighborhood", [
    "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst",
    "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes"
])
recent_trend = st.slider("Recent Price Trend (%)", -5.0, 5.0, 0.0, step=0.1)
property_tax = st.number_input("Annual Property Tax (USD)", value=2500)
crime_rate = st.slider("Crime Rate (1=Low, 10=High)", 1.0, 10.0, 5.0)
school_rating = st.slider("School Rating (1=Low, 10=High)", 1, 10, 7)
distance_to_city = st.slider("Distance to City Center (km)", 0.5, 30.0, 5.0, step=0.5)

# Feature engineering
price_per_sqft = gr_liv_area and 1 or 0

# Function to create input dataframe
def make_input_df(trend):
    return pd.DataFrame({
        "GrLivArea": [gr_liv_area],
        "OverallQual": [overall_qual],
        "YearBuilt": [year_built],
        "Neighborhood": [neighborhood],
        "RecentPriceTrend": [trend],
        "PropertyTax": [property_tax],
        "CrimeRate": [crime_rate],
        "SchoolRating": [school_rating],
        "Price_per_sqft": [price_per_sqft],
        "Distance_to_city": [distance_to_city]
    })

# Main prediction logic
def predict_house_price(input_df):
    if model:
        prediction = model.predict(input_df)
        st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
        st.markdown("---")
        st.write("### Model Input Summary")
        st.dataframe(input_df)
    else:
        st.warning("Model is not loaded. Please upload the model file.")

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Predict House Price"):
        input_df = make_input_df(recent_trend)
        predict_house_price(input_df)

with col2:
    if st.button("Predict with Optimistic Market Trend"):
        input_df = make_input_df(min(recent_trend + 3, 5.0))
        predict_house_price(input_df)

with col3:
    if st.button("Predict with Pessimistic Market Trend"):
        input_df = make_input_df(max(recent_trend - 3, -5.0))
        predict_house_price(input_df)
