import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Predictor")
st.markdown("Provide property details below to estimate the house price.")

# Load model if available
model_path = "house_price_predictor.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

# Sidebar: global input
st.sidebar.header("Global Settings")
recent_trend = st.sidebar.slider("Recent Price Trend (%)", -5.0, 5.0, 0.0, step=0.1)

# Input form
with st.form("input_form"):
    st.subheader("Property Details")

    col1, col2 = st.columns(2)
    with col1:
        gr_liv_area = st.number_input("Living Area (sq. ft)", value=1500)
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2005)
        property_tax = st.number_input("Annual Property Tax (USD)", value=2500)
        crime_rate = st.slider("Crime Rate (1=Low, 10=High)", 1.0, 10.0, 5.0)

    with col2:
        overall_qual = st.selectbox("Overall Quality (1-10)", list(range(1, 11)), index=7)
        school_rating = st.slider("School Rating (1=Low, 10=High)", 1, 10, 7)
        distance_to_city = st.slider("Distance to City Center (km)", 0.5, 30.0, 5.0, step=0.5)
        neighborhood = st.selectbox("Neighborhood", [
            "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst",
            "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes"
        ])

    submitted = st.form_submit_button("Predict House Price")

# Function to make input dataframe
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
        "Price_per_sqft": [1 if gr_liv_area else 0],
        "Distance_to_city": [distance_to_city]
    })

# Show prediction
if submitted:
    input_df = make_input_df(recent_trend)
    st.markdown("## Prediction Summary")
    st.dataframe(input_df)

    if model:
        prediction = model.predict(input_df)
        st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
    else:
        st.warning("Model file not found. Please upload `house_price_predictor.pkl` to enable predictions.")
