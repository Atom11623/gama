# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model, scaler, and feature list
model = joblib.load("rf_model_compressed.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")  # List of columns after get_dummies

# Streamlit app UI
st.set_page_config(page_title="California House Price Predictor", layout="centered")
st.title("🏠 California House Price Predictor")
st.markdown("Enter the housing details below to predict the median house value:")

# User inputs
longitude = st.number_input("Longitude", -125.0, -113.0, value=-120.0)
latitude = st.number_input("Latitude", 32.0, 43.0, value=36.0)
housing_median_age = st.slider("Housing Median Age", 1, 52, 20)
total_rooms = st.number_input("Total Rooms", 2, 50000, value=2000)
total_bedrooms = st.number_input("Total Bedrooms", 1, 10000, value=400)
population = st.number_input("Population", 1, 50000, value=1000)
households = st.number_input("Households", 1, 10000, value=400)
median_income = st.number_input("Median Income (×$10,000)", 0.0, 20.0, value=3.0)
ocean_proximity = st.selectbox("Ocean Proximity", [
    '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
])

# Create DataFrame from input
input_df = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# Encode the categorical variable using get_dummies
input_encoded = pd.get_dummies(input_df, columns=["ocean_proximity"], drop_first=True)

# Reindex columns to match training features
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_encoded)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.subheader("💵 Predicted Median House Value:")
    st.success(f"${prediction:,.2f}")
