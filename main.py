import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("BMW_Car_Sales_Classification.csv")


df = load_data()


# Encode and scale
def preprocess(df):
    df_processed = df.copy()
    label_encoders = {}

    for col in df_processed.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    X = df_processed.drop("Sales_Classification", axis=1)
    y = df_processed["Sales_Classification"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders


X_scaled, y, scaler, label_encoders = preprocess(df)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Streamlit UI
st.title("BMW Sales Classification Predictor 🚗📊")

with st.form("prediction_form"):
    st.subheader("Enter Car Details:")

    model_input = st.selectbox("Model", df["Model"].unique())
    year_input = st.number_input("Year", min_value=2000, max_value=2025, value=2020)
    region_input = st.selectbox("Region", df["Region"].unique())
    color_input = st.selectbox("Color", df["Color"].unique())
    fuel_input = st.selectbox("Fuel Type", df["Fuel_Type"].unique())
    transmission_input = st.selectbox("Transmission", df["Transmission"].unique())
    engine_input = st.number_input("Engine Size (L)", min_value=1.0, max_value=6.0, step=0.1, value=2.0)
    mileage_input = st.number_input("Mileage (KM)", min_value=0, value=50000)
    price_input = st.number_input("Price (USD)", min_value=1000, value=30000)
    sales_volume_input = st.number_input("Sales Volume", min_value=0, value=100)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert input to DataFrame
    input_data = pd.DataFrame([{
        "Model": model_input,
        "Year": year_input,
        "Region": region_input,
        "Color": color_input,
        "Fuel_Type": fuel_input,
        "Transmission": transmission_input,
        "Engine_Size_L": engine_input,
        "Mileage_KM": mileage_input,
        "Price_USD": price_input,
        "Sales_Volume": sales_volume_input
    }])

    # Encode input
    for col in input_data.select_dtypes(include='object').columns:
        le = label_encoders[col]
        input_data[col] = le.transform(input_data[col])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_label = label_encoders['Sales_Classification'].inverse_transform(prediction)

    st.success(f"Predicted Sales Classification: **{prediction_label[0]}**")

