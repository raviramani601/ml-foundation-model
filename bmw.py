import streamlit as st
import pandas as pd
from photo import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("BMW_Car_Sales_Classification.csv")
    label_encoders = {}
    df_encoded = df.copy()
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    scaler = StandardScaler()
    X = df_encoded.drop("Sales_Classification", axis=1)
    y = df_encoded["Sales_Classification"]
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler, label_encoders, df

model, scaler, label_encoders, df = load_data()

# Optional: Custom HTML header
st.markdown("""
    <div style="background-color:#2c3e50;padding:20px;border-radius:10px;">
        <h2 style="color:white;text-align:center;">🚗 BMW Sales Classification Predictor</h2>
    </div>
    <br>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Input Car Details")
form_data = {
    "Model": st.sidebar.selectbox("Model", df["Model"].unique()),
    "Year": st.sidebar.number_input("Year", min_value=2000, max_value=2025, value=2020),
    "Region": st.sidebar.selectbox("Region", df["Region"].unique()),
    "Color": st.sidebar.selectbox("Color", df["Color"].unique()),
    "Fuel_Type": st.sidebar.selectbox("Fuel Type", df["Fuel_Type"].unique()),
    "Transmission": st.sidebar.selectbox("Transmission", df["Transmission"].unique()),
    "Engine_Size_L": st.sidebar.number_input("Engine Size (L)", min_value=1.0, max_value=6.0, step=0.1, value=2.0),
    "Mileage_KM": st.sidebar.number_input("Mileage (KM)", value=30000),
    "Price_USD": st.sidebar.number_input("Price (USD)", value=40000),
    "Sales_Volume": st.sidebar.number_input("Sales Volume", value=100)
}

if st.sidebar.button("Predict Classification"):
    input_df = pd.DataFrame([form_data])
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = label_encoders[col].transform(input_df[col])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    predicted_label = label_encoders['Sales_Classification'].inverse_transform(prediction)

    st.success(f"🧠 Predicted Sales Classification: **{predicted_label[0]}**")
