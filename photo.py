from pathlib import Path
import streamlit as st
import base64

def set_background(image_file):
    # Build path relative to project root
    img_path = Path(__file__).parent / image_file
    with open(img_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
