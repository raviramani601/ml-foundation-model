import streamlit as st
import base64

# ---- Helper Function ----
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---- Set background before any UI rendering ----
set_background("997526.jpg")  # your image file

# ---- Streamlit UI ----

# Example Input
sales_value = st.number_input("Enter Sales Value ($)", min_value=0)

# Predict button
if st.button("Predict Sales Class"):
    # Dummy logic for demonstration
    if sales_value > 50000:
        st.success("High-end BMW buyer")
    else:
        st.warning("Mid-range BMW buyer")
