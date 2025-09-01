# photo.py
import streamlit as st
from pathlib import Path
import base64
import mimetypes

def set_background(image_file: str):
    """
    image_file can be a relative path like "assets/997526.jpg" or just "997526.jpg".
    This function resolves it relative to this file's directory and sets CSS background.
    """
    base_dir = Path(__file__).parent  # directory where photo.py lives
    img_path = (base_dir / image_file).resolve()

    if not img_path.exists():
        st.error(f"Background image not found: {img_path}")
        # Optionally log for debugging in server logs:
        st.write("Make sure the file is in the repository and pushed to GitHub.")
        return

    # Read file and embed as base64 to avoid external hosting
    with open(img_path, "rb") as f:
        data = f.read()

    mime_type, _ = mimetypes.guess_type(str(img_path))
    if not mime_type:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:{mime_type};base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
