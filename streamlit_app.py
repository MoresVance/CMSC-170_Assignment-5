import streamlit as st
from PIL import Image
import pandas as pd
import io

from sympy import false


# Set wide layout and page configuration
st.set_page_config(
    page_title="AI Classification App", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

## Sidebar Panel
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("AI Image Classifier")
    st.markdown("""
        **A simple application to classify uploaded images using a machine learning model.**
        
        ---
        
        ### **Instructions**
        1. **Upload** an image file below (.jpg, .jpeg, or .png).
        2. Click the **'Predict'** button.
        3. View the top prediction and confidence scores in the main panel.
    """)

## Main Panel
# ----------------------------------------------------------------------
st.header("Upload Image for Classification")

uploaded_file = st.file_uploader(
    "Choose an image file...",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=False,
    help="Drag and drop or browse to upload an image (.jpg, .jpeg, or .png)"
)


# Layout columns for image display and button
col1, col2 = st.columns([1, 2])

with col1:
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Predict Button
        if st.button("🚀 **Predict**", key="predict_button", use_container_width=True):
            with st.spinner('Processing image and running model...'):
                # Convert uploaded file to bytes for the model function
                image_bytes = io.BytesIO(uploaded_file.getvalue())
                
                
    else:
        st.markdown(
            """
            <div style="
                border: 2px dashed #ccc; 
                padding: 20px; 
                text-align: center; 
                border-radius: 10px;
                height: 300px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            ">
                <p style="font-size: 1.1em; color: #888;">No Image Uploaded</p>
                <p style="font-size: 0.9em; color: #aaa;">Upload an image on the left to begin.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Results Display Area
with col2:
    if false:
        st.success("✅ **Prediction Complete!**")
        
        st.subheader("🥇 Top Prediction")
        # Custom HTML to display the top prediction highlighted in green
        top_class_html = f"""
            <div style="
                background-color: #e6ffe6; 
                color: #008000; 
                padding: 15px; 
                border-radius: 8px; 
                font-size: 1.5em; 
                font-weight: bold;
                text-align: center;
                border: 1px solid #008000;
            ">
                {st.session_state.top_1_class}
            </div>
        """
        st.markdown(top_class_html, unsafe_allow_html=True)
        
        st.markdown("---")

        ### Bar Graph Showing Top-5 Predicted Classes
        st.subheader("📊 Top 5 Predictions and Confidence")
        
        
        st.caption("Confidence score represents the model's certainty (0.0 to 1.0).")
        
    else:
        st.subheader("Results Display Area")
        st.info("Upload an image and click 'Predict' to see the classification results here.")

