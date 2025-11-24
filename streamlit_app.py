import streamlit as st
from PIL import Image
import pandas as pd
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image


# Set wide layout and page configuration
st.set_page_config(
    page_title="AI Image Classifier", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    model = EfficientNetB0(weights='imagenet')
    return model

model = load_model()

# Prediction function
def predict(image_bytes, model):
    img = Image.open(image_bytes).convert('RGB')
    img = img.resize((224, 224))
    
    # Preprocess the image
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Make prediction
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=5)[0]
    
    return decoded_preds

def clear_predictions():
    if 'predictions' in st.session_state:
        del st.session_state.predictions
    if 'top_1_class' in st.session_state:
        del st.session_state.top_1_class

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
    help="Drag and drop or browse to upload an image (.jpg, .jpeg, or .png)",
    on_change=clear_predictions,
)


# Layout columns for image display and button
col1, col2 = st.columns([1, 2])

with col1:
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Predict Button
        if st.button("**Predict**", key="predict_button", use_container_width=True):
            with st.spinner('Processing image and running model...'):
                # Convert uploaded file to bytes for the model function
                image_bytes = io.BytesIO(uploaded_file.getvalue())
                
                # Get predictions
                predictions = predict(image_bytes, model)
                st.session_state.predictions = predictions
                st.session_state.top_1_class = predictions[0][1].replace('_', ' ').title()
                
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
    if 'predictions' in st.session_state:
        st.success("**Prediction Complete!**")
        
        st.subheader("Top Prediction")
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
        st.subheader("Top 5 Predictions and Confidence")
        
        # Create a DataFrame for the bar chart
        df_preds = pd.DataFrame(
            [(class_name.replace('_', ' ').title(), prob) for _, class_name, prob in st.session_state.predictions],
            columns=['Class', 'Confidence']
        )
        
        st.bar_chart(df_preds.set_index('Class'))
        
        st.caption("Confidence score represents the model's certainty (0.0 to 1.0).")
        
    else:
        st.subheader("Results Will Appear Here:")
        st.info("Tip: Upload an image and click 'Predict' to see the classification results here.")

