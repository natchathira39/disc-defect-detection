import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Page configuration
st.set_page_config(page_title="Disc Defect Detection", page_icon="🔧", layout="wide")

# Configuration
IMG_SIZE = (224, 224)  # Your model was trained with 224x224
CLASS_NAMES = ['good', 'patches', 'rolled_pits', 'scratches', 'waist_folding']

# *** REPLACE WITH YOUR GOOGLE DRIVE FILE ID ***
GOOGLE_DRIVE_FILE_ID = "1soeqMV4kO9EZg6JMny-c5CYEhf5yIB01"
MODEL_PATH = "disc_defect_model.h5"

@st.cache_resource
def download_model():
    """Download model from Google Drive"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_file = download_model()
    model = tf.keras.models.load_model(model_file)
    return model

def preprocess_image(image):
    """Preprocess image for model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_defect(image, model):
    """Make prediction"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    return predicted_class, confidence, predictions[0] * 100

# Main App
st.title("🔧 Disc Defect Detection")
st.write("Upload a disc image to detect defects")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("**Defect Types:**")
    st.write("✅ Good")
    st.write("⚠️ Patches")
    st.write("❌ Rolled Pits")
    st.write("⚠️ Scratches")
    st.write("❌ Waist Folding")

# Upload
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Prediction")
        model = load_model()
        
        with st.spinner("Analyzing..."):
            predicted_class, confidence, all_probs = predict_defect(image, model)
        
        # Display result
        if predicted_class == 'good':
            st.success(f"✅ **{predicted_class.upper()}**")
        else:
            st.error(f"❌ **DEFECT: {predicted_class.upper().replace('_', ' ')}**")
        
        # Confidence
        st.progress(confidence / 100)
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        # All probabilities
        st.write("**All Probabilities:**")
        for name, prob in zip(CLASS_NAMES, all_probs):
            st.write(f"{name.replace('_', ' ').title()}: {prob:.1f}%")
