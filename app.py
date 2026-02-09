import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import zipfile

st.set_page_config(page_title="Disc Defect Detection", page_icon="🔧", layout="wide")

IMG_SIZE = (224, 224)
CLASS_NAMES = ['good', 'patches', 'rolled_pits', 'scratches', 'waist_folding']

GOOGLE_DRIVE_FILE_ID = "1FUWdx1mJSY0P4cA4st5VBzQcCEEbq0Ap"
MODEL_ZIP = "disc_model.zip"
MODEL_DIR = "saved_model_dir"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_DIR):
        with st.spinner("📥 Downloading model..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_ZIP, quiet=False)
            
            with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(MODEL_ZIP)
    return MODEL_DIR

@st.cache_resource
def load_model():
    model_dir = download_model()
    model = tf.saved_model.load(model_dir)
    return model

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_defect(image, model):
    processed_image = preprocess_image(image)
    predictions = model(processed_image).numpy()[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100
    return predicted_class, confidence, predictions * 100

st.title("🔧 Disc Defect Detection")
st.write("Upload a disc image to detect defects")

with st.sidebar:
    st.header("About")
    st.write("**Defect Types:**")
    st.write("✅ Good")
    st.write("⚠️ Patches")
    st.write("❌ Rolled Pits")
    st.write("⚠️ Scratches")
    st.write("❌ Waist Folding")

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
        
        if predicted_class == 'good':
            st.success(f"✅ **{predicted_class.upper()}**")
        else:
            st.error(f"❌ **DEFECT: {predicted_class.upper().replace('_', ' ')}**")
        
        st.progress(confidence / 100)
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        st.write("**All Probabilities:**")
        for name, prob in zip(CLASS_NAMES, all_probs):
            st.write(f"{name.replace('_', ' ').title()}: {prob:.1f}%")
