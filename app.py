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
EXTRACT_DIR = "extracted_model"

@st.cache_resource
def download_and_extract_model():
    """Download and extract the model, return the path to saved_model directory"""
    if not os.path.exists(MODEL_ZIP):
        with st.spinner("📥 Downloading model..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_ZIP, quiet=False)
    
    # Extract the ZIP file
    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    
    # Find the directory containing saved_model.pb
    def find_saved_model_dir(root_dir):
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if 'saved_model.pb' in filenames:
                return dirpath
        return None
    
    model_path = find_saved_model_dir(EXTRACT_DIR)
    
    if model_path is None:
        raise FileNotFoundError("Could not find saved_model.pb in extracted files")
    
    return model_path

@st.cache_resource
def load_model():
    model_path = download_and_extract_model()
    st.info(f"Loading model from: {model_path}")
    model = tf.saved_model.load(model_path)
    return model

def preprocess_image(image):
    """Preprocess the image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_defect(image, model):
    """Make prediction on the image"""
    processed_image = preprocess_image(image)
    
    # Get the inference function
    infer = model.signatures['serving_default']
    
    # Convert to tensor
    input_tensor = tf.convert_to_tensor(processed_image)
    
    # Make prediction
    predictions = infer(input_tensor)
    
    # Extract output (key might vary, common ones are 'output_0' or 'dense')
    output_key = list(predictions.keys())[0]
    predictions = predictions[output_key].numpy()[0]
    
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100
    
    return predicted_class, confidence, predictions * 100

# UI Components
st.title("🔧 Disc Defect Detection")
st.write("Upload a disc image to detect defects")

with st.sidebar:
    st.header("About")
    st.write("**Defect Types:**")
    for class_name in CLASS_NAMES:
        emoji = "✅" if class_name == "good" else "❌"
        st.write(f"{emoji} {class_name.replace('_', ' ').title()}")
    
    st.divider()
    st.write("**Model Info:**")
    st.write(f"Input Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    st.write(f"Classes: {len(CLASS_NAMES)}")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        try:
            model = load_model()
            
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, all_probs = predict_defect(image, model)
            
            # Display main result
            if predicted_class == 'good':
                st.success(f"✅ **{predicted_class.upper()}**")
                st.balloons()
            else:
                st.error(f"❌ **DEFECT DETECTED: {predicted_class.upper().replace('_', ' ')}**")
            
            # Confidence meter
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(confidence / 100)
            
            # Detailed probabilities
            st.divider()
            st.write("**Detailed Probabilities:**")
            
            # Sort by probability
            prob_data = sorted(zip(CLASS_NAMES, all_probs), key=lambda x: x[1], reverse=True)
            
            for name, prob in prob_data:
                emoji = "✅" if name == "good" else "❌"
                st.write(f"{emoji} **{name.replace('_', ' ').title()}:** {prob:.2f}%")
                st.progress(prob / 100)
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.exception(e)

# Footer
st.divider()
st.caption("Upload a clear image of the disc surface for best results")
