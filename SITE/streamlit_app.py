import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime

# App configuration
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile optimization
st.markdown("""
    <style>
    .stApp {
        max-width: 100% !important;
        padding: 10px !important;
    }
    .stCameraInput > div {
        border-radius: 15px;
        overflow: hidden;
    }
    .st-eb {
        padding: 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(45deg, #6e48aa, #9d50bb);
        color: white;
        font-weight: bold;
        border: none;
    }
    .stFileUploader {
        border-radius: 20px;
    }
    .prediction-box {
        border-radius: 15px;
        padding: 15px;
        background: #f8f9fa;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load model function with caching
@st.cache_resource
def load_model():
    # Replace with your actual model loading code
    # Example: return tf.keras.models.load_model('my_model.h5')
    # For demo purposes, we'll use a placeholder
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    image = image.resize((48, 48))
    image = np.array(image)
    # Convert RGBA to RGB if needed
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Main app
def main():
    st.title("Traffic Sign Detector")
    st.markdown("Upload an image or use your camera to get predictions")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Take Photo"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner('Processing...'):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)
                    display_results(prediction)
    
    with tab2:
        st.write("Position your object and take a photo:")
        img_file_buffer = st.camera_input("Take a picture", label_visibility="collapsed")
        
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            st.image(image, caption='Captured Image', use_column_width=True)
            
            if st.button("Analyze Photo"):
                with st.spinner('Processing...'):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)
                    display_results(prediction)

def display_results(prediction):
    # Replace with your actual class labels
    class_names = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", 
                   "Class 6", "Class 7", "Class 8", "Class 9", "Class 10"]
    
    # Get top prediction
    confidence = tf.nn.softmax(prediction[0])
    top_idx = np.argmax(confidence)
    
    st.markdown("""
    <div class="prediction-box">
        <h3>Analysis Results</h3>
    """, unsafe_allow_html=True)
    
    # Top prediction
    st.metric(label="Most Likely", 
              value=class_names[top_idx],
              delta=f"{confidence[top_idx]*100:.1f}% confidence")
    
    # All classes
    st.subheader("All Class Probabilities:")
    for i, (name, conf) in enumerate(zip(class_names, confidence)):
        st.progress(float(conf), text=f"{name}: {conf*100:.1f}%")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
