import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime

# App configuration - optimized for mobile
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered",  # Changed from "wide" to "centered" for better mobile display
    initial_sidebar_state="collapsed"
)

# Mobile-optimized CSS with visual enhancements
st.markdown("""
    <style>
    /* Base mobile-friendly styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 8px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Full-width elements on mobile */
    .stTabs, .stCameraInput, .stFileUploader, .stButton, .stImage {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Header styling for mobile */
    .st-emotion-cache-10trblm {
        color: #2c3e50;
        font-size: 1.8rem !important;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    /* Tab styling for mobile */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        padding: 0.5rem 0.3rem !important;
        margin: 0 !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.7) !important;
        font-size: 0.85rem !important;
        min-width: auto !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #6e48aa !important;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Camera and image styling */
    .stCameraInput > div, .stImage > div {
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    
    /* Button styling for mobile */
    .stButton button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(45deg, #6e48aa, #9d50bb);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.7rem;
        font-size: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stButton button:active {
        transform: scale(0.98);
    }
    
    /* Prediction box for mobile */
    .prediction-box {
        border-radius: 12px;
        padding: 12px;
        background: rgba(255,255,255,0.95);
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6e48aa, #9d50bb);
        height: 0.5rem !important;
        border-radius: 4px;
    }
    
    /* Metric styling for mobile */
    .stMetric {
        background: rgba(255,255,255,0.8);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
    }
    
    /* Hide unnecessary elements on mobile */
    @media (max-width: 768px) {
        /* Remove extra padding */
        .main > div {
            padding: 0.5rem !important;
        }
        
        /* Make text more readable */
        .stMarkdown p, .stMarkdown li {
            font-size: 0.9rem !important;
            line-height: 1.4 !important;
        }
        
        /* Adjust progress bar labels */
        .stProgress .st-emotion-cache-1ru4j3m {
            font-size: 0.8rem !important;
        }
    }
    
    /* Very small devices adjustments */
    @media (max-width: 400px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 0.75rem !important;
            padding: 0.4rem 0.2rem !important;
        }
        
        .stButton button {
            padding: 0.6rem;
            font-size: 0.9rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model function with caching
@st.cache_resource
def load_model():
    # Replace with your actual model loading code
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
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Main app - simplified for mobile
def main():
    st.title("🚦 Traffic Sign Detector")
    st.markdown("Snap or upload a photo of a traffic sign to identify it")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload", "📷 Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image", 
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
        st.write("Center the sign and tap the button below")
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
    class_names = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing for vehicles over 3.5 metric tons"
    }
    
    confidence = tf.nn.softmax(prediction[0])
    top_idx = np.argmax(confidence)
    
    st.markdown("""
    <div class="prediction-box">
        <h3 style="color: #2c3e50; text-align: center;">Results</h3>
    """, unsafe_allow_html=True)
    
    # Top prediction - mobile optimized
    st.metric(label="Most Likely", 
              value=class_names[top_idx],
              delta=f"{confidence[top_idx]*100:.1f}%")
    
    # All classes with compact display
    st.subheader("All probabilities:")
    for i, (name, conf) in enumerate(zip(class_names, confidence)):
        if conf > 0.01:  # Only show significant probabilities
            st.progress(float(conf), text=f"{name}: {conf*100:.1f}%")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
