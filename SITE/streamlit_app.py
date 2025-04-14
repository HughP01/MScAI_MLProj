import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# App configuration - optimized for mobile
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state for theme if not already set
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Function to toggle theme
def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'
    st.rerun()

# Add theme toggle button at the top
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🚦 Traffic Sign Detector")
with col2:
    # Display the theme toggle button
    if st.session_state.theme == 'light':
        st.button('🌙', on_click=toggle_theme, help="Switch to dark mode")
    else:
        st.button('☀️', on_click=toggle_theme, help="Switch to light mode")

# Mobile-optimized CSS with light/dark mode support
st.markdown(f"""
    <style>
    /* Base mobile-friendly styles */
    .stApp {{
        background: {'#121212' if st.session_state.theme == 'dark' else 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)'} !important;
        padding: 8px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {'white' if st.session_state.theme == 'dark' else 'black'} !important;
    }}
    
    /* Theme-aware text colors */
    .stMarkdown, .stText, .stMetric, .stProgress .st-emotion-cache-1ru4j3m {{
        color: {'white' if st.session_state.theme == 'dark' else 'black'} !important;
    }}
    
    /* Full-width elements on mobile */
    .stTabs, .stCameraInput, .stFileUploader, .stButton, .stImage {{
        width: 100% !important;
        max-width: 100% !important;
    }}
    
    /* Header styling for mobile */
    .st-emotion-cache-10trblm {{
        color: {'#ffffff' if st.session_state.theme == 'dark' else '#2c3e50'} !important;
        font-size: 1.8rem !important;
        margin-bottom: 0.5rem;
    }}
    
    /* Tab styling for mobile */
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        gap: 4px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        flex: 1;
        padding: 0.5rem 0.3rem !important;
        margin: 0 !important;
        border-radius: 12px !important;
        background: {'rgba(30,30,30,0.7)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.7)'} !important;
        font-size: 0.85rem !important;
        min-width: auto !important;
        color: {'white' if st.session_state.theme == 'dark' else 'black'} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {'#333333' if st.session_state.theme == 'dark' else 'white'} !important;
        color: {'#9d50bb' if st.session_state.theme == 'dark' else '#6e48aa'} !important;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Camera and image styling */
    .stCameraInput > div, .stImage > div {{
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
        background-color: {'#333333' if st.session_state.theme == 'dark' else 'white'} !important;
    }}
    
    /* Button styling for mobile */
    .stButton button {{
        width: 20%;
        border-radius: 12px;
        background: linear-gradient(45deg, #6e48aa, #9d50bb);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.7rem;
        font-size: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    .stButton button:active {{
        transform: scale(0.98);
    }}
    
    /* Theme toggle button specific styling */
    .st-emotion-cache-7ym5gk {{
        min-height: 2.5rem !important;
        height: 2.5rem !important;
        width: 2.5rem !important;
        padding: 0 !important;
        margin-top: 0.5rem !important;
    }}
    
    /* Prediction box */
    .prediction-box {{
        border-radius: 12px;
        padding: 12px;
        background: {'rgba(30,30,30,0.95)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.95)'} !important;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid {'rgba(255,255,255,0.1)' if st.session_state.theme == 'dark' else 'rgba(0,0,0,0.05)'} !important;
        color: {'white' if st.session_state.theme == 'dark' else 'black'} !important;
    }}
    
    /* Progress bar styling */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, #6e48aa, #9d50bb);
        height: 0.5rem !important;
        border-radius: 4px;
    }}
    
    /* Metric styling for mobile */
    .stMetric {{
        background: {'rgba(30,30,30,0.8)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.8)'} !important;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
    }}
    
    /* Hide unnecessary elements on mobile */
    @media (max-width: 768px) {{
        /* Remove extra padding */
        .main > div {{
            padding: 0.5rem !important;
        }}
        
        /* Make text more readable */
        .stMarkdown p, .stMarkdown li {{
            font-size: 0.9rem !important;
            line-height: 1.4 !important;
        }}
        
        /* Adjust progress bar labels */
        .stProgress .st-emotion-cache-1ru4j3m {{
            font-size: 0.8rem !important;
        }}
    }}
    
    /* Very small devices adjustments */
    @media (max-width: 400px) {{
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.75rem !important;
            padding: 0.4rem 0.2rem !important;
        }}
        
        .stButton button {{
            padding: 0.6rem;
            font-size: 0.9rem;
        }}
        
        .st-emotion-cache-7ym5gk {{
            min-height: 2rem !important;
            height: 2rem !important;
            width: 2rem !important;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# [Rest of your code remains the same...]
# Load model function, image processing, and main app logic would follow here
# Make sure to keep all your existing functionality

def main():
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
