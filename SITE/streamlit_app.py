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

# Add header with theme toggle
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🚦 Traffic Sign Detector")
with col2:
    # Display the theme toggle button
    if st.session_state.theme == 'light':
        st.button('🌙', on_click=toggle_theme, help="Switch to dark mode", 
                 key='theme_toggle', use_container_width=True)
    else:
        st.button('☀️', on_click=toggle_theme, help="Switch to light mode",
                 key='theme_toggle', use_container_width=True)

# Mobile-optimized CSS with light/dark mode support
def get_custom_css():
    return f"""
    <style>
    .stApp {{
        background: {'#121212' if st.session_state.theme == 'dark' else 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)'} !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {'white' if st.session_state.theme == 'dark' else 'black'} !important;
    }}

    /* Title with dynamic font size */
    .stMarkdown h1, .st-emotion-cache-10trblm {{
        font-size: clamp(0.8rem, 2vw, 1.2rem); /* Smaller font size for better mobile fit */
        line-height: 1.2;
        word-break: break-word;
        margin-bottom: 0.5rem;
        color: {'white' if st.session_state.theme == 'dark' else '#2c3e50'} !important;
    }}

    button[data-testid="baseButton-secondary"] {{
        width: 2.5rem !important;
        height: 2.5rem !important;
        padding: 0 !important;
        font-size: 1.2rem !important;
        line-height: 1.2 !important;
        margin-top: 0.5rem !important;
        border-radius: 50% !important;
    }}

    .stButton, .stButton>button {{
        width: 50px !important;  /* Set width */
        height: 50px !important;  /* Set height to match width */
        padding: 0 !important;  /* Remove internal padding */
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        font-size: 1.2rem;  /* Adjust font size */
        border-radius: 12px;  /* Rounded corners */
        background-color: {'#2c3e50' if st.session_state.theme == 'light' else '#1e1e1e'} !important;  /* Dark background for dark mode, light for light mode */
    }}

    /* Enhanced Tab container styling */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent;
        padding: 0.2rem;
        border-radius: 10px;
        gap: 0.5rem;
        display: flex;
        justify-content: space-evenly;
    }}

    /* Individual tab styles for visibility */
    .stTabs [data-baseweb="tab"] {{
        background: rgba(255, 255, 255, 0.3);
        border: 2px solid transparent;
        border-radius: 12px;
        color: {'#ffffff' if st.session_state.theme == 'dark' else '#333333'} !important;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        backdrop-filter: blur(4px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem;
    }}

    /* Active tab styling */
    .stTabs [aria-selected="true"] {{
        background: {'#9d50bb' if st.session_state.theme == 'dark' else '#6e48aa'};
        color: white !important;
        border-color: white;
        font-weight: bold;
        box-shadow: 0 3px 8px rgba(0,0,0,0.2);
    }}

    @media screen and (max-width: 480px) {{
        .stMarkdown h1 {{
            font-size: 1.4rem !important;
        }}
        .stButton button {{
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }}
    }}
    </style>
    """

# Inject custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Load model function with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("SITE/traffic_sign_cnn.h5")
model = load_model()

# Image preprocessing
def preprocess_image(image):
    image = image.resize((32, 32)) 
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]  
    image = image / 255.0
    return np.expand_dims(image, axis=0)


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

    st.markdown(f"""
    <div class="prediction-box">
        <h3 style="color: {'#ffffff' if st.session_state.theme == 'dark' else '#2c3e50'}; text-align: center;">Results</h3>
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

# Main app function
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

if __name__ == "__main__":
    main()
