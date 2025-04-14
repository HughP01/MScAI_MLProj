import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# App configuration
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Toggle theme
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.rerun()

# CSS styling in a function
def get_custom_css():
    return f"""
    <style>
    .stApp {{
        background: {'#121212' if st.session_state.theme == 'dark' else 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)'} !important;
        padding: 8px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {'white' if st.session_state.theme == 'dark' else 'black'} !important;
    }}
    .stMarkdown, .stText, .stMetric {{ color: {'white' if st.session_state.theme == 'dark' else 'black'} !important; }}
    .stTabs, .stCameraInput, .stFileUploader, .stButton, .stImage {{ width: 100% !important; max-width: 100% !important; }}
    .st-emotion-cache-10trblm {{
        color: {'#ffffff' if st.session_state.theme == 'dark' else '#2c3e50'} !important;
        font-size: 1.8rem !important;
        margin-bottom: 0.5rem;
    }}
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
    .stCameraInput > div, .stImage > div {{
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
        background-color: {'#333333' if st.session_state.theme == 'dark' else 'white'} !important;
    }}
    .stButton button {{
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
    }}
    .stButton button:active {{ transform: scale(0.98); }}
    button[data-testid="baseButton-secondary"] {{
        min-height: 2.5rem !important;
        height: 2.5rem !important;
        width: 2.5rem !important;
        padding: 0 !important;
        margin-top: 0.5rem !important;
    }}
    .prediction-box {{
        border-radius: 12px;
        padding: 12px;
        background: {'rgba(30,30,30,0.95)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.95)'} !important;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid {'rgba(255,255,255,0.1)' if st.session_state.theme == 'dark' else 'rgba(0,0,0,0.05)'} !important;
        color: {'white' if st.session_state.theme == 'dark' else 'black'} !important;
    }}
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, #6e48aa, #9d50bb);
        height: 0.5rem !important;
        border-radius: 4px;
    }}
    .stMetric {{
        background: {'rgba(30,30,30,0.8)' if st.session_state.theme == 'dark' else 'rgba(255,255,255,0.8)'} !important;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
    }}
    </style>
    """

# Load a dummy model
@st.cache_resource
def load_model():
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

# Preprocess image
def preprocess_image(image):
    image = image.resize((48, 48))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Show prediction results
def display_results(prediction):
    class_names = {
        0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
        3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
        6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
        9: "No passing"
    }
    confidence = tf.nn.softmax(prediction[0])
    top_idx = np.argmax(confidence)

    st.markdown(f"""
    <div class="prediction-box">
        <h3 style="text-align: center;">Results</h3>
    """, unsafe_allow_html=True)

    st.metric(label="Most Likely", value=class_names[top_idx], delta=f"{confidence[top_idx]*100:.1f}%")

    st.subheader("All probabilities:")
    for i, (name, conf) in enumerate(zip(class_names.values(), confidence)):
        if conf > 0.01:
            st.progress(float(conf), text=f"{name}: {conf*100:.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)

# Main app layout
def main():
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🚦 Traffic Sign Detector")
    with col2:
        icon = '🌙' if st.session_state.theme == 'light' else '☀️'
        label = 'Switch to dark mode' if st.session_state.theme == 'light' else 'Switch to light mode'
        st.button(icon, on_click=toggle_theme, help=label, key='theme_toggle', use_container_width=True)

    # Apply CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    st.markdown("Snap or upload a photo of a traffic sign to identify it")

    tab1, tab2 = st.tabs(["📁 Upload", "📷 Camera"])

    with tab1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button("Analyze Image"):
                with st.spinner('Processing...'):
                    processed = preprocess_image(image)
                    prediction = model.predict(processed)
                    display_results(prediction)

    with tab2:
        st.write("Center the sign and tap the button below")
        img_buffer = st.camera_input("Take a picture", label_visibility="collapsed")
        if img_buffer is not None:
            image = Image.open(img_buffer)
            st.image(image, caption='Captured Image', use_column_width=True)
            if st.button("Analyze Photo"):
                with st.spinner('Processing...'):
                    processed = preprocess_image(image)
                    prediction = model.predict(processed)
                    display_results(prediction)

# Run the app
if __name__ == "__main__":
    main()
