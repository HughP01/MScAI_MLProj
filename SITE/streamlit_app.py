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
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {'white' if st.session_state.theme == 'dark' else 'black'} !important;
    }}

    .stMarkdown h1, .st-emotion-cache-10trblm {{
        font-size: clamp(1.5rem, 5vw, 2.2rem);
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

    .stButton button {{
        width: auto !important;
        min-width: 160px;
        max-width: 100%;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        background: linear-gradient(45deg, #6e48aa, #9d50bb);
        color: white;
        font-weight: bold;
        font-size: 1rem;
        margin: 0.5rem auto;
        display: block;
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
