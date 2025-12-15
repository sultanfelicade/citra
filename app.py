import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Lemon Quality Detection",
    layout="wide"
)

# Custom CSS for aesthetic look
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right bottom, #ffffff, #f0f2f6);
    }
    .main-title {
        font-size: 3.5rem;
        color: #333333;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #666666;
        text-align: center;
        margin-bottom: 40px;
        font-style: italic;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #333333;
        color: white;
        text-align: center;
        padding: 15px;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #333333;
        color: white;
        font-weight: bold;
        border-radius: 25px;
        border: none;
        padding: 15px 30px;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #555555;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model_terbaik.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

class_names = ['Bad Quality', 'Empty', 'Good Quality']

# Header Section
st.markdown('<div class="main-title">Lemon Quality Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Artificial Intelligence Based Quality Control</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader("Choose a lemon image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.markdown("### Analysis Results")
        analyze_button = st.button('Analyze Image')
        
        if analyze_button:
            if model is None:
                st.error("Model not loaded. Please check the model file.")
            else:
                with st.spinner('Analyzing lemon quality...'):
                    target_size = (224, 224)
                    img = image.resize(target_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    
                    predicted_class = class_names[np.argmax(predictions[0])]
                    confidence = 100 * np.max(score)

                    # Dynamic Color Logic
                    if predicted_class == 'Bad Quality':
                        result_color = "#FF4B4B" # Red
                    elif predicted_class == 'Good Quality':
                        result_color = "#2E8B57" # Green
                    else:
                        result_color = "#808080" # Gray for Empty

                    # Custom Result Display
                    st.markdown(f"""
                        <div class="result-card" style="border-left: 8px solid {result_color};">
                            <h2 style="color: {result_color}; margin:0;">Prediction: {predicted_class}</h2>
                            <p style="color: #666; font-size: 1.2rem; margin-top: 10px;">Confidence Level: <strong>{confidence:.2f}%</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### Probability Distribution")
                    chart_data = {label: float(pred) for label, pred in zip(class_names, predictions[0])}
                    st.bar_chart(chart_data, color=result_color)

# Footer
st.markdown('<div class="footer">Created by Sultan & Fathan</div>', unsafe_allow_html=True)
