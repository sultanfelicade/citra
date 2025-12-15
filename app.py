import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Lemon Quality Detection",
    layout="wide"
)

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

st.title("Lemon Quality Assessment System")
st.markdown("### Artificial Intelligence Based Quality Control")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        analyze_button = st.button('Analyze Image', type="primary")
        
        if analyze_button:
            if model is None:
                st.error("Model not loaded. Please check the model file.")
            else:
                with st.spinner('Processing...'):
                    target_size = (224, 224)
                    img = image.resize(target_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    
                    predicted_class = class_names[np.argmax(predictions[0])]
                    confidence = 100 * np.max(score)

                    st.success(f"Prediction: {predicted_class}")
                    st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
                    
                    st.markdown("#### Probability Distribution")
                    chart_data = {label: float(pred) for label, pred in zip(class_names, predictions[0])}
                    st.bar_chart(chart_data)
