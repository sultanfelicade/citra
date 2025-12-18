import streamlit as st
try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error("TensorFlow module not found. Please try running the app with: `python -m streamlit run app.py`")
    st.stop()
import numpy as np
import matplotlib.cm as cm
import cv2
from PIL import Image

st.set_page_config(
    page_title="Lemon Quality Detection",
    layout="wide"
)

# --- CSS STYLES ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to right bottom, #ffffff, #f0f2f6); }
    .main-title { font-size: 3.5rem; color: #333; text-align: center; font-weight: 800; }
    .sub-title { font-size: 1.5rem; color: #666; text-align: center; margin-bottom: 40px; font-style: italic; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #333; color: white; text-align: center; padding: 15px; }
    .result-card { background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI GRAD-CAM (MODIFIKASI PENTING DISINI) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Get the model up to the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.models.Model(model.inputs, last_conv_layer.output)

    # 2. Create the classifier model (from last conv layer to output)
    # We iterate over layers after the last conv layer
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    # Find the index of the last conv layer
    try:
        layer_names = [layer.name for layer in model.layers]
        last_conv_index = layer_names.index(last_conv_layer_name)
        
        # Reconstruct the classifier part
        for layer in model.layers[last_conv_index+1:]:
            x = layer(x)
            
        classifier_model = tf.keras.models.Model(classifier_input, x)
    except Exception as e:
        st.error(f"Error reconstructing classifier model: {e}")
        return None

    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer
        last_conv_layer_output = last_conv_layer_model(img_array)
        
        # Watch the conv output variable
        tape.watch(last_conv_layer_output)
        
        # Compute predictions
        preds = classifier_model(last_conv_layer_output)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Hitung gradien output target terhadap output layer konvolusi terakhir
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap ke range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Gunakan colormap jet (seperti contoh kamu)
    jet = cm.get_cmap("jet")
    
    # Gunakan nilai RGB dari colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Buat gambar heatmap dari array warna
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose heatmap ke gambar asli
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img

def find_last_conv_layer(model):
    """Mencari layer Conv2D terakhir secara otomatis"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model_terbaik.keras') # Sesuaikan nama file
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
class_names = ['Bad Quality', 'Empty Background', 'Good Quality']

# --- UI LAYOUT ---
st.markdown('<div class="main-title">Lemon Quality Check</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Tugas Pengolahan citraa</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader("Choose a lemon image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Original Uploaded Image', use_container_width=True)

    with col2:
        st.markdown("### Analysis Results")
        analyze_button = st.button('Analyze Image')
        
        if analyze_button:
            if model is None:
                st.error("Model Error.")
            else:
                with st.spinner('Analyzing patterns & texture...'):
                    # --- PREPROCESSING ---
                    target_size = (224, 224)
                    img_resized = image.resize(target_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = img_array / 255.0
                    img_input = np.expand_dims(img_array, axis=0)

                    # --- PREDICTION ---
                    predictions = model.predict(img_input)
                    score = predictions[0]
                    predicted_index = np.argmax(score)
                    predicted_class = class_names[predicted_index]
                    confidence = 100 * np.max(score)

                    if predicted_index == 0: result_color = "#FF4B4B"
                    elif predicted_index == 2: result_color = "#2E8B57"
                    else: result_color = "#808080"

                    # --- GRAD-CAM VISUALIZATION ---
                    # 1. Cari layer terakhir
                    last_layer_name = find_last_conv_layer(model)
                    
                    if last_layer_name:
                        # 2. Buat heatmap
                        heatmap = make_gradcam_heatmap(img_input, model, last_layer_name, predicted_index)
                        
                        # 3. Gabungkan dengan gambar asli (Perlu denormalisasi x255 biar kelihatan)
                        cam_image = display_gradcam(img_array * 255, heatmap)
                    else:
                        cam_image = None
                        st.warning("Could not find Convolutional Layer for visualization.")

                    # --- DISPLAY RESULT CARD ---
                    st.markdown(f"""
                        <div class="result-card" style="border-left: 8px solid {result_color};">
                            <h2 style="color: {result_color}; margin:0;">Prediction: {predicted_class}</h2>
                            <p style="color: #666; font-size: 1.2rem; margin-top: 10px;">Confidence Level: <strong>{confidence:.2f}%</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # --- DISPLAY XAI (HEATMAP) ---
                    st.markdown("#### Visual Focus")
                    if cam_image:
                        st.image(cam_image, caption=f"Heatmap Area ", use_container_width=True)
                    else:
                        st.image(img_resized, caption="Processed Image", width=224)

                    # Bar Chart
                    st.markdown("#### Probability Distribution")
                    chart_data = {label: float(pred) for label, pred in zip(class_names, score)}
                    st.bar_chart(chart_data, color=result_color)

st.markdown('<div class="footer">Created by Sultan</div>', unsafe_allow_html=True)