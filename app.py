import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# Page config
st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")

# Title
st.title("🔢 Handwritten Digit Recognizer")
st.write("Upload a photo of a handwritten digit (0–9) and the model will predict it.")

# Load model — cached so it only loads once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('devesh_mnist_model.keras')

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show original image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your image")
        st.image(uploaded_file, width=200)

    # Preprocess
    img = Image.open(uploaded_file).convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = ImageOps.expand(img, border=30, fill=0)
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img) / 255.0

    # Normalize contrast
    if img_array.max() > img_array.min():
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

    # Show what model sees
    with col2:
        st.subheader("What model sees")
        st.image(img_array, width=200, clamp=True)

    # Predict
    prediction = model.predict(img_array[np.newaxis, ...], verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = prediction[0][predicted_digit] * 100

    # Show result
    st.markdown("---")
    st.subheader("Prediction")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Predicted Digit", predicted_digit)
    with col4:
        st.metric("Confidence", f"{confidence:.1f}%")

    # Show all probabilities as a bar chart
    st.subheader("Probabilities for all digits")
    probs = {str(i): float(prediction[0][i]) for i in range(10)}
    st.bar_chart(probs)