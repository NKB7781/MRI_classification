import streamlit as st
import numpy as np
import os
import gdown
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Class labels
class_labels = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Load models with caching
@st.cache_resource
def load_models():
    # Load Custom CNN
    cnn_model = load_model("custom_cnn_model.h5")

    # Check if ResNet model exists, if not download from Google Drive
    resnet_path = "best_resnet_model.h5"
    if not os.path.exists(resnet_path):
        # 🔁 Replace this with your Google Drive shareable file ID
        file_id = "1EoL148o3_WQYt-eL2DxC7kopbZA6ZGGD"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", resnet_path, quiet=False)

    resnet_model = load_model(resnet_path)
    return cnn_model, resnet_model

# Image preprocessing
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load both models
cnn_model, resnet_model = load_models()

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")
st.title("🧠 Brain Tumor Classification - Auto Model Selection")

st.write("Upload a brain MRI image. The app will use both models and select the better prediction automatically.")

uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img = preprocess_image(image)

        # Predictions
        cnn_pred = cnn_model.predict(img)
        resnet_pred = resnet_model.predict(img)

        # Confidence
        cnn_conf = np.max(cnn_pred)
        resnet_conf = np.max(resnet_pred)

        # Choose better prediction
        if cnn_conf >= resnet_conf:
            final_pred = cnn_pred
            model_used = "Custom CNN"
        else:
            final_pred = resnet_pred
            model_used = "ResNet50 Fine-tuned"

        pred_class = class_labels[np.argmax(final_pred)]
        confidence = np.max(final_pred) * 100

        st.success(f"🎯 Tumor Type: **{pred_class}**")
        st.info(f"🤖 Model Used: **{model_used}** | Confidence: **{confidence:.2f}%**")
