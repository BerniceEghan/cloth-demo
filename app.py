import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Clothing Classifier", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()

# Load class mapping
with open("class_mapping.json", "r") as f:
    class_map = json.load(f)

idx_to_name = class_map["idx_to_name"]

# Preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

st.title("Clothing Classifier (EfficientNetB0)")
st.write("Upload an image to classify the clothing type.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img_array = preprocess_image(image)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_name = idx_to_name[pred_idx]
    confidence = float(np.max(preds))

    st.subheader("Prediction Results")
    st.write(f"**Class:** {pred_name}")
    st.write(f"**Confidence:** {confidence:.4f}")
