import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from efficientnet_utils import load_model, predict

model = load_model()

st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    if st.button("🔍 Diagnose"):
        probs = predict(image, model)
        class_names = ["Normal", "Pneumonia"]
        pred_class = class_names[np.argmax(probs)]
        st.success(f"**Prediction:** {pred_class}")

        st.subheader("Confidence")
        fig, ax = plt.subplots()
        ax.bar(class_names, probs, color=['skyblue', 'salmon'])
        ax.set_ylim([0, 1])
        for i, v in enumerate(probs):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
        st.pyplot(fig)