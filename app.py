import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from efficientnet_pytorch import EfficientNet

# Load the model with locally uploaded weights
@st.cache_resource
def load_model():
    model = EfficientNet.from_name("efficientnet-b4", num_classes=2)
    weight_path = os.path.join(os.path.dirname(__file__), "efficientnetv2m_pneumonia_weights.pth")
    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("Pneumonia Detection from Chest X-ray")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_tensor = transform(image).unsqueeze(0)

    with st.spinner("Predicting..."):
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        label = "Pneumonia" if predicted.item() == 1 else "Normal"
        st.success(f"Prediction: {label}")
