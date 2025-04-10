import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path="efficientnetv2m_pneumonia_weights.pth"):
    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
    model = efficientnet_v2_m(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def predict(image, model):
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs