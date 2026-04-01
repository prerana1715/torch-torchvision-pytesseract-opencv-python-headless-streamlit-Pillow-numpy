# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pytesseract
import cv2
import re
import streamlit as st
import numpy as np

# =========================
# CONFIG (Tesseract Path - Windows)
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("fraud_model.pth", map_location=device))
model.to(device)
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# FRAUD PREDICTION FUNCTION
# =========================
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    return "Fake" if pred.item() == 1 else "Real"


# =========================
# OCR FUNCTION
# =========================
def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)
    return text


# =========================
# AADHAAR VALIDATION
# =========================
def validate_aadhaar(text):
    pattern = r"\d{4}\s\d{4}\s\d{4}"
    match = re.search(pattern, text)

    return match.group() if match else None


# =========================
# FINAL DECISION
# =========================
def verify_aadhaar(image_path):

    cnn_result = predict(image_path)
    text = extract_text(image_path)
    aadhaar_number = validate_aadhaar(text)

    if cnn_result == "Fake":
        return "❌ Fraud Detected"

    elif aadhaar_number is None:
        return "⚠️ Suspicious (Invalid Aadhaar Number)"

    else:
        return f"✅ Valid Aadhaar: {aadhaar_number}"


# =========================
# STREAMLIT UI
# =========================
st.title("🪪 Aadhaar Fraud Detection System")

uploaded_file = st.file_uploader("Upload Aadhaar Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    # Save uploaded file
    file_path = "temp.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Show image
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    # Run verification
    result = verify_aadhaar(file_path)

    st.subheader("Result:")
    st.write(result)