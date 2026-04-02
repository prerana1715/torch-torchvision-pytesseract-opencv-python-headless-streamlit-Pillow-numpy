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

# =========================
# CONFIG (Tesseract Path - Windows)
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL (MATCH TRAINING)
# =========================
model = models.resnet18(pretrained=False)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)

model.load_state_dict(torch.load("fraud_model.pth", map_location=device))
model.to(device)
model.eval()

# =========================
# TRANSFORM (MATCH TRAINING)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# =========================
# 🔥 PREDICT FUNCTION
# =========================
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = "Fake" if pred.item() == 1 else "Real"
    return label, confidence.item()


# =========================
# OCR FUNCTION
# =========================
def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text


# =========================
# 🔥 EXTRACT ALL AADHAAR NUMBERS
# =========================
def extract_all_aadhaar_numbers(text):
    text = text.replace('O', '0')
    text = text.replace('S', '5')

    pattern = r"\d{4}\s\d{4}\s\d{4}"
    matches = re.findall(pattern, text)

    return matches


# =========================
# FINAL DECISION LOGIC
# =========================
def verify_aadhaar(image_path):

    label, confidence = predict(image_path)
    text = extract_text(image_path)
    aadhaar_numbers = extract_all_aadhaar_numbers(text)

    # Debug (optional)
    print("Prediction:", label)
    print("Confidence:", confidence)
    print("Detected Numbers:", aadhaar_numbers)

    # ❌ FRAUD: Multiple Aadhaar numbers (overlay attack)
    if len(aadhaar_numbers) > 1:
        return "❌ Fraud Detected (Multiple Aadhaar Numbers)"

    # ❌ FRAUD: Strong CNN prediction
    if label == "Fake" and confidence > 0.6:
        return f"❌ Fraud Detected (Confidence: {confidence:.2f})"

    # ⚠️ Suspicious: No valid number
    if len(aadhaar_numbers) == 0:
        return "⚠️ Suspicious (Invalid Aadhaar Number)"

    # ⚠️ Low confidence
    if confidence < 0.6:
        return f"⚠️ Suspicious (Low Confidence: {confidence:.2f})"

    # ✅ VALID
    return f"✅ Valid Aadhaar: {aadhaar_numbers[0]} (Confidence: {confidence:.2f})"


# =========================
# STREAMLIT UI
# =========================
st.title("🪪 Aadhaar Fraud Detection System")

uploaded_file = st.file_uploader("Upload Aadhaar Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    file_path = "temp.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    result = verify_aadhaar(file_path)

    st.subheader("Result:")
    st.write(result)
