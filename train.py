# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from collections import Counter
from PIL import Image

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# TRANSFORMS
# =========================

# TRAIN TRANSFORM (augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# VALIDATION TRANSFORM (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# =========================
# LOAD DATASET
# =========================
dataset = datasets.ImageFolder("dataset", transform=train_transform)

print("Classes:", dataset.classes)

# =========================
# CHECK CLASS DISTRIBUTION
# =========================
labels = [label for _, label in dataset]
class_counts = Counter(labels)
print("Class Distribution:", class_counts)

# =========================
# TRAIN / VALIDATION SPLIT
# =========================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = random_split(dataset, [train_size, val_size])

# Apply validation transform
val_data.dataset.transform = val_transform

# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# =========================
# MODEL (ResNet18)
# =========================
model = models.resnet18(pretrained=True)

# 🔥 Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# 🔥 Train only final layer
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# =========================
# LOSS (CLASS BALANCING)
# =========================
total = sum(class_counts.values())
weights = [total/class_counts[i] for i in range(len(class_counts))]
weights = torch.tensor(weights).float().to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

# =========================
# OPTIMIZER
# =========================
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)

# =========================
# TRAINING
# =========================
epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

    # =========================
    # VALIDATION EACH EPOCH
    # =========================
    model.eval()
    correct = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_acc = correct / total_val
    print(f"Validation Accuracy: {val_acc:.4f}")

# =========================
# FINAL EVALUATION
# =========================
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nFinal Accuracy:", accuracy_score(y_true, y_pred))

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "fraud_model.pth")
print("Model saved as fraud_model.pth")

# =========================
# PREDICTION FUNCTION
# =========================
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = val_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = "Fake" if pred.item() == 1 else "Real"
    return label, confidence.item()

# =========================
# TEST
# =========================
test_image = "dataset/real/card_0.png"  # change this
label, conf = predict(test_image)

print(f"\nPrediction: {label}, Confidence: {conf:.2f}")
