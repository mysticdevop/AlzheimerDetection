import os
import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityd
)
from monai.data import Dataset
from monai.networks.nets import resnet18
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score

DATA_DIR = r"C:\Users\krita\OneDrive\alzymer_dataset\dataset split patient\train"

def load_data(folder):
    data = []
    
    for label_name, label_val in [("CN", 0), ("AD", 1)]:
        path = os.path.join(folder, label_name)
        
        if not os.path.exists(path):
            continue

        for img in os.listdir(path):
            if img.endswith(".png"):
                data.append({
                    "image": os.path.join(path, img),
                    "label": label_val
                })
    return data

train_data = load_data(r"C:\Users\krita\OneDrive\alzymer_dataset\dataset split patient\train")
val_data = load_data(r"C:\Users\krita\OneDrive\alzymer_dataset\dataset split patient\val")

print("Train:", len(train_data))
print("Val:", len(val_data))

train_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(224, 224)),
    ScaleIntensityd(keys=["image"])
])

val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(224, 224)),
    ScaleIntensityd(keys=["image"])
])
train_dataset = Dataset(data=train_data, transform=train_transforms)
val_dataset = Dataset(data=val_data, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = resnet18(
    spatial_dims=2,
    n_input_channels=1,
    num_classes=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {acc:.4f}")

model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)


print("Classification Report:")
print(classification_report(all_labels, all_preds))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("AUC:", roc_auc_score(all_labels, all_probs))

print("Balanced Accuracy:", balanced_accuracy_score(all_labels, all_preds))

torch.save(model.state_dict(), "binary_resnet_monai.pth")

