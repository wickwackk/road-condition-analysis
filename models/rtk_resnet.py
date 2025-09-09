import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
from torchvision.models import resnet50, ResNet50_Weights

# Use the default pretrained weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# =========================
# 1. Data Transforms
# =========================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images to 224x224 (ResNet input size)
    transforms.RandomHorizontalFlip(), # Data augmentation
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]) # Standard normalization for ResNet
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# =========================
# 2. Load Dataset
# =========================
train_dataset = datasets.ImageFolder("RTK/train", transform=transform_train)
val_dataset   = datasets.ImageFolder("RTK/val", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

# =========================
# 3. Load Pretrained ResNet50
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)

# Replace final layer (ResNet50 has 1000 classes by default → we need 4)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

model = model.to(device)

# =========================
# 4. Loss & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# =========================
# 5. Training Loop
# =========================
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Each epoch has training and validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Loop through batches
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

# =========================
# 6. Run Training
# =========================
start = time.time()
model = train_model(model, criterion, optimizer, num_epochs=10)
end = time.time()
print("Training complete in {:.0f}m {:.0f}s".format((end-start)//60, (end-start)%60))

# =========================
# 7. Save Model
# =========================
torch.save(model.state_dict(), "road_resnet50.pth")
print("Model saved as road_resnet50.pth")
