import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# ==== Data loader ====

def load_count_annotations(txt_path):
    count_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                image_id = parts[0].zfill(4) + '.jpg'
                count = int(parts[1])
                count_dict[image_id] = count
    return count_dict

class CountRegressionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.count_dict = load_count_annotations(annotation_file)
        self.image_names = sorted(self.count_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # Normalized count (divide by max approx. value)
        count = self.count_dict[img_name] / 2000.0
        count = torch.tensor([count], dtype=torch.float32)

        return image, count

# ==== Model ====

class CountCNN(nn.Module):
    def __init__(self):
        super(CountCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# ==== Setup ====

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

image_dir = '/Users/nick/Downloads/jhu_crowd_v2.0/train/images'
annotation_file = '/Users/nick/Artificial Neural Networks and Deep Learning/https:/trainimage_labels.txt'

from torch.utils.data import Subset  # 👈 add this at the top with your imports
import random

dataset = CountRegressionDataset(image_dir, annotation_file, transform=transform)
indices = list(range(len(dataset)))
random.shuffle(indices)
subset = Subset(dataset, indices[:200])
dataloader = DataLoader(subset, batch_size=8, shuffle=True)

val_image_dir = "/Users/nick/Downloads/jhu_crowd_v2.0/val/images"
val_annotation_file = "/Users/nick/Artificial Neural Networks and Deep Learning/https:/valimage_labels.txt"
val_dataset = CountRegressionDataset(val_image_dir, val_annotation_file, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ==== Model + Optimizer ====

model = CountCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Load previous weights if available
if os.path.exists("count_model.pth"):
    model.load_state_dict(torch.load("count_model.pth", map_location=device))
    print("✅ Loaded saved model weights")

# ==== Training ====

best_mae = float('inf')
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, counts in dataloader:
        images = images.to(device)
        counts = counts.to(device)

        preds = model(images)
        loss = criterion(preds, counts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ==== Validation ====
    model.eval()
    true_counts = []
    pred_counts = []

    with torch.no_grad():
        for images, counts in val_loader:
            images = images.to(device)
            counts = counts.to(device)

            preds = model(images).squeeze(1)
            # Undo normalization
            true_counts.extend((counts.cpu().numpy().flatten()) * 2000)
            pred_counts.extend((preds.cpu().numpy().flatten()) * 2000)

    mae = mean_absolute_error(true_counts, pred_counts)
    mse = mean_squared_error(true_counts, pred_counts)
    rmse = math.sqrt(mse)

    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation MSE: {mse:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")

    # Save best model
    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Best model saved at epoch {epoch+1} (MAE: {mae:.2f})")

# Final save
torch.save(model.state_dict(), "count_model.pth")
print("✅ Final model saved as count_model.pth")
