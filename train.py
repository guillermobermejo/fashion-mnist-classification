import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
SEED = 42

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Data
# -----------------------------
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------
# Model: Simple CNN
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 28 -> 26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # after pool: 13 -> 11
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # (N, 32, 26, 26)
        x = self.pool(x)              # (N, 32, 13, 13)
        x = self.relu(self.conv2(x))  # (N, 64, 11, 11)
        x = self.pool(x)              # (N, 64, 5, 5)
        x = self.flatten(x)           # (N, 64*5*5)
        x = self.relu(self.fc1(x))    # (N, 128)
        x = self.fc2(x)               # (N, 10)
        return x

# -----------------------------
# Model: Simple MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total if total else 0.0
    return acc, np.array(all_preds), np.array(all_labels)


def save_confusion_matrix_png(cm: np.ndarray, out_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def print_per_class_accuracy(cm: np.ndarray) -> None:
    print("Per-class accuracy:")

    per_class = []
    for i in range(cm.shape[0]):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        acc = correct / total if total > 0 else 0.0
        per_class.append(acc)

        print(f"- {CLASS_NAMES[i]:12s}: {acc:.4f} ({correct}/{total})")

    worst_i = int(np.argmin(per_class))
    best_i = int(np.argmax(per_class))

    print(f"\nLowest class accuracy: {CLASS_NAMES[worst_i]} ({per_class[worst_i]:.4f})")
    print(f"Highest class accuracy: {CLASS_NAMES[best_i]} ({per_class[best_i]:.4f})")


def run_experiment(model: nn.Module, name: str):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"[{name}] Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

    acc, preds, labels = evaluate(model, test_loader)
    cm = confusion_matrix(labels, preds)

    print_per_class_accuracy(cm)

    print(f"\n[{name}] Test Accuracy: {acc:.4f}\n")
    save_confusion_matrix_png(cm, f"results/{name}_confusion_matrix.png", f"{name} Confusion Matrix")

    return acc


def main():
    mlp_acc = run_experiment(MLP(), "MLP")
    cnn_acc = run_experiment(CNN(), "CNN")

    print("=== Summary ===")
    print(f"MLP Accuracy: {mlp_acc:.4f}")
    print(f"CNN Accuracy: {cnn_acc:.4f}")


if __name__ == "__main__":
    main()