"""
CORN (Mısır Yaprak Hastalığı) veri seti üzerinde
DenseNet169 mimarisi ile **sıfırdan (random initialization)** eğitim yapan betik.

Bu dosya, ön-eğitimli (pretrained) ağırlıklar kullanmadan, DenseNet169 modelini
tamamen rastgele ağırlıklarla başlatır ve veri seti üzerinde baştan eğitir.
"""

# ============================================
#   CORN (MISIR YAPRAK HASTALIĞI) - DenseNet169 (From Scratch)
# ============================================

import time

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms

try:
    # Google Colab ortamında Drive bağlamak için
    from google.colab import drive  # type: ignore
except ImportError:  # pragma: no cover
    drive = None

# ======================
# 1. AYARLAR
# ======================
DATA_DIR = "/content/drive/MyDrive/data"  # Healthy, Blight, Common_Rust, Gray_Leaf_Spot
BATCH_SIZE = 8
EPOCHS = 50
LR = 0.0001
SEED = 42

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Tekrarlanabilirlik için rastgelelik tohumunu (seed) ayarla."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================
# 2. TRANSFORM
# ======================
train_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ======================
# 3. VERİ SETİ YAPISI
# ======================
class TransformSubset(Dataset):
    """random_split ile elde edilen alt kümelere transform uygulayan yardımcı Dataset sınıfı."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.transform(img)
        return img, label


def create_dataloaders(data_dir: str, batch_size: int):
    """ImageFolder veri setini okuyup eğitim/validasyon/test DataLoader'larını döndür."""
    base_dataset = datasets.ImageFolder(data_dir)
    classes = base_dataset.classes
    num_classes = len(classes)
    print("Sınıflar:", classes)

    # Bölme (70-15-15)
    total = len(base_dataset)
    train_len = int(total * 0.7)
    val_len = int(total * 0.15)
    test_len = total - train_len - val_len

    train_base, val_base, test_base = random_split(
        base_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_set = TransformSubset(train_base, train_tf)
    val_set = TransformSubset(val_base, test_tf)
    test_set = TransformSubset(test_base, test_tf)

    # Donma sorunu giderilmiş DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, classes, num_classes


# ======================
# 4. MODEL (SIFIRDAN)
# ======================
def build_model(num_classes: int) -> nn.Module:
    """
    DenseNet169 modelini ön-eğitimli ağırlıklar olmadan, sıfırdan başlat.

    Not: weights=None -> rastgele ağırlıklar.
    """
    model = models.densenet169(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(DEVICE)


# ======================
# 5. EĞİTİM VE DOĞRULAMA
# ======================
def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
):
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    print("Kullanılan cihaz:", DEVICE)
    print("\n--- Eğitim Başladı (Sıfırdan) ---\n")
    start = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        ep_loss = running_loss / len(train_loader)
        ep_acc = 100.0 * correct / total if total > 0 else 0.0

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                v_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        v_loss = v_loss / len(val_loader)
        v_acc = 100.0 * v_correct / v_total if v_total > 0 else 0.0

        train_loss_history.append(ep_loss)
        val_loss_history.append(v_loss)
        train_acc_history.append(ep_acc)
        val_acc_history.append(v_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {ep_loss:.4f} | Acc: {ep_acc:.2f}% | "
            f"Val Loss: {v_loss:.4f} | Acc: {v_acc:.2f}%"
        )

    elapsed_min = (time.time() - start) / 60.0
    print(f"\nEğitim tamamlandı ({elapsed_min:.2f} dk)")

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history


# ======================
# 6. GRAFİKLER
# ======================
def plot_training_curves(
    train_loss, val_loss, train_acc, val_acc, epochs: int
) -> None:
    """Loss ve accuracy eğrilerini çiz."""
    epoch_indices = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_indices, train_loss, label="Train Loss")
    plt.plot(epoch_indices, val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epoch_indices, train_acc, label="Train Acc")
    plt.plot(epoch_indices, val_acc, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.show()


# ======================
# 7. TEST SONUÇLARI
# ======================
def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    classes,
) -> None:
    """Test kümesi üzerinde sınıflandırma raporu ve karışıklık matrisi üret."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def main():
    """Ana çalışma fonksiyonu (sıfırdan eğitim)."""
    set_seed(SEED)

    # Google Colab ortamında çalışılıyorsa Drive'ı bağla
    if drive is not None:
        drive.mount("/content/drive")
    else:
        print("google.colab bulunamadı, Drive bağlama atlandı.")

    train_loader, val_loader, test_loader, classes, num_classes = create_dataloaders(
        DATA_DIR, BATCH_SIZE
    )

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss, val_loss, train_acc, val_acc = train_and_validate(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS
    )
    plot_training_curves(train_loss, val_loss, train_acc, val_acc, EPOCHS)
    evaluate_on_test(model, test_loader, classes)


if __name__ == "__main__":
    main()

# ============================================
#   CORN (MISIR YAPRAK HASTALIĞI) - DenseNet169
# ============================================
# = 0 dan eğitim bu şekilde yapılmaktadır
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os

# ======================
# 1. GOOGLE DRIVE BAĞLA
# ======================
from google.colab import drive
drive.mount('/content/drive')

# ======================
# 2. AYARLAR
# ======================
DATA_DIR = "/content/drive/MyDrive/data"   # Healthy, Blight, Common_Rust, Gray_Leaf_Spot
BATCH_SIZE = 8
EPOCHS = 50
LR = 0.0001

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# ======================
# 3. TRANSFORM
# ======================
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ======================
# 4. VERİ OKU
# ======================
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.transform(img)
        return img, label

base_dataset = datasets.ImageFolder(DATA_DIR)
classes = base_dataset.classes
num_classes = len(classes)
print("Sınıflar:", classes)

# Bölme (70-15-15)
total = len(base_dataset)
train_len = int(total * 0.7)
val_len = int(total * 0.15)
test_len = total - train_len - val_len

train_base, val_base, test_base = random_split(
    base_dataset, [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(42)
)

train_set = TransformSubset(train_base, train_tf)
val_set   = TransformSubset(val_base, test_tf)
test_set  = TransformSubset(test_base, test_tf)

# 🔥 DONMA SORUNU GİDERİLMİŞ DATA LOADER
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ======================
# 5. DENSENET169 MODEL
# ======================
# = 0 dan eğitim bu şekilde yapılmaktadır
model = models.densenet169(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ======================
# 6. EĞİTİM
# ======================
train_loss, val_loss = [], []
train_acc, val_acc = [], []

print("\n--- Eğitim Başladı ---\n")
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    ep_loss = running_loss / len(train_loader)
    ep_acc = 100 * correct / total

    # Validation
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            v_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            v_correct += (preds == labels).sum().item()
            v_total += labels.size(0)

    v_loss = v_loss / len(val_loader)
    v_acc = 100 * v_correct / v_total

    train_loss.append(ep_loss)
    val_loss.append(v_loss)
    train_acc.append(ep_acc)
    val_acc.append(v_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {ep_loss:.4f} | Acc: {ep_acc:.2f}% | "
          f"Val Loss: {v_loss:.4f} | Acc: {v_acc:.2f}%")

print(f"\nEğitim tamamlandı ({(time.time()-start)/60:.2f} dk)")

# ======================
# 7. GRAFİKLER
# ======================
epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, val_acc, label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.show()

# ======================
# 8. TEST SONUÇLARI
# ======================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(all_labels, all_preds, target_names=classes))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
