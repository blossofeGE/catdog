import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 8
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")


class ConvBlock(nn.Module):
    """
    Один блок:
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class CatDogCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        # Не зависит от точного размера после последнего pooling.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """He/Kaiming initialization для ReLU-сетей."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# =========================
# DATA
# =========================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1)
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15
    ),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


# =========================
# VALIDATION
# =========================
def evaluate(model, criterion, val_loader):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# =========================
# TRAIN
# =========================
def train_model():
    # Загружаем датасеты
    train_dataset = datasets.ImageFolder(
        'dataset/train',
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        'dataset/val',
        transform=val_transforms
    )

    print("Классы:", train_dataset.class_to_idx)
    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = CatDogCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    # AdamW обычно удобнее
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # Если validation перестала расти, уменьшаем learning rate.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.3,
        patience=3
    )

    best_val_acc = 0.0
    epochs_without_improvement = 0

    os.makedirs('models', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()

        total = 0
        correct = 0
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, criterion, val_loader)

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"\nЭпоха {epoch + 1}/{EPOCHS}"
            f"\nTrain Loss: {train_loss:.4f}"
            f" | Train Acc: {train_acc * 100:.2f}%"
            f"\nVal Loss:   {val_loss:.4f}"
            f" | Val Acc:   {val_acc * 100:.2f}%"
            f"\nLearning rate: {current_lr:.6f}"
        )

        # Сохраняем лучшую, а не последнюю модель.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': train_dataset.class_to_idx,
                    'img_size': IMG_SIZE,
                    'val_acc': val_acc,
                },
                'models/cat_dog_custom_cnn_v2.pth'
            )

            print("!Сохранена лучшая модель!")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping.")
            break

    print(
        f"\nГотово. Лучшая Val Accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )


if __name__ == '__main__':
    train_model()
