import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Проверяем доступность видеокарты (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство для вычислений: {device}")


# 2. ПРОДВИНУТАЯ СВЕРТОЧНАЯ НЕЙРОСЕТЬ (С GAP и PReLU)
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        
        # Блок 1: Вход (3, 224, 224) -> (32, 112, 112)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Блок 2: (32, 112, 112) -> (64, 56, 56)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Блок 3: (64, 56, 56) -> (128, 28, 28)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        # Блок 4: (128, 28, 28) -> (256, 14, 14)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.PReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global Average Pooling: сжимает (256, 14, 14) до (256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Полносвязный классификатор
        self.fc1 = nn.Linear(256, 128)
        self.act5 = nn.PReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 2)  # 2 класса: 0 = cat, 1 = dog

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = self.pool4(self.act4(self.bn4(self.conv4(x))))
        
        # Применяем Global Average Pooling и убираем лишние измерения
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.act5(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 3. Подготовка данных с высоким разрешением (224x224) и аугментацией
IMG_SIZE = (224, 224)

train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder('dataset/train', transform=train_transforms)
val_dataset = datasets.ImageFolder('dataset/val', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Инициализация модели, оптимизатора AdamW и шедулера
model = CatDogCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Косинусный планировщик скорости обучения
EPOCHS = 60
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# 5. Цикл обучения
def train_model():
    for epoch in range(EPOCHS):
        print(f"\n--- Эпоха {epoch + 1}/{EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")
        
        # Обучение
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train.double() / total_train
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}%")

        # Валидация
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        val_acc = correct_val.double() / total_val
        print(f"Val Acc: {val_acc * 100:.2f}%")

        # Шаг планировщика LR
        scheduler.step()

    # Сохранение весов
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/cat_dog_custom_cnn.pth')
    print("\nПродвинутая модель успешно сохранена в `models/cat_dog_custom_cnn_v2.pth`!")


if __name__ == '__main__':
    train_model()
