import os
# Обход ошибки дублирования OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство для вычислений: {device}")

# 2. ОПРЕДЕЛЯЕМ СОБСТВЕННУЮ СВЕРТОЧНУЮ НЕЙРОСЕТЬ (CNN)
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        
        # Сверточный блок 1: Вход (3, 128, 128) -> Выход (32, 64, 64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Сверточный блок 2: Вход (32, 64, 64) -> Выход (64, 32, 32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Сверточный блок 3: Вход (64, 32, 32) -> Выход (128, 16, 16)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Полносвязные слои (Классификатор)
        # Размерность входного вектора: 128 каналов * 16 * 16 px = 32768
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)  # 2 класса: 0 = cat, 1 = dog

    def forward(self, x):
        # Проход через сверточные блоки
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Преобразование 3D тензора в 1D вектор (Flatten)
        x = x.view(x.size(0), -1)
        
        # Проход через полносвязные слои
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 3. Аугментация и подготовка данных (размер 128x128)
IMG_SIZE = (128, 128)

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

# 4. Инициализация модели
model = CatDogCNN().to(device)

# 5. Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 6. Цикл обучения
def train_model(epochs=5):
    for epoch in range(epochs):
        print(f"\n--- Эпоха {epoch + 1}/{epochs} ---")
        
        # Режим обучения
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

        # Режим валидации
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

    # Сохраняем веса обученной модели
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/cat_dog_custom_cnn.pth')
    print("\nСобственная модель успешно сохранена в `models/cat_dog_custom_cnn.pth`!")


if __name__ == '__main__':
    train_model(epochs=60)