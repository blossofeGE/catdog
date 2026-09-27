Cat vs Dog --- CNN v1

Структура

project/
├── download_data.py
├── preprocess.py
├── train_cnn.py
├── predict.py
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── val/
│       ├── cats/
│       └── dogs/
└── models/
    └── cat_dog_custom_cnn.pth

Данные

download_data.py:

скачивает архив с Яндекс Диска;

распаковывает изображения;

определяет класс по имени файла (cat... / dog...);

делит каждый класс на train и validation;

использует val_ratio=0.2;

использует seed 42.

Предобработка

preprocess.py проверяет изображения через OpenCV и удаляет нечитаемые
файлы.

Для визуализации используется размер 128×128.

Архитектура

Input: 3 × 128 × 128

Conv 3 → 32
BatchNorm
ReLU
MaxPool
        ↓
32 × 64 × 64

Conv 32 → 64
BatchNorm
ReLU
MaxPool
        ↓
64 × 32 × 32

Conv 64 → 128
BatchNorm
ReLU
MaxPool
        ↓
128 × 16 × 16

Flatten
        ↓
32768

Linear 32768 → 256
ReLU
Dropout(0.5)
Linear 256 → 2

Модель полностью обучается с нуля.

Аугментация

Training:

Resize до 128×128;

RandomHorizontalFlip;

RandomRotation(15°);

ColorJitter для brightness и contrast;

ToTensor;

Normalize с mean/std [0.5, 0.5, 0.5].

Validation использует только Resize, ToTensor и Normalize.

Обучение

Loss: CrossEntropyLoss

Optimizer: Adam

Learning rate: 0.001

Batch size: 32

Максимум: 60 эпох

На каждой эпохе выводятся Train Loss, Train Accuracy и Validation
Accuracy.

После обучения веса сохраняются в:

models/cat_dog_custom_cnn.pth

В v1 сохраняется модель последней эпохи, а не обязательно лучшая модель
по validation accuracy.

Предсказание

predict.py:

открывает изображение;

переводит его в RGB;

изменяет размер до 128×128;

нормализует;

загружает веса;

получает logits;

применяет Softmax;

выводит класс и confidence.

Запуск

pip install torch torchvision opencv-python matplotlib requests tqdm pillow

python download_data.py
python preprocess.py
python train_cnn.py
python predict.py path/to/image.jpg


Эта версия используется как baseline для сравнения с более новой CNN.
