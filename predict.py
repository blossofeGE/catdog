import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# Импортируем архитектуру моей модели из train_cnn.py
from train_cnn import CatDogCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(image_path, model_path='models/cat_dog_custom_cnn.pth'):
    """
    Функция предсказания класса для произвольного изображения
    """
    if not os.path.exists(image_path):
        print(f"Ошибка: файл '{image_path}' не найден!")
        return

    if not os.path.exists(model_path):
        print(f"Ошибка: файл весов '{model_path}' не найден. Сначала обучите модель!")
        return

    # 1. Предобработка входящего изображения (такая же, как при валидации)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device) # Добавляем размерность батча (Batch Size = 1)

    # 2. Инициализация модели и загрузка весов
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Переводим модель в режим инференса

    # 3. Инференс (предсказание)
    with torch.no_grad():
        outputs = model(input_tensor)
        # Применяем Softmax для получения вероятностей
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_names = ['Cat', 'Dog']
    pred_class = class_names[predicted.item()]
    conf_score = confidence.item() * 100

    print("=" * 40)
    print(f" Файл: {image_path}")
    print(f" Результат: {pred_class}")
    print(f" Уверенность модели: {conf_score:.2f}%")
    print("=" * 40)

if __name__ == '__main__':
    # Если путь передан через аргументы командной строки, используем его, 
    # иначе берем тестовую картинку по умолчанию
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Пример пути к случайной картинке из валидационной выборки
        img_path = 'dataset/test/4.JPG' 
    
    predict_image(img_path)
