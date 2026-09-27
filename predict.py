import os
import sys
import torch
from torchvision import transforms
from PIL import Image

from train_cnn import CatDogCNN, IMG_SIZE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(
    image_path,
    model_path='models/cat_dog_custom_cnn_v2.pth'
):
    """Предсказывает класс для одной картинки."""

    if not os.path.exists(image_path):
        print(f"Ошибка: файл '{image_path}' не найден.")
        return

    if not os.path.exists(model_path):
        print(
            f"Ошибка: модель '{model_path}' не найдена. "
            f"Сначала запустите train_cnn.py."
        )
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    model = CatDogCNN().to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    class_to_idx = checkpoint.get(
        'class_to_idx',
        {'cats': 0, 'dogs': 1}
    )

    idx_to_class = {
        index: name
        for name, index in class_to_idx.items()
    }

    confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = idx_to_class[predicted.item()]

    print("=" * 45)
    print(f"Файл:        {image_path}")
    print(f"Результат:   {predicted_class}")
    print(f"Уверенность: {confidence.item() * 100:.2f}%")

    if 'cats' in class_to_idx and 'dogs' in class_to_idx:
        cat_prob = probabilities[0][class_to_idx['cats']].item()
        dog_prob = probabilities[0][class_to_idx['dogs']].item()

        print(f"Cat:         {cat_prob * 100:.2f}%")
        print(f"Dog:         {dog_prob * 100:.2f}%")

    print("=" * 45)


if __name__ == '__main__':
    path = "dataset/test/7.jpg"

    predict_image(path)
