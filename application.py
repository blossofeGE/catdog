import os
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from pyngrok import ngrok
import hashlib

from train_cnn import CatDogCNN, IMG_SIZE

EASTER_EGGS_BY_HASH = {
    "1280e2d617708f1aff1c85c83908ccf3": "Гений, миллиардер, плейбой, филантроп",
    "7072fbabfb1fa52b3b5c1c5117446bdc": "Котопёс"
}

# Настройка устройства и путей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/cat_dog_custom_cnn_v2.pth'

# Трансформации изображения
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])

# Глобальная загрузка модели при запуске
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Файл модели '{MODEL_PATH}' не найден! Проверьте путь."
        )

    model = CatDogCNN().to(device)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_to_idx = checkpoint.get('class_to_idx', {'cats': 0, 'dogs': 1})
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    
    return model, idx_to_class

print("Загрузка модели...")
model, idx_to_class = load_model()
print("Модель успешно загружена!")

# Функция предсказания для Gradio
def predict(image: Image.Image):
    if image is None:
        return {}

    filename = os.path.basename(getattr(image, 'filename', '')).lower()
    img_bytes = image.tobytes()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    if img_hash in EASTER_EGGS_BY_HASH:
        return {EASTER_EGGS_BY_HASH[img_hash]: 1.0}

    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    results = {}
    for idx, prob in enumerate(probabilities):
        raw_label = idx_to_class.get(idx, f"Class {idx}")
        if raw_label == 'cats':
            label_display = "Кошка 🐱"
        elif raw_label == 'dogs':
            label_display = "Собака 🐶"
        else:
            label_display = raw_label

        results[label_display] = float(prob)

    return results

# Интерфейс Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите картинку"),
    outputs=gr.Label(label="Результат", num_top_classes=2),
    title="Классификатор Кошек и Собак 🐱🐶",
    description="Загрузите изображение, и модель определит, кто на нем изображен.",
)

if __name__ == "__main__":
    ngrok.set_auth_token("") # Необходимо вставить свой токен с сайта ngrok

    public_url = ngrok.connect(7861)

    print("\n" + "=" * 50)
    print(f"🚀 Публичная ссылка: {public_url}")
    print("=" * 50 + "\n")

    demo.launch(server_port=7861, share=False)