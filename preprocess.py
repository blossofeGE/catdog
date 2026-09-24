import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_and_clean_images(data_dir='dataset'):
    """
    Проходит по всей папке dataset и удаляет битые/нечитаемые картинки.
    """
    print("Начинаем валидацию изображений...")
    corrupted_count = 0
    total_count = 0

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_count += 1
                file_path = os.path.join(root, file)

                # Пробуем прочитать файл через OpenCV
                img = cv2.imread(file_path)

                # Если файл пустой или поврежден
                if img is None or img.size == 0:
                    print(f"Удален поврежденный файл: {file_path}")
                    os.remove(file_path)
                    corrupted_count += 1

    print(f"Проверка завершена. Проверено: {total_count}, удалено битых: {corrupted_count}")


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Полный цикл предобработки одного изображения:
    1. Чтение с диска (OpenCV)
    2. Конвертация BGR -> RGB
    3. Изменение размера (Resize)
    4. Нормализация пикселей [0, 255] -> [0.0, 1.0] (NumPy)
    """
    # 1. Чтение (по умолчанию OpenCV читает в формате BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Не удалось прочитать файл: {image_path}")

    # 2. Перевод из формата BGR в стандартный RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Изменение размера до target_size
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    # 4. Преобразование в float32 и нормализация [0.0, 1.0]
    img_normalized = img_resized.astype(np.float32) / 255.0

    return img_normalized


def visualize_sample(dataset_dir='dataset/train'):
    """
    Берет по одному случайному файлу кошки и собаки, обработает и покажет на экране.
    """
    cat_dir = os.path.join(dataset_dir, 'cats')
    dog_dir = os.path.join(dataset_dir, 'dogs')

    cat_file = os.path.join(cat_dir, os.listdir(cat_dir)[0])
    dog_file = os.path.join(dog_dir, os.listdir(dog_dir)[0])

    cat_img = load_and_preprocess_image(cat_file)
    dog_img = load_and_preprocess_image(dog_file)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(cat_img)
    axes[0].set_title(f"Cat (Shape: {cat_img.shape})")
    axes[0].axis('off')

    axes[1].imshow(dog_img)
    axes[1].set_title(f"Dog (Shape: {dog_img.shape})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. Очищаем датасет от невалидных файлов
    check_and_clean_images()
    
    # 2. Визуализируем пример после предобработки
    visualize_sample()