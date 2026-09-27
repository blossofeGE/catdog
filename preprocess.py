import os
import cv2
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def check_and_clean_images(data_dir='dataset'):
    """Проверяет изображения и удаляет нечитаемые файлы."""
    corrupted_count = 0
    total_count = 0

    print("Начинаем проверку изображений...")

    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            total_count += 1
            file_path = os.path.join(root, file)
            image = cv2.imread(file_path)

            if image is None or image.size == 0:
                print(f"Удален поврежденный файл: {file_path}")
                os.remove(file_path)
                corrupted_count += 1

    print(
        f"Проверка завершена. Проверено: {total_count}, "
        f"удалено: {corrupted_count}"
    )


def visualize_samples(dataset_dir='dataset/train'):
    """Показывает примеры изображений из train."""
    cat_dir = os.path.join(dataset_dir, 'cats')
    dog_dir = os.path.join(dataset_dir, 'dogs')

    cat_files = [
        os.path.join(cat_dir, f)
        for f in os.listdir(cat_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ][:3]

    dog_files = [
        os.path.join(dog_dir, f)
        for f in os.listdir(dog_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ][:3]

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    for ax, path in zip(axes[0], cat_files):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title('Cat')
        ax.axis('off')

    for ax, path in zip(axes[1], dog_files):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title('Dog')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    check_and_clean_images()
    visualize_samples()
