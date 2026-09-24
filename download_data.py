import os
import random
import shutil
import urllib.parse
import zipfile
import requests
from tqdm import tqdm  # Библиотека для отрисовки шкал прогресса

PUBLIC_LINK = 'https://disk.yandex.ru/d/VzV5KXL8Y7Vvgg'


def download_archive(public_key, dest_archive='dataset/images.zip'):
    """Скачивание архива с Яндекс Диска с отображением прогресс-бара"""
    print("Получаем ссылку на скачивание с Яндекс Диска...")
    api_url = f'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={urllib.parse.quote(public_key)}'

    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Ошибка API Яндекса: {response.status_code} - {response.text}")

    download_url = response.json()['href']
    print("Ссылка получена. Скачиваем архив...")

    os.makedirs(os.path.dirname(dest_archive), exist_ok=True)

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()

        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 8192

        # Шкала прогресса скачивания файла
        with open(dest_archive, 'wb') as f, tqdm(
            desc="Загрузка архива",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                bar.update(size)

    print(f"Архив успешно сохранен в: {dest_archive}\n")


def extract_and_split_dataset(zip_path='dataset/images.zip', dest_dir='dataset', val_ratio=0.2):
    """Распаковка и автоматическая сортировка файлов на train/val с шкалой прогресса"""
    temp_dir = os.path.join(dest_dir, 'temp_unpacked')

    # Шкала прогресса распаковки архива
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Распаковка архива", unit="файлов"):
            zip_ref.extract(member, temp_dir)

    # Создаем итоговую структуру папок
    dirs = {
        'train_cats': os.path.join(dest_dir, 'train', 'cats'),
        'train_dogs': os.path.join(dest_dir, 'train', 'dogs'),
        'val_cats': os.path.join(dest_dir, 'val', 'cats'),
        'val_dogs': os.path.join(dest_dir, 'val', 'dogs'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Ищем все файлы картинок во временной папке
    all_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, file))

    cats = [f for f in all_files if os.path.basename(f).lower().startswith('cat')]
    dogs = [f for f in all_files if os.path.basename(f).lower().startswith('dog')]

    print(f"\nНайдено изображений -> Кошки: {len(cats)}, Собаки: {len(dogs)}")

    def move_files(file_list, train_target, val_target, animal_name="файлов"):
        random.seed(42)
        random.shuffle(file_list)

        val_size = int(len(file_list) * val_ratio)
        val_files = file_list[:val_size]
        train_files = file_list[val_size:]

        # Шкала прогресса копирования
        total_files = train_files + val_files
        for f in tqdm(total_files, desc=f"Сортировка {animal_name}", unit="картинок"):
            target_folder = val_target if f in val_files else train_target
            shutil.copy(f, os.path.join(target_folder, os.path.basename(f)))

    move_files(cats, dirs['train_cats'], dirs['val_cats'], animal_name="кошек")
    move_files(dogs, dirs['train_dogs'], dirs['val_dogs'], animal_name="собак")

    print("\nОчищаем временные файлы...")
    shutil.rmtree(temp_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("ГОТОВО! Данные разложены по папкам `dataset/train/` и `dataset/val/`.")


if __name__ == '__main__':
    download_archive(PUBLIC_LINK)
    extract_and_split_dataset()