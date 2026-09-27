import os
import random
import shutil
import urllib.parse
import zipfile
import requests
from tqdm import tqdm

PUBLIC_LINK = 'https://disk.yandex.ru/d/VzV5KXL8Y7Vvgg'


def download_archive(public_key, dest_archive='dataset/images.zip'):
    """Скачивание архива с Яндекс Диска."""
    print("Получаем ссылку на скачивание с Яндекс Диска...")
    api_url = (
        'https://cloud-api.yandex.net/v1/disk/public/resources/download'
        f'?public_key={urllib.parse.quote(public_key)}'
    )

    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    download_url = response.json()['href']
    os.makedirs(os.path.dirname(dest_archive), exist_ok=True)

    print("Ссылка получена. Скачиваем архив...")
    with requests.get(download_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))

        with open(dest_archive, 'wb') as f, tqdm(
            desc="Загрузка архива",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print(f"Архив сохранен в: {dest_archive}")


def extract_and_split_dataset(
    zip_path='dataset/images.zip',
    dest_dir='dataset',
    val_ratio=0.2,
    seed=42
):
    """Распаковка и воспроизводимый train/validation split."""
    temp_dir = os.path.join(dest_dir, 'temp_unpacked')

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Распаковка архива", unit="файлов"):
            zip_ref.extract(member, temp_dir)

    dirs = {
        'train_cats': os.path.join(dest_dir, 'train', 'cats'),
        'train_dogs': os.path.join(dest_dir, 'train', 'dogs'),
        'val_cats': os.path.join(dest_dir, 'val', 'cats'),
        'val_dogs': os.path.join(dest_dir, 'val', 'dogs'),
    }

    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)

    all_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, file))

    cats = [f for f in all_files if os.path.basename(f).lower().startswith('cat')]
    dogs = [f for f in all_files if os.path.basename(f).lower().startswith('dog')]

    print(f"Найдено: кошек={len(cats)}, собак={len(dogs)}")

    rng = random.Random(seed)

    def split_files(files):
        files = files.copy()
        rng.shuffle(files)
        val_size = int(len(files) * val_ratio)
        return files[val_size:], files[:val_size]

    def copy_files(files, train_dir, val_dir, name):
        train_files, val_files = split_files(files)

        for file_path in tqdm(train_files, desc=f"Train {name}", unit="картинок"):
            shutil.copy2(file_path, os.path.join(train_dir, os.path.basename(file_path)))

        for file_path in tqdm(val_files, desc=f"Val {name}", unit="картинок"):
            shutil.copy2(file_path, os.path.join(val_dir, os.path.basename(file_path)))

    copy_files(cats, dirs['train_cats'], dirs['val_cats'], 'кошки')
    copy_files(dogs, dirs['train_dogs'], dirs['val_dogs'], 'собаки')

    shutil.rmtree(temp_dir)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("ГОТОВО: dataset/train и dataset/val созданы.")


if __name__ == '__main__':
    download_archive(PUBLIC_LINK)
    extract_and_split_dataset()
