Cat vs Dog --- CNN v2 (from scratch)

Цель v2 --- улучшить собственную CNN из v1 и одновременно сделать
архитектуру удобнее для дальнейших экспериментов.

Данные

Проект рассчитан на датасет примерно из 10 000 изображений кошек и
собак.

download_data.py:

скачивает архив;

распаковывает изображения;

определяет классы по именам файлов;

делает train/validation split;

использует val_ratio=0.2;

использует seed 42.

Важно: отдельный test split пока автоматически не создается. Поэтому
validation используется для контроля качества во время разработки.

Архитектура v2

Input: 3 × 224 × 224
        ↓
ConvBlock 3 → 32
        ↓
112 × 112 × 32
        ↓
ConvBlock 32 → 64
        ↓
56 × 56 × 64
        ↓
ConvBlock 64 → 128
        ↓
28 × 28 × 128
        ↓
ConvBlock 128 → 256
        ↓
14 × 14 × 256
        ↓
Adaptive Average Pooling
        ↓
256
        ↓
Linear 256 → 128
        ↓
ReLU
        ↓
Dropout(0.5)
        ↓
Linear 128 → 2

ConvBlock

Каждый блок содержит две convolution:

Conv
 ↓
BatchNorm
 ↓
ReLU
 ↓
Conv
 ↓
BatchNorm
 ↓
ReLU
 ↓
MaxPool

В v1 была только одна convolution на блок.

Большее количество convolution позволяет сети строить более сложное
представление признаков.

Условно:

пиксели
  ↓
края и текстуры
  ↓
простые формы
  ↓
части объекта
  ↓
сложные признаки
  ↓
Cat / Dog

Почему 224×224

В v1 использовалось 128×128, в v2 --- 224×224.

Большее разрешение сохраняет больше деталей исходной фотографии:

форму ушей;

глаза;

детали морды;

текстуру шерсти;

контуры головы.

Это экспериментальная гипотеза, а не гарантированное улучшение.
Имеет смысл отдельно сравнить 128, 224 и 256.

Adaptive Average Pooling

В v1 использовалось:

Flatten
→
Linear(128 * 16 * 16, 256)

Из-за этого classifier был жестко связан с размером входного
изображения.

В v2 используется:

AdaptiveAvgPool2d((1, 1))

Например:

256 × 14 × 14
       ↓
256 × 1 × 1
       ↓
256 признаков

После этого:

256 → 128 → 2

Преимущества:

меньше параметров в classifier;

нет жесткой привязки к 128×128;

проще экспериментировать с разрешением.

Инициализация

Поскольку сеть обучается с нуля, convolutional и linear layers получают
Kaiming initialization.

Она подходит для сети с ReLU и обеспечивает более подходящий старт для
обучения.

BatchNorm инициализируется стандартно:

weight = 1
bias = 0

Аугментация

Training:

Resize 224×224
RandomHorizontalFlip
RandomRotation(10°)
RandomResizedCrop
ColorJitter
ToTensor
Normalize

Validation:

Resize 224×224
ToTensor
Normalize

RandomResizedCrop помогает уменьшить зависимость модели от точного
положения и масштаба объекта.

Обучение

v2 использует:

Loss:
CrossEntropyLoss

Optimizer:
AdamW

Learning rate:
0.001

Weight decay:
0.0001

Batch size:
32

Maximum epochs:
50

Learning-rate scheduler

Используется ReduceLROnPlateau.

Если validation accuracy перестает улучшаться, learning rate уменьшается
в 3.33 раза:

0.001
 ↓
0.0003
 ↓
0.00009
...

Это позволяет продолжить оптимизацию более маленькими шагами.

Early stopping

PATIENCE = 8.

Если validation accuracy не улучшается 8 эпох подряд, обучение
прекращается.

Best checkpoint

Вместо сохранения только последней эпохи v2 сохраняет модель всякий раз,
когда достигается новая лучшая validation accuracy.

В checkpoint также сохраняются:

model_state_dict;

class_to_idx;

img_size;

val_acc.

Сравнение v1 и v2

Характеристика           v1                 v2

Conv-блоков              3                  4
Conv на блок             1                  2
Всего Conv               3                  8
Каналы                   3→32→64→128        3→32→64→128→256
Input                    128×128            224×224
BatchNorm                Да                 Да
Activation               ReLU               ReLU
Dropout                  0.5                0.5
Global Average Pooling   Нет                Да
Kaiming initialization   Нет явно           Да
Optimizer                Adam               AdamW
Weight decay             Нет                0.0001
Scheduler                Нет                ReduceLROnPlateau
Early stopping           Нет                Да
Максимум эпох            60                 50
Сохраняется              Последняя модель   Лучшая модель
Отдельный test split     Нет                Нет

Что именно пытается улучшить v2

1. Более глубокое извлечение признаков

v1:

3 Conv

v2:

8 Conv

v2 может выполнять больше последовательных преобразований признаков.

2. Более высокая емкость

v1 заканчивается на 128 каналах.

v2 --- на 256.

Это дает модели больше возможностей хранить разные высокоуровневые
признаки.

3. Больше информации на входе

224×224 вместо 128×128 потенциально позволяет сохранить больше
деталей.

4. Более компактный classifier

Global Average Pooling заменяет огромный Flatten → Linear.

5. Более контролируемое обучение

v2 автоматически уменьшает learning rate при плато и может остановиться
раньше.

6. Лучший checkpoint

v2 сохраняет лучшую модель по validation accuracy, а не просто
последнюю.

Результаты эксперимента

Текущий запуск v2 показал, например, на 41-й и 42-й эпохах:

Epoch 41
Train Loss: 0.1169
Train Acc:  95.80%
Val Loss:   0.1798
Val Acc:    93.63%
LR:         0.000300

Epoch 42
Train Loss: 0.1130
Train Acc:  95.90%
Val Loss:   0.1696
Val Acc:    93.43%
LR:         0.000300

По этим двум эпохам явного сильного переобучения не видно: разрыв между
train и validation accuracy составляет примерно 2--2.5 процентного
пункта, а validation loss между этими эпохами даже снизился.

Для окончательной оценки нужно смотреть всю историю обучения и,
желательно, отдельный test set.

Как корректно сравнить v1 и v2

Чтобы сравнение было честным, желательно использовать:

один и тот же датасет;

один и тот же train/validation split;

одинаковый test set;

одинаковый критерий оценки.

Сравнивать стоит как минимум:

Train Accuracy
Validation Accuracy
Test Accuracy
F1
Precision
Recall
Количество параметров
Время обучения

Запуск

pip install torch torchvision opencv-python matplotlib requests tqdm pillow

python download_data.py
python preprocess.py
python train_cnn.py

Предсказание:

python predict.py path/to/image.jpg

