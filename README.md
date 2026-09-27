## Архитектура v2

```text
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
```

## ConvBlock

Каждый блок содержит две convolution:

```text
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
```

В v1 была только одна convolution на блок.

Большее количество convolution позволяет сети строить более сложное
представление признаков.

Условно:

```text
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
```
