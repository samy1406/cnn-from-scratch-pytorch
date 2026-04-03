# CNN from Scratch — PyTorch + NumPy

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Building a Convolutional Neural Network completely from scratch — no `nn.Conv2d`, no `nn.Linear`, no autograd. Every forward pass, every gradient, every weight update written by hand using NumPy. Trained on CIFAR-10.

---

## Problem Statement

Most ML engineers use `torch.nn` as a black box. This project breaks open that black box — implementing convolutions, pooling, activations, and backpropagation manually to deeply understand what happens inside every CNN call.

**Why this matters:** At Qualcomm, NVIDIA, and similar companies, you will be asked to derive or explain these operations on a whiteboard. This project is the answer to that.

---

## Architecture

```
Input (3 × 32 × 32)
        │
        ▼
┌───────────────────┐
│   Conv Layer 1    │  3 → 32 filters, 3×3, padding=1
│   (from scratch)  │  Output: (32 × 32 × 32)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│      ReLU         │  Non-linear activation
│   (from scratch)  │  Output: (32 × 32 × 32)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    MaxPool 2×2    │  Stride=2, spatial halving
│   (from scratch)  │  Output: (32 × 16 × 16)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│     Flatten       │  32 × 16 × 16 = 8192
│   (from scratch)  │  Output: (8192,)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   FC Layer 1      │  8192 → 128
│   (from scratch)  │  Output: (128,)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│      ReLU         │  Output: (128,)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   FC Layer 2      │  128 → 10
│   (from scratch)  │  Output: (10,)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Softmax +         │  Numerically stable
│ CrossEntropy Loss │  Combined for stability
└───────────────────┘
```

---

## What "From Scratch" Means Here

| Component | Implemented | NOT used |
|-----------|-------------|----------|
| Conv Layer forward | Manual sliding window + tensordot | `nn.Conv2d` |
| Conv Layer backward | Transposed convolution gradients | `autograd` |
| ReLU forward/backward | Element-wise max + gradient gate | `nn.ReLU` |
| MaxPool forward/backward | Max with argmax mask | `nn.MaxPool2d` |
| FC Layer forward/backward | Matrix multiply + manual gradients | `nn.Linear` |
| Weight initialization | He initialization | default init |
| Loss function | Stable softmax + NLL | `nn.CrossEntropyLoss` |
| Optimizer | Manual SGD loop | `torch.optim` |
| Data loading | `torchvision` (allowed) | — |

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | CIFAR-10 (10 classes) |
| Initial Loss (random) | ~2.302 (theoretical: -log(0.1)) |
| Loss after 1 epoch | 1.6419 |
| Hardware | CPU (NumPy) |
| Parameters | ~1.1M |

**Loss sanity check:** Random initialization gives loss ≈ 2.302 (`-log(1/10)`). Achieving 1.64 after 1 epoch confirms the model is learning correctly.

---

## Key Concepts Implemented

### 1. Convolution (Cross-Correlation)
Filter slides over input with stride and padding. Each position produces one scalar via element-wise multiply + sum:
```
output[i][j] = sum(input[i:i+k, j:j+k] * filter) + bias
```

### 2. Backprop Through Conv — Two Gradients
```
# Gradient w.r.t. weights (accumulate across all positions)
grad_filter[m][n] = sum over (i,j): input[i+m][j+n] × upstream_grad[i][j]

# Gradient w.r.t. input (transposed convolution)
grad_input[i][j] = sum over (m,n): filter[m][n] × upstream_grad[i-m][j-n]
```

### 3. MaxPool Backward — The Mask Trick
During forward pass, save which position was the maximum (the "mask"). During backward, only route gradient to that position. All other positions get zero.

### 4. Softmax + CrossEntropy — Numerically Stable
Combined gradient simplifies beautifully:
```
grad[correct_class] = softmax(logit) - 1
grad[other_classes] = softmax(logit)
```

### 5. He Initialization
```
weights ~ N(0, sqrt(2 / fan_in))
```
Prevents vanishing/exploding gradients in ReLU networks.

---

## Project Structure

```
cnn-from-scratch-pytorch/
├── README.md
├── LEARNING.md              ← personal concept notes
├── requirements.txt
├── notebooks/
│   └── cnn_cifar10.ipynb    ← Colab notebook
├── src/
│   ├── dataloader.py        ← CIFAR-10 loading and preprocessing
│   ├── model.py             ← All layers from scratch
│   ├── loss.py              ← CrossEntropy with stable softmax
│   └── train.py             ← Training loop with SGD
├── demo/
│   └── (loss curve GIF)
└── data/
    └── (auto-downloaded CIFAR-10)
```

---

## Installation & Running

```bash
git clone https://github.com/samy1406/cnn-from-scratch-pytorch.git
cd cnn-from-scratch-pytorch
pip install -r requirements.txt
```

**Run training:**
```python
from src.train import train
train(epochs=10, lr=0.01, batch_size=64)
```

**Run on Google Colab (recommended — GPU):**

Open `notebooks/cnn_cifar10.ipynb` in Colab → Runtime → Change runtime type → T4 GPU.

---

## Interview Questions This Project Answers

- **"Implement a conv layer forward pass"** → `src/model.py ConvLayer.forward()`
- **"What happens during backprop through a conv layer?"** → Two gradients: transposed conv for input, correlation for weights
- **"Why do we need non-linear activations?"** → Stacking linear layers collapses to one linear transform
- **"Why does MaxPool need to save positions during forward?"** → Backward needs to route gradient only to the max position
- **"Why combine Softmax and CrossEntropy?"** → Numerical stability + gradient simplification
- **"What is He initialization and why?"** → Compensates for ReLU killing half the neurons on average

---

## Related Projects

- [LearningNeuralNet](https://github.com/samy1406/LearningNeuralNet) — micrograd autograd engine (scalar autograd, prerequisite to this)

---

## Author

**Samy** — AI/ML Engineer | Computer Vision + Edge AI | BITS Pilani M.Tech AI/ML

[GitHub](https://github.com/samy1406) • [LinkedIn](#) • [YouTube](#)