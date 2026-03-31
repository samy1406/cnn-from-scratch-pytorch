# CNN from Scratch — PyTorch + NumPy

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

> Building a Convolutional Neural Network from scratch — no `nn.Conv2d`, no `nn.Linear`. Every forward pass, every gradient, every weight update written by hand. Trained on CIFAR-10.

---

## Problem Statement

Most engineers use `torch.nn` as a black box. This project breaks open that black box — implementing convolutions, pooling, activations, and backpropagation manually to deeply understand what happens inside every CNN call.

**Why this matters:** At Qualcomm, NVIDIA, and similar companies, you will be asked to derive or explain these operations on a whiteboard. This project is the answer to that.

---

## Architecture

```
Input (3 × 32 × 32)
        ↓
Conv Layer 1  (3 → 32 filters, 3×3)   ← implemented from scratch
        ↓
ReLU                                   ← implemented from scratch
        ↓
MaxPool (2×2)                          ← implemented from scratch
        ↓
Conv Layer 2  (32 → 64 filters, 3×3)  ← implemented from scratch
        ↓
ReLU
        ↓
MaxPool (2×2)
        ↓
Flatten
        ↓
Fully Connected (64×8×8 → 256)        ← implemented from scratch
        ↓
ReLU
        ↓
Fully Connected (256 → 10)
        ↓
Softmax + CrossEntropy Loss
```

---

## What "From Scratch" Means Here

| Component | What's allowed | What's NOT allowed |
|-----------|---------------|-------------------|
| Conv Layer | NumPy for the math, PyTorch tensors for data | `nn.Conv2d` |
| Backprop | Manual gradient computation | `autograd` for layer internals |
| Weight update | Manual SGD loop | `torch.optim` |
| Data loading | `torchvision.datasets`, `DataLoader` | — |
| Loss | Manual CrossEntropy | `nn.CrossEntropyLoss` |

---

## Results


| Metric | Value |
|--------|-------|
| Dataset | CIFAR-10 (10 classes) |
| Final Test Accuracy | TBD |
| Training Time / Epoch | TBD |
| Hardware | Google Colab T4 GPU |

*Results will be updated as training progresses.*

---

## Demo

*Demo GIF will be added after training is complete.*

---

## Project Structure

```
cnn-from-scratch-pytorch/
├── README.md
├── LEARNING.md              ← personal notes on every concept
├── requirements.txt
├── notebooks/
│   └── cnn_cifar10.ipynb    ← main Colab notebook
├── src/
│   ├── dataloader.py        ← CIFAR-10 loading and preprocessing
│   ├── model.py             ← CNN layers implemented from scratch
│   └── train.py             ← training loop and evaluation
├── demo/
│   └── (demo GIF here)
└── data/
    └── (auto-downloaded by torchvision)
```

---

## Installation & Running

```bash
git clone https://github.com/samy1406/cnn-from-scratch-pytorch.git
cd cnn-from-scratch-pytorch
pip install -r requirements.txt
```

**To run in Google Colab:**

Open `notebooks/cnn_cifar10.ipynb` directly in Colab — GPU runtime recommended (Runtime → Change runtime type → T4 GPU).

---

## Key Concepts Implemented

- **Convolution (cross-correlation):** filter sliding over input with stride and padding
- **Backprop through Conv:** gradient w.r.t. weights = input ⊛ upstream gradient; gradient w.r.t. input = upstream gradient ⊛ flipped filter
- **MaxPool backward:** gradient flows only through the max-position (argmax mask)
- **ReLU backward:** gate — gradient passes if forward activation > 0, else blocked
- **Softmax + CrossEntropy:** combined for numerical stability

---

## What I Learned

*This section will be expanded throughout the project. See [LEARNING.md](LEARNING.md) for detailed notes.*

- Why backprop through a conv layer requires a transposed convolution
- How gradient accumulation works when a filter weight touches multiple input patches
- Why combining Softmax and CrossEntropy improves numerical stability
- The real cost of naive conv vs optimized im2col approaches

---

## Related Projects

- [LearningNeuralNet](https://github.com/samy1406/LearningNeuralNet) — micrograd autograd engine (prerequisite to this project)

---

## Author

**Samy** — AI/ML Engineer | Computer Vision + Edge AI | BITS Pilani M.Tech AI/ML

[GitHub](https://github.com/samy1406) • [LinkedIn](#) • [YouTube](#)
