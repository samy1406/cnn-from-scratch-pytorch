# LEARNING.md — CNN from Scratch

> Personal notes written while building this project. These are the concepts I had to deeply understand to implement each component. Written so I can explain every one of these in an interview without hesitation.

---

## 1. Backpropagation — The Chain Rule in a CNN

**The core question backprop answers:**
> "How much does each weight contribute to the final loss?"

For a weight deep in the network (e.g., Conv Layer 1), it doesn't directly touch the loss. There are many functions stacked between them. The chain rule lets us walk backwards through each one:

```
∂Loss/∂w1 = ∂Loss/∂out_fc × ∂out_fc/∂out_relu2 × ∂out_relu2/∂out_conv2 × ... × ∂out_relu1/∂w1
```

Each layer computes two things during backprop:
1. Its own gradient → to update its weights
2. The gradient to pass backwards → for the layer before it

This is why every layer needs both a `forward()` and a `backward()` method.

---

## 2. What a Gradient Actually Means

For a 3×3 conv filter with gradient shape `(3, 3)`:

> `grad[i][j]` = "If I increase filter weight at position `[i][j]` by a tiny amount ε, the loss changes by `grad[i][j] × ε`"

This is why the optimizer update is:
```
filter[i][j] = filter[i][j] - learning_rate × grad[i][j]
```

If `grad[i][j]` is large and positive → weight is pushing loss up → decrease it.
If `grad[i][j]` is large and negative → weight is pushing loss down → increase it.

---

## 3. Convolution vs Cross-Correlation

What CNNs actually compute in the forward pass is **cross-correlation**, not true convolution (which would require flipping the filter). In practice, the filter learns the flipped version anyway, so the distinction doesn't affect results — but it matters for understanding the backward pass.

**Forward pass (cross-correlation):**
```
output[i][j] = sum over (m, n) of: input[i+m][j+n] × filter[m][n]
```

---

## 4. Backprop Through a Conv Layer

A filter weight `filter[m][n]` touches every patch it slides over during the forward pass. During backprop, we must accumulate the gradient from every one of those patches.

**Gradient w.r.t. filter weights:**
```
∂Loss/∂filter[m][n] = sum over all positions (i,j) of: input[i+m][j+n] × upstream_grad[i][j]
```
This is a **convolution/cross-correlation** between the input and the upstream gradient.

**Gradient w.r.t. input (to pass to the previous layer):**
```
∂Loss/∂input[i][j] = sum over all (m,n) of: filter[m][n] × upstream_grad[i-m][j-n]
```
This is a **transposed convolution** — the filter is effectively flipped and correlated with the upstream gradient.

---

## 5. ReLU Backward

Forward: `output = max(0, input)`

Backward: the gradient is a **gate**
```
∂Loss/∂input = upstream_grad  if input > 0
∂Loss/∂input = 0              if input ≤ 0
```

ReLU kills the gradient wherever the forward activation was negative. This is what causes "dying ReLU" — if a neuron always outputs 0, its gradient is always 0, and it never updates.

---

## 6. MaxPool Backward

Forward: for each 2×2 window, take the maximum value.

Backward: gradient flows **only to the position that was the maximum** during the forward pass. All other positions get gradient = 0.

This requires saving the **argmax positions** (which pixel was max) during the forward pass — a technique called "max switches" or "mask."

```
if input[i][j] was the max in its pool window:
    grad_input[i][j] = upstream_grad for that window
else:
    grad_input[i][j] = 0
```

---

## 7. Softmax + CrossEntropy — Why Combine Them

**Softmax** converts raw logits to probabilities:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**CrossEntropy loss** for the correct class `c`:
```
Loss = -log(softmax(x_c))
```

The gradient of the combined operation simplifies beautifully:
```
∂Loss/∂x_i = softmax(x_i) - 1  (for the correct class i = c)
∂Loss/∂x_i = softmax(x_i)      (for all other classes)
```

This is why they're always implemented together — the math cancels out the unstable `exp()` in the denominator and gives a clean gradient.

---

## 8. CIFAR-10 Dataset

- 60,000 images: 50,000 train + 10,000 test
- Image size: 32 × 32 × 3 (RGB)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) per channel

**Why normalize?** Without normalization, pixel values range 0–255. Large input values → large activations → large gradients → unstable training. Normalizing to ~zero mean and unit std keeps all layers in a healthy operating range.

---

## 9. Weight Initialization — Why It Matters

If weights are initialized to zero → all neurons compute the same thing → gradients are identical → network never learns different features. This is called the **symmetry problem**.

**He initialization** (used for ReLU networks):
```
weights ~ N(0, sqrt(2 / fan_in))
```
The `2/fan_in` factor accounts for the fact that ReLU kills half the neurons on average — so we scale up to compensate and keep variance stable through layers.

---

## 10. Questions I Can Now Answer in an Interview

- [ ] Derive the chain rule for a 3-layer CNN on a whiteboard
- [ ] Explain what `grad[i][j]` of a conv filter means in plain English
- [ ] Why is the backward pass of a conv layer a transposed convolution?
- [ ] Why does MaxPool need to save argmax positions during the forward pass?
- [ ] Why do we combine Softmax and CrossEntropy instead of implementing them separately?
- [ ] What happens if you initialize all weights to zero?
- [ ] What is a dying ReLU and how do you fix it?

---

*Updated continuously as the project progresses.*