import numpy as np
from src.model import ConvLayer, ReLU, MaxPool, Flatten, FCLayer
from src.loss import CrossEntropyLoss
from src.dataloader import get_dataloaders

def train(epochs=10, lr=0.01, batch_size=64):
    # 1. Get data
    train_loader, test_loader = get_dataloaders(batch_size)
    
    # 2. Build layers
    conv1 = ConvLayer(3, 32)
    relu1 = ReLU()
    pool1 = MaxPool()
    flat  = Flatten()
    fc1   = FCLayer(8192, 128)
    relu2 = ReLU()
    fc2   = FCLayer(128, 10)
    loss_fn = CrossEntropyLoss()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            x = images.numpy()
            y = labels.numpy()
            
            # 3. Forward pass
            out_conv  = conv1.forward(x)
            out_relu1 = relu1.forward(out_conv)
            out_pool  = pool1.forward(out_relu1)
            out_flat  = flat.forward(out_pool)
            out_fc1   = fc1.forward(out_flat)
            out_relu2 = relu2.forward(out_fc1)
            logits    = fc2.forward(out_relu2)
            
            # 4. Compute loss (assuming loss_fn.forward returns loss and probabilities)
            loss = loss_fn.forward(logits, y)
            epoch_loss += loss
            
            # 5. Backward pass (Chain Rule: start from the end)
            grad = loss_fn.backward()
            grad = fc2.backward(grad)
            grad = relu2.backward(grad)
            grad = fc1.backward(grad)
            grad = flat.backward(grad)
            grad = pool1.backward(grad)
            grad = relu1.backward(grad)
            conv1.backward(grad)
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.4f}")
            
            # 6. SGD update
            # Update FC2
            fc2.weights -= lr * fc2.grad_weights
            fc2.bias    -= lr * fc2.grad_bias
            # Update FC1
            fc1.weights -= lr * fc1.grad_weights
            fc1.bias    -= lr * fc1.grad_bias
            # Update Conv1
            conv1.weights -= lr * conv1.grad_weights
            conv1.bias    -= lr * conv1.grad_bias
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
