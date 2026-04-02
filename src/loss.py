import numpy as np

class CrossEntropyLoss:
    def forward(self, logits, labels):
        # logits shape: (batch_size, 10)
        batch_size = logits.shape[0]
        # labels shape: (batch_size,) — integer class indices
        batch_size_label = labels.shape
        # step 1: stable softmax
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        # step 2: compute cross entropy loss
        corect_probs = probs[np.arange(batch_size), labels]
        # step 3: save for backward
        loss = -np.mean(np.log(corect_probs))
        self.probs = probs
        self.labels = labels
        return loss
    
    def backward(self):
        batch_size = self.probs.shape[0]
        d_logits = self.probs.copy()
        d_logits[np.arange(batch_size), self.labels] -= 1
        return d_logits / batch_size