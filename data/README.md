# Data

CIFAR-10 dataset is automatically downloaded here when you run the training script.

```python
from src.dataloader import get_dataloaders
train_loader, test_loader = get_dataloaders()  # downloads automatically
```

Dataset details:
- 50,000 training images
- 10,000 test images  
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Image size: 32×32×3 (RGB)
- Download size: ~170MB

The dataset files are excluded from git via `.gitignore`.