# 🧠 EfficientNet-B0 Deepfake Image Classifier

## 📘 Overview
This project implements a **partially fine-tuned EfficientNet-B0 model** to classify images as **Real** or **AI-generated (Fake)**.  
The model is trained on a Kaggle dataset with GPU acceleration, achieving high accuracy while maintaining training efficiency.

---

## ⚙️ Features
✅ Pretrained EfficientNet-B0 backbone  
✅ **Partially unfrozen** fine-tuning for top layers  
✅ GPU-accelerated (CUDA) training  
✅ Kaggle dataset-based training  
✅ 90.54% best validation accuracy  
✅ Model saving and reloading supported  

---

## 📂 Dataset
Dataset used:  
📦 [AI or Not: Human vs AI Generated Images (Kaggle)](https://www.kaggle.com/datasets/trainingdatapro/ai-or-not-human-vs-ai-generated-images)

Folder structure:
```
dataset/
├── Train/
│   ├── real/
│   └── fake/
└── Val/
    ├── real/
    └── fake/
```

---

## 🧩 Model Architecture

We use **EfficientNet-B0 (pretrained on ImageNet)** as a feature extractor.  
Only the final classifier and the last few convolutional layers are **unfrozen** for fine-tuning.

```python
from torchvision import models
import torch.nn as nn

model = models.efficientnet_b0(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the top few layers
for name, param in list(model.features.named_parameters())[-20:]:
    param.requires_grad = True

# Modify classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
```

---

## 🚀 Training Configuration

| Parameter      | Value           |
|----------------|----------------|
| Model          | EfficientNet-B0 |
| Epochs         | 10              |
| Batch Size     | 64              |
| Optimizer      | Adam            |
| Learning Rate  | 1e-4            |
| Loss Function  | CrossEntropyLoss|
| Device         | GPU (CUDA)      |

---

## 📊 Results

| Metric             | Score    |
|--------------------|----------|
| Training Accuracy  | 97.12%   |
| Validation Accuracy| 90.54%   |
| Epochs             | 10       |
| Batch Size         | 64       |

---

## 💾 Saving the Model

```python
torch.save(model.state_dict(), 'efficientnet_b0_fakereal.pth')
```

---

## 🔁 Loading the Model

```python
from torchvision import models
import torch.nn as nn
import torch

# Rebuild the model structure
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# Load trained weights
model.load_state_dict(torch.load('efficientnet_b0_fakereal.pth', map_location='cuda'))
model.eval()
```

---

## 🧠 Inference Example

Test your trained model on any image:

```python
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# Load model
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load('efficientnet_b0_fakereal.pth', map_location='cuda'))
model.eval()

# Transform for test image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load image
img = Image.open("test_image.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to('cuda')

# Predict
with torch.no_grad():
    output = model(img_tensor)
    pred = torch.argmax(output, dim=1).item()

print("Prediction:", "Real" if pred == 0 else "Fake")
```

---

## ⚡ GPU Optimization Tips

- Enable CUDNN benchmark for performance:
    ```python
    torch.backends.cudnn.benchmark = True
    ```
- Use `batch_size=64` or higher
- Try mixed precision training:
    ```python
    from torch.cuda.amp import GradScaler, autocast
    ```
- Close unnecessary apps during training

---

## 🧰 Requirements

```
torch
torchvision
tqdm
Pillow
numpy
matplotlib
```

Install all dependencies:
```bash
pip install torch torchvision tqdm Pillow numpy matplotlib
```

---

## 🏁 Future Improvements

- Add Grad-CAM visualization for feature attention
- Implement data augmentation (rotation, blur, jitter)
- Deploy model via Flask or Streamlit
- Add inference API endpoint

---

## 🧾 Summary

| Component   | Details                       |
|-------------|------------------------------|
| Model       | EfficientNet-B0 (Partially Unfrozen) |
| Dataset     | Kaggle “AI or Not”           |
| Epochs      | 10                           |
| Batch Size  | 64                           |
| Validation Accuracy | 90.54%               |
| Save File   | efficientnet_b0_fakereal.pth |
| Device      | CUDA (GPU)                   |

---

## 🧬 Author

Project by: **Drishi Kachchhawaha**  
Model: EfficientNet-B0 (Fine-tuned)  
Accuracy: 90.54%