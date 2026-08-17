# MonReader – Book Page Flip Classification

## Overview

MonReader is a computer vision project that uses deep learning to classify images of a book page into two categories:

- **flip** – the page is being flipped
- **notflip** – the page is not being flipped

The project uses PyTorch and a pretrained EfficientNet-B0 model to perform binary image classification.

The notebook covers the complete machine learning workflow:

1. Dataset loading
2. Dataset exploration and visualization
3. Image validation
4. Image preprocessing
5. Data augmentation
6. PyTorch Dataset and DataLoader creation
7. Transfer learning with EfficientNet-B0
8. Model training
9. Model evaluation
10. Model prediction on individual images

---

## Dataset

The dataset is organized into separate training and testing directories:

```text
data/
├── training/
│   ├── flip/
│   └── notflip/
│
└── testing/
    ├── flip/
    └── notflip/
