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

Dataset Size

The notebook reports:

Dataset	Number of Images
Training	2,392
Testing	597
Total	2,989
Class Distribution
Training
Class	Images
notflip	1,230
flip	1,162
Testing
Class	Images
notflip	307
flip	290

The classes are relatively well balanced.

The notebook also verifies the image dimensions and reports that all images have the same size:

Width: 1080 pixels
Height: 1920 pixels
Unique image sizes: 1
Corrupted images: 0

These dataset statistics are directly reported by the notebook.

Exploratory Data Analysis

The notebook performs an initial inspection of the dataset, including:

Displaying sample images
Examining class distributions
Checking image dimensions
Checking for corrupted images
Comparing the training and testing datasets

This provides an overview of the data before model training.

Image Preprocessing

The original images have dimensions of:

1080 × 1920

The images are processed before being passed to the neural network.

The preprocessing pipeline includes resizing the images and converting them into tensors suitable for PyTorch.

Training images additionally undergo image augmentation to improve the model's ability to generalize to variations in the input images.

Data Loading

A custom PyTorch dataset is used to load the images and their corresponding labels.

Each sample contains:

filename
label
filepath

The labels correspond to:

flip    → 1
notflip → 0

The training and testing data are then loaded using PyTorch DataLoaders.

Model

The project uses EfficientNet-B0 as the underlying image classification model.

EfficientNet-B0 is initialized with pretrained weights and then adapted for the two-class MonReader classification problem.

The final classification layer is modified so that the model produces two outputs:

0 → notflip
1 → flip

This approach uses transfer learning, allowing the model to take advantage of visual features learned from a larger image dataset while adapting the final classifier to the MonReader task.

Training

The model is trained using PyTorch.

The training process consists of:

Loading a batch of training images
Applying image transformations
Passing the images through EfficientNet-B0
Calculating the classification loss
Performing backpropagation
Updating the model weights
Evaluating the model on the testing dataset

Model performance is monitored during training to determine the best-performing model.

Evaluation

The notebook evaluates the trained model using classification metrics including:

Accuracy
Precision
Recall
F1 score
Confusion matrix
Classification report

These metrics provide a more complete picture of model performance than accuracy alone.

Prediction

The notebook also includes functionality for making predictions on individual images.

The prediction workflow:

Input Image
     ↓
Resize / Preprocessing
     ↓
EfficientNet-B0
     ↓
Class Prediction
     ↓
flip / notflip
