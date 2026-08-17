# MonReader – Book Page Flip Classification

## 📖 Overview

**MonReader** is a computer vision project that uses deep learning to classify images of a book page into two categories:

* **flip** – the page is being flipped
* **notflip** – the page is not being flipped

The project uses **PyTorch** and a pretrained **EfficientNet-B0** model to perform binary image classification using transfer learning.

The notebook demonstrates an end-to-end deep learning workflow, including dataset exploration, image preprocessing, data augmentation, model training, evaluation, and individual image prediction.

### Workflow

1. Dataset loading
2. Exploratory data analysis
3. Image validation
4. Image preprocessing
5. Data augmentation
6. PyTorch Dataset and DataLoader creation
7. Transfer learning with EfficientNet-B0
8. Model training
9. Model evaluation
10. Individual image prediction

---

## 📂 Dataset

The dataset contains separate training and testing directories:

```text
data/
├── training/
│   ├── flip/
│   └── notflip/
│
└── testing/
    ├── flip/
    └── notflip/
```

### Dataset Size

| Dataset   | Number of Images |
| --------- | ---------------: |
| Training  |            2,392 |
| Testing   |              597 |
| **Total** |        **2,989** |

### Class Distribution

#### Training Set

| Class     |    Images |
| --------- | --------: |
| `notflip` |     1,230 |
| `flip`    |     1,162 |
| **Total** | **2,392** |

#### Testing Set

| Class     |  Images |
| --------- | ------: |
| `notflip` |     307 |
| `flip`    |     290 |
| **Total** | **597** |

The two classes are relatively well balanced.

### Image Validation

The original images have the following characteristics:

| Property           |       Result |
| ------------------ | -----------: |
| Width              | 1,080 pixels |
| Height             | 1,920 pixels |
| Unique image sizes |            1 |
| Corrupted images   |            0 |

All images have the same original dimensions, and no corrupted images were detected.

---

## 🔍 Exploratory Data Analysis

The notebook performs an initial exploration of the dataset before model training.

The analysis includes:

* Displaying sample images
* Examining class distributions
* Checking image dimensions
* Checking for corrupted images
* Comparing training and testing datasets

This provides an understanding of the dataset and verifies that the images are suitable for model training.

---

## 🖼️ Image Size: 128×128 vs 224×224

Image resolution is an important consideration in computer vision because it determines how much visual information is available to the model.

### 128×128

A **128×128** input image contains:

```text
128 × 128 = 16,384 pixels
```

Advantages:

* Lower computational cost
* Faster training and inference
* Lower GPU memory usage
* Smaller tensors

Disadvantages:

* More image information is discarded
* Fine details may be lost
* Page edges, curvature, shadows, and other subtle visual features may become less distinguishable

### 224×224

A **224×224** input image contains:

```text
224 × 224 = 50,176 pixels
```

Advantages:

* Preserves substantially more visual information
* Better suited for detecting fine image features
* Commonly used with pretrained computer vision models
* Better aligned with the input resolution used by the pretrained EfficientNet-B0 model

Disadvantages:

* Requires more computational resources than 128×128
* Uses more GPU memory
* May increase training time

### Comparison

| Feature                       |  128×128 |       224×224 |
| ----------------------------- | -------: | ------------: |
| Pixels per image              |   16,384 |        50,176 |
| Relative pixel information    |       1× |        ~3.06× |
| Computational cost            |    Lower |        Higher |
| Memory usage                  |    Lower |        Higher |
| Fine visual details           |    Lower |        Higher |
| Training speed                |   Faster |        Slower |
| EfficientNet-B0 compatibility | Possible | **Preferred** |
| Used in this project          |       No |       **Yes** |

For this project, **224×224 was selected** because the model uses pretrained EfficientNet-B0 weights and the higher resolution preserves more visual information.

## The notebooks resize the original 1080×1920 images to **224×224** for both training and validation/testing.

## 🧹 Image Preprocessing

The original images are:

```text
1080 × 1920 pixels
```

Before being passed to EfficientNet-B0, the images are resized to:

```text
224 × 224 pixels
```

The preprocessing pipeline includes:

* Resizing images to 224×224
* Converting images to PyTorch tensors
* Normalizing RGB channels

The normalization uses:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

These transformations are used for validation and individual image prediction as well.

---

## 🔄 Data Augmentation

Training images undergo additional augmentation to improve generalization.

The training transformation includes:

* Resizing to 224×224
* Random rotation of approximately ±5 degrees
* Color jitter with brightness variation
* Tensor conversion
* Image normalization

The validation/testing images are resized and normalized without the random training augmentations.

---

## 📦 Data Loading

A custom PyTorch `Dataset` is used to load images and their corresponding labels.

Each sample contains:

* Filename
* Label
* Filepath

The class labels are encoded as:

```text
notflip → 0
flip    → 1
```

The datasets are then loaded using PyTorch `DataLoader` objects for efficient batching during training and evaluation.

---

## 🧠 Model

The project uses **EfficientNet-B0** as the underlying image classification architecture.

EfficientNet-B0 is initialized with pretrained weights and adapted for the two-class MonReader classification problem.

### Transfer Learning

Instead of training a convolutional neural network from scratch, the project uses **transfer learning**.

The pretrained model provides visual features learned from a large image dataset. The final classification layer is adapted to classify the two MonReader classes:

```text
0 → notflip
1 → flip
```

This allows the model to leverage previously learned visual representations while adapting the classifier to the MonReader task.

---

## 🏋️ Model Training

The model is trained using **PyTorch**.

The training process follows:

```text
Training Images
      ↓
Image Augmentation
      ↓
224 × 224 Images
      ↓
DataLoader
      ↓
EfficientNet-B0
      ↓
Classification Loss
      ↓
Backpropagation
      ↓
Weight Updates
      ↓
Validation
```

The model's validation accuracy and F1 score are monitored during training.

### Training Performance

The first notebook reached 100% validation accuracy and 100% validation F1 score beginning at Epoch 3. Training stopped early at Epoch 6 after the model showed no further F1 improvement.

The second notebook reached 100% validation accuracy and 100% validation F1 score at Epoch 2, with later epochs maintaining the same scores.

---

## 📊 Model Evaluation

The trained model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### Final Evaluation Metrics

| Metric        | MonReader Training Model | MonReader Training/Test Model |
| ------------- | -----------------------: | ----------------------------: |
| **Accuracy**  |                 **100%** |                      **100%** |
| **Precision** |                 **100%** |                      **100%** |
| **Recall**    |                 **100%** |                      **100%** |
| **F1 Score**  |                 **100%** |                      **100%** |

The first notebook reports 1.0 for accuracy, precision, recall, and F1 score.

The second notebook also reports 1.0 for all four metrics.

### Confusion Matrix

#### MonReader Training Model

The evaluation used 479 validation samples:

```text
[[246   0]
 [  0 233]]
```

This means:

* 246 `notflip` images were correctly classified
* 233 `flip` images were correctly classified
* 0 false positives
* 0 false negatives

#### MonReader Training/Test Model

The evaluation used all 597 testing samples:

```text
[[307   0]
 [  0 290]]
```

This means:

* 307 `notflip` images were correctly classified
* 290 `flip` images were correctly classified
* 0 false positives
* 0 false negatives

---

## 📈 Performance Summary

| Model Evaluation             | Samples Evaluated | Accuracy | Precision |   Recall | F1 Score |
| ---------------------------- | ----------------: | -------: | --------: | -------: | -------: |
| Training Notebook Validation |               479 | **100%** |  **100%** | **100%** | **100%** |
| Training/Test Notebook       |               597 | **100%** |  **100%** | **100%** | **100%** |

The second evaluation is particularly important because it evaluates the model across the complete **597-image testing dataset**.

---

## 🔮 Prediction

The notebook also includes functionality for predicting the class of an individual image.

The prediction workflow is:

```text
Input Image
     ↓
Convert to RGB
     ↓
Resize to 224 × 224
     ↓
Convert to Tensor
     ↓
Normalize
     ↓
EfficientNet-B0
     ↓
Softmax
     ↓
Class Prediction
     ↓
flip / notflip
```

The model also outputs a confidence score for the predicted class.

For example, the notebook demonstrated:

```text
Prediction: flip
Confidence: 0.9973
```

and:

```text
Prediction: notflip
Confidence: 0.9996
```

---

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **EfficientNet-B0**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Pillow**

---

## 📁 Project Structure

```text
MonReader/
├── MonReaderTraining.ipynb
├── MonReaderTrainingTest.ipynb
├── MonReaderTraining.py
├── MonReaderTraingTest.py
├── README.md

```

---

## 🎯 Project Objective

The objective of MonReader is to develop a deep learning model capable of automatically identifying whether a book page is **being flipped** or **not being flipped** from an image.

The project combines:

* Image preprocessing
* Data augmentation
* Transfer learning
* EfficientNet-B0
* PyTorch
* Binary image classification

## Using **224×224 input images**, the trained models achieved **100% accuracy, precision, recall, and F1 score** on the reported evaluation datasets.

## 📝 Notes

The reported 100% evaluation metrics reflect the results recorded in the provided notebooks. They should be interpreted in the context of the available dataset and evaluation methodology. A perfect score on a particular test set does not necessarily guarantee the same performance on completely new images captured under different conditions.
