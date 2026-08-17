# MonReader – Book Page Flip Classification

## 📖 Overview

**MonReader** is a computer vision project that uses deep learning to classify images of a book page into two categories:

* **flip** – the page is being flipped
* **notflip** – the page is not being flipped

The project uses **PyTorch** and a pretrained **EfficientNet-B0** model to perform binary image classification through transfer learning.

The notebook demonstrates an end-to-end machine learning workflow, from dataset exploration and preprocessing to model training, evaluation, and image-level prediction.

### Workflow

1. Dataset loading
2. Exploratory data analysis
3. Image validation
4. Image preprocessing
5. Data augmentation
6. PyTorch `Dataset` and `DataLoader` creation
7. Transfer learning with EfficientNet-B0
8. Model training
9. Model evaluation
10. Individual image prediction

---

## 📂 Dataset

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

The two classes are relatively well balanced in both the training and testing datasets.

### Image Validation

The notebook validates the image files and reports:

| Property           | Result       |
| ------------------ | ------------ |
| Image width        | 1,080 pixels |
| Image height       | 1,920 pixels |
| Unique image sizes | 1            |
| Corrupted images   | 0            |

All images have a consistent resolution, and no corrupted images were detected.

---

## 🔍 Exploratory Data Analysis

Before model training, the notebook performs an initial exploration of the dataset.

The analysis includes:

* Displaying sample images
* Examining class distributions
* Checking image dimensions
* Detecting corrupted images
* Comparing training and testing datasets

This provides an overview of the dataset and helps verify that the images are suitable for model training.

---

## 🖼️ Image Preprocessing

The original images have dimensions of:

```text
1080 × 1920 pixels
```

Images are processed before being passed to the neural network.

The preprocessing pipeline includes:

* Resizing images to the input dimensions required by EfficientNet-B0
* Converting images into PyTorch tensors
* Normalizing image data

### Data Augmentation

Training images additionally undergo image augmentation to improve the model's ability to generalize to variations in the input images.

Augmentation helps reduce overfitting by exposing the model to modified versions of the training images.

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

The training and testing data are then loaded using PyTorch `DataLoader` objects.

This enables efficient batching and iteration during model training and evaluation.

---

## 🧠 Model

The project uses **EfficientNet-B0** as the underlying image classification architecture.

EfficientNet-B0 is initialized with pretrained weights and adapted for the two-class MonReader classification problem.

### Transfer Learning

Instead of training a convolutional neural network from scratch, the project uses **transfer learning**.

The pretrained EfficientNet-B0 model provides visual features learned from a large image dataset. The final classification layer is modified to specialize the model for the MonReader task.

The model produces two output classes:

```text
0 → notflip
1 → flip
```

This approach allows the model to leverage previously learned visual representations while adapting to the specific requirements of book-page classification.

---

## 🏋️ Model Training

The model is trained using **PyTorch**.

The training workflow consists of:

```text
Training Images
      ↓
Image Transformations
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
Model Evaluation
```

During training, the model:

1. Loads a batch of training images.
2. Applies the required image transformations.
3. Passes the images through EfficientNet-B0.
4. Calculates the classification loss.
5. Performs backpropagation.
6. Updates the model weights.
7. Evaluates performance on the testing dataset.

Model performance is monitored during training to identify the best-performing model.

---

## 📊 Model Evaluation

The trained model is evaluated using multiple classification metrics rather than relying solely on accuracy.

The notebook includes:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**
* **Classification Report**

These metrics provide a more comprehensive assessment of how well the model distinguishes between `flip` and `notflip` images.

### Evaluation Metrics

| Metric                | Purpose                                                   |
| --------------------- | --------------------------------------------------------- |
| Accuracy              | Percentage of correctly classified images                 |
| Precision             | Proportion of predicted positive cases that are correct   |
| Recall                | Proportion of actual positive cases correctly identified  |
| F1 Score              | Harmonic mean of precision and recall                     |
| Confusion Matrix      | Shows correct and incorrect predictions by class          |
| Classification Report | Provides a detailed summary of classification performance |

---

## 🔮 Prediction

The notebook also includes functionality for making predictions on individual images.

The prediction workflow is:

```text
Input Image
     ↓
Resize / Preprocessing
     ↓
EfficientNet-B0
     ↓
Class Prediction
     ↓
flip / notflip
```

The model takes an unseen image, applies the same preprocessing used during training, and predicts whether the page is being flipped.

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

A typical project structure is:

```text
MonReader/
│
├── data/
│   ├── training/
│   │   ├── flip/
│   │   └── notflip/
│   │
│   └── testing/
│       ├── flip/
│       └── notflip/
│
├── notebooks/
│   └── MonReader.ipynb
│
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd MonReader
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the dataset under the `data/` directory using the following structure:

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

### 4. Run the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/MonReader.ipynb
```

Run the notebook cells sequentially to reproduce the data exploration, preprocessing, model training, evaluation, and prediction workflow.

---

## 🎯 Project Objective

The primary objective of MonReader is to develop a deep learning model capable of automatically identifying whether a book page is **being flipped** or **not being flipped** from an image.

By combining image preprocessing, data augmentation, and transfer learning with EfficientNet-B0, the project demonstrates an end-to-end approach to solving a real-world binary image classification problem.



