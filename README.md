
# MNIST Digit Classification

This repository contains implementations of various machine learning classifiers for the MNIST dataset, including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest Classifier (RFC), and Convolutional Neural Networks (CNN).

## Overview

The MNIST dataset consists of 70,000 handwritten digit images, each of size 28x28 pixels, and the corresponding labels (0-9). This project demonstrates different methods to classify these digits using machine learning and deep learning algorithms.

## Classifiers Implemented

1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Machine (SVM)**
3. **Random Forest Classifier (RFC)**
4. **Convolutional Neural Network (CNN)**

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mnist-digit-classification.git
    cd mnist-digit-classification
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the MNIST dataset and place it in the `MNIST_Dataset_Loader/dataset/` directory.

## Usage

### K-Nearest Neighbors (KNN)

Run the KNN classifier:
```bash
python knn_classifier.py
```

### Support Vector Machine (SVM)

Run the SVM classifier:
```bash
python svm_classifier.py
```

### Random Forest Classifier (RFC)

Run the Random Forest classifier:
```bash
python rfc_classifier.py
```

### Convolutional Neural Network (CNN)

Run the CNN classifier:
```bash
python cnn_classifier.py --save_model 1 --save_weights weights.hdf5
```

### Arguments

- `--save_model`: Save the trained model weights.
- `--load_model`: Load pre-trained model weights.
- `--save_weights`: File path to save or load weights.

## Results

The classifiers will output the accuracy, confusion matrix, and display random sample predictions along with the actual labels.

## Dependencies

- Python 3.x
- numpy
- scikit-learn
- matplotlib
- keras
- tensorflow
- opencv-python
- argparse

## Directory Structure

```
mnist-digit-classification/
│
├── MNIST_Dataset_Loader/
│   └── dataset/
│       ├── train-images-idx3-ubyte
│       ├── train-labels-idx1-ubyte
│       ├── t10k-images-idx3-ubyte
│       └── t10k-labels-idx1-ubyte
│
├── knn_classifier.py
├── svm_classifier.py
├── rfc_classifier.py
├── cnn_classifier.py
├── requirements.txt
├── README.md
└── summary.log
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
