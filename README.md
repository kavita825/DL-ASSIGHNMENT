# Handwritten Digit Recognition with a Feedforward Neural Network

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-3B4CCA)

A Deep Learning assignment that classifies handwritten digits (0–9) from the **MNIST** dataset using a fully connected neural network built with **TensorFlow/Keras**. A baseline model with 128 hidden neurons is compared against a wider model with 256 hidden neurons.

## Table of Contents

- [Objectives](#objectives)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Future Work](#future-work)
- [Author](#author)

## Objectives

- Load and explore the MNIST handwritten-digit dataset and prepare it for training.
- Build a baseline feedforward neural network (**Model A**, 128 hidden neurons).
- Train it while monitoring accuracy and loss on training and validation data.
- Evaluate it on unseen test data and check predictions on individual digits.
- Modify the architecture (**Model B**, 256 hidden neurons) and compare it with the baseline.

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) is loaded directly through `keras.datasets.mnist`, so no manual download is needed.

| Split | Images | Shape |
|-------|--------|-------|
| Training | 60,000 | 28 × 28 (greyscale) |
| Test | 10,000 | 28 × 28 (greyscale) |

- **Classes:** 10 (digits 0–9)
- **Pre-processing:** pixel values are divided by 255 to scale them to the range 0–1.
- **Validation:** 10% of the training set (6,000 images) is held out during training, leaving 54,000 images for weight updates.

## Model Architecture

```
Input (28×28)  →  Flatten (784)  →  Dense (128 or 256, ReLU)  →  Dense (10, Softmax)  →  Digit 0–9
```

Both models are built with `keras.Sequential`. Only the width of the hidden layer differs.

| Configuration | Model A (baseline) | Model B (modified) |
|---------------|--------------------|--------------------|
| Hidden layer | Dense(128, ReLU) | Dense(256, ReLU) |
| Output layer | Dense(10, Softmax) | Dense(10, Softmax) |
| Trainable parameters | 101,770 | 203,530 |
| Optimiser | Adam | Adam |
| Loss | Sparse categorical cross-entropy | Sparse categorical cross-entropy |
| Epochs / validation split | 5 / 10% | 5 / 10% |

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

## Results

| Metric | Model A (128) | Model B (256) |
|--------|---------------|---------------|
| Final training accuracy | 98.54% | 98.93% |
| Final validation accuracy | 97.68% | 98.12% |
| **Test accuracy** | **97.71%** | **97.80%** |
| Test loss | 0.0769 | 0.0714 |
| Approx. time per epoch | 6–7 s | 9–13 s |

**Sample predictions (Model A, first five test images): all five correct.**

| Test image | 1 | 2 | 3 | 4 | 5 |
|------------|---|---|---|---|---|
| Actual label | 7 | 2 | 1 | 0 | 4 |
| Predicted label | 7 | 2 | 1 | 0 | 4 |

### Key observations

- Both models learn quickly: validation accuracy is already near 97% after the first epoch.
- The baseline generalises well: its test accuracy (97.71%) closely matches its final validation accuracy (97.68%).
- Doubling the hidden layer gives a small gain of **+0.09 percentage points** (9 more correct images out of 10,000). With a single run per model, this difference may be within normal run-to-run variation.
- Validation loss rises slightly in later epochs while training loss keeps falling, an early sign of overfitting.
- Model B has twice the parameters and takes roughly 1.5× longer per epoch.

## Getting Started

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

**2. (Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install tensorflow numpy matplotlib jupyter
```

**4. Run the notebook**

```bash
jupyter notebook DL_Code.ipynb
```

Run all cells from top to bottom. MNIST is downloaded automatically on first run, and each model trained in under a minute in the original run (about 6–13 s per epoch).

## Repository Structure

```
.
├── DL_Code.ipynb        # Full code: data, training, evaluation, comparison
└── README.md
```

## Future Work

- Add Dropout or early stopping to reduce overfitting.
- Repeat each experiment with several random seeds to confirm the difference between models.
- Plot a confusion matrix to see which digits are confused most often.
- Try a convolutional neural network (CNN), which typically exceeds 99% accuracy on MNIST.

## Author

**Kavita P** — Rai Technology University
