# Quantum ML-based Classification of Leukoplakia with Dysplasia

This repository implements a deep learning and quantum machine learning framework for classification of histopathological oral images into four diagnostic categories:

1. Leukoplakia with dysplasia
2. Leukoplakia without dysplasia
3. Oral Squamous Cell Carcinoma (OSCC)
4. Normal

## Workflow

C-GAN Augmentation → CML/QML Classification → Baseline Comparison → Performance Evaluation → Explainability

## Dataset Structure

```text
DATA_ROOT/
├── Dataset1/
│   ├── With_Dysplasia/
│   ├── Without_Dysplasia/
│   └── OSCC/
└── Dataset2/
    ├── OSCC/
    └── Normal/
```

Supported formats: PNG, JPG/JPEG, TIFF, and BMP.

## Requirements

```bash
pip install tensorflow pennylane scikit-learn matplotlib seaborn pillow
```

A GPU is recommended.

## Image Preprocessing

- Original images are converted to RGB.
- Images are resized to 150×150.
- Data are split using a stratified 70/20/10 train/validation/test split.
- For C-GAN, images are converted to grayscale and resized to 28×28.
- For QML, images are converted to grayscale and resized to 40×40.

## Data Leakage Prevention

The default configuration is:

```python
LEAK_FREE = True
```

Real images are split before C-GAN training. The C-GAN is trained only on the training split, and generated synthetic images are added only to the training set.

Setting `LEAK_FREE=False` reproduces the alternative augmentation-before-splitting procedure but may introduce leakage and inflate test performance.

## Conditional GAN

The C-GAN generates class-conditioned synthetic images.

Main configuration:

- Image size: 28×28×1
- Latent dimension: 100
- Epochs: 500
- Batch size: 64
- Learning rate: 0.0001
- Adam beta1: 0.5
- Dropout: 0.5

Configured synthetic targets:

| Class | Synthetic Images |
|---|---:|
| Leukoplakia with dysplasia | 4,237 |
| Leukoplakia without dysplasia | 1,608 |
| OSCC | 5,379 |
| Normal | 765 |

Generated images are converted to 150×150×3 for CML and baseline classifiers.

## Quantum Machine Learning

The QML pipeline uses a fixed quanvolutional quantum circuit.

Process:

1. 40×40 grayscale image
2. 2×2 non-overlapping patches
3. Four-qubit angle encoding using RY(π×x)
4. Fixed PennyLane RandomLayers quantum circuit
5. Pauli-Z expectation values
6. Resulting quantum representation: 20×20×4

Quantum representations are cached to reduce repeated circuit computation.

## CML Model

The CML model uses:

```text
150×150×3
     ↓
Conv2D (32 filters, 3×3, ReLU)
     ↓
Flatten
     ↓
Dropout
     ↓
Dense (64, ReLU)
     ↓
Softmax (4 classes)
```

## QML Classifier

```text
20×20×4 quantum representation
          ↓
Flatten
          ↓
Dense (512, ReLU)
          ↓
Dense (64, ReLU)
          ↓
Dropout
          ↓
Softmax (4 classes)
```

## Baseline Models

Four baseline architectures are implemented:

- VGG19
- DenseNet201
- ResNet152
- InceptionV3

All use 150×150×3 inputs.

The notebook uses `BASELINE_WEIGHTS=None` by default, meaning the baseline networks are initialized from scratch.

## Training

Classifier configuration:

- Epochs: 500
- Batch size: 32
- Learning rate: 0.0001
- Independent runs: 5
- Optimizer: Adam
- Early stopping: enabled
- Patience: 20 epochs

The best validation-loss weights are restored when early stopping is triggered.

## Evaluation Metrics

Six metrics are calculated:

1. Specificity
2. Sensitivity
3. Precision
4. Accuracy
5. F1-score
6. Matthews Correlation Coefficient (MCC)

Multi-class metrics are calculated using a one-vs-rest formulation. Class-wise Specificity, Sensitivity, Precision, and F1-score are macro-averaged.

Results from five independent runs are summarized as:

**Mean ± Standard Deviation**

## Explainability

The notebook provides:

- Grad-CAM for the CML model
- Gradient-based saliency visualization for the QML model

The final visualization is saved as:

```text
outputs/gradcam.png
```

## Output Files

The notebook can generate:

```text
outputs/
├── cgan_generator.keras
├── CML_best.keras
├── QML_best.keras
├── VGG19_best.keras
├── DenseNet201_best.keras
├── ResNet152_best.keras
├── InceptionV3_best.keras
├── quanv_train.npy
├── quanv_val.npy
├── quanv_test.npy
├── all_runs.csv
├── summary_mean_sd.csv
└── gradcam.png
```

## Running the Notebook

1. Install the required packages.
2. Place the datasets under the configured `DATA_ROOT`.
3. Set the experiment configuration.
4. Run the notebook cells sequentially.

Recommended publication configuration:

```python
QUICK_TEST = False
LEAK_FREE = True
N_RUNS = 5
EPOCHS = 500
GAN_EPOCHS = 500
```

## Quick Test Mode

For checking whether the pipeline works, set:

```python
QUICK_TEST = True
```

This reduces epochs, runs, and synthetic image generation. It should only be used for pipeline testing and not for final reported results.

## Important Implementation Notes

Several methodological details are ambiguous in the source methodology and are therefore explicitly implemented as assumptions:

- Latent dimension is set to 100 where different values are reported.
- The classifier uses four softmax outputs because there are four classes.
- ResNet152 is used because standard Keras does not provide ResNet151.
- The CML convolutional layer uses 32 filters because the number of filters is not specified in the methodology.
- `LEAK_FREE=True` is recommended for publication-quality experiments to prevent augmentation leakage.

## Overall Pipeline

```text
Histopathological Images
          ↓
Data Preprocessing
          ↓
Stratified 70/20/10 Split
          ↓
       C-GAN
          ↓
Synthetic Training Images
          ↓
Augmented Training Set
          ↓
   ┌───────────────┐
   │               │
   ↓               ↓
  CML             QML
   │               │
150×150×3       40×40 Gray
                   ↓
                2×2 Patches
                   ↓
              4-Qubit Circuit
                   ↓
                20×20×4
   │               │
   └───────┬───────┘
           ↓
    Baseline Models
           ↓
 Performance Evaluation
           ↓
Specificity / Sensitivity /
Precision / Accuracy / F1 / MCC
           ↓
    Mean ± SD (5 Runs)
           ↓
Explainability Analysis
```

## Reproducibility

The base random seed is:

```python
SEED = 42
```

Individual classifier runs use different seeds derived from the base seed.

For final experiments, record the exact dataset composition, train/validation/test counts, synthetic-image counts, model initialization settings, and whether pretrained weights were used.
