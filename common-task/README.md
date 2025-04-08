# Common Task 1: Electron/Photon Classification

## Dataset:
- **Photon**: [CERNBox Link](https://cernbox.cern.ch/index.php/s/AtBT8y4MiQYFcgc)
- **Electron**: [CERNBox Link](https://cernbox.cern.ch/index.php/s/FbXw3V4XNyYB3oA )

## Objective:
The goal is to train a **ResNet-15-like model** to classify electrons vs. photons based on detector hits, using **energy** and **time** channels.

## Solution Overview

### Preprocessing:
- Loaded **HDF5** dataset, reshaped to dimensions (C=2, 32, 32) — combining both photon and electron classes with proper labels.
- **Data Split**: Instead of the suggested 80:20 split, used a **70:15:15** split for **train/val/test** to monitor generalization and avoid overfitting.

### Augmentations:
- **Train**: Flip, rotate, affine transformations, random erasing, normalization.
- **Validation/Test**: Only normalization.

### Model:
Custom **ResNet-15-like** architecture with the following components:
- Pre-activation residual blocks
- **Squeeze-and-Excitation (SE)** blocks for improved feature recalibration
- Attention mechanism before the final fully connected layer (FC)
- Grouped convolutions and dropout (0.5) for regularization

### Training Tricks:
- **Focal Loss** to handle class imbalance
- **Mixup** (data-level regularization)
- **OneCycleLR** learning rate scheduler
- **Gradient clipping** to avoid exploding gradients
- **Mixed precision** using `GradScaler` for stability and speed

### Evaluation:
- **Classification Report** and **Confusion Matrix** for model performance.
- **Accuracy metrics**
- **Electron class**:
  - Precision: 0.72
  - Recall: 0.69
  - F1-Score: 0.71
  - Support: 37,567

- **Photon class**:
  - Precision: 0.70
  - Recall: 0.73
  - F1-Score: 0.71
  - Support: 37,133

  ### Graphs:
1. **Confusion matrix**:  
   ![Confusion matrix](graphs/Confusion%20matrix.png)  
   *This graph shows Confusion matrix for test set*

## Model:
You can download the trained model from [Drive link](https://drive.google.com/file/d/1i4Xgtwy3hxz2EL3OzMInG0RMBiwzXMht/view?usp=sharing).

## Known Issues:
- During final feature visualization using **t-SNE**, a **RuntimeError** occurred due to a shape mismatch in the linear layer. This issue likely stems from an incorrect flattening or mismatch in the expected input shape for the linear layer during feature extraction.
- **Note**: This error only impacted the final visualization step. The training and evaluation processes were completed successfully, and the rest of the pipeline and results remain valid.
- I was unable to rerun the cell to debug this due to exhausted Colab GPU credits. Once GPU access is restored, I will revisit and fix this issue.

## To Do:
- Continue improving the model based on future testing and adjustments.


