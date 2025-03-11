Deep Learning in Medical Image Analysis

This repository contains multiple labs for the Deep Learning in Medical Image Analysis course. Each lab focuses on training different models for various medical image analysis tasks.


# Lab 1
---
Objective: Train a deep learning model to classify handwritten digits (0-9).

Techniques Used:

Python (pandas, numpy, torch)

Deep Learning Framework: PyTorch

Data Processing:

Used sklearn.model_selection.train_test_split to split data into training and testing sets.

Data visualization with matplotlib and plotly.

Model Architecture:

SimpleNN: A simple neural network with fully connected layers.

Layers include ReLU, BatchNorm, and Dropout.

Training & Evaluation:

Loss Function: CrossEntropyLoss

Optimizer: Adam

Metrics: accuracy_score, f1_score

Testing & Output:

Model inference on test data.

Output predictions saved as CSV (Hw1_submission_Z.csv).

# Lab 2
---
Objective: Train and compare different deep learning models for medical image classification.

Techniques Used:

Deep Learning Framework: PyTorch

Model Architectures:

VGG16:

Used pre-trained VGG16 for feature extraction.

Modified MaxPool2d to Identity() and AdaptiveAvgPool2d for better adaptability to medical images.

Transfer Learning to initialize weights.

ResNet50:

Used pre-trained ResNet50, replacing the global pooling layer with AdaptiveAvgPool2d.

Removed ResNet’s classifier, keeping only feature extraction.

2048-dimensional feature vectors were extracted and concatenated with age and gender features before classification.

Swin Transformer (Swin ViT):

Used pre-trained Swin Transformer with image size set to 56x56.

Removed the classification head (head.fc) to retain extracted features.

Concatenated features with age and gender data, then passed through a linear classifier.

Data Processing:

Three-channel transformation: Selected the previous, current, and next image slices to form a three-channel input.

Data Augmentation: Applied carefully to avoid distorting medical image properties.

Training & Evaluation:

Loss Function: CrossEntropyLoss

Optimizer: Adam

Learning Rate Scheduler and Early Stopping to prevent overfitting.

# Lab 3
---

Objective: Investigate the impact of different image fusion techniques on medical image classification.

Techniques Used:

Deep Learning Framework: PyTorch

Model Architectures:

ResNet50:

Used pre-trained ResNet50 for 2D image classification.

Applied AdaptiveAvgPool2d and a fully connected classification layer.

3D CNN:

Used 3D Convolutional Layers for volumetric feature extraction.

Early Fusion:

Combined multiple image slices into a three-channel input for ResNet50.

Late Fusion:

Trained separate ResNet50 models and combined their predictions using Weighted Sum.

Single Slice:

Used only a single image slice for classification (baseline comparison).

Data Processing:

Medical image dataset preprocessing (Normalization, Resizing).

Three-channel transformation by selecting previous, current, and next slices.

Data Augmentation: Rotation, Contrast Adjustment.

Training & Evaluation:

Loss Function: CrossEntropyLoss

Optimizer: Adam

Evaluation Metrics: Accuracy, F1 Score, ROC-AUC Score

Testing & Results:

ResNet50_6c.csv: ResNet50 model predictions.

3D.csv: 3D CNN predictions.

early.csv: Early Fusion predictions.

late.csv: Late Fusion predictions.

single.csv: Single Slice predictions.

# Lab 4
---

Objective: Medical image segmentation using U-Net and FCN-8s.

Techniques Used:

Deep Learning Framework: PyTorch

Model Architectures:

U-Net:

Encoder: Multi-layer convolution and pooling to extract features at different scales.

Decoder: Upsampling layers to restore spatial resolution.

Skip Connections: Connect encoder and decoder to preserve high-resolution features.

FCN-8s:

Encoder: Feature extraction using a deep convolutional network (e.g., VGG-16).

Decoder: Upsampling layers to restore segmentation maps.

Skip Connections: Combine multi-scale features for enhanced segmentation accuracy.

Training & Evaluation:

Loss Function: Dice Loss, CrossEntropyLoss.

Optimizer: Adam.

Metrics: Dice Score, IoU (Intersection over Union).

# Lab 5
---

Objective: Chest X-ray disease detection using Faster R-CNN.

Techniques Used:

Deep Learning Framework: PyTorch

Model Architecture:

Faster R-CNN:

Used pre-trained weights from COCO dataset.

Trained on labeled chest X-ray images.

Data Processing:

Converted DICOM images to a standardized format.

Applied Log Transformation and Simplest Color Balance for contrast enhancement.

Ensured patient-wise data split to prevent data leakage.

Training & Evaluation:

Loss Function: IoU, mAP.

Optimizer: SGD with Momentum and Nesterov.

Evaluation Metrics: AP (Average Precision), Recall.

Visualization:

Used EigenCAM and AblationCAM to analyze model focus areas.