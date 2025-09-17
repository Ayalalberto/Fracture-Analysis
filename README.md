# Bone Fracture Analysis using Machine Learning and Deep Learning

This project explores the detection and classification of bone fractures from X-ray images using various machine learning techniques. The analysis begins with unsupervised learning methods and progresses towards a more robust deep learning approach.

## Overview

The primary goal is to build a model capable of automatically identifying whether a bone in an X-ray image is healthy or fractured, potentially classifying the type of fracture and location in the image. This repository documents the journey from initial exploratory analysis with clustering algorithms to the implementation of a convolutional neural network (CNN) for improved accuracy.

The project is structured in two main phases:
1.  **Phase 1: Unsupervised Learning (Completed)**: An initial analysis using dimensionality reduction (PCA) and clustering algorithms (K-Means, GMM, etc.) to discover patterns in the data without relying on labels. This phase, detailed in [`this notebook`](Fracturas.ipynb), concluded that these methods are insufficient for this complex task, paving the way for a more advanced approach.
2.  **Phase 2: Deep Learning (In Progress)**: The current focus is on implementing a deep learning model to leverage the hierarchical feature-learning capabilities of neural networks for better classification performance.

## Dataset

The project uses the **[Human Bone Fractures Multi-modal Image Dataset (HBFMID)](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)**.

- **Structure**: The dataset is divided into `train`, `test`, and `valid` sets.
- **Labels**: Annotations are provided in YOLO format (`.txt` files), specifying the class ID and bounding box for each fracture or healthy bone.
- **Classes**: The dataset includes multiple fracture types and a 'Healthy' class.

## Methodology

### Phase 1: Unsupervised Learning Analysis

The initial approach involved the following steps:
1.  **Data Preprocessing**: Images were loaded, converted to grayscale, resized to a uniform dimension (64x64), and flattened into vectors.
2.  **Dimensionality Reduction**: Principal Component Analysis (PCA) was applied to reduce the feature space while retaining most of the variance.
3.  **Class Imbalance**: The SMOTE (Synthetic Minority Over-sampling Technique) was used on the PCA-transformed data to balance the training set.
4.  **Clustering**: Several algorithms were evaluated:
    - K-Means
    - Gaussian Mixture Models (GMM)
    - BIRCH
    - Hierarchical Clustering
    - DBSCAN
5.  **Conclusion**: The clustering models achieved low Silhouette scores, indicating that they struggled to find meaningful, well-separated clusters. This suggests the visual patterns are too complex for these methods, justifying the need for a deep learning approach.

### Phase 2: Deep Learning Implementation (Next Steps)

The next stage of this project will focus on building, training, and evaluating a Convolutional Neural Network (CNN). The planned steps are:

1.  **Data Pipeline**: Create an efficient data loader for the image dataset that handles data augmentation (rotation, flipping, zooming) to increase the robustness of the model.
2.  **Model Architecture**:
    - Start with a custom CNN architecture tailored for image classification.
    - Explore using **transfer learning** with pre-trained models like VGG16, ResNet, or EfficientNet, which are powerful feature extractors.
3.  **Training**: Train the model on the `train` set and use the `valid` set to monitor for overfitting and tune hyperparameters.
4.  **Evaluation**: Assess the final model's performance on the unseen `test` set using metrics such as accuracy, precision, recall, F1-score, and the confusion matrix.
