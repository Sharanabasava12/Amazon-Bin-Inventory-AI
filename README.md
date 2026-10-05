# Amazon Bin Inventory AI

## Computer Vision Based Warehouse Inventory Audit System

A machine learning project that uses computer vision to estimate the quantity of products inside Amazon inventory bins and compare the predicted quantity with the recorded warehouse quantity.

The project reproduces the machine-learning approaches described in the reference work **"Amazon Inventory Reconciliation Using AI"** and extends the work by implementing a **Random Forest classifier**.

---

# 1. Project Overview

Warehouse inventory reconciliation requires comparing the physical quantity of products inside storage bins with the quantity recorded in inventory systems.

Manual inventory checking is time-consuming and can result in:

- Inventory shortages
- Inventory overages
- Incorrect inventory records
- Increased warehouse auditing time

This project develops a computer-vision-based inventory auditing system.

The system:

1. Loads Amazon bin images.
2. Loads the corresponding inventory metadata.
3. Extracts the expected quantity.
4. Prepares the image dataset.
5. Trains machine-learning models.
6. Predicts the quantity class from an image.
7. Compares predicted quantity with expected quantity.
8. Classifies the bin as:
   - MATCHED
   - SHORTAGE
   - OVERAGE
9. Provides batch auditing for all available bins.
10. Provides an interactive Streamlit application.

---

# 2. Reference Work

### Reference

**Amazon Inventory Reconciliation Using AI**

The reference work investigates computer-vision techniques for inventory reconciliation using Amazon bin images.

The reference uses:

- Convolutional Neural Networks
- ResNet architectures
- Support Vector Machines
- HOG/PCA-based image features
- Inventory quantity prediction
- Accuracy and RMSE-based evaluation

This project reproduces the main model categories that were implemented in the reference and adds Random Forest as an extension.

---

# 3. Project Objectives

### Objective 1 — Dataset Preparation

Prepare the Amazon bin image and metadata files for machine-learning experiments.

### Objective 2 — CNN Reproduction

Implement a ResNet-34 based image-classification model.

### Objective 3 — SVM Reproduction

Implement an SVM classifier using HOG image features.

### Objective 4 — Extension

Implement Random Forest using the same HOG features.

### Objective 5 — Evaluation

Compare the models using:

- Accuracy
- RMSE
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix

### Objective 6 — Inventory Auditing

Build an application that compares predicted physical inventory with recorded inventory.

---

# 4. Dataset

The downloaded dataset contains Amazon bin images and corresponding metadata.

The project contains:

```text
data/
├── images/
├── metadata/
├── raw/
├── samples/
└── processed/