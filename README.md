# Pneumonia Detection from Chest X-Ray using Deep Learning

##  Overview
This project focuses on detecting **Pneumonia** from chest X-ray images using **Deep Learning**. Pneumonia is a serious lung infection that can be life-threatening if not diagnosed early. This system automates the detection process using a **Convolutional Neural Network (CNN)** based on **ResNet18**.

The model classifies chest X-ray images into two categories:
- **NORMAL**
- **PNEUMONIA**

This project demonstrates how deep learning can assist in **medical image analysis**, reducing manual effort and improving diagnostic efficiency.

---

##  Objectives
- Build a deep learning model to classify chest X-ray images  
- Achieve high accuracy in detecting pneumonia  
- Evaluate performance using classification metrics  
- Enable prediction on new unseen images  

---

##  Dataset

Dataset used:  
 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  

---

## Model Architecture

- Model: **ResNet18 (Pretrained on ImageNet)**
- Transfer Learning applied
- Final layer modified for **binary classification**

---

## Tech Used

- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## Data Preprocessing

- Resize → 224 × 224  
- Normalization (ImageNet mean & std)  
- Data Augmentation:
  - Random Horizontal Flip  
  - Color Jitter  
  - Random Resized Crop  

---

## Training Details

- Optimizer: **Adam**
- Loss Function: **CrossEntropyLoss**
- Epochs: **15**
- Batch Size: **32**
- Training Time: **11 minutes 31 seconds**
- Best Validation Accuracy: **81.25%**

---

## Results

### Test Performance

- **Accuracy:** 83%

| Class       | Precision | Recall | F1-Score |
|------------|----------|--------|----------|
| NORMAL     | 0.70     | 0.97   | 0.81     |
| PNEUMONIA  | 0.97     | 0.75   | 0.84     |

---

## Confusion Matrix

|                | Predicted NORMAL | Predicted PNEUMONIA |
|----------------|----------------|---------------------|
| Actual NORMAL  | 226            | 8                   |
| Actual PNEUMONIA | 99           | 291                 |

### Analysis
- High **recall for NORMAL (0.97)** → very good at identifying healthy cases  
- High **precision for PNEUMONIA (0.97)** → predictions are reliable  
- Some pneumonia cases are missed (FN = 99) → improvement area  

---

## Usage

### 1. Train the model

python train.py

### 2. Test/Evaluate the Model

python evaluate.py

