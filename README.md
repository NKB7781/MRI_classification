# 🧠 Brain Tumor MRI Image Classification

This project uses deep learning to classify brain tumors from MRI images into four categories: **glioma**, **meningioma**, **pituitary**, and **no tumor**. It implements both a **Custom CNN** and a **fine-tuned ResNet50 model**, with automatic model selection based on prediction confidence.

## 📂 Dataset

The dataset contains MRI scan images grouped into four folders:
- `glioma`
- `meningioma`
- `pituitary`
- `no_tumor`

> Preprocessing and augmentation were applied using Keras `ImageDataGenerator`.

---

## 🧠 Models Used

### ✅ 1. Custom CNN Model
A deep CNN built from scratch, trained on augmented MRI data.

### ✅ 2. ResNet50 (Transfer Learning)
Fine-tuned on MRI scans using a pretrained ResNet50 base.

### 🔁 Model Switching Logic
During prediction, the system uses **both models**, compares their confidence scores, and returns the prediction from the more confident one.

---

## 🚀 Streamlit Deployment

The app is built with **Streamlit** and allows users to:
- Upload brain MRI images
- See predictions and confidence
- Automatically use the better model per case

> ResNet50 model is loaded via Google Drive to handle large file size.

---

## 🛠 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier

Conclusion
## This project shows how deep learning can support radiologists by automating brain tumor detection. It balances speed and accuracy by integrating two models and dynamically selecting the best one per image.
The deployment with Streamlit makes it practical and user-friendly for real-time diagnosis.
