# 🦠 COVID-19 Detection from Chest X-Rays using Deep Learning
This project focuses on building and evaluating deep learning models to automatically detect COVID-19 infection from chest X-ray images and distinguish it from pneumonia and normal conditions. The goal is to assist medical professionals in the early diagnosis of COVID-19, improving treatment response time and patient outcomes.

## 🚀 Project Overview

In this notebook-based project, I implemented and compared multiple deep learning models to classify chest X-ray images into three classes:

- **COVID-19**
- **Pneumonia**
- **Normal (Healthy)**

The models were trained and evaluated using publicly available datasets, aiming to achieve high accuracy, precision, and recall for medical-grade performance.

---

## 🧠 Models Implemented

- **Convolutional Neural Networks (CNN)**
- **Transfer Learning using Pretrained Models:**
  - VGG16
  - ResNet50
  - InceptionV3
  - EfficientNetB0

Each model was fine-tuned and validated on a well-prepared dataset. Data augmentation techniques were applied to enhance the model’s robustness and reduce overfitting.

---

## 📂 Dataset

The dataset consists of labeled chest X-ray images sourced from:

- COVID-19 image data collection
- RSNA Pneumonia Detection Challenge
- NIH Chest X-ray dataset

The dataset was preprocessed, normalized, and split into training, validation, and testing sets.

---

## 🧪 Key Features

- 📷 Image classification of X-rays into 3 classes
- 📊 Performance evaluation using confusion matrices and classification reports
- 🧬 Comparative analysis of model performance
- ⚙️ Transfer learning with model fine-tuning
- 🧼 Data cleaning and augmentation

---

## 📁 Notebooks

- `COVID-19 detection vs pneumonia and normal.ipynb`: Full pipeline including preprocessing, training, and evaluation.
- `COVID-19 DETECTION MODELS TESTING REPORT.ipynb`: Summary and comparative analysis of different model results.
- `COVID-19 detection vs pneumonia.ipynb`: Binary classification approach for detecting COVID-19 vs Pneumonia.

---

## 📌 Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy, OpenCV, Matplotlib, Seaborn
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📎 Conclusion

This project demonstrates the feasibility of using deep learning to detect COVID-19 and differentiate it from other lung infections with promising accuracy. The model can serve as a supportive diagnostic tool in clinical settings.

---

## 🙌 Acknowledgments

- Inspired by open-source medical datasets and the global fight against COVID-19.
- Special thanks to the researchers and dataset contributors.
