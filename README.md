# Breast Cancer Prediction System

## Overview
This project is a machine learning–based breast cancer diagnostic system with a Flask web application.
Users can enter tumor feature values through a web interface, and the system predicts whether the tumor
is benign or malignant.

## Dataset
Breast Cancer Wisconsin Diagnostic Dataset

## Machine Learning Models
The following models were implemented and evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (Linear Kernel)
- Decision Tree

Based on evaluation metrics such as recall, precision, F1-score, and confusion matrix analysis,
the Support Vector Machine (Linear Kernel) was selected as the final model.

## Best Model
**Support Vector Machine (Linear Kernel)**  
Chosen due to its strong balance between recall and precision, which is critical for minimizing
false negative cases in breast cancer diagnosis.

## Tech Stack
- Python
- Flask
- Scikit-learn
- NumPy
- HTML / CSS

## Project Structure
breast-cancer-prediction/
│
├── app.py
├── breast_cancer_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
├── templates/
│ └── index.html
│
└── static/