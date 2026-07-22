# Heart Disease Prediction

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Flask-2.0%2B-green)

A machine learning pipeline for predicting heart disease.

## System Architecture
```
[Data Ingestion] -> [Data Transformation] -> [Model Trainer] -> [Model Evaluation] -> [Flask App]
```

## Dataset & Features
13 clinical feature inputs (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) predicting target heart disease.

## Local Setup & Installation
```bash
git clone https://github.com/hardikkaurani/Heart-Disease-Prediction.git
pip install -r requirements.txt
python app.py
```
