import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from src.Heart.exception import customexception
from src.Heart.logger import logging
from src.Heart.utils.utils import load_object

class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        with mlflow.start_run():
                mlflow.log_metric("Testing Accuracy", acc)
                mlflow.log_metric("Precision Score", precision)
                mlflow.log_metric("Recall Score", recall)
                mlflow.log_metric("F1 Score", f1)
                return acc

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            model_path = os.path.join("Artifacts", "model.pkl")
            model = load_object(model_path)
            predictions = model.predict(X_test)
            (acc, precision, recall, f1) = self.eval_metrics(y_test, predictions)
            with mlflow.start_run():
                mlflow.log_metric("Testing Accuracy", acc)
                mlflow.log_metric("Precision Score", precision)
                mlflow.log_metric("Recall Score", recall)
                mlflow.log_metric("F1 Score", f1)
                return acc
        except Exception as e:
            raise customexception(e, sys)
