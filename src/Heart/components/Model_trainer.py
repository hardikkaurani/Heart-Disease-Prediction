from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os
import sys
from dataclasses import dataclass

from src.Heart.exception import customexception
from src.Heart.logger import logging
from src.Heart.utils.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "SVC": SVC()
            }
            params = {
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128]},
                "Decision Tree": {'criterion': ['gini', 'entropy']},
                "Logistic Regression": {'C': [0.1, 1.0, 10.0]},
                "SVC": {'C': [0.1, 1.0, 10.0]}
            }
            model_report:dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )
            return model_report
        except Exception as e:
            raise customexception(e, sys)
